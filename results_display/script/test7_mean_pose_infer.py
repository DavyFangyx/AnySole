"""Test7: mean-pose inference — feed mean/GT poses into the model, export SMPL.

Goal: decide whether the model output moves with its pose input and where it
lands relative to GT.  Uses the trained checkpoint (default
results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt) with real VT
conditioning and five input arms:

    tau0_mean    mean pose, tau=0           clean-input pass-through
    tau0_gt      GT pose, tau=0             clean GT reconstruction (-> 0 ideally)
    ddim_noise   standard DDIM from noise   same path as eval.py VT2M (repro check)
    ddim_mean500 mean pose noised to tau=500, then DDIM (mean pose truly "in the model")
    ddim_gt500   GT pose noised to tau=500, then DDIM (init upper bound)

Reading: tau0_gt MPJPE >> 0 means the model cannot even reconstruct its
training target (undertrained / weak conditioning); |tau0_mean - tau0_gt|
shows how much the output follows the input pose; ddim_mean500 vs ddim_gt500
shows how much the sampled output depends on the init pose.

Outputs under results_display/Test7_mean_pose/:
    <session>/<session>_<arm>.npz   per-arm SMPL archive (first --export-sessions sessions)
    test7_report.json               per-arm MPJPE + pairwise output distances

Usage (run from the repository root):
    python results_display/script/test7_mean_pose_infer.py
    python results_display/script/test7_mean_pose_infer.py --session S10103 --export-sessions 1
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata  # noqa: E402
from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.diffusion import GaussianDiffusion  # noqa: E402
from anysole.eval import _load_model  # noqa: E402
from anysole.geometry import fk_pose6d  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, FPS, POSE_DIM  # noqa: E402
from test6_dataset_check import ensure_mean_pose  # noqa: E402

ARMS = ("tau0_mean", "tau0_gt", "ddim_noise", "ddim_mean500", "ddim_gt500")

DEFAULT_CKPT = cli_common.RESULTS_ROOT / "AnySole" / "anysolev1_joint_and" / "checkpoints" / "ckpt_last.pt"


def ddim_from(diffusion, model, cond: dict, x_init: torch.Tensor, tau_start: int, steps: int, device) -> torch.Tensor:
    """Deterministic DDIM from tau_start down to 0 (mirrors the training schedule)."""
    values = np.linspace(int(tau_start), 0, num=int(steps)).round().astype(np.int64)
    ordered = []
    for value in values.tolist():
        if int(value) not in ordered:
            ordered.append(int(value))
    if ordered[-1] != 0:
        ordered.append(0)
    x = x_init
    batch_size = x.shape[0]
    for i, t in enumerate(ordered):
        tau = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        out = model(cond["V_feat"], cond["T_raw"], cond["T_phys"], x, tau, cond["config_id"], cond.get("session_id"), V_hmr=cond.get("V_hmr"))
        if int(t) == 0:
            return out["x0_hat"]
        tau_prev = torch.full((batch_size,), int(ordered[i + 1]), device=device, dtype=torch.long)
        x = diffusion.ddim_step(x, tau, tau_prev, out["x0_hat"], eta=0.0)
    return x


def load_references(ckpt: Path) -> dict:
    """Reference numbers from Test6 and the checkpoint's own eval metrics."""
    refs = {}
    try:
        b1_path = cli_common.DISPLAY_ROOT / "Test6_dataset_check" / "mean_baseline.json"
        if b1_path.is_file():
            payload = json.loads(b1_path.read_text(encoding="utf-8"))
            refs["mean_baseline_B1_mm"] = payload.get("B1_mean_pose_gt_root_MPJPE_mm")
    except (OSError, ValueError) as exc:
        log.warning("could not read Test6 mean_baseline.json: {}", exc)
    try:
        metrics_path = ckpt.parent.parent / "metrics" / "test.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            vt2m = (payload.get("metrics") or {}).get("VT2M") or {}
            refs["model_VT2M_MPJPE_mm"] = vt2m.get("MPJPE")
    except (OSError, ValueError) as exc:
        log.warning("could not read model metrics: {}", exc)
    return refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test7: mean-pose inference with the AnySole model.")
    cli_common.add_common_args(
        parser,
        session=True,
        split_csv=True,
        split=True,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "Test7_mean_pose",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "v1.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--contact-method", type=str, default="", help="Dataset contact labels; default: checkpoint's, then config's.")
    parser.add_argument("--mean-pose", type=str, default=str(cli_common.DISPLAY_ROOT / "Test6_dataset_check" / "mean_pose.npz"),
                        help="Train mean pose cache (auto-computed when missing).")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Cap the number of sessions (0 = all).")
    parser.add_argument("--export-sessions", type=int, default=4, help="First N sessions get SMPL NPZ exports (0 = all).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--init-tau", type=int, default=500, help="Noise level for the mean/GT-init DDIM arms.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli_common.outputs_ready([out_dir / "test7_report.json"]) and not args.force:
        log.info("Skip Test7: {} already exists (use --force to overwrite)", out_dir / "test7_report.json")
        return 0
    config = load_config(Path(args.config))
    device = resolve_device(args.device)
    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    model = _load_model(checkpoint, config, device)
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])
    ckpt_config = checkpoint.get("config", {})
    contact_method = str(args.contact_method or ckpt_config.get("contact_method") or config.get("contact_method", "tactile_abs"))

    split_csv = Path(cli_common.resolve_path(args.split_csv))
    session_ids = cli_common.load_test_sessions(args.session, split_csv, args.split)
    if args.limit_sessions:
        session_ids = session_ids[: args.limit_sessions]
    export_sessions = set(session_ids[: args.export_sessions]) if args.export_sessions else set(session_ids)
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=split_csv,
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        session_ids=session_ids,
        contact_method=contact_method,
        tactile_input=str(ckpt_config.get("tactile_input", "raw108")),
        no_imu=bool(ckpt_config.get("no_imu", False)),
        v_input=str(ckpt_config.get("v_input", config.get("v_input", "hrnet"))),
        v_hmr_model=str(ckpt_config.get("v_hmr_model", config.get("v_hmr_model", "gvhmr"))),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    beta_by_session = {
        session["session_id"]: np.asarray(session.get("betas", np.zeros(10)), dtype=np.float32)
        for session in dataset.sessions
    }

    mean_pose_np, _ = ensure_mean_pose(args.mean_pose, config, contact_method)
    mean_pose = torch.from_numpy(mean_pose_np).float().to(device)  # (24, 6)

    arm_sums = {arm: {"mpjpe": 0.0, "n": 0} for arm in ARMS}
    pair_sums = {"%s__%s" % (a, b): 0.0 for a, b in combinations(ARMS, 2)}
    pair_count = 0
    per_session = {}
    exports = {}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            bsz = batch["pose_gt"].shape[0]
            cid = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
            v_feat, t_raw, t_phys = condition_inputs(batch, cid)
            sid = raw_batch["session_id"]
            cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                    "config_id": cid, "session_id": sid, "V_hmr": batch.get("V_hmr")}
            v_kw = {"V_hmr": batch.get("V_hmr")}
            zero = torch.zeros(bsz, device=device, dtype=torch.long)
            anchor = batch["trans_anchor"][:, None, :]
            mean_x = mean_pose.reshape(1, 1, POSE_DIM).expand(bsz, int(config["tw"]), POSE_DIM)

            poses = {}
            trans_world = {}
            # tau=0 clean-input arms.
            for arm, x_in in (("tau0_mean", mean_x), ("tau0_gt", batch["pose_gt"])):
                out = model(v_feat, t_raw, t_phys, x_in, zero, cid, sid, **v_kw)
                poses[arm] = out["x0_hat"]
                trans_world[arm] = out["trans_hat"] + anchor
            # Standard DDIM from noise (same path as eval.py VT2M).
            pred = diffusion.ddim_sample_loop(
                model,
                tau_related_kwargs=cond,
                shape=(bsz, int(config["tw"]), POSE_DIM),
                steps=sample_steps,
                eta=0.0,
                device=device,
            )
            out = model(v_feat, t_raw, t_phys, pred, zero, cid, sid, **v_kw)
            poses["ddim_noise"] = pred
            trans_world["ddim_noise"] = out["trans_hat"] + anchor
            # DDIM initialized from a half-noised mean / GT pose.
            for arm, x0 in (("ddim_mean500", mean_x), ("ddim_gt500", batch["pose_gt"])):
                tau_init = torch.full((bsz,), int(args.init_tau), device=device, dtype=torch.long)
                pred = ddim_from(diffusion, model, cond, diffusion.q_sample(x0, tau_init), int(args.init_tau), sample_steps, device)
                out = model(v_feat, t_raw, t_phys, pred, zero, cid, sid, **v_kw)
                poses[arm] = pred
                trans_world[arm] = out["trans_hat"] + anchor

            kps = {arm: fk_pose6d(poses[arm], trans_world[arm], batch["offsets"], batch["parents"]) for arm in ARMS}
            for arm in ARMS:
                err = torch.linalg.vector_norm(kps[arm] - batch["kp_gt"], dim=-1)
                arm_sums[arm]["mpjpe"] += float(err.sum().item())
                arm_sums[arm]["n"] += err.numel()
                for i, s in enumerate(sid):
                    per_session.setdefault(s, {a: [0.0, 0] for a in ARMS})
                    per_session[s][arm][0] += float(err[i].sum().item())
                    per_session[s][arm][1] += err[i].numel()
                for i, s in enumerate(sid):
                    if s in export_sessions:
                        exports.setdefault((s, arm), []).append(
                            (
                                int(raw_batch["frame_start"][i]),
                                poses[arm][i].cpu().numpy(),
                                trans_world[arm][i].cpu().numpy(),
                            )
                        )
            for a, b in combinations(ARMS, 2):
                pair_sums["%s__%s" % (a, b)] += float(torch.linalg.vector_norm(kps[a] - kps[b], dim=-1).sum().item())
            pair_count += kps[ARMS[0]].numel()

    arms_report = {arm: {"MPJPE_mm": arm_sums[arm]["mpjpe"] / max(arm_sums[arm]["n"], 1) * 1000.0} for arm in ARMS}
    pairwise = {key: value / max(pair_count, 1) * 1000.0 for key, value in pair_sums.items()}
    session_report = {
        s: {arm: values[0] / max(values[1], 1) * 1000.0 for arm, values in arms.items()}
        for s, arms in per_session.items()
    }

    for (session_id, arm), windows in exports.items():
        windows.sort(key=lambda item: item[0])
        pose = np.concatenate([item[1] for item in windows], axis=0)
        trans = np.concatenate([item[2] for item in windows], axis=0)
        output_path = out_dir / session_id / ("%s_%s.npz" % (session_id, arm))
        smpl_poses = smpl24_pose6d_to_poses(pose)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        betas = beta_by_session.get(session_id, np.zeros((10,), dtype=np.float32))
        np.savez_compressed(output_path, poses=smpl_poses, trans=pelvis_to_smpl_trans(pose, trans, betas),
                            root_orient=smpl_poses[:, :3], pose_body=smpl_poses[:, 3:],
                            betas=betas,
                            mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
                            source_frame_times_s=np.arange(len(pose), dtype=np.float32) / FPS,
                            **smpl_archive_metadata())
        log.info("wrote {}", output_path)

    report = {
        "checkpoint": str(args.ckpt),
        "sample_steps": sample_steps,
        "init_tau": int(args.init_tau),
        "split": args.split,
        "mean_pose_source": str(args.mean_pose),
        "arms_MPJPE_mm": arms_report,
        "pairwise_output_MPJPE_mm": pairwise,
        "per_session_MPJPE_mm": session_report,
        "references": load_references(Path(args.ckpt)),
    }
    (out_dir / "test7_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote {}", out_dir / "test7_report.json")

    print("== Test7 mean-pose inference ==")
    for arm in ARMS:
        print("  %-14s MPJPE = %7.1f mm" % (arm, arms_report[arm]["MPJPE_mm"]))
    print("  pairwise output distances (mm): " + "  ".join("%s=%.1f" % (key, value) for key, value in pairwise.items()))
    refs = report["references"]
    if refs.get("mean_baseline_B1_mm") is not None:
        print("  Test6 B1 (mean pose + GT root) = %.1f mm" % refs["mean_baseline_B1_mm"])
    if refs.get("model_VT2M_MPJPE_mm") is not None:
        print("  model eval VT2M               = %.1f mm" % refs["model_VT2M_MPJPE_mm"])
    print("  reading:")
    print("    tau0_gt MPJPE >> 0        -> model cannot even reconstruct its target (undertrained / weak conditioning)")
    print("    tau0_mean ~ tau0_gt       -> output barely follows the input pose")
    print("    ddim_mean500 ~ ddim_gt500 -> sampled output does not depend on the init pose")
    print("    ddim_noise vs eval VT2M   -> reproduction sanity (should be close)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
