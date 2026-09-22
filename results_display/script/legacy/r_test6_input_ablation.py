"""Test8: input-output correlation — condition ablation + gradient check.

Decides whether the model output depends on its conditioning inputs at all.
Uses the trained checkpoint (default results/AnySole/anysolev1_joint_and/
checkpoints/ckpt_last.pt) and four conditioning variants, each run through
standard DDIM (noise init) and a tau=0 GT-pose reconstruction:

    real     the window's own V/T (same path as eval.py VT2M)
    zero     V=0, T=0
    mean     per-timestep train-split means of V/T (Test6 input_means.npz)
    shuffle  V/T rolled one window within the batch (real inputs, wrong window)

Plus a gradient check on the first batch: ||dL/dV||, ||dL/dT||, ||dL/dT_phys||
against ||dL/dx|| for a tau=0 reconstruction loss.  Gradients near zero mean
the conditioning path is gradient-dead.

Reading: MPJPE(shuffle) - MPJPE(real) small -> conditioning is ignored;
pairwise output distances across variants ~ 0 -> output is input-independent;
gradient ratios ~ 0 -> conditioning cannot be learned (dead path).

Outputs under results_display/r_test6_input_ablation/:
    <session>/<session>_<variant>_<arm>.npz
    test8_report.json

Usage (run from the repository root):
    python results_display/script/r_test6_input_ablation.py
    python results_display/script/r_test6_input_ablation.py --session S10103 --export-sessions 1
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata  # noqa: E402
from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.utils.diffusion import GaussianDiffusion  # noqa: E402
from anysole.eval import _load_model  # noqa: E402
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, FPS, POSE_DIM  # noqa: E402
from d_test2_dataset_check import ensure_input_means  # noqa: E402

VARIANTS = ("real", "zero", "mean", "shuffle")
ARMS = ("ddim", "tau0_gt")

DEFAULT_CKPT = cli_common.RESULTS_ROOT / "AnySole" / "anysolev1_joint_and" / "checkpoints" / "ckpt_last.pt"


def build_variants(v_feat, t_raw, t_phys, input_means, device, v_hmr=None) -> dict:
    """Return variants including the optional GVHMR channel.

    Keeping V_hmr in lock-step with V_feat is essential for hmr_gvhmr
    checkpoints; passing the real HMR tensor to a zero/shuffle ablation would
    make that probe report a false conditioning dependency.
    """
    bsz = v_feat.shape[0]
    zero = (torch.zeros_like(v_feat), torch.zeros_like(t_raw), torch.zeros_like(t_phys))
    mean = tuple(
        torch.from_numpy(input_means[key]).float().to(device).unsqueeze(0).expand(bsz, -1, -1)
        for key in ("V_feat", "T_raw", "T_phys")
    )
    shuffle = (torch.roll(v_feat, shifts=1, dims=0), torch.roll(t_raw, shifts=1, dims=0), torch.roll(t_phys, shifts=1, dims=0))
    if v_hmr is None:
        return {"real": (*((v_feat, t_raw, t_phys)), None),
                "zero": (*zero, None), "mean": (*mean, None), "shuffle": (*shuffle, None)}
    h_zero = torch.zeros_like(v_hmr)
    h_mean = torch.zeros_like(v_hmr)  # no fitted HMR mean cache; zero is explicit
    h_shuffle = torch.roll(v_hmr, shifts=1, dims=0)
    return {"real": (v_feat, t_raw, t_phys, v_hmr),
            "zero": (*zero, h_zero), "mean": (*mean, h_mean),
            "shuffle": (*shuffle, h_shuffle)}


def gradient_check(model, batch, device) -> dict:
    """Norm of dL/d(V,T) vs dL/dx for a tau=0 reconstruction loss on one batch."""
    bsz = batch["pose_gt"].shape[0]
    cid = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
    zero = torch.zeros(bsz, device=device, dtype=torch.long)
    x_in = batch["pose_gt"].clone().requires_grad_(True)
    v_in = batch["V_feat"].clone().requires_grad_(True)
    t_in = batch["T_raw"].clone().requires_grad_(True)
    tp_in = batch["T_phys"].clone().requires_grad_(True)
    with torch.enable_grad():
        out0 = model(v_in, t_in, tp_in, x_in, zero, cid, batch.get("session_id"), V_hmr=batch.get("V_hmr"))
        loss = F.mse_loss(out0["x0_hat"], batch["pose_gt"])
        grads = torch.autograd.grad(loss, (x_in, v_in, t_in, tp_in), allow_unused=True)

    def norm(g):
        return 0.0 if g is None else float(g.norm().item())

    dx, dv, dt, dtp = (norm(g) for g in grads)
    return {
        "dL_dx": dx,
        "dL_dV": dv,
        "dL_dT_raw": dt,
        "dL_dT_phys": dtp,
        "dL_dV_over_dL_dx": dv / max(dx, 1e-12),
        "dL_dT_over_dL_dx": dt / max(dx, 1e-12),
    }


def load_references(ckpt: Path) -> dict:
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
    parser = argparse.ArgumentParser(description="Test8: input-output correlation ablation.")
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
        out_dir_default=cli_common.DISPLAY_ROOT / "result/r_test6_input_ablation",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "anysole" / "configs" / "v1.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--contact-method", type=str, default="", help="Dataset contact labels; default: checkpoint's, then config's.")
    parser.add_argument("--mean-inputs", type=str, default=str(cli_common.DISPLAY_ROOT / "Test6_dataset_check" / "input_means.npz"),
                        help="Train V/T means cache (auto-computed when missing).")
    parser.add_argument("--limit-sessions", type=int, default=0, help="Cap the number of sessions (0 = all).")
    parser.add_argument("--export-sessions", type=int, default=4, help="First N sessions get SMPL NPZ exports (0 = all).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli_common.outputs_ready([out_dir / "test8_report.json"]) and not args.force:
        log.info("Skip Test8: {} already exists (use --force to overwrite)", out_dir / "test8_report.json")
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
    input_means = ensure_input_means(args.mean_inputs, config, contact_method)

    sums = {arm: {variant: {"mpjpe": 0.0, "n": 0} for variant in VARIANTS} for arm in ARMS}
    pair_sums = {arm: {"%s__%s" % (a, b): 0.0 for a, b in combinations(VARIANTS, 2)} for arm in ARMS}
    pair_count = 0
    # The gradient check needs real autograd: torch.inference_mode disables
    # gradient tracking and cannot be re-enabled from inside, so run it on a
    # throwaway iterator over the first batch before the main loop.
    grad_report = None
    exports = {}
    first_raw = next(iter(loader))
    first_batch = move_batch(first_raw, device)
    grad_report = gradient_check(model, first_batch, device)
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            bsz = batch["pose_gt"].shape[0]
            cid = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
            sid = raw_batch["session_id"]
            zero = torch.zeros(bsz, device=device, dtype=torch.long)
            anchor = batch["trans_anchor"][:, None, :]
            v_feat, t_raw, t_phys = condition_inputs(batch, cid)
            variants = build_variants(v_feat, t_raw, t_phys, input_means, device, batch.get("V_hmr"))

            kps = {arm: {} for arm in ARMS}
            poses = {arm: {} for arm in ARMS}
            trans_world = {arm: {} for arm in ARMS}
            for variant, (v, t, tp, v_hmr) in variants.items():
                cond = {"V_feat": v, "T_raw": t, "T_phys": tp,
                        "config_id": cid, "session_id": sid, "V_hmr": v_hmr}
                pred = diffusion.ddim_sample_loop(
                    model,
                    tau_related_kwargs=cond,
                    shape=(bsz, int(config["tw"]), POSE_DIM),
                    steps=sample_steps,
                    eta=0.0,
                    device=device,
                )
                out = model(v, t, tp, pred, zero, cid, sid, V_hmr=v_hmr)
                poses["ddim"][variant] = pred
                trans_world["ddim"][variant] = out["trans_hat"] + anchor
                kps["ddim"][variant] = fk_pose6d(pred, trans_world["ddim"][variant], batch["offsets"], batch["parents"])
                out0 = model(v, t, tp, batch["pose_gt"], zero, cid, sid, V_hmr=v_hmr)
                poses["tau0_gt"][variant] = out0["x0_hat"]
                trans_world["tau0_gt"][variant] = out0["trans_hat"] + anchor
                kps["tau0_gt"][variant] = fk_pose6d(out0["x0_hat"], trans_world["tau0_gt"][variant], batch["offsets"], batch["parents"])

            for arm in ARMS:
                for variant in VARIANTS:
                    err = torch.linalg.vector_norm(kps[arm][variant] - batch["kp_gt"], dim=-1)
                    sums[arm][variant]["mpjpe"] += float(err.sum().item())
                    sums[arm][variant]["n"] += err.numel()
                    for i, s in enumerate(sid):
                        if s in export_sessions:
                            exports.setdefault((s, variant, arm), []).append(
                                (
                                    int(raw_batch["frame_start"][i]),
                                    poses[arm][variant][i].cpu().numpy(),
                                    trans_world[arm][variant][i].cpu().numpy(),
                                )
                            )
                for a, b in combinations(VARIANTS, 2):
                    pair_sums[arm]["%s__%s" % (a, b)] += float(torch.linalg.vector_norm(kps[arm][a] - kps[arm][b], dim=-1).sum().item())
            pair_count += kps["ddim"][VARIANTS[0]].numel()

    report = {
        "checkpoint": str(args.ckpt),
        "sample_steps": sample_steps,
        "split": args.split,
        "variants": list(VARIANTS),
        "arms_MPJPE_mm": {
            arm: {variant: sums[arm][variant]["mpjpe"] / max(sums[arm][variant]["n"], 1) * 1000.0 for variant in VARIANTS}
            for arm in ARMS
        },
        "pairwise_variant_output_MPJPE_mm": {
            arm: {key: value / max(pair_count, 1) * 1000.0 for key, value in pair_sums[arm].items()}
            for arm in ARMS
        },
        "gradient_norms": grad_report,
        "references": load_references(Path(args.ckpt)),
    }
    for (session_id, variant, arm), windows in exports.items():
        windows.sort(key=lambda item: item[0])
        pose = np.concatenate([item[1] for item in windows], axis=0)
        trans = np.concatenate([item[2] for item in windows], axis=0)
        output_path = out_dir / session_id / ("%s_%s_%s.npz" % (session_id, variant, arm))
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
    (out_dir / "test8_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote {}", out_dir / "test8_report.json")

    print("== Test8 input-output correlation ==")
    for arm in ARMS:
        line = "  %-8s" % arm
        for variant in VARIANTS:
            line += "  %s=%6.1f" % (variant, report["arms_MPJPE_mm"][arm][variant])
        print(line + "  (mm)")
    print("  pairwise variant outputs (ddim, mm): " + "  ".join("%s=%.1f" % (k, v) for k, v in report["pairwise_variant_output_MPJPE_mm"]["ddim"].items()))
    g = grad_report or {}
    print("  grad norms: dL/dx=%.3e dL/dV=%.3e dL/dT=%.3e  ratios V/x=%.3e T/x=%.3e"
          % (g.get("dL_dx", 0.0), g.get("dL_dV", 0.0), g.get("dL_dT_raw", 0.0), g.get("dL_dV_over_dL_dx", 0.0), g.get("dL_dT_over_dL_dx", 0.0)))
    refs = report["references"]
    if refs.get("mean_baseline_B1_mm") is not None:
        print("  Test6 B1 (mean pose + GT root) = %.1f mm" % refs["mean_baseline_B1_mm"])
    if refs.get("model_VT2M_MPJPE_mm") is not None:
        print("  model eval VT2M               = %.1f mm" % refs["model_VT2M_MPJPE_mm"])
    print("  reading:")
    print("    shuffle MPJPE ~ real MPJPE  -> conditioning is ignored (output ~ input-independent)")
    print("    pairwise outputs ~ 0        -> all conditions collapse to the same pose")
    print("    grad ratios ~ 0             -> conditioning path is gradient-dead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
