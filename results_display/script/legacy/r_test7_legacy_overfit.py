"""Test9: single-batch overfit with pose-only loss and an LR sweep.

Trains a fresh AnySole model on ONE fixed batch of train windows with:
  * modality dropout off (config_id fixed to VT; the train-side equivalent of
    `--dropoutVT 0,0`),
  * model dropout off (--dropout 0.0),
  * pose-only loss: L_pose only (lambda_traj = lambda_kp = lambda_Trec =
    lambda_Vrec = lambda_con = 0),
and sweeps learning rates (default 1e-3, 3e-4, 1e-4, 3e-5) for 3000 steps each
on a small fixed batch (default 8 windows).

Both reconstruction monitors (tau=0 clean GT input and full DDIM sampling)
use the GT trajectory, so they measure pose fitting alone — the traj head is
untrained under the pose-only loss.

If L_pose reaches ~0 (and the reconstruction MPJPEs collapse on the same
batch), the pose target and pipeline are sound; generalization failures are
then a matter of undertraining / weak conditioning (see Test7/Test8).  If it
cannot fit, the target or the pipeline is broken (see Test6).

Outputs under results_display/r_test7_legacy_overfit/:
    overfit_log.csv       lr, step, L_pose, tau0 MPJPE, DDIM MPJPE
    overfit_curves.png    one curve per lr
    overfit_report.json   per-lr final numbers + PASS/FAIL verdict
    overfit_ckpt_<lr>.pt  only with --save-ckpt

Usage (run from the repository root):
    python results_display/script/r_test7_legacy_overfit.py
    python results_display/script/r_test7_legacy_overfit.py --lrs 3e-4 --steps 100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.utils.diffusion import GaussianDiffusion  # noqa: E402
from anysole.utils.geometry import fk_pose6d  # noqa: E402
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, POSE_DIM, assert_batch_shapes  # noqa: E402

# Verdict thresholds (pose-only fitting on the fixed batch).
PASS_L_POSE = 0.05
PASS_TAU0_MPJPE_MM = 20.0
PASS_DDIM_MPJPE_MM = 50.0

DEFAULT_LRS = "1e-3,3e-4,1e-4,3e-5"


def parse_lrs(value: str) -> List[float]:
    lrs = [float(item) for item in value.replace(";", ",").split(",") if item.strip()]
    if not lrs:
        raise ValueError("--lrs must contain at least one learning rate")
    return lrs


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _tau0_mpjpe_mm(model, v_feat, t_raw, t_phys, batch, device) -> float:
    """Clean GT pose input, tau=0, GT trajectory: can the model reproduce its target?"""
    with torch.inference_mode():
        zero = torch.zeros(batch["pose_gt"].shape[0], device=device, dtype=torch.long)
        out0 = model(v_feat, t_raw, t_phys, batch["pose_gt"], zero, batch["config_id"], batch.get("session_id"))
        anchor = batch["trans_anchor"][:, None, :]
        kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
        return float(torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0


def _ddim_mpjpe_mm(diffusion, model, v_feat, t_raw, t_phys, batch, device, sample_steps) -> float:
    """Full DDIM sampling on the fixed batch, GT trajectory (pose-only metric)."""
    with torch.inference_mode():
        bsz = batch["pose_gt"].shape[0]
        cond = {
            "V_feat": v_feat,
            "T_raw": t_raw,
            "T_phys": t_phys,
            "config_id": batch["config_id"],
            "session_id": batch.get("session_id"),
        }
        pred = diffusion.ddim_sample_loop(
            model,
            tau_related_kwargs=cond,
            shape=(bsz, int(batch["pose_gt"].shape[1]), POSE_DIM),
            steps=sample_steps,
            eta=0.0,
            device=device,
        )
        anchor = batch["trans_anchor"][:, None, :]
        kp = fk_pose6d(pred, batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
        return float(torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean().item()) * 1000.0


def plot_curves(rows: list, out_path: Path) -> None:
    lrs = sorted({row["lr"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for lr in lrs:
        lr_rows = [row for row in rows if row["lr"] == lr]
        steps = [row["step"] for row in lr_rows]
        label = "lr=%g" % lr
        axes[0].plot(steps, [row["L_pose"] for row in lr_rows], label=label)
        axes[1].plot(steps, [row["tau0_mpjpe_mm"] for row in lr_rows], label=label)
        ddim_rows = [row for row in lr_rows if row["ddim_mpjpe_mm"] is not None]
        if ddim_rows:
            axes[1].plot([row["step"] for row in ddim_rows], [row["ddim_mpjpe_mm"] for row in ddim_rows], marker="o", linestyle="")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L_pose (log)")
    axes[0].set_title("pose-only loss")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MPJPE (mm)")
    axes[1].set_title("reconstruction MPJPE (GT trajectory; lines=tau0, dots=DDIM)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test9: single-batch pose-only overfit with an LR sweep.")
    cli_common.add_common_args(
        parser,
        session=False,
        split_csv=False,
        split=False,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "result/r_test7_legacy_overfit",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "anysole" / "configs" / "v1.yaml"))
    parser.add_argument("--contact-method", type=str, default="", help="Contact-label scheme (default: config's).")
    parser.add_argument("--modal", type=str, default=MODEL_ANYSOLEV1)
    parser.add_argument("--batch-size", type=int, default=8, help="Windows in the fixed batch.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lrs", type=str, default=DEFAULT_LRS, help="Comma-separated learning-rate sweep.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Model nn.Dropout rate (modality dropout is off by construction: VT-only).")
    parser.add_argument("--seed", type=int, default=0, help="Reseeded per lr so every sweep point starts from the same init.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-ckpt", action="store_true", help="Also save one checkpoint per lr.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli_common.outputs_ready([out_dir / "overfit_report.json"]) and not args.force:
        log.info("Skip Test9: {} already exists (use --force to overwrite)", out_dir / "overfit_report.json")
        return 0
    config = load_config(Path(args.config))
    if args.contact_method:
        config["contact_method"] = str(args.contact_method)
    device = resolve_device(args.device)

    dataset = AnySoleDataset(
        mode="train",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        contact_method=str(config["contact_method"]),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_windows)
    batch = move_batch(next(iter(loader)), device)
    bsz = batch["pose_gt"].shape[0]
    # Modality dropout off: every window trains under the full VT config.
    batch["config_id"] = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
    assert_batch_shapes(batch, bsz)
    lrs = parse_lrs(args.lrs)
    log.info("fixed batch: {} windows, steps={}, lrs={}, dropout={}", bsz, args.steps, lrs, args.dropout)

    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(config["diffusion_sample_steps"])
    v_feat, t_raw, t_phys = condition_inputs(batch, batch["config_id"])

    all_rows = []
    per_lr = {}
    for lr in lrs:
        seed_everything(args.seed)
        model = AnySoleModel(
            d=int(config["d_model"]),
            tw=int(config["tw"]),
            dropout=float(args.dropout),
            modal=str(args.modal),
        ).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        log.info("lr sweep: lr={}", lr)
        rows = []
        for step in range(int(args.steps)):
            tau = torch.randint(0, int(config["diffusion_train_steps"]), (bsz,), device=device, dtype=torch.long)
            x_tau = diffusion.q_sample(batch["pose_gt"], tau)
            optimizer.zero_grad(set_to_none=True)
            out = model(v_feat, t_raw, t_phys, x_tau, tau, batch["config_id"], batch.get("session_id"))
            # Pose-only loss: L_pose only (lambda_traj = lambda_kp = lambda_Trec =
            # lambda_Vrec = lambda_con = 0).
            loss = F.mse_loss(out["x0_hat"], batch["pose_gt"])
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                log.warning("non-finite at step {}; skipped", step)
                continue
            loss.backward()
            optimizer.step()

            if step == 0 or (step + 1) % args.log_every == 0:
                want_sample = step == 0 or (step + 1) % args.sample_every == 0
                row = {
                    "lr": lr,
                    "step": step + 1,
                    "L_pose": float(loss.detach().item()),
                    "tau0_mpjpe_mm": _tau0_mpjpe_mm(model, v_feat, t_raw, t_phys, batch, device),
                    "ddim_mpjpe_mm": _ddim_mpjpe_mm(diffusion, model, v_feat, t_raw, t_phys, batch, device, sample_steps) if want_sample else None,
                }
                rows.append(row)
                print("lr=%g step=%d L_pose=%.6f tau0=%.1fmm%s"
                      % (lr, row["step"], row["L_pose"], row["tau0_mpjpe_mm"],
                         (" ddim=%.1fmm" % row["ddim_mpjpe_mm"]) if row["ddim_mpjpe_mm"] is not None else ""))
        all_rows.extend(rows)
        final = rows[-1]
        ddim_final = final["ddim_mpjpe_mm"] if final["ddim_mpjpe_mm"] is not None else [row["ddim_mpjpe_mm"] for row in rows if row["ddim_mpjpe_mm"] is not None][-1]
        verdict_ok = (
            final["L_pose"] < PASS_L_POSE
            and final["tau0_mpjpe_mm"] < PASS_TAU0_MPJPE_MM
            and ddim_final < PASS_DDIM_MPJPE_MM
        )
        per_lr[str(lr)] = {
            "final_L_pose": final["L_pose"],
            "final_tau0_mpjpe_mm": final["tau0_mpjpe_mm"],
            "final_ddim_mpjpe_mm": ddim_final,
            "verdict": "PASS" if verdict_ok else "FAIL",
        }
        if args.save_ckpt:
            torch.save({"model": model.state_dict(), "config": dict(config)}, out_dir / ("overfit_ckpt_%g.pt" % lr))
            log.info("wrote {}", out_dir / ("overfit_ckpt_%g.pt" % lr))

    csv_path = out_dir / "overfit_log.csv"
    fieldnames = ["lr", "step", "L_pose", "tau0_mpjpe_mm", "ddim_mpjpe_mm"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in fieldnames})
    plot_curves(all_rows, out_dir / "overfit_curves.png")
    log.info("wrote {}", csv_path)

    report = {
        "thresholds": {"L_pose": PASS_L_POSE, "tau0_mpjpe_mm": PASS_TAU0_MPJPE_MM, "ddim_mpjpe_mm": PASS_DDIM_MPJPE_MM},
        "per_lr": per_lr,
        "config": {"batch_windows": bsz, "steps": int(args.steps), "dropout": float(args.dropout),
                   "modal": str(args.modal), "contact_method": str(config["contact_method"]), "seed": int(args.seed),
                   "loss": "pose-only (L_pose; lambda_traj=lambda_kp=lambda_Trec=lambda_Vrec=lambda_con=0)"},
    }
    (out_dir / "overfit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote {}", out_dir / "overfit_report.json")

    print("== Test9 single-batch pose-only overfit ==")
    for lr_key, values in per_lr.items():
        print("  lr=%-8s %s  final L_pose=%.6f  tau0 MPJPE=%.1f mm  DDIM MPJPE=%.1f mm"
              % (lr_key, values["verdict"], values["final_L_pose"], values["final_tau0_mpjpe_mm"], values["final_ddim_mpjpe_mm"]))
    if all(values["verdict"] == "PASS" for values in per_lr.values()):
        print("  全部 lr 均 PASS -> 单 batch 能训到 ~0：姿态目标/管线正常；泛化差 = 欠训 / 条件弱（结合 Test7/Test8 解读）")
    elif any(values["verdict"] == "PASS" for values in per_lr.values()):
        print("  部分 lr PASS -> 目标/管线正常但收敛依赖 lr；用收敛的 lr 作为重训起点")
    else:
        print("  全部 FAIL -> 训不下去：目标/管线有问题；先回查 Test6 的 FK/单位，再看 Test8 的梯度死区")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
