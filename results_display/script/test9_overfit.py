"""Test9: single-batch overfit — one fixed batch, dropout off, VT-only.

Trains a fresh AnySole model on ONE fixed batch of train windows with:
  * modality dropout off (config_id fixed to VT; the train-side equivalent of
    `--dropoutVT 0,0`),
  * model dropout off (--dropout 0.0, the nn.Dropout inside the transformers),
and drives the loss down as far as it goes.

If the loss reaches ~0 (and tau=0 / DDIM reconstruction MPJPE collapse on the
same batch), the target and the pipeline are sound — generalization failures
are then a matter of undertraining / weak conditioning (see Test7/Test8).  If
it cannot fit, the target or the pipeline is broken (see Test6).

Outputs under results_display/Test9_overfit/:
    overfit_log.csv       step, loss terms, tau0 MPJPE, DDIM MPJPE
    overfit_curves.png    loss + MPJPE curves
    overfit_report.json   final numbers + PASS/FAIL verdict
    overfit_ckpt.pt       only with --save-ckpt

Usage (run from the repository root):
    python results_display/script/test9_overfit.py
    python results_display/script/test9_overfit.py --steps 300 --log-every 25 --sample-every 150
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.diffusion import GaussianDiffusion  # noqa: E402
from anysole.geometry import fk_pose6d  # noqa: E402
from anysole.losses import compute_losses  # noqa: E402
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, POSE_DIM, assert_batch_shapes  # noqa: E402

# Verdict thresholds (same spirit as the Task-2 control run: ~300 steps of
# single-batch training reached L_pose 0.086 after the T_phys normalization).
PASS_L_POSE = 0.05
PASS_TAU0_MPJPE_MM = 20.0
PASS_DDIM_MPJPE_MM = 50.0


def _tau0_mpjpe_mm(model, v_feat, t_raw, t_phys, batch, device) -> float:
    """Clean GT pose input, tau=0: can the model reproduce its target at all?"""
    with torch.inference_mode():
        zero = torch.zeros(batch["pose_gt"].shape[0], device=device, dtype=torch.long)
        out0 = model(v_feat, t_raw, t_phys, batch["pose_gt"], zero, batch["config_id"], batch.get("session_id"))
        anchor = batch["trans_anchor"][:, None, :]
        kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
        return float(torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0


def _ddim_mpjpe_mm(diffusion, model, v_feat, t_raw, t_phys, batch, device, sample_steps) -> float:
    """Full DDIM sampling on the fixed batch (the inference path)."""
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
        zero = torch.zeros(bsz, device=device, dtype=torch.long)
        out = model(v_feat, t_raw, t_phys, pred, zero, batch["config_id"], batch.get("session_id"))
        anchor = batch["trans_anchor"][:, None, :]
        kp = fk_pose6d(pred, out["trans_hat"] + anchor, batch["offsets"], batch["parents"])
        return float(torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean().item()) * 1000.0


def plot_curves(rows: list, out_path: Path) -> None:
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key in ("loss_total", "L_pose", "L_traj", "L_kp", "L_con"):
        axes[0].plot(steps, [row[key] for row in rows], label=key)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log)")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(steps, [row["tau0_mpjpe_mm"] for row in rows], label="tau0 MPJPE")
    ddim_rows = [row for row in rows if row["ddim_mpjpe_mm"] is not None]
    if ddim_rows:
        axes[1].plot([row["step"] for row in ddim_rows], [row["ddim_mpjpe_mm"] for row in ddim_rows], label="DDIM MPJPE", marker="o")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MPJPE (mm)")
    axes[1].set_title("reconstruction MPJPE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test9: single-batch overfit (dropout off, VT-only).")
    cli_common.add_common_args(
        parser,
        session=False,
        split_csv=False,
        split=False,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        force=False,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "Test9_overfit",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "v1.yaml"))
    parser.add_argument("--contact-method", type=str, default="", help="Contact-label scheme (default: config's).")
    parser.add_argument("--modal", type=str, default=MODEL_ANYSOLEV1)
    parser.add_argument("--batch-size", type=int, default=256, help="Windows in the fixed batch.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--dropout", type=float, default=0.0, help="Model nn.Dropout rate (modality dropout is off by construction: VT-only).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--save-ckpt", action="store_true", help="Also save the overfit checkpoint.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    log.info("fixed batch: {} windows, steps={}, lr={}, dropout={}", bsz, args.steps, args.lr, args.dropout)

    model = AnySoleModel(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(args.dropout),
        modal=str(args.modal),
    ).to(device)
    model.train()
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(config["diffusion_sample_steps"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    v_feat, t_raw, t_phys = condition_inputs(batch, batch["config_id"])

    rows = []
    for step in range(int(args.steps)):
        tau = torch.randint(0, int(config["diffusion_train_steps"]), (bsz,), device=device, dtype=torch.long)
        x_tau = diffusion.q_sample(batch["pose_gt"], tau)
        optimizer.zero_grad(set_to_none=True)
        out = model(v_feat, t_raw, t_phys, x_tau, tau, batch["config_id"], batch.get("session_id"))
        losses = compute_losses(out, batch, batch["config_id"], config)
        if not (torch.isfinite(losses["loss"]) and torch.isfinite(out["F"]).all()):
            optimizer.zero_grad(set_to_none=True)
            log.warning("non-finite at step {}; skipped", step)
            continue
        losses["loss"].backward()
        optimizer.step()

        if step == 0 or (step + 1) % args.log_every == 0:
            want_sample = step == 0 or (step + 1) % args.sample_every == 0
            row = {
                "step": step + 1,
                "loss_total": float(losses["loss"].detach().item()),
                "L_pose": float(losses["L_pose"].detach().item()),
                "L_traj": float(losses["L_traj"].detach().item()),
                "L_kp": float(losses["L_kp"].detach().item()),
                "L_con": float(losses["L_con"].detach().item()),
                "L_Trec": float(losses["L_Trec"].detach().item()),
                "L_Vrec": float(losses["L_Vrec"].detach().item()),
                "tau0_mpjpe_mm": _tau0_mpjpe_mm(model, v_feat, t_raw, t_phys, batch, device),
                "ddim_mpjpe_mm": _ddim_mpjpe_mm(diffusion, model, v_feat, t_raw, t_phys, batch, device, sample_steps) if want_sample else None,
            }
            rows.append(row)
            print("step=%d loss=%.6f L_pose=%.6f L_traj=%.6f L_kp=%.6f tau0=%.1fmm%s"
                  % (row["step"], row["loss_total"], row["L_pose"], row["L_traj"], row["L_kp"], row["tau0_mpjpe_mm"],
                     (" ddim=%.1fmm" % row["ddim_mpjpe_mm"]) if row["ddim_mpjpe_mm"] is not None else ""))

    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "overfit_log.csv"
    fieldnames = ["step", "loss_total", "L_pose", "L_traj", "L_kp", "L_con", "L_Trec", "L_Vrec", "tau0_mpjpe_mm", "ddim_mpjpe_mm"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in fieldnames})
    plot_curves(rows, out_dir / "overfit_curves.png")
    log.info("wrote {}", csv_path)

    final = rows[-1]
    ddim_final = final["ddim_mpjpe_mm"] if final["ddim_mpjpe_mm"] is not None else [row["ddim_mpjpe_mm"] for row in rows if row["ddim_mpjpe_mm"] is not None][-1]
    verdict_ok = (
        final["L_pose"] < PASS_L_POSE
        and final["tau0_mpjpe_mm"] < PASS_TAU0_MPJPE_MM
        and ddim_final < PASS_DDIM_MPJPE_MM
    )
    report = {
        "verdict": "PASS" if verdict_ok else "FAIL",
        "thresholds": {"L_pose": PASS_L_POSE, "tau0_mpjpe_mm": PASS_TAU0_MPJPE_MM, "ddim_mpjpe_mm": PASS_DDIM_MPJPE_MM},
        "final": final,
        "config": {"batch_windows": bsz, "steps": int(args.steps), "lr": float(args.lr), "dropout": float(args.dropout),
                   "modal": str(args.modal), "contact_method": str(config["contact_method"]), "seed": int(args.seed)},
    }
    (out_dir / "overfit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote {}", out_dir / "overfit_report.json")
    if args.save_ckpt:
        torch.save({"model": model.state_dict(), "config": dict(config)}, out_dir / "overfit_ckpt.pt")
        log.info("wrote {}", out_dir / "overfit_ckpt.pt")

    print("== Test9 single-batch overfit verdict ==")
    print("  final: L_pose=%.6f  tau0 MPJPE=%.1f mm  DDIM MPJPE=%.1f mm" % (final["L_pose"], final["tau0_mpjpe_mm"], ddim_final))
    if verdict_ok:
        print("  PASS -> 单 batch 能训到 ~0：目标/管线正常；泛化差 = 欠训 / 条件弱（结合 Test7/Test8 的敏感度结果解读）")
    else:
        print("  FAIL -> 训不下去：目标/管线有问题；先回查 Test6 的 FK/单位，再看 Test8 的梯度死区")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
