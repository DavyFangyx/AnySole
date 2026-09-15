"""Test11: τ 训练区间消融 — τ≡0 恒等映射与 τ~U(0,100) 低噪声带训练.

Trains a fresh AnySole model on ONE fixed batch of train windows with:
  * modality dropout off (config_id fixed to VT; the train-side equivalent of
    `--dropoutVT 0,0`),
  * model dropout off (--dropout 0.0),
  * pose-only loss: L_pose only (lambda_traj = lambda_kp = lambda_Trec =
    lambda_Vrec = lambda_con = 0),
and sweeps learning rates (default 1e-3, 3e-4, 1e-4, 3e-5) for 3000 steps each
on a small fixed batch (default 8 windows), under these τ regimes:

  tau0         τ ≡ 0: x_tau = x0，任务退化为恒等映射 x0_hat = x_τ.
               若 L_pose 收敛不到 ~0（tau0 MPJPE 压不到个位数），说明
               x_tau → 输出根本没有可用带宽（分区 Linear 投影丢信息 /
               cross-attn 把残差流冲掉），门控不是主因，改结构的重点也要变.
  lowband      τ ~ U(0, tau_band_max)（默认 100）: 低噪声区训练. 若 τ0
               MPJPE 压到个位数 mm → 门控 + SNR 是主因，结构改法对症；
               压不下去 → 同上（结构瓶颈）.
  full         τ ~ U(0, diffusion_train_steps): 全区间对照（与 Test9 同
               设置），是 "压下去" 的参照.

任意 arm 可加 ``-nocond`` 后缀（或全局 --no-cond 自动追加）：V/T 置零，隔离
纯 x_tau → 输出路径。``lowband-nocond`` 压到个位数而 ``lowband`` 压不到 →
cross-attn 条件路径（F）是低噪声区通路的污染源，对 F 做 τ 感知门控对症；
两者都压不到 → τ 注入路径本身破坏干净输入直通，需给 x_tau 建 τ 门控高速通道。

监控：L_pose（训练损失）与 tau0 MPJPE（τ=0 干净 GT 姿态直通重建，GT 轨迹，
23 关节 FK，mm）。DDIM 仅在 full arm 采样（τ≡0 / 低噪声带训练出的模型没有
见过高噪声区，DDIM 无意义）。

判定阈值：
  identity PASS: L_pose < 1e-3 且 tau0 MPJPE < 5 mm（任一 lr 达到即可）
  lowband  PASS: tau0 MPJPE < 10 mm（个位数）

诊断逻辑：
  tau0 FAIL                       → 结构瓶颈：带宽被卡死，门控不是主因.
  tau0 PASS 且 lowband PASS       → 门控 + SNR 主因确认，结构改法对症.
  tau0 PASS 且 lowband FAIL       → 低噪声区也压不到个位数，带宽不足（同上）.

Outputs under results_display/Test11_tau_regime/:
    tau_regime_log.csv       arm, lr, step, L_pose, tau0_mpjpe_mm, ddim_mpjpe_mm
    tau_regime_curves.png    L_pose / tau0 MPJPE 曲线 + 各 arm 最终 tau0 柱状图
    tau_regime_report.json   各 arm × lr 最终数值 + PASS/FAIL 判定 + 诊断结论
    tau_regime_ckpt_<arm>_<lr>.pt  仅 --save-ckpt 时输出

Usage (run from the repository root):
    python results_display/script/test11_tau_regime.py
    python results_display/script/test11_tau_regime.py --arms tau0 --lrs 1e-3 --steps 1000
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

import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.diffusion import GaussianDiffusion  # noqa: E402
from anysole.geometry import fk_pose6d  # noqa: E402
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, POSE_DIM, assert_batch_shapes  # noqa: E402

# Verdict thresholds (fixed-batch pose-only fitting).
PASS_IDENTITY_L_POSE = 1.0e-3
PASS_IDENTITY_TAU0_MM = 5.0
PASS_LOWBAND_TAU0_MM = 10.0

ARMS = ("tau0", "lowband", "full", "tau0-nocond", "lowband-nocond", "full-nocond")
DEFAULT_ARMS = "tau0,lowband,full"
DEFAULT_LRS = "1e-3,3e-4,1e-4,3e-5"
DEFAULT_TAU_BAND_MAX = 100
NOCOND_SUFFIX = "-nocond"


def parse_lrs(value: str) -> List[float]:
    lrs = [float(item) for item in value.replace(";", ",").split(",") if item.strip()]
    if not lrs:
        raise ValueError("--lrs must contain at least one learning rate")
    return lrs


def parse_arms(value: str) -> List[str]:
    arms = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError("unknown arm(s) %r; expected subset of %s" % (unknown, ", ".join(ARMS)))
    if not arms:
        raise ValueError("--arms must contain at least one arm")
    return arms


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_tau(arm: str, bsz: int, tau_band_max: int, n_train_steps: int, device) -> torch.Tensor:
    base = arm.removesuffix(NOCOND_SUFFIX)
    if base == "tau0":
        return torch.zeros(bsz, device=device, dtype=torch.long)
    if base == "lowband":
        return torch.randint(0, int(tau_band_max), (bsz,), device=device, dtype=torch.long)
    return torch.randint(0, int(n_train_steps), (bsz,), device=device, dtype=torch.long)


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


def plot_curves(rows: list, arms: List[str], out_path: Path) -> None:
    lrs = sorted({row["lr"] for row in rows})
    arm_colors = {arm: "C%d" % (i % 10) for i, arm in enumerate(arms)}
    lr_styles = {lr: style for lr, style in zip(lrs, ("-", "--", "-.", ":"))}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for arm in arms:
        for lr in lrs:
            group = [row for row in rows if row["arm"] == arm and row["lr"] == lr]
            if not group:
                continue
            label = "%s lr=%g" % (arm, lr)
            steps = [row["step"] for row in group]
            axes[0].plot(steps, [row["L_pose"] for row in group], label=label,
                         color=arm_colors[arm], linestyle=lr_styles[lr])
            axes[1].plot(steps, [row["tau0_mpjpe_mm"] for row in group], label=label,
                         color=arm_colors[arm], linestyle=lr_styles[lr])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L_pose (log)")
    axes[0].set_title("pose-only loss per τ regime")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_yscale("log")
    axes[1].axhline(PASS_LOWBAND_TAU0_MM, color="r", linestyle="--", linewidth=1)
    axes[1].axhline(PASS_IDENTITY_TAU0_MM, color="orange", linestyle=":", linewidth=1)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("tau0 MPJPE (mm, log)")
    axes[1].set_title("τ=0 clean-input reconstruction (10mm low-band / 5mm identity thresholds)")
    axes[1].legend(fontsize=7, ncol=2)

    # Final tau0 MPJPE per arm (best lr), against the two thresholds.
    finals = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        best = min(arm_rows, key=lambda row: row["tau0_mpjpe_mm"])
        finals[arm] = best["tau0_mpjpe_mm"]
    if finals:
        names = list(finals)
        bars = axes[2].bar(names, [finals[name] for name in names],
                           color=[arm_colors[name] for name in names])
        axes[2].axhline(PASS_LOWBAND_TAU0_MM, color="r", linestyle="--", linewidth=1)
        axes[2].axhline(PASS_IDENTITY_TAU0_MM, color="orange", linestyle=":", linewidth=1)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("final tau0 MPJPE (mm, log)")
        axes[2].set_title("best-lr final τ0 MPJPE per arm")
        for bar, value in zip(bars, finals.values()):
            axes[2].text(bar.get_x() + bar.get_width() / 2, value * 1.15,
                         "%.1f" % value, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def arm_verdict(arm: str, final: dict) -> str:
    base = arm.removesuffix(NOCOND_SUFFIX)
    if base == "tau0":
        ok = final["L_pose"] < PASS_IDENTITY_L_POSE and final["tau0_mpjpe_mm"] < PASS_IDENTITY_TAU0_MM
    elif base == "lowband":
        ok = final["tau0_mpjpe_mm"] < PASS_LOWBAND_TAU0_MM
    else:
        return "reference"
    return "PASS" if ok else "FAIL"


def diagnose(per_arm: dict) -> str:
    def passed(arm: str) -> bool:
        return any(entry["verdict"] == "PASS" for entry in per_arm.get(arm, {}).values())

    def present(arm: str) -> bool:
        return arm in per_arm

    identity_ok = passed("tau0") or passed("tau0-nocond")
    lowband_ok = passed("lowband")
    lowband_nocond_ok = passed("lowband-nocond")
    has_identity = present("tau0") or present("tau0-nocond")
    has_lowband = present("lowband")
    if has_identity and not identity_ok:
        return ("结构瓶颈：τ≡0 恒等映射收敛不到 ~0，x_tau → 输出没有可用带宽"
                "（分区 Linear 投影丢信息 / cross-attn 把残差流冲掉），门控不是主因，改结构重点要相应调整")
    if has_lowband and lowband_ok:
        return "门控 + SNR 主因确认：低噪声区训练能把 τ0 MPJPE 压到个位数，结构改法（门控）对症"
    if has_lowband and not lowband_ok:
        if present("lowband-nocond") and lowband_nocond_ok:
            return ("低噪声区无条件下压到个位数、真实 VT 条件下压不到：cross-attn 条件路径（F）是"
                    "低噪声区通路的污染源 → 对 F 做 τ 感知门控对症")
        return "低噪声区训练也压不到个位数：τ/噪声注入路径本身破坏干净输入直通，门控条件 F 不够，需给 x_tau 建 τ 门控高速通道"
    return "信息不足：诊断需要 tau0 与 lowband 两个 arm，请用 --arms tau0,lowband 运行"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test11: τ training-regime ablation (τ≡0 identity + low-noise band).")
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
        out_dir_default=cli_common.DISPLAY_ROOT / "Test11_tau_regime",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "v1.yaml"))
    parser.add_argument("--contact-method", type=str, default="", help="Contact-label scheme (default: config's).")
    parser.add_argument("--modal", type=str, default=MODEL_ANYSOLEV1)
    parser.add_argument("--batch-size", type=int, default=8, help="Windows in the fixed batch.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lrs", type=str, default=DEFAULT_LRS, help="Comma-separated learning-rate sweep.")
    parser.add_argument("--arms", type=str, default=DEFAULT_ARMS, help="Comma-separated arms: %s." % ", ".join(ARMS))
    parser.add_argument("--tau-band-max", type=int, default=DEFAULT_TAU_BAND_MAX, help="Lowband arm upper bound: tau ~ U(0, N).")
    parser.add_argument("--no-cond", action="store_true", help="Also run the V/T-zeroed (-nocond) twin of every selected arm, isolating the pure x_tau path.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Model nn.Dropout rate (modality dropout is off by construction: VT-only).")
    parser.add_argument("--seed", type=int, default=0, help="Reseeded per (arm, lr) so every point starts from the same init.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500, help="DDIM sampling interval (full arm only).")
    parser.add_argument("--save-ckpt", action="store_true", help="Also save one checkpoint per (arm, lr).")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    arms = parse_arms(args.arms)
    if args.no_cond:
        # Append the V/T-zeroed twin of every selected arm (e.g. --no-cond turns
        # tau0,lowband,full into tau0,tau0-nocond,lowband,lowband-nocond,...).
        twins = [
            arm + NOCOND_SUFFIX
            for arm in list(arms)
            if not arm.endswith(NOCOND_SUFFIX) and arm + NOCOND_SUFFIX not in arms
        ]
        arms = arms + twins
    log.info("fixed batch: {} windows, steps={}, lrs={}, arms={}, dropout={}",
             bsz, args.steps, lrs, arms, args.dropout)

    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    n_train_steps = int(config["diffusion_train_steps"])
    sample_steps = int(config["diffusion_sample_steps"])
    v_feat, t_raw, t_phys = condition_inputs(batch, batch["config_id"])

    all_rows = []
    per_arm = {}
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        per_arm[arm] = {}
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
            # Conditioning per arm: a -nocond suffix zeroes V/T so the model only
            # has x_tau to work with; every other arm keeps the real VT input.
            if arm.endswith(NOCOND_SUFFIX):
                cond_v = torch.zeros_like(v_feat)
                cond_t_raw = torch.zeros_like(t_raw)
                cond_t_phys = torch.zeros_like(t_phys)
            else:
                cond_v, cond_t_raw, cond_t_phys = v_feat, t_raw, t_phys
            log.info("arm sweep: arm={} lr={}", arm, lr)
            rows = []
            for step in range(int(args.steps)):
                tau = sample_tau(arm, bsz, int(args.tau_band_max), n_train_steps, device)
                if arm.startswith("tau0"):
                    x_tau = batch["pose_gt"]
                else:
                    x_tau = diffusion.q_sample(batch["pose_gt"], tau)
                optimizer.zero_grad(set_to_none=True)
                out = model(cond_v, cond_t_raw, cond_t_phys, x_tau, tau, batch["config_id"], batch.get("session_id"))
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
                    want_ddim = arm == "full" and (step == 0 or (step + 1) % args.sample_every == 0)
                    row = {
                        "arm": arm,
                        "lr": lr,
                        "step": step + 1,
                        "L_pose": float(loss.detach().item()),
                        "tau0_mpjpe_mm": _tau0_mpjpe_mm(model, cond_v, cond_t_raw, cond_t_phys, batch, device),
                        "ddim_mpjpe_mm": _ddim_mpjpe_mm(diffusion, model, cond_v, cond_t_raw, cond_t_phys, batch, device, sample_steps) if want_ddim else None,
                    }
                    rows.append(row)
                    print("%s lr=%g step=%d L_pose=%.6f tau0=%.1fmm%s"
                          % (arm, lr, row["step"], row["L_pose"], row["tau0_mpjpe_mm"],
                             (" ddim=%.1fmm" % row["ddim_mpjpe_mm"]) if row["ddim_mpjpe_mm"] is not None else ""))
            all_rows.extend(rows)
            final = rows[-1]
            per_arm[arm][str(lr)] = {
                "final_L_pose": final["L_pose"],
                "final_tau0_mpjpe_mm": final["tau0_mpjpe_mm"],
                "best_L_pose": min(row["L_pose"] for row in rows),
                "best_tau0_mpjpe_mm": min(row["tau0_mpjpe_mm"] for row in rows),
                "verdict": arm_verdict(arm, final),
            }
            if arm == "full":
                ddim_final = final["ddim_mpjpe_mm"]
                if ddim_final is None:
                    ddim_final = [row["ddim_mpjpe_mm"] for row in rows if row["ddim_mpjpe_mm"] is not None][-1]
                per_arm[arm][str(lr)]["final_ddim_mpjpe_mm"] = ddim_final
            if args.save_ckpt:
                torch.save({"model": model.state_dict(), "config": dict(config)},
                           out_dir / ("tau_regime_ckpt_%s_%g.pt" % (arm, lr)))
                log.info("wrote {}", out_dir / ("tau_regime_ckpt_%s_%g.pt" % (arm, lr)))

    csv_path = out_dir / "tau_regime_log.csv"
    fieldnames = ["arm", "lr", "step", "L_pose", "tau0_mpjpe_mm", "ddim_mpjpe_mm"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in fieldnames})
    plot_curves(all_rows, arms, out_dir / "tau_regime_curves.png")
    log.info("wrote {}", csv_path)

    diagnosis = diagnose(per_arm)
    report = {
        "thresholds": {"identity_L_pose": PASS_IDENTITY_L_POSE, "identity_tau0_mpjpe_mm": PASS_IDENTITY_TAU0_MM,
                       "lowband_tau0_mpjpe_mm": PASS_LOWBAND_TAU0_MM},
        "per_arm": per_arm,
        "diagnosis": diagnosis,
        "config": {"batch_windows": bsz, "steps": int(args.steps), "dropout": float(args.dropout),
                   "modal": str(args.modal), "contact_method": str(config["contact_method"]), "seed": int(args.seed),
                   "arms": arms, "tau_band_max": int(args.tau_band_max),
                   "loss": "pose-only (L_pose; lambda_traj=lambda_kp=lambda_Trec=lambda_Vrec=lambda_con=0)"},
    }
    (out_dir / "tau_regime_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote {}", out_dir / "tau_regime_report.json")

    print("== Test11 τ 训练区间消融 ==")
    for arm in arms:
        for lr_key, values in per_arm[arm].items():
            extra = ("  final DDIM MPJPE=%.1f mm" % values["final_ddim_mpjpe_mm"]) if "final_ddim_mpjpe_mm" in values else ""
            print("  %-13s lr=%-8s %-9s final L_pose=%.6f  tau0 MPJPE=%.1f mm%s"
                  % (arm, lr_key, values["verdict"], values["final_L_pose"], values["final_tau0_mpjpe_mm"], extra))
    print("诊断结论：%s" % diagnosis)
    if "full" in per_arm:
        full_best = min(per_arm["full"].values(), key=lambda values: values["final_tau0_mpjpe_mm"])
        print("参照：full 全区间对照 best tau0 MPJPE=%.1f mm（同 Test9 设置，供 '压下去' 参照）" % full_best["final_tau0_mpjpe_mm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
