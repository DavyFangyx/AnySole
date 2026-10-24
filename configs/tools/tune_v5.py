#!/usr/bin/env python3
"""V5 先验塔配方的 Optuna 调参入口（2026-10-24，λ 格子消融第一轮）。

背景：V5_Av1 的 cam3 运行 T2M 塌缩（5-fold 138.6±80.8；生产轴 113.1 >
外部 T 基线 100.2），V5_Av4 的 λ 3/1 把 T2M 钉在 85–94 但 V 流从零、
occ050 兑现缩水（Δ踝足 −0.3~−4.2）。缺格 = V5 结构 + E 载体 + λ 3/1。
本入口把缺格扩成完整网格：E 载体冻结（V2M 逐位不变，归因干净），只
扫先验塔配方。

trial 结构（每 trial = 1 训练 + 2 评估）：
  1. 训练 occ050 配置（采 λ_prior_pose / λ_prior_traj，其余同 V5_Av1
     基座规格）——兑现只能在 occ 档训练出现；T2M 是 T-only 行、与视觉
     轴无关，occ050 配置的 T2M 即 E2 U@T 块读数。
  2. eval occ050 test → T2M PA（门槛 1）+ Δ踝足 VT−V（主目标）。
  3. eval cam3 test（同一 ckpt、换数据轴）→ 零税门槛 |ΔPA| ≤ 0.5，
     顺带读出 U@V 块 V2M PA。

目标 = 两道门槛内最大化兑现：
  门槛 1（E2 U@T）：T2M PA ≤ --t2m-gate（默认 95，对 100.2 基线留余量）
  门槛 2（零税契约）：cam3 |ΔPA| ≤ --zerotax-pa（默认 0.5）
  主目标：occ050 Δ踝足 VT−V，越小越好（Av1 生产轴参考 ≈ −14）
  目标值 = Δ踝足 + 线性惩罚（T2M 超门槛每 1mm 罚 5；|ΔPA| 超门槛每 1mm 罚 100）
  三原始数全部入 user_attrs + 终表，T2M-vs-Δ 散布直接回答
  「修 T2M 是否吃兑现」。

用法：
  python configs/tools/tune_v5.py --trials 8 --dry-run        # 只打印命令
  python configs/tools/tune_v5.py --trials 8 --device cuda    # 真跑
  CUDA_VISIBLE_DEVICES=N bash configs/bg_tune.sh <log> --trials 8   # 后台
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import optuna

# V5_Av1 基座规格（configs/z_gen/model_registry.py 同款；λ 对由 trial 采样）。
BASE_TRAIN_ARGS: list[tuple[str, str, ...]] = [
    ("--contact-method", "f6_soft"),
    ("--config-probs", "1.0", "0.0", "0.0"),
    ("--lambda-contact", "1.0"),
    ("--lambda-air", "2.0"),
    ("--lambda-phase", "1.0"),
    ("--lambda-cop", "1.0"),
    ("--lambda-trec", "0.0"),
    ("--lambda-vrec", "0.0"),
    ("--lambda-nll", "0.0"),
    ("--lr-warmup-frac", "0.05"),
    ("--grad-clip", "5.0"),
]

# 搜索空间（第一轮 = λ 格子；扩空间只改这一处）
PRIOR_POSE_GRID = [1.0, 2.0, 3.0, 4.0]
PRIOR_TRAJ_GRID = [0.0, 1.0]

DEFAULT_CARRIER = (
    "results/AnySole/V3_Ev1_f6_soft/"
    "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
    "checkpoints/ckpt_best.pt")

_EVAL_LINE = re.compile(r"epoch (\d+) eval:")
_T2M_VAL = re.compile(r"val/tau0_mpjpe/T=([0-9.eE+-]+)")


def build_train_cmd(trial_dir: Path, lp: float, lt: float,
                    args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "anysole.train",
           "--config", args.occ_config,
           "--device", args.device,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--seed", str(args.seed),
           "--out-dir", str(trial_dir / "ckpts"),
           "--carrier-from", args.carrier_from,
           "--lambda-prior-pose", "%.4g" % lp,
           "--lambda-prior-traj", "%.4g" % lt]
    for key, *vals in BASE_TRAIN_ARGS:
        cmd += [key, *vals]
    if args.limit_sessions:
        cmd += ["--limit-sessions", str(args.limit_sessions)]
    return cmd


def build_eval_cmd(config: str, ckpt: Path, out_dir: Path,
                   args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "anysole.eval",
            "--config", config,
            "--ckpt", str(ckpt),
            "--split", "test",
            "--device", args.device,
            "--contact-method", "f6_soft",
            "--metrics-out", str(out_dir / "test_brief.json"),
            "--protocol-out", str(out_dir / "test.json")]


def read_test_metrics(json_path: Path) -> dict[str, float]:
    """test.json → {t2m_pa, d_ankle, d_pa, v2m_pa}（Δ = VT−V）。"""
    m = json.load(open(json_path))["metrics"]
    vt, v, t = m["VT2M"], m["V2M"], m["T2M"]
    return {
        "t2m_pa": float(t["PA-MPJPE"]),
        "d_ankle": float(vt["MPJPE_anklefoot"]) - float(v["MPJPE_anklefoot"]),
        "d_pa": float(vt["PA-MPJPE"]) - float(v["PA-MPJPE"]),
        "v2m_pa": float(v["PA-MPJPE"]),
    }


def run_train(cmd: list[str], trial: optuna.Trial, trial_dir: Path) -> int:
    log = trial_dir / "train.log"
    with log.open("w") as out:
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=_subenv())
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            out.flush()
            m_ev = _EVAL_LINE.search(line)
            m_t = _T2M_VAL.search(line)
            if m_ev and m_t:
                epoch = int(m_ev.group(1))
                trial.report(float(m_t.group(1)), epoch)
                if trial.should_prune():
                    proc.kill()
                    proc.wait()
                    raise optuna.TrialPruned()
            elif line.startswith("saved best"):
                print("  [%s]" % line.strip(), flush=True)
        return proc.wait()


def _subenv() -> dict[str, str]:
    import os
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled",
                "ANYSOLE_WORKSPACE": str(REPO / "AnysoleWorkspace"),
                "ANYSOLE_RESULTS": str(REPO / "results")})
    return env


def objective(trial: optuna.Trial) -> float:
    lp = trial.suggest_categorical("lambda_prior_pose", PRIOR_POSE_GRID)
    lt = trial.suggest_categorical("lambda_prior_traj", PRIOR_TRAJ_GRID)
    trial_dir = (REPO / _ARGS.out_root / _ARGS.study_name /
                 ("trial_%04d_lp%s_lt%s" % (trial.number, lp, lt)))
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("lambda_prior_pose", lp)
    trial.set_user_attr("lambda_prior_traj", lt)

    print("trial %d: lp=%.4g lt=%.4g" % (trial.number, lp, lt), flush=True)
    train_cmd = build_train_cmd(trial_dir, lp, lt, _ARGS)
    print("  $ " + " ".join(train_cmd), flush=True)
    rc = run_train(train_cmd, trial, trial_dir)
    if rc != 0:
        trial.set_user_attr("status", "train_failed_rc%d" % rc)
        return 999.0

    ckpt = trial_dir / "ckpts" / "ckpt_best.pt"
    occ_dir = trial_dir / "eval_occ050"
    cam_dir = trial_dir / "eval_cam3"
    for config, out_dir in ((_ARGS.occ_config, occ_dir),
                            (_ARGS.cam3_config, cam_dir)):
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = build_eval_cmd(config, ckpt, out_dir, _ARGS)
        print("  $ " + " ".join(eval_cmd), flush=True)
        rc = subprocess.run(eval_cmd, cwd=REPO, env=_subenv()).returncode
        if rc != 0:
            trial.set_user_attr("status", "eval_failed_rc%d" % rc)
            return 999.0

    occ = read_test_metrics(occ_dir / "test.json")
    cam = read_test_metrics(cam_dir / "test.json")
    t2m, d_ankle = occ["t2m_pa"], occ["d_ankle"]
    d_pa_cam3 = cam["d_pa"]

    pen_t2m = max(0.0, t2m - _ARGS.t2m_gate) * 5.0
    pen_tax = max(0.0, abs(d_pa_cam3) - _ARGS.zerotax_pa) * 100.0
    value = d_ankle + pen_t2m + pen_tax

    for k, v in {"lambda_prior_pose": lp, "lambda_prior_traj": lt,
                 "t2m_pa": t2m, "d_ankle": d_ankle,
                 "d_pa_cam3": d_pa_cam3, "v2m_pa_cam3": cam["v2m_pa"],
                 "pen_t2m": pen_t2m, "pen_tax": pen_tax,
                 "feasible": int(pen_t2m == 0 and pen_tax == 0),
                 "status": "ok"}.items():
        trial.set_user_attr(k, v)
    print("  T2M=%.1f  Δ踝足=%+.1f  ΔPA(cam3)=%+.2f  obj=%.1f%s" %
          (t2m, d_ankle, d_pa_cam3, value,
           "" if pen_t2m == 0 and pen_tax == 0 else "  [不可行]"), flush=True)
    return value


def report(study: optuna.Study) -> None:
    rows = [t for t in study.trials if t.state.is_finished()]
    rows.sort(key=lambda t: t.value if t.value is not None else 1e9)
    print("\n=== 全部 trial（按目标值排序）===")
    print("%-5s %-6s %-6s %8s %9s %9s %8s %9s %s" %
          ("#", "lp", "lt", "T2M PA", "Δ踝足", "ΔPA c3", "V2M c3", "obj", "状态"))
    for t in rows:
        u = t.user_attrs
        if t.value is None:
            print("%-5d %-6s %-6s %s" % (t.number, u.get("lambda_prior_pose"),
                                         u.get("lambda_prior_traj"), t.state))
            continue
        print("%-5d %-6g %-6g %8.1f %+9.1f %+9.2f %8.1f %9.1f %s" %
              (t.number, u.get("lambda_prior_pose"), u.get("lambda_prior_traj"),
               u.get("t2m_pa", 0), u.get("d_ankle", 0), u.get("d_pa_cam3", 0),
               u.get("v2m_pa_cam3", 0), t.value,
               "✓" if u.get("feasible") else u.get("status", "✗")))
    summary = [{**{"number": t.number, "value": t.value}, **t.user_attrs}
               for t in rows]
    out = REPO / _ARGS.out_root / _ARGS.study_name / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nsummary: %s" % out)


def dry_run(args: argparse.Namespace) -> None:
    for i, (lp, lt) in enumerate((lpt for lpt in
                                  ((p, t) for p in PRIOR_POSE_GRID
                                   for t in PRIOR_TRAJ_GRID))):
        trial_dir = Path(args.out_root) / args.study_name / \
            ("trial_%02d_lp%s_lt%s" % (i, lp, lt))
        print("lp=%.4g lt=%.4g" % (lp, lt))
        print("  $ " + " ".join(
            build_train_cmd(trial_dir, lp, lt, args)))
        print("  $ " + " ".join(build_eval_cmd(
            args.occ_config, trial_dir / "ckpts" / "ckpt_best.pt",
            trial_dir / "eval_occ050", args)))
        print("  $ " + " ".join(build_eval_cmd(
            args.cam3_config, trial_dir / "ckpts" / "ckpt_best.pt",
            trial_dir / "eval_cam3", args)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-name", default="v5_prior_tune")
    ap.add_argument("--out-root", default="results/AnySole/optuna_tune")
    ap.add_argument("--occ-config", default="anysole/configs/v1_cam3_occ050.yaml")
    ap.add_argument("--cam3-config", default="anysole/configs/v1.yaml")
    ap.add_argument("--carrier-from", default=DEFAULT_CARRIER)
    ap.add_argument("--epochs", type=int, default=740)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-sessions", type=int, default=None,
                    help="冒烟：限制会话数")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--sampler", choices=("grid", "tpe"), default="grid")
    ap.add_argument("--pruner", choices=("median", "none"), default="none")
    ap.add_argument("--n-startup-trials", type=int, default=2)
    ap.add_argument("--t2m-gate", type=float, default=95.0,
                    help="T2M PA 门槛（外部 T 基线 100.2 留余量）")
    ap.add_argument("--zerotax-pa", type=float, default=0.5,
                    help="cam3 零税门槛 |ΔPA|")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印每个格子的命令，不训练")
    args = ap.parse_args()

    for p in (args.occ_config, args.cam3_config, args.carrier_from):
        if not (REPO / p).exists():
            print("路径不存在: %s" % p)
            return 2
    if args.dry_run:
        dry_run(args)
        return 0

    global _ARGS
    _ARGS = args
    out_root = (REPO / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///%s/%s.db" % (out_root, args.study_name)
    sampler: Any = (optuna.samplers.GridSampler(
        {"lambda_prior_pose": PRIOR_POSE_GRID,
         "lambda_prior_traj": PRIOR_TRAJ_GRID})
        if args.sampler == "grid" else optuna.samplers.TPESampler(seed=args.seed))
    pruner: Any = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials, n_warmup_steps=60)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, sampler=sampler,
        pruner=pruner, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=args.trials)
    report(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
