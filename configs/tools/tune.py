#!/usr/bin/env python3
"""Optuna 调参驱动（2026-10-25 泛化版，取代 tune_v5.py）。

模型 × 参数空间 = configs/tools/tune_space.py（唯一编辑点）；每个 trial =
训练一个模型的一个参数点（multi_metric ckpt 机制，ckpt规则.md v2），训练
结束直接读 multi_metric manifest 的 val 记录算目标值——零额外评估。

目标值（与 ckpt规则 v2 的正式选点规则同构）：
  候选 = manifest 里 gate_pass=true 的 epoch 权重；
  四阈值 = 三门（V ≤ 40 / T ≤ 90 / VT ≤ 40，读 manifest gates）
           + 税门 ΔPA = val PA(VT) − val PA(V) ≤ --dpa-gate（默认 0.5）；
  目标 = 全过候选中 VT PA 最小；无人通过 → 999（该参数点报告失败，与
  正式规则「无人通过 → 不选」一致）。
  正式选点（checkpoint_review 的 val 十指标复评）仍只在最终候选上做——
  本目标值是调参用的轻量代理，不替代正式选点。

用法：
  python configs/tools/tune.py --dry-run                  # 打印采样计划
  python configs/tools/tune.py --models V7_Av1,V7_Bv1 --trials 20
  python configs/tools/tune.py --models V7_Av1 --defaults-only   # 只跑默认点
"""

from __future__ import annotations

import argparse
import glob
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

from configs.tools.tune_space import MODEL_SPACE
from configs.tools.runner import _train_command
from configs.tools.common import load_registry

_EVAL_LINE = re.compile(r"epoch (\d+) eval:")
_VT_VAL = re.compile(r"val/tau0_mpjpe/VT=([0-9.eE+-]+)")


def sample_params(trial: optuna.Trial, model: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in MODEL_SPACE[model].items():
        kind = spec["kind"]
        if kind == "categorical":
            out[key] = trial.suggest_categorical(
                "%s:%s" % (model, key), spec["values"])
        elif kind == "float":
            out[key] = trial.suggest_float(
                "%s:%s" % (model, key), spec["low"], spec["high"],
                log=bool(spec.get("log", False)))
        elif kind == "int":
            out[key] = trial.suggest_int(
                "%s:%s" % (model, key), spec["low"], spec["high"],
                step=spec.get("step", 1))
        else:
            raise ValueError("unknown kind %r for %s:%s" % (kind, model, key))
    return out


def default_params(model: str) -> dict[str, Any]:
    return {k: v["default"] for k, v in MODEL_SPACE[model].items()}


def find_manifest(model: str, started: float) -> Path | None:
    import time
    best: Path | None = None
    for p in glob.glob(str(REPO / "results/AnySole" / ("%s_f6_soft" % model) /
                           "*/checkpoints/multi_metric/manifest.json")):
        q = Path(p)
        if q.stat().st_mtime >= started and (best is None or
                                             q.stat().st_mtime > best.stat().st_mtime):
            best = q
    return best


def objective_from_manifest(manifest_path: Path, dpa_gate: float) -> float:
    d = json.load(open(manifest_path))
    gates = d["gates"]["val_pa_max_mm"]
    best_vt: float | None = None
    n_pass = 0
    for cand in d.get("candidates", {}).values():
        if not cand.get("gate_pass"):
            continue
        s = cand["scores"]
        dpa = s["VT/PA-MPJPE"] - s["V/PA-MPJPE"]
        if s["V/PA-MPJPE"] <= gates["V"] and s["T/PA-MPJPE"] <= gates["T"] \
                and s["VT/PA-MPJPE"] <= gates["VT"] and dpa <= dpa_gate:
            n_pass += 1
            best_vt = s["VT/PA-MPJPE"] if best_vt is None \
                else min(best_vt, s["VT/PA-MPJPE"])
    return best_vt if best_vt is not None else 999.0


def _subenv() -> dict[str, str]:
    import os
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled",
                "ANYSOLE_WORKSPACE": str(REPO / "AnysoleWorkspace"),
                "ANYSOLE_RESULTS": str(REPO / "results")})
    return env


def run_train(cmd: list[str], trial: optuna.Trial,
              trial_dir: Path) -> int:
    with (trial_dir / "train.log").open("w") as out:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=_subenv())
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            out.flush()
            m_ev = _EVAL_LINE.search(line)
            m_vt = _VT_VAL.search(line)
            if m_ev and m_vt:
                trial.report(float(m_vt.group(1)), int(m_ev.group(1)))
                if trial.should_prune():
                    proc.kill()
                    proc.wait()
                    raise optuna.TrialPruned()
        return proc.wait()


def objective(trial: optuna.Trial) -> float:
    import time
    models = _ARGS.models
    model = (trial.suggest_categorical("model", models)
             if len(models) > 1 else models[0])
    params = (default_params(model) if _ARGS.defaults_only
              else sample_params(trial, model))
    started = time.time()
    trial.set_user_attr("model", model)
    trial.set_user_attr("params", json.dumps(params))

    registry = load_registry()
    cmd = _train_command(registry, model, train_overrides=params)
    trial_dir = REPO / _ARGS.out_root / _ARGS.study_name / \
        ("trial_%04d_%s" % (trial.number, model))
    trial_dir.mkdir(parents=True, exist_ok=True)
    # 每 trial 独立输出目录（ckpt规则 §3：新规则 run 不与历史目录混写），
    # 并强制 multi_metric 候选机制（旧基座条目可能未带）。
    cmd += ["--epochs", str(_ARGS.epochs),
            "--out-dir", str(trial_dir / "ckpts"),
            "--checkpoint-policy", "multi_metric",
            "--checkpoint-role", "auto",
            "--wandb_mode", "disabled"]
    if _ARGS.batch_size:
        cmd += ["--batch-size", str(_ARGS.batch_size)]
    if _ARGS.limit_sessions:
        cmd += ["--limit-sessions", str(_ARGS.limit_sessions)]

    print("trial %d %s %s" % (trial.number, model, params), flush=True)
    print("  $ " + " ".join(cmd), flush=True)
    rc = run_train(cmd, trial, trial_dir)
    if rc != 0:
        trial.set_user_attr("status", "train_failed_rc%d" % rc)
        return 999.0

    manifest = (trial_dir / "ckpts" / "multi_metric" / "manifest.json")
    if not manifest.is_file():
        manifest = find_manifest(model, started)
    if manifest is None or not manifest.is_file():
        trial.set_user_attr("status", "manifest_missing")
        return 999.0
    value = objective_from_manifest(manifest, _ARGS.dpa_gate)
    trial.set_user_attr("manifest", str(manifest))
    trial.set_user_attr("status", "ok" if value < 999 else "no_candidate")
    print("  value %.2f (%s)" % (value, manifest), flush=True)
    return value


def report(study: optuna.Study) -> None:
    rows = [t for t in study.trials if t.state.is_finished()]
    rows.sort(key=lambda t: t.value if t.value is not None else 1e9)
    print("\n=== trials（按目标值排序）===")
    for t in rows:
        u = t.user_attrs
        if t.value is None:
            print("%-4d %-8s %s" % (t.number, u.get("model", "?"), t.state))
            continue
        print("%-4d %-8s %-6.2f  %s  %s" %
              (t.number, u.get("model", "?"), t.value,
               u.get("params", ""), u.get("status", "")))
    out = REPO / _ARGS.out_root / _ARGS.study_name / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{**{"number": t.number, "value": t.value}, **t.user_attrs}
         for t in rows], indent=2, ensure_ascii=False))
    print("\nsummary: %s" % out)


def dry_run(args: argparse.Namespace) -> None:
    registry = load_registry()
    for model in args.models:
        params = default_params(model)
        cmd = _train_command(registry, model, train_overrides=params)
        cmd += ["--epochs", str(args.epochs)]
        print("%s defaults %s" % (model, params))
        print("  $ " + " ".join(cmd))
        if args.trials > 1 and not args.defaults_only:
            print("  # 采样空间（其余 trial 在网格/TPE 中采样）")
            for k, spec in MODEL_SPACE[model].items():
                print("  #   %s: %s" % (k, spec))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODEL_SPACE),
                    help="逗号分隔模型列表（默认全部 MODEL_SPACE 条目）")
    ap.add_argument("--study-name", default="v7_tune")
    ap.add_argument("--out-root", default="results/AnySole/optuna_tune")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--sampler", choices=("tpe", "grid"), default="tpe")
    ap.add_argument("--pruner", choices=("median", "none"), default="none")
    ap.add_argument("--n-startup-trials", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=740)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-sessions", type=int, default=None,
                    help="冒烟：限制会话数")
    ap.add_argument("--dpa-gate", type=float, default=0.5,
                    help="税门 ΔPA = val PA(VT)−val PA(V) 上限")
    ap.add_argument("--defaults-only", action="store_true",
                    help="每个模型只跑默认参数点（基线对照）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印命令，不训练")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in args.models:
        if m not in MODEL_SPACE:
            print("模型 %s 不在 MODEL_SPACE；可用：%s"
                  % (m, ", ".join(MODEL_SPACE)))
            return 2
    if args.dry_run:
        dry_run(args)
        return 0

    global _ARGS
    _ARGS = args
    out_root = (REPO / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///%s/%s.db" % (out_root, args.study_name)
    sampler: Any = (optuna.samplers.TPESampler(seed=1)
                    if args.sampler == "tpe" else None)
    if args.sampler == "grid":
        spaces: dict[str, list[Any]] = {}
        if len(args.models) > 1:
            spaces["model"] = args.models
        for model in args.models:
            for k, spec in MODEL_SPACE[model].items():
                if spec["kind"] == "categorical":
                    spaces["%s:%s" % (model, k)] = spec["values"]
        sampler = optuna.samplers.GridSampler(spaces)
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
