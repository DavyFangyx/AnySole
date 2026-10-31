#!/usr/bin/env python3
"""One-step baseline evaluation: run the native entry, then export the unified
contract.

2026-10-05 user ruling: the export step is folded into evaluation — after the
native run produces its outputs, the unified archive is written automatically;
no standalone export invocation is needed.

Per model:
  motionpro       test_frappe writes the unified contract directly (T2-01a);
                  this wrapper only runs it and verifies split coverage.
  vp_mocap        run_full_mmvp (mmvp env) -> export_baseline_motion vp_mocap.
  pressure_toolkit  run_full_mmvp init_shape+fit (mmvp env) -> export_pressure.
  fpp_v2t         infer_smplcont (touch_gait) -> canonical export_v2t.
  wham/sam3db/gvhmr  package-saved predictions -> export_baseline_motion (no native run).

Examples:
  python AnysoleWorkspace/tool/eval_baseline.py --model vp_mocap --split test --gpu 1
  python AnysoleWorkspace/tool/eval_baseline.py --model pressure_toolkit --split all --gpu 3 --per-gpu 8
  python AnysoleWorkspace/tool/eval_baseline.py --model motionpro --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
BASELINES = ROOT / "Baselines"
TOUCH_GAIT = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
MMVP = "/data/fangyuxuan/miniconda3/envs/mmvp/bin/python"
MALES = "S5,S6,S7,S8,S10,S11,S12,S13"
FEMALES = "S14"


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def split_sessions(split: str) -> list[str]:
    rows = list(csv.DictReader(open(WORKSPACE / "protocol/splits/default/splits.csv", encoding="utf-8")))
    parts = ("train", "val", "test") if split == "all" else (split,)
    seen, out = set(), []
    for row in rows:
        for part in parts:
            sid = row.get(part)
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def eval_motionpro(args: argparse.Namespace) -> None:
    run([TOUCH_GAIT, "-m", "app.test_frappe"], cwd=BASELINES / "MotionPRO")
    root = ROOT / "results" / "baselines" / "MotionPRO" / "predictions" / "carpet_v2" / "eval_motion"
    missing = [s for s in split_sessions(args.split) if not (root / f"{s}.npz").is_file()]
    if missing:
        raise SystemExit(f"motionpro eval_motion missing {len(missing)}: {missing[:10]}")
    print(f"motionpro eval_motion verified: {len(split_sessions(args.split))} sessions")


def eval_vp_mocap(args: argparse.Namespace) -> None:
    cmd = [MMVP, "run_full_mmvp.py", "--split", args.split, "--gpu", str(args.gpu),
           "--no-visualization"]
    if args.force:
        cmd.append("--force")
    run(cmd, cwd=BASELINES / "VP-MoCap" / "PoseTransOpt")
    run([TOUCH_GAIT, str(WORKSPACE / "tool/export_baseline_motion.py"), "--model", "vp_mocap",
         "--split", args.split, "--force"], cwd=ROOT)


def eval_pressure_toolkit(args: argparse.Namespace) -> None:
    base = [MMVP, "run_full_mmvp.py", "--split", args.split, "--male", args.male,
            "--female", args.female, "--gpu", str(args.gpu), "--per-gpu", str(args.per_gpu)]
    run(base + ["--stage", "init_shape"], cwd=BASELINES / "pressure_tookit")
    run(base + ["--stage", "fit"], cwd=BASELINES / "pressure_tookit")
    run([TOUCH_GAIT, str(WORKSPACE / "tool/export_baseline_motion.py"), "--model",
         "pressure_toolkit", "--split", args.split, "--force"], cwd=ROOT)


def eval_fpp_v2t(args: argparse.Namespace) -> None:
    cfg = "configs/temporalKPSMPLCont_series5_mlp.yaml"
    phases = ("train", "val", "test") if args.split == "all" else (args.split,)
    cwd = BASELINES / "VP-MoCap" / "FPP-Net"
    env = os.environ.copy()
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    for phase in phases:
        run([TOUCH_GAIT, "-m", "app.infer_smplcont", "--config", cfg, "--phase", phase,
             "--batch_size", "32", "--num_threads", "4", "--no_visualization", "--gpus",
             str(args.gpu)], cwd=cwd, env=env)
    exportable = split_sessions(args.split)
    run([TOUCH_GAIT, str(WORKSPACE / "tool/adapters/mmvp_series/fpp/export_v2t.py"),
         "--sessions", ",".join(exportable), "--force"], cwd=ROOT)


def eval_visual(args: argparse.Namespace) -> None:
    # 决策 11（2026-10-29）：WHAM/SAM3DB/GVHMR 预测来自传输包已保存结果，
    # 无 native 运行步骤；一体入口 = 导出统一契约 + 覆盖校验。
    exportable = split_sessions(args.split)
    command = [TOUCH_GAIT, str(WORKSPACE / "tool/export_baseline_motion.py"),
               "--model", args.model, "--split", args.split, "--cam", str(args.cam),
               "--package-root", str(WORKSPACE / "work" / "visual_baselines_local"),
               "--force"]
    run(command, cwd=ROOT)
    root = ROOT / "results" / "baselines" / {"wham": "WHAM", "sam3db": "SAM3DB",
                                             "gvhmr": "GVHMR"}[args.model] / "predictions" \
        / args.variant / "eval_motion"
    missing = [s for s in exportable if not (root / f"{s}.npz").is_file()]
    if missing:
        raise SystemExit(f"{args.model} eval_motion missing {len(missing)}: {missing[:10]}")
    print(f"{args.model} eval_motion verified: {len(exportable)} sessions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=("motionpro", "vp_mocap", "pressure_toolkit", "fpp_v2t",
                                 "wham", "sam3db", "gvhmr"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU id (mmvp models)")
    parser.add_argument("--per-gpu", type=int, default=8, help="pressure_toolkit slots per GPU")
    parser.add_argument("--male", default=MALES)
    parser.add_argument("--female", default=FEMALES)
    parser.add_argument("--cam", type=int, choices=(1, 2, 3, 4), default=3,
                        help="camera manifest (visual baselines)")
    parser.add_argument("--variant", default="local_gen",
                        help="prediction variant subdir (visual baselines)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    {"motionpro": eval_motionpro, "vp_mocap": eval_vp_mocap,
     "pressure_toolkit": eval_pressure_toolkit, "fpp_v2t": eval_fpp_v2t,
     "wham": eval_visual, "sam3db": eval_visual, "gvhmr": eval_visual}[args.model](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
