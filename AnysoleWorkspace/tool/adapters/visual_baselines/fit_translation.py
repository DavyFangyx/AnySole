#!/usr/bin/env python3
"""视觉基线 世界系平移拟合（与 MMVP 漂移点 m/n 同机制，2026-11-01 用户裁定）。

三个视觉基线的导出链 camera→checkerboard→mocap world 与动捕世界原点之间存在
常数偏差（WHAM/GVHMR ~+2.5m 同向、SAM3DB ~0.5m）——标定链世界端锚定于棋盘格
标定测量，非动捕原点。本脚本逐日期逐模型用已导出预测根与 smpl GT 骨盆的
**池化最小二乘**拟合平移 b，落盘
``protocol/calibration/<date>.<model>_native_to_mocap.json``；
导出器（export_wham/export_sam3db/export_gvhmr）经 ``common.load_translation``
消费（无 artifact 的日期 = 零修正）。

b 逐日期在该日期**已导出且未经修正**的预测上拟合（3 参数全局平移，拟合来源与
会话清单登记于 artifact）：首次导出时该日期无 artifact = 零修正，拟合后重新
导出即带上 b。已有 artifact 的日期再拟合会读到已修正的预测（残差 ≈ 0），
故需 --force 显式覆盖。
覆盖现状：test = 20260810；act8020 的 val 口径另含 20260804 / 20260808
（2026-11-05 WHAM 补跑，19 个会话）。

用法（touch_gait）：
  python AnysoleWorkspace/tool/adapters/visual_baselines/fit_translation.py \
      --date 20260810 --model wham
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
CALIBRATION_DIR = WORKSPACE / "protocol/calibration"
RESULTS = REPO_ROOT / "results"
MODEL_DIRS = {
    "wham": "WHAM",
    "sam3db": "SAM3DB",
    "gvhmr": "GVHMR",
}


def _manifest_rows() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def _sessions_of_date(date: str) -> list[str]:
    """One recording date's sessions, across every split column.

    The membership table stays the canonical session source; the date of a
    session comes from its recording path (``common.session_parts``, the same
    helper the exporters use). Scanning all columns is what makes val/train
    dates fittable — the test-only era read the ``test`` column alone.
    """
    from AnysoleWorkspace.tool.adapters.visual_baselines import common

    rows = _manifest_rows()
    out = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for split in ("train", "val", "test"):
                value = (row.get(split) or "").strip()
                if not value or value in out or value not in rows:
                    continue
                if common.session_parts(rows[value])[0] == date:
                    out.append(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_DIRS), required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing artifact (see the warning below)")
    args = parser.parse_args()

    out = CALIBRATION_DIR / f"{args.date}.{args.model}_native_to_mocap.json"
    if out.is_file() and not args.force:
        # b is fitted on *pre-correction* predictions. Once an artifact exists the
        # exporter folds b into the npz, so refitting on those would measure the
        # residual (~0) and overwrite a good artifact. Re-export without the
        # artifact first if a refit is really wanted.
        raise SystemExit(f"{out} exists; refit needs --force (fit consumes pre-correction predictions)")

    from Baselines.utils.gt_loading import protocol_gt

    rows = _manifest_rows()
    sessions = [s for s in _sessions_of_date(args.date) if s in rows]
    pred_root = RESULTS / "baselines" / MODEL_DIRS[args.model] / "predictions" / "local_gen" / "eval_motion"
    deltas: list[np.ndarray] = []
    used = []
    for sid in sessions:
        path = pred_root / f"{sid}.npz"
        if not path.is_file():
            continue
        d = np.load(path, allow_pickle=False)
        mask = np.asarray(d["valid_mask"], dtype=bool)
        pred = np.asarray(d["joint_xyz_world"])[mask][:, 0]
        gt, _ = protocol_gt(rows[sid], "smpl24")
        g = gt[mask][:, 0]
        deltas.append(g - pred)
        used.append(sid)
    if not deltas:
        raise SystemExit(f"no predictions for {args.date} ({args.model})")
    delta = np.concatenate(deltas, axis=0)
    b = delta.mean(axis=0)
    residual = np.linalg.norm(delta - b, axis=1)
    payload = {
        "date": args.date,
        "model": args.model,
        "b": [float(v) for v in b],
        "residual_rmse_mm": float(np.sqrt((residual ** 2).mean()) * 1000.0),
        "n_sessions": len(used),
        "sessions": used,
        "fit_source": "exported local_gen predictions (pre-correction) vs smpl GT pelvis, pooled least squares",
        "note": "3-parameter global translation; per-date; same mechanism as MMVP drift points m/n (04_E §8.4)",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"b = {b}  residual RMSE {payload['residual_rmse_mm']:.1f} mm  ({len(used)} sessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
