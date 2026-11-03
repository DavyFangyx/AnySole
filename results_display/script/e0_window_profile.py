#!/usr/bin/env python3
"""E0 窗内帧号剖面（修改 01「评估标尺补全」，2026-11-03）：jerk / PA 按
窗内帧号（tw=20，frame % 20）拆解 + 接缝分解。

- 输入：results/cells/<cell>/predictions/eval_motion/<session>_<mode>.npz
  的 ``joint_xyz_world``（display 帧 +Z 上）+ ``valid_mask`` +
  ``frame_indices``；GT 行读 AnySoleDataset kp_gt（native 世界系）。
- jerk 剖面：逐帧 j[t] = FPS³·(J[t] − 2·J[t−1] + J[t−2])（t ≥ 2，帧号取
  len(pr) 前补 +2），‖j‖ 24 关节均值，按窗内帧号 %20 分桶、桶内帧数加权
  均值。全帧/非接缝/接缝 = 20 个桶均值的等权均值（与主会话 2026-11-03
  实测口径逐位一致；原实测脚本 np.diff 链首段 n=0 为恒等 → 冻结量实为
  二阶差分 ×FPS³，见下方「定义对照」）。
- 接缝帧 = 窗内帧号 ∈ {18, 19, 0, 1}（4/20 帧）；接缝占总量 =
  4·接缝均值/(20·全帧均值)。jerk 剖面沿用实测口径全帧参与（不筛
  valid_mask——冻结表五行即此口径）。
- PA 剖面：逐有效帧（valid_mask）pa_mpjpe(pred, gt[frame_indices])
  （metrics.py 逐帧相似对齐），同桶均值；PA 全帧 = 20 桶等权均值；
  PA 非接缝 = 去 {0,19} 桶（主会话「接缝不吃 PA」口径）。
- GT 行必出（jerk 均匀参照；PA 恒为 0）。
- 定义对照（detail json）：同桶聚合下真三阶差分（字面口径）的值与表内
  冻结量并列，供口径裁定。

产物（幂等）：
- results_display/Etest/E0_ReferenceFloors/Full_Table2_E0_WindowProfile.csv
- results_display/Etest/E0_ReferenceFloors/detail/e0_window_profile.json
- results_display/Etest/receipts/e0_window_profile_receipt.json（sha256）

用法（touch_gait env）：
  python results_display/script/e0_window_profile.py \
      --cells V3_Bv1,V7_Cv5_s1,V7_Cv5AR2M_s1,V7_Cv5AR2R_s1 \
      --mode T2M,V2M,VT2M
（--mode 支持逗号分隔多模式同表输出；缺省 T2M 复现冻结五行。）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import FPS, HRNET_CACHE_ROOT, SEQ_ROOT, SPLIT_CSV  # noqa: E402
from anysole.utils.metrics import pa_mpjpe  # noqa: E402

ETEST = REPO / "results_display" / "Etest"
OUT_DIR = ETEST / "E0_ReferenceFloors"
DETAIL = OUT_DIR / "detail"
RECEIPTS = ETEST / "receipts"
CELLS_ROOT = REPO / "results" / "cells"
CONTACT_METHOD = "f6_soft_time_accel_state_smpl24_v1"
CONTACT_VERSION = "state_smpl24"
SEAM_POS = (18, 19, 0, 1)        # 接缝帧（jerk 分解）
PA_DROP_POS = (0, 19)            # PA 非接缝 = 去 {0,19}（主会话口径）
DEFAULT_CELLS = ("V3_Bv1", "V7_Cv5_s1", "V7_Cv5AR2M_s1", "V7_Cv5AR2R_s1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gt() -> dict:
    """test 会话 kp_gt（native 世界系；display 转换由调用方决定——jerk 为
    旋转不变量、PA 逐帧对齐亦不变，与主会话实测一致直接使用原生帧）。"""
    ids = load_split_ids(SPLIT_CSV, "test")
    ds = AnySoleDataset(
        mode="eval", seq_root=SEQ_ROOT, split_csv=SPLIT_CSV,
        cache_root=HRNET_CACHE_ROOT, window_length=20, session_ids=ids,
        contact_method=CONTACT_METHOD, allow_missing_video=True)
    return {str(s["session_id"]): np.asarray(
        s["kp_gt"], np.float64).reshape(-1, 24, 3) for s in ds.sessions}


def jerk_frozen(J: np.ndarray) -> np.ndarray:
    """冻结实测口径：j[t] = FPS³·(J[t] − 2·J[t−1] + J[t−2])，t ∈ [2, T)。

    主会话实测脚本为三段 np.diff 链（首段 n=0 恒等），数值上即此二阶差分
    ×FPS³；帧号 = len(pr) 前补 +2。返回 (T,) 关节均值 ‖j‖（m·FPS³）。
    """
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] < 3:
        return np.full(J.shape[0], np.nan)
    v = np.diff(J, 0, axis=0) * FPS          # 恒等（冻结口径）
    a = np.diff(v, axis=0) * FPS
    j = np.diff(a, axis=0) * FPS
    return np.linalg.norm(j, axis=-1).mean(axis=-1)


def jerk_third(J: np.ndarray) -> np.ndarray:
    """字面口径：真三阶差分，j[t] = FPS³·(J[t+1] − 3·J[t] + 3·J[t−1] −
    J[t−2])，t ∈ [2, T−1)（np.diff 三连，帧号前补 +2）。"""
    J = np.asarray(J, dtype=np.float64)
    if J.shape[0] < 4:
        return np.full(J.shape[0], np.nan)
    v = np.diff(J, axis=0) * FPS
    a = np.diff(v, axis=0) * FPS
    j = np.diff(a, axis=0) * FPS
    return np.linalg.norm(j, axis=-1).mean(axis=-1)


def _bucket(pr: np.ndarray, frame_index: np.ndarray) -> dict:
    """逐帧值按窗内帧号 %20 分桶；返回 {pos: (sum, count)}。"""
    acc = {p: [0.0, 0] for p in range(20)}
    ok = np.isfinite(pr)
    pos = frame_index % 20
    for p in range(20):
        m = ok & (pos == p)
        acc[p] = [float(pr[m].sum()), int(m.sum())]
    return acc


def _bucket_concat(parts: list) -> np.ndarray:
    """跨会话分桶拼接（桶内帧数加权），返回 20 桶均值。"""
    acc = {p: [0.0, 0] for p in range(20)}
    for part in parts:
        for p in range(20):
            acc[p][0] += part[p][0]
            acc[p][1] += part[p][1]
    return np.array([acc[p][0] / acc[p][1] if acc[p][1] else np.nan
                     for p in range(20)])


def _summary(profile: np.ndarray, seam_pos: tuple = SEAM_POS) -> dict:
    """全帧/非接缝/接缝 = 20 桶等权均值；占总量 = 4·接缝/(20·全帧)。"""
    m = np.ones(20, bool)
    m[list(seam_pos)] = False
    all_mean = float(profile.mean())
    non = float(profile[m].mean())
    seam = float(profile[~m].mean())
    share = 100.0 * 4.0 * seam / (20.0 * all_mean) if all_mean else float("nan")
    return {"all": all_mean, "non_seam": non, "seam": seam,
            "seam_share_pct": share}


def profile_row(name: str, sources: list) -> dict:
    """一行的 jerk（冻结+字面对照）与 PA 剖面。

    sources = [(J (T,24,3), valid (T,) 或 None, frame_index (T,), gt (T,24,3)
    或 None), ...]；gt 缺省（模型行）时 PA 由 npz 与 GT 配对，GT 行 PA=0。
    """
    jerk_f_parts, jerk_t_parts, pa_parts = [], [], []
    for J, valid, frame_index, gt in sources:
        if J.shape[0] < 8:
            continue
        jf = jerk_frozen(J)
        jt = jerk_third(J)
        k = np.arange(len(jf)) + 2
        jerk_f_parts.append(_bucket(jf, k))
        kt = np.arange(len(jt)) + 2
        jerk_t_parts.append(_bucket(jt, kt))
        if gt is None:
            continue
        # PA：逐有效帧 pa_mpjpe（pred vs gt[frame_indices]）
        if frame_index.max() >= len(gt):
            continue
        v = (valid if valid is not None else np.ones(J.shape[0], bool))
        v = v & np.isfinite(frame_index)
        if not v.any():
            continue
        err = pa_mpjpe(J[v], gt[frame_index[v]])
        kp = frame_index[v]
        pa_parts.append(_bucket(err, kp))
    jerk_f = _bucket_concat(jerk_f_parts)
    jerk_t = _bucket_concat(jerk_t_parts)
    out = {
        "name": name,
        "jerk_profile": jerk_f,
        "jerk": _summary(jerk_f),
        "jerk_third_diff_profile": jerk_t,
        "jerk_third_diff": _summary(jerk_t),
    }
    if pa_parts:
        pa = _bucket_concat(pa_parts)
        m = np.ones(20, bool)
        m[list(PA_DROP_POS)] = False
        out["pa_profile"] = pa
        out["pa"] = {"all": float(pa.mean()),
                     "non_seam": float(pa[m].mean())}
    else:
        out["pa_profile"] = np.zeros(20)
        out["pa"] = {"all": 0.0, "non_seam": 0.0}   # GT 行：均匀参照
    return out


def load_cell(cell: str, mode: str, gt: dict) -> list:
    sources = []
    files = []
    for sid, g in gt.items():
        f = CELLS_ROOT / cell / "predictions" / "eval_motion" / (
            "%s_%s.npz" % (sid, mode))
        if not f.is_file():
            continue
        d = np.load(f)
        J = np.asarray(d["joint_xyz_world"], np.float64)
        valid = (np.asarray(d["valid_mask"]).astype(bool)
                 if "valid_mask" in d.files else None)
        fi = np.asarray(d["frame_indices"], np.int64)
        sources.append((J, valid, fi, g))
        files.append(f)
    return sources, files


def write_table(rows: list) -> Path:
    header = (["模型", "模式"]
              + ["jerk_p%02d" % i for i in range(20)]
              + ["jerk_全帧", "jerk_非接缝", "jerk_接缝", "接缝占总量%"]
              + ["PA_p%02d" % i for i in range(20)]
              + ["PA_全帧", "PA_非接缝(去{0,19})"])
    out_rows = []
    for r in rows:
        line = [r["name"], r["mode"]]
        line += ["%.1f" % v for v in r["jerk_profile"]]
        line += ["%.1f" % r["jerk"]["all"], "%.1f" % r["jerk"]["non_seam"],
                 "%.1f" % r["jerk"]["seam"], "%.1f" % r["jerk"]["seam_share_pct"]]
        line += ["%.1f" % v for v in r["pa_profile"]]
        line += ["%.1f" % r["pa"]["all"], "%.1f" % r["pa"]["non_seam"]]
        out_rows.append(line)
    path = OUT_DIR / "Full_Table2_E0_WindowProfile.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print("[csv] %s (%d rows)" % (path, len(out_rows)))
    return path


def write_json(rows: list, cells: list, modes: list, files_by_cell: dict) -> Path:
    payload = {
        "schema_version": "e0_window_profile_v1",
        "split": "test",
        "modes": modes,
        "contact_version": CONTACT_VERSION,
        "contact_method": CONTACT_METHOD,
        "fps": float(FPS),
        "window_length": 20,
        "seam_positions": list(SEAM_POS),
        "pa_drop_positions": list(PA_DROP_POS),
        "jerk_definition": (
            "冻结实测口径：j[t] = FPS³·(J[t] − 2·J[t−1] + J[t−2])（主会话"
            "实测脚本 np.diff 链首段 n=0 恒等 → 二阶差分 ×FPS³；帧号前补"
            "+2）。字面「三阶差分」对照见各行 jerk_third_diff。"),
        "gt_source": ("AnySoleDataset(mode=eval, session_ids=test 列) kp_gt，"
                      "24 关节世界系（jerk 旋转不变、PA 逐帧对齐不变）"),
        "rows": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items()}
            for r in rows],
        "sources": files_by_cell,
        "gt_sources": [{"path": str(SPLIT_CSV), "sha256": _sha256(SPLIT_CSV)}],
    }
    path = DETAIL / "e0_window_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("[json] %s" % path)
    return path


def write_detail_md(rows: list, modes: list) -> Path:
    lines = ["# E0 窗内帧号剖面说明（修改 01）", ""]
    lines.append("口径：tw=20，窗内帧号 = frame % 20；jerk 逐帧分桶、桶内帧数加权，"
                 "全帧/非接缝/接缝 = 20 桶等权均值；PA 逐有效帧（valid_mask）"
                 "pa_mpjpe(pred, gt[frame_indices]) 同桶均值。模式 = "
                 + ",".join(modes) + "；GT 行每模式必出（jerk 均匀参照，PA 恒 0）。")
    lines.append("")
    lines.append("jerk 冻结口径（与主会话 2026-11-03 实测逐位一致）：j[t] = "
                 "FPS³·(J[t] − 2·J[t−1] + J[t−2])（t ≥ 2）。注意：实测脚本的 "
                 "np.diff 链首段 n=0 为恒等，冻结量实为**二阶差分 ×FPS³**，与"
                 "文档字面「三阶差分」不符；字面真三阶差分的同桶聚合值已并列"
                 "写入 detail json（jerk_third_diff），供口径裁定。")
    lines.append("")
    lines.append("接缝帧 = 窗内帧号 ∈ {18, 19, 0, 1}（4/20 帧）；接缝占总量 = "
                 "4·接缝均值/(20·全帧均值)。")
    lines.append("")
    lines.append("| 模型 | jerk 全帧 | 非接缝 | 接缝 | 接缝占总量% | PA 全帧 | PA 非接缝 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            r["name"], r["jerk"]["all"], r["jerk"]["non_seam"],
            r["jerk"]["seam"], r["jerk"]["seam_share_pct"],
            r["pa"]["all"], r["pa"]["non_seam"]))
    lines.append("")
    lines.append("判读：Jitter 大部分长在窗口接缝 4 帧上（GT 接缝占总量仅 "
                 "~19% 即均匀分布；模型行 74–91% 集中在接缝）。PA 剖面不受接缝"
                 "影响（Cv5 全帧 89.4 / 去 {0,19} 后 89.2），即「接缝不吃 PA」。")
    path = DETAIL / "E0_窗内剖面说明.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[md] %s" % path)
    return path


def write_receipt(cells: list, artifacts: list[Path],
                  files_by_cell: dict) -> Path:
    sources = [{"cell": "splits.csv", "path": str(SPLIT_CSV),
                "sha256": _sha256(SPLIT_CSV)}]
    for key in sorted(files_by_cell):
        for f in files_by_cell[key]:
            sources.append({"cell": key, "path": f["path"],
                            "sha256": f["sha256"]})
    payload = {
        "split": "test",
        "contact_version": CONTACT_VERSION,
        "contact_method": CONTACT_METHOD,
        "sources": sources,
        "artifacts": [{"path": str(p), "sha256": _sha256(p)}
                      for p in artifacts],
    }
    path = RECEIPTS / "e0_window_profile_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("[receipt] %s" % path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                    help="逗号分隔 cell 名（缺省 = 实测四模型）")
    ap.add_argument("--mode", default="T2M",
                    help="逗号分隔模式 T2M|V2M|VT2M（缺省 = T2M；多模式"
                         "同表输出）")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    for m in modes:
        if m not in ("T2M", "V2M", "VT2M"):
            raise SystemExit("未知模式 %s；可选 T2M,V2M,VT2M" % m)

    print("[E0-window] build GT ...")
    gt = build_gt()
    sids = sorted(gt)
    print("[E0-window] GT sessions=%d" % len(gt))
    rows = []
    files_by_cell = {}
    for mode in modes:
        # GT 行（均匀参照；PA 恒 0）
        rows.append(profile_row("GT", [(gt[sid], None,
                                        np.arange(len(gt[sid])), gt[sid])
                                       for sid in sids]))
        rows[-1]["mode"] = mode
        for cell in cells:
            sources, files = load_cell(cell, mode, gt)
            if not sources:
                print("[skip] %s: 无 %s npz" % (cell, mode))
                continue
            key = "%s/%s" % (cell, mode)
            files_by_cell[key] = [{"path": str(f), "sha256": _sha256(f)}
                                  for f in files]
            rows.append(profile_row(cell, sources))
            rows[-1]["mode"] = mode
            r = rows[-1]
            print("[E0-window] %-16s %s jerk 全帧 %.1f 非接缝 %.1f 接缝 %.1f "
                  "占总量 %.1f%% | PA 全帧 %.1f 非接缝 %.1f"
                  % (cell, mode, r["jerk"]["all"], r["jerk"]["non_seam"],
                     r["jerk"]["seam"], r["jerk"]["seam_share_pct"],
                     r["pa"]["all"], r["pa"]["non_seam"]))
            print("            第三差分对照: 全帧 %.1f 非接缝 %.1f 接缝 %.1f "
                  "占总量 %.1f%%" % (r["jerk_third_diff"]["all"],
                                    r["jerk_third_diff"]["non_seam"],
                                    r["jerk_third_diff"]["seam"],
                                    r["jerk_third_diff"]["seam_share_pct"]))
    artifacts = [write_table(rows), write_json(rows, cells, modes,
                                               files_by_cell),
                 write_detail_md(rows, modes)]
    write_receipt(cells, artifacts, files_by_cell)
    print("[E0-window] done -> %s" % OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
