"""f6_soft_offfix vs f6_soft 标签一致性自检（2026-10-20，offset 修正轴审计）。

检查全部 140 session（splits 并集）：
  - A/C/D 类（shift=0）：两版标签必须逐位全等；
  - B 类（shift≠0）：给出差值统计（差异帧数/格数/每脚、首末差异帧位置、
    平均差异帧位置 vs 期望平移 lag_eff_frames）；
  - 抽样 6 会话明细（3 A：S13011/S14022/S5011；3 B：S5091/S13112/S12042）。

只读：不写任何标签。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO_ROOT))

from anysole.data.workspace_adapter import LABEL_ROOT, PROTOCOL_SPLIT  # noqa: E402
from anysole.data.dataset import load_offset_correction  # noqa: E402

DETAIL_A = ("S13011", "S14022", "S5011")
DETAIL_B = ("S5091", "S13112", "S12042")


def main() -> int:
    rows = list(csv.DictReader(PROTOCOL_SPLIT.open(encoding="utf-8-sig")))
    ids = []
    for col in ("train", "val", "test"):
        ids += [r[col].strip() for r in rows if r[col].strip()]
    ids = sorted(set(ids))
    assert len(ids) == 140, len(ids)

    shifts = load_offset_correction()
    n_zero_identical, n_zero_diff = 0, []
    n_shift_identical, n_shift_diff = 0, []
    stats = {}
    detail = {}
    for sid in ids:
        pa = LABEL_ROOT / "f6_soft" / f"{sid}.npz"
        pb = LABEL_ROOT / "f6_soft_offfix" / f"{sid}.npz"
        a = np.load(pa)["contact"]
        b = np.load(pb)["contact"]
        shift = shifts.get(sid, 0.0)
        identical = bool(np.array_equal(a, b))
        frame_diff = (a != b).any(axis=1)
        cell_diff = int((a != b).sum())
        n_frames = len(a)
        lag_frames = shift / 0.025
        st = {
            "identical": identical,
            "shift_s": shift,
            "lag_eff_frames": lag_frames,
            "n_frames": n_frames,
            "n_diff_frames": int(frame_diff.sum()),
            "n_diff_cells": cell_diff,
            "n_diff_left": int((a[:, 0] != b[:, 0]).sum()),
            "n_diff_right": int((a[:, 1] != b[:, 1]).sum()),
        }
        if frame_diff.any():
            idx = np.where(frame_diff)[0]
            st["first_diff_frame"] = int(idx[0])
            st["last_diff_frame"] = int(idx[-1])
            st["mean_diff_frame"] = float(idx.mean())
        stats[sid] = st
        if shift == 0.0:
            if identical:
                n_zero_identical += 1
            else:
                n_zero_diff.append(sid)
        else:
            if identical:
                n_shift_identical += 1
            else:
                n_shift_diff.append(sid)
        if sid in DETAIL_A or sid in DETAIL_B:
            detail[sid] = st

    payload = {
        "n_sessions": len(ids),
        "n_zero_shift": n_zero_identical + len(n_zero_diff),
        "zero_shift_identical": n_zero_identical,
        "zero_shift_diff": n_zero_diff,
        "n_shift": n_shift_identical + len(n_shift_diff),
        "shift_identical": n_shift_identical,
        "shift_diff": n_shift_diff,
        "detail_sampled": detail,
        "per_session": stats,
    }
    out = REPO_ROOT / "z_note" / "data_offset_fixpipe" / "审计_F1_CoP" / "f6_soft_offfix_selfcheck.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("== 全量 140 session 自检")
    print("shift=0（A/C/D）: %d 全等, 不一致=%s" % (n_zero_identical, n_zero_diff))
    print("shift!=0（B）: %d 全等, 不一致=%s" % (n_shift_identical, n_shift_diff))
    print()
    print("== 抽样明细")
    for sid, st in detail.items():
        print("  %s shift=%+.4fs lag=%+.1ff n=%d identical=%s diff_frames=%d "
              "(L %d / R %d cells=%d) first=%s last=%s mean=%.1f"
              % (sid, st["shift_s"], st["lag_eff_frames"], st["n_frames"],
                 st["identical"], st["n_diff_frames"], st["n_diff_left"],
                 st["n_diff_right"], st["n_diff_cells"],
                 st.get("first_diff_frame"), st.get("last_diff_frame"),
                 st.get("mean_diff_frame", float("nan"))))
    print()
    print("== B 类 50 会话 diff 帧数分布（n_diff_frames vs lag_eff_frames）")
    bs = [(sid, s) for sid, s in stats.items() if s["shift_s"] != 0]
    for sid, s in sorted(bs, key=lambda x: -x[1]["n_diff_frames"]):
        print("  %-8s lag=%+6.1ff diff_frames=%4d/%4d  (L %d / R %d)" % (
            sid, s["lag_eff_frames"], s["n_diff_frames"], s["n_frames"],
            s["n_diff_left"], s["n_diff_right"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
