#!/usr/bin/env python3
"""corrected_time —— 按 offset_correction.csv 平移动捕时间轴（只读 shared facts，不写任何文件）。

用途：后续「纯触觉测试」用修正后的动捕时间轴采样 GT/SMPL，消除每 session 的时钟残差。

规则（与 build_offset_table.py / offset_correction.csv 同口径）：
  - A 类（已对齐）：mocap_time_s 原样返回
  - B 类（可平移修复）：返回 mocap_time_s − lag_eff_frames × 0.025
  - C 类（修不动）/ D 类（证据不足）：原样返回；调用方务必用 class_of() 查询并自行处置

典型用法（触觉测试采 GT）：
    from corrected_time import corrected_mocap_time, class_of

    t_mocap = corrected_mocap_time("S5091")       # ndarray，B 类已平移
    if class_of("S5091") in ("C", "D"):
        ...  # 该 session 无有效修正，按原时间轴 + 登记处置
    gt_on_video_clock = smpl_gt(t_mocap)          # 在修正后时间轴上采样 GT

自检（__main__）：S5091 首帧修正前后差 −0.2375、长度=帧数等（输出贴入 README）。
"""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DT = 0.025  # 40 Hz 帧间隔（秒）

HERE = Path(__file__).resolve().parent


def _find_repo(start):
    p = start
    for _ in range(8):
        if (p / "AnysoleWorkspace").is_dir() and (p / "z_note").is_dir():
            return p
        p = p.parent
    raise RuntimeError(f"未找到仓库根（从 {start} 向上找 AnysoleWorkspace 失败）")


REPO = _find_repo(HERE)
FACTS_ROOT = REPO / "AnysoleWorkspace/shared/facts/sessions"  # 其下 cam3/<date>/<subject>/<session>/
TABLE_CSV = HERE / "offset_correction.csv"                    # build_offset_table.py 产出


@lru_cache(maxsize=1)
def load_table(path=None):
    """读 offset_correction.csv → {session_id: row}（row 为 dict，数值列转 float）。"""
    p = Path(path) if path else TABLE_CSV
    if not p.is_file():
        raise FileNotFoundError(f"{p} 不存在；先运行 z_note/data_offset_fixpipe/build_offset_table.py 生成")
    with open(p, newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            row = dict(r)
            row["lag_eff_frames"] = float(row["lag_eff_frames"])
            row["offset_s_old"] = float(row["offset_s_old"])
            if row["offset_s_new"]:
                row["offset_s_new"] = float(row["offset_s_new"])
            else:
                row["offset_s_new"] = None
            rows[row["session_id"]] = row
    return rows


@lru_cache(maxsize=1)
def _session_index(root=None):
    """枚举 cam3 目录树 → {session_id: (date, subject)}。"""
    root = Path(root) if root else FACTS_ROOT
    idx = {}
    cam3 = root / "cam3"
    if not cam3.is_dir():
        raise FileNotFoundError(f"facts cam3 目录不存在: {cam3}")
    for date_d in sorted(cam3.iterdir()):
        if not date_d.is_dir():
            continue
        for subj_d in sorted(date_d.iterdir()):
            if not subj_d.is_dir():
                continue
            for sess_d in sorted(subj_d.iterdir()):
                if sess_d.is_dir() and (sess_d / "session.json").is_file():
                    idx[sess_d.name] = (date_d.name, subj_d.name)
    return idx


def class_of(session_id, table_path=None):
    """返回 conf_class（'A'/'B'/'C'/'D'）。"""
    return load_table(table_path)[session_id]["conf_class"]


def lag_eff_frames(session_id, table_path=None):
    """返回裁定的有效滞后（帧；A 类恒 0）。"""
    return load_table(table_path)[session_id]["lag_eff_frames"]


def corrected_offset_s(session_id, table_path=None):
    """修正后 offset_s：B 类 = old + lag×0.025；A 类 = old；C/D 类 = None。"""
    row = load_table(table_path)[session_id]
    return row["offset_s_new"]


def corrected_mocap_time(session_id, root=None, table_path=None):
    """读 shared facts 的 mocap_time_s，按修正表平移后返回（np.ndarray，浮点）。

    B 类：mocap_time_s − lag_eff_frames × 0.025；
    A 类：原样；C/D 类：原样（调用方应先用 class_of() 查询类标记）。
    只读 shared facts，不写任何文件。
    """
    date, subject = _session_index(root)[session_id]
    base = (Path(root) if root else FACTS_ROOT) / "cam3" / date / subject / session_id
    frames = np.load(base / "frames.npz")
    t = np.asarray(frames["mocap_time_s"], dtype=np.float64).copy()
    cls = class_of(session_id, table_path=table_path)
    if cls == "B":
        t = t - lag_eff_frames(session_id, table_path=table_path) * DT
    return t


def _selfcheck():
    """模块自检：锚点 session 的平移量、长度、类标记。"""
    anchors = [
        ("S5091", "B", +9.5, -0.2375),   # 首帧修正前后差 = −lag×dt
        ("S13112", "B", -9.0, +0.2250),
        ("S12042", "B", -3.0, +0.0750),
        ("S5011", "A", 0.0, 0.0),        # A 类原样
        ("S5013", "D", 27.5, 0.0),       # D 类原样（lag_eff 仅排序参考，不应用）
        ("S11031", "C", -16.5, 0.0),     # C 类原样
    ]
    print("corrected_time 自检")
    print("-" * 60)
    all_ok = True
    for sid, expect_cls, expect_lag, expect_shift in anchors:
        cls = class_of(sid)
        lag = lag_eff_frames(sid)
        t_raw = np.load(
            FACTS_ROOT / "cam3" / _session_index()[sid][0] / _session_index()[sid][1] / sid / "frames.npz"
        )["mocap_time_s"]
        t = corrected_mocap_time(sid)
        shift = float(t[0] - t_raw[0])
        frame_count = int(json.load(
            open(FACTS_ROOT / "cam3" / _session_index()[sid][0] / _session_index()[sid][1] / sid / "session.json")
        )["frame_count"])
        ok_line = (cls == expect_cls and abs(lag - expect_lag) < 1e-9
                   and abs(shift - expect_shift) < 1e-9 and len(t) == frame_count)
        all_ok &= ok_line
        print(f"{sid}: class={cls} lag_eff={lag:+.1f}f 首帧差={shift:+.6f}s "
              f"len={len(t)}==frame_count({frame_count})  [{'OK' if ok_line else 'FAIL'}]")
    print("-" * 60)
    print("S5091 断言: 首帧修正前后差 == −0.2375 且长度 == frame_count",
          "→ OK" if all_ok else "→ FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _selfcheck()
