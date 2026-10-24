#!/usr/bin/env python3
"""验证视觉↔动捕时钟修正的部署一致性（只读，不写任何文件）。

offset_correction.csv 是 10-22 已固化进 facts 的冻结时间配置（B 类 50 session
的 mocap 轴已按表平移）；lag_fleet.csv 是 S1 全库扫描终表（表的血缘证据）。
本脚本核对：fleet / splits / facts / 修正表四向集合一致、facts 内部恒等式与
帧数、facts 的 offset_s 与表期望值逐 session 一致、表内部行级规则与已知锚点。

口径（Baselines/决策/05_R1-R3_风险裁定.md §三，S1 全库扫描）：
  - DT = 0.025 s（40 Hz）
  - 符号约定 UV[i] ≈ kp[i + lag]；lag > 0 ⟺ 图像被配到"未来"的动捕姿态（GT 超前）
    → 修法 mocap_time_s[j] -= lag·dt，等价 offset_s_new = offset_s_old + lag_eff_frames × DT
  - A 已对齐：lag_eff 恒 0，不修；B 可平移修复：平移；C 修不动 / D 证据不足：不修，留空

内置断言（任一失败即非零退出）：
  1. fleet 行数 == 140，与修正表、splits（train/val/test 去重）、facts cam3
     目录树四向集合一致
  2. 全 140 session 恒等式 offset_s == visual_time_s[0] − mocap_time_s[0]，
     且 len(mocap_time_s) == session.json frame_count
  3. 部署一致性：B 类 facts offset_s == 表 offset_s_new；A/C/D 类 == 表 offset_s_old
  4. 表内部行级规则：A 类 lag_eff==0 且 new==old；B 类 new==old+lag_eff×DT；C/D 类 new 留空
  5. 三个已知值锚点：S5091 / S13112 / S12042（fleet 值 + 表值 + facts 值三方核对）

用法：python3 offset_verify.py   （可选 --table/--fleet/--facts-root/--splits 覆盖默认路径）
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DT = 0.025  # 40 Hz 帧间隔（秒）

HERE = Path(__file__).resolve().parent

# 已知值锚点（05_R1-R3_风险裁定.md §三 复算值）：
#   session: (conf_class, lag_eff_frames, offset_s_old, offset_s_new)
ANCHORS = {
    "S5091": ("B", +9.5, 0.631612, 0.869112),   # 0.631612 + 9.5×0.025
    "S13112": ("B", -9.0, 0.126726, -0.098274),  # 0.126726 − 9.0×0.025
    "S12042": ("B", -3.0, 0.326787, 0.251787),   # 0.326787 − 3.0×0.025
}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg):
    print(f"PASS: {msg}")


def read_fleet(path):
    """读 lag_fleet.csv → {session_id: row}，row 附加 fleet_line（文件行号，1 起）。"""
    with open(path, newline="") as f:
        rows = {}
        for lineno, r in enumerate(csv.DictReader(f), start=2):
            sid = r["session"].split("/")[-1]
            r["fleet_line"] = lineno
            rows[sid] = r
    return rows


def read_table(path):
    """读 offset_correction.csv → {session_id: row}。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            rows[r["session_id"]] = r
    return rows


def read_splits(path):
    """splits.csv train/val/test 三列去重 → set(session_id)。"""
    ids = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for c in ("train", "val", "test"):
                v = (r.get(c) or "").strip()
                if v:
                    ids.add(v)
    return ids


def scan_facts(root):
    """枚举 cam3/<date>/<subject>/<session>/session.json → {session_id: (date, subject)}。"""
    found = {}
    cam3 = root / "cam3"
    if not cam3.is_dir():
        fail(f"facts cam3 目录不存在: {cam3}")
    for date_d in sorted(cam3.iterdir()):
        if not date_d.is_dir():
            continue
        for subj_d in sorted(date_d.iterdir()):
            if not subj_d.is_dir():
                continue
            for sess_d in sorted(subj_d.iterdir()):
                sj = sess_d / "session.json"
                if sess_d.is_dir() and sj.is_file():
                    found[sess_d.name] = (date_d.name, subj_d.name)
    return found


def load_session(root, date, subject, sid):
    """读 session.json + frames.npz，并做恒等式与帧数校验。"""
    base = root / "cam3" / date / subject / sid
    with open(base / "session.json") as f:
        meta = json.load(f)
    frames = np.load(base / "frames.npz")
    offset_s = float(meta["offset_s"])
    ident = float(frames["visual_time_s"][0] - frames["mocap_time_s"][0])
    if abs(ident - offset_s) > 1e-6:
        fail(f"{sid}: 恒等式破坏 visual[0]−mocap[0]={ident:.9f} != offset_s={offset_s}")
    if len(frames["mocap_time_s"]) != int(meta["frame_count"]):
        fail(f"{sid}: len(mocap_time_s)={len(frames['mocap_time_s'])} != frame_count={meta['frame_count']}")
    return offset_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(HERE / "offset_correction.csv"))
    ap.add_argument("--fleet", default=str(HERE / "lag_fleet.csv"))
    ap.add_argument("--facts-root",
                    default=str(HERE.parents[1] / "shared" / "facts" / "sessions"))
    ap.add_argument("--splits",
                    default=str(HERE.parents[1] / "protocol" / "splits" / "default" / "splits.csv"))
    args = ap.parse_args()
    root = Path(args.facts_root)

    fleet = read_fleet(args.fleet)
    table = read_table(args.table)
    splits = read_splits(args.splits)
    facts = scan_facts(root)

    # 断言 1：四向集合一致
    if len(fleet) != 140:
        fail(f"fleet 行数 {len(fleet)} != 140")
    if len(table) != 140:
        fail(f"修正表行数 {len(table)} != 140")
    if len(splits) != 140:
        fail(f"splits 去重 session 数 {len(splits)} != 140")
    if len(facts) != 140:
        fail(f"facts 目录树 session 数 {len(facts)} != 140")
    for a, b, name in ((fleet, table, "fleet vs 表"), (fleet, splits, "fleet vs splits"),
                       (fleet, facts, "fleet vs facts"), (table, splits, "表 vs splits"),
                       (table, facts, "表 vs facts")):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        if only_a or only_b:
            fail(f"{name} 不一致: 仅前者 {only_a} / 仅后者 {only_b}")
    ok("四向集合一致: fleet=140 / 表=140 / splits=140 / facts=140，无缺失无多出")

    # 断言 2：全 140 恒等式 + 帧数
    facts_off = {}
    for sid, (date, subject) in facts.items():
        facts_off[sid] = load_session(root, date, subject, sid)
    ok("全 140 session 恒等式 offset_s == visual_time_s[0] − mocap_time_s[0] 且帧数一致")

    # 断言 4（表内部行级规则）+ 断言 3（部署一致性，顺带逐行核对）
    class_stat = {"A": 0, "B": 0, "C": 0, "D": 0}
    for sid in sorted(table):
        r = table[sid]
        cls = r["conf_class"]
        lag = float(r["lag_eff_frames"])
        old = float(r["offset_s_old"])
        new_s = r["offset_s_new"]
        class_stat[cls] = class_stat.get(cls, 0) + 1
        if cls == "A":
            if abs(lag) > 1e-9:
                fail(f"{sid}: A 类但 lag_eff={lag} 非零")
            new = old
            if new_s and abs(float(new_s) - new) > 1e-6:
                fail(f"{sid}: A 类 new={new_s} != old={new:.6f}")
        elif cls == "B":
            new = old + lag * DT
            if abs(float(new_s) - new) > 1e-6:
                fail(f"{sid}: B 类表复算 {new:.6f} != new 列 {new_s}")
        elif cls in ("C", "D"):
            new = old
            if new_s:
                fail(f"{sid}: {cls} 类 new 列应留空，实际 {new_s}")
        else:
            fail(f"{sid}: 未知 conf_class {cls}")
        if abs(facts_off[sid] - new) > 1e-6:
            fail(f"{sid}: 部署不一致 facts offset_s={facts_off[sid]:.6f} != 期望 {new:.6f}（{cls} 类）")
    ok("部署一致性: 140/140 facts offset_s == 表期望值（B 类=new / A/C/D 类=old）")
    ok("表内部行级规则: A lag==0 且 new==old / B new==old+lag×DT / C、D new 留空")

    # 断言 5：已知值锚点（fleet + 表 + facts 三方）
    for sid, (cls, lag, old, new) in ANCHORS.items():
        r = fleet.get(sid)
        if r is None:
            fail(f"锚点 {sid} 不在 fleet 表")
        if r["conf_class"] != cls:
            fail(f"锚点 {sid}: fleet conf_class={r['conf_class']} != {cls}")
        if abs(float(r["lag_eff_frames"]) - lag) > 1e-6:
            fail(f"锚点 {sid}: fleet lag_eff_frames={r['lag_eff_frames']} != {lag}")
        tr = table[sid]
        if abs(float(tr["offset_s_old"]) - old) > 1e-6:
            fail(f"锚点 {sid}: 表 old={tr['offset_s_old']} != {old}")
        if abs(float(tr["offset_s_new"]) - new) > 1e-6:
            fail(f"锚点 {sid}: 表 new={tr['offset_s_new']} != {new}")
        if abs(facts_off[sid] - new) > 1e-6:
            fail(f"锚点 {sid}: facts offset_s={facts_off[sid]:.6f} != {new}")
    ok("已知值锚点通过: S5091 0.631612→0.869112 / S13112 0.126726→−0.098274 / S12042 0.326787→0.251787")

    b_new = [float(table[s]["offset_s_new"]) for s in table if table[s]["conf_class"] == "B"]
    print(f"类别分布: {class_stat}")
    print(f"B 类 offset_s_new 范围: [{min(b_new):.6f}, {max(b_new):.6f}]（n={len(b_new)}）")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
