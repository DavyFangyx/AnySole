#!/usr/bin/env python3
"""生成 offset_correction.csv：视觉↔动捕逐 session 时钟修正表（可复现）。

输入（只读）：
  - lag 终表  z_note/对齐扫描归档/s1_scratch/lag_fleet.csv
      （s80_final.py 产出；列 lag_eff_frames = 已裁定的有效滞后，conf_class ∈ A/B/C/D）
  - shared facts  AnysoleWorkspace/shared/facts/sessions/cam3/<date>/<subject>/<session>/
      session.json（offset_s） + frames.npz（visual_time_s / mocap_time_s / frame_id）
  - splits  AnysoleWorkspace/protocol/splits/default/splits.csv（train/val/test 共 140 session）

口径（Baselines/决策/05_R1-R3_风险裁定.md §三，S1 全库扫描）：
  - DT = 0.025 s（40 Hz）
  - 符号约定 UV[i] ≈ kp[i + lag]；lag > 0 ⟺ 图像被配到"未来"的动捕姿态（GT 超前）
    → 修法 mocap_time_s[j] -= lag·dt，等价 offset_s_new = offset_s_old + lag_eff_frames × DT
  - A 已对齐：lag_eff 恒 0，不修；B 可平移修复：平移；C 修不动 / D 证据不足：不修，留空

输出：
  z_note/data_offset_fixpipe/offset_correction.csv
  列 = session_id, date, subject, conf_class, lag_eff_frames, offset_s_old, offset_s_new,
       remark, source（fleet 行号）

内置一致性断言（任一失败即非零退出）：
  1. fleet 行数 == 140，且与 splits（train/val/test 去重）三向集合一致
  2. shared facts cam3 目录树 140 个 session 且与 fleet/splits 集合一致
  3. 全 140 session 恒等式 offset_s == visual_time_s[0] − mocap_time_s[0]，
     且 len(mocap_time_s) == session.json frame_count
  4. 三个已知值锚点：S5091 / S13112 / S12042（见 ANCHORS）
  5. 行级规则：A 类 lag_eff==0 且 new==old；B 类 new==old+lag_eff×DT；C/D 类 new 留空

用法：python3 build_offset_table.py   （可选 --fleet/--facts-root/--splits/--out 覆盖默认路径）
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

DT = 0.025  # 40 Hz 帧间隔（秒）

HERE = Path(__file__).resolve().parent


def _find_repo(start):
    """从脚本目录向上找包含 AnysoleWorkspace 与 z_note 的仓库根（与调用 cwd 无关）。"""
    p = start
    for _ in range(8):
        if (p / "AnysoleWorkspace").is_dir() and (p / "z_note").is_dir():
            return p
        p = p.parent
    raise RuntimeError(f"未找到仓库根（从 {start} 向上找 AnysoleWorkspace 失败）")


REPO = _find_repo(HERE)

FLEET_CSV = REPO / "z_note/对齐扫描归档/s1_scratch/lag_fleet.csv"
FACTS_ROOT = REPO / "AnysoleWorkspace/shared/facts/sessions"  # 其下 cam3/<date>/<subject>/<session>/
SPLITS_CSV = REPO / "AnysoleWorkspace/protocol/splits/default/splits.csv"
OUT_CSV = HERE / "offset_correction.csv"

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


def fmt(v):
    return f"{v:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=str(FLEET_CSV))
    ap.add_argument("--facts-root", default=str(FACTS_ROOT))
    ap.add_argument("--splits", default=str(SPLITS_CSV))
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    root = Path(args.facts_root)

    fleet = read_fleet(args.fleet)
    splits = read_splits(args.splits)
    facts = scan_facts(root)

    # 断言 1/2：三向集合一致
    if len(fleet) != 140:
        fail(f"fleet 行数 {len(fleet)} != 140")
    if len(splits) != 140:
        fail(f"splits 去重 session 数 {len(splits)} != 140")
    if len(facts) != 140:
        fail(f"facts 目录树 session 数 {len(facts)} != 140")
    for a, b, name in ((fleet, splits, "fleet vs splits"), (fleet, facts, "fleet vs facts"), (splits, facts, "splits vs facts")):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        if only_a or only_b:
            fail(f"{name} 不一致: 仅前者 {only_a} / 仅后者 {only_b}")
    ok("三向集合一致: fleet=140 / splits=140 / facts=140，无缺失无多出")

    # 断言 3：全 140 恒等式 + 帧数（顺带取 offset_s_old）
    old_off = {}
    for sid, (date, subject) in facts.items():
        old_off[sid] = load_session(root, date, subject, sid)
    ok("全 140 session 恒等式 offset_s == visual_time_s[0] − mocap_time_s[0] 且帧数一致")

    # 断言 4：已知值锚点
    for sid, (cls, lag, old, new) in ANCHORS.items():
        r = fleet.get(sid)
        if r is None:
            fail(f"锚点 {sid} 不在 fleet 表")
        if r["conf_class"] != cls:
            fail(f"锚点 {sid}: conf_class={r['conf_class']} != {cls}")
        if abs(float(r["lag_eff_frames"]) - lag) > 1e-6:
            fail(f"锚点 {sid}: lag_eff_frames={r['lag_eff_frames']} != {lag}")
        if abs(old_off[sid] - old) > 1e-6:
            fail(f"锚点 {sid}: offset_s_old={old_off[sid]:.6f} != {old}")
        new_calc = old + lag * DT
        if abs(new_calc - new) > 1e-9:
            fail(f"锚点 {sid}: 复算 {new_calc:.6f} != {new}")
    ok(f"已知值锚点通过: S5091 0.631612→0.869112 / S13112 0.126726→−0.098274 / S12042 0.326787→0.251787")

    # 行级规则 + 写 CSV
    rows_out = []
    class_stat = {"A": 0, "B": 0, "C": 0, "D": 0}
    for sid in sorted(fleet):
        r = fleet[sid]
        date, subject = facts[sid]
        cls = r["conf_class"]
        lag = float(r["lag_eff_frames"])
        old = old_off[sid]
        class_stat[cls] = class_stat.get(cls, 0) + 1
        if cls == "A":
            if abs(lag) > 1e-9:
                fail(f"{sid}: A 类但 lag_eff={lag} 非零")
            new, remark = old, "A 已对齐，不修（lag_eff 恒 0）"
        elif cls == "B":
            new = old + lag * DT
            if abs((new - old) - lag * DT) > 1e-9:
                fail(f"{sid}: B 类复算异常")
            remark = f"B 可平移修复：mocap_time_s 平移 {-lag * DT:+.6f}s（offset_s_new = offset_s_old {lag * DT:+.6f}s）"
        elif cls == "C":
            new, remark = None, "C 修不动（align*≥12px，2D 串人非时钟），不修"
        elif cls == "D":
            new, remark = None, f"D 证据不足（fleet flag: {r['flag'] or '—'}），不修"
        else:
            fail(f"{sid}: 未知 conf_class {cls}")
        rows_out.append({
            "session_id": sid,
            "date": date,
            "subject": subject,
            "conf_class": cls,
            "lag_eff_frames": f"{lag:.1f}",
            "offset_s_old": fmt(old),
            "offset_s_new": fmt(new) if new is not None else "",
            "remark": remark,
            "source": f"lag_fleet.csv 行{r['fleet_line']}",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "session_id", "date", "subject", "conf_class", "lag_eff_frames",
            "offset_s_old", "offset_s_new", "remark", "source"])
        w.writeheader()
        w.writerows(rows_out)
    ok(f"已写出 {args.out}（{len(rows_out)} 行）")

    b_new = [float(x["offset_s_new"]) for x in rows_out if x["conf_class"] == "B"]
    print(f"类别分布: {class_stat}")
    print(f"B 类 offset_s_new 范围: [{min(b_new):.6f}, {max(b_new):.6f}]（n={len(b_new)}）")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
