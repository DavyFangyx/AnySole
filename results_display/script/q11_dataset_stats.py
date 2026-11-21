#!/usr/bin/env python3
"""Q11 数据集统计：140 会话清单（帧数/时长）+ 各模态原生采样率。

只读输入：
  - AnysoleWorkspace/shared/facts/sessions/cam3/<date>/<subject>/<session>/session.json
    （40 Hz 统一网格帧数 frame_count）
  - AnysoleWorkspace/protocol/manifests/session_manifest.csv（动作编号）
  - AnysoleWorkspace/hf-raw/rgb/*/*/*/meta.json（视频原生 fps）
  - AnysoleWorkspace/hf-raw/pressure/data/dataset/<date>/<subject>/<session>/left.npz
    （压力原生到达间隔 native_receive_time_s）

输出：
  results_display/Etest_crossfit_physimu_vgate/Supplement/05_数据集与采集真实性/
    Q11：数据集与接触真值统计/实验结果/q11_数据集统计.csv

用法：python3 results_display/script/q11_dataset_stats.py
"""

import csv
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FACTS = REPO / "AnysoleWorkspace/shared/facts/sessions/cam3"
MANIFEST = REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.csv"
HFRAW = REPO / "AnysoleWorkspace/hf-raw"
OUT = (REPO / "results_display/Etest_crossfit_physimu_vgate/Supplement"
       / "05_数据集与采集真实性/Q11：数据集与接触真值统计/实验结果"
       / "q11_数据集统计.csv")

ACTION_NAMES = {
    "01": "站立重心移", "02": "单脚站立", "03": "折返慢走", "04": "折返快走",
    "05": "绕圈跑", "06": "深蹲", "07": "开合跳", "08": "折返跑", "09": "横移",
    "10": "坐下站起", "11": "上楼梯",
}

GRID_HZ = 40.0


def session_key(session_id: str) -> tuple[int, int, int]:
    """S5011 -> (5, 1, 1)；S10101 -> (10, 10, 1)。"""
    core = session_id[1:]
    return int(core[:-3]), int(core[-3:-1]), int(core[-1])


def main() -> None:
    facts: dict[str, dict] = {}
    for d in glob.glob(str(FACTS / "*" / "*" / "*")):
        jf = Path(d) / "session.json"
        if jf.exists():
            j = json.loads(jf.read_text())
            facts[j["session_id"]] = j

    action_of = {
        r["session_id"]: r["action"]
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig"))
    }
    missing = set(facts) - set(action_of)
    if missing:
        print("manifest 缺会话:", sorted(missing), file=sys.stderr)

    rows = []
    for sid in sorted(facts, key=session_key):
        j = facts[sid]
        rows.append({
            "session_id": sid,
            "date": j["date"],
            "subject": j["subject"],
            "action": action_of.get(sid, "?"),
            "action_name": ACTION_NAMES.get(action_of.get(sid, ""), ""),
            "n_frames_40hz": j["frame_count"],
            "duration_s": round(j["frame_count"] / GRID_HZ, 3),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {OUT}（{len(rows)} 行）")

    # ---- 汇总 ----
    tot_frames = sum(r["n_frames_40hz"] for r in rows)
    print(f"会话 {len(rows)} | 帧数 {tot_frames} | 时长 {tot_frames / GRID_HZ:.1f} s")
    durs = [r["duration_s"] for r in rows]
    print(f"会话时长 min {min(durs):.1f} / 中位 {statistics.median(durs):.1f} / max {max(durs):.1f} s")

    by_subj = defaultdict(lambda: [0, 0])
    for r in rows:
        by_subj[r["subject"]][0] += 1
        by_subj[r["subject"]][1] += r["n_frames_40hz"]
    print("按受试者:")
    for s in sorted(by_subj, key=lambda x: int(x[1:])):
        n, f = by_subj[s]
        print(f"  {s}: {n} 会话, {f / GRID_HZ:.1f} s")

    by_act = defaultdict(lambda: [0, 0])
    for r in rows:
        by_act[r["action"]][0] += 1
        by_act[r["action"]][1] += r["n_frames_40hz"]
    print("按动作:")
    for a in sorted(by_act):
        n, f = by_act[a]
        print(f"  动作{a} {ACTION_NAMES.get(a, '')}: {n} 会话, {f / GRID_HZ:.1f} s")

    # ---- 原生采样率 ----
    fps = []
    n_meta = 0
    for p in glob.glob(str(HFRAW / "rgb" / "*" / "*" / "*" / "meta.json")):
        n_meta += 1
        m = json.loads(open(p).read())
        for c in m.get("cameras", []):
            fps.append(c["fps"])
    print(f"视频原生 fps（{n_meta} 个 meta.json、{len(fps)} 条相机记录）: "
          f"min {min(fps):.2f} / max {max(fps):.2f} / 中位 {statistics.median(fps):.2f}")

    p_rates = []
    for r in rows:
        p = (HFRAW / "pressure/data/dataset" / r["date"] / r["subject"]
             / r["session_id"] / "left.npz")
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        t = z["native_receive_time_s"]
        if len(t) > 1:
            p_rates.append(1.0 / float(np.median(np.diff(t))))
    print(f"压力原生采样率（{len(p_rates)} 个左脚文件的中位间隔）: "
          f"{statistics.median(p_rates):.1f} Hz")

    native_mocap = 3 * tot_frames - 2 * len(rows)
    print(f"动捕原生 120 Hz 总帧数（按 3:1 抽取逆算）≈ {native_mocap}")


if __name__ == "__main__":
    main()
