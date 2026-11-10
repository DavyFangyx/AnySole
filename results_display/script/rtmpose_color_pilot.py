#!/usr/bin/env python3
"""RTMPose 黑 vs 蓝遮挡条 cheap pilot（2026-11-09）。

同一几何 occlude_lower_strip，只改 fill 颜色：
  black = out[cut:] = 0；blue = out[cut:] = (0,0,255) RGB。
在 3 会话 × 4 档（020/040/060/080）上跑 RTMPose，取下肢 6 点
keypoints/scores，对 V<T（V8_Cv5 V 行 vs T 行 full PA）做 within-level
Spearman/AUC，比较 black 与 blue 的符号/幅度，排除“只检测黑条”。

用法（mmpose 环境，GPU；先 smoke 一档）：
  python results_display/script/rtmpose_color_pilot.py --device cuda:2 \
      --levels 020,040,060,080 --sessions S6073,S11021,S5112
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_display/script"))

from anysole.data.extract_hrnet import occlude_lower_strip, read_rgb  # noqa: E402
from anysole.utils.metrics import pa_mpjpe  # noqa: E402
from gate_metric_crossfit_export import load_motion  # noqa: E402
from AnysoleWorkspace.tool.adapters.visual_baselines import generate as gb  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

GATE_FOLD = "fS5S6S11"
SESSIONS = ("S6073", "S11021", "S5112")
LEVELS = ("020", "040", "060", "080")
CAM = 3
LOWER_HALPE = tuple(range(11, 23))  # hips/knees/ankles/toes/heels
OUT = REPO / "results/diagnostics/20261108_gate_level/rtmpose_color_pilot"


def read_manifest():
    rows = {}
    path = REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def first_person(prediction):
    people = prediction.get("predictions", [[]])
    people = people[0] if people else []
    person = max(people, key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))))
    return (np.asarray(person["keypoints"], dtype=np.float32),
            np.asarray(person["keypoint_scores"], dtype=np.float32).reshape(-1))


def load_unit(pkg, date, session, cam):
    unit_id = f"{date}/method_1_offset/{session}/camera_{cam}"
    path = Path(pkg) / "reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl"
    for line in path.read_text().splitlines():
        value = json.loads(line)
        if value.get("unit_id") == unit_id:
            return value
    raise FileNotFoundError(unit_id)
STAGE = OUT / "staged"


def _jpeg_wall_seconds(name: str) -> float:
    clock = name.split("_")[-1].rsplit(".", 1)[0]
    return int(clock[:2]) * 3600.0 + int(clock[2:4]) * 60.0 + float(clock[4:])


def stage_color(level: str, session: str, color: str) -> list[Path]:
    """按 canonical grid 重绘遮挡条；返回 staged PNG（长度 = bbox/visual_time_s）。"""
    import cv2
    from datetime import datetime
    rows = read_manifest()
    row = rows[session]
    date, subject = gb.session_parts(row)
    recording = resolve_uri(row["video_path"], must_exist=True)
    meta = json.loads((recording / "meta.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(str(meta["started_at_iso"]))
    start_s = start.hour * 3600.0 + start.minute * 60.0 + start.second + start.microsecond / 1e6
    frames = gb.recording_frames(row, CAM)
    stamps = np.asarray([_jpeg_wall_seconds(p.name) for p in frames])
    facts = (REPO / "AnysoleWorkspace/shared/facts/sessions/cam3" / date / subject
             / session / "frames.npz")
    visual = np.asarray(np.load(facts)["visual_time_s"], dtype=np.float64)
    boxes = np.load(gb.BBOX_CAM3_ROOT / ("%s.npy" % session))[:, 1:5]
    if len(boxes) != len(visual):
        raise RuntimeError("%s boxes %d != visual %d" % (session, len(boxes), len(visual)))
    stage_dir = STAGE / level / session / color
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (box, vt) in enumerate(zip(boxes, visual)):
        idx = int(np.argmin(np.abs(stamps - (start_s + vt))))
        path = frames[idx]
        target = stage_dir / ("%06d.png" % i)
        if not target.is_file():
            rgb = read_rgb(path)
            rgb = occlude_lower_strip(rgb, box, int(level) / 100.0, "black")
            if color == "blue":
                y1, y2 = float(box[1]), float(box[3])
                cut = int(round(y1 + (1.0 - int(level) / 100.0) * (y2 - y1)))
                if 0 < cut < rgb.shape[0] - 1:
                    rgb[cut:] = np.asarray([0, 0, 255], dtype=np.uint8)
            tmp = stage_dir / ("%06d.tmp.png" % i)
            if not cv2.imwrite(str(tmp), np.ascontiguousarray(rgb[:, :, ::-1])):
                raise RuntimeError("failed to write %s" % tmp)
            import os
            os.replace(str(tmp), str(target))
        out.append(target)
    return out

def v8_vt(level: str, session: str):
    v = load_motion("V8_Cv5", GATE_FOLD, level, session, "V2M")
    t = load_motion("V8_Cv5", GATE_FOLD, level, session, "T2M")
    rows = np.load(REPO / "results/diagnostics/20261108_gate_level" /
                   ("rows_%s_test.npz" % GATE_FOLD), allow_pickle=True)
    m = (rows["session"] == session) & (rows["level"] == int(level))
    starts, gt_win = rows["start"][m].astype(int), rows["gt"][m]
    if v is None or t is None:
        return None
    frames = v["frame"]
    ind, gt = [], []
    for wi, start in enumerate(starts):
        for j in range(gt_win.shape[1]):
            idx = np.where(frames == (start + j))[0]
            if len(idx):
                ind.append(int(idx[0])); gt.append(gt_win[wi, j])
    ind = np.asarray(ind); gt = np.asarray(gt)
    e_v = np.asarray([float(pa_mpjpe(v["joint"][i][None], gt[i][None]).mean())
                      for i in range(len(ind))])
    e_t = np.asarray([float(pa_mpjpe(t["joint"][i][None], gt[i][None]).mean())
                      for i in range(len(ind))])
    return ind, e_v, e_t


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5:
        return float("nan")
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        for val in np.unique(v):
            idx = np.where(v == val)[0]; r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])[order]
    ranks = np.arange(1, len(lab) + 1)
    return float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--sessions", default=",".join(SESSIONS))
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--config", default=str(
        REPO / "AnysoleWorkspace/assets/third_party/mmpose/configs/body_2d_keypoint/"
        "rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"))
    ap.add_argument("--model", default=str(
        REPO / "AnysoleWorkspace/assets/third_party/rtmpose/"
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"))
    args = ap.parse_args()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    mmpose_root = REPO / "AnysoleWorkspace/assets/third_party/mmpose"
    if str(mmpose_root) not in sys.path:
        sys.path.insert(0, str(mmpose_root))
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(pose2d=args.config, pose2d_weights=args.model,
                                  device=args.device, show_progress=False)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for level in levels:
        for session in sessions:
            vt = v8_vt(level, session)
            if vt is None:
                continue
            ind, e_v, e_t = vt
            vlt = (e_v > e_t).astype(float)
            for color in ("black", "blue"):
                imgs = stage_color(level, session, color)
                # staged frames correspond to the full unit; align by frame index
                kp, sc = [], []
                for image in imgs:
                    pred = next(inferencer(str(image)))
                    pts, scores = first_person(pred)
                    kp.append(pts); sc.append(scores)
                kp = np.asarray(kp); sc = np.asarray(sc)
                # 只取 V8 用到的 canonical frame（staged 顺序 = unit 顺序）
                if len(kp) < len(ind):
                    continue
                low_s = sc[ind][:, LOWER_HALPE]
                low_k = kp[ind][:, LOWER_HALPE]
                score_mean = low_s.mean(axis=1)
                score_p05 = np.percentile(low_s, 5, axis=1)
                jitter = np.linalg.norm(np.diff(low_k, axis=0), axis=-1).mean(axis=1)
                jitter = np.r_[jitter, jitter[-1]]
                for i in range(len(ind)):
                    all_rows.append({"level": level, "session": session,
                                     "color": color, "frame": int(ind[i]),
                                     "vlt": int(vlt[i]),
                                     "score_mean": float(score_mean[i]),
                                     "score_p05": float(score_p05[i]),
                                     "jitter": float(jitter[i])})
                print("[%s/%s/%s] n=%d score=%.3f jitter=%.2f"
                      % (level, session, color, len(ind),
                         score_mean.mean(), jitter.mean()))
    # per color / level / pooled within-level
    report = {}
    for color in ("black", "blue"):
        rec = {"per_level": {}}
        for level in levels:
            rows = [r for r in all_rows if r["color"] == color and r["level"] == level]
            y = np.asarray([r["vlt"] for r in rows])
            rec["per_level"][level] = {"n": len(rows), "vlt_rate": float(y.mean()) if len(y) else None}
            for k in ("score_mean", "score_p05", "jitter"):
                s = np.asarray([r[k] for r in rows])
                rec["per_level"][level][k] = {
                    "spearman": float(spearman(s, y)),
                    "auc": float(auc_score(y, s))}
        # pooled, level-centered
        rows = [r for r in all_rows if r["color"] == color]
        y = np.asarray([r["vlt"] for r in rows])
        lv = np.asarray([r["level"] for r in rows])
        rec["pooled"] = {"n": len(rows), "vlt_rate": float(y.mean()) if len(y) else None}
        for k in ("score_mean", "score_p05", "jitter"):
            s = np.asarray([r[k] for r in rows])
            s_c = s.copy()
            for l in levels:
                m = lv == l
                if m.any():
                    s_c[m] -= s[m].mean()
            rec["pooled"][k] = {"raw_spearman": float(spearman(s, y)),
                                "within_level_spearman": float(spearman(s_c, y)),
                                "auc": float(auc_score(y, s))}
        report[color] = rec
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[out] %s" % (OUT / "report.json"))


if __name__ == "__main__":
    main()
