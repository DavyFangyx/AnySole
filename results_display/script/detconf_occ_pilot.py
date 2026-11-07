#!/usr/bin/env python3
"""检测器置信度 × 遮挡 pilot（2026-11-06）。

目的：验证「RTMPose-Halpe26 逐关节 score 能否作为 V 可靠性的门控载体」。
只在少量会话 × 5 个遮挡档上跑，不写生产前端缓存。

做法：
  * 遮挡 = `anysole/data/extract_hrnet.py::occlude_lower_strip` 的同一逻辑
    （bbox 底部 frac 全宽涂黑；bbox 与 bbox_info 不动）。
  * 推理 = MMPoseInferencer（RTMPose-m HALPE-26，config/model 见 shared
    frontend artifact）；干净档用现有 shared frontend 的 score（已逐位
    核对过同一帧重跑 = 0 diff），只新跑 occ 档。
  * 逐关节 score → 下肢组（Halpe-26 11-16,20-25）与远端组（15,16,20-25）
    的窗级/帧级统计。
  * 对 test 会话，与 V8_Cv5_fS5S6S11_s1[_evaloccNNN] 的
    window_detail_test.json 按 (session, frame_start) 对齐，算 score 与
    V2M PA / lower / anklefoot 误差的 Spearman。

用法（bbox_scan env，仓库根；需要 GPU）：
  python results_display/script/detconf_occ_pilot.py \
      --sessions S11101,S11011,S10101 --device cuda --stride 5
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO / "AnysoleWorkspace"
FACTS_CAM3 = WORKSPACE / "shared/facts/sessions/cam3"
RTMPOSE_FRONTEND = WORKSPACE / "shared/frontends/rtmpose_halpe26/v1"
BBOX_CAM3 = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet/bbox"
MMPOSE_ROOT = WORKSPACE / "assets/third_party/mmpose"
MODEL = (WORKSPACE / "assets/third_party/rtmpose"
         / "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth")
CONFIG = (MMPOSE_ROOT / "configs/body_2d_keypoint/rtmpose/body8"
          / "rtmpose-m_8xb512-700e_body8-halpe26-256x192.py")

LEVELS = ("020", "040", "060", "080", "100")
LOWER = (11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25)
DISTAL = (15, 16, 20, 21, 22, 23, 24, 25)
CELL = "V8_Cv5_fS5S6S11_s1"
FOLD = "fS5S6S11"
OUT = REPO / "results/diagnostics/20261106_detconf_pilot"


def find_session(sid: str) -> Path:
    hits = sorted(FACTS_CAM3.glob("*/*/%s" % sid))
    if not hits:
        raise FileNotFoundError("no facts session dir for %s" % sid)
    return hits[0]


def find_frontend(sid: str) -> Path:
    hits = sorted(RTMPOSE_FRONTEND.glob("*/*/%s" % sid))
    if not hits:
        raise FileNotFoundError("no RTMPose frontend dir for %s" % sid)
    return hits[0]


def frame_paths(sid: str) -> list[Path]:
    root = find_session(sid) / "rgb"
    frames = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise FileNotFoundError("no RGB frames under %s" % root)
    return frames


def occlude_lower_strip(bgr: np.ndarray, box, frac: float,
                        fill: str = "black") -> np.ndarray:
    """Same cut as extract_hrnet.occlude_lower_strip.

    fill="black"（评估缓存口径）或 "blue"（亮蓝，OOD 诊断：排除
    「黑色被当成衣物/阴影」的歧义）。
    """
    y1, y2 = float(box[1]), float(box[3])
    cut = int(round(y1 + (1.0 - frac) * (y2 - y1)))
    if cut <= 0 or cut >= bgr.shape[0] - 1:
        return bgr
    out = bgr.copy()
    if fill == "black":
        out[cut:] = 0
    elif fill == "blue":
        out[cut:] = (255, 0, 0)  # BGR 纯亮蓝
    else:
        raise ValueError("unknown fill: %s" % fill)
    return out


def first_person(prediction: dict):
    people = prediction.get("predictions", [[]])
    people = people[0] if people else []
    if not people:
        return None
    return max(people, key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))))


def evaluate_scores(scores: np.ndarray, idx: np.ndarray) -> dict:
    """scores (N,26) with NaN rows for no-detection; idx关节子集。"""
    sub = scores[:, list(idx)]
    valid = ~np.isnan(sub).any(axis=1)
    vals = sub[valid].reshape(-1)
    if vals.size == 0:
        return {"n": int(len(scores)), "detected": 0, "detected_frac": 0.0,
                "mean": float("nan"), "median": float("nan"),
                "p05": float("nan"), "min": float("nan")}
    return {"n": int(len(scores)), "detected": int(valid.sum()),
            "detected_frac": float(valid.mean()),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)), "min": float(vals.min())}


def load_window_errors(cell: str, level: str, sid: str) -> dict:
    """window_detail_test.json 的 V2M：frame_start -> per-window errors."""
    if level == "000":
        path = REPO / "results/cells" / cell / "metrics/window_detail_test.json"
    else:
        path = (REPO / "results/cells" / ("%s_evalocc%s" % (cell, level))
                / "metrics/window_detail_test.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))["conditions"]["V2M"]
    out = {}
    for s, start, pa, lo, af, gl in zip(
            data["session_ids"], data["frame_starts"],
            data["metrics"]["pa_mpjpe_mm"], data["metrics"]["PA-MPJPE_lower"],
            data["metrics"]["PA-MPJPE_anklefoot"], data["metrics"]["global_mpjpe_mm"]):
        if str(s) == sid:
            out[int(start)] = {"pa": float(pa), "lower": float(lo),
                               "anklefoot": float(af), "global": float(gl)}
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", default="S11101,S11011,S10101")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fill", choices=("black", "blue"), default="black",
                    help="遮挡填充：black = 评估缓存口径；blue = 亮蓝 OOD 诊断")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer(pose2d=str(CONFIG), pose2d_weights=str(MODEL),
                                  device=args.device, show_progress=False)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    corr_rows = []
    strip_rows = []

    for sid in [s.strip() for s in args.sessions.split(",") if s.strip()]:
        frames = frame_paths(sid)
        bbox = np.load(BBOX_CAM3 / ("%s.npy" % sid))
        if bbox.shape[0] != len(frames):
            raise ValueError("%s bbox %s != frames %d" % (sid, bbox.shape, len(frames)))
        clean_dir = find_frontend(sid) / "keypoints_cam3"
        picked = list(range(0, len(frames), max(1, args.stride)))
        # ---- clean（用生产前端 score，已核同一帧重跑逐位一致）----
        clean = np.full((len(picked), 26), np.nan, dtype=np.float32)
        for j, i in enumerate(picked):
            p = clean_dir / ("%06d.npy" % i)
            if p.is_file():
                clean[j] = np.load(p, allow_pickle=True).item()["keypoint_scores"]
        summary.append({"session": sid, "level": "000", "group": "lower",
                        **evaluate_scores(clean, LOWER)})
        summary.append({"session": sid, "level": "000", "group": "distal",
                        **evaluate_scores(clean, DISTAL)})
        np.savez_compressed(out_dir / ("scores_%s_000.npz" % sid),
                            frames=np.asarray(picked), scores=clean)

        # ---- occ 档：现场涂黑 + RTMPose（按 batch 向量化）----
        for level in levels:
            frac = int(level) / 100.0
            scores = np.full((len(picked), 26), np.nan, dtype=np.float32)
            kps = np.full((len(picked), 26, 2), np.nan, dtype=np.float32)
            step = max(1, args.batch_size)
            for start in range(0, len(picked), step):
                chunk = picked[start:start + step]
                imgs = []
                for i in chunk:
                    bgr = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise FileNotFoundError(frames[i])
                    imgs.append(occlude_lower_strip(bgr, bbox[i][1:5], frac, args.fill))
                outs = list(inferencer(imgs, batch_size=len(imgs)))
                for k, _ in enumerate(chunk):
                    people = outs[k].get("predictions", [[]])
                    people = people[0] if people else []
                    person = (max(people, key=lambda item: float(
                        np.mean(item.get("keypoint_scores", [0.0]))))
                        if people else None)
                    if person is not None:
                        scores[start + k] = np.asarray(
                            person["keypoint_scores"], dtype=np.float32)
                        kps[start + k] = np.asarray(
                            person["keypoints"], dtype=np.float32)
            np.savez_compressed(out_dir / ("scores_%s_%s.npz" % (sid, level)),
                                frames=np.asarray(picked), scores=scores,
                                keypoints=kps)
            # 关节是否落在遮挡条内 + 条内/条外 score（检测到 vs 靠先验补）
            in_vals, out_vals = [], []
            for j, i in enumerate(picked):
                y1, y2 = float(bbox[i][1]), float(bbox[i][3])
                cut = int(round(y1 + (1.0 - frac) * (y2 - y1)))
                for q in DISTAL:
                    if np.isnan(scores[j, q]):
                        continue
                    (in_vals if kps[j, q, 1] >= cut else out_vals).append(float(scores[j, q]))
            in_frac = len(in_vals) / max(len(in_vals) + len(out_vals), 1)
            strip_rows.append({
                "session": sid, "level": level, "fill": args.fill,
                "in_frac": in_frac,
                "score_in": float(np.mean(in_vals)) if in_vals else float("nan"),
                "score_out": float(np.mean(out_vals)) if out_vals else float("nan"),
                "n_in": len(in_vals), "n_out": len(out_vals)})
            for group, idx in (("lower", LOWER), ("distal", DISTAL)):
                summary.append({"session": sid, "level": level, "group": group,
                                **evaluate_scores(scores, idx)})
            # 窗误差相关（test 会话；窗长 20 帧）
            errs = load_window_errors(CELL, level, sid)
            if errs:
                starts = sorted(errs)
                xs, pa, lo, af, gl = [], [], [], [], []
                for s in starts:
                    sel = [(j, i) for j, i in enumerate(picked)
                           if s <= i < s + 20 and not np.isnan(scores[j]).any()]
                    if len(sel) < 2:
                        continue
                    xs.append(float(np.mean([scores[j, list(DISTAL)].mean() for j, _ in sel])))
                    pa.append(errs[s]["pa"]); lo.append(errs[s]["lower"])
                    af.append(errs[s]["anklefoot"]); gl.append(errs[s]["global"])
                if len(xs) >= 3:
                    corr_rows.append({
                        "session": sid, "level": level, "n_windows": len(xs),
                        "rho_pa": spearman(np.array(xs), np.array(pa)),
                        "rho_lower": spearman(np.array(xs), np.array(lo)),
                        "rho_anklefoot": spearman(np.array(xs), np.array(af)),
                        "rho_global": spearman(np.array(xs), np.array(gl)),
                    })
            print("[%s/%s] %s  strip_in=%.2f score_in=%.3f score_out=%.3f" % (
                sid, level, " ".join(
                    "%s median=%.3f det=%.2f" % (r["group"], r["median"], r["detected_frac"])
                    for r in summary[-2:]),
                in_frac,
                float(np.mean(in_vals)) if in_vals else float("nan"),
                float(np.mean(out_vals)) if out_vals else float("nan")))

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["session", "level", "group", "n", "detected",
                                           "detected_frac", "mean", "median", "p05", "min"])
        w.writeheader(); w.writerows(summary)
    with (out_dir / "strip.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["session", "level", "fill", "in_frac",
                                           "score_in", "score_out", "n_in", "n_out"])
        w.writeheader(); w.writerows(strip_rows)
    with (out_dir / "corr.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        if corr_rows:
            w = csv.DictWriter(fh, fieldnames=list(corr_rows[0]))
            w.writeheader(); w.writerows(corr_rows)
    (out_dir / "report.json").write_text(json.dumps(
        {"sessions": args.sessions, "stride": args.stride, "fill": args.fill,
         "levels": list(LEVELS), "summary": summary, "corr": corr_rows,
         "strip": strip_rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[summary]")
    print("%-8s %-5s %-7s %8s %8s %8s %8s" % ("session", "occ", "group", "median", "p05", "det", "n"))
    for r in summary:
        print("%-8s %-5s %-7s %8.3f %8.3f %8.2f %8d"
              % (r["session"], r["level"], r["group"], r["median"], r["p05"],
                 r["detected_frac"], r["n"]))
    if corr_rows:
        print("\n[score vs V2M error, Spearman]")
        print("%-8s %-5s %6s %9s %9s %9s %9s" % ("session", "occ", "nwin", "rho_pa", "rho_lower", "rho_af", "rho_global"))
        for r in corr_rows:
            print("%-8s %-5s %6d %9.3f %9.3f %9.3f %9.3f"
                  % (r["session"], r["level"], r["n_windows"], r["rho_pa"],
                     r["rho_lower"], r["rho_anklefoot"], r["rho_global"]))
    print("\n[out] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
