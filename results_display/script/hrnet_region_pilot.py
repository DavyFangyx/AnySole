#!/usr/bin/env python3
"""HRNet 前置特征图 lower-region 激活 pilot（2026-11-06）。

目的：验证「CLIFF-HRNet 的 2048×H×W 特征图在下 bbox 区域的能量」能否作为
V 可靠性信号。它不是逐关节热力图（CLIFF-HRNet 是分类式 backbone），而是
模型自己的输入表示在遮挡区域还剩多少能量。

做法：
  * 遮挡 = extract_hrnet.occlude_lower_strip（黑填充，评估缓存口径）。
  * Hook `model.final_layer`（2048×H×W，avg_pool 之前），按行算能量
    `row_energy[n, h] = mean_{c,w} F[n,c,h,w]^2`。
  * 对每个 occ 档 frac，lower = 底部 round(frac*H) 行的能量；
    statistic = lower / full、lower_drop = (clean_lower - occ_lower)/clean_lower。
  * 对 test 会话，与 V8_Cv5_fS5S6S11_s1[_evaloccNNN] 的
    window_detail_test.json 按 (session, frame_start) 对齐，算 Spearman。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/hrnet_region_pilot.py \
      --sessions S11101,S11011,S10101 --device cuda --stride 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
WORKSPACE = REPO / "AnysoleWorkspace"
FACTS_CAM3 = WORKSPACE / "shared/facts/sessions/cam3"
BBOX_CAM3 = WORKSPACE / "model_inputs/AnySole/adapter_v1/visual/hrnet/bbox"
LEVELS = ("000", "020", "040", "060", "080", "100")
CELL = "V8_Cv5_fS5S6S11_s1"
OUT = REPO / "results/diagnostics/20261106_hrnet_region_pilot"


def find_session(sid: str) -> Path:
    hits = sorted(FACTS_CAM3.glob("*/*/%s" % sid))
    if not hits:
        raise FileNotFoundError("no facts session dir for %s" % sid)
    return hits[0]


def frame_paths(sid: str) -> list[Path]:
    root = find_session(sid) / "rgb"
    frames = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise FileNotFoundError("no RGB frames under %s" % root)
    return frames


def load_window_errors(cell: str, level: str, sid: str) -> dict:
    if level == "000":
        path = REPO / "results/cells" / cell / "metrics/window_detail_test.json"
    else:
        path = (REPO / "results/cells" / ("%s_evalocc%s" % (cell, level))
                / "metrics/window_detail_test.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))["conditions"]["V2M"]
    out = {}
    for s, start, lo, af in zip(
            data["session_ids"], data["frame_starts"],
            data["metrics"]["PA-MPJPE_lower"], data["metrics"]["PA-MPJPE_anklefoot"]):
        if str(s) == sid:
            out[int(start)] = {"lower": float(lo), "anklefoot": float(af)}
    return out


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        for val in np.unique(v):
            idx = np.where(v == val)[0]
            r[idx] = r[idx].mean()
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    from anysole.data.extract_hrnet import (CROP_IMG_HEIGHT, CROP_IMG_WIDTH,
                                            load_encoder, occlude_lower_strip,
                                            process_image, read_rgb)
    device = torch.device(args.device)
    model = load_encoder(device)
    captured = {}

    def hook(_module, _inputs, output):
        captured["f"] = output

    handle = model.final_layer.register_forward_hook(hook)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, corr_rows = [], []

    for sid in [s.strip() for s in args.sessions.split(",") if s.strip()]:
        frames = frame_paths(sid)
        bbox = np.load(BBOX_CAM3 / ("%s.npy" % sid))
        if bbox.shape[0] != len(frames):
            raise ValueError("%s bbox %s != frames %d" % (sid, bbox.shape, len(frames)))
        picked = list(range(0, len(frames), max(1, args.stride)))
        row_energy = {}
        for level in LEVELS:
            frac = int(level) / 100.0
            rows = np.full((len(picked), 8), np.nan, dtype=np.float32)
            step = max(1, args.batch_size)
            for start in range(0, len(picked), step):
                chunk = picked[start:start + step]
                xs = []
                for i in chunk:
                    rgb = read_rgb(frames[i])
                    box = bbox[i][1:5]
                    if frac > 0:
                        rgb = occlude_lower_strip(rgb, box, frac, "black")
                    norm_img, *_ = process_image(
                        rgb, box, crop_height=CROP_IMG_HEIGHT,
                        crop_width=CROP_IMG_WIDTH)
                    xs.append(np.asarray(norm_img, dtype=np.float32))
                x = torch.from_numpy(np.stack(xs, axis=0)).to(device)
                with torch.no_grad():
                    model(x)
                fmap = captured["f"]  # (B,2048,H,W)
                rows[start:start + len(chunk)] = (
                    (fmap.float() ** 2).mean(dim=(1, 3)).cpu().numpy())
            row_energy[level] = rows
            np.savez_compressed(out_dir / ("row_energy_%s_%s.npz" % (sid, level)),
                                frames=np.asarray(picked), row_energy=rows)
            print("[%s/%s] H=%d mean_row_energy=%s" % (
                sid, level, rows.shape[1],
                " ".join("%.3f" % v for v in np.nanmean(rows, axis=0))))

        clean = row_energy["000"]
        for level in LEVELS:
            frac = int(level) / 100.0
            h = row_energy[level].shape[1]
            n_low = max(1, int(round(frac * h))) if frac > 0 else 0
            if n_low == 0:
                continue
            occ_low = np.nanmean(row_energy[level][:, h - n_low:], axis=1)
            clean_low = np.nanmean(clean[:, h - n_low:], axis=1)
            full = np.nanmean(row_energy[level], axis=1)
            ratio = occ_low / np.maximum(full, 1e-8)
            drop = (clean_low - occ_low) / np.maximum(clean_low, 1e-8)
            summary.append({"session": sid, "level": level, "frac": frac,
                            "n_low": n_low,
                            "lower_ratio_med": float(np.nanmedian(ratio)),
                            "lower_drop_med": float(np.nanmedian(drop)),
                            "clean_low_med": float(np.nanmedian(clean_low)),
                            "occ_low_med": float(np.nanmedian(occ_low))})
            # window correlation
            errs = load_window_errors(CELL, level, sid)
            if errs:
                xs, ds, ys = [], [], []
                for st in sorted(errs):
                    sel = [j for j, fr in enumerate(picked) if st <= fr < st + 20]
                    if len(sel) < 2:
                        continue
                    xs.append(float(np.nanmean(ratio[sel])))
                    ds.append(float(np.nanmean(drop[sel])))
                    ys.append(errs[st]["lower"])
                if len(xs) >= 3:
                    corr_rows.append({
                        "session": sid, "level": level, "n_windows": len(xs),
                        "rho_ratio": spearman(xs, ys),
                        "rho_drop": spearman(ds, ys),
                    })
    handle.remove()
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    with (out_dir / "corr.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        if corr_rows:
            w = csv.DictWriter(fh, fieldnames=list(corr_rows[0]))
            w.writeheader(); w.writerows(corr_rows)
    (out_dir / "report.json").write_text(json.dumps(
        {"sessions": args.sessions, "stride": args.stride,
         "levels": list(LEVELS), "summary": summary, "corr": corr_rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[summary]")
    for r in summary:
        print("%-7s %s lower_ratio=%.3f lower_drop=%.3f"
              % (r["session"], r["level"], r["lower_ratio_med"], r["lower_drop_med"]))
    if corr_rows:
        print("\n[ratio vs V lower error]")
        for r in corr_rows:
            print("%-7s %s nwin=%d rho_ratio=%+.3f rho_drop=%+.3f"
                  % (r["session"], r["level"], r["n_windows"], r["rho_ratio"], r["rho_drop"]))
    print("\n[out] %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
