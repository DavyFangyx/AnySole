"""E9 teaser companion strips: visual (cam3 RGB) and tactile (MMVP) filmstrips
across the same light -> dark -> light window used by Teaser_Model_*.png.

The vision-loss window (n0, e0, start, end) is derived purely from GT + the
session manifest (``events_unified05``), so it is identical to the window the
model teaser (e9_teaser_scene_model.py) draws for the same session -- no
model inference is needed here, only GT trajectories and the manifest.

Visual strip : cam3 rgb/%06d.jpg thumbnails; frames inside the occlusion
               event are replaced by a solid black tile (vision lost).
Tactile strip: MMVP 31x11 L/R insole heatmaps for the same frame indices;
               never masked (touch stays available throughout).

Usage:
    python results_display/script/e9_vision_touch_strip.py \
        --sid S6092 --action 09 --out-dir .../E9_Trajectory_Teaser/S6092_action09
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path("/data/fangyuxuan/projects/gait")
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import render_common as rc  # noqa: E402  (import before Step2Motion/src
                                        # shadows the stdlib-style `utils` name below)

PROTO_DIR = ROOT / "results/diagnostics/20261111_final_protocol"
for p in (
    PROTO_DIR,
    ROOT / "Baselines/Step2Motion/src",
    ROOT / "results/diagnostics/20261108_hybrid",
    ROOT / "results/diagnostics/20261107_ti_step_stage1",
    ROOT / "results/diagnostics/20261108_traj_sceneocc",
    ROOT / "results/diagnostics/20261110_vtbridge",
    ROOT / "results/diagnostics/20261111_variant_sweep",
    ROOT / "results/diagnostics/20261111_s2m_next",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e9_final_protocol as P  # noqa: E402

FOLDS = ["fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"]
FACTS_ROOT = ROOT / "AnysoleWorkspace/shared/facts/sessions/cam3"
MMVP_ROOT = ROOT / "AnysoleWorkspace/shared/representations/tactile/mmvp_31x11/v1"
PRE, POST = 60, 30  # same window as e9_teaser_scene_model.py defaults

TILE_H = 150
TILE_W = round(1624 * TILE_H / 1240)  # fixed per-tile footprint, shared by both strips
GAP = 4
BG = (247, 246, 243)
SEP_GRAY = (167, 167, 167)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FRAME_B, FRAME_W = 2, 1               # black-white-black sandwich, per side
FRAME_TOTAL = FRAME_B + FRAME_W + FRAME_B
CONTENT_W = TILE_W - 2 * FRAME_TOTAL  # picture itself shrinks to make room for the frame
CONTENT_H = TILE_H - 2 * FRAME_TOTAL


def one_way_score(gt: np.ndarray, event: dict, n: int) -> tuple | None:
    n0i, e0i = int(event["n0"]), int(event["e0"])
    loi, hii = n0i - PRE, e0i + POST + 1
    if loi < 0 or hii > n:
        return None
    direction = gt[e0i, [0, 2]] - gt[n0i, [0, 2]]
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        return None
    path = gt[loi:hii][:, [0, 2]]
    progress = (path - gt[n0i, [0, 2]]) @ (direction / distance)
    steps = np.diff(path, axis=0)
    travel = float(np.linalg.norm(steps, axis=1).sum())
    straightness = abs(float(progress[-1] - progress[0])) / max(travel, 1e-9)
    reverse_fraction = float(np.mean(np.diff(progress) < -0.003))
    return straightness - reverse_fraction, straightness, reverse_fraction


def find_window(sid: str, action: str) -> dict:
    """Same event-selection logic as e9_teaser_scene_model.py, GT-only."""
    gt = None
    fold_used = None
    for fold in FOLDS:
        path = P.VT_DIR % fold + "/" + sid + "_VT2M.npz"
        if os.path.isfile(path):
            gt, _, _, _ = P.load_vt(fold, sid)
            fold_used = fold
            break
    if gt is None:
        raise FileNotFoundError(f"{sid}: no VT2M npz found in any fold")
    man = json.load(open(P.MANI + f"/{sid}.json"))
    if str(man["action"]).zfill(2) != str(action).zfill(2):
        raise ValueError(f"{sid}: manifest action {man['action']} != requested {action}")
    n = len(gt)
    events = P.events_unified05(man, n)
    scored = [(one_way_score(gt, e, n), e) for e in events]
    scored = [(s, e) for s, e in scored if s is not None]
    if not scored:
        raise ValueError(f"{sid}: no event fits pre={PRE} post={POST}")
    score, ev = max(scored, key=lambda item: item[0][0])
    n0, e0 = int(ev["n0"]), int(ev["e0"])
    return {
        "fold": fold_used, "date": str(man["date"]), "subject": sid[:-3],
        "n": n, "n0": n0, "e0": e0,
        "start": int(ev["start"]), "end": int(ev["end"]),
        "lo": n0 - PRE, "hi": e0 + POST + 1,
    }


def sample_indices(win: dict, count: int = 28) -> list[int]:
    lo, hi = win["lo"], win["hi"]
    idx = set(np.linspace(lo, hi - 1, count).round().astype(int).tolist())
    idx.update([win["n0"], win["e0"], win["start"], win["end"], lo, hi - 1])
    return sorted(i for i in idx if lo <= i <= hi - 1)


def stretch_into(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize ``img`` to exactly (width, height), stretching as needed (no
    letterbox padding) -- every tile's picture is the same size with no
    blank margin, so frame i lines up between the visual and tactile strips.
    """
    return img.resize((width, height), Image.BILINEAR)


def framed(img: Image.Image) -> Image.Image:
    """Wrap a CONTENT_W x CONTENT_H picture in a black-white-black sandwich
    border (inner black touches the picture, then a white hairline, then an
    outer black line), growing it back out to the full TILE_W x TILE_H
    footprint. Marks every tile as its own discrete frame, independent of
    the GAP background between tiles.
    """
    img = ImageOps.expand(img, border=FRAME_B, fill=BLACK)
    img = ImageOps.expand(img, border=FRAME_W, fill=WHITE)
    img = ImageOps.expand(img, border=FRAME_B, fill=BLACK)
    return img


def build_strip(tiles: list[np.ndarray]) -> Image.Image:
    """Plain tile concatenation: no title, no labels, no markers."""
    th = tiles[0].shape[0]
    tw_total = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    canvas = Image.new("RGB", (tw_total, th), BG)
    x = 0
    for tile in tiles:
        canvas.paste(Image.fromarray(tile), (x, 0))
        x += tile.shape[1] + GAP
    return canvas


def visual_strip(sid: str, action: str, win: dict) -> Image.Image:
    rgb_dir = FACTS_ROOT / win["date"] / win["subject"] / sid / "rgb"
    idxs = sample_indices(win)
    tiles = []
    for i in idxs:
        if win["start"] <= i <= win["end"]:
            content = Image.fromarray(
                np.zeros((CONTENT_H, CONTENT_W, 3), dtype=np.uint8))
        else:
            img = Image.open(rgb_dir / f"{i:06d}.jpg").convert("RGB")
            content = stretch_into(img, CONTENT_W, CONTENT_H)
        tiles.append(np.asarray(framed(content)))
    return build_strip(tiles)


def load_mmvp_frame(mmvp_dir: Path, i: int) -> np.ndarray:
    payload = np.load(mmvp_dir / f"{i:06d}.npy")
    if payload.shape != (2, 31, 11):
        raise ValueError(f"MMVP frame {i}: shape {payload.shape} != (2,31,11)")
    return payload


def tactile_strip(sid: str, action: str, win: dict) -> Image.Image:
    mmvp_dir = MMVP_ROOT / win["date"] / win["subject"] / sid / "insole"
    idxs = sample_indices(win)
    tiles = []
    for i in idxs:
        left, right = load_mmvp_frame(mmvp_dir, i)
        left_img = rc.pressure_to_heatmap(left, vmax=1023.0)
        right_img = rc.pressure_to_heatmap(right, vmax=1023.0)
        sep = np.full((left_img.shape[0], 3, 3), SEP_GRAY, np.uint8)
        pair = np.concatenate([left_img, sep, right_img], axis=1)
        content = stretch_into(Image.fromarray(pair), CONTENT_W, CONTENT_H)
        tiles.append(np.asarray(framed(content)))
    return build_strip(tiles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    win = find_window(args.sid, args.action)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vis = visual_strip(args.sid, args.action, win)
    vis_path = args.out_dir / f"Teaser_Visual_{args.sid}_action{args.action}.png"
    vis.save(vis_path)
    print("wrote", vis_path, win)

    tac = tactile_strip(args.sid, args.action, win)
    tac_path = args.out_dir / f"Teaser_Tactile_{args.sid}_action{args.action}.png"
    tac.save(tac_path)
    print("wrote", tac_path)


if __name__ == "__main__":
    main()
