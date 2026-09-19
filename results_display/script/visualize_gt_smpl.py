"""Render raw SMPL-NPZ ground truth with the same skeleton style as Test1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cli_common
from smpl_visual import smpl_joints
from visualize_gt_bvh import render_frame
from anysole.data.dataset import find_session_dir, resolve_bvh_path
from anysole.data.smpl_io import resolve_smpl_path
from anysole.types import SMPL_ROOTS


def main() -> int:
    p = argparse.ArgumentParser(description="Render SMPL NPZ ground truth.")
    cli_common.add_common_args(p, seq_root=True, fps_default=0.0, stride_default=4,
                               out_dir_default=cli_common.DISPLAY_ROOT / "Test1_visualization" / "gt_smpl")
    p.add_argument("--smpl", default="", help="Optional single motion_neutral_smpl.npz path.")
    args = p.parse_args()
    seq_root = Path(cli_common.resolve_path(args.seq_root))
    out_root = Path(cli_common.resolve_path(args.out_dir))
    if args.smpl:
        samples = [(Path(args.smpl).stem, Path(args.smpl), None)]
    else:
        ids = cli_common.load_test_sessions(args.session, Path(cli_common.resolve_path(args.split_csv)), args.split)
        samples = []
        for sid in ids:
            seq = find_session_dir(seq_root, sid)
            meta = json.loads((seq / "align_meta.json").read_text())
            try:
                samples.append((sid, resolve_smpl_path(meta, SMPL_ROOTS), resolve_bvh_path(meta)))
            except FileNotFoundError as exc:
                print(f"[skip] {sid}: {exc}")
    for sid, path, bvh in samples:
        data = np.load(path, allow_pickle=True)
        n = int(np.asarray(data["trans"]).shape[0])
        fps = float(np.asarray(data.get("mocap_frame_rate", 120.0)).reshape(-1)[0])
        query = np.arange(n, dtype=np.float64) / fps
        pos, parents = smpl_joints(path, query, bvh)
        stride = max(int(args.stride), 1)
        ids = list(range(0, n, stride))[: args.max_frames or None]
        edges = [(int(parent), int(child)) for child, parent in enumerate(parents) if parent >= 0]
        out = out_root / sid / Path(path).stem
        npz = out / "skeleton_zup.npz"
        gen = cli_common.media_path(out, "skeleton_zup", args.gen)
        if cli_common.outputs_ready([npz, gen]) and not args.force:
            continue
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, positions=pos[ids], frame_ids=np.asarray(ids), edges=np.asarray(edges), raw_fps=np.asarray([fps]))
        frames = [render_frame(pos[i], edges, i, n) for i in ids]
        gen.parent.mkdir(parents=True, exist_ok=True)
        (cli_common.write_gif if args.gen == "gif" else cli_common.write_mp4)(frames, gen, cli_common.viz_fps(args.fps or fps, stride))
        print(f"SMPL: {path}\noutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
