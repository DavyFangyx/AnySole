"""MotionPRO-InsoleAdapted private adapter over the frozen shared facts contract.

Ownership: this module is the single MotionPRO-owned place that turns a
``shared.session.v1`` artifact into MotionPRO model inputs.  Shared facts stay
read-only; every output lands below ``model_inputs/MotionPRO/``.

Frozen conversions (audited in D_Test4, do not redefine):

  shared pressure_48 (2 x 4x12) --rasterize_feet--> dual-sole raster (T,160,120)
  raster --bilinear (align_corners=False)--> 96x96, divided by 255 at dataset load

The raster is the current fixed left/right block layout and is now called a
*dual-sole raster*, never a pressure carpet.  ``contact.npy`` is NOT a public
contact ground truth: it is the MotionPRO-private four-level soft-f6
{0.05, 0.30, 0.70, 0.95} foot loss weight in the ten-column upstream contract,
non-zero only in columns 6/7.

The soft-f6 criterion re-implements the frozen F6 design (fix_plan_v2.md F6a;
global constants, no per-session fitting).  The same constants are mirrored in
``anysole/data/contact_adapter.py`` and ``AnysoleWorkspace/tool/contact_labels.py``;
MotionPRO never reads the binarized AnySole label NPYs.

Two-phase gate (task book 03): C1 = audit artifacts for representative
sessions only, no training; C2 = full model input generation after user
confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri  # noqa: E402
from AnysoleWorkspace.tool._bvh_aligner_pose import parse_bvh_aligner  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

ADAPTER_VERSION = "adapter_v1"
WORKSPACE_ROOT = REPO_ROOT / "AnysoleWorkspace"
SHARED_SESSION_ROOT = WORKSPACE_ROOT / "shared/facts/sessions/cam3"
PROTOCOL_SPLIT = WORKSPACE_ROOT / "protocol/splits/default/splits.csv"
MODEL_INPUT_ROOT = WORKSPACE_ROOT / "model_inputs" / "MotionPRO" / ADAPTER_VERSION / "cam3"
REPORT_ROOT = WORKSPACE_ROOT / "reports" / "MotionPRO_insole_adapted"

# ---------------------------------------------------------------- frozen raster
# Audited by the user in D_Test4.  These values must not drift; they are also
# mirrored in the D_Test4 display layer (results_display/script/utils/
# render_common.py LEFT/RIGHT_FOOT_BOX) and were the frozen conversion of the
# removed generate_baseline_tactile.py.
PRESSURE_HW = (160, 120)
PRESSURE_CLIP = 1023.0
LEFT_FOOT_BOX = (slice(40, 120), slice(6, 54))
RIGHT_FOOT_BOX = (slice(40, 120), slice(66, 114))
BLOCK_REPEAT = (20, 4)  # rows, cols of one 4x12 cell inside the raster

# ---------------------------------------------------------------- frozen soft-f6
FPS = 40.0
CONTACT_COLS = 10
LEFT_COL, RIGHT_COL = 6, 7
F6_V_LO, F6_V_HI, F6_A_THR, F6_H_HIGH = 0.3, 0.6, 3.0, 0.20
F6_MIN_AIR = 10
F6_SOFT = {"contact_loaded": 0.95, "contact_unloaded": 0.70,
           "air_loaded": 0.30, "air_unloaded": 0.05}


def rasterize_feet(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    """Frozen 4x12-per-foot -> dual-sole raster (T,160,120), values 0..255."""
    height, width = PRESSURE_HW
    img = np.zeros((left48.shape[0], height, width), dtype=np.float32)
    left = np.clip(left48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    right = np.clip(right48, 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
    left = left.reshape(-1, 4, 12)
    right = right.reshape(-1, 4, 12)
    left_block = np.repeat(np.repeat(left, BLOCK_REPEAT[0], axis=1), BLOCK_REPEAT[1], axis=2)
    right_block = np.repeat(np.repeat(right, BLOCK_REPEAT[0], axis=1), BLOCK_REPEAT[1], axis=2)
    img[:, LEFT_FOOT_BOX[0], LEFT_FOOT_BOX[1]] = left_block
    img[:, RIGHT_FOOT_BOX[0], RIGHT_FOOT_BOX[1]] = right_block
    return img


def crop_cells(pressure_t: np.ndarray) -> np.ndarray:
    """Frozen inverse (audit parity helper): (T,160,120) -> (T,2,4,12)."""
    out = []
    for box in (LEFT_FOOT_BOX, RIGHT_FOOT_BOX):
        block = np.asarray(pressure_t[:, box[0], box[1]], dtype=np.float32)
        out.append(block.reshape(-1, 4, BLOCK_REPEAT[0], 12, BLOCK_REPEAT[1]).mean(axis=(2, 4)))
    return np.stack(out, axis=1)


# ---------------------------------------------------------------- shared facts

def shared_session_dir(session_id: str) -> Path:
    matches = sorted(
        path.parent
        for path in SHARED_SESSION_ROOT.glob(f"*/*/{session_id}/session.json")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"shared session {session_id!r} not found under {SHARED_SESSION_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"session id is ambiguous in shared facts: {session_id}: {matches}")
    return matches[0]


def load_shared_session(session_id: str) -> dict:
    session_dir = shared_session_dir(session_id)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = dict(np.load(session_dir / "frames.npz", allow_pickle=True))
    pressure = dict(np.load(session_dir / "pressure_48.npz", allow_pickle=True))
    frame_id = np.asarray(frames["frame_id"])
    if not np.array_equal(frame_id, np.arange(len(frame_id), dtype=frame_id.dtype)):
        raise ValueError(f"{session_dir}: shared frame_id must be unique and monotonic from zero")
    for key in ("left48", "right48"):
        if np.asarray(pressure[key]).shape != (len(frame_id), 48):
            raise ValueError(f"{session_dir}: pressure.{key} shape != (T,48)")
    if not np.array_equal(np.asarray(pressure["frame_id"]), frame_id):
        raise ValueError(f"{session_dir}: pressure frame_id differs from frames.npz")
    if str(meta.get("session_id")) != str(session_id):
        raise ValueError(f"{session_dir}: session.json identity mismatch")
    return {
        "session_id": str(session_id),
        "dir": session_dir,
        "meta": meta,
        "frames": frames,
        "pressure": pressure,
        "source_artifact": session_dir / "artifact.json",
    }


# ---------------------------------------------------------------- soft-f6 (F6 frozen)

def _foot_signal(ctx: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, name in enumerate(ctx["names"])
               if name.lower().startswith(f"{side}foot") or name.lower().startswith(f"{side}toe")]
    if not indices:
        raise ValueError(f"no {side} foot joints in BVH")
    foot = ctx["pts"][:, indices, :]
    height = foot[:, :, 2].min(axis=1) - float(ctx["floor"])
    speed = np.zeros(len(foot), dtype=np.float64)
    speed[1:] = np.linalg.norm(foot[1:] - foot[:-1], axis=2).mean(axis=1) * FPS
    return height.astype(np.float64), speed


def _f6_signals(ctx: dict, side: str) -> dict:
    indices = [i for i, name in enumerate(ctx["names"])
               if name.lower().startswith(f"{side}foot") or name.lower().startswith(f"{side}toe")]
    filtered = median_filter(ctx["pts"][:, indices, :], size=(3, 1, 1))
    center = filtered.mean(axis=1)
    velocity = np.zeros_like(center)
    velocity[1:] = (center[1:] - center[:-1]) * FPS
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_z = np.zeros(len(center), dtype=np.float64)
    acceleration_z[2:] = (center[2:, 2] - center[:-2, 2]) / 2.0 * FPS * FPS
    height, _ = _foot_signal(ctx, side)
    return {"v": speed, "az": acceleration_z, "h": height}


def _schmitt(values: np.ndarray, low: float, high: float) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    result[0] = values[0] > high
    for index in range(1, len(values)):
        result[index] = values[index] >= low if result[index - 1] else values[index] > high
    return result


def _f6_pressure(ctx: dict, side: str, airborne: np.ndarray) -> dict:
    cells = ctx["left48"] if side == "left" else ctx["right48"]
    calibrated = int(airborne.sum()) >= F6_MIN_AIR
    baseline = np.percentile(cells[airborne], 90, axis=0) if calibrated else np.zeros(48)
    corrected = np.clip(cells - baseline, 0.0, None).sum(axis=1)
    stance = ~airborne
    reference = float(np.median(corrected[stance])) if stance.any() else float(np.median(corrected))
    low = max(80.0, 0.2 * reference)
    return {"corrected": corrected, "loaded": _schmitt(corrected, low, 1.5 * low)}


def _f6_machine(ctx: dict, side: str, loaded: np.ndarray) -> np.ndarray:
    signals = _f6_signals(ctx, side)
    speed, accel, height = signals["v"], signals["az"], signals["h"]
    state = np.ones(len(speed), dtype=np.int8)
    for index in range(2, len(speed)):
        to_air = speed[index] > F6_V_HI and speed[index - 1] > F6_V_HI
        to_contact = ((speed[index] < F6_V_LO and speed[index - 1] < F6_V_LO and
                       abs(accel[index]) < F6_A_THR and abs(accel[index - 1]) < F6_A_THR)
                      or (bool(loaded[index]) and speed[index] < F6_V_HI))
        if to_contact and height[index] >= F6_H_HIGH and not loaded[index]:
            to_contact = False
        if state[index - 1] == 1 and to_air:
            state[index] = 0
        elif state[index - 1] == 0 and to_contact:
            state[index] = 1
        else:
            state[index] = state[index - 1]
    return state


def _f6_pipeline(ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """(motion_f6 states, pressure_f6 loaded) per foot, frozen F6 criterion."""
    states, loaded_values = [], []
    for side in ("left", "right"):
        signals = _f6_signals(ctx, side)
        seed_air = signals["v"] > F6_V_HI
        first = _f6_pressure(ctx, side, seed_air)
        state = _f6_machine(ctx, side, first["loaded"])
        second = _f6_pressure(ctx, side, state == 0)
        state = _f6_machine(ctx, side, second["loaded"])
        states.append(state)
        loaded_values.append(second["loaded"])
    return np.stack(states, axis=1), np.stack(loaded_values, axis=1)


def soft_f6_contact(ctx: dict) -> np.ndarray:
    """Private (T,10) contact: columns 6/7 hold the four-level soft-f6
    foot loss weights; the other eight columns are always zero."""
    states, loaded_values = _f6_pipeline(ctx)
    out = np.zeros((ctx["n"], CONTACT_COLS), dtype=np.float32)
    for column, side in enumerate(("left", "right")):
        state, loaded = states[:, column], loaded_values[:, column]
        values = np.zeros(len(state), dtype=np.float32)
        values[state == 1] = np.where(loaded[state == 1], F6_SOFT["contact_loaded"],
                                      F6_SOFT["contact_unloaded"])
        values[state == 0] = np.where(loaded[state == 0], F6_SOFT["air_loaded"],
                                      F6_SOFT["air_unloaded"])
        out[:, (LEFT_COL, RIGHT_COL)[column]] = values
    return out


def _context(session: dict) -> dict:
    """F6 context on the shared aligned frame grid (no re-alignment)."""
    frames = session["frames"]
    pressure = session["pressure"]
    n = len(frames["frame_id"])
    bvh_path = resolve_uri(session["meta"]["source_files"]["bvh"], must_exist=True)
    parsed = parse_bvh_aligner(bvh_path, trim_leading_seconds=0.0)
    joints_src = np.asarray(parsed["joints"], dtype=np.float64)
    src_t = np.arange(len(joints_src), dtype=np.float64) * float(parsed["frame_time"])
    query_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    points = np.empty((n,) + joints_src.shape[1:], dtype=np.float64)
    clipped = np.clip(query_t, src_t[0], src_t[-1])
    for joint in range(joints_src.shape[1]):
        for axis in range(3):
            points[:, joint, axis] = np.interp(clipped, src_t, joints_src[:, joint, axis])
    if points.size and float(np.ptp(points[0], axis=0).max()) > 5.0:
        points *= 0.01
    floor = float(np.percentile(points[:, :, 2], 5))
    return {
        "meta": {"subject": session["meta"].get("subject", ""),
                 "session_id": session["session_id"]},
        "n": n,
        "pts": points.astype(np.float32),
        "names": parsed["names"],
        "floor": floor,
        "left48": np.asarray(pressure["left48"], dtype=np.float64),
        "right48": np.asarray(pressure["right48"], dtype=np.float64),
    }


# ---------------------------------------------------------------- output paths

def model_input_dir(session_id: str) -> Path:
    session_dir = shared_session_dir(session_id)
    date, subject = session_dir.parts[-3], session_dir.parts[-2]
    return MODEL_INPUT_ROOT / date / subject / session_id


def build_one(session_id: str, *, force: bool = False) -> Path:
    """Generate the C1/C2 model input pieces owned by this adapter:
    pressure.npz (dual-sole raster), contact.npy (private soft-f6),
    frame_id.npy and artifact.json."""
    session = load_shared_session(session_id)
    out_dir = model_input_dir(session_id)
    pressure_file = out_dir / "pressure.npz"
    contact_file = out_dir / "contact.npy"
    frame_file = out_dir / "frame_id.npy"
    if not force and all(path.is_file() for path in (pressure_file, contact_file, frame_file)):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    left48 = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right48 = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    raster = rasterize_feet(left48, right48)
    ctx = _context(session)
    contact = soft_f6_contact(ctx)

    temporary = out_dir / f".pressure.{id(session)}.tmp.npz"
    np.savez_compressed(temporary, pressure=raster.astype(np.float32))
    temporary.replace(pressure_file)
    np.save(out_dir / ".contact.tmp.npy", contact)
    Path(out_dir / ".contact.tmp.npy").replace(contact_file)
    np.save(out_dir / ".frame_id.tmp.npy", frame_id)
    Path(out_dir / ".frame_id.tmp.npy").replace(frame_file)

    write_artifact(
        out_dir,
        schema_version="model_input.motionpro.insole_adapted.v1",
        producer="AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py",
        repository_root=REPO_ROOT,
        parameters={
            "adapter_version": ADAPTER_VERSION,
            "raster": {
                "semantic": "dual_sole_raster",
                "layout": "fixed left/right blocks, D_Test4-audited, frozen",
                "shape": [int(PRESSURE_HW[0]), int(PRESSURE_HW[1])],
                "clip": float(PRESSURE_CLIP),
                "left_box": [b.start for b in LEFT_FOOT_BOX] + [b.stop for b in LEFT_FOOT_BOX],
                "right_box": [b.start for b in RIGHT_FOOT_BOX] + [b.stop for b in RIGHT_FOOT_BOX],
                "cell_block": list(BLOCK_REPEAT),
                "model_input": "bilinear 96x96 align_corners=False /255 at dataset load",
            },
            "contact": {
                "physical_file": "contact.npy",
                "semantic_role": "motionpro_private_soft_f6_foot_loss_weight",
                "value_set": [0.05, 0.30, 0.70, 0.95],
                "formal_metric_input": False,
                "native_diagnostic_only": True,
            },
        },
        source_artifacts=[
            canonical_uri(session["dir"] / "pressure_48.npz"),
            canonical_uri(session["dir"] / "frames.npz"),
            session["meta"].get("source_files", {}).get("bvh", ""),
        ],
        source_hashes={
            "pressure_48": sha256_file(session["dir"] / "pressure_48.npz"),
            "frames": sha256_file(session["dir"] / "frames.npz"),
            "session_artifact": sha256_file(session["source_artifact"])
            if session["source_artifact"].is_file() else "",
        },
        frame_id_min=int(frame_id.min()),
        frame_id_max=int(frame_id.max()),
        frame_count=len(frame_id),
        consumers=["MotionPRO-InsoleAdapted"],
    )
    return out_dir


# ---------------------------------------------------------------- C1 audit

def window_stats(n_frames: int, window_length: int, invalid: np.ndarray) -> dict:
    """M5 accounting: complete windows only (train/eval); fake/invalid
    exclusions and the tail frames that do not fill a window."""
    n_complete = n_frames // window_length
    kept = 0
    dropped_fake = 0
    for index in range(n_complete):
        left, right = index * window_length, (index + 1) * window_length
        if invalid[left:right].any():
            dropped_fake += 1
        else:
            kept += 1
    tail = n_frames - n_complete * window_length
    return {"n_frames": n_frames, "window_length": window_length,
            "n_complete_windows": n_complete, "kept_windows": kept,
            "dropped_invalid_windows": dropped_fake, "tail_frames_excluded": tail}


def audit_one(session_id: str, out_dir: Path) -> dict:
    """C1 audit artifacts: raw 4x12, dual-sole raster, soft-f6 time series
    and value distributions, plus per-foot statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session = load_shared_session(session_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    valid = np.asarray(session["frames"]["valid"], dtype=np.uint8)
    fake = np.asarray(session["frames"]["fake"], dtype=np.uint8)
    invalid = (valid == 0).astype(bool)
    left48 = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right48 = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    ctx = _context(session)
    contact = soft_f6_contact(ctx)
    states, loaded = _f6_pipeline(ctx)
    raster = rasterize_feet(left48, right48)

    t = np.arange(len(frame_id)) / FPS
    n = len(frame_id)

    # Fig 1: raw 4x12 per foot (mean stance heatmap + total sum time series)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for row, (cells, title, color) in enumerate(
        ((left48, "left 4x12", "Blues"), (right48, "right 4x12", "Reds"))
    ):
        im = axes[row, 0].imshow(cells.reshape(n, 4, 12).mean(axis=0), aspect="auto", cmap=color)
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046)
        axes[row, 0].set_title(f"{title} mean cell value")
        axes[row, 1].plot(t, cells.sum(axis=1), lw=0.7)
        axes[row, 1].set_title(f"{title} total sum / frame")
        axes[row, 1].set_xlabel("t (s, shared frame grid)")
    fig.suptitle(f"{session_id} raw 4x12 (shared pressure_48)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{session_id}_raw_4x12.png", dpi=110)
    plt.close(fig)

    # Fig 2: dual-sole raster + bilinear 96x96 model input at a stance frame
    stance = int(np.argmax(left48.sum(axis=1) + right48.sum(axis=1)))
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(raster)).float()
    r96 = torch.nn.functional.interpolate(tensor.unsqueeze(1), size=(96, 96),
                                          mode="bilinear", align_corners=False).squeeze(1).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(raster[stance], cmap="inferno")
    axes[0].set_title(f"dual-sole raster 160x120, frame {stance}")
    axes[1].imshow(r96[stance] / 255.0, cmap="inferno")
    axes[1].set_title("model input bilinear 96x96 /255")
    fig.suptitle(f"{session_id} dual-sole raster (frozen fixed left/right blocks)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{session_id}_raster.png", dpi=110)
    plt.close(fig)

    # Fig 3: soft-f6 time series per foot
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for row, (side, col) in enumerate((("left", LEFT_COL), ("right", RIGHT_COL))):
        axes[row].plot(t, contact[:, col], lw=0.8, color="tab:blue")
        axes[row].set_ylim(-0.08, 1.08)
        axes[row].set_yticks(sorted(F6_SOFT.values()))
        axes[row].set_title(f"{side} soft-f6 foot loss weight (col {col})")
        for frame in np.where(fake)[0]:
            axes[row].axvspan(frame / FPS, (frame + 1) / FPS, color="red", alpha=0.25)
    axes[0].legend(["weight", "fake frame"], loc="upper right")
    axes[1].set_xlabel("t (s, shared frame grid)")
    fig.suptitle(f"{session_id} MotionPRO private four-level soft-f6")
    fig.tight_layout()
    fig.savefig(out_dir / f"{session_id}_soft_f6_series.png", dpi=110)
    plt.close(fig)

    # Fig 4: value distribution per foot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    stats = {}
    for column, (side, col) in enumerate((("left", LEFT_COL), ("right", RIGHT_COL))):
        values = contact[:, col]
        counts = {level: int((values == level).sum()) for level in sorted(F6_SOFT.values())}
        axes[column].bar([str(k) for k in counts], list(counts.values()), color="tab:blue")
        axes[column].set_title(f"{side} soft value counts")
        stats[side] = {
            "counts": counts,
            "contact_rate": float((states[:, column] == 1).mean()),
            "loaded_rate": float(loaded[:, column].mean()),
            "contact_loaded": counts[0.95],
            "contact_unloaded": counts[0.70],
            "air_loaded": counts[0.30],
            "air_unloaded": counts[0.05],
        }
    fig.suptitle(f"{session_id} soft-f6 value distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"{session_id}_soft_f6_hist.png", dpi=110)
    plt.close(fig)

    stats.update({
        "session_id": session_id,
        "n_frames": n,
        "n_valid": int(valid.sum()),
        "n_fake": int(fake.sum()),
        "soft_value_set": sorted(F6_SOFT.values()),
        "nonzero_columns": [int(c) for c in range(CONTACT_COLS)
                            if np.any(contact[:, c] != 0)],
        "windows": window_stats(n, 20, invalid),
        "raster_value_range": [float(raster.min()), float(raster.max())],
    })
    (out_dir / f"{session_id}_audit.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------- CLI

def split_session_ids(split: str) -> list[str]:
    with Path(PROTOCOL_SPLIT).open(encoding="utf-8-sig", newline="") as handle:
        return [row[split].strip() for row in csv.DictReader(handle)
                if (row.get(split) or "").strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MotionPRO-InsoleAdapted private adapter")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--session", action="append", default=[],
                       help="session id(s); representative sessions only during C1")
    scope.add_argument("--split", choices=("train", "val", "test"),
                       help="canonical split column (C2 bulk generation only)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit", action="store_true",
                        help="also write C1 audit figures and statistics")
    parser.add_argument("--audit-dir", type=Path,
                        default=REPORT_ROOT / "C1_audit",
                        help="audit output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = list(args.session)
    if args.split:
        sessions = split_session_ids(args.split)
    for session_id in sessions:
        out_dir = build_one(session_id, force=args.force)
        print(f"{session_id}: {out_dir}")
        if args.audit:
            stats = audit_one(session_id, args.audit_dir)
            print(f"{session_id}: audit -> {args.audit_dir}; "
                  f"counts L={stats['left']['counts']} R={stats['right']['counts']}; "
                  f"windows={stats['windows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
