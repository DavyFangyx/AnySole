"""MotionPRO-InsoleAdapted private adapter over the frozen shared facts contract.

Ownership: this module is the single MotionPRO-owned place that turns a
``shared.session.v1`` artifact into MotionPRO model inputs.  Shared facts stay
read-only; every output lands below ``model_inputs/MotionPRO/``.

Frozen conversions (audited in D_Test4, do not redefine):

  shared pressure_48 (2 x 4x12) --rasterize_feet--> dual-sole raster (T,160,120)
  raster --bilinear (align_corners=False)--> 96x96, divided by 255 at dataset load

The raster is the fixed left/right block layout and is now called a
*dual-sole raster*, never a pressure carpet.  ``contact.npy`` is NOT a public
contact ground truth: it is the MotionPRO-private binary f6 foot loss weight
{0, 1} (= motion_f6 hard decision) in the ten-column upstream contract,
non-zero only in columns 6/7.  The four-level soft-f6 intermediate
{0.05, 0.30, 0.70, 0.95} is computed internally and never persisted
(2026-10-03 ruling: back to the native binary contract, unified with the
AnySole f6_soft export).

T1 (09-28 user rulings, z_note/重构执行/11_T系列_运行时输入与接入方案.md §4)
replaces that raster as the model pressure input with a **virtual carpet**
(``virtual_carpet``), and adds the missing ``smpl.npy`` supervision producer:

  shared pressure_48 --virtual_carpet--> carpet raster (T,320,120), 0..255
  raw c3d SMPL npz --build_smpl_npy--> smpl.npy (betas/body_pose/global_orient/transl)

The virtual carpet is a fixed world-frame 4.0m x 1.5m plane (1.25cm/px) in the
z-up world frame; each foot's 48-cell insole is painted as a 24cm x 12cm
footprint centred on the *foot* joint and oriented by the toe-minus-foot
direction.  Out-of-carpet pixels are simply not painted (V1: fixed world-frame
anchor, out-of-bounds means no tactile signal -- no invalid flag).  The
``rasterize_feet`` dual-sole raster is kept only for comparability
(``--legacy-raster``) and because the C1 audit artifacts were produced with it.

The world frame is the adapter's existing mocap frame: ``parse_bvh_aligner``
already emits z-up via ``new = (x_old, -z_old, y_old)`` from the y-up raw BVH
(the same "MocapVideoAligner default z-up display conversion" used by the F6
context), so no second, different frame is introduced here.

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

# ------------------------------------------------------- virtual carpet (T1-A)
# 09-28 user rulings (11 号文档 §4.1/V1/V2, FINALIZED).  The native contract is
# `lib/util/gen_contact.py`:
#     scale = 2.0 / pressure.shape[1]                      # 2m / 160 = 1.25cm/px
#     carpet_pos[i, j, 0] = i*scale + 0.5*scale            # x in [0, 1.5m), 120 cells
#     carpet_pos[i, j, 1] = -j*scale - 0.5*scale           # y in (-2m, 0], 160 cells
#     carpet_pos[i, j, 2] = 0.                             # ground plane z=0, z-up world
# The long axis is doubled to the 4m test corridor (user ruling): 320 cells on
# the same 1.25cm pitch, so scale = 4.0/320 == 2.0/160 == 1.25cm/px.
CARPET_HW = (320, 120)          # (long axis j, wide axis i) == pressure (y, x)
CARPET_CELL_M = 1.25e-2         # 1.25cm/px; == native 2.0/160
CARPET_LONG_M = 4.0             # long axis j, y_j = -(j*cell + cell/2) < 0
CARPET_WIDE_M = 1.5             # wide axis i, x_i =   i*cell + cell/2 > 0
CARPET_Y_UP_TO_Z_UP = "new = (x_old, -z_old, y_old) from y-up raw BVH; " \
                      "parse_bvh_aligner's z-up display conversion (see module docstring)"

# V2 footprint: the insole is 12cm wide (4 rows, 3cm each) x 24cm long
# (12 columns, 2cm each).  Cell order is the frozen shared contract
# (assets/foot_sensor_layout/README.md, verified numerically): channel index is
# row-major over 4 width rows x 12 length columns and **column 0 is the toe**,
# column 11 the heel.  The user's formula "(c-5.5)*2cm along the toe-minus-foot
# direction" assumes column 0 == heel, so the length offset is written here as
# (5.5 - c)*2cm to keep the frozen channel order physically correct.
FOOT_ROWS, FOOT_COLS = 4, 12
FOOT_ROW_M, FOOT_COL_M = 0.030, 0.020
FOOT_HALF_COL_M = FOOT_COL_M / 2.0
FOOT_HALF_ROW_M = FOOT_ROW_M / 2.0
FOOT_JOINT_SUFFIX = ("foot", "toe")   # BVH-23 names: LeftFoot/LeftToeBase, ...

# ---------------------------------------------------------------- frozen soft-f6
FPS = 40.0
CONTACT_COLS = 10
LEFT_COL, RIGHT_COL = 6, 7
F6_V_LO, F6_V_HI, F6_A_THR, F6_H_HIGH = 0.3, 0.6, 3.0, 0.20
F6_MIN_AIR = 10
# Four-level soft-f6 intermediate (motion_f6 state x pressure_f6 consistency).
# Computed internally for the F6 state machine; never persisted (2026-10-03
# ruling: contact.npy exports the binary state only, native {0,1} contract).
F6_SOFT = {"contact_loaded": 0.95, "contact_unloaded": 0.70,
           "air_loaded": 0.30, "air_unloaded": 0.05}


def rasterize_feet(left48: np.ndarray, right48: np.ndarray) -> np.ndarray:
    """SUPERSEDED by :func:`virtual_carpet` (T1-A, 09-28).  Frozen 4x12-per-foot
    -> dual-sole raster (T,160,120), values 0..255.

    Kept only so ``--legacy-raster`` can reproduce the C1-audit artifact
    (D_Test4-audited fixed left/right blocks) for comparability; it is no longer
    the model pressure input.
    """
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


# ---------------------------------------------------------------- virtual carpet

def carpet_grid_x() -> np.ndarray:
    """Wide axis (i, model columns): x_i = i*1.25cm + 0.625cm in [0, 1.5m)."""
    return np.arange(CARPET_HW[1], dtype=np.float64) * CARPET_CELL_M + 0.5 * CARPET_CELL_M


def carpet_grid_y() -> np.ndarray:
    """Long axis (j, model rows): y_j = -(j*1.25cm + 0.625cm) in (-4m, 0]."""
    return -(np.arange(CARPET_HW[0], dtype=np.float64) * CARPET_CELL_M + 0.5 * CARPET_CELL_M)


def _joint_index(names: list[str], side: str, suffix: str) -> int:
    """BVH joint index for e.g. ("left", "foot") -> LeftFoot.

    Exact name first (LeftFoot / LeftToeBase), else the first non-EndSite joint
    whose lowercased name starts with the side+suffix prefix.
    """
    lowered = [str(name).lower() for name in names]
    for preferred in (f"{side}{suffix}", f"{side}{suffix}base"):
        if preferred in lowered:
            return lowered.index(preferred)
    for index, name in enumerate(lowered):
        if name.startswith(f"{side}{suffix}") and "endsite" not in name:
            return index
    raise ValueError(f"no {side} {suffix} joint in BVH names: {names}")


def foot_tracks(ctx: dict) -> dict[str, dict[str, np.ndarray]]:
    """Per-side carpet-plane foot tracks from the aligned BVH context.

    Footprint center = the foot (ankle) joint projected on the carpet plane;
    direction = toe joint - foot joint (both projected, z ignored).  Degenerate
    directions (zero length / non-finite) inherit the previous valid frame and,
    failing that, the carpet-forward default (0, -1).
    """
    points = np.asarray(ctx["pts"], dtype=np.float64)
    tracks = {}
    for side in ("left", "right"):
        foot = _joint_index(ctx["names"], side, FOOT_JOINT_SUFFIX[0])
        toe = _joint_index(ctx["names"], side, FOOT_JOINT_SUFFIX[1])
        center = points[:, foot, :2].copy()
        toe_xy = points[:, toe, :2].copy()
        direction = toe_xy - center
        length = np.linalg.norm(direction, axis=1)
        bad = ~np.isfinite(length) | (length <= 1e-6)
        carry = np.array([0.0, -1.0])
        for index in range(len(direction)):
            if bad[index]:
                direction[index] = carry
            else:
                carry = direction[index] / length[index]
                direction[index] = carry
        tracks[side] = {"center": center, "toe": toe_xy, "direction": direction,
                        "joints": {"foot": ctx["names"][foot], "toe": ctx["names"][toe]}}
    return tracks


def _paint_footprint(image_t: np.ndarray, center: np.ndarray, direction: np.ndarray,
                     values: np.ndarray) -> int:
    """Paint one 48-cell insole footprint into one carpet frame (in place).

    Cell (row r, col c) is a FOOT_ROW_M x FOOT_COL_M rectangle centred at
    ``center + (5.5 - c)*FOOT_COL_M * d + (r - 1.5)*FOOT_ROW_M * p`` where d is
    the unit heel->toe direction and p = (-d_y, d_x) is its +90 degree
    right-hand rotation about +z (the documented consistent perpendicular; the
    medial/lateral sign of the row axis is not fixed by the shared contract).
    A pixel takes the max value of every cell rectangle covering its center, and
    cells outside the carpet are simply dropped.  Returns painted pixel count.
    """
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(direction)):
        return 0
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
    rows = np.arange(FOOT_ROWS, dtype=np.float64)[:, None]
    cols = np.arange(FOOT_COLS, dtype=np.float64)[None, :]
    along = (5.5 - cols) * FOOT_COL_M            # (1,12) toe-ward, col 0 = toe
    across = (rows - 1.5) * FOOT_ROW_M           # (4,1)
    centers = center + along[..., None] * direction + across[..., None] * perpendicular
    centers = centers.reshape(-1, 2)

    support = np.abs(FOOT_HALF_COL_M * direction) + np.abs(FOOT_HALF_ROW_M * perpendicular)
    low = centers.min(axis=0) - support
    high = centers.max(axis=0) + support
    height, width = image_t.shape
    i0 = int(np.floor((low[0] - 0.5 * CARPET_CELL_M) / CARPET_CELL_M))
    i1 = int(np.ceil((high[0] - 0.5 * CARPET_CELL_M) / CARPET_CELL_M))
    j0 = int(np.floor((-high[1] - 0.5 * CARPET_CELL_M) / CARPET_CELL_M))
    j1 = int(np.ceil((-low[1] - 0.5 * CARPET_CELL_M) / CARPET_CELL_M))
    if i1 < 0 or j1 < 0 or i0 >= width or j0 >= height:
        return 0
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, width - 1), min(j1, height - 1)

    grid_x, grid_y = np.meshgrid(
        np.arange(i0, i1 + 1) * CARPET_CELL_M + 0.5 * CARPET_CELL_M,
        -(np.arange(j0, j1 + 1) * CARPET_CELL_M + 0.5 * CARPET_CELL_M),
        indexing="xy",
    )
    pixels = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    relative = pixels[:, None, :] - centers[None, :, :]
    along_offset = relative @ direction
    across_offset = relative @ perpendicular
    inside = ((np.abs(along_offset) <= FOOT_HALF_COL_M)
              & (np.abs(across_offset) <= FOOT_HALF_ROW_M))
    painted = np.where(inside, values[None, :], 0.0).max(axis=1)
    patch = painted.reshape(j1 - j0 + 1, i1 - i0 + 1)
    region = image_t[j0:j1 + 1, i0:i1 + 1]
    np.maximum(region, patch, out=region)
    return int(np.count_nonzero(patch))


def virtual_carpet(ctx: dict, left48: np.ndarray, right48: np.ndarray, *,
                   anchor: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """(T,320,120) virtual tactile carpet + painting statistics.

    Ruled geometry (11 号文档 §4.1): 4.0m x 1.5m, 1.25cm/px, fixed world frame,
    z=0 plane in the adapter's z-up frame.  Values keep the frozen
    ``clip(raw, 0, 1023)/1023*255`` convention of ``rasterize_feet``; the
    bilinear 96x96 resize and the /255 are consumer-side dataset contracts and
    are deliberately NOT applied here.  ``anchor`` is a diagnostic-only world
    (dx, dy) shift used by the V3 acceptance test to probe the fixed-anchor
    ruling; production always calls with the default ``None``.
    """
    image = np.zeros((np.asarray(left48).shape[0], CARPET_HW[0], CARPET_HW[1]), dtype=np.float32)
    tracks = foot_tracks(ctx)
    if anchor is not None:
        shift = np.asarray(anchor, dtype=np.float64)
    else:
        shift = np.zeros(2)
    x_axis, y_axis = carpet_grid_x(), carpet_grid_y()
    stats = {"painted_px": 0, "frames_any_paint": 0, "per_side_painted": {},
             "per_side_in_bounds_frames": {}, "anchor": None if anchor is None else [float(v) for v in shift]}
    for side, cells in (("left", left48), ("right", right48)):
        values = np.clip(np.asarray(cells, dtype=np.float64), 0.0, PRESSURE_CLIP) / PRESSURE_CLIP * 255.0
        values = values.reshape(-1, FOOT_ROWS, FOOT_COLS)
        painted_total = 0
        in_bounds = np.zeros(image.shape[0], dtype=bool)
        for frame in range(image.shape[0]):
            center = tracks[side]["center"][frame] + shift
            in_bounds[frame] = (x_axis[0] <= center[0] <= x_axis[-1]
                                and y_axis[-1] <= center[1] <= y_axis[0])
            painted_total += _paint_footprint(image[frame], center,
                                              tracks[side]["direction"][frame],
                                              values[frame].reshape(-1))
        stats["per_side_painted"][side] = painted_total
        stats["per_side_in_bounds_frames"][side] = int(in_bounds.sum())
        stats["painted_px"] += painted_total
    any_paint = np.any(image != 0.0, axis=(1, 2))
    stats["frames_any_paint"] = int(any_paint.sum())
    stats["frames_out_of_bounds"] = int((~any_paint).sum())
    stats["joints"] = {side: tracks[side]["joints"] for side in tracks}
    return image, stats


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
    """Private (T,10) contact: columns 6/7 hold the binary f6 foot loss
    weights (= the motion_f6 hard decision); the other eight columns are
    always zero.  2026-10-03 ruling: the four-level soft intermediate is
    not persisted — this returns the native {0,1} contract, unified with
    the AnySole f6_soft export."""
    states, _ = _f6_pipeline(ctx)
    out = np.zeros((ctx["n"], CONTACT_COLS), dtype=np.float32)
    for column, side in enumerate(("left", "right")):
        out[:, (LEFT_COL, RIGHT_COL)[column]] = states[:, column].astype(np.float32)
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


# ---------------------------------------------------------------- smpl.npy (D1)

SMPL_KEYS = ("betas", "body_pose", "global_orient", "transl")


def load_smpl_source(session: dict) -> dict:
    """Read the raw c3d/mosh SMPL npz behind ``meta.source_files.smpl``.

    Raw contract (D4, verified): poses (Ts,72) axis-angle for 24 joints with
    ``poses[:, :3] == root_orient``, ``poses[:, 3:66] == pose_body`` and
    ``poses[:, 66:72] == 0``; trans (Ts,3) m; betas (10,); source_frame_times_s
    (Ts,) at 120Hz on the mocap clock.
    """
    uri = session["meta"]["source_files"]["smpl"]
    source = resolve_uri(uri, must_exist=True)
    data = dict(np.load(source, allow_pickle=True))
    poses = np.asarray(data["poses"], dtype=np.float64)
    times = np.asarray(data["source_frame_times_s"], dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"{source}: poses must be (Ts,72), got {poses.shape}")
    if len(times) != len(poses):
        raise ValueError(f"{source}: source_frame_times_s {len(times)} != poses {len(poses)}")
    if not np.array_equal(poses[:, :3], np.asarray(data["root_orient"])):
        raise ValueError(f"{source}: poses[:,:3] != root_orient (raw contract drift)")
    if not np.array_equal(poses[:, 3:66], np.asarray(data["pose_body"])):
        raise ValueError(f"{source}: poses[:,3:66] != pose_body (raw contract drift)")
    if float(np.abs(poses[:, 66:]).max()) != 0.0:
        raise ValueError(f"{source}: poses[:,66:72] must be exactly zero")
    return {
        "uri": uri,
        "path": source,
        "poses": poses,
        "times": times,
        "trans": np.asarray(data["trans"], dtype=np.float64),
        "betas": np.asarray(data["betas"], dtype=np.float64).reshape(-1),
    }


def _interp_to_grid(source_times: np.ndarray, values: np.ndarray,
                    query_times: np.ndarray) -> tuple[np.ndarray, int]:
    """Column-wise linear interpolation onto the shared mocap grid.

    The 120Hz source grid is not phase-aligned with the 40Hz grid (S5091 offset
    0.529ms), so resampling must be interpolation, never decimation.  Query
    times are clipped into the source span and the out-of-span count is
    returned for the artifact.
    """
    clipped = np.clip(query_times, source_times[0], source_times[-1])
    flat = values.reshape(len(values), -1)
    out = np.empty((len(query_times), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(clipped, source_times, flat[:, column])
    out_of_span = int(np.count_nonzero((query_times < source_times[0])
                                       | (query_times > source_times[-1])))
    return out.reshape((len(query_times),) + values.shape[1:]), out_of_span


def build_smpl_npy(session: dict) -> dict:
    """Write ``smpl.npy`` next to the other model inputs.

    Four-key contract consumed by ``lib/dataset/image_pressure.py`` (and
    ``lib/util/gen_kps.py``): ``betas`` (10,), ``body_pose`` (T,69),
    ``global_orient`` (T,3), ``transl`` (T,3), T == len(frame_id).  Kept
    independent of the carpet so it can be produced even where painting fails.
    """
    out_dir = model_input_dir(session["session_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    source = load_smpl_source(session)
    query = np.asarray(session["frames"]["mocap_time_s"], dtype=np.float64)
    global_orient, missed_go = _interp_to_grid(source["times"], source["poses"][:, :3], query)
    body_pose, missed_bp = _interp_to_grid(source["times"], source["poses"][:, 3:], query)
    transl, missed_tr = _interp_to_grid(source["times"], source["trans"], query)
    payload = {
        "betas": np.asarray(source["betas"][:10], dtype=np.float32),
        "body_pose": np.ascontiguousarray(body_pose, dtype=np.float32),
        "global_orient": np.ascontiguousarray(global_orient, dtype=np.float32),
        "transl": np.ascontiguousarray(transl, dtype=np.float32),
    }
    temporary = out_dir / f".smpl.{id(session)}.tmp.npy"
    np.save(temporary, payload, allow_pickle=True)
    temporary.replace(out_dir / "smpl.npy")
    return {
        "source_uri": source["uri"],
        "path": out_dir / "smpl.npy",
        "n_frames": int(len(query)),
        "source_frames": int(len(source["poses"])),
        "source_frame_times_s": [float(source["times"][0]), float(source["times"][-1])],
        "source_fps": float(1.0 / np.median(np.diff(source["times"]))) if len(source["times"]) > 1 else 0.0,
        "interp": "np.interp per component from source_frame_times_s onto frames.mocap_time_s "
                  "(clipped to the source span; no decimation)",
        "out_of_span_frames": int(max(missed_go, missed_bp, missed_tr)),
        "shapes": {key: list(np.asarray(value).shape) for key, value in payload.items()},
    }


# ---------------------------------------------------------------- output paths

def model_input_dir(session_id: str) -> Path:
    session_dir = shared_session_dir(session_id)
    date, subject = session_dir.parts[-3], session_dir.parts[-2]
    return MODEL_INPUT_ROOT / date / subject / session_id


def color_dir_for(session: dict) -> Path:
    """Shared per-frame rgb directory of one session (``rgb/<frame:06d>.jpg``)."""
    return Path(session["dir"]) / "rgb"


def write_color_symlinks(session: dict, out_dir: Path) -> dict:
    """Link ``color/<index:06d>.jpg`` -> the shared rgb frames.

    The native visual chain (gen_bbox / gen_image_feature) globs and sorts
    ``<seq>/color/*.jpg`` and indexes it by frame number, so the link name is
    the shared frame index (which is the model frame index: shared frame_id is
    unique, monotonic and zero-based).  gen_kps does not need ``color/``.
    """
    source = color_dir_for(session)
    target = out_dir / "color"
    target.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    linked, missing, existing = 0, [], 0
    for index in frames:
        original = source / f"{int(index):06d}.jpg"
        if not original.is_file():
            missing.append(int(index))
            continue
        link = target / f"{int(index):06d}.jpg"
        if link.is_symlink() or link.exists():
            existing += 1
            continue
        link.symlink_to(original)
        linked += 1
    return {
        "source_uri": canonical_uri(source),
        "source_dir": str(source),
        "link_dir": str(target),
        "n_frames": int(len(frames)),
        "n_linked": linked,
        "n_already_present": existing,
        "missing_frames": missing[:16],
        "n_missing": len(missing),
    }


def build_one(session_id: str, *, force: bool = False, legacy_raster: bool = False,
              color: bool = True) -> Path:
    """Generate the model input pieces owned by this adapter: pressure.npz
    (virtual carpet by default, legacy dual-sole raster with ``legacy_raster``),
    smpl.npy, contact.npy (private soft-f6), frame_id.npy, the ``color/`` link
    farm and artifact.json."""
    session = load_shared_session(session_id)
    out_dir = model_input_dir(session_id)
    pressure_file = out_dir / "pressure.npz"
    contact_file = out_dir / "contact.npy"
    frame_file = out_dir / "frame_id.npy"
    smpl_file = out_dir / "smpl.npy"
    if not force and all(path.is_file() for path in
                         (pressure_file, contact_file, frame_file, smpl_file)):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    left48 = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right48 = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    ctx = _context(session)
    if legacy_raster:
        carpet = rasterize_feet(left48, right48)
        carpet_stats = None
    else:
        carpet, carpet_stats = virtual_carpet(ctx, left48, right48)
    contact = soft_f6_contact(ctx)
    smpl_provenance = build_smpl_npy(session)
    color_stats = write_color_symlinks(session, out_dir) if color else None

    temporary = out_dir / f".pressure.{id(session)}.tmp.npz"
    np.savez_compressed(temporary, pressure=carpet.astype(np.float32))
    temporary.replace(pressure_file)
    np.save(out_dir / ".contact.tmp.npy", contact)
    Path(out_dir / ".contact.tmp.npy").replace(contact_file)
    np.save(out_dir / ".frame_id.tmp.npy", frame_id)
    Path(out_dir / ".frame_id.tmp.npy").replace(frame_file)

    if legacy_raster:
        raster_section = {
            "semantic": "dual_sole_raster",
            "layout": "fixed left/right blocks, D_Test4-audited, frozen",
            "shape": [int(PRESSURE_HW[0]), int(PRESSURE_HW[1])],
            "clip": float(PRESSURE_CLIP),
            "left_box": [b.start for b in LEFT_FOOT_BOX] + [b.stop for b in LEFT_FOOT_BOX],
            "right_box": [b.start for b in RIGHT_FOOT_BOX] + [b.stop for b in RIGHT_FOOT_BOX],
            "cell_block": list(BLOCK_REPEAT),
            "model_input": "bilinear 96x96 align_corners=False /255 at dataset load",
        }
    else:
        raster_section = {
            "semantic": "virtual_tactile_carpet",
            "replaces": "dual_sole_raster (rasterize_feet, D_Test4-audited)",
            "shape": [int(CARPET_HW[0]), int(CARPET_HW[1])],
            "long_axis": {
                "semantic": "y (pressure rows, j)",
                "cells": int(CARPET_HW[0]),
                "extent_m": float(CARPET_LONG_M),
                "grid_m": float(CARPET_CELL_M),
                "formula": "y_j = -(j*1.25cm + 0.625cm), j=0..319",
            },
            "wide_axis": {
                "semantic": "x (pressure columns, i)",
                "cells": int(CARPET_HW[1]),
                "extent_m": float(CARPET_WIDE_M),
                "grid_m": float(CARPET_CELL_M),
                "formula": "x_i = i*1.25cm + 0.625cm, i=0..119",
            },
            "frame": {
                "name": "fixed world frame, z-up, carpet plane z=0",
                "anchor": "world origin (V1 ruling: fixed world frame, no per-session recentring)",
                "y_up_to_z_up": CARPET_Y_UP_TO_Z_UP,
                "source": "raw BVH foot/toe joints via parse_bvh_aligner on frames.mocap_time_s",
            },
            "footprint": {
                "rows": int(FOOT_ROWS), "cols": int(FOOT_COLS),
                "row_pitch_m": float(FOOT_ROW_M), "col_pitch_m": float(FOOT_COL_M),
                "size_m": [FOOT_ROWS * FOOT_ROW_M, FOOT_COLS * FOOT_COL_M],
                "center": "foot joint projected on the carpet plane",
                "orientation": "unit toe - foot direction; columns run toe->heel "
                               "(shared cell contract col 0 = toe), rows run along the "
                               "perpendicular p = (-d_y, d_x) (+90deg right-hand about +z)",
                "overlap": "max over covering cells and over both feet",
                "out_of_bounds": "not painted; no invalid flag (V1 ruling)",
            },
            "value": {
                "convention": "clip(raw_cell, 0, 1023)/1023*255, float32",
                "range": [0.0, 255.0],
                "model_entry": "consumer-side bilinear 96x96 align_corners=False then /255 "
                               "(lib/dataset/image_pressure.py); NOT applied in the adapter",
            },
            "legacy_raster_flag": bool(legacy_raster),
            "stats": carpet_stats,
        }

    artifact_parameters = {
        "adapter_version": ADAPTER_VERSION,
        "raster": raster_section,
        "carpet_geometry": {
            "shape": [int(CARPET_HW[0]), int(CARPET_HW[1])],
            "extent_m": [float(CARPET_LONG_M), float(CARPET_WIDE_M)],
            "grid_m": float(CARPET_CELL_M),
            "plane": "z=0 in the z-up world frame",
            "y_up_to_z_up": CARPET_Y_UP_TO_Z_UP,
            "anchor": "fixed world frame",
        },
        "footprint_geometry": {
            "size_m": [FOOT_ROWS * FOOT_ROW_M, FOOT_COLS * FOOT_COL_M],
            "rows": int(FOOT_ROWS), "cols": int(FOOT_COLS),
            "row_pitch_m": float(FOOT_ROW_M), "col_pitch_m": float(FOOT_COL_M),
        },
        "value_convention": "clip(raw, 0, 1023)/1023*255 float32 (model /255 and 96x96 bilinear are consumer-side)",
        "legacy_raster": bool(legacy_raster),
        "smpl_npy": {
            "physical_file": "smpl.npy",
            "source_uri": smpl_provenance["source_uri"],
            "source_hash_key": "source_hashes.smpl (session.json)",
            "source_hashes_smpl": session["meta"].get("source_hashes", {}).get("smpl", ""),
            "keys": list(SMPL_KEYS),
            "shapes": smpl_provenance["shapes"],
            "interp": smpl_provenance["interp"],
            "source_frames": smpl_provenance["source_frames"],
            "source_fps": smpl_provenance["source_fps"],
            "source_frame_times_s": smpl_provenance["source_frame_times_s"],
            "out_of_span_frames": smpl_provenance["out_of_span_frames"],
            "derivation": "global_orient=poses[:,:3]; body_pose=poses[:,3:] (T,69, 6 trailing zeros); "
                          "transl=trans; betas=betas[:10]",
        },
        "contact": {
            "physical_file": "contact.npy",
            "semantic_role": "motionpro_private_f6_foot_loss_weight",
            "value_set": [0, 1],
            "soft_levels_intermediate_not_persisted": [0.05, 0.30, 0.70, 0.95],
            "nonzero_columns": [int(LEFT_COL), int(RIGHT_COL)],
            "formal_metric_input": False,
            "native_diagnostic_only": True,
        },
    }
    if color_stats is not None:
        artifact_parameters["color_links"] = {
            "physical_dir": "color/",
            "source_uri": color_stats["source_uri"],
            "n_frames": color_stats["n_frames"],
            "n_present": color_stats["n_linked"] + color_stats["n_already_present"],
            "n_linked_this_run": color_stats["n_linked"],
            "n_already_present": color_stats["n_already_present"],
            "n_missing": color_stats["n_missing"],
            "missing_frames": color_stats["missing_frames"],
            "consumer": "lib/util/gen_bbox.py + lib/util/gen_image_feature.py (glob color/*.jpg, sorted)",
        }

    write_artifact(
        out_dir,
        schema_version="model_input.motionpro.insole_adapted.v1",
        producer="AnysoleWorkspace/tool/adapters/MotionPRO/adapter.py",
        repository_root=REPO_ROOT,
        parameters=artifact_parameters,
        source_artifacts=[
            canonical_uri(session["dir"] / "pressure_48.npz"),
            canonical_uri(session["dir"] / "frames.npz"),
            session["meta"].get("source_files", {}).get("bvh", ""),
            session["meta"].get("source_files", {}).get("smpl", ""),
        ],
        source_hashes={
            "pressure_48": sha256_file(session["dir"] / "pressure_48.npz"),
            "frames": sha256_file(session["dir"] / "frames.npz"),
            "session_artifact": sha256_file(session["source_artifact"])
            if session["source_artifact"].is_file() else "",
            "smpl": session["meta"].get("source_hashes", {}).get("smpl", ""),
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
    raster_mode = parser.add_mutually_exclusive_group()
    raster_mode.add_argument("--carpet", dest="legacy_raster", action="store_false",
                             help="write the virtual tactile carpet (T,320,120) into "
                                  "pressure.npz (default, T1-A ruling)")
    raster_mode.add_argument("--legacy-raster", dest="legacy_raster", action="store_true",
                             help="write the superseded dual-sole raster (T,160,120) into "
                                  "pressure.npz instead of the virtual carpet "
                                  "(comparability only)")
    parser.set_defaults(legacy_raster=False)
    parser.add_argument("--no-color", action="store_true",
                        help="skip the color/ symlink farm (the gen_bbox/gen_image_feature "
                             "input contract)")
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
        out_dir = build_one(session_id, force=args.force,
                            legacy_raster=args.legacy_raster, color=not args.no_color)
        print(f"{session_id}: {out_dir}"
              f"{' [legacy raster]' if args.legacy_raster else ' [virtual carpet]'}")
        if args.audit:
            stats = audit_one(session_id, args.audit_dir)
            print(f"{session_id}: audit -> {args.audit_dir}; "
                  f"counts L={stats['left']['counts']} R={stats['right']['counts']}; "
                  f"windows={stats['windows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
