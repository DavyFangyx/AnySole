"""Build AnySole-private contact labels from shared facts.

The algorithms are intentionally the existing AnySole contact semantics.  The
adapter changes only the source/output boundary: pressure comes from the
public ``pressure_48.npz`` and labels land in ``model_inputs/AnySole``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool._bvh_aligner_pose import parse_bvh_aligner
from scipy.ndimage import median_filter
from anysole.data.workspace_adapter import (
    PROTOCOL_SPLIT,
    SHARED_SESSION_ROOT,
    load_shared_session,
    shared_bvh_path,
    write_label,
)


FPS = 40.0
CONTACT_SUM_THRESH = 100.0
F6_V_LO, F6_V_HI, F6_A_THR, F6_H_HIGH = 0.3, 0.6, 3.0, 0.20
F6_MIN_AIR = 10
F6_SOFT = {"contact_loaded": 0.95, "contact_unloaded": 0.70,
           "air_loaded": 0.30, "air_unloaded": 0.05}


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


def _bvh_h(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, side in enumerate(("left", "right")):
        height, _ = _foot_signal(ctx, side)
        state = 1.0
        for index, value in enumerate(height):
            if value >= 0.06:
                state = 0.0
            elif value <= 0.04:
                state = 1.0
            out[index, k] = state
    return out


def _bvh_soft(ctx: dict) -> np.ndarray:
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for k, side in enumerate(("left", "right")):
        height, speed = _foot_signal(ctx, side)
        hp = 1.0 / (1.0 + np.exp(-(0.05 - height) / 0.02))
        sp = 1.0 / (1.0 + np.exp(-(0.20 - speed) / 0.05))
        out[:, k] = (hp * sp > 0.5).astype(np.float32)
    return out


def _tactile_abs(ctx: dict) -> np.ndarray:
    return np.stack([ctx["sums_l"] > CONTACT_SUM_THRESH,
                     ctx["sums_r"] > CONTACT_SUM_THRESH], axis=1).astype(np.float32)


def _tactile_rel(ctx: dict) -> np.ndarray:
    values = []
    for sums in (ctx["sums_l"], ctx["sums_r"]):
        values.append(sums > float(sums.min()) + 0.25 * float(np.ptp(sums)))
    return np.stack(values, axis=1).astype(np.float32)


def _gmm_valley(sums: np.ndarray):
    values = np.log1p(np.maximum(sums, 0.0).astype(np.float64))
    if values.size < 40 or float(np.ptp(values)) < 1.0:
        return None
    low, high = float(np.percentile(values, 20)), float(np.percentile(values, 80))
    mask = np.ones(values.size, dtype=bool)
    for _ in range(50):
        new_mask = np.abs(values - high) < np.abs(values - low)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        if mask.sum() == 0 or (~mask).sum() == 0:
            return None
        low, high = float(values[~mask].mean()), float(values[mask].mean())
    if low > high:
        low, high = high, low
    weight = float(mask.mean())
    std0, std1 = float(values[~mask].std()), float(values[mask].std())
    if weight < 0.12 or weight > 0.88 or high - low < 2.0 * max(std0 + std1, 1e-3):
        return None
    return float(np.expm1((low * std1 + high * std0) / max(std0 + std1, 1e-9)))


def _tactile_gmm(ctx: dict) -> np.ndarray:
    values = []
    for sums in (ctx["sums_l"], ctx["sums_r"]):
        threshold = _gmm_valley(sums)
        values.append(np.ones(len(sums)) if threshold is None else sums > threshold)
    return np.stack(values, axis=1).astype(np.float32)


def _joint(ctx: dict, mode: str) -> np.ndarray:
    bvh, tactile = _bvh_h(ctx), _tactile_gmm(ctx)
    air = (bvh <= 0.5) | (tactile <= 0.5) if mode == "or" else (bvh <= 0.5) & (tactile <= 0.5)
    return (~air).astype(np.float32)


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
    states, loaded_values = [], []
    for column, side in enumerate(("left", "right")):
        signals = _f6_signals(ctx, side)
        seed_air = signals["v"] > F6_V_HI
        first = _f6_pressure(ctx, side, seed_air)
        state = _f6_machine(ctx, side, first["loaded"])
        second = _f6_pressure(ctx, side, state == 0)
        state = _f6_machine(ctx, side, second["loaded"])
        states.append(state)
        loaded_values.append(second["loaded"])
    return np.stack(states, axis=1), np.stack(loaded_values, axis=1)


def _motion_f6(ctx: dict) -> np.ndarray:
    return _f6_pipeline(ctx)[0].astype(np.float32)


def _pressure_f6(ctx: dict) -> np.ndarray:
    return _f6_pipeline(ctx)[1].astype(np.float32)


def _f6_soft(ctx: dict) -> np.ndarray:
    states, loaded_values = _f6_pipeline(ctx)
    out = np.zeros((ctx["n"], 2), dtype=np.float32)
    for column in range(2):
        state, loaded = states[:, column], loaded_values[:, column]
        out[state == 1, column] = np.where(loaded[state == 1], 0.95, 0.70)
        out[state == 0, column] = np.where(loaded[state == 0], 0.30, 0.05)
    return (out > 0.5).astype(np.float32)


METHODS = {
    "tactile_abs": _tactile_abs, "bvh_soft": _bvh_soft, "bvh_h": _bvh_h,
    "tactile_gmm": _tactile_gmm, "tactile_rel": _tactile_rel,
    "joint_or": lambda ctx: _joint(ctx, "or"),
    "joint_and": lambda ctx: _joint(ctx, "and"),
    "motion_f6": _motion_f6, "pressure_f6": _pressure_f6,
    "f6_soft": _f6_soft,
}


def _load_split_ids(split: str) -> list[str]:
    import csv

    with Path(PROTOCOL_SPLIT).open(encoding="utf-8-sig", newline="") as handle:
        return [
            row[split].strip()
            for row in csv.DictReader(handle)
            if (row.get(split) or "").strip()
        ]


def _context(session: dict) -> dict:
    frames = session["frames"]
    pressure = session["pressure"]
    n = len(frames["frame_id"])
    bvh_path = shared_bvh_path(session)
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
        "meta": {
            "subject": session["meta"].get("subject", ""),
            "session_id": session["session_id"],
        },
        "n": n,
        "pts": points.astype(np.float32),
        "names": parsed["names"],
        "floor": floor,
        "left48": np.asarray(pressure["left48"], dtype=np.float64),
        "right48": np.asarray(pressure["right48"], dtype=np.float64),
        "sums_l": np.asarray(pressure["left48"], dtype=np.float64).sum(axis=1),
        "sums_r": np.asarray(pressure["right48"], dtype=np.float64).sum(axis=1),
    }


def build_one(session_id: str, methods: list[str], *, force: bool = False,
              shared_root: Path = SHARED_SESSION_ROOT) -> list[Path]:
    session = load_shared_session(session_id, shared_root)
    ctx = None
    outputs = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown contact method {method!r}; choose from {sorted(METHODS)}")
        from anysole.data.workspace_adapter import label_path

        output = label_path(session_id, method)
        if output.is_file() and not force:
            outputs.append(output)
            continue
        if ctx is None:
            ctx = _context(session)
        # These methods are pure functions over the adapter context.  No
        # label from another method is read or used as an input.
        contact = np.asarray(METHODS[method](ctx), dtype=np.float32)
        write_label(
            output,
            session=session,
            method=method,
            contact=contact,
            parameters={
                "source": "shared/facts pressure_48 + raw BVH when required",
                "bvh_path": str(shared_bvh_path(session)),
                "semantics": "AnySole legacy method; f6_soft is exported binary (>0.5)",
            },
        )
        outputs.append(output)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build private AnySole contact labels from shared facts.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--session", action="append", default=[])
    scope.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--methods", default="f6_soft",
                        help="comma-separated methods, or 'all' (default: f6_soft)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    sessions = list(args.session)
    if args.split:
        sessions = _load_split_ids(args.split)
    if not sessions:
        raise SystemExit("provide --session or --split; refusing an implicit full rebuild")
    methods = sorted(METHODS) if args.methods == "all" else [v.strip() for v in args.methods.split(",") if v.strip()]
    for session_id in sessions:
        paths = build_one(session_id, methods, force=args.force)
        print(f"{session_id}: " + ", ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
