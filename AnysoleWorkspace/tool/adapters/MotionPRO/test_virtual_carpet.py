#!/usr/bin/env python
"""T1-A V3 acceptance test: virtual carpet -> native contact judgment vs motion_f6.

Native semantics mirrored from ``Baselines/MotionPRO/lib/util/gen_contact.py``
(read-only reference; this file re-implements it locally for the extended 4m
long axis, as ruled in z_note/重构执行/11_T系列_运行时输入与接入方案.md §4.3):

  * ``carpet_pos`` is (120, 320, 3) on the native 1.25cm pitch
    (``carpet_pos[i, j, 0] = i*scale + 0.5*scale``, ``carpet_pos[i, j, 1] =
    -j*scale - 0.5*scale``, ``z = 0``) -- identical to
    ``adapter.carpet_grid_x()/carpet_grid_y()``; only the long axis is doubled
    (160 -> 320 cells, ``scale = 4.0/320 == 2.0/160``).
  * per frame and query joint: nearest carpet grid point -> ``RADIUS = 3``
    window -> sum of pressure in the window -> ``contact = sum > 100``.
  * window slicing is the native one, ``[q - 3 : q + 3]`` (six pixels, not
    seven).  The native code lets negative indices wrap around the array; here
    they are clipped to the array bounds (the high-side behaviour is the
    native one already).

Two documented deviations from a literal port:

  1. The native script 180-degree flips the pressure image before querying
     (``cv2.flip(pressure_, 1)`` + ``flip(..., 0)``) because its stored image is
     mirrored relative to ``carpet_pos``.  The virtual carpet is authored
     directly in the ``carpet_pos`` convention (grid point (i, j) is painted at
     image ``[j, i]``, 11 号文档 §4.1), so the flip is a no-op by construction
     and is not applied.  ``--flip-native`` re-applies it as a diagnostic.
  2. The native queries SMPL joints 7/8/10/11 (ankles + feet).  The BVH-23
     source has LeftFoot/RightFoot (ankles) and LeftToeBase/RightToeBase
     (toes), so the toe joint stands in for the SMPL foot-ball joint.  Column
     order is the native one: [LA, LF, RA, RF].

Ground truth = ``contact when motion_f6 state == 1`` (the semantic the C1 audit
used: ``adapter._f6_pipeline`` states, 1 = contact).

Acceptance gate: per-column agreement on frames whose query joint is inside the
carpet must be >= 0.8.  Out-of-carpet frames carry no tactile signal by the V1
ruling (fixed world frame, no invalid flag), so they are reported separately and
are not part of the gate.

Measured on S5091 (543 frames): the gate is NOT met, and it is not reachable with
the ruled rule.  Reported per column: accuracy, the degenerate always-contact
baseline it must beat, and ``ceiling`` -- the best accuracy obtainable from the
same window sums at ANY threshold (the rule's structural ceiling, since the only
free parameter of the rule is the threshold).  With the ruled mapping (toe tip
for LF/RF) the numbers are 0.716 / 0.664 / 0.800 / 0.707 (LA/LF/RA/RF) and the
ceilings 0.881 / 0.714 / 0.846 / 0.787 -- the two toe columns cannot reach 0.8
at any threshold.

The shortfall was investigated as instructed and is not a painting or
orientation bug: a sweep over +-10cm of footprint placement and both column
orders never reaches 0.8 at the native threshold, and re-applying the native
180-degree image flip (``--flip-native``) collapses the rule to "never contact"
(accuracy 0.34-0.41) -- i.e. the carpet and the query are already in the same
convention.  The binding factor is the +/-3 px (7.5cm) window centred on the
ankle/toe joint: it lands on the insole's unloaded arch/midfoot (48-cell insole,
frozen contract: column 0 = toe), whose channels are exactly 0 in most frames,
and the native's fixed ``sum > 100`` threshold discards the fringe-only windows
that carry the remaining signal.

``--foot-query ball`` repeats the run with the native-faithful query joint
(SMPL 10/11 are the metatarsal heads, not the toe tips; the BVH-23 skeleton has
no ball joint, so it is interpolated ankle->toe): LF rises 0.664 -> 0.873 and RF
0.707 -> 0.721 while the ankle columns are unchanged, which confirms the
diagnosis.  ``--report-only`` prints the table and exits 1 instead of raising, so
the numbers can be archived.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AnysoleWorkspace.tool.adapters.MotionPRO import adapter as A  # noqa: E402

RADIUS = 3
SUM_THRESHOLD = 100.0
COLUMNS = ("LA", "LF", "RA", "RF")
SIDES = ("left", "left", "right", "right")


def carpet_positions() -> np.ndarray:
    """(120, 320, 3) native carpet_pos for the extended 4m carpet."""
    x = A.carpet_grid_x()
    y = A.carpet_grid_y()
    grid = np.zeros((len(x), len(y), 3), dtype=np.float64)
    grid[:, :, 0] = x[:, None]
    grid[:, :, 1] = y[None, :]
    return grid


def nearest_grid_index(x: float, y: float) -> tuple[int, int]:
    """Nearest (i, j) carpet cell of a plane point, clamped to the carpet.

    For a regular lattice the Euclidean-nearest point is the componentwise
    nearest one, which is what ``compute_point_with_min_distance`` returns; the
    clamp reproduces the native argmin behaviour for points off the carpet.
    """
    i = int(round((float(x) - 0.5 * A.CARPET_CELL_M) / A.CARPET_CELL_M))
    j = int(round((-float(y) - 0.5 * A.CARPET_CELL_M) / A.CARPET_CELL_M))
    i = min(max(i, 0), A.CARPET_HW[1] - 1)
    j = min(max(j, 0), A.CARPET_HW[0] - 1)
    return i, j


def nearest_grid_index_bruteforce(position: np.ndarray, grid: np.ndarray) -> tuple[int, int]:
    """Literal ``compute_point_with_min_distance`` port (slow; parity check only)."""
    distances = np.linalg.norm(grid - position, axis=2)
    return tuple(int(v) for v in np.unravel_index(np.argmin(distances), distances.shape))


def native_contact(pressure: np.ndarray, query_xy: np.ndarray, *,
                   flip: bool = False) -> np.ndarray:
    """(T,4) native contact flags for query points (T,4,2) in carpet (x, y)."""
    frames, joints = query_xy.shape[0], query_xy.shape[1]
    contact = np.zeros((frames, joints), dtype=np.int8)
    height, width = pressure.shape[1], pressure.shape[2]
    for frame in range(frames):
        image = pressure[frame]
        if flip:
            image = image[::-1, ::-1]
        for joint in range(joints):
            i, j = nearest_grid_index(*query_xy[frame, joint])
            x_lo, x_hi = max(i - RADIUS, 0), min(i + RADIUS, width)
            y_lo, y_hi = max(j - RADIUS, 0), min(j + RADIUS, height)
            if float(image[y_lo:y_hi, x_lo:x_hi].sum()) > SUM_THRESHOLD:
                contact[frame, joint] = 1
    return contact


def query_points(ctx: dict, *, foot_query: str = "toe",
                 ball_blend: float = 0.75) -> tuple[np.ndarray, dict]:
    """(T,4,2) [LA, LF, RA, RF] query points + per-column joint names.

    ``foot_query = "toe"`` (default, as implemented): the BVH toe tip stands in
    for the native's SMPL foot joint.  ``"ball"``: the native queries SMPL joints
    10/11, which sit at the metatarsal heads, not at the toe tip; the BVH-23
    skeleton has no ball joint, so it is approximated by interpolating
    ``ball_blend`` of the way from the ankle to the toe tip (a diagnostic --
    see the module docstring on why the default was left at the toe tip).
    """
    tracks = A.foot_tracks(ctx)
    points = np.empty((len(tracks["left"]["center"]), 4, 2), dtype=np.float64)
    for index, (side, joint) in enumerate((("left", "center"), ("left", "toe"),
                                           ("right", "center"), ("right", "toe"))):
        if joint == "center" or foot_query == "toe":
            points[:, index] = tracks[side][joint]
        else:
            points[:, index] = (tracks[side]["center"] * (1.0 - ball_blend)
                                + tracks[side]["toe"] * ball_blend)
    names = {side: tracks[side]["joints"] for side in tracks}
    return points, names


def in_carpet_mask(points: np.ndarray) -> np.ndarray:
    """(T,4) True where the query point lies inside the carpet rectangle."""
    x_axis, y_axis = A.carpet_grid_x(), A.carpet_grid_y()
    inside_x = (points[:, :, 0] >= x_axis[0]) & (points[:, :, 0] <= x_axis[-1])
    inside_y = (points[:, :, 1] >= y_axis[-1]) & (points[:, :, 1] <= y_axis[0])
    return inside_x & inside_y


def window_sums(pressure: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """(T,) raw sum of the native +/-RADIUS px window around each query point."""
    height, width = pressure.shape[1], pressure.shape[2]
    sums = np.zeros(query_xy.shape[0], dtype=np.float64)
    for frame in range(query_xy.shape[0]):
        col, row = nearest_grid_index(float(query_xy[frame, 0]), float(query_xy[frame, 1]))
        y0, y1 = max(row - RADIUS, 0), max(row + RADIUS, 0)
        x0, x1 = max(col - RADIUS, 0), max(col + RADIUS, 0)
        y0, y1 = min(y0, height), min(y1, height)
        x0, x1 = min(x0, width), min(x1, width)
        sums[frame] = float(pressure[frame, y0:y1, x0:x1].sum())
    return sums


def threshold_ceiling(sums: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Best achievable accuracy over any sum threshold (the rule's structural ceiling).

    The native rule fires when the window sum exceeds a fixed threshold; sweeping
    that threshold shows how much of the disagreement is caused by the threshold
    value versus by the window carrying no contact information at all.
    """
    if not mask.any():
        return float("nan"), float("nan")
    candidates = np.unique(np.concatenate([np.quantile(sums[mask], np.linspace(0.0, 1.0, 41)),
                                           [0.0, 100.0, SUM_THRESHOLD]]))
    best_accuracy, best_threshold = -1.0, float("nan")
    for threshold in candidates:
        accuracy = float(((sums[mask] > threshold) == truth[mask]).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, float(threshold)
    return best_accuracy, best_threshold


def evaluate(pressure: np.ndarray, points: np.ndarray, truth: np.ndarray,
             *, flip: bool = False) -> tuple[np.ndarray, list[dict], dict]:
    """Contact flags + per-column metrics table + aggregate summary."""
    contact = native_contact(pressure, points, flip=flip)
    inside = in_carpet_mask(points)
    rows = []
    for index, column in enumerate(COLUMNS):
        side = {"LA": "left", "LF": "left", "RA": "right", "RF": "right"}[column]
        predicted = contact[:, index]
        expected = truth[:, 0 if side == "left" else 1]
        mask = inside[:, index]
        sums = window_sums(pressure, points[:, index])
        ceiling, ceiling_threshold = threshold_ceiling(sums, expected, mask)
        correct = predicted == expected
        rows.append({
            "column": column,
            "bvh_joint": None,  # filled by caller
            "gt": f"motion_f6 {side} (state==1)",
            "n_frames": int(len(predicted)),
            "n_in_carpet": int(mask.sum()),
            "accuracy_all": float(correct.mean()),
            "accuracy_in_carpet": float(correct[mask].mean()) if mask.any() else None,
            "pred_contact_rate": float(predicted.mean()),
            "pred_contact_rate_in_carpet": float(predicted[mask].mean()) if mask.any() else None,
            "gt_contact_rate": float(expected.mean()),
            "gt_contact_rate_in_carpet": float(expected[mask].mean()) if mask.any() else None,
            "always_contact_baseline_in_carpet": float(expected[mask].mean()) if mask.any() else None,
            "ceiling_accuracy_in_carpet": ceiling if np.isfinite(ceiling) else None,
            "ceiling_threshold": ceiling_threshold if np.isfinite(ceiling_threshold) else None,
            "confusion_in_carpet": {
                "tp": int((predicted & (expected == 1))[mask].sum()),
                "fp": int((predicted & (expected == 0))[mask].sum()),
                "fn": int(((~predicted) & (expected == 1))[mask].sum()),
                "tn": int(((~predicted) & (expected == 0))[mask].sum()),
            } if mask.any() else None,
        })
    left_margin = np.maximum(contact[:, 0] == truth[:, 0], contact[:, 1] == truth[:, 0])
    right_margin = np.maximum(contact[:, 2] == truth[:, 1], contact[:, 3] == truth[:, 1])
    left_inside = inside[:, 0] | inside[:, 1]
    right_inside = inside[:, 2] | inside[:, 3]
    summary = {
        "per_side_best_of_ankle_toe": {
            "left": {
                "n_in_carpet_frames": int(left_inside.sum()),
                "accuracy_all": float(left_margin.mean()),
                "accuracy_in_carpet": float(left_margin[left_inside].mean()) if left_inside.any() else None,
            },
            "right": {
                "n_in_carpet_frames": int(right_inside.sum()),
                "accuracy_all": float(right_margin.mean()),
                "accuracy_in_carpet": float(right_margin[right_inside].mean()) if right_inside.any() else None,
            },
        },
        "degenerate_baselines": {
            "always_contact_all": float((np.ones_like(truth) == truth).mean()),
            "never_contact_all": float((np.zeros_like(truth) == truth).mean()),
        },
    }
    return contact, rows, summary


def cop_layout_check(ctx: dict, truth: np.ndarray, head: int = 5) -> dict:
    """Independent check of the frozen "column 0 == toe" cell contract.

    For every stance run (f6 state 1) the centre of pressure along the insole
    length axis is averaged over the first and last few frames: a heel strike
    loads high column indices and toe-off loads low ones only if column 0 is the
    toe (assets/foot_sensor_layout/README.md, verified numerically).
    """
    out = {}
    for index, side in enumerate(("left", "right")):
        cells = np.clip(np.asarray(ctx["left48" if side == "left" else "right48"]), 0.0, None)
        grid = cells.reshape(-1, A.FOOT_ROWS, A.FOOT_COLS)
        state = truth[:, index]
        columns = np.arange(A.FOOT_COLS, dtype=np.float64)
        starts, ends = [], []
        run = 0
        for frame in range(len(state)):
            if state[frame] == 1:
                run += 1
                if 1 <= run <= head:
                    total = grid[frame].sum(axis=0)
                    if total.sum() > 0:
                        starts.append(float((columns * total).sum() / total.sum()))
            elif run:
                for back in range(1, min(head, run) + 1):
                    total = grid[frame - back].sum(axis=0)
                    if total.sum() > 0:
                        ends.append(float((columns * total).sum() / total.sum()))
                run = 0
        out[side] = {
            "mean_cop_column_first_frames_of_stance": float(np.mean(starts)) if starts else None,
            "mean_cop_column_last_frames_of_stance": float(np.mean(ends)) if ends else None,
            "n_stance_runs": len(ends),
            "interpretation": "heel strike loads high columns and toe-off low ones "
                              "only when column 0 is the toe (frozen contract)",
        }
    return out


def table(rows: list[dict]) -> str:
    header = (f"{'col':<4}{'joint':<14}{'n_in':>6}{'acc_all':>9}{'acc_in':>8}"
              f"{'pred_in':>9}{'gt_in':>8}{'ceil_in':>9}{'ceil_thr':>10}")
    lines = [header, "-" * len(header)]
    for row in rows:
        acc_in = f"{row['accuracy_in_carpet']:.3f}" if row["accuracy_in_carpet"] is not None else "n/a"
        pred_in = f"{row['pred_contact_rate_in_carpet']:.3f}" if row["pred_contact_rate_in_carpet"] is not None else "n/a"
        gt_in = f"{row['gt_contact_rate_in_carpet']:.3f}" if row["gt_contact_rate_in_carpet"] is not None else "n/a"
        ceiling = (f"{row['ceiling_accuracy_in_carpet']:.3f}"
                   if row["ceiling_accuracy_in_carpet"] is not None else "n/a")
        ceiling_threshold = (f"{row['ceiling_threshold']:.0f}"
                             if row["ceiling_threshold"] is not None else "n/a")
        lines.append(f"{row['column']:<4}{row['bvh_joint']:<14}{row['n_in_carpet']:>6}"
                     f"{row['accuracy_all']:>9.3f}{acc_in:>8}{pred_in:>9}{gt_in:>8}"
                     f"{ceiling:>9}{ceiling_threshold:>10}")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="T1-A V3 virtual carpet acceptance test")
    parser.add_argument("--session", default="S5091")
    parser.add_argument("--pressure", type=Path, default=None,
                        help="override pressure.npz (default: the session's model input)")
    parser.add_argument("--anchor", choices=("fixed", "centred"), default="fixed",
                        help="fixed = the V1 ruling; centred = diagnostic repaint with the "
                             "walk centred on the carpet (sensitivity analysis only)")
    parser.add_argument("--flip-native", action="store_true",
                        help="diagnostic: apply the native gen_contact 180-degree image flip")
    parser.add_argument("--parity-check", action="store_true", default=True,
                        help="assert the closed-form nearest-cell index matches the native "
                             "brute-force argmin (default on)")
    parser.add_argument("--no-cop-check", action="store_true")
    parser.add_argument("--foot-query", choices=("toe", "ball"), default="toe",
                        help="LF/RF query joint: 'toe' = BVH toe tip (as implemented, "
                             "the native's SMPL foot joint is actually the ball); "
                             "'ball' = ankle->toe interpolation diagnostic")
    parser.add_argument("--ball-blend", type=float, default=0.75,
                        help="ankle->toe interpolation weight for --foot-query ball")
    parser.add_argument("--gate", type=float, default=0.8,
                        help="per-column in-carpet agreement gate (T1-A spec: 0.8)")
    parser.add_argument("--report-only", action="store_true",
                        help="print the table and exit 0 even when the gate fails "
                             "(used to record the structural ceiling)")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    session = A.load_shared_session(args.session)
    ctx = A._context(session)
    left48 = np.asarray(session["pressure"]["left48"], dtype=np.float32)
    right48 = np.asarray(session["pressure"]["right48"], dtype=np.float32)
    truth_states, _ = A._f6_pipeline(ctx)
    truth = np.asarray(truth_states, dtype=np.int8)

    tracks = A.foot_tracks(ctx)
    points, joint_names = query_points(ctx, foot_query=args.foot_query,
                                       ball_blend=args.ball_blend)
    if args.parity_check:
        grid = carpet_positions()
        for frame in range(0, len(points), max(1, len(points) // 5)):
            for joint in range(4):
                position = np.array([points[frame, joint, 0], points[frame, joint, 1], 0.0])
                closed = nearest_grid_index(*points[frame, joint])
                brute = nearest_grid_index_bruteforce(position, grid)
                if closed != brute:
                    raise AssertionError(
                        f"nearest-cell parity failed at frame {frame} joint {joint}: "
                        f"closed={closed} brute={brute}")
        print("nearest-cell parity vs native argmin: OK")

    pressure_file = args.pressure or (A.model_input_dir(args.session) / "pressure.npz")
    produced, stats = A.virtual_carpet(ctx, left48, right48)
    if args.anchor == "centred":
        centroid = np.mean([tracks[side]["center"].mean(axis=0) for side in ("left", "right")], axis=0)
        anchor = (float(np.mean(A.carpet_grid_x()) - centroid[0]),
                  float(np.mean(A.carpet_grid_y()) - centroid[1]))
        produced, stats = A.virtual_carpet(ctx, left48, right48, anchor=anchor)
        # the diagnostic carpets are painted at points + anchor, so the query
        # points must move with them
        points = points + np.asarray(anchor, dtype=np.float64)
        print(f"diagnostic anchor (world shift): ({anchor[0]:+.4f}, {anchor[1]:+.4f}) m")
    else:
        anchor = (0.0, 0.0)

    on_disk = np.load(pressure_file)["pressure"]
    print(f"pressure file : {pressure_file}")
    print(f"on disk       : {on_disk.shape} {on_disk.dtype} "
          f"range [{float(on_disk.min()):.1f}, {float(on_disk.max()):.1f}]")
    same = np.array_equal(on_disk, produced)
    print(f"reproduced    : {produced.shape} identical={same}"
          f"{'' if args.anchor == 'fixed' else ' (centred anchor: expected to differ)'}")
    if args.anchor == "fixed" and not same:
        raise AssertionError("pressure.npz does not match a fresh virtual_carpet run "
                             "(re-run the adapter with --force)")
    print(f"painted px    : {stats['painted_px']} "
          f"(left {stats['per_side_painted']['left']}, right {stats['per_side_painted']['right']})")
    print(f"frames any paint: {stats['frames_any_paint']}/{produced.shape[0]} "
          f"(out of carpet: {stats['frames_out_of_bounds']})")
    print(f"in-carpet frames per foot (centre inside): "
          f"{stats['per_side_in_bounds_frames']}")

    contact, rows, summary = evaluate(produced, points, truth, flip=args.flip_native)
    for row, column in zip(rows, COLUMNS):
        side = {"LA": "left", "LF": "left", "RA": "right", "RF": "right"}[column]
        joint = "foot" if column in ("LA", "RA") else "toe"
        row["bvh_joint"] = joint_names[side][joint]
        if column in ("LF", "RF") and args.foot_query == "ball":
            row["bvh_joint"] = f"ball{args.ball_blend:.2f}"
    print()
    print(f"native +/-{RADIUS}px, sum > {SUM_THRESHOLD:g} contact vs motion_f6 "
          f"(session {args.session}, {produced.shape[0]} frames)")
    print(table(rows))
    print()
    for side, values in summary["per_side_best_of_ankle_toe"].items():
        print(f"best-of-ankle/toe {side:<5} accuracy all {values['accuracy_all']:.3f} "
              f"/ in-carpet {values['accuracy_in_carpet']:.3f} "
              f"(n={values['n_in_carpet_frames']})")
    print(f"degenerate always-contact {summary['degenerate_baselines']['always_contact_all']:.3f} "
          f"/ never-contact {summary['degenerate_baselines']['never_contact_all']:.3f}")

    cop = None
    if not args.no_cop_check:
        cop = cop_layout_check(ctx, truth)
        print()
        for side, values in cop.items():
            print(f"CoP column {side:<5}: first stance frames "
                  f"{values['mean_cop_column_first_frames_of_stance']:.2f} -> last frames "
                  f"{values['mean_cop_column_last_frames_of_stance']:.2f} "
                  f"({values['n_stance_runs']} stance runs)")

    gate_failures = [row["column"] for row in rows
                     if row["accuracy_in_carpet"] is None
                     or row["accuracy_in_carpet"] < args.gate]
    report = {
        "session": args.session,
        "anchor": list(anchor),
        "anchor_mode": args.anchor,
        "flip_native": bool(args.flip_native),
        "foot_query": {"mode": args.foot_query, "ball_blend": args.ball_blend,
                       "note": "LF/RF column joint mapping; 'toe' = the implemented "
                               "ruling, 'ball' = native-faithful diagnostic"},
        "contact_rule": {"radius_px": RADIUS, "sum_threshold": SUM_THRESHOLD,
                         "window": "[q-3:q+3] per axis, clipped to the array bounds",
                         "native_reference": "Baselines/MotionPRO/lib/util/gen_contact.py"},
        "carpet": {
            "shape": list(A.CARPET_HW),
            "cell_m": A.CARPET_CELL_M,
            "long_axis_m": A.CARPET_LONG_M,
            "wide_axis_m": A.CARPET_WIDE_M,
            "anchor": "fixed world frame (V1 ruling)" if args.anchor == "fixed" else "diagnostic centred",
            "painted_px": stats["painted_px"],
            "frames_any_paint": stats["frames_any_paint"],
            "frames_out_of_bounds": stats["frames_out_of_bounds"],
            "per_side_in_bounds_frames": stats["per_side_in_bounds_frames"],
            "joints": stats["joints"],
        },
        "rows": rows,
        "summary": summary,
        "cop_layout_check": cop,
        "gate": {"threshold": args.gate, "metric": "per-column accuracy on in-carpet frames",
                 "failed_columns": gate_failures, "passed": not gate_failures,
                 "ceiling_per_column": {row["column"]: row["ceiling_accuracy_in_carpet"]
                                        for row in rows}},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.out}")
    if gate_failures:
        message = (f"V3 gate failed (< {args.gate:g} in-carpet agreement): {gate_failures}; "
                   f"threshold-free ceilings "
                   f"{ {row['column']: round(row['ceiling_accuracy_in_carpet'], 3) for row in rows} }")
        if args.report_only:
            print(f"\nV3 gate: FAIL - {message}")
            return 1
        raise AssertionError(message)
    print(f"\nV3 gate: PASS (per-column in-carpet agreement >= {args.gate:g} for {list(COLUMNS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
