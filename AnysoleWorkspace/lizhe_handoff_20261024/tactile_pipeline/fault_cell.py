"""Fault-cell burst repair (v3, 2026-10-20): runs on raw rows, BEFORE
analyze/process/baseline, so bursts can never enter the airborne baseline.

Background (GT足底×触觉冲突审计_20261020.md §2.3 / §F): right-foot fault
cells 35 and 38 (1-based) emit intermittent bursts (4096..18916 g) stacked on
top of the normal signal (healthy segment p50 ~117 g), in runs of 1..20+
rows. v2 fed those bursts into the airborne-baseline percentiles, which
clipped the whole normal segment of the cell to 0. Repair must therefore
happen directly on the raw rows, before any baseline estimation.

Criterion (per row, per cell):
    value > 4095 AND p99(其余47格同帧) < 4095/3  ->  burst
The p99 guard excludes the single whole-row byte-corruption frame (several
cells >4095 at once raise the p99 of the rest above the guard) while flagging
single-cell bursts in an otherwise quiet row. The rule is value-based and
side-agnostic: it is applied identically to both feet (no channel list is
hardcoded — the previous washer hardcoded one channel and missed the other).

Run extension: a burst row whose neighbors are themselves elevated (cross-
talk to adjacent cells, or the burst landing on a loaded stance row) can
raise the row's p99 above the guard even though the cell is clearly inside
a burst run. Such rows are flagged as bursts too when an immediately
adjacent row of the SAME cell is already flagged and the cell is still
>4095; iterated to a fixed point this closes crosstalk/stance gaps inside
runs. It never touches the byte-corruption frame (no flagged neighbor of
the same cell) and only ever flags cells already >4095.

Repair: replace the burst value with the median of that cell's non-burst
rows inside the +-10 row window (edge-clamped); when the window holds fewer
than 5 non-burst rows, fall back to the whole-file median of that cell's
non-burst rows. Values are never zeroed: the median preserves the normal
signal underneath the burst (including a loaded stance level when the burst
sits on one).

Record: row0 / channel (1-based) / before / after for every repaired value.
"""
import numpy as np

BURST_THRESHOLD = 4095.0
GUARD_THRESHOLD = BURST_THRESHOLD / 3.0  # p99 of the other 47 cells
WINDOW = 10
MIN_WINDOW_SAMPLES = 5


def find_bursts(values):
    """Boolean mask of burst cells (rows x 48), primary criterion + run extension."""
    values = np.asarray(values, dtype=float)
    n, nc = values.shape
    mask = np.zeros(values.shape, bool)
    for c in range(nc):
        hot = np.flatnonzero(values[:, c] > BURST_THRESHOLD)
        for i in hot:
            if np.quantile(np.delete(values[i], c), 0.99) < GUARD_THRESHOLD:
                mask[i, c] = True
    # Close gaps inside burst runs (crosstalk / loaded stance rows).
    while True:
        added = 0
        for c in range(nc):
            hot = np.flatnonzero(values[:, c] > BURST_THRESHOLD)
            for i in hot:
                if mask[i, c]:
                    continue
                if (i > 0 and mask[i - 1, c]) or (i + 1 < n and mask[i + 1, c]):
                    mask[i, c] = True
                    added += 1
        if not added:
            break
    return mask


def repair(values):
    """Return (repaired, burst_mask, (row0, channel, before, after)).

    A cell whose every burst-flagged row leaves no non-burst reference in
    the whole file (fully stuck channel) is left untouched: the median
    repair semantics has no normal-signal reference and no value is
    invented for it.
    """
    values = np.asarray(values, dtype=float)
    mask = find_bursts(values)
    repaired = values.copy()
    n, nc = values.shape
    rows, cells, before, after = [], [], [], []
    for c in range(nc):
        burst_rows = np.flatnonzero(mask[:, c])
        if not len(burst_rows):
            continue
        good = ~mask[:, c]
        if not good.any():
            continue  # fully stuck channel: no non-burst reference exists
        file_median = np.median(values[good, c])
        for i in burst_rows:
            lo = max(0, i - WINDOW)
            hi = min(n, i + WINDOW + 1)
            window = values[lo:hi, c][good[lo:hi]]
            fill = np.median(window) if len(window) >= MIN_WINDOW_SAMPLES else file_median
            if not np.isfinite(fill):
                raise ValueError(
                    "no non-burst reference rows for cell %d (row %d)" % (c + 1, i))
            rows.append(i)
            cells.append(c + 1)
            before.append(float(values[i, c]))
            after.append(float(fill))
            repaired[i, c] = fill
    return (repaired, mask,
            (np.asarray(rows, dtype=np.int64), np.asarray(cells, dtype=np.int64),
             np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)))
