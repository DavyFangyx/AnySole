#!/usr/bin/env python3
"""Tests for the cell-35 fault cleaning stage (R3 option A, raw data side).

Self-contained: run directly —

    python AnysoleWorkspace/tool/pressure_washer/test_clean_fault_cell.py

Gold numbers come from the R3 audit (Baselines/决策/05_R1-R3_风险裁定.md §四)
and the 2026-10-02 CSV-criterion verification:
  - protocol sessions hit by the CSV-side criterion: 52
  - override sessions (CV knife-edge): S13072, S14033, S6101 -> 55 zeroed
  - non-protocol recordings hit by the criterion: 48
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import clean_fault_cell as cfc  # noqa: E402

TREE = REPO / "AnysoleWorkspace/work/data_pipeline/pressure_washer/final_fake_marked"
_ARCHIVE = REPO / "AnysoleWorkspace/work/data_pipeline/pressure_washer/archive"
_PRE_CLEAN = sorted(_ARCHIVE.glob("final_fake_marked*")) if _ARCHIVE.is_dir() else []
# The gold criterion must be evaluated on pre-clean data (the live tree is cleaned).
GOLD_TREE = _PRE_CLEAN[-1] if _PRE_CLEAN else TREE
FLEET = {r["sid"]: r for r in __import__("json").load(open("/tmp/r3-scratch/fleet_scan.json"))}
MANIFEST_ROWS = [__import__("json").loads(line) for line in
                 (REPO / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl").read_text().splitlines() if line.strip()]
PROTOCOL_RECS = {row["pressure_path"].rstrip("/").split("/")[-1]: row["session_id"] for row in MANIFEST_ROWS}
OVERRIDE_RECS = cfc.override_rec_dirs()


def _crit_values(rec_dir: Path):
    right, _, _, keep_r = cfc.read_foot(rec_dir / "pressure_right.csv")
    left, _, _, keep_l = cfc.read_foot(rec_dir / "pressure_left.csv")
    empty = dict(max=np.nan, cv=np.nan, hit=False)
    if not keep_r.any():
        e_r = empty
    else:
        e_r = cfc.evaluate_criterion(right[keep_r][:, cfc.FAULT_CHANNEL], np.delete(right[keep_r], cfc.FAULT_CHANNEL, axis=1))
    if not keep_l.any():
        e_l = empty
    else:
        e_l = cfc.evaluate_criterion(left[keep_l][:, cfc.FAULT_CHANNEL], np.delete(left[keep_l], cfc.FAULT_CHANNEL, axis=1))
    return e_r, e_l


def test_criterion_units():
    # sustained rail spike -> hit; random noise -> no hit
    cell = np.array([100.0] * 60 + [5000.0] * 30 + [100.0] * 60)
    others = np.full((len(cell), 47), 120.0)
    assert cfc.evaluate_criterion(cell, others)["hit"] is True
    noisy = np.abs(np.random.default_rng(0).normal(1500, 900, size=len(cell)))
    assert cfc.evaluate_criterion(noisy, others)["hit"] is False
    assert cfc.longest_run(np.array([1, 1, 0, 1, 1, 1], dtype=bool)) == 3
    assert abs(cfc.cv_of(np.array([2.0, 2.0])) ) < 1e-12


def test_fleet_reproduction():
    """The criterion over the pre-clean tree must reproduce the audited hit sets."""
    hits, extras, flagged = [], [], []
    for rec_dir in sorted(GOLD_TREE.glob("*/S*/rec*/pressure_right.csv")):
        rec_dir = rec_dir.parent
        e_r, e_l = _crit_values(rec_dir)
        if e_r["hit"]:
            hits.append(rec_dir.name)
        sid = PROTOCOL_RECS.get(rec_dir.name)
        if sid is None:
            extras.append(rec_dir.name)
        if e_l["hit"]:
            flagged.append((rec_dir.name, "L"))
    protocol_hits = [r for r in hits if r in PROTOCOL_RECS]
    non_protocol_hits = [r for r in hits if r not in PROTOCOL_RECS]
    assert len(protocol_hits) == 52, f"protocol criterion hits {len(protocol_hits)} != 52: {sorted(protocol_hits)}"
    assert len(non_protocol_hits) == 48, f"non-protocol hits {len(non_protocol_hits)} != 48"
    assert not flagged, f"left-foot hits unexpected: {flagged}"
    for sid in ("S13072", "S14033", "S6101"):
        assert PROTOCOL_RECS_inverse(sid) in OVERRIDE_RECS  # override resolves to a rec dir
    # knife-edge check: the overrides do NOT hit the criterion (else they'd not need it)
    for rec in OVERRIDE_RECS:
        e_r, _ = _crit_values(next(GOLD_TREE.glob(f"*/S*/{rec}/pressure_right.csv")).parent)
        assert not e_r["hit"], f"{rec} hits criterion; override redundant"


def PROTOCOL_RECS_inverse(sid: str) -> str:
    return next(r for r, s in PROTOCOL_RECS.items() if s == sid)


def _make_synthetic_tree(root: Path):
    header = "frame_idx,t_us,valid_mask,fake," + ",".join(str(c) for c in range(1, 49))
    hit = np.array([120.0] * 100 + [6000.0] * 20 + [120.0] * 100)  # channel 34 rail
    normal = np.full(220, 150.0)
    def write(dirname: str, ch34: np.ndarray, left_ch34: np.ndarray | None = None):
        rec = root / "20260804" / "S9" / dirname
        rec.mkdir(parents=True, exist_ok=True)
        for side, ch in (("left", left_ch34 if left_ch34 is not None else normal),
                         ("right", ch34)):
            with (rec / f"pressure_{side}.csv").open("w", encoding="utf-8") as f:
                f.write(header + "\n")
                for i, v in enumerate(ch):
                    cols = ["150.0"] * 48
                    cols[34] = f"{v:.1f}"
                    f.write(f"{i},{i*25000},1,0," + ",".join(cols) + "\n")
    write("rec20260804_000000_test_S901_1", hit)
    write("rec20260804_000001_test_S902_1", normal)


def test_byte_level_behavior():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"; out = Path(td) / "out"
        _make_synthetic_tree(src)
        # synthetic tree has no manifest entries -> no overrides, both sides evaluated
        rc = __import__("subprocess").run(
            [sys.executable, str(HERE / "clean_fault_cell.py"), "-input", str(src),
             "--output-dir", str(out)],
            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        hit_src = src / "20260804/S9/rec20260804_000000_test_S901_1/pressure_right.csv"
        hit_out = out / "20260804/S9/rec20260804_000000_test_S901_1/pressure_right.csv"
        miss_src = src / "20260804/S9/rec20260804_000001_test_S902_1/pressure_right.csv"
        miss_out = out / "20260804/S9/rec20260804_000001_test_S902_1/pressure_right.csv"
        # hit file: identical except column '35' is all 0.0
        assert hit_src.read_bytes() != hit_out.read_bytes()
        src_lines = hit_src.read_text().splitlines()
        out_lines = hit_out.read_text().splitlines()
        assert len(src_lines) == len(out_lines)
        col = src_lines[0].split(",").index("35")
        for s, o in zip(src_lines[1:], out_lines[1:]):
            sf, of = s.split(","), o.split(",")
            sf[col], of[col] = "0.0", "0.0"
            assert sf == of, f"unexpected diff outside col 35: {sf} vs {of}"
            assert of[col] == "0.0"
        # miss file: byte-identical copy
        assert miss_src.read_bytes() == miss_out.read_bytes()
        # idempotency: second pass on the cleaned tree hits nothing (criterion max=0)
        out2 = Path(td) / "out2"
        rc2 = __import__("subprocess").run(
            [sys.executable, str(HERE / "clean_fault_cell.py"), "-input", str(out),
             "--output-dir", str(out2)],
            capture_output=True, text=True)
        assert rc2.returncode == 0
        assert (out / "20260804/S9/rec20260804_000000_test_S901_1/pressure_right.csv").read_bytes() == \
               (out2 / "20260804/S9/rec20260804_000000_test_S901_1/pressure_right.csv").read_bytes()


def test_manifest_written():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"; out = Path(td) / "out"
        _make_synthetic_tree(src)
        rc = __import__("subprocess").run(
            [sys.executable, str(HERE / "clean_fault_cell.py"), "-input", str(src),
             "--output-dir", str(out), "--manifest-only-path", str(out / "fault_cell_manifest.csv")],
            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        text = (out / "fault_cell_manifest.csv").read_text()
        assert "criterion (D7" in text and "zero_R34" in text and "S901" in text


if __name__ == "__main__":
    test_criterion_units()
    test_fleet_reproduction()
    test_byte_level_behavior()
    test_manifest_written()
    print("clean_fault_cell tests: 4/4 passed")
