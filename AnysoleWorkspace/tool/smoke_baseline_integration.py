#!/usr/bin/env python3
"""Read-only smoke tests for the unified baseline integration layer."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
MANIFEST = WORKSPACE / "manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "splits/default/splits.csv"


def check_manifest() -> dict[str, int]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    split_counts = {key: 0 for key in ("train", "val", "test")}
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    for row in split_rows:
        for key in split_counts:
            if (row.get(key) or "").strip():
                split_counts[key] += 1
    assert len(rows) == 144, len(rows)
    assert split_counts == {"train": 92, "val": 12, "test": 36}, split_counts
    return split_counts


def check_mmvp_artifacts(session: str) -> dict:
    manifest = {
        json.loads(line)["session_id"]: json.loads(line)
        for line in MANIFEST.read_text().splitlines() if line.strip()
    }
    row = manifest[session]
    parts = Path(row["pressure_path"]).parts
    cam = parts.index("cam3")
    date, subject = parts[cam + 1], parts[cam + 2]
    n = int(row["n_frames"])
    toolkit = WORKSPACE / "derived/pressure_tookit/images" / date / subject / session
    fpp = WORKSPACE / "derived/VP-MoCap" / date / subject / session
    tool_files = sorted((toolkit / "insole").glob("*.npy"))
    fpp_files = sorted((fpp / "insole").glob("*.npy"))
    assert len(tool_files) == len(fpp_files) == n, (len(tool_files), len(fpp_files), n)
    first_tool = np.load(tool_files[0], allow_pickle=True).item()
    first_fpp = np.load(fpp_files[0], allow_pickle=True).item()
    assert np.array_equal(first_tool["insole"][0], first_fpp["insole"][0])
    assert np.array_equal(first_tool["insole"][1], first_fpp["insole"][1])
    assert np.asarray(first_tool["insole"][0]).shape == (31, 11)
    assert np.asarray(first_tool["insole"][1]).shape == (31, 11)
    assert (toolkit / "calibration.npy").is_file()
    floor = WORKSPACE / "derived/pressure_tookit/annotations" / date / "floor_info" / f"floor_{subject}.npy"
    assert floor.is_file()
    floor_data = np.load(floor, allow_pickle=True).item()
    transform = np.asarray(floor_data["depth2floor"])
    assert transform.shape == (4, 4)
    assert np.allclose(transform @ np.linalg.inv(transform), np.eye(4), atol=1e-5)
    adapter_report = fpp / "adapter_manifest.json"
    assert adapter_report.is_file()
    return {"session": session, "frames": n, "insole_shape": [31, 11],
            "depth_ready": bool(json.loads(adapter_report.read_text())["depth"]),
            "keypoints_ready": bool(json.loads(adapter_report.read_text())["keypoints_fpp"])}


def check_fpp_metadata() -> dict:
    path = WORKSPACE / "derived/VP-MoCap/dataset_split_temporal5.npy"
    assert path.is_file()
    data = np.load(path, allow_pickle=True).item()
    assert set(data) == {"train", "val", "test"}
    counts = {phase: 0 for phase in data}
    for phase, dates in data.items():
        for subjects in dates.values():
            for sequences in subjects.values():
                counts[phase] += len(sequences)
                for frames in sequences.values():
                    # An assigned session may be retained with zero windows
                    # when every frame is fake; it is still part of the
                    # canonical audit population.
                    if not frames:
                        continue
                    assert all(str(frame).endswith(".npy") for frame in frames)
    assert counts == {"train": 92, "val": 12, "test": 36}, counts
    return counts


def check_motionpro_cache(session: str) -> dict:
    motion_root = ROOT / "Baselines/MotionPRO"
    if str(motion_root) not in sys.path:
        sys.path.insert(0, str(motion_root))
    from lib.dataset.image_pressure import ImagePressureDataset  # noqa: E402

    seq_root = WORKSPACE / "derived/MotionPRO/sequences/cam3"
    cfg = {"task": {
        "window_length": 20, "cam_id": 3,
        "split_csv": str(SPLITS), "seq_root": str(seq_root),
        "contact_method": "tactile_abs",
    }}
    dataset = ImagePressureDataset(cfg, "test")
    assert len(dataset) > 0
    item = dataset[0]
    assert tuple(item["pressure"].shape[-2:]) == (96, 96)
    assert item["pressure"].shape[0] == 20
    return {"windows": len(dataset), "pressure_shape": list(item["pressure"].shape)}


def compile_sources() -> int:
    paths = [
        "AnysoleWorkspace/tool/build_fpp_metadata.py",
        "AnysoleWorkspace/tool/prepare_mmvp_observations.py",
        "AnysoleWorkspace/tool/run_rtmpose_halpe26.py",
        "AnysoleWorkspace/tool/run_cliff_mmvp.py",
        "AnysoleWorkspace/tool/smoke_baseline_integration.py",
        "Baselines/pressure_tookit/main_singleview.py",
        "Baselines/pressure_tookit/lib/dataextra/data_loader.py",
        "Baselines/VP-MoCap/FPP-Net/app/infer_smplcont.py",
        "Baselines/VP-MoCap/FPP-Net/lib/config/config.py",
        "Baselines/VP-MoCap/PoseTransOpt/app/optimize.py",
        "Baselines/VP-MoCap/PoseTransOpt/lib/initial_trans/initial_trans.py",
        "Baselines/Step2Motion/src/process_gait.py",
        "Baselines/Step2Motion/src/test.py",
        "results_display/script/r_test1_visualize_mmvp.py",
    ]
    for value in paths:
        path = ROOT / value
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="S13013")
    parser.add_argument("--motionpro", action="store_true")
    args = parser.parse_args()
    result = {
        "split": check_manifest(),
        "mmvp": check_mmvp_artifacts(args.session),
        "fpp": check_fpp_metadata(),
        "compiled": compile_sources(),
    }
    if args.motionpro:
        result["motionpro"] = check_motionpro_cache(args.session)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
