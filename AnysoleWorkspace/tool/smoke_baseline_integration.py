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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402
MANIFEST = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
SPLITS = WORKSPACE / "protocol/splits/default/splits.csv"


def check_manifest() -> dict[str, int]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    manifest_ids = {row["session_id"] for row in rows}
    split_counts = {key: 0 for key in ("train", "val", "test")}
    with SPLITS.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    for row in split_rows:
        for key in split_counts:
            if (row.get(key) or "").strip():
                split_counts[key] += 1
    # 08 全局审阅：split 以用户生成的 canonical CSV 为准（当前 104/36/36），
    # 不再按固定 92/12/36 断言。
    for row in split_rows:
        for key in split_counts:
            value = (row.get(key) or "").strip()
            if value:
                assert value in manifest_ids, f"split session {value} not in manifest"
    return split_counts


def check_mmvp_artifacts(session: str) -> dict:
    manifest = {
        json.loads(line)["session_id"]: json.loads(line)
        for line in MANIFEST.read_text().splitlines() if line.strip()
    }
    row = manifest[session]
    recording = resolve_uri(row["video_path"], must_exist=True)
    date, subject = recording.parts[-3], recording.parts[-2]
    n = int(row["n_frames"])
    toolkit = WORKSPACE / "model_inputs/pressure_toolkit/v1/images" / date / subject / session
    mmvp = WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1" / date / subject / session
    # 00 §2.1：31×11 只保留一份公共表示；pressure_toolkit v1 的 insole
    # 副本必须与公共表示逐帧一致。
    tool_files = sorted((toolkit / "insole").glob("*.npy"))
    mmvp_files = sorted((mmvp / "insole").glob("*.npy"))
    assert len(tool_files) == len(mmvp_files) == n, (len(tool_files), len(mmvp_files), n)
    for tool_file, mmvp_file in zip(tool_files, mmvp_files):
        # 共享表示：裸 (2,31,11) float32；pressure_toolkit v1：上游 dict 包装
        # {'insole': [L, R], 'frame_id', 'source'}。数值必须逐位一致。
        tool = np.load(tool_file, allow_pickle=True).item()
        rep = np.load(mmvp_file)
        assert rep.shape == (2, 31, 11), rep.shape
        assert np.array_equal(np.asarray(tool["insole"][0]), rep[0])
        assert np.array_equal(np.asarray(tool["insole"][1]), rep[1])
    assert (toolkit / "calibration.npy").is_file()
    floor = WORKSPACE / "model_inputs/pressure_toolkit/v1/annotations" / date / "floor_info" / f"floor_{subject}.npy"
    assert floor.is_file()
    floor_data = np.load(floor, allow_pickle=True).item()
    transform = np.asarray(floor_data["depth2floor"])
    assert transform.shape == (4, 4)
    assert np.allclose(transform @ np.linalg.inv(transform), np.eye(4), atol=1e-5)
    join_report = WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1" / date / subject / session / "join_manifest.json"
    assert join_report.is_file()
    join_data = json.loads(join_report.read_text(encoding="utf-8"))
    return {"session": session, "frames": n, "insole_shape": [31, 11],
            "joined_frames": int(join_data.get("joined_count", 0)),
            "depth_ready": (toolkit / "depth").is_dir(),
            "keypoints_ready": (toolkit / "input" / subject / session / "keypoints").is_dir()}


def check_fpp_metadata() -> dict:
    path = WORKSPACE / "model_inputs/FPP-Net/adapter_v1/dataset_split_temporal5.npy"
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
    # 08 全局审阅：split 以 canonical CSV 为准，不再按 92/12/36 断言。
    return counts


def check_motionpro_cache(session: str) -> dict:
    motion_root = ROOT / "Baselines/MotionPRO"
    if str(motion_root) not in sys.path:
        sys.path.insert(0, str(motion_root))
    from lib.dataset.image_pressure import ImagePressureDataset  # noqa: E402

    seq_root = WORKSPACE / "shared/facts/sessions/cam3"
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
        "AnysoleWorkspace/tool/adapters/FPP-Net/build_inputs.py",
        "AnysoleWorkspace/tool/adapters/FPP-Net/build_metadata.py",
        "AnysoleWorkspace/tool/adapters/FPP-Net/run_rtmpose.py",
        "AnysoleWorkspace/tool/adapters/PoseTransOpt/build_inputs.py",
        "AnysoleWorkspace/tool/adapters/PoseTransOpt/run_cliff.py",
        "AnysoleWorkspace/tool/adapters/pressure_toolkit/build_inputs.py",
        "AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py",
        "AnysoleWorkspace/tool/export_baseline_motion.py",
        "Baselines/pressure_tookit/main_singleview.py",
        "Baselines/pressure_tookit/lib/dataextra/data_loader.py",
        "Baselines/VP-MoCap/FPP-Net/app/infer_smplcont.py",
        "Baselines/VP-MoCap/FPP-Net/lib/config/config.py",
        "Baselines/VP-MoCap/PoseTransOpt/app/optimize.py",
        "Baselines/VP-MoCap/PoseTransOpt/lib/initial_trans/initial_trans.py",
        "Baselines/Step2Motion/src/test.py",
        "results_display/script/r_test1_visualize_mmvp.py",
    ]
    for value in paths:
        path = ROOT / value
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    # S10101 同时具备 pressure_toolkit v1 输入与 PoseTransOpt adapter 产物，
    # 是当前磁盘上可完整跑通 mmvp 一致性检查的代表 session。
    parser.add_argument("--session", default="S10101")
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
