#!/usr/bin/env python3
"""Step2Motion adapter_v3: identical to adapter_v2 except the IMU block.

The 50-dim insole layout, pressure pooling, clip policy, pose targets and
normalizer contract are unchanged (``build_gait.py`` stays untouched).  Only
the acc/gyro channels are read from the virtual instep IMU facts produced by
``workspace/virtual_imu/build_instep.py``:

    AnysoleWorkspace/shared/facts/virtual_imu/<date>/<sid>/imu_instep.npz

The facts store the physical sensor-frame measurements (mount rotation ``M``
applied, no left-foot y flip).  This adapter applies the legacy left y flip
exactly like ``build_gait.synthesize_imu`` and writes the world acceleration
from the facts into the native ``is_acceleration_world=True`` slots.  The
per-frame sensor quaternions (including ``M``) are exposed through
``dataset.quat_lIMU`` / ``quat_rIMU``, so local <-> world stays consistent.

Reads the linked AnysoleWorkspace (splits / shared facts / BVH) read-only and
writes everything under this project's own workspace:

    AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3/gait/{gait_train,gait_val,gait_test}.pt
    AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3/gait/normalizer_gait.pth
    AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3/gait/split_sessions.json
    AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3/{artifact.json,pressure_ranges.csv}

Run (gait repo root, touch_gait env):

    python3 workspace/virtual_imu/build_gait_v3.py
    python3 workspace/virtual_imu/build_gait_v3.py --sessions S10103
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]   # .../gait
HERE = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROJECT_ROOT = REPO_ROOT

import build_gait as v2  # noqa: E402  (同目录)
from AnysoleWorkspace.tool.artifacts import sha256_file, write_artifact  # noqa: E402
from AnysoleWorkspace.tool.workspace import WORKSPACE, canonical_uri  # noqa: E402

ADAPTER_VERSION = "adapter_v3"
OUTPUT_ROOT = WORKSPACE / "model_inputs" / "Step2Motion" / ADAPTER_VERSION
INPUT_ROOT = OUTPUT_ROOT
VIRTUAL_IMU_ROOT = WORKSPACE / "shared/facts/virtual_imu"
TARGET_HZ = v2.TARGET_HZ

ACC_WORLD_LEFT = slice(16, 19)
ACC_WORLD_RIGHT = slice(41, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step2Motion adapter_v3 gait inputs.")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs; default: every canonical split session")
    parser.add_argument("--min-frames", type=int, default=v2.MIN_FRAMES)
    parser.add_argument("--virtual-imu-root", default=str(VIRTUAL_IMU_ROOT),
                        help="directory containing <date>/<sid>/imu_instep.npz")
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT),
                        help="adapter_v3 output root (default: workspace/virtual_imu/adapter_v3)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--splits-csv", default=str(v2.SPLIT_CSV),
                        help="split CSV (default: canonical splits.csv; "
                             "four-fold LSO uses the per-fold CSV)")
    return parser.parse_args()


def virtual_imu_path(session: dict) -> Path:
    date = str(session["meta"].get("date", ""))
    if not date:
        raise ValueError(f"{session['session_id']}: session.json has no date")
    return VIRTUAL_IMU_ROOT / date / session["session_id"] / "imu_instep.npz"


def load_virtual_imu(session: dict) -> dict:
    """原始测量（imu_instep.npz）+ 派生/状态量（imu_instep_state.npz）合并加载。

    文件契约（2026-11-06 拆分）：imu_instep.npz 只含真实传感器的原始输出
    （acc_g/gyro_rad_s + 时间轴键）；世界加速度、传感器四元数等派生量与
    生成状态在 imu_instep_state.npz。
    """
    path = virtual_imu_path(session)
    if not path.is_file():
        raise FileNotFoundError(
            f"{session['session_id']}: missing virtual IMU facts {path}; "
            "run AnysoleWorkspace/tool/virtual_imu/build_instep.py first"
        )
    state_path = path.with_name("imu_instep_state.npz")
    if not state_path.is_file():
        raise FileNotFoundError(
            f"{session['session_id']}: missing virtual IMU state {state_path}"
        )
    data = dict(np.load(path, allow_pickle=False))
    data.update(dict(np.load(state_path, allow_pickle=False)))
    frame_id = np.asarray(data["frame_id"], dtype=np.int64)
    expected = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    if not np.array_equal(frame_id, expected):
        raise ValueError(f"{session['session_id']}: virtual IMU frame_id differs from shared frames")
    required = ("left_acc_g", "right_acc_g", "left_gyro_rad_s", "right_gyro_rad_s",
                "left_acc_world_g", "right_acc_world_g",
                "left_sensor_quat_wxyz", "right_sensor_quat_wxyz")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing virtual IMU keys {missing}")
    return data


def convert_session_v3(session: dict, min_frames: int) -> list[dict] | None:
    """Same clip/skeleton path as v2; only the IMU block is replaced."""
    sid = session["session_id"]
    frames = session["frames"]
    pressure = session["pressure"]
    n = int(session["meta"]["frame_count"])
    fake_mask = np.asarray(frames["fake"], dtype=bool)
    valid_mask = np.asarray(frames["valid"], dtype=bool)
    frame_ids = np.asarray(frames["frame_id"], dtype=np.int64)
    mocap_t = np.asarray(frames["mocap_time_s"], dtype=np.float64)
    fps = float(session["meta"].get("fps", TARGET_HZ))
    dt = 1.0 / fps

    clips = v2.split_fake_clips(fake_mask, min_frames)
    if not clips:
        return None

    left48 = np.asarray(pressure["left48"], dtype=np.float32)
    right48 = np.asarray(pressure["right48"], dtype=np.float32)

    bvh = v2.shared_bvh_path(session)
    bvh_data = v2.load_bvh24(bvh, mocap_t)
    dst_pos = np.asarray(bvh_data["joints"], dtype=np.float64)
    dst_global_rot = np.asarray(bvh_data["global_rotations"], dtype=np.float64)
    dst_offsets = np.asarray(bvh_data["offsets"], dtype=np.float64)
    root_pos = dst_pos[:, 0]

    imu = load_virtual_imu(session)
    if len(imu["left_acc_g"]) != n:
        raise ValueError(f"{sid}: virtual IMU length {len(imu['left_acc_g'])} != frame_count {n}")

    # Legacy Step2Motion convention: the left insole acceleration y axis is
    # inverted (build_gait.synthesize_imu: flip_local_y=True).  Facts store the
    # physical sensor value; flip it here before it enters the .pt channel.
    left_acc_pt = np.asarray(imu["left_acc_g"], dtype=np.float32).copy()
    left_acc_pt[:, 1] *= -1.0
    right_acc_pt = np.asarray(imu["right_acc_g"], dtype=np.float32)
    left_gyro = np.asarray(imu["left_gyro_rad_s"], dtype=np.float32)
    right_gyro = np.asarray(imu["right_gyro_rad_s"], dtype=np.float32)

    insole = np.concatenate(
        [
            v2.foot_features(left48, left_acc_pt, left_gyro),
            v2.foot_features(right48, right_acc_pt, right_gyro),
        ],
        axis=1,
    )

    local_pos_root = dst_pos - root_pos[:, None, :]
    pose = local_pos_root[:, 1:, :].reshape(n, -1).astype(np.float32)
    quats = v2.rotmat_to_quat_wxyz(dst_global_rot)
    displacements = np.zeros((n, 3), dtype=np.float32)
    displacements[1:] = (root_pos[1:] - root_pos[:-1]).astype(np.float32)
    root_quats = quats[:, 0]

    left_acc_world = np.asarray(imu["left_acc_world_g"], dtype=np.float32)
    right_acc_world = np.asarray(imu["right_acc_world_g"], dtype=np.float32)
    left_sensor_quat = np.asarray(imu["left_sensor_quat_wxyz"], dtype=np.float32)
    right_sensor_quat = np.asarray(imu["right_sensor_quat_wxyz"], dtype=np.float32)

    out = []
    for start, end in clips:
        t_of = lambda k: mocap_t[0] + k * dt  # noqa: E731
        out.append(
            {
                "insole": insole[start:end],
                "pose": pose[start:end],
                "quats": quats[start:end],
                "displacements": displacements[start:end],
                "global_rots": root_quats[start:end],
                "parents": v2.JOINT_PARENTS.copy(),
                "offsets": dst_offsets.astype(np.float32),
                "joint_names": list(v2.JOINT_NAMES),
                "session_id": sid,
                "clip_index": int(len(out)),
                "n_session_clips": int(len(clips)),
                "frame_ids": frame_ids[start:end],
                "valid": valid_mask[start:end],
                "raw_bvh_path": str(bvh),
                "raw_start_time": float(t_of(start + 1)),
                "raw_end_time": float(t_of(end)),
                # adapter_v3-only payload, consumed after the v2 state builder.
                "left_acc_world_g": left_acc_world[start:end],
                "right_acc_world_g": right_acc_world[start:end],
                "left_acc_local_phys": left_acc_pt[start:end],
                "right_acc_local_phys": right_acc_pt[start:end],
                "left_sensor_quat": left_sensor_quat[start:end],
                "right_sensor_quat": right_sensor_quat[start:end],
            }
        )
    return out


def build_motion_dataset_state_v3(clips_data: list[dict]):
    """v2 state builder + overwrite the IMU fields from the virtual facts.

    The v2 builder derives the world acceleration from the local channels and
    the nominal rest axes (no mount rotation).  The overwrite makes every IMU
    field consistent with the virtual sensor frame ``S = R_foot @ rest_axes @ M``:

    * world acceleration slots  <- facts (physics, mount-corrected)
    * local acceleration copies <- physical sensor frame (legacy left y flip)
    * quat_{l,r}IMU             <- sensor orientation including M
    * gyro slots                <- already physical (never rotated by v2)
    """
    import torch

    state = v2.build_motion_dataset_state(clips_data, no_imu=False)
    for index, clip in enumerate(clips_data):
        n_kept = clip["pose"].shape[0] - 1
        if n_kept <= 0:
            continue
        state.insole[index][:, ACC_WORLD_LEFT] = torch.from_numpy(clip["left_acc_world_g"][1:]).float()
        state.insole[index][:, ACC_WORLD_RIGHT] = torch.from_numpy(clip["right_acc_world_g"][1:]).float()
        state.left_acceleration_local[index] = torch.from_numpy(clip["left_acc_local_phys"][1:]).float()
        state.right_acceleration_local[index] = torch.from_numpy(clip["right_acc_local_phys"][1:]).float()
        state.quat_lIMU[index] = torch.from_numpy(clip["left_sensor_quat"][1:]).float()
        state.quat_rIMU[index] = torch.from_numpy(clip["right_sensor_quat"][1:]).float()
    return state


def create_dataset_v3(clips_data: list[dict], out_path: Path, dry_run: bool):
    if not clips_data:
        return None
    if dry_run:
        return clips_data
    import torch

    state = build_motion_dataset_state_v3(clips_data)
    shim = v2._dataset_shim()
    dataset = shim.__new__(shim)
    dataset.__dict__.update(state.__dict__)
    dataset.virtual_imu = {
        "adapter_version": ADAPTER_VERSION,
        "facts_root": str(VIRTUAL_IMU_ROOT),
        "source": "tool/virtual_imu/build_instep.py (instep mount, noise layers per artifact)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    return dataset


def collect_clips(session_ids, splits, min_frames, cache, sessions_cache, errors, details):
    converted: list[dict] = []
    for sid in session_ids:
        if sid in errors:
            continue
        if sid not in cache:
            try:
                session = sessions_cache.setdefault(sid, v2.load_shared_session(sid))
                cache[sid] = convert_session_v3(session, min_frames)
            except Exception as exc:  # noqa: BLE001
                errors[sid] = f"{type(exc).__name__}: {exc}"
                print(f"skip {sid}: {errors[sid]}")
                continue
        clips = cache[sid]
        if not clips:
            fake_all = bool(np.asarray(sessions_cache[sid]["frames"]["fake"], dtype=bool).all())
            errors[sid] = "all_frames_fake" if fake_all else f"too_short_or_fake(min_frames={min_frames})"
            continue
        converted.extend(clips)
        details[sid] = {
            "split": v2.split_of(sid, splits),
            "n_clips": len(clips),
            "n_frames": sum(int(len(c["frame_ids"][1:])) for c in clips),
            "clip_ranges": [(int(c["frame_ids"][1]), int(c["frame_ids"][-1])) for c in clips],
        }
        print(f"{details[sid]['split']} {sid}: {len(clips)} clip(s), kept_frames={details[sid]['n_frames']}")
    return converted


def write_pressure_ranges(splits: dict[str, list[str]], errors: dict) -> None:
    rows = []
    for name in ("train", "val", "test"):
        for sid in splits[name]:
            if sid in errors:
                continue
            session = v2.load_shared_session(sid)
            pressure = session["pressure"]
            rows.append({
                "session_id": sid,
                "date": session["meta"].get("date", ""),
                "split": name,
                "value_min": float(pressure["value_min"]),
                "value_max": float(pressure["value_max"]),
                "unit": str(pressure["unit"]),
                "source_left": str(pressure["source_file_left"]),
                "source_right": str(pressure["source_file_right"]),
            })
    if not rows:
        return
    path = INPUT_ROOT / "pressure_ranges.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def build_normalizer_v3(dataset_path: Path, out_path: Path, name: str) -> dict:
    """Same stats as v2.build_normalizer, without canonical_uri on workspace paths."""
    v2._prefer_env_site()
    v2._dataset_shim()
    if str(v2.STEP2MOTION_SRC) not in sys.path:
        sys.path.insert(0, str(v2.STEP2MOTION_SRC))
    import torch

    normalizer_mod = importlib.import_module("normalizer")
    db = normalizer_mod.MotionDataset.load(str(dataset_path), torch.device("cpu"))

    def compute_mean_std(data):
        mean = data.mean(dim=0).float()
        std = data.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std).float()
        return mean, std

    normalizer = normalizer_mod.Normalizer.__new__(normalizer_mod.Normalizer)
    normalizer.poses_mean, normalizer.poses_std = compute_mean_std(torch.cat(db.poses, dim=0))
    normalizer.distances_mean, normalizer.distances_std = compute_mean_std(db.distances)
    normalizer.insoles_mean, normalizer.insoles_std = compute_mean_std(torch.cat(db.insole, dim=0))
    normalizer.id = hash(
        normalizer.poses_mean.mean().item()
        + normalizer.poses_std.mean().item()
        + normalizer.distances_mean.mean().item()
        + normalizer.distances_std.mean().item()
        + normalizer.insoles_mean.mean().item()
        + normalizer.insoles_std.mean().item()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalizer, out_path)
    return {"name": name, "source": str(dataset_path), "output": str(out_path)}


def run_self_test() -> None:
    # The IMU slices match the v2 contract.
    assert v2.MIN_FRAMES == 101
    session = v2.load_shared_session("S10103")
    clips = convert_session_v3(session, v2.MIN_FRAMES)
    assert clips, "S10103 produced no clips"
    clip = clips[0]
    assert clip["insole"].shape[1] == 50
    assert v2.TARGET_HZ == TARGET_HZ
    print("self-test ok")


def main() -> int:
    global VIRTUAL_IMU_ROOT, INPUT_ROOT, OUTPUT_ROOT
    args = parse_args()
    VIRTUAL_IMU_ROOT = Path(args.virtual_imu_root)
    INPUT_ROOT = Path(args.out_root)
    OUTPUT_ROOT = INPUT_ROOT
    if args.self_test:
        run_self_test()
        return 0
    splits = v2.read_split(Path(args.splits_csv))
    if args.sessions:
        wanted = {value.strip() for value in args.sessions.split(",") if value.strip()}
        unknown = wanted - set().union(*map(set, splits.values()))
        if unknown:
            raise SystemExit(f"sessions not in canonical split: {sorted(unknown)}")
        splits = {name: [sid for sid in ids if sid in wanted] for name, ids in splits.items()}

    cache: dict = {}
    sessions_cache: dict = {}
    errors: dict = {}
    details: dict = {}
    normalizers: dict = {}
    for split_name in ("train", "val", "test"):
        out_dir = INPUT_ROOT / "gait"
        out_path = out_dir / f"gait_{split_name}.pt"
        clips = collect_clips(splits[split_name], splits, args.min_frames, cache,
                              sessions_cache, errors, details)
        n_frames = sum(int(c["pose"].shape[0]) for c in clips)
        print(f"gait/{split_name}: clips={len(clips)} frames={n_frames} insole=50")
        dataset = create_dataset_v3(clips, out_path, args.dry_run)
        if dataset is None:
            print(f"gait/{split_name}: empty")
        elif not args.dry_run:
            print(f"wrote {out_path}")
            if split_name == "train":
                normalizers["gait"] = build_normalizer_v3(
                    out_path, out_dir / "normalizer_gait.pth", "gait"
                )
    if not args.dry_run:
        v2.write_split_sessions(INPUT_ROOT / "gait", splits=splits, details=details,
                                skipped=errors, split_csv=Path(args.splits_csv),
                                manifest=v2.MANIFEST, variant="gait")
        imu_files = sorted(VIRTUAL_IMU_ROOT.glob("*/*/imu_instep.npz"))
        write_artifact(
            INPUT_ROOT,
            schema_version="model_input.step2motion.v1",
            producer="tool/adapters/Step2Motion/build_gait_v3.py",
            repository_root=PROJECT_ROOT,
            parameters={
                "adapter_version": ADAPTER_VERSION,
                "base_adapter": "adapter_v2 (build_gait.py, unchanged)",
                "variants": ["gait"],
                "min_frames": args.min_frames,
                "target_hz": TARGET_HZ,
                "input_dims": {"gait": 50},
                "output_dim": 72,
                "imu_source": "shared/facts/virtual_imu/<date>/<sid>/imu_instep.npz",
                "imu_facts_count": len(imu_files),
                "imu_contract": (
                    "adapter_v2 channel layout unchanged; acc world slots from facts; "
                    "left local acc keeps the legacy y flip; quat_*IMU includes the mount rotation M"
                ),
                "time_axis": "shared frames.npz mocap_time_s",
                "normalizers": normalizers,
            },
            source_artifacts=[canonical_uri(v2.SPLIT_CSV), str(VIRTUAL_IMU_ROOT)],
            source_hashes={"split_csv": sha256_file(v2.SPLIT_CSV)},
            consumers=["Baselines/Step2Motion"],
        )
        write_pressure_ranges(splits, errors)

    print("empty_or_skipped:", dict(sorted(errors.items())))
    unexpected = {sid: reason for sid, reason in errors.items() if reason != "all_frames_fake"}
    if unexpected:
        print(f"ERROR: {len(unexpected)} unexpected skips: {unexpected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
