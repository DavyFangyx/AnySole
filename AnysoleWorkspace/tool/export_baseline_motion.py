#!/usr/bin/env python3
"""Export native MMVP baseline results to the unified eval_motion contract.

The native baseline outputs remain untouched.  This adapter writes only:

    results/baselines/<Model>/predictions/eval_motion/<session>.npz   (V2M motion)
    results/baselines/FPP-Net/predictions/v2t/<session>.npz           (V2T pressure)

Each archive contains ``joint_xyz_world``, ``joint_names`` and ``valid_mask``
plus provenance fields.  Coordinates are converted from the SMPL/floor
convention (x, y-up, z) to the display/world convention (x, z-up, y ->
``(x, -z, y)``), matching results_display's SMPL GT reader.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"
RESULTS = ROOT / "results"
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri as _frozen_resolve_uri  # noqa: E402
MANIFEST = WORKSPACE / "protocol/manifests" / "session_manifest.jsonl"
SMPL_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def split_ids(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def rows() -> dict[str, dict]:
    result = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["session_id"]] = row
    return result


def split_sessions(split: str) -> list[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    out = []
    with (WORKSPACE / "protocol/splits" / "default" / "splits.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    out.append(value)
    return sorted(set(out))


def session_parts(row: dict) -> tuple[str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2]


def valid_frames(row: dict, n: int) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    raw = row.get("fake_frame_indices", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    for index in raw or []:
        index = int(index)
        if 0 <= index < n:
            keep[index] = False
    valid = row.get("valid_frame_indices", [])
    if isinstance(valid, str) and valid:
        try:
            valid = json.loads(valid)
        except json.JSONDecodeError:
            valid = []
    if valid:
        keep[:] = False
        for index in valid:
            index = int(index)
            if 0 <= index < n:
                keep[index] = True
    return keep


def smpl_yup_to_display(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def resolve_uri(value: str, **kwargs) -> Path:
    """Delegate to the frozen workspace resolver; ``workspace://`` stays removed."""
    text = str(value)
    if text.startswith("workspace://"):
        raise ValueError("workspace:// has been removed; use a canonical URI")
    return _frozen_resolve_uri(text, **kwargs)


def write_archive(path: Path, joints: np.ndarray, valid: np.ndarray,
                  source: str, session: str, frame_indices: np.ndarray | None = None,
                  vertices: np.ndarray | None = None,
                  poses: np.ndarray | None = None,
                  source_type: str = "model_prediction",
                  extra_meta: dict | None = None) -> None:
    joints = np.asarray(joints, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"unified SMPL joints must be (T,24,3), got {joints.shape}")
    if len(valid) != len(joints):
        raise ValueError("valid_mask length does not match joint_xyz_world")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_xyz_world": joints,
        "joint_names": np.asarray(SMPL_NAMES),
        "valid_mask": valid,
        "frame_indices": np.arange(len(joints), dtype=np.int64) if frame_indices is None else np.asarray(frame_indices, dtype=np.int64),
        "motion_protocol": np.asarray("smpl24"),
        "model_units": np.asarray("m"),
        "coordinate_system": np.asarray("world_z_up"),
        "joint_coordinate_system": np.asarray("world_z_up"),
        "session_id": np.asarray(session),
        "source_native_output": np.asarray(source),
        # 评估整改任务 01/02：输出来源类型进入导出元数据，
        # 只有 model_prediction / method_optimization 及其确定性派生可进入公共指标。
        "provenance_source_type": np.asarray(source_type),
        "target_fps": np.asarray(40.0, dtype=np.float32),
    }
    for key, value in (extra_meta or {}).items():
        payload[key] = np.asarray(value)
    if vertices is not None:
        payload["vertices_world"] = np.asarray(vertices, dtype=np.float32)
    if poses is not None:
        payload["poses"] = np.asarray(poses, dtype=np.float32).reshape(len(joints), 72)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}  joints={joints.shape} valid={int(valid.sum())}")


def pressure_frame_paths(root: Path, date: str, subject: str, session: str) -> list[Path]:
    directory = root / "results" / date / subject / session
    paths = sorted(directory.glob("smpl_*.npz"))
    if not paths:
        raise FileNotFoundError(f"pressure_toolkit frame outputs missing: {directory}")
    return paths


def export_pressure(row: dict, args: argparse.Namespace, output: Path) -> None:
    date, subject = session_parts(row)
    session = row["session_id"]
    gender = "female" if session in split_ids(args.female) or subject in split_ids(args.female) else "male"
    frame_paths = pressure_frame_paths(resolve_uri(args.pressure_root), date, subject, session)
    if str(ROOT / "Baselines" / "pressure_tookit") not in sys.path:
        sys.path.insert(0, str(ROOT / "Baselines" / "pressure_tookit"))
    from lib.core.smpl_mmvp import SMPL_MMVP
    import torch

    essential = WORKSPACE / "assets/third_party" / "pressure_toolkit" / "essential"
    model = SMPL_MMVP(str(essential), gender=gender, stage="tracking").cpu()
    n = int(row["n_frames"])
    joints = np.zeros((n, 24, 3), dtype=np.float32)
    vertices = np.zeros((n, 6890, 3), dtype=np.float32)
    poses = np.zeros((n, 72), dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    for path in frame_paths:
        frame = int(path.stem.split("_")[-1])
        if not 0 <= frame < n:
            continue
        data = dict(np.load(path, allow_pickle=True))
        shape = np.asarray(data["shape"], dtype=np.float32).reshape(-1)
        if shape.size == 10:
            shape = np.concatenate(([1.0], shape))
        if shape.size != 11:
            raise ValueError(f"{path}: expected scale+10 betas, got {shape.shape}")
        body_pose = np.asarray(data["body_pose"], dtype=np.float32).reshape(1, -1)
        global_rot = np.asarray(data.get("global_rot", np.zeros((1, 3))), dtype=np.float32).reshape(1, 3)
        transl = np.asarray(data["transl"], dtype=np.float32).reshape(1, 3)
        scale = np.asarray(data.get("model_scale_opt", shape[:1]), dtype=np.float32).reshape(1)
        model.setPose(
            betas=torch.from_numpy(shape.reshape(1, 11)),
            model_scale_opt=torch.from_numpy(scale),
            body_pose=torch.from_numpy(body_pose),
            global_orient=torch.from_numpy(global_rot),
            transl=torch.from_numpy(transl),
        )
        model.update_shape()
        model.init_plane()
        model_output = model.update_pose()
        native = model_output.smpl_joints[0].detach().cpu().numpy()
        joints[frame] = smpl_yup_to_display(native)
        vertices[frame] = smpl_yup_to_display(model_output.vertices[0].detach().cpu().numpy())
        pose_aa = np.concatenate([global_rot.reshape(-1), body_pose.reshape(-1)])
        poses[frame, :min(72, pose_aa.size)] = pose_aa[:72]
        available[frame] = True
    valid = valid_frames(row, n) & available
    if not np.all(available[valid_frames(row, n)]):
        missing = np.flatnonzero(valid_frames(row, n) & ~available)
        raise ValueError(f"{session}: missing pressure_toolkit frames: {missing[:20].tolist()}")
    # 2026-09-27 生成清单裁定：pressure_toolkit 的 shape/表面是其拟合
    # pipeline 原生生成，PVE 可评估（表面对 GT 表面比较）。
    write_archive(output, joints, valid, str(frame_paths[0].parent), session,
                  vertices=vertices, poses=poses,
                  source_type="method_optimization",
                  extra_meta={
                      "surface_source": "method_optimization",
                      "shape_source": "method_optimization",
                      "public_surface_metrics": "true",
                  })


def export_vp_mocap(row: dict, args: argparse.Namespace, output: Path) -> None:
    session = row["session_id"]
    date, subject = session_parts(row)
    native_path = WORKSPACE / "work" / "VP-MoCap" / "v1" / "pose_optimization" / date / subject / session / "opt_result.pth"
    if not native_path.is_file():
        raise FileNotFoundError(f"PoseTransOpt output missing: {native_path}")
    import torch
    import smplx

    data = torch.load(native_path, map_location="cpu")
    pose = data["pose"].detach().cpu() if torch.is_tensor(data["pose"]) else torch.as_tensor(data["pose"])
    beta = data.get("beta", data.get("betas"))
    trans = data["trans"].detach().cpu() if torch.is_tensor(data["trans"]) else torch.as_tensor(data["trans"])
    beta = beta.detach().cpu() if torch.is_tensor(beta) else torch.as_tensor(beta)
    if pose.ndim != 4 or tuple(pose.shape[1:]) != (24, 3, 3):
        raise ValueError(f"{native_path}: expected pose (T,24,3,3), got {tuple(pose.shape)}")
    if beta.ndim == 1:
        beta = beta.reshape(1, -1).repeat(len(pose), 1)
    # PoseTransOpt saves float64 tensors; smplx's lbs matmul requires float32.
    pose, beta, trans = pose.float(), beta.float(), trans.float()
    smpl_path = WORKSPACE / "assets/third_party" / "smpl" / "SMPL_NEUTRAL.pkl"
    model = smplx.create(str(smpl_path), "smpl", gender="neutral", batch_size=len(pose), num_betas=10)
    with torch.no_grad():
        result = model(
            betas=beta[:, :10],
            body_pose=pose[:, 1:],
            global_orient=pose[:, [0]],
            transl=trans,
            pose2rot=False,
        )
    native = result.joints[:, :24].detach().cpu().numpy()
    native_vertices = result.vertices.detach().cpu().numpy()
    from scipy.spatial.transform import Rotation
    poses_aa = Rotation.from_matrix(pose.numpy().reshape(-1, 3, 3)).as_rotvec().reshape(len(pose), 72)
    n = int(row["n_frames"])
    full = np.zeros((n, 24, 3), dtype=np.float32)
    full_vertices = np.zeros((n, 6890, 3), dtype=np.float32)
    full_poses = np.zeros((n, 72), dtype=np.float32)
    start = 2  # Dataset.load_pose/load_2d_keypoints discard the first/last 2 frames.
    end = min(n, start + len(native))
    full[start:end] = smpl_yup_to_display(native[: end - start])
    full_vertices[start:end] = smpl_yup_to_display(native_vertices[: end - start])
    full_poses[start:end] = poses_aa[: end - start]
    available = np.zeros(n, dtype=bool)
    available[start:end] = True
    # 2026-10-01 U7 拍 A / E6：三基线 provenance 统一——VP-MoCap 的 pose/shape
    # 来自 PoseTransOpt 优化 pipeline 原生生成（与注册表 sources 一致），PVE 可评估。
    write_archive(output, full, valid_frames(row, n) & available, str(native_path), session,
                  np.arange(n, dtype=np.int64), vertices=full_vertices, poses=full_poses,
                  source_type="method_optimization",
                  extra_meta={
                      "surface_source": "method_optimization",
                      "shape_source": "method_optimization",
                      "public_surface_metrics": "true",
                  })


def export_fpp_v2t(row: dict, output: Path) -> None:
    # E3-A (2026-10-03): the canonical exporter owns the V2T archive contract
    # (press2Cont th=0.5 binary vertex GT + threshold/revision provenance).
    # This entry point delegates so the two writers cannot drift again — the
    # previous copy read the retired %03d sidecars and labelled the GT f6_soft.
    # The canonical function writes the same output path.
    from AnysoleWorkspace.tool.adapters.mmvp_series.fpp.export_v2t import export_session

    expected = RESULTS / "baselines" / "FPP-Net" / "predictions" / "v2t" / f"{row['session_id']}.npz"
    if output != expected:
        raise ValueError(f"fpp_v2t output path drift: {output} != {expected}")
    print(export_session(row, force=not output.is_file()))


def choose_sessions(args: argparse.Namespace, manifest: dict[str, dict]) -> list[str]:
    if args.session:
        return [value.strip() for value in args.session.split(",") if value.strip()]
    return split_sessions(args.split)


def split_sessions(split: str) -> list[str]:
    columns = ("train", "val", "test") if split == "all" else (split,)
    values = []
    with (WORKSPACE / "protocol/splits" / "default" / "splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                value = (row.get(column) or "").strip()
                if value:
                    values.append(value)
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("motionpro", "pressure_toolkit", "vp_mocap", "fpp_v2t", "all"), default="all")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--session", default="", help="comma-separated session IDs")
    # 评估整改任务 02：pressure 输入默认读 workspace fitting 根；
    # 迁移期旧树可通过 --pressure-root results://baselines/pressure_toolkit 显式传入。
    parser.add_argument("--pressure-root", default="work://pressure_toolkit/fitting")
    parser.add_argument("--female", "--famale", dest="female", default="S14")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = rows()
    selected = choose_sessions(args, manifest)
    models = ("motionpro", "pressure_toolkit", "vp_mocap", "fpp_v2t") if args.model == "all" else (args.model,)
    for model in models:
        for session in selected:
            if session not in manifest:
                raise KeyError(f"session not in manifest: {session}")
            if model == "fpp_v2t":
                # 评估整改任务 02：FPP-Net V2T 与 VP-MoCap V2M 分离，
                # V2T 写入 results/baselines/FPP-Net/predictions/v2t/。
                output = RESULTS / "baselines" / "FPP-Net" / "predictions" / "v2t" / f"{session}.npz"
                if output.is_file() and not args.force:
                    print(f"skip existing {output}")
                else:
                    export_fpp_v2t(manifest[session], output)
                continue
            model_dir = {
                "motionpro": "baselines/MotionPRO",
                "pressure_toolkit": "baselines/pressure_toolkit",
                "vp_mocap": "baselines/VP-MoCap",
            }[model]
            output = RESULTS / model_dir / "predictions" / "eval_motion" / f"{session}.npz"
            if output.is_file() and not args.force:
                print(f"skip existing {output}")
                continue
            if model == "motionpro":
                if output.is_file():
                    print(f"skip existing {output}")
                    continue
                raise FileNotFoundError(
                    f"MotionPRO unified output missing: {output}. Run app.test_frappe; "
                    "MotionPRO writes this contract during evaluation."
                )
            elif model == "pressure_toolkit":
                export_pressure(manifest[session], args, output)
            else:
                export_vp_mocap(manifest[session], args, output)


if __name__ == "__main__":
    main()
