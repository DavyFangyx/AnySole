#!/usr/bin/env python3
"""CLIFF_results.npz → pressure_tookit ``<session>_cliff_hr48.npz`` 初值契约。

T1-B（11 号文档 §5）：cliff/ 是 PoseTransOpt 与 pressure_tookit 共用的前端。
``run_cliff.py`` 产出的 ``CLIFF_results.npz`` 是 MMVP 方法族的通用单人舞台；
本模块把它转换为 pressure_tookit 上游 ``load_init_pose`` 需要的初值格式。

消费端契约（实测 ``Baselines/pressure_tookit/lib/dataextra/data_loader.py``）：

- ``Pressure_Dataset`` 的 ``stage='init_pose'`` 读
  ``init_root/<dataset>/<sub>/<seq>/<seq>_cliff_hr48.npz``（L341-343），即
  ``model_inputs/pressure_toolkit/v1/initialization/<date>/<subject>/<session>/``；
- ``load_init_pose(path, form='cliff')``（L114-146）：

  ```python
  init_data = dict(np.load(data_fn).items())          # allow_pickle=False
  init_pose = np.expand_dims(init_data['pose'][0], 0) if init_data['pose'].shape[0] > 1 \
              else init_data['pose']
  return init_pose[:, 3:], None, None                 # body_pose (1,69)，丢全局朝向
  ```

  即 **实际读取的键只有 ``pose``**：(n,72) = global_orient(3) + body_pose(69)，
  消费端取第 0 行并丢掉前 3 维。
- ``load_init_shape(path)``（L157-159）读 ``shape`` 与 ``model_scale_opt``，但它作用于
  ``annotations/<dataset>/smpl_pose/<sub>/init_shape_<sub>.npz``（L244-247），
  不是本文件；这里仍写出这两键，使 upstream cliff npz 契约（pose/shape/
  model_scale_opt 三键）完整、可被将来任何按该契约读取的调用方直接使用。
  ``model_scale_opt`` 与 init_shape 阶段的常量一致：
  ``torch.tensor([1.0])``（``lib/core/fit_single_frame.py:86``、
  ``lib/core/smpl_mmvp.py:137``）。
- 文件必须能被 ``np.load(path)``（默认 ``allow_pickle=False``）读取，因此只写
  数值/字符串数组，不写 pickle dict。

写出键：

    pose             float32[n,72]   消费端读取（第 0 行 → body_pose 69）
    shape            float32[n,10]   SMPL betas（upstream 契约键）
    model_scale_opt  float32[1]      [1.0]（init_shape 常量，upstream 契约键）
    frame_id         int64[n]        provenance：canonical shared frame id
    valid            uint8[n]        provenance：CLIFF 单人有效性
    global_t         float32[n,3]    provenance：消费端丢弃（load_init_pose 返回 None）
    source_cliff_npz / source_pose_selector / conversion  provenance 字符串
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

CLIFF_ROOT = WORKSPACE / "shared/frontends/cliff_hr48/v1"
POSE_SELECTOR = "row 0 (load_init_pose: init_data['pose'][0], global orient dropped)"


def read_manifest() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2], row["session_id"]


def convert(cliff_npz: Path, output: Path, force: bool = False) -> dict:
    """One session: ``CLIFF_results.npz`` → ``<session>_cliff_hr48.npz``."""
    if not cliff_npz.is_file():
        raise FileNotFoundError(f"CLIFF_results.npz missing: {cliff_npz}")
    if output.is_file() and not force:
        return {"session": output.name, "status": "ok_existing", "path": str(output)}

    data = np.load(cliff_npz, allow_pickle=False)
    pose = np.asarray(data["pose"], dtype=np.float32)
    shape = np.asarray(data["shape"], dtype=np.float32)
    frame_id = np.asarray(data["frame_id"], dtype=np.int64)
    valid = np.asarray(data["valid"], dtype=np.uint8)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"{cliff_npz}: expected pose (n,72), got {pose.shape}")
    if shape.ndim != 2 or shape.shape[1] != 10:
        raise ValueError(f"{cliff_npz}: expected shape (n,10), got {shape.shape}")
    if len(frame_id) != len(pose) or len(valid) != len(pose):
        raise ValueError(
            f"{cliff_npz}: row count mismatch pose={len(pose)} "
            f"frame_id={len(frame_id)} valid={len(valid)}")
    if not np.array_equal(frame_id, np.arange(len(frame_id), dtype=np.int64)):
        raise ValueError(f"{cliff_npz}: frame ids are not canonical 0..n-1")
    global_t = np.asarray(data["global_t"], dtype=np.float32)

    payload = {
        "pose": pose,
        "shape": shape,
        # init_shape stage constant: torch.tensor([1.0]) (fit_single_frame.py:86)
        "model_scale_opt": np.asarray([1.0], dtype=np.float32),
        # provenance (the consumer reads only 'pose' from this file)
        "frame_id": frame_id,
        "valid": valid,
        "global_t": global_t,
        "source_cliff_npz": np.asarray(str(cliff_npz)),
        "source_pose_selector": np.asarray(POSE_SELECTOR),
        "conversion": np.asarray(
            "CLIFF_results.npz -> pressure_tookit load_init_pose(form='cliff') "
            "contract; pose kept as (n,72), shape/model_scale_opt written for "
            "upstream key parity"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(output)
    return {
        "session": output.stem,
        "status": "built",
        "path": str(output),
        "frames": int(len(pose)),
        "cliff_valid": int(valid.astype(bool).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sessions", default="", help="comma-separated session IDs")
    parser.add_argument("--cliff-root", default=str(CLIFF_ROOT),
                        help="tree holding <date>/<subject>/<session>/CLIFF_results.npz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cliff_root = Path(args.cliff_root)
    rows = read_manifest()
    sessions = ([s for s in args.sessions.split(",") if s] if args.sessions
                else split_sessions(args.split))
    reports = []
    for sid in sessions:
        row = rows[sid]
        date, subject, _ = session_parts(row)
        session_root = cliff_root / date / subject / sid
        reports.append(convert(session_root / "CLIFF_results.npz",
                               session_root / f"{sid}_cliff_hr48.npz",
                               args.force))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
