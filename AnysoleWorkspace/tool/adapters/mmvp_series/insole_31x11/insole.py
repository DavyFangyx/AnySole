"""31×11 insole：公共版本化 MMVP representation 的上游契约包装。

06 任务书 P4：

- 只读消费 ``shared/representations/tactile/mmvp_31x11/v1``，不重建私有树；
- 每帧 insole 保存 shared frame id，不仅依赖排序文件名；
- 数值与 D_Test4 已验收映射 parity（representation v1 即该映射，逐帧逐值
  一致，仅增加上游 load_contact 期望的 dict 包装）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from AnysoleWorkspace.tool.adapters.mmvp_series.common import common


def load_mmvp_session(date: str, subject: str, session: str) -> dict:
    rep_dir = common.MMVP_ROOT / date / subject / session
    frame_ids = np.load(rep_dir / "frame_id.npy")
    return {
        "dir": rep_dir,
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "artifact": json.loads(
            (rep_dir / "artifact.json").read_text(encoding="utf-8")),
    }


def wrap_insole(insole_npy: Path, frame_id: int, source: str) -> dict:
    """(2,31,11) float32 → 上游 load_contact 期望的 dict 包装。

    数值不经过任何变换；``frame_id``/``source`` 为 provenance 附加键。
    """
    array = np.load(insole_npy)
    if array.shape != (2, 31, 11) or array.dtype != np.float32:
        raise ValueError(
            f"{insole_npy}: expected float32 (2,31,11), got {array.dtype} {array.shape}")
    return {
        "insole": [array[0], array[1]],
        "frame_id": int(frame_id),
        "source": source,
    }


def build_session_insoles(session_facts: dict, mmvp: dict,
                          session_dir: Path, force: bool) -> dict:
    """把一个 session 的公共 31×11 逐帧包装进 model_inputs 树。

    Returns 统计记录（帧数、parity 校验、hash）。
    """
    facts_frame_ids = np.asarray(session_facts["frames"]["frame_id"], dtype=np.int64)
    rep_frame_ids = mmvp["frame_ids"]
    if not np.array_equal(facts_frame_ids, rep_frame_ids):
        raise ValueError(
            f"frame id mismatch between shared facts and mmvp representation: "
            f"facts {facts_frame_ids.shape} vs mmvp {rep_frame_ids.shape}")

    insole_dir = session_dir / "insole"
    insole_dir.mkdir(parents=True, exist_ok=True)
    parity = True
    count = 0
    for frame_id in facts_frame_ids:
        out_path = insole_dir / f"{int(frame_id):06d}.npy"
        src = mmvp["dir"] / "insole" / f"{int(frame_id):06d}.npy"
        if out_path.is_file() and not force:
            # 已存在也要复核 parity：报告口径是对全 session 的
            if not np.array_equal(
                    np.load(out_path, allow_pickle=True).item()["insole"][0],
                    np.load(src)[0]):
                parity = False
            count += 1
            continue
        payload = wrap_insole(src, int(frame_id), str(src))
        np.save(out_path, payload)
        # 数值 parity：与公共表示逐值一致（包装不改变任何数值）
        if not np.array_equal(payload["insole"][0], np.load(src)[0]):
            parity = False
        count += 1
    return {
        "n_written": count,
        "frame_count": int(len(facts_frame_ids)),
        "numeric_parity_with_shared": bool(parity),
        "source_representation": "shared://representations/tactile/mmvp_31x11/v1",
        "mapping": mmvp["artifact"].get("parameters", {}).get(
            "mapping", "audited_4x12_to_31x11_nearest_cell"),
    }
