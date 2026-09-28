"""pressure_toolkit depth mask：human_mask_sam31 ∩ finite_depth ∩ depth∈[0.4m,5m]。

06 任务书 P2 契约：

- 直接消费 Agent A 索引的外部 SAM3.1 单受试者 mask，不运行新分割模型；
- 交集在保存前完成；loader 里上游原生的 3×3 膨胀作用于交集后的人体 mask，
  不会重新引入全景地面/背景；
- 逐帧保存 frame_id、human-mask path/hash、匹配误差、交集前后面积和空 mask 检查。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_MIN_M = 0.4
DEPTH_MAX_M = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_depth_mask(depth_png: Path, human_mask_png: Path,
                     output_png: Path) -> dict:
    """生成单帧交集 mask，返回该帧的 provenance 记录。"""
    depth = np.asarray(Image.open(depth_png), dtype=np.uint16)
    mask_img = Image.open(human_mask_png)
    human = np.asarray(mask_img.convert("L"), dtype=np.uint8) > 0
    if human.shape != depth.shape:
        raise ValueError(
            f"mask/depth size mismatch: {human_mask_png} {human.shape} "
            f"vs {depth_png} {depth.shape}")

    finite = np.isfinite(depth) & (depth > 0)
    in_range = (depth >= DEPTH_MIN_M * 1000.0) & (depth <= DEPTH_MAX_M * 1000.0)
    # 交集：人体前景 ∩ 有限深度 ∩ [0.4m,5m]。膨胀在 loader（上游行为）
    # 中作用于本交集结果。
    intersected = human & finite & in_range

    human_area = int(np.count_nonzero(human))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    # 三通道 0/255 PNG：上游 loader 读入后 3×3 膨胀再按通道取均值
    # （np.mean(mask, axis=-1)），单通道 PNG 会错误地在宽度上求均值。
    rgb_mask = np.repeat((intersected.astype(np.uint8) * 255)[..., None], 3, axis=2)
    Image.fromarray(rgb_mask, mode="RGB").save(output_png)

    return {
        "frame_id": None,  # caller fills shared frame id
        "depth_path": str(depth_png),
        "human_mask_path": str(human_mask_png),
        "human_mask_hash": sha256_file(human_mask_png),
        "depth_hash": sha256_file(depth_png),
        "human_mask_area": human_area,
        "intersected_area": int(np.count_nonzero(intersected)),
        "empty": bool(not np.any(intersected)),
    }
