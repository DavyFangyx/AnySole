"""35 解剖表面点选择器：SMPL 均值模板 7 区域 × 5 点最远点采样。

论文 TTA 的 L_2D 使用「44 关节 + 35 解剖表面点」重投影；本选择器给出该 35 点
在本项目中的确定性定义。区域划分 = 蒙皮权重最近身体关节（7 区域，SMPL 关节索引）：

  头[15] / 躯干[3,6,9,12] / 骨盆[0,1,2] / 左臂[13,16,18,20,22] /
  右臂[14,17,19,21,23] / 左腿[4,7,10] / 右腿[5,8,11]

用法：python data/surface_sel35.py（输出 data/surface_sel35.npy + .json）。
装配与 TTA 双方共用该产物。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hmr2_gait 根
from hmr2_tta.paths import RESULTS_ROOT, SMPL_MODEL_PATH  # noqa: E402

REGIONS = {
    "head": [15],
    "torso": [3, 6, 9, 12],
    "pelvis": [0, 1, 2],
    "left_arm": [13, 16, 18, 20, 22],
    "right_arm": [14, 17, 19, 21, 23],
    "left_leg": [4, 7, 10],
    "right_leg": [5, 8, 11],
}
POINTS_PER_REGION = 5


def farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    idx = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))]
    dist = np.full(len(points), np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(points - points[idx[-1]], axis=1)
        dist = np.minimum(dist, d)
        idx.append(int(np.argmax(dist)))
    return np.array(idx, dtype=np.int64)


def main():
    with open(SMPL_MODEL_PATH, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    v_template = np.asarray(model["v_template"], dtype=np.float64)   # (6890,3)
    weights = np.asarray(model["weights"], dtype=np.float64)         # (6890,24)
    nearest_joint = weights.argmax(axis=1)

    sel = {}
    for region, joints in REGIONS.items():
        mask = np.isin(nearest_joint, joints)
        verts = np.where(mask)[0]
        if len(verts) < POINTS_PER_REGION:
            raise ValueError(f"region {region}: only {len(verts)} vertices")
        picked = verts[farthest_point_sample(v_template[verts], POINTS_PER_REGION)]
        sel[region] = [int(v) for v in picked]

    out_npy = RESULTS_ROOT / "data/surface_sel35.npy"
    out_json = RESULTS_ROOT / "data/surface_sel35.json"
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray([v for region in REGIONS for v in sel[region]], dtype=np.int64)
    np.save(out_npy, indices)
    out_json.write_text(json.dumps(
        {"regions": REGIONS, "indices": sel,
         "order": [r for r in REGIONS], "source_model": str(SMPL_MODEL_PATH)},
        indent=1), encoding="utf-8")
    print(f"{out_npy} ({indices.shape})")
    print(f"{out_json}")


if __name__ == "__main__":
    main()
