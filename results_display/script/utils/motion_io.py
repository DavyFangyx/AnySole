"""Re-export shim: the canonical unified motion reader lives in
``Baselines/utils/motion_io.py`` (2026-09-26 结构裁定：基线侧数据协议
统一放入 Baselines/utils)。

T1 数据适配后（2026-09-28）补充 display 侧 GT 读取：旧 per-session
``align_meta.json`` 已废弃，新契约下时间网格在 ``frames.npz`` 的
visual_time_s/mocap_time_s，SMPL/BVH 路径在 ``session.json`` 的
source_files URI——``load_session_gt`` 在无 align_meta.json 时改走
shared facts（与 anysole/data/workspace_adapter.py 同源）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]  # utils/ -> script/ -> results_display/ -> root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Baselines.utils.motion_io import *  # noqa: F401,F403
from Baselines.utils.motion_io import LEGACY_BVH_NAMES  # noqa: F401

# 保存冻结侧原实现（供旧布局回退），随后用 shared-facts 版本覆盖同名。
_baselines_load_session_gt = load_session_gt


def load_session_gt(seq_dir: Path, n_frames: int, fps: float = 40.0) -> dict:
    """Display GT reader, shared-facts contract first (见模块 docstring)。

    Legacy layout (align_meta.json present) keeps the Baselines-side
    original implementation; this definition only shadows it when the old
    file is gone."""
    if (Path(seq_dir) / "align_meta.json").is_file():
        return _baselines_load_session_gt(seq_dir, n_frames, fps)
    from anysole.data.workspace_adapter import load_shared_session, shared_bvh_path, shared_smpl_path
    shared = load_shared_session(Path(seq_dir).name)
    mocap_t = np.asarray(shared["frames"]["mocap_time_s"], dtype=np.float64)
    if n_frames > len(mocap_t):
        mocap_t = mocap_t[:1] + np.arange(n_frames, dtype=np.float64) / float(fps)
    else:
        mocap_t = mocap_t[:n_frames]
    try:
        smpl = shared_smpl_path(shared)
        result = load_motion(smpl, query_t=mocap_t)
        result["source_path"] = str(smpl)
    except FileNotFoundError:
        bvh = shared_bvh_path(shared)
        result = load_motion(bvh, query_t=mocap_t)
        result["source_path"] = str(bvh)
    result["t_mocap"] = mocap_t
    return result
