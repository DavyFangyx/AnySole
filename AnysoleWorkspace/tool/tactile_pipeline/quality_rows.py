"""帧级 valid/fake 掩码的行级质量数据（管线自带的 mask 源）。

每段录制一份 npz（``data/quality_rows/<date>/<subject>/<recording>.npz``）：
``{left,right}_{t_us,raw_t_us,valid,fake}``。值列不含（值是清洗链的
``processed48``），本模块只供对齐步推导帧级掩码。

来源：旧 PressureWasher mark-fake 链的行级 valid/fake 标记（2026-08；
代码已于 2026-10-24 删除），2026-10-24 吸收为本管线的数据工件，见
``data/quality_rows/provenance.json``。质量包已由生产端烤进 dataset npz
（2026-10-24 定妆照），消费端不读本模块。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 生产端包内相对路径（包根/data/quality_rows，随包分发）；消费端不再持有
# quality_rows（已烤进 dataset npz）。
QUALITY_ROWS_REL = Path("data") / "quality_rows"
QUALITY_ROWS_ROOT = Path(__file__).resolve().parent / QUALITY_ROWS_REL


def load_quality_rows(date: str, subject: str, recording: str, *, root=None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-side ``(t_us, raw_t_us, valid, fake)`` row packs.

    ``root`` = 包根（重建时传 --package-root；缺省 = 本模块所在的包根）。
    """
    base = (Path(root) / QUALITY_ROWS_REL) if root is not None else QUALITY_ROWS_ROOT
    path = base / date / subject / f"{recording}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing quality rows: {path}")
    data = np.load(path)
    out = {}
    for side in ("left", "right"):
        out[side] = (data[f"{side}_t_us"], data[f"{side}_raw_t_us"],
                     data[f"{side}_valid"], data[f"{side}_fake"])
    return out
