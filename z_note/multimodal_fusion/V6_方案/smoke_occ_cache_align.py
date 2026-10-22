#!/usr/bin/env python3
"""V6_Av1 occ_mix 的数据级闸门：cam3 vs occ010 特征缓存逐帧对齐审计（① 前置测试）。

occ_mix（批量级缓存混合）的前置条件 = 双缓存同帧轴、同窗口网格、可同窗替换。
本脚本核对：会话集合一致、逐会话帧数一致、occ 缓存确有特征差异（非 cam3 复制）。

结论（2026-10-19 实测）：140/140 会话零缺失、帧数全等、全部特征确有差异
（diff 幅度 0.01~0.3，符合 bbox 底部 10% 遮挡预期）。

用法（仓库根目录，touch_gait 环境）：
  python z_note/multimodal_fusion/V6_方案/smoke_occ_cache_align.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

BASE = (REPO / "AnysoleWorkspace/model_inputs/AnySole/adapter_v1/visual/hrnet")
A, B = BASE / "cam3", BASE / "cam3_occ_lower010"


def main():
    sa = {os.path.basename(f) for f in glob.glob(str(A / "S*.pt"))}
    sb = {os.path.basename(f) for f in glob.glob(str(B / "S*.pt"))}
    print("cam3: %d sessions, occ010: %d sessions" % (len(sa), len(sb)))
    print("cam3-only: %d, occ-only: %d, common: %d" % (len(sa - sb), len(sb - sa), len(sa & sb)))
    mismatch = identical = checked = 0
    for s in sorted(sa & sb):
        a = torch.load(A / s, map_location="cpu")
        b = torch.load(B / s, map_location="cpu")
        checked += 1
        if a.shape != b.shape:
            mismatch += 1
            print("SHAPE MISMATCH", s, a.shape, b.shape)
            continue
        if (a - b).abs().max().item() == 0:
            identical += 1
    print("checked %d: shape mismatch %d, identical-features %d" % (checked, mismatch, identical))
    ok = not (sa - sb) and not (sb - sa) and mismatch == 0 and identical == 0
    print("GATE: %s" % ("PASS (occ_mix 双缓存可同窗替换)" if ok else "FAIL"))


if __name__ == "__main__":
    main()
