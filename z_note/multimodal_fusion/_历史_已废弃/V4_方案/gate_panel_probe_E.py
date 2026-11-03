#!/usr/bin/env python3
"""V4_Av1 判据②的载体闸门：E 载体面板探针复测（2026-10-19，④ 前置测试）。

薄封装 `z_note/probes/probe_info_panel.py`（共享探针管线，勿改）——在 E v2.1
ckpt_best 上重测命题 1/2 的线性增量，为 V4_Av1 判据②（非对齐位置口径）提供
载体级期望值。

复测结论（2026-10-19，已落盘 `results/AnySole/probes/probe_info_panel_E.json`）：
- 命题 1：GlobalMPJPE −76.66* / −78.51*、traj_ATE −73.37*（B 载体为
  −43/−45/−44）——T 全局/世界系线性增量在 E 载体上更强；
- 命题 2：PA 路径踝足翻正（+6.44*，B 为 −1.96*），非对齐位置口径保留负
  信号（MPJPE_anklefoot 路径 B −4.54*、MPJPE_r_foot 路径 B −4.54*）——
  V4_Av1 判据② 用非对齐位置口径的依据。

用法（仓库根目录，touch_gait 环境）：
  python z_note/multimodal_fusion/V4_方案/gate_panel_probe_E.py            # 全量
  python z_note/multimodal_fusion/V4_方案/gate_panel_probe_E.py --smoke    # 冒烟 2 会话
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/data/fangyuxuan/projects/gait")
E_CKPT = (REPO / "results/AnySole/V3_Ev1_f6_soft/"
          "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
          "checkpoints/ckpt_best.pt")
PROBE = REPO / "z_note/probes/probe_info_panel.py"


def main(argv=None):
    args = list(argv) if argv is not None else sys.argv[1:]
    cmd = [sys.executable, str(PROBE), "--ckpt", str(E_CKPT), "--device",
           "cuda:3", "--tag", "E"]
    if "--smoke" in args:
        cmd += ["--max-sessions", "2"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
