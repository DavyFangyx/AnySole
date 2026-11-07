"""Cv5VG 一折（fS5S6S11）g_v/g_t 随遮挡轴诊断（2026-11-07，用户问"0遮挡下
视觉退化了"的排查）。

目的：
  1) 验证 g_v 在 V-only 行是否严格钉死 1.0（_route_v_gate 的 t_present 掩码）。
  2) 读出 g_v / g_t 在 VT 行随 occ000→100 的 median/mean 曲线，判断 V 门
     是否已经在 clean 档闭合、以及是否随遮挡朝正确方向响应。

复用 e6_gate_values_occlusion.py 的 build_model / build_loader。

用法（touch_gait env，仓库根；需要 GPU 或退化到 CPU）：
  python results_display/script/vg_gate_occlusion_check.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from results_display.script.e6_gate_values_occlusion import (  # noqa: E402
    build_model, build_loader)
from anysole.train import condition_inputs, move_batch  # noqa: E402
from anysole.types import CONFIG_VT, CONFIG_V  # noqa: E402

CELL = "V8_Cv5VG_fS5S6S11_s1"
LEVELS = [("000", "anysole/configs/v1_fold_S5S6S11.yaml")] + [
    ("%03d" % n, "anysole/configs/v1_occ%03d_fold_S5S6S11.yaml" % n)
    for n in (20, 40, 60, 80, 100)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    net, ckpt_path, t_phys_v2 = build_model(CELL, device)
    print("ckpt:", ckpt_path)

    for level, cfg_path in LEVELS:
        loader = build_loader(cfg_path, "test", batch_size=32, t_phys_v2=t_phys_v2)
        gv_list, gt_list, gv_v_only = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                n = batch["V_feat"].shape[0]
                cid_vt = torch.full((n,), CONFIG_VT, dtype=torch.long, device=device)
                v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid_vt)
                out = net(v_feat, t_raw, t_phys, cid_vt, batch.get("subject_ids"), T_s2m=t_s2m)
                gv = out.get("route_v_gate")
                gt = out.get("route_gate")
                if gv is not None:
                    gv_list.append(gv[..., 0].cpu().numpy().reshape(-1))
                if gt is not None:
                    gt_list.append(gt[..., 0].cpu().numpy().reshape(-1))

                cid_v = torch.full((n,), CONFIG_V, dtype=torch.long, device=device)
                v_feat2, t_raw2, t_phys2, t_s2m2 = condition_inputs(batch, cid_v)
                out2 = net(v_feat2, t_raw2, t_phys2, cid_v, batch.get("subject_ids"), T_s2m=t_s2m2)
                gv2 = out2.get("route_v_gate")
                if gv2 is not None:
                    gv_v_only.append(gv2[..., 0].cpu().numpy().reshape(-1))

        gv_all = np.concatenate(gv_list) if gv_list else np.array([])
        gt_all = np.concatenate(gt_list) if gt_list else np.array([])
        gv_vonly_all = np.concatenate(gv_v_only) if gv_v_only else np.array([])
        print(f"occ{level}: "
              f"g_v(VT) mean={gv_all.mean():.4f} median={np.median(gv_all):.4f} "
              f"p05={np.percentile(gv_all,5):.4f} p95={np.percentile(gv_all,95):.4f} | "
              f"g_t(VT) mean={gt_all.mean():.4f} median={np.median(gt_all):.4f} | "
              f"g_v(V-only) max|dev-1|={np.abs(gv_vonly_all-1.0).max() if gv_vonly_all.size else float('nan'):.6f}")


if __name__ == "__main__":
    main()
