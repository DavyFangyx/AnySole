#!/usr/bin/env python3
"""修改 15 阶段 2：Cv5 + V 门（T 门 Cv5），冻结主干只训 V 门。

对每折：
1. 载入 V8_Cv5 折 ckpt 到 V8_Cv5VIG（新增 route_v_mlp，strict=False）；
   冻结全部参数，只开 route_v_mlp。
2. val × clean+occ 上算逐窗 w*（输出空间 V/T 插值最优权重，网格搜索；
   门乘在 V 支路的语义 = g=1 全 V、g=0 只剩 T，w* 是门的一一对应目标）；
   000/020 健康档目标钉 1（clean 税归零，T 伤害由 T 门负责）。
3. 用 val 的 v_tok/t_tok 训练 V 门（target=w*，MSE）。
4. (α,β) 校准（val 输出空间代理）：g_c = 2σ(αz+β)，000 税 ≤ +0.5、
   020 税 ≤ +1.0 硬约束，最大化全档收益；折进 route_v_mlp 末层。
5. test 上报：逐档 (1−g)T+gV 相对 VT 的兑现、clean 税、与按档常数 w 的对比。
6. 存 checkpoint（主干 + V 门，config 含 model_name 供 anysole.eval 消费）。

用法（touch_gait env，仓库根；需要 GPU）：
  python results_display/script/stage2_train_gate.py --fold fS5S6S11 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "results_display/script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ab_decomp import run_rows  # noqa: E402
from e6_gate_values_occlusion import build_loader, build_model  # noqa: E402
from pa_grid_oracle import const_w, w_oracle  # noqa: E402
from stage2_gate_pilot import spearman, win_lower_pa  # noqa: E402
from anysole.models.v8_cv5vig import build as build_vig  # noqa: E402

LEVELS = ("000", "020", "040", "060", "080", "100")
OUT = REPO / "results/diagnostics/20261107_stage2_vig"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="fS5S6S11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    fold_yaml = args.fold[1:] if args.fold.startswith("f") else args.fold
    cell = "V8_Cv5_%s_s1" % args.fold

    # 基座 ckpt → VIG（只多 route_v_mlp）
    base_net, ckpt, tp = build_model(cell, device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    src_cfg = payload.get("config", {}) or {}
    cfg = {"d_model": 256, "tw": 20, "dropout": 0.1, "pose_layers": 6,
           "v7_task_routes": True, "v7_root_sd": True,
           "v7_v_gate": True, "v7_gate_2sigma": False,
           "v7_trans_anchor_head": True}
    net = build_vig(cfg).to(device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print("[ckpt] %s  missing=%s unexpected=%s" % (ckpt, missing, unexpected))
    net.eval()
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith("route_v_mlp"))
    trainable = [p for p in net.parameters() if p.requires_grad]
    print("[trainable] %d tensors" % len(trainable))

    # 收集 split 的 w* / 按档常数 w 与 v_tok/t_tok
    rows = {}
    for level in levels:
        cfg_path = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                    else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = build_loader(cfg_path, args.split, args.batch_size, t_phys_v2=tp)
        r = run_rows(net, loader, device, args.split, level)
        w_raw, _ = w_oracle(r["p_v"], r["p_t"], r["gt"])
        w_l, _ = const_w(r["p_v"], r["p_t"], r["gt"])
        # 健康档（000/020）目标钉 1：clean 税必须归零，V 门不动作，
        # clean 的 T 伤害由 T 门负责；只有 040+ 才学逐窗 w*。
        w_star = np.ones_like(w_raw) if level in ("000", "020") else w_raw
        rows[level] = {"v_tok": r["v_tok"], "t_tok": r["t_tok"],
                       "w_star": w_star, "w_const": w_l,
                       "e_vt": r["e_vt"], "p_v": r["p_v"], "p_t": r["p_t"],
                       "gt": r["gt"]}
        print("[%s/%s] n=%d w*=%s%.2f (raw %.2f) w_const=%.2f"
              % (args.split, level, len(w_star),
                 "PIN1 " if level in ("000", "020") else "",
                 float(w_star.mean()), float(w_raw.mean()), w_l))

    # 训练 V 门：val 全部档；target = 逐窗 w*；MSE
    xt = torch.from_numpy(np.concatenate([rows[l]["v_tok"] for l in levels])).float().to(device)
    tt = torch.from_numpy(np.concatenate([rows[l]["t_tok"] for l in levels])).float().to(device)
    yt = torch.from_numpy(np.concatenate([rows[l]["w_star"] for l in levels])).float().to(device)
    level_ids = np.concatenate([np.full(len(rows[l]["w_star"]), l) for l in levels])
    clean_mask = torch.from_numpy(np.isin(level_ids, ("000", "020"))).to(device)
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        opt.zero_grad()
        # 直接调 _route_v_gate（窗口级；t_present=1）
        g_v = net._route_v_gate(xt[:, None, :], tt[:, None, :],
                                torch.ones(len(xt), device=device))
        pred = g_v.mean(dim=(1, 2, 3))
        loss = F.mse_loss(pred, yt)
        # 健康档 margin：z ≥ 2.0（g 饱和进 1），为 test OOD 左尾制造余量；
        # 与 (α,β) 校准形成两个杠杆。
        if clean_mask.any():
            z = net.route_v_mlp(net._gate_input(xt[:, None, :], tt[:, None, :])).squeeze(-1)
            loss = loss + 2.0 * F.relu(2.0 - z[clean_mask]).mean()
        loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print("[train] ep=%d loss=%.4f g_med=%.3f w_med=%.3f"
                  % (ep + 1, float(loss), float(pred.detach().median()),
                     float(yt.median())))
    net.eval()
    with torch.no_grad():
        g_all = net._route_v_gate(xt[:, None, :], tt[:, None, :],
                                  torch.ones(len(xt), device=device)).mean(dim=(1, 2, 3))
    print("[train] final g median=%.3f vs w*=%.3f" % (float(g_all.median()), float(yt.median())))

    # (α,β) 校准（val 输出空间代理）：g_c = 2σ(αz+β)，z = route_v_mlp 输出。
    # 约束：000 税 ≤ +0.5、020 税 ≤ +1.0（硬罚 100×），目标 = 最大化全档
    # (e_vt − e_mix)。校准折进 route_v_mlp 末层（W←αW, b←αb+β），
    # 模型结构不变；test 窗的 clean 左尾被抬进 sigmoid 饱和区。
    for level in levels:
        xl = torch.from_numpy(rows[level]["v_tok"]).float().to(device)
        tl = torch.from_numpy(rows[level]["t_tok"]).float().to(device)
        with torch.no_grad():
            z = net.route_v_mlp(net._gate_input(xl[:, None, :], tl[:, None, :])).squeeze(-1)
        rows[level]["z"] = z.cpu().numpy()
    best = None
    for alpha in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            total = 0.0; taxes = {}
            for level in levels:
                g_l = np.minimum(2.0 / (1.0 + np.exp(-(alpha * rows[level]["z"] + beta))), 1.0)
                mix = ((1.0 - g_l)[:, None, None, None] * rows[level]["p_t"]
                       + g_l[:, None, None, None] * rows[level]["p_v"])
                e_l = float(win_lower_pa(mix, rows[level]["gt"]).mean())
                total += e_l
                taxes[level] = e_l - float(np.mean(rows[level]["e_vt"]))
                if level == "000" and taxes[level] > 0.5:
                    total += 100.0 * (taxes[level] - 0.5)
                if level == "020" and taxes[level] > 1.0:
                    total += 100.0 * (taxes[level] - 1.0)
            if best is None or total < best[0]:
                best = (total, alpha, beta, taxes)
    _, alpha, beta, taxes = best
    with torch.no_grad():
        last = net.route_v_mlp[-1]
        last.weight.mul_(alpha)
        last.bias.mul_(alpha).add_(beta)
    print("[calib] alpha=%.2f beta=%.2f taxes={%s}"
          % (alpha, beta, ", ".join("%s:%+.1f" % (l, taxes[l]) for l in levels)))

    # test 上评门兑现：(1−g)T + gV vs VT；按档常数 w 作参照
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"fold": args.fold, "ckpt": str(ckpt),
               "calib": {"alpha": alpha, "beta": beta}, "per_level": []}
    for level in levels:
        cfg_path = ("anysole/configs/v1_fold_%s.yaml" % fold_yaml if level == "000"
                    else "anysole/configs/v1_occ%s_fold_%s.yaml" % (level, fold_yaml))
        loader = build_loader(cfg_path, "test", args.batch_size, t_phys_v2=tp)
        r = run_rows(net, loader, device, "test", level)
        w_star_t, _ = w_oracle(r["p_v"], r["p_t"], r["gt"])
        xg = torch.from_numpy(r["v_tok"]).float().to(device)
        tg = torch.from_numpy(r["t_tok"]).float().to(device)
        with torch.no_grad():
            g_v = net._route_v_gate(xg[:, None, :], tg[:, None, :],
                                    torch.ones(len(xg), device=device)).mean(dim=(1, 2, 3)).cpu().numpy()
        mix = ((1.0 - g_v)[:, None, None, None] * r["p_t"]
               + g_v[:, None, None, None] * r["p_v"])
        e_mix = win_lower_pa(mix, r["gt"])
        w_l = rows[level]["w_const"]
        lc = (1.0 - w_l) * r["p_t"] + w_l * r["p_v"]
        e_lc = win_lower_pa(lc, r["gt"])
        e_vt = r["e_vt"]
        metrics["per_level"].append({
            "level": level, "n": len(e_vt),
            "w_star_mean_test": float(w_star_t.mean()),
            "gate_mean": float(g_v.mean()),
            "gate_p05": float(np.percentile(g_v, 5)),
            "gate_p95": float(np.percentile(g_v, 95)),
            "spearman_g_w": spearman(g_v, w_star_t),
            "e_vt": float(e_vt.mean()), "e_mix": float(e_mix.mean()),
            "e_levelconst": float(e_lc.mean()),
            "gain_vig": float(np.mean(e_vt - e_mix)),
            "gain_levelconst": float(np.mean(e_vt - e_lc)),
            "clean_tax": float(np.mean(e_mix - e_vt)) if level == "000" else None})
        print("[test/%s] g=%.2f [p05 %.2f p95 %.2f] w*=%.2f rho=%+.2f | e_vt=%.1f e_gate=%.1f e_lc=%.1f | gain=%+.1f lc=%+.1f"
              % (level, metrics["per_level"][-1]["gate_mean"],
                 metrics["per_level"][-1]["gate_p05"],
                 metrics["per_level"][-1]["gate_p95"],
                 metrics["per_level"][-1]["w_star_mean_test"],
                 metrics["per_level"][-1]["spearman_g_w"],
                 metrics["per_level"][-1]["e_vt"],
                 metrics["per_level"][-1]["e_mix"],
                 metrics["per_level"][-1]["e_levelconst"],
                 metrics["per_level"][-1]["gain_vig"],
                 metrics["per_level"][-1]["gain_levelconst"]))
    ckpt_out = out_dir / ("V8_Cv5VIG_%s.pt" % args.fold)
    vig_cfg = dict(src_cfg)
    vig_cfg.update({"model_name": "V8_Cv5VIG_%s" % args.fold,
                    "v7_v_gate": True, "v7_gate_2sigma": False,
                    "lambda_tanchor": 1.0})
    torch.save({"model": net.state_dict(), "config": vig_cfg, "fold": args.fold},
               ckpt_out)
    (out_dir / ("metrics_%s.json" % args.fold)).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[out] %s" % ckpt_out)


if __name__ == "__main__":
    main()
