"""续写 ctx 判别实验（2026-11-01）：v1 ckpt（ckpt_last）链式评估三臂。

full（原链，应复现 test.json）/ velzero（ctx 速度三维置零）/ gtctx（ctx=GT 锚）。
零训练成本、不改模型（forward 包装器注入）；只打印数字、不落任何仓库产物。

用法（queue 走 TASK=script 的 SCRIPT 行；直接跑亦可）：
  python results_display/Etest/ctx_discrim.py V2M --device cuda:0 [--arms full,gtctx]
  cond ∈ {T2M, V2M, VT2M}；--device 默认 cuda:0（尊重 worker 的
  CUDA_VISIBLE_DEVICES）；--arms 逗号分隔，默认 full,gtctx（可选 velzero）。

判读口径：
  full = v1 实测；velzero = 速度通道贡献；gtctx = 好锚点下链式机制上限。
  对照 Cv5 基线（test Δ，来自 v1 结果分析）：T2M PA≈89.4/RootOrient≈32.3°；
  V2M PA≈37.5/RO≈4.4°；VT2M PA≈40.5/RO≈4.9°。
"""
import argparse
import sys
import torch

sys.path.insert(0, "/data/fangyuxuan/projects/gait")
from pathlib import Path
from anysole.registry import get_builder
from anysole.data.dataset import AnySoleDataset, load_split_ids
from anysole.utils.eval_protocol import (
    _v7_ctx_session_seq, _session_metrics, _Accum, _pick_forward_axis)
from anysole.eval import _session_window_groups
from anysole.types import CONFIG_T, CONFIG_V, CONFIG_VT

ap = argparse.ArgumentParser()
ap.add_argument("cond", choices=("T2M", "V2M", "VT2M"))
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--arms", default="full,gtctx")
args = ap.parse_args()

COND = {"T2M": CONFIG_T, "V2M": CONFIG_V, "VT2M": CONFIG_VT}[args.cond]
ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
DEVICE = torch.device(args.device)

ck = torch.load(
    "results/AnySole/V7_Cv5Ctx_f6_soft_time_accel_state_smpl24_v1/"
    "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
    "checkpoints/ckpt_last.pt", map_location="cpu")
cfg = ck["config"]
tw, half = int(cfg["tw"]), int(cfg["tw"]) // 2

model = get_builder("V7_Cv5Ctx").build(cfg, None)
model.load_state_dict(ck["model"])
model.eval().to(DEVICE)

split_csv = Path(cfg["split_csv"])
test_ids = load_split_ids(split_csv, "test")
ds = AnySoleDataset(
    mode="eval", seq_root=Path(cfg["seq_root"]), split_csv=split_csv,
    cache_root=Path(cfg["cache_root"]), window_length=tw,
    session_ids=test_ids, contact_method=cfg["contact_method"],
    smpl_support_policy=str(cfg.get("smpl_support_policy", "exclude")),
    tactile_input=cfg.get("tactile_input", "raw108"),
    no_imu=bool(cfg.get("no_imu", False)), v_input=cfg.get("v_input", "hrnet"),
    f2_repr=bool(cfg.get("f2_repr", False)), smpl_roots=cfg.get("smpl_roots"),
    cam_id=int(cfg.get("cam_id", 3)), t_phys_v2=bool(cfg.get("t_phys_v2", False)),
    d8_heads=bool(cfg.get("d8_heads", False)), stride=half)
groups = _session_window_groups(ds)
fwd_axis = _pick_forward_axis(ds, DEVICE)
print("sessions=%d windows=%d cond=%s arms=%s device=%s"
      % (len(groups), sum(len(v) for v in groups.values()),
         args.cond, ARMS, args.device), flush=True)

KEYS = ("pa_mpjpe_mm", "mpjpe_mm", "global_mpjpe_mm", "root_ate_mm",
        "root_orientation_deg", "root_orientation_drift_deg",
        "jitter_pred_m_s3", "accel_error_m_s2")


def run_arm(name):
    accum = _Accum()
    contact = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.inference_mode():
        for n, (sid, idxs) in enumerate(groups.items()):
            seq, _ = _v7_ctx_session_seq(ds, model, sid, idxs, COND,
                                         DEVICE, half, {})
            _session_metrics(seq, tw, fwd_axis, COND, accum, contact)
            if (n + 1) % 6 == 0:
                print("  [%s] %d/%d" % (name, n + 1, len(groups)), flush=True)
    m = accum.means()
    print("\n== %s %s ==" % (args.cond, name))
    for k in KEYS:
        print("  %-26s %.3f" % (k, m.get(k, float("nan"))))
    return m


orig = model.forward


def _velzero(*a, **k):
    pc = k.get("prev_ctx")
    if pc is not None:
        pc = pc.clone()
        pc[..., 6:9] = 0.0
        k = {**k, "prev_ctx": pc}
    return orig(*a, **k)


gt_map = {}
for sid, idxs in groups.items():
    lst, prev_start = [], None
    for i in idxs:
        raw = ds[i]
        start = int(raw["frame_start"])
        reset = prev_start is None or start != prev_start + half
        lst.append(torch.zeros(10) if reset else torch.cat(
            [raw["pose_gt"][0, :6], raw["vel_gt"][0, :3], torch.ones(1)]))
        prev_start = start
    gt_map[sid] = lst
pos = {sid: 0 for sid in groups}


def _gtctx(*a, **k):
    sid = a[4][0]
    pc = gt_map[sid][pos[sid]].to(DEVICE).unsqueeze(0)
    pos[sid] += 1
    return orig(*a, **{**k, "prev_ctx": pc})


if "full" in ARMS:
    run_arm("full")
if "velzero" in ARMS:
    model.forward = _velzero
    run_arm("velzero")
    model.forward = orig
if "gtctx" in ARMS:
    model.forward = _gtctx
    run_arm("gtctx")
