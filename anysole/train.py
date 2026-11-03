"""Train AnySole V1."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anysole.data.dataset import (
    AnySoleDataset,
    collate_windows,
    load_split_ids,
    sample_config_ids,
)
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.losses import (
    compute_losses,
    compute_losses_v3f,
    compute_losses_v4_1,
    compute_losses_v5_1,
    per_sample_contact_bce,
)
from anysole.models import AnySoleModel, MODEL_NAMES, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2
from anysole.types import (
    CONFIG_PROBS,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    DEFAULT_CONFIG_PATH,
    FOOT_JOINTS,
    GAIT_ROOT,
    HRNET_CACHE_ROOT,
    JOINT_PROTOCOL_CHECKSUM,
    KNEE_ANKLE_FOOT_JOINTS,
    MOTION_PROTOCOL,
    N_JOINTS,
    FPS,
    PART_NAMES,
    POSE_DIM,
    SEQ_ROOT,
    SMPL_ROOTS,
    SPLIT_CSV,
    STATE_DIM,
    T_S2M_DIM,
    WORKSPACE_ROOT,
    anysole_model_dir,
    assert_batch_shapes,
    variant_from_config,
)
from anysole.registry import get_builder, infer_anysole_paths
from anysole.data.workspace_adapter import ADAPTER_VERSION, sha256_file
from AnysoleWorkspace.tool.workspace import resolve_uri
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.utils.geometry import (
    ar2_to_world, compose_rot6d, f2_to_world, fk_pose6d,
    rot6d_to_rotmat, rot_root6d)
from anysole.utils.metrics import pa_mpjpe
from anysole.utils.constraints import apply_v3e_constraints
from anysole.utils.constraints_g import apply_v3g_constraints
from anysole.utils.constraints_h import apply_v3h
from anysole.models.common.model_v3f import _split_state, sample_v3f
from anysole.models.common.model_v3h import apply_foot_residual
from anysole.models.common.model_v5_1 import v5_1_z_src
from anysole.data.occ_mix import SRC_CAM3, SRC_OCC, grouped_mean
from anysole.utils.losses import soft_contact_from_keypoints
from anysole.types import CONFIG_NAMES
from anysole.utils.checkpoint_policy import CandidateArchive, DashboardUnavailable, infer_role
from anysole.utils.v5_conditioning import apply_availability, conditioning_policy, require_fresh_output


DEFAULT_CONFIG = {
    "d_model": 256,
    "dropout": 0.1,
    "tw": 20,
    "batch_size": 256,
    # Native SMPL-24 reinitializes the pose/traj heads when warm-starting a
    # BVH-23 checkpoint.  A real-batch F4 fixed-batch probe showed 1e-3
    # increases full MPJPE even with 20-step warmup; 1e-4 is the verified
    # stable starting rate for this protocol.
    "lr": 1.0e-4,
    "epochs": 200,
    "num_workers": 4,
    "ckpt_interval": 5,         # ckpt_last 保存间隔（epoch）；2026-11-01 训练提速：sdb 写回节流
    "smpl_support_policy": "exclude",  # new runs reject unsupported SMPL targets
    "diffusion_train_steps": 1000,
    "diffusion_sample_steps": 50,
    "pose_layers": 6,
    "denoise_layers": 4,        # V3_Fv1: denoiser transformer depth
    "v_cond_dropout": 0.1,      # V3_Fv1: V-condition dropout (the unconditional pass)
    # ---- E6.x 系列开关（默认值保持既有行为；E3 基础用 noise_scaled=false +
    # lr_schedule=constant，详见 model_fix_note/archived/E6x_series_plan.md）----
    "pose_repr": "6d",          # E6.1: "pos" = 根局部位置表示
    "stride": None,             # E6.3: 训练窗 stride（None = 非重叠）
    "lambda_pose_vel": 0.0,     # E6.2: 帧间平滑损失权重（0 = 关闭）
    "lambda_bone": 0.0,         # E6.2: 骨长刚性损失权重（0 = 关闭，仅 pos 模式）
    "continuation": False,      # E6.4: eval/infer 续写式推理
    "warm_start": False,        # E6.5: 采样起点对齐训练边缘分布
    "noise_scaled": False,      # E3 口径（E4 的噪声缩放被证伪）；默认与 anysole/configs/v1.yaml 对齐
    "lr_schedule": "constant",  # E3 口径（E5 的 cosine 被证伪）
    "t_phys_v2": False,  # 维度 2 显式物理特征（2026-10-21）：T_phys 12→20 维
    "d8_heads": False,  # 维度 3 足-地监督三头（2026-10-21）
    "lambda_tcop": 0.0,
    "lambda_strike": 0.0,
    "lambda_gtcop": 0.0,
    "lambda_pose": 3.0,
    "lambda_traj": 1.0,
    "lambda_trec": 0.1,
    "lambda_vrec": 0.1,
    "lambda_kp": 1.0,
    "lambda_con": 0.0,          # E3 口径（接触标签默认不参与训练损失）
    "traj_velocity_w": 1.0,
    "traj_delta_w": 1.0,
    "traj_delta_weight_power": 1.0,
    "traj_deltas": [2, 4, 8, 19],
    "config_probs": list(CONFIG_PROBS),
    "seed": 1,
    "cam_id": 3,
    # 路径一律引用 anysole/types.py 的集中常量，避免与配置文件的字面量双重维护。
    "seq_root": str(SEQ_ROOT),
    "split_csv": str(SPLIT_CSV),
    "cache_root": str(HRNET_CACHE_ROOT),
    "use_insole_drift": False,
    "contact_method": "f6_soft",
    "smpl_roots": [str(p) for p in SMPL_ROOTS],
    "modal": MODEL_ANYSOLEV1,
        "template_path": str(WORKSPACE_ROOT / "assets" / "insole_templates.json"),
    "wandb_mode": "disabled",
    "wandb_project": "Anysole",
    "wandb_entity": "davyfangyuxuan-nanjing-university-of-aeronautics-and-ast",
    "wandb_eval_interval": 5,
    "wandb_experiment_tag": "anysole_v1",
    # 在线监控后端（2026-11-03）：WANDB 访问受限后默认改投 SwanLab；
    # "wandb" 仍可选，行为与改造前一致。
    "monitor_backend": "swanlab",
    "swanlab_workspace": None,
}

# V5_Av1 z 通道监控键（epoch 均值；V5_A方案.md 判据③ 分歧审计 +
# G 前科接触读数塌缩）。batch 级累计进 epoch_sums，epoch 末打印
# 一行 + wandb 记 z/*；正式判读以 test 面板与分歧审计 JSON 为准。
_V5_Av1_Z_KEYS = (
    "z_div_contact",     # |sigmoid(logit_gT) − sigmoid(logit_gV)| 均值
    "z_div_phase",       # |μ_T − μ_V| 相位通道（phase_mask 掩码）
    "z_div_cop",         # |μ_T − μ_V| CoP 通道（cop_mask 掩码）
    "z_poe_prob_std",    # sigmoid(logit_poe) 批内 std（→0 = G 前科形态）
    "z_air_recall",      # PoE 接触读数的 air recall（G/H 前科诊断项）
    "z_logsig_gap",      # (log σ_T − log σ_V) 均值（精度分化方向）
)


def load_config(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read %s" % path) from exc
    loaded = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping: %s" % path)
    if "contact_method" not in loaded:
        raise ValueError(
            f"{path}: contact_method is required explicitly; choose an AnySole-private "
            "label method such as 'f6_soft'"
        )
    for key in ("seq_root", "split_csv", "cache_root"):
        value = loaded.get(key)
        if isinstance(value, str) and "://" in value:
            loaded[key] = str(resolve_uri(value))
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_observation_kwargs(batch: dict) -> dict:
    return {"obs_history": batch["obs_history"]} if "obs_history" in batch else {}


def condition_inputs(batch: dict, config_id: torch.Tensor, t_s2m_dim: int = T_S2M_DIM) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero absent modalities without modifying reconstruction targets in batch.

    Returns (v_feat, t_raw, t_phys, t_s2m); t_s2m is only consumed when the
    model was trained with tactile_input="s2m50" and is zeroed for V-only rows.
    ``t_s2m_dim`` selects the fallback width for hand-built batches that omit
    the E6.6 channel (50 by default, 38 for --no-imu checkpoints).
    """
    drop_v = (config_id == CONFIG_T).view(-1, 1, 1)
    drop_t = (config_id == CONFIG_V).view(-1, 1, 1)
    v_feat = torch.where(drop_v, torch.zeros_like(batch["V_feat"]), batch["V_feat"])
    # F1: zero the GVHMR channel for T-only rows (same rule as V_feat); the
    # batch dict is mutated so model call sites can pick it up directly.
    if batch.get("V_hmr") is not None:
        batch["V_hmr"] = torch.where(drop_v, torch.zeros_like(batch["V_hmr"]), batch["V_hmr"])
    t_raw = torch.where(drop_t, torch.zeros_like(batch["T_raw"]), batch["T_raw"])
    t_phys = torch.where(drop_t, torch.zeros_like(batch["T_phys"]), batch["T_phys"])
    t_s2m = batch.get("T_s2m")
    if t_s2m is None:
        # Manually-built batches (probes) may omit the E6.6 channel.
        t_s2m = t_raw.new_zeros((t_raw.shape[0], t_raw.shape[1], t_s2m_dim))
    else:
        t_s2m = torch.where(drop_t, torch.zeros_like(t_s2m), t_s2m)
    return v_feat, t_raw, t_phys, t_s2m


def _ar_rollout_curriculum(global_step: int, total_steps: int, tw: int,
                           l_max: int = 8, schedule_frac: float = 0.6) -> int:
    """修改 04 改动 A（2026-11-03）：rollout 长度课程——L 从 1 线性升到
    l_max（斜坡占 schedule_frac 总步数，其后固定 l_max），上界 tw−1
    （帧 0 永远 batch 携带 = 推理口径，不可 rollout）。"""
    if not 1 <= l_max < tw:
        raise ValueError("ar_rollout_max must be in [1, tw-1]")
    prog = min(1.0, global_step / max(1, int(total_steps * schedule_frac)))
    return min(1 + round((l_max - 1) * prog), tw - 1)


def _ar_ss_cache(out, prev_state):
    """AR SS 缓存（2026-11-01 修正）：帧 1..tw-1 = 模型自生成输出（批内
    同窗口同序列）；帧 0 = batch 携带反馈原样保留（连续窗口 = GT 前一帧、
    首窗 = 全零，与推理口径一致）——帧 0 的模型自生成反馈需要上一窗
    输出，批内不存在；此前帧 0 置零会覆盖连续首帧携带修复。"""
    bsz = out["x0_hat"].shape[0]
    tail = torch.cat([out["x0_hat"][:, :-1], out["v_hat"][:, :-1]],
                     dim=-1).detach()
    if prev_state is not None:
        first0 = prev_state[:, :1].detach()
    else:
        first0 = out["x0_hat"].new_zeros((bsz, 1, tail.shape[-1]))
    return torch.cat([first0, tail], dim=1)


def _ctx4_residual_batch(batch):
    """v4-a（2026-11-01）：T 行目标转锚点系残差（锚 = GT 窗首帧根朝向）。

    残差 pose = R_aᵀ⊙pose_gt（全 24 关节）、残差 vel = R_aᵀ⊙vel_gt、
    trans_gt 由残差 vel 重积分。V/VT 行不调用（目标保持绝对）。
    首窗/复位（ctx flag=0）→ R_a = I：首窗训练 = 绝对模式（与评估首窗
    R_a=I 同口径；2026-11-01 补齐 v4 方案 §4 的 flag 检查）。
    """
    b = dict(batch)
    R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])          # (B,3,3)
    ctx = batch.get("ctx")
    if ctx is not None:
        first = ctx[..., -1].mean(dim=1) <= 0.5                # (B,) 首窗/复位
        R_a[first] = torch.eye(3, device=R_a.device, dtype=R_a.dtype)
    RaT = R_a.transpose(-1, -2)
    # 只转根关节 6D（世界朝向）+ 根速度；非根关节为父相对局部旋转，
    # 不含绝对航向、不参与锚变换（2026-11-01 修正）。
    b["pose_gt"] = rot_root6d(RaT[:, None, :, :], batch["pose_gt"])
    b["vel_gt"] = torch.einsum("bij,btj->bti", RaT, batch["vel_gt"])
    b["trans_gt"] = torch.cumsum(b["vel_gt"], dim=1) / float(FPS)
    return b


def _t_prefix_rows(batch: dict, cid: torch.Tensor, tw: int, p: float):
    """V7 继承路线（2026-11-03）：T 档里以概率 p 把有观测历史的样本改成前缀继承样本。

    送模型：config 改 VT（历史窗口的 V 流保持真实、不被整流换成 null），当前窗的 V
    由 mask_v 整窗换成 null token，T 全程真实——「历史有视觉、当前窗只剩触觉」。
    损失侧仍按 T 行（调用方保留原 cid）：视觉航向项只对 V 在场行定义，而这类样本
    当前窗的 V 全是 null，不能让 vis_yaw / vis_gain 去拟合 null。无历史（会话首窗）
    的样本保持普通 T。返回 (model_cid, mask_v)；mask_v 为 (B, tw) bool，选中行整行 True。
    """
    history = batch.get("obs_history")
    if history is None:
        raise ValueError("--t-prefix-p requires obs_history in the batch (obs_context > tw)")
    has_history = torch.tensor([int(h["V_feat"].shape[0]) > 0 for h in history],
                               dtype=torch.bool, device=cid.device)
    chosen = has_history & (torch.rand(cid.shape[0], device=cid.device) < p)
    model_cid = cid.clone()
    model_cid[chosen] = CONFIG_VT
    mask_v = torch.zeros(cid.shape[0], tw, dtype=torch.bool, device=cid.device)
    mask_v[chosen] = True
    return model_cid, mask_v


def backward_v7_enum(model, batch, config, ctx_ss_p: float = 0.0,
                     ctx_cache=None, ar_ss_p: float = 0.0,
                     ar_noise: float = 0.0, ar_cache=None,
                     ar_rollout_len: int = 0):
    """Enumerate the same windows, accumulate gradients, then update once.

    Motion keeps config_probs weights. T-only auxiliary terms normalize over
    T-present modes, matching their masked-mean definition in sampled batches.
    Graphs are released after each backward, rather than retaining 3 batches.

    ``ar_rollout_len``（修改 04 改动 A，2026-11-03）：>0 且 model.ar2 时
    用闭环 rollout 前向替换本步模型调用（帧 1..L 反馈 = 模型自预测、
    帧 0 保留 batch 携带 = 推理口径；损失在全部 L 帧定义、分量沿用现有
    定义）。默认 0 = 原行为不变（改动全关 → AR2 逐位契约）。
    """
    probs = [float(x) for x in config["config_probs"]]
    if (len(probs) != 3 or not all(math.isfinite(x) and x >= 0 for x in probs)
            or sum(probs) <= 0):
        raise ValueError("enum requires three finite nonnegative config_probs")
    probs = [x / sum(probs) for x in probs]
    t_probability = probs[CONFIG_VT] + probs[CONFIG_T]
    mode_config = dict(config)
    for key in ("lambda_contact", "lambda_foot"):
        if key in mode_config:
            mode_config[key] = float(config[key]) / t_probability if t_probability > 0 else 0.0
    if ctx_ss_p > 0 and bool(getattr(model, "ctx3", False)):
        raise ValueError("--ctx-ss requires single-frame ctx (v1/v2); "
                         "ctx3/4/5 block ctx is unsupported")
    if ar_rollout_len > 0 and (ar_ss_p > 0 or ar_noise > 0):
        # 修改 04：闭环 rollout 与 scheduled sampling/收缩噪声是两种不同
        # 训练机制（SS=逐帧概率替换、rollout=连续 L 帧闭环），混用语义
        # 含糊——显式互斥而非静默叠加。
        raise ValueError("--ar-rollout is incompatible with --ar-ss/--ar-noise")
    # 纯回归收口（2026-11-02，v7_c2ar2r）：ar2_regression 下双向注意力、
    # 无因果掩码、无反馈（ar2_fb=False 本就不消费 prev_state）——回归格
    # 的 SS/噪声由 conf 置零，此处双保险关闭因果。
    ar_causal = (bool(getattr(model, "ar", False) or getattr(model, "ar2", False))
                 and not bool(config.get("ar2_regression", False)))
    # 缓存只在同一 batch 的枚举模式间复用（同窗口、同序列——scheduled
    # sampling 的替换来源与目标对齐）；跨 batch 必须清零（2026-11-01：
    # 跨 minibatch 复用 = 「另一段动作的历史预测当前动作目标」，不是
    # 当前序列上的 SS）。
    if ctx_cache is not None:
        ctx_cache[0] = None
    if ar_cache is not None:
        ar_cache[0] = None
    metrics = {}
    batch_size = batch["pose_gt"].shape[0]
    for mode, probability in enumerate(probs):
        if probability == 0:
            continue
        cid = torch.full((batch_size,), mode, dtype=torch.long, device=batch["pose_gt"].device)
        # 前缀继承样本（--t-prefix-p，默认关）：T 档里一部分有历史的样本送模型时改 VT +
        # 当前窗 mask_v；损失仍用原 cid（T 行）。关闭时 model_cid 即 cid，前向逐位不变。
        model_cid, prefix_mask_v = cid, None
        if mode == CONFIG_T and float(config.get("t_prefix_p", 0.0)) > 0.0:
            model_cid, prefix_mask_v = _t_prefix_rows(
                batch, cid, int(config["tw"]), float(config["t_prefix_p"]))
        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, model_cid)
        ctx = batch.get("ctx")
        if ctx is not None and ctx_cache is not None and ctx_ss_p > 0:
            # G1 scheduled sampling（续写 v2 实验矩阵）：以概率 p 用上一步
            # 缓存的「模型预测 ctx」替换 GT ctx——打破 teacher forcing 的
            # oracle 契约（零额外前向；缓存分布 = 模型生成分布）。
            if ctx_cache[0] is not None and torch.rand(1).item() < ctx_ss_p:
                c = ctx_cache[0].to(ctx.dtype)
                n = min(c.shape[0], ctx.shape[0])
                ctx = ctx.clone()
                ctx[:n] = c[:n]
        prev_state = batch.get("prev_state")
        if (getattr(model, "ar2", False) and not config.get("ar2_regression", False)
                and ar_ss_p > 0 and prev_state is not None):
            # AR2 同序列两遍式 scheduled sampling（2026-11-01，V7_AR方案
            # §5.1：禁无会话/帧位置匹配的 minibatch 缓存）：第一遍 GT 反馈
            # no_grad 生成同一序列的自反馈 [Δyaw+tilt/body+v_head]；第二遍
            # （梯度遍）以概率 p 替换（帧 0 保留 batch 携带反馈——真实前驱
            # 或首窗全零，与 v7_ar SS 同口径）。
            with torch.inference_mode():
                out_det = model(v_feat, t_raw, t_phys, cid,
                                batch.get("session_id"), T_s2m=t_s2m,
                                prev_ctx=ctx, prev_state=prev_state,
                                causal=True, **model_observation_kwargs(batch))
            cache = torch.cat([out_det["dyaw_hat"] + (
                prev_state[..., :1]
                if str(config.get("ar2_dyaw_base", "zero")) == "history"
                else 0.0), out_det["x0_hat"],
                               out_det["v_hat"]], dim=-1)      # (B,tw,148)
            if torch.rand(1).item() < ar_ss_p:
                prev_state = torch.cat([prev_state[:, :1],
                                        cache[:, :-1]], dim=1)
        elif (prev_state is not None and ar_cache is not None
                and ar_cache[0] is not None and torch.rand(1).item() < ar_ss_p):
            # AR 反馈 scheduled sampling（2026-11-01）：以概率 p 用上一步
            # 缓存的「模型自生成反馈」替换 GT 反馈（左移一格，帧 0 全零）。
            prev_state = ar_cache[0]
        if prev_state is not None and ar_noise > 0.0:
            # 反馈收缩正则：只扰姿态分量（归一化空间 ~O(1)），速度分量
            # (m/s) 不加噪。ar2：扰 tilt/body 切片 [1:145]；ar：[:144]。
            prev_state = prev_state.clone()
            sl = slice(1, 145) if getattr(model, "ar2", False) else slice(0, 144)
            prev_state[..., sl] = (
                prev_state[..., sl]
                + torch.randn_like(prev_state[..., sl]) * ar_noise)
        if ar_rollout_len > 0:
            if not (getattr(model, "ar2", False)
                    and getattr(model, "ar2_fb", False)
                    and prev_state is not None):
                # 死开关防御（2026-11-03）：rollout 要求带反馈的 ar2——
                # AR2M 无反馈支路 TF≡rollout，静默回退 = 假实验。
                raise ValueError("--ar-rollout requires ar2 with feedback "
                                 "(ar2_fb=True) and prev_state in the batch")
            # 修改 04 改动 A（2026-11-03）：闭环多步 rollout 前向——帧
            # 1..L 反馈 = 模型自预测（同序列、逐帧闭环，与 eval
            # ar2_session_chunks 更新式同构），帧 0 保留 batch 携带
            # （= 推理口径，连续窗口 GT 前驱/首窗零），帧 L+1.. 保持
            # GT 反馈（TF 尾帧）。损失在合并窗口上沿用现有分量定义。
            out = model.rollout_forward(
                v_feat, t_raw, t_phys, cid, batch.get("session_id"),
                T_s2m=t_s2m, prev_state=prev_state,
                rollout_len=ar_rollout_len,
                detach=bool(config.get("ar_rollout_detach", True)),
                dyaw_base=str(config.get("ar2_dyaw_base", "zero")),
                **model_observation_kwargs(batch))
        else:
            mask_kw = {"mask_v": prefix_mask_v} if prefix_mask_v is not None else {}
            out = model(v_feat, t_raw, t_phys, model_cid, batch.get("session_id"), T_s2m=t_s2m,
                        prev_ctx=ctx, prev_state=prev_state, causal=ar_causal,
                        **mask_kw, **model_observation_kwargs(batch))
        if (ar_cache is not None and not getattr(model, "ar2", False)
                and out.get("x0_hat") is not None):
            ar_cache[0] = _ar_ss_cache(out, prev_state)
        if ctx_cache is not None:
            bsz = out["x0_hat"].shape[0]
            cache_root = out["x0_hat"][:, 10, :6].detach()
            cache_vel = out["v_hat"][:, 10, :3].detach()
            if getattr(model, "ctx4", False) and mode == CONFIG_T:
                # ctx4 输出是残差——缓存必须组合回世界态（锚 = GT 窗首帧）。
                R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])
                cache_root = compose_rot6d(R_a, cache_root)
                cache_vel = torch.einsum("bij,bj->bi", R_a, cache_vel)
            ctx_cache[0] = torch.cat([
                cache_root, cache_vel,
                torch.ones(bsz, 1, device=out["x0_hat"].device)], dim=1)
        loss_batch = batch
        if getattr(model, "ctx4b", False):
            # v4-b：全条件目标 = 锚点系残差（锚 = GT 窗首帧根朝向；首窗
            # flag=0 → R_a=I 绝对模式）。
            loss_batch = _ctx4_residual_batch(batch)
        elif getattr(model, "ctx4", False) and mode == CONFIG_T:
            # v4-a：T 行目标 = 锚点系残差（模型输出残差，目标同空间）。
            loss_batch = _ctx4_residual_batch(batch)
        losses = compute_losses(out, loss_batch, cid, mode_config)
        if not bool(torch.isfinite(losses["loss"]) & torch.isfinite(out["F"]).all()):
            # The caller discards all accumulated gradients on failure.
            return out, losses
        (probability * losses["loss"]).backward()
        for key, value in losses.items():
            weight = probability
            if key in ("L_contact", "L_foot") and t_probability > 0:
                weight /= t_probability
            contribution = weight * value.detach()
            metrics[key] = metrics.get(key, 0.0) + contribution
    return out, metrics


def fit_pose_stats(loader: DataLoader, device: torch.device, key: str = "pose_gt",
                   residual_weight: float = 0.0,
                   first_abs_share: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """One pass over the training loader: per-dim mean/std of the pose target.

    The pose head normalizes its input/output with these fixed stats (MDM/RoHM
    normalize the motion representation before diffusing), so the values are
    fitted once on the training set and frozen for the whole run. ``key``
    selects "pose_gt" (6D) or "pose_gt_pos" (E6.1 positions).
    """
    sums = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    sumsq = torch.zeros(POSE_DIM, dtype=torch.float64, device=device)
    count = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            pose = raw_batch[key].to(device=device, dtype=torch.float64, non_blocking=True)
            if sums.shape[0] != pose.shape[-1]:
                sums = torch.zeros(pose.shape[-1], dtype=torch.float64, device=device)
                sumsq = torch.zeros(pose.shape[-1], dtype=torch.float64, device=device)
            flat = pose.reshape(-1, pose.shape[-1])
            if residual_weight > 0 or first_abs_share is not None:
                # 统计拟合混合口径（2026-11-01）：
                #   v4-a：绝对全窗权重 1 + 残差权重 residual_weight；
                #   v4-b：残差全窗权重 1（首窗降为 1−first_abs_share）+ 绝对
                #         仅首窗（T 行绝对模式）权重 first_abs_share = p_T。
                # 残差分量 = R_aᵀ 只转根关节（与训练目标变换同方向、同范围）。
                shape = pose.shape
                R_a = rot6d_to_rotmat(pose[:, 0, :6].float())          # (B,3,3)
                res = rot_root6d(R_a.transpose(-1, -2)[:, None, :, :].double(),
                                 pose)
                flat_res = res.reshape(-1, res.shape[-1])
                if first_abs_share is not None:
                    ctx = raw_batch.get("ctx")
                    if ctx is None:
                        raise RuntimeError(
                            "first_abs_share requires the ctx column "
                            "(v7_ctx dataset)")
                    first = (ctx[..., -1].mean(dim=1) <= 0.5).to(
                        device).double()                                # (B,)
                    w_abs = (first * first_abs_share).repeat_interleave(
                        shape[1])                                       # (B*tw,)
                    w_res = 1.0 - w_abs
                    sums += (flat * w_abs.view(-1, 1)).sum(dim=0)
                    sumsq += (flat.square() * w_abs.view(-1, 1)).sum(dim=0)
                    count += w_abs.sum()
                    sums += (flat_res * w_res.view(-1, 1)).sum(dim=0)
                    sumsq += (flat_res.square() * w_res.view(-1, 1)).sum(dim=0)
                    count += w_res.sum()
                else:
                    sums += flat.sum(dim=0)
                    sumsq += flat.square().sum(dim=0)
                    count += flat.shape[0]
                    sums += residual_weight * flat_res.sum(dim=0)
                    sumsq += residual_weight * flat_res.square().sum(dim=0)
                    count += residual_weight * flat_res.shape[0]
            else:
                sums += flat.sum(dim=0)
                sumsq += flat.square().sum(dim=0)
                count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit pose stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    return mean.float(), var.sqrt().float()


def fit_pressure_cell_means(loader: DataLoader, device: torch.device) -> Tuple[float, ...]:
    """One pass over the training loader: per-cell train mean of raw pressure
    (96 cells, grams).  V3_Bv2/Bv3 Q-NN/Q-FACT initialize their output bias
    from these values, so the zero-initialized readout starts at the training
    prior; fitted once at run start, then frozen in the checkpoint config.
    """
    sums = torch.zeros(96, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            t_raw = raw_batch["T_raw"].to(device=device, dtype=torch.float64, non_blocking=True)
            flat = t_raw.reshape(-1, t_raw.shape[-1])
            if flat.shape[-1] != 96:
                raise ValueError("pressure cell statistics require 96 raw cells")
            sums += flat.sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit pressure cell means: training loader is empty")
    mean = sums / count
    return tuple(float(x) for x in mean.cpu())


def fit_state_stats(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """V3_Fv1: per-component mean/std of the diffused state [pose|trans|contact].

    One pass over the training loader; the values normalize the full state to
    unit scale before diffusing (MDM convention) and are frozen for the run.
    """
    sums = torch.zeros(STATE_DIM, dtype=torch.float64, device=device)
    sumsq = torch.zeros(STATE_DIM, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            state = torch.cat(
                [raw_batch["pose_gt"], raw_batch["trans_gt"], raw_batch["contact_gt"]],
                dim=-1,
            ).to(device=device, dtype=torch.float64, non_blocking=True)
            flat = state.reshape(-1, STATE_DIM)
            sums += flat.sum(dim=0)
            sumsq += flat.square().sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit state stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    return mean.float(), var.sqrt().float()


def fit_traj_stats(loader: DataLoader, device: torch.device) -> Dict[str, list]:
    """One pass over the training loader: per-dim mean/std of traj_gt_f2 (4).

    F2a: the trajectory loss compares psi_dot / v_h / h in one normalized
    space, so the per-dim scales are balanced by construction.  Saved into
    the checkpoint config (traj_f2_stats) and consumed by losses.py.
    """
    dim = 4  # TRAJ_F2_DIM
    sums = torch.zeros(dim, dtype=torch.float64, device=device)
    sumsq = torch.zeros(dim, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            traj = raw_batch["traj_gt_f2"].to(device=device, dtype=torch.float64, non_blocking=True)
            flat = traj.reshape(-1, traj.shape[-1])
            sums += flat.sum(dim=0)
            sumsq += flat.square().sum(dim=0)
            count += flat.shape[0]
    if count == 0:
        raise RuntimeError("cannot fit traj stats: training loader is empty")
    mean = sums / count
    var = (sumsq / count - mean.square()).clamp(min=0.0)
    std = var.sqrt().clamp(min=1e-2)
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def init_from_checkpoint(model, path: Path, drop_prefixes=()) -> None:
    """Warm-start: copy matching parameters from another checkpoint's model.

    A key is copied only when it exists in the target state dict with the
    same shape; everything else (missing keys, shape changes, explicit
    ``drop_prefixes``) keeps the target's fresh initialization.  This makes
    anysolev1 -> anysolev2 warm-start work with no key mapping: the V2 pose
    head's ``query`` replaces the V1 ``proj_*``/``timestep_token`` (skipped
    by the shape/name rule), while encoders/fusion/traj/aux transfer as-is.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("--init-from checkpoint has no 'model' state dict: %s" % path)
    src = checkpoint["model"]
    src_pose_dim = int(getattr(src.get("pose_head.pose_mean"), "shape", (0,))[0])
    src_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    src_joints = int(src_config.get("motion_n_joints", src_pose_dim // 6 if src_pose_dim else 0))
    src_protocol = str(src_config.get("motion_protocol", ""))
    src_checksum = str(src_config.get("joint_protocol_checksum", ""))
    protocol_mismatch = (
        src_pose_dim != POSE_DIM
        or src_joints != N_JOINTS
        or src_protocol != MOTION_PROTOCOL
        or src_checksum != JOINT_PROTOCOL_CHECKSUM
    )
    effective_drop = tuple(drop_prefixes)
    if protocol_mismatch:
        # A motion head from another/unknown protocol is not merely a tensor
        # with a possibly different width: joint order, parent tree, body/leg
        # grouping and trajectory target are semantic state.  In particular,
        # early 24/144 SMPL migration checkpoints used a wrong collar parent
        # tree and therefore must not pass this gate on shape alone.
        # Shape-only loading would still copy most decoder and trajectory
        # tensors and silently retain that language.  Transfer only the
        # protocol-independent encoders/fusion/auxiliary heads.
        effective_drop += ("pose_head.", "traj_head.")
        print("=" * 79)
        reason = []
        if src_pose_dim != POSE_DIM or src_joints != N_JOINTS:
            reason.append("shape=%d-joint/%dD" % (src_joints, src_pose_dim))
        if src_protocol != MOTION_PROTOCOL:
            reason.append("motion_protocol=%r" % src_protocol)
        if src_checksum != JOINT_PROTOCOL_CHECKSUM:
            reason.append("joint-tree checksum=%s" % (src_checksum or "missing"))
        print("WARNING: --init-from uses a legacy or unknown motion protocol (%s)." %
              ", ".join(reason))
        print("         Protocol-specific pose query/decoder/output/stats and traj_head.* are reinitialized")
        print("         for SMPL-24/144D; shared embeddings plus encoder/fusion/aux may transfer.")
        print("=" * 79)
    dst = model.state_dict()
    loaded, skipped = [], []
    for key, value in src.items():
        if any(key.startswith(p) for p in effective_drop) or key not in dst or dst[key].shape != value.shape:
            skipped.append(key)
            continue
        dst[key] = value.to(dst[key].dtype)
        loaded.append(key)
    model.load_state_dict(dst)
    pose_skipped = [key for key in skipped if key.startswith("pose_head.")]
    if pose_skipped and not protocol_mismatch:
        print("WARNING: %d pose-head tensors were reinitialized because their name/shape did not match."
              % len(pose_skipped))
    print("init-from %s: %d/%d keys copied%s" % (
        path, len(loaded), len(dst),
        " (skipped: %s)" % ", ".join(skipped) if skipped else ""))


def warn_legacy_output_checkpoint(out_dir: Path) -> None:
    """Warn when a run directory contains a non-current motion checkpoint."""
    candidates = [Path(out_dir) / "ckpt_last.pt", Path(out_dir) / "ckpt_best.pt"]
    legacy = []
    v3f_ckpt = False
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            old = torch.load(candidate, map_location="cpu", weights_only=False)
            state = old.get("model", {}) if isinstance(old, dict) else {}
            dim = int(getattr(state.get("pose_head.pose_mean"), "shape", (0,))[0])
            cfg = old.get("config", {}) if isinstance(old, dict) else {}
            if cfg.get("v3f", False):
                # V3_Fv1 has no pose_head; its diffused state dim (149) is not
                # POSE_DIM and would false-positive the legacy protocol check.
                v3f_ckpt = True
                continue
            joints = int(cfg.get("motion_n_joints", dim // 6 if dim else 0))
            protocol = str(cfg.get("motion_protocol", ""))
            checksum = str(cfg.get("joint_protocol_checksum", ""))
            if (dim != POSE_DIM or joints != N_JOINTS
                    or protocol != MOTION_PROTOCOL
                    or checksum != JOINT_PROTOCOL_CHECKSUM):
                legacy.append(
                    "%s (%d joints/%dD, protocol=%r, tree=%s)"
                    % (candidate.name, joints, dim, protocol, checksum[:12] or "missing")
                )
        except Exception as exc:
            print("WARNING: could not inspect existing output checkpoint %s: %s" % (candidate, exc))
    if legacy or v3f_ckpt:
        print("=" * 79)
        if legacy:
            print("WARNING: output directory contains legacy checkpoints: %s" % ", ".join(legacy))
            print("         They are not resumable/evaluable under the current SMPL-24 joint tree")
            print("         and will be replaced by this run.")
        if v3f_ckpt:
            print("NOTE: output directory contains V3_Fv1 checkpoints (state-space model, no pose_head);")
            print("      the legacy SMPL-24 pose-head protocol check does not apply and is skipped.")
        print("         Until the first new checkpoint is saved, do not treat this directory as a new result.")
        print("=" * 79)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AnySole V1.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training RNG seed. Saved into the checkpoint config for registered experiments.",
    )
    parser.add_argument(
        "--tw",
        type=int,
        default=None,
        help="Window length in frames; overrides the config tw (e.g. --tw 100).",
    )
    parser.add_argument("--obs-context", type=int, default=None,
                        help="AR2 maximum observed frames; output length remains --tw.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Model dropout rate (0 disables dropout; used by Test9 overfitting and retrain experiments). Saved into the checkpoint config.",
    )
    parser.add_argument(
        "--config-probs",
        type=float,
        nargs=3,
        metavar=("VT", "V", "T"),
        default=None,
        help="Sampling probabilities for VT, V-only, and T-only configs.",
    )
    parser.add_argument(
        "--dropoutVT",
        dest="dropout_vt",
        type=str,
        nargs="+",
        metavar=("DROP_V_PCT", "DROP_T_PCT"),
        default=None,
        help="Modality dropout as percentages: V is dropped DROP_V_PCT%% of steps, T is dropped "
        "DROP_T_PCT%% (config_probs becomes [1-v-t, t, v] in VT/V/T order; overrides --config-probs). "
        "Accepts '--dropoutVT 20 30' or '--dropoutVT 20,30'; '0,0' disables modality dropout.",
    )
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--condition-sampling", choices=("sample", "enum"), default=None,
                        help="enum: each window sees VT/V/T, with config_probs loss weights and one optimizer update")
    parser.add_argument("--final-eval-split", choices=("val", "test", "none"), default=None,
                        help="Post-training evaluation split; new feasibility runs use val")
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="Upper bound for training-step sampling: tau ~ U(0, tau_max) instead of "
        "U(0, diffusion_train_steps). Test11 low-noise band training (e.g. --tau-max 100).",
    )
    parser.add_argument(
        "--tau-fixed",
        type=int,
        default=None,
        help="Fix tau to this value for every training step. Test11 tau0 identity training "
        "(--tau-fixed 0: x_tau = x0 clean input, task degenerates to the identity map).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="E6.3: training window stride (None = non-overlapping, stride == tw; "
        "1 = every-frame alignment, Step2Motion-style).",
    )
    parser.add_argument(
        "--part-json",
        type=Path,
        default=None,
        help="V4B only: load a learned hard partition (JSON, either a list "
        "of joint-index lists or the probe output with a 'partition' key of "
        "joint-name lists) and train the hard pose head with it. The "
        "partition is structural data for the V4B entry; other models reject "
        "it. Saved into the checkpoint config so eval/infer reconstruct it.",
    )
    parser.add_argument(
        "--lambda-assign",
        type=float,
        default=None,
        help="Weight of L_assign (assignment entropy + slot-load balance; "
        "only active for soft-A models, i.e. V3_3A/B / V3_4b/c / V4A). "
        "Default 0.05. V4A clustering passes 1.0 here and scales the three "
        "cluster terms with --lambda-assign-*.",
    )
    parser.add_argument(
        "--assign-cluster",
        action="store_true",
        help="V4A only (its entry fixes 24 slots + soft A): learn the part "
        "grouping from a near-uniform assignment. Replaces the V3-3 L_assign "
        "with annealed entropy + concentration reward + dead-slot tax; A gets "
        "its own optimizer lr (--assign-lr-mult) and is frozen after "
        "--assign-lock-frac of the run. Saved into the checkpoint.",
    )
    parser.add_argument(
        "--lambda-assign-ent", type=float, default=0.05,
        help="V4A: weight of the ANNEALED per-joint assignment entropy (early "
        "softness; anneals to 0 by --assign-anneal-frac).",
    )
    parser.add_argument(
        "--lambda-assign-conc", type=float, default=0.01,
        help="V4A: weight of the concentration reward (N_JOINTS^2 - sum mass^2, "
        ">= 0). Bigger = fewer, larger slots (K emerges from the recon-loss tension).",
    )
    parser.add_argument(
        "--lambda-assign-dead", type=float, default=0.02,
        help="V4A: weight of the dead-slot tax (sum exp(-beta*mass)) that "
        "rounds crumb loads to 0 so the effective K reads cleanly.",
    )
    parser.add_argument(
        "--assign-dead-beta", type=float, default=2.0,
        help="V4A: sharpness of the dead-slot tax exp(-beta*mass).",
    )
    parser.add_argument(
        "--assign-temp-init", type=float, default=1.0,
        help="V4A: assignment softmax temperature at step 0 (soft start).",
    )
    parser.add_argument(
        "--assign-temp-final", type=float, default=0.2,
        help="V4A: assignment softmax temperature at the end of the anneal "
        "(sharp). eval/readout restore this value from the checkpoint.",
    )
    parser.add_argument(
        "--assign-anneal-frac", type=float, default=0.7,
        help="V4A: fraction of total steps over which temperature and entropy "
        "anneal linearly to their final/zero values.",
    )
    parser.add_argument(
        "--assign-lock-frac", type=float, default=0.7,
        help="V4A: fraction of total steps after which part_logits (A) is "
        "frozen so the heads settle against a fixed grouping (M step).",
    )
    parser.add_argument(
        "--assign-lr-mult", type=float, default=5.0,
        help="V4A: optimizer lr multiplier for part_logits (A) relative to the "
        "rest of the model. Only active with --assign-cluster.",
    )
    parser.add_argument(
        "--lambda-sigma",
        type=float,
        default=None,
        help="V3-3: weight of the beta-NLL sigma supervision (plan §3, mixture-"
        "variance form; only active with --gate sigma). Default 0 (off); "
        "recommended 0.01 when enabling sigma calibration.",
    )
    parser.add_argument(
        "--sigma-freeze-frac",
        type=float,
        default=None,
        help="V3-3: fraction of total steps where sigma is frozen at 0 "
        "(gate runs on the tau_p prior bias alone, plan §3 'first 10%% sigma "
        "fixed'). Default 0.1 with --gate sigma.",
    )
    parser.add_argument(
        "--lr-warmup-frac",
        type=float,
        default=None,
        help="V3 structural steps (fix_plan_v3 §2.2): linear LR warmup over the "
        "first this-fraction of total steps (e.g. 0.05), multiplying the "
        "epoch cosine LR. 0/absent = no warmup (F0b/F4a behavior).",
    )
    parser.add_argument("--lambda-pose", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_pose.")
    parser.add_argument("--lambda-kp", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_kp.")
    parser.add_argument("--lambda-pos", type=float, default=None, metavar="W",
                        help="V3_Dv1: override yaml loss weight lambda_pos (position-head L_pos).")
    parser.add_argument("--lambda-traj", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_traj.")
    parser.add_argument("--lambda-trec", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_trec.")
    parser.add_argument("--lambda-vrec", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_vrec.")
    parser.add_argument("--lambda-con", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_con.")
    parser.add_argument("--lambda-contact", type=float, default=None, metavar="W",
                        help="V3_Ev1: override yaml loss weight lambda_contact (T contact head BCE).")
    parser.add_argument("--ctx-ss", type=float, default=0.0,
                        help="V7 续写 G1: scheduled sampling 目标概率 p"
                        "（线性升至 p 用模型预测 ctx 替换 GT ctx；0=关）。")
    parser.add_argument("--ar-ss", type=float, default=0.0,
                        help="V7 AR 反馈 scheduled sampling 目标概率 p"
                        "（线性升至 p 用模型自生成反馈替换 GT 反馈；0=关）。")
    parser.add_argument("--ar-noise", type=float, default=0.0,
                        help="V7 AR 反馈收缩正则：训练反馈姿态分量加高斯噪声"
                        "标准差（0=关）。")
    parser.add_argument("--ar-rollout", action="store_true", default=None,
                        help="V7 AR2 修改 04 改动 A：训练期闭环多步 rollout 损失"
                        "（rollout 长度 L 从 1 线性升到 L_max；默认关）。"
                        "与 --ar-ss/--ar-noise 互斥。")
    parser.add_argument("--ar-rollout-max", type=int, default=None, metavar="L",
                        help="V7 AR2 修改 04：rollout 长度上限 L_max（默认 8）。")
    parser.add_argument("--ar-rollout-schedule-frac", type=float, default=None,
                        metavar="F",
                        help="V7 AR2 修改 04：L 课程斜坡占总步数比例（默认 0.6，"
                        "剩余步数固定 L_max）。")
    parser.add_argument("--ar-rollout-detach", action="store_true", default=None,
                        help="V7 AR2 修改 04：rollout 反馈梯度截断（默认开；"
                        "只影响优化、不改反馈来源）。")
    parser.add_argument("--t-prefix-p", type=float, default=None, metavar="P",
                        help="V7 AR2 继承路线（2026-11-03）：enum 的 T 档里，有观测历史的样本"
                        "以概率 P 改成「历史窗口 V 真实、当前窗 V 换 null、T 全程真实」的前缀"
                        "继承样本（模型走 VT 配置 + 当前窗 mask_v，损失仍按 T 行；默认关）。")
    parser.add_argument("--v7-ar2-lowdim", action="store_true", default=None,
                        help="V7 AR2 修改 04 改动 B：反馈降维为低维相对状态"
                        "（Δyaw+根倾斜+body 投影+航向系速度，默认关）。")
    parser.add_argument("--v7-ar2-fb-dim", type=int, default=None, metavar="K",
                        help="V7 AR2 修改 04：body 投影维度 k（默认 16，"
                        "总反馈维 10+k ≤ 32）。")
    parser.add_argument("--lambda-nll", type=float, default=None, metavar="W",
                        help="V3_Ev1: override yaml loss weight lambda_nll (V pose sigma NLL).")
    parser.add_argument("--lambda-prior-pose", type=float, default=None, metavar="W",
                        help="V3_Ev1: override yaml loss weight lambda_prior_pose (T prior pose MSE).")
    parser.add_argument("--lambda-prior-nll", type=float, default=None, metavar="W",
                        help="V3_Ev1: override yaml loss weight lambda_prior_nll (T prior sigma NLL).")
    parser.add_argument("--lambda-prior-traj", type=float, default=None, metavar="W",
                        help="V3_Ev1: override yaml loss weight lambda_prior_traj (T prior trajectory).")
    parser.add_argument("--lambda-dyaw", type=float, default=None, metavar="W",
                        help="V7 AR2: Δyaw 帧级监督权重（mrad 口径）。")
    parser.add_argument("--lambda-cumyaw", type=float, default=None, metavar="W",
                        help="V7 AR2: 窗口累计 Δyaw 监督权重（mrad 口径）。")
    parser.add_argument("--lambda-dyaw-td", type=float, default=None, metavar="W",
                        help="V7 AR2: Δyaw 时间差分平滑权重（mrad 口径）。")
    parser.add_argument("--lambda-visyaw", type=float, default=None, metavar="W",
                        help="V7 AR2: 视觉航向观测监督权重（rad 口径，V 行）。")
    parser.add_argument("--lambda-yawcorr", type=float, default=None, metavar="W",
                        help="V7 AR2: 视觉校正链监督权重（rad 口径，V 行）。")
    parser.add_argument("--lambda-world-traj", type=float, default=None, metavar="W",
                        help="V7 AR2 yaw-fix: 世界系轨迹损失权重（m 口径，V/VT 行）。")
    parser.add_argument("--v7-ar2-yaw-fix", action="store_true", default=None,
                        help="V7 AR2 yaw-fix（2026-11-02 修订）：正面 Δψ 结构 "
                             "（历史延续基线 + 修正项 + T 行弱监督）+ L_yawcorr "
                             "自由生成口径 + 世界系轨迹损失。")
    parser.add_argument("--ar2-dyaw-base", type=str, default=None,
                        choices=("zero", "history"), metavar="MODE",
                        help="V7 AR2 yaw-fix: Δψ 基线来源（history=历史延续，"
                             "zero=零基线/旧语义；默认 zero）。")
    parser.add_argument("--ar2-dyaw-t-w", type=float, default=None, metavar="W",
                        help="V7 AR2 yaw-fix: T 行航向损失权重（默认 0.1）。")
    parser.add_argument("--lambda-air", type=float, default=None, metavar="W",
                        help="V3_Gv1: override yaml loss weight lambda_air (swing/air-frame contact BCE weight).")
    parser.add_argument("--lambda-foot-res", type=float, default=None, metavar="W",
                        help="V3_Hv3: override yaml loss weight lambda_foot_res "
                             "(first-order foot residual head).")
    parser.add_argument("--lambda-phase", type=float, default=None, metavar="W",
                        help="V5_Av1: override yaml loss weight lambda_phase (z 专家相位 μ "
                             "监督，掩码跳过；单点预注册 1.0).")
    parser.add_argument("--lambda-cop", type=float, default=None, metavar="W",
                        help="V5_Av1: override yaml loss weight lambda_cop (z 专家 CoP μ "
                             "监督，接触帧掩码；单点预注册 1.0).")
    parser.add_argument("--lambda-tcop", type=float, default=None,
                        help="维度 3：触觉 CoP 自重建权重")
    parser.add_argument("--lambda-strike", type=float, default=None,
                        help="维度 3：着地模式三分类权重")
    parser.add_argument("--lambda-gtcop", type=float, default=None,
                        help="维度 3：GT CoP 回归权重")
    parser.add_argument("--lambda-cond", type=float, default=None,
                        help="V5_Av5/V4_Av2: conditional contract weight "
                             "(V 健康窗口罚修正量，V 失效窗口自由；0 = 关).")
    parser.add_argument("--carrier-lr-mult", type=float, default=None,
                        help="V4_Av4: two-stage fine-tune — the B backbone "
                             "trains at this lr multiplier (0.1), other "
                             "modules at 1.0; 0 = freeze (V4_Av3 口径).")
    parser.add_argument("--g-floor-src", type=str, default=None, choices=("gt", "t"),
                        help="V3_Gv1: observation-layer floor source (gt = E v2.1 口径, default; t = T-derived diagnostic).")
    parser.add_argument("--carrier-from", type=Path, default=None,
                        help="V3_Gv1: E v2.1 ckpt_best whose V stream is the frozen carrier.")
    parser.add_argument("--beta", type=float, default=None, metavar="B",
                        help="V4_Av1: kin-mask beta (pre-registered sweep {0, 1.0, 2.0}; "
                             "default 1.0 from the entry).")
    parser.add_argument("--occ-mix", action="store_true",
                        help="V6_Av1 occ_mix: window-level structured-dropout augmentation "
                             "(default off; --p-occ/--p-nullv/--p-nullt tune it).")
    parser.add_argument("--t-phys-v2", action="store_true", default=None,
                        help="维度 2：T_phys 12→20 维显式物理特征（着地几何+双足关系）")
    parser.add_argument("--d8-heads", action="store_true", default=None,
                        help="维度 3：足-地监督三头（tcop 自重建 + 着地模式 + GT CoP）")
    parser.add_argument("--tactile-gainbias", action="store_true",
                        help="阶段 3 G-b：训练侧鞋垫增益偏置增强（每窗每足 gain[0.7,1.3] + bias±5%%，"
                             "默认关）。")
    parser.add_argument("--tactile-frame-dropout", type=float, default=None, metavar="P",
                        help="阶段 3 G-c：训练侧整帧 T token dropout 概率 p（默认关；建议 0.2）。")
    parser.add_argument("--tactile-weight-norm", action="store_true",
                        help="T_raw 会话体重代理归一（每会话总压力中位数除；跨受试者尺度对齐，默认关）。")
    parser.add_argument("--p-occ", type=float, default=None, metavar="P",
                        help="V6_Av1 occ_mix: P(occ010 source window) (default 0.2).")
    parser.add_argument("--p-nullv", type=float, default=None, metavar="P",
                        help="V6_Av1 occ_mix: P(null-V window) (default 0.1).")
    parser.add_argument("--p-nullt", type=float, default=None, metavar="P",
                        help="V6_Av1 occ_mix: P(null-T window) (default 0.1).")
    parser.add_argument("--occ-a3-monitor", action="store_true",
                        help="V6_Av1 A3 val 分层监控（窗口级 VT2M<=V2M 占比 "
                             "occ vs cam3 对照；默认关，仅 occ_mix 启用时有效).")
    parser.add_argument("--lambda-pose-vel", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_pose_vel (E6.2, pos mode).")
    parser.add_argument("--lambda-bone", type=float, default=None, metavar="W",
                        help="E6.7: override yaml loss weight lambda_bone (E6.2, pos mode).")
    parser.add_argument("--lambda-s", type=float, default=None, metavar="W",
                        help="S 显式收益监督：四组门 (vel/other/left/right) 对离线相对"
                             "收益标签 BCE 权重（0 = 关）。")
    parser.add_argument("--v7-s-label", type=Path, default=None,
                        help="S 显式收益监督：s_gate_diag.py 产出目录（含 "
                             "tlr_gate_diag.npz/.json）。")
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu.")
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--contact-method",
        default=None,
        help="AnySole-private contact-label scheme for contact_gt. Labels are generated "
        "under model_inputs/AnySole by anysole.data.contact_adapter.",
    )
    parser.add_argument("--smpl-support-policy", choices=("exclude", "legacy_clip"), default=None,
                        help="exclude unsupported SMPL target windows; legacy_clip only for historical reproduction")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override checkpoint output directory.")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None,
                        help="LR schedule override: constant (E3 口径) or "
                        "cosine (epoch decay; V7 设计稿写过 cosine 但从未实际执行).")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model directory identifier (e.g. F0b / V3_4a). With --out-dir omitted the "
        "output dir is INFERRED: <model-name>_<contact-method>[/<stacked fields>]/checkpoints, "
        "where the stacked fields (tw/st/lr/lp/lt/lk, defaults omitted) are built from the "
        "effective config — model+contact+fields == the ckpt address (2026-09-22 naming). "
        "--out-dir remains an explicit override only.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Explicit variant dir name override (e.g. --variant tw20 puts a "
        "default-hyperparameter run in <model>_<contact>/tw20/ instead of the "
        "bare base dir, keeping the results tree uniform). When omitted, the "
        "variant is auto-derived from the stacked non-default fields.",
    )
    parser.add_argument("--init-from", type=Path, default=None,
                        help="DEBUG OVERRIDE ONLY: warm-start from an existing "
                             "checkpoint.  Independent-models rule (2026-09-24): "
                             "every run starts from random init by default — no "
                             "warm-start is ever inferred.  Only state-dict keys "
                             "whose names AND shapes match the current model are "
                             "copied (everything else keeps its fresh init). "
                             "--init-drop adds explicit prefixes to skip.")
    parser.add_argument("--init-drop", type=str, default="",
                        help="Comma-separated state-dict key prefixes to skip in addition "
                             "to the automatic name/shape matching (e.g. 'pose_head.' to "
                             "re-init the whole pose head while reusing the encoders).")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_eval_interval", type=int, default=None,
                        help="Epochs between validation evals (3 condition configs x full val set, "
                             "DDIM sampling). One eval costs ~16 normal epochs of wall-clock, so this "
                             "knob dominates total time: 10 -> 57s stall every 10 epochs (~64%% overhead), "
                             "50 -> ~26%% overhead. Runs regardless of wandb mode.")
    parser.add_argument("--wandb_experiment_tag", default=None, help="Batch tag used by Wandb_Analyzer to group runs.")
    parser.add_argument("--monitor_backend", choices=("wandb", "swanlab"), default=None,
                        help="Online monitoring backend used when --wandb_mode != disabled.")
    parser.add_argument("--swanlab_workspace", default=None,
                        help="SwanLab workspace (user or team); empty means the personal workspace.")
    parser.add_argument("--v5-conditioning", choices=("legacy", "availability_v1"), default=None,
                        help="V5 training recipe: legacy reproduces old conditioning; availability_v1 "
                             "aligns V/T/VT supervision with config and structured/random input dropout. "
                             "Use a fresh output directory; saved in checkpoint config.")
    parser.add_argument("--checkpoint-policy", choices=("legacy", "multi_metric"), default=None,
                        help="Default legacy: keep only ckpt_last/ckpt_best. "
                             "multi_metric ADDS the candidate pool: every epoch whose val PA "
                             "passes its three condition gates is retained in full (plus "
                             "per-axis PA/GlobalMPJPE champions for gate-failing epochs). "
                             "Selection happens after training, on formal ten-metric validation.")
    parser.add_argument("--checkpoint-role", choices=("auto", "U", "S_V", "S_T", "S_VT"), default=None,
                        help="Candidate objective inputs; auto infers from config_probs [VT,V,T]. "
                             "Explicit roles must agree with the actual training inputs.")
    parser.add_argument("--checkpoint-gate-v", type=float, default=None,
                        help="Candidate pool gate: max val V-only PA-MPJPE in mm (default 40; "
                             "2026-10-25 裁定). Only used by multi_metric.")
    parser.add_argument("--checkpoint-gate-t", type=float, default=None,
                        help="Candidate pool gate: max val T-only PA-MPJPE in mm (default 90; "
                             "2026-10-25 裁定). Only used by multi_metric.")
    parser.add_argument("--checkpoint-gate-vt", type=float, default=None,
                        help="Candidate pool gate: max val VT PA-MPJPE in mm (default 40; "
                             "2026-10-25 裁定). Only used by multi_metric.")
    parser.add_argument("--checkpoint-all", action="store_true", default=None,
                        help="Candidate pool retention switch (2026-10-26 裁定): default false "
                             "keeps only the most recent gate-passer as a rolling single file "
                             "(a gate-failing epoch keeps the previous passer); true keeps every "
                             "gate-passing epoch in full (the 2026-10-25 protocol). "
                             "Only used by multi_metric.")
    parser.add_argument("--loss-cap", type=float, default=None,
                        help="If set, clip the logged train loss metrics (loss_total, loss_pose, loss_traj, "
                             "loss_con, loss_kp, loss_Trec_Tmissing, loss_Vrec_Vmissing, grad_norm) at this "
                             "value, keeping their original names (no *_cap duplicates). Raw charts get "
                             "y-squashed by the first epochs' spike (loss_total 30 -> 0.1, loss_pose 1.5 -> "
                             "0.01), hiding convergence; e.g. --loss-cap 1.0 pins the y-axis to 0-1.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Max-norm gradient clipping (torch.nn.utils.clip_grad_norm_). "
                             "None = off. F0 lesson: the original F0b_warm run exploded at "
                             "ep~370 via a FINITE huge gradient (nonfinite guard never fired) "
                             "and never recovered; healthy regression runs stay at "
                             "grad_norm <= 1.3 after warm-up, so 5.0 only bites on explosions.")
    parser.add_argument("--lr", type=float, default=None,
                        help="V3-6 Optuna tuning: override the yaml learning rate per "
                             "trial. The manual's base commands keep the yaml value; "
                             "this flag exists for the tuning loop only.")
    return parser.parse_args(argv)


def _evaluate_v3f(model, diffusion, loader, config, device):
    """V3_Fv1 val dashboard: the three conditions are sampler switches.

    VT2M = V cond + T guidance + refinement; V2M = V cond only; T2M =
    unconditional (V zeroed) + T guidance + refinement.  The T stream is
    live under every condition (the contact head reads the real stream).
    """
    result = {}
    sample_steps = int(config["diffusion_sample_steps"])
    with torch.inference_mode():
        for config_value, config_name in zip((CONFIG_VT, CONFIG_V, CONFIG_T), CONFIG_NAMES):
            totals = {"mpjpe": 0.0, "root_ate": 0.0, "n": 0,
                      "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                      "foot_slide_sum": 0.0, "foot_slide_n": 0}
            tau0_mpjpe = 0.0
            first_batch = True
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                bsz = batch["pose_gt"].shape[0]
                v_feat = batch["V_feat"].clone()
                if config_value == CONFIG_T:
                    v_feat.zero_()
                anchor = batch["trans_anchor"][:, None, :]
                out = sample_v3f(
                    model, v_feat, batch["T_raw"], batch["T_phys"],
                    anchor=anchor, floor_y=batch["floor_y"],
                    offsets=batch["offsets"], parents=batch["parents"],
                    use_t=config_value != CONFIG_V,
                    refine=config_value != CONFIG_V,
                    diffusion=diffusion, steps=sample_steps, device=device)
                pred_pose = out["x0_hat"]
                pred_trans = out["trans_hat"] + anchor
                gt_trans = batch["trans_gt"] + anchor
                pred_kp = fk_pose6d(pred_pose, pred_trans, batch["offsets"], batch["parents"])
                totals["mpjpe"] += float(torch.linalg.vector_norm(
                    pred_kp - batch["kp_gt"], dim=-1).mean().item()) * bsz * 1000.0
                if first_batch:
                    # tau=0 clean-input reconstruction (the ckpt_best tracker):
                    # x_tau = the clean normalized state, denoised at tau=0.
                    x0_n = model.denoiser.normalize(torch.cat(
                        [batch["pose_gt"], batch["trans_gt"], batch["contact_gt"]], dim=-1))
                    zero = torch.zeros(bsz, device=device, dtype=torch.long)
                    out0 = model(v_feat, batch["T_raw"], batch["T_phys"], x0_n, zero)
                    pose0 = _split_state(model.denoiser.unnormalize(out0["x0_hat"]))[0]
                    kp0 = fk_pose6d(pose0, gt_trans, batch["offsets"], batch["parents"])
                    tau0_mpjpe = float(torch.linalg.vector_norm(
                        kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0
                    first_batch = False
                pred_rel = pred_trans - pred_trans[:, :1]
                gt_rel = gt_trans - gt_trans[:, :1]
                totals["root_ate"] += float(torch.linalg.vector_norm(
                    pred_rel - gt_rel, dim=-1).mean().item()) * bsz
                # The contact row measures the T contact head itself (the
                # guidance/refinement input), the V3_Ev1 convention.
                pred_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                gt_contact = batch["contact_gt"] > 0.5
                totals["tp"] += int((pred_contact & gt_contact).sum().item())
                totals["fp"] += int((pred_contact & ~gt_contact).sum().item())
                totals["fn"] += int((~pred_contact & gt_contact).sum().item())
                totals["tn"] += int((~pred_contact & ~gt_contact).sum().item())
                foot = pred_kp[:, :, FOOT_JOINTS, :][:, :, :, [0, 2]]
                speed = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1) * 1000.0
                slide = speed[pred_contact[:, 1:]]
                if slide.numel():
                    totals["foot_slide_sum"] += float(slide.sum().item())
                    totals["foot_slide_n"] += int(slide.numel())
                totals["n"] += bsz
            count = max(totals["n"], 1)
            p = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
            r = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
            air_recall = totals["tn"] / max(totals["tn"] + totals["fp"], 1)
            result.update({
                f"val/mpjpe/{config_name}": totals["mpjpe"] / count,
                f"val/root_ate/{config_name}": totals["root_ate"] / count,
                f"val/contact_f1/{config_name}": 2 * p * r / max(p + r, 1e-8),
                f"val/contact_recall/{config_name}": r,
                f"val/air_recall/{config_name}": air_recall,
                f"val/contact_balanced_acc/{config_name}": 0.5 * (r + air_recall),
                f"val/foot_slide/{config_name}": totals["foot_slide_sum"] / max(totals["foot_slide_n"], 1),
                f"val/tau0_mpjpe/{config_name}": tau0_mpjpe})
    result["val/gap_mpjpe_dropT"] = result["val/mpjpe/V"] - result["val/mpjpe/VT"]
    result["val/gap_mpjpe_dropV"] = result["val/mpjpe/T"] - result["val/mpjpe/VT"]
    return result


def _evaluate(model, diffusion, loader, config, device):
    """Return the epoch-level VT/V/T metrics used by the training dashboard."""
    # V3_Fv1 has no pose_head — route before the first pose_head access
    # (the pos_mode probe below would raise AttributeError).  The model
    # attribute is the source of truth (the caller's config may not carry
    # the v3f flag, e.g. the yaml config alone).
    if getattr(model, "v3f", False):
        return _evaluate_v3f(model, diffusion, loader, config, device)
    pos_mode = getattr(model.pose_head, "repr", "6d") == "pos"
    regress_mode = str(config.get("modal")) == MODEL_ANYSOLEV2
    v3e_mode = bool(config.get("v3e"))
    v3g_mode = bool(config.get("v3g"))
    v3h_mode = bool(config.get("v3h"))
    v5_1_mode = bool(config.get("v5_1"))
    v4_1_mode = bool(config.get("v4_1"))
    f2_mode = bool(config.get("f2_repr", False))
    result = {}
    with torch.inference_mode():
        for config_value, config_name in zip((0, 1, 2), CONFIG_NAMES):
            totals = {"mpjpe": 0.0, "root_ate": 0.0, "contact_f1": 0.0, "foot_slide": 0.0, "n": 0}
            tau0_mpjpe = 0.0
            first_batch = True
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                bsz = batch["pose_gt"].shape[0]
                cid = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
                if v5_1_mode or v4_1_mode:
                    # V5_Av1/V4_Av1：T 流输入三条件恒定（g_T/接触头/XAttn
                    # 恒读真实压力；干预实验要求同输入比来源）——恢复被
                    # condition_inputs 零化的 T；V 零化保留（T2M 行）。
                    t_raw, t_phys = batch["T_raw"], batch["T_phys"]
                v_kw = {"V_hmr": batch.get("V_hmr")} if str(config.get("v_input", "hrnet")) == "hmr_gvhmr" else {}
                if regress_mode:
                    # F0b: single forward, no DDIM sampling.
                    z_kw = {"z_src": v5_1_z_src(config_value)} if v5_1_mode else {}
                    out = model(v_feat, t_raw, t_phys, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw, **z_kw,
                    prev_ctx=batch.get("ctx"), prev_state=batch.get("prev_state"),
                    causal=bool(getattr(model, "ar", False) or getattr(model, "ar2", False)),
                    **model_observation_kwargs(batch))
                    pred_pose = out["x0_hat"]
                    if getattr(model, "ar2", False):
                        # ar2 val：teacher-forced 单步口径，相对表示 →
                        # 世界组装（psi_base = GT 窗前一帧航向）。
                        pred_pose, pred_trans_ar2 = ar2_to_world(
                            out["x0_hat"], out["dyaw_hat"][..., 0],
                            out["v_hat"], batch["psi_base"],
                            batch["trans_anchor"])
                        out = dict(out)
                        out["trans_hat"] = pred_trans_ar2 - batch[
                            "trans_anchor"][:, None, :]
                else:
                    warm_prior = None
                    if bool(config.get("warm_start", False)):
                        # E6.5: initialize the chain on the training marginal
                        # (mean pose), not pure noise.
                        warm_prior = model.pose_head.pose_mean.view(1, 1, -1).expand(
                            bsz, int(config["tw"]), -1)
                    pred_pose = diffusion.ddim_sample_loop(
                        model, tau_related_kwargs={"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                        "T_s2m": t_s2m, "config_id": cid, "session_id": batch.get("session_id")},
                        shape=(bsz, int(config["tw"]), model.pose_head.pose_dim),
                        steps=int(config["diffusion_sample_steps"]), eta=0.0, device=device,
                        prior=warm_prior)
                    zero = torch.zeros(bsz, device=device, dtype=torch.long)
                    out = model(v_feat, t_raw, t_phys, pred_pose, zero, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                anchor = batch["trans_anchor"][:, None, :]
                pred_trans = out["trans_hat"] + anchor
                gt_trans = batch["trans_gt"] + anchor
                if getattr(model, "ctx4b", False):
                    # v4-b：val T 行 = GT 锚（teacher-forced，首窗 I）；
                    # V/VT 行 = 锚估计头输出。
                    if config_value == CONFIG_T:
                        R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])
                        ctx = batch.get("ctx")
                        if ctx is not None:
                            first = ctx[..., -1].mean(dim=1) <= 0.5
                            R_a[first] = torch.eye(3, device=R_a.device,
                                                   dtype=R_a.dtype)
                    else:
                        R_a = out["anchor_rot"]
                    pred_pose = rot_root6d(R_a[:, None, :, :], out["x0_hat"])
                    v_w = torch.einsum("bij,btj->bti", R_a, out["v_hat"])
                    pred_trans = torch.cumsum(v_w, dim=1) / float(FPS) + anchor
                elif getattr(model, "ctx4", False) and config_value == CONFIG_T:
                    # v4-a：val T 行输出 = 残差，组合 R_a（GT 锚，teacher-forced
                    # val；首窗 I——与训练目标同口径）后进 FK 指标。
                    R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])
                    ctx = batch.get("ctx")
                    if ctx is not None:
                        first = ctx[..., -1].mean(dim=1) <= 0.5
                        R_a[first] = torch.eye(3, device=R_a.device,
                                               dtype=R_a.dtype)
                    pred_pose = rot_root6d(R_a[:, None, :, :], out["x0_hat"])
                    v_w = torch.einsum("bij,btj->bti", R_a, out["v_hat"])
                    pred_trans = torch.cumsum(v_w, dim=1) / float(FPS) + anchor
                if v3e_mode:
                    # V3_Ev1: the three dashboard conditions are three readouts
                    # of one forward — raw V pose, constraint-processed VT,
                    # T pose prior.  The forward itself ignores config_id.
                    if config_value == CONFIG_VT:
                        pred_pose, pred_trans = apply_v3e_constraints(
                            out["x0_hat"], out["trans_hat"] + anchor,
                            batch["floor_y"],
                            batch["offsets"], batch["parents"])
                    elif config_value == CONFIG_V:
                        pred_pose, pred_trans = out["x0_hat"], out["trans_hat"] + anchor
                    else:
                        pred_pose, pred_trans = out["x0_prior"], out["trans_prior"] + anchor
                elif v3g_mode:
                    # V3_Gv1: VT = 分部观测层 on the V stream; V = raw carrier
                    # (= E); T = the T prior + 观测层 (observation-only sanity).
                    # floor_src stays the registered default "gt" (E v2.1 口径).
                    if config_value == CONFIG_VT:
                        pred_pose, pred_trans = apply_v3g_constraints(
                            out["x0_hat"], out["trans_hat"] + anchor,
                            out["contact_logits"],
                            batch["offsets"], batch["parents"],
                            floor_y=batch["floor_y"])
                    elif config_value == CONFIG_V:
                        pred_pose, pred_trans = out["x0_hat"], out["trans_hat"] + anchor
                    else:
                        pred_pose, pred_trans = apply_v3g_constraints(
                            out["x0_prior"], out["trans_prior"] + anchor,
                            out["contact_logits"],
                            batch["offsets"], batch["parents"],
                            floor_y=batch["floor_y"])
                elif v3h_mode:
                    # V3_Hv1: VT = 残差通道 on the V stream; V = raw V tower;
                    # T = null-V 先验 + 残差通道 (H 无 T 塔——同一 V 塔、V 已
                    # 在上游零化)。floor_src stays the registered default
                    # "gt" (E v2.1 口径)。V3_Hv3：VT/T 行先应用一阶足部
                    # 残差（V2M 行不应用——零税）。
                    pose_vt = apply_foot_residual(
                        out["x0_hat"], out["foot_res"],
                        landing=str(getattr(model, "res_landing", "foot"))) if out.get("foot_res") is not None else out["x0_hat"]
                    if config_value == CONFIG_VT:
                        pred_pose, pred_trans = apply_v3h(
                            bool(getattr(model, "no_anchor", False)),
                            pose_vt, out["trans_hat"] + anchor,
                            out["contact_logits"],
                            batch["offsets"], batch["parents"],
                            floor_y=batch["floor_y"])
                    elif config_value == CONFIG_V:
                        pred_pose, pred_trans = out["x0_hat"], out["trans_hat"] + anchor
                    else:
                        pred_pose, pred_trans = apply_v3h(
                            bool(getattr(model, "no_anchor", False)),
                            pose_vt, out["trans_hat"] + anchor,
                            out["contact_logits"],
                            batch["offsets"], batch["parents"],
                            floor_y=batch["floor_y"])
                elif v4_1_mode:
                    # V4_Av1: V2M = V 载体（修正不应用）；VT2M = V + pos_corr
                    # 软骨架（下方 FK 块按条件应用）；T2M = 先验 + pos_corr_prior。
                    if config_value == CONFIG_VT:
                        pred_pose, pred_trans = out["x0_hat"], out["trans_hat"] + anchor
                    elif config_value == CONFIG_V:
                        pred_pose, pred_trans = out["x0_hat"], out["trans_hat"] + anchor
                    else:
                        pred_pose, pred_trans = out["x0_prior"], out["trans_prior"] + anchor
                n = bsz
                if pos_mode:
                    # E6.1: compare in the metric space directly (23 joints,
                    # root-local; root covered by root_ate below).
                    err = torch.linalg.vector_norm(
                        pred_pose.reshape(bsz, -1, N_JOINTS - 1, 3)
                        - batch["pose_gt_pos"].reshape(bsz, -1, N_JOINTS - 1, 3),
                        dim=-1,
                    )
                    totals["mpjpe"] += float(err.mean().item()) * n * 1000.0
                else:
                    if f2_mode:
                        # F2a: recover the world pose/trans from the tilt pose
                        # + heading-frame trajectory, then the same FK scope.
                        world_pose, world_trans = f2_to_world(
                            pred_pose, out["v_hat"], batch["psi_anchor"],
                            batch["trans_anchor"],
                        )
                        pred_kp = fk_pose6d(world_pose, world_trans,
                                            batch["offsets"], batch["parents"])
                        pred_trans = world_trans
                    else:
                        pred_kp = fk_pose6d(pred_pose, pred_trans, batch["offsets"], batch["parents"])
                        if v4_1_mode:
                            # V4_Av1 软骨架按条件应用（契约 1 守卫生效：
                            # V2M 行不加修正——V3_Dv1 遗留块对 v4_1 禁用）。
                            if config_value == CONFIG_VT:
                                corr = out["pos_corr"]
                            elif config_value == CONFIG_T:
                                corr = out["pos_corr_prior"]
                            else:
                                corr = None
                            if corr is not None:
                                pred_kp = pred_kp.clone()
                                pred_kp[:, :, KNEE_ANKLE_FOOT_JOINTS, :] = (
                                    pred_kp[:, :, KNEE_ANKLE_FOOT_JOINTS, :]
                                    + corr.reshape(bsz, -1, len(KNEE_ANKLE_FOOT_JOINTS), 3)
                                )
                        elif out.get("pos_corr") is not None:
                            # V3_Dv1: the val dashboard sees the corrected skeleton
                            # (soft skeleton), so ckpt_best tracks the position
                            # head's contribution, not the frozen base's constant.
                            pred_kp = pred_kp.clone()
                            pred_kp[:, :, KNEE_ANKLE_FOOT_JOINTS, :] = (
                                pred_kp[:, :, KNEE_ANKLE_FOOT_JOINTS, :]
                                + out["pos_corr"].reshape(bsz, -1, len(KNEE_ANKLE_FOOT_JOINTS), 3)
                            )
                    totals["mpjpe"] += float(torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1).mean().item()) * n * 1000.0
                    if not pos_mode and totals.get("pa_ok", True):
                        # Candidate pool gates (2026-10-25): frame-level
                        # PA-MPJPE on the FK keypoints, same formula family as
                        # eval.py (similarity_align + mean_point_error, mm).
                        # The three gates read val/pa_mpjpe/{V,T,VT}.
                        try:
                            _pa = pa_mpjpe(
                                pred_kp.reshape(bsz * pred_kp.shape[1], -1, 3).detach().float().contiguous().cpu().numpy(),
                                batch["kp_gt"].reshape(bsz * batch["kp_gt"].shape[1], -1, 3).detach().float().contiguous().cpu().numpy())
                            totals["pa_sum"] = totals.get("pa_sum", 0.0) + float(_pa.mean()) * n
                        except ValueError:
                            # Degenerate predicted point set (variance ~0): the
                            # condition metric is undefined for this epoch —
                            # mark the row unusable instead of poisoning the sum.
                            totals["pa_ok"] = False
                if first_batch and not regress_mode:
                    # tau=0 clean-input reconstruction MPJPE on one batch (no
                    # diffusion sampling): tells apart "cannot fit the training
                    # data" from "cannot denoise", on the same joint scope as
                    # eval.py (all 23 joints).  F0b regress has no clean-input
                    # reconstruction — its direct output IS the prediction, so
                    # tau0 is set to the full MPJPE below.
                    pose_in = batch["pose_gt_pos"] if pos_mode else batch["pose_gt"]
                    zero = torch.zeros(bsz, device=device, dtype=torch.long)
                    out0 = model(v_feat, t_raw, t_phys, pose_in, zero, cid, batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                    if pos_mode:
                        err0 = torch.linalg.vector_norm(
                            out0["x0_hat"].reshape(bsz, -1, N_JOINTS - 1, 3)
                            - batch["pose_gt_pos"].reshape(bsz, -1, N_JOINTS - 1, 3),
                            dim=-1,
                        )
                        tau0_mpjpe = float(err0.mean().item()) * 1000.0
                    else:
                        kp0 = fk_pose6d(out0["x0_hat"], batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
                        tau0_mpjpe = float(torch.linalg.vector_norm(kp0 - batch["kp_gt"], dim=-1).mean().item()) * 1000.0
                    first_batch = False
                pred_rel = pred_trans - pred_trans[:, :1]
                gt_rel = gt_trans - gt_trans[:, :1]
                totals["root_ate"] += float(torch.linalg.vector_norm(pred_rel - gt_rel, dim=-1).mean().item()) * n
                if not pos_mode:
                    if (v3e_mode or v3g_mode or v3h_mode or v5_1_mode or v4_1_mode) and config_value == CONFIG_VT and out.get("contact_logits") is not None:
                        # V3_Ev1/G/H/V5_Av1: the val dashboard contact row measures
                        # the T contact head itself (the residual/observation
                        # layer's gate input / G 配方接触头).
                        pred_contact = torch.sigmoid(out["contact_logits"]) > 0.5
                    else:
                        pred_contact = soft_contact_from_keypoints(pred_kp, batch["floor_y"]) > 0.5
                    gt_contact = batch["contact_gt"] > 0.5
                    tp = (pred_contact & gt_contact).sum().item()
                    fp = (pred_contact & ~gt_contact).sum().item()
                    fn = (~pred_contact & gt_contact).sum().item()
                    tn = (~pred_contact & ~gt_contact).sum().item()
                    totals.setdefault("tp", 0); totals.setdefault("fp", 0); totals.setdefault("fn", 0); totals.setdefault("tn", 0)
                    totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn; totals["tn"] += tn
                    foot = pred_kp[:, :, FOOT_JOINTS, :][:, :, :, [0, 2]]
                    # Keypoints are meters; report horizontal contact displacement
                    # in mm/frame, matching eval_protocol.py.  Each foot is masked
                    # by its own contact state.
                    # Each foot is masked by its own contact state. The old
                    # any-foot mask averaged the fast swing foot into the
                    # stationary stance foot and over-reported sliding.
                    speed = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1) * 1000.0
                    slide = speed[pred_contact[:, 1:]]
                    if slide.numel():
                        totals.setdefault("foot_slide_sum", 0.0)
                        totals.setdefault("foot_slide_n", 0)
                        totals["foot_slide_sum"] += float(slide.sum().item())
                        totals["foot_slide_n"] += int(slide.numel())
                totals["n"] += n
            count = max(totals["n"], 1)
            p = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fp", 0), 1)
            r = totals.get("tp", 0) / max(totals.get("tp", 0) + totals.get("fn", 0), 1)
            air_recall = totals.get("tn", 0) / max(totals.get("tn", 0) + totals.get("fp", 0), 1)
            balanced_acc = 0.5 * (r + air_recall)
            if regress_mode:
                # F0b: no denoising chain — the direct regression output is
                # both the final prediction and the "clean-input" value.
                tau0_mpjpe = totals["mpjpe"] / count
            result.update({f"val/mpjpe/{config_name}": totals["mpjpe"] / count,
                           f"val/root_ate/{config_name}": totals["root_ate"] / count,
                           f"val/contact_f1/{config_name}": 2 * p * r / max(p + r, 1e-8),
                           f"val/contact_recall/{config_name}": r,
                           f"val/air_recall/{config_name}": air_recall,
                           f"val/contact_balanced_acc/{config_name}": balanced_acc,
                           f"val/foot_slide/{config_name}": totals.get("foot_slide_sum", 0.0) / max(totals.get("foot_slide_n", 0), 1),
                           f"val/tau0_mpjpe/{config_name}": tau0_mpjpe})
            if totals.get("pa_ok", True) and not pos_mode:
                result[f"val/pa_mpjpe/{config_name}"] = totals.get("pa_sum", 0.0) / count
    result["val/gap_mpjpe_dropT"] = result["val/mpjpe/V"] - result["val/mpjpe/VT"]
    result["val/gap_mpjpe_dropV"] = result["val/mpjpe/T"] - result["val/mpjpe/VT"]
    return result


def _occ_a3_monitor(model, loader, config, device):
    """V6_Av1 A3 val 分层监控（开关 occ_a3_monitor，默认关；方案 §4.1）。

    窗口级 VT2M ≤ V2M 占比按窗源分组（occ 组 vs cam3 组对照）：val
    loader 同挂 occ 标签，输入按标签来源喂（vz/tz 置零；occ 源已在
    dataset 侧选定）。VT 行 = 结构线读出：V4_Av1 软骨架（x0_hat +
    pos_corr 六关节）；V5_Av1 z_src="VT"（二次前向）；其余模型退回
    V 行同读数（占比 0.5，无信号）。"""
    v4_1_mode = bool(config.get("v4_1", False))
    v5_1_mode = bool(config.get("v5_1", False))
    v_hmr_mode = str(getattr(model, "v_input", "hrnet")) == "hmr_gvhmr"
    stats = {"cam3": [0, 0], "occ": [0, 0]}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if batch.get("occ_src") is None:
                continue
            bsz = batch["pose_gt"].shape[0]
            cid = torch.zeros(bsz, dtype=torch.long, device=device)  # VT 行
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, cid)
            if v4_1_mode or v5_1_mode:
                t_raw, t_phys = batch["T_raw"], batch["T_phys"]
            vz = batch["occ_v_zeroed"].view(-1, 1, 1).to(device, dtype=torch.bool)
            tz = batch["occ_t_zeroed"].view(-1, 1, 1).to(device, dtype=torch.bool)
            v_feat = torch.where(vz, torch.zeros_like(v_feat), v_feat)
            t_raw = torch.where(tz, torch.zeros_like(t_raw), t_raw)
            t_phys = torch.where(tz, torch.zeros_like(t_phys), t_phys)
            t_s2m = torch.where(tz, torch.zeros_like(t_s2m), t_s2m)
            v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
            anchor = batch["trans_anchor"][:, None, :]
            if v5_1_mode:
                out_v = model(v_feat, t_raw, t_phys, cid, batch.get("session_id"),
                              T_s2m=t_s2m, **v_kw, **{"z_src": "V"})
                out_vt = model(v_feat, t_raw, t_phys, cid, batch.get("session_id"),
                               T_s2m=t_s2m, **v_kw, **{"z_src": "VT"})
                kp_v = fk_pose6d(out_v["x0_hat"], out_v["trans_hat"] + anchor,
                                 batch["offsets"], batch["parents"])
                kp_vt = fk_pose6d(out_vt["x0_hat"], out_vt["trans_hat"] + anchor,
                                  batch["offsets"], batch["parents"])
            else:
                out = model(v_feat, t_raw, t_phys, cid, batch.get("session_id"),
                            T_s2m=t_s2m, **v_kw)
                kp_v = fk_pose6d(out["x0_hat"], out["trans_hat"] + anchor,
                                 batch["offsets"], batch["parents"])
                if v4_1_mode and out.get("pos_corr") is not None:
                    kp_vt = kp_v.clone()
                    kp_vt[:, :, KNEE_ANKLE_FOOT_JOINTS, :] = (
                        kp_vt[:, :, KNEE_ANKLE_FOOT_JOINTS, :]
                        + out["pos_corr"].reshape(bsz, -1, len(KNEE_ANKLE_FOOT_JOINTS), 3))
                else:
                    kp_vt = kp_v
            err_v = torch.linalg.vector_norm(kp_v - batch["kp_gt"], dim=-1).mean(dim=(-1, -2))
            err_vt = torch.linalg.vector_norm(kp_vt - batch["kp_gt"], dim=-1).mean(dim=(-1, -2))
            vt_le_v = err_vt <= err_v
            src = batch["occ_src"].to(device)
            for src_val, name in ((SRC_CAM3, "cam3"), (SRC_OCC, "occ")):
                mask = src == src_val
                if bool(mask.any()):
                    stats[name][0] += int(vt_le_v[mask].sum().item())
                    stats[name][1] += int(mask.sum().item())
    return {
        "val/occ_a3/vt_le_v_frac_cam3": stats["cam3"][0] / max(stats["cam3"][1], 1),
        "val/occ_a3/vt_le_v_frac_occ": stats["occ"][0] / max(stats["occ"][1], 1),
        "val/occ_a3/n_cam3": stats["cam3"][1],
        "val/occ_a3/n_occ": stats["occ"][1],
    }


def _grad_probe_groups(model):
    """按模块划组的梯度范数探针（2026-11-03 分支挤压审计用）。

    encoders 持有共享 embeddings 的引用，所以按 id 归组、共享块优先：
    每个参数只属于第一个拥有它的组。
    """
    enc = getattr(model, "encoders", None)
    blocks = []
    if enc is not None:
        for name in ("embeddings", "v_enc", "t_enc"):
            module = getattr(enc, name, None)
            if module is not None:
                blocks.append(("share_enc" if name == "embeddings" else name, module))
    if getattr(model, "pose_head", None) is not None:
        blocks.append(("pose_head", model.pose_head))
    seen = set()
    groups = {}
    for name, module in blocks:
        params = [param for param in module.parameters() if id(param) not in seen]
        seen.update(id(param) for param in params)
        groups[name] = params
    groups["other"] = [param for param in model.parameters() if id(param) not in seen]
    return groups


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config["input_adapter"] = "AnySole"
    config["adapter_version"] = ADAPTER_VERSION
    config["shared_schema_version"] = "shared.session.v1"
    config["split_hash"] = sha256_file(Path(config["split_csv"]))
    config["contact_method"] = str(config["contact_method"])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("training seed: %d" % seed)
    if args.tw is not None:
        if args.tw < 1:
            raise ValueError("--tw must be positive")
        config["tw"] = int(args.tw)
    if args.obs_context is not None:
        config["obs_context"] = int(args.obs_context)
    if args.dropout is not None:
        config["dropout"] = float(args.dropout)
    if args.config_probs is not None:
        config["config_probs"] = args.config_probs
    for key in ("condition_sampling", "final_eval_split"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config.get("condition_sampling", "sample") not in ("sample", "enum"):
        raise ValueError("condition_sampling must be sample or enum")
    if config.get("final_eval_split", "test") not in ("val", "test", "none"):
        raise ValueError("final_eval_split must be val, test or none")
    if args.dropout_vt is not None:
        tokens = [item for item in " ".join(args.dropout_vt).replace(",", " ").split() if item]
        if len(tokens) != 2:
            raise ValueError("--dropoutVT needs two percentages (e.g. '--dropoutVT 20 30' or '--dropoutVT 20,30')")
        drop_v, drop_t = float(tokens[0]) / 100.0, float(tokens[1]) / 100.0
        if drop_v < 0.0 or drop_t < 0.0 or drop_v + drop_t > 1.0:
            raise ValueError("--dropoutVT percentages must be non-negative and sum to at most 100")
        config["config_probs"] = [1.0 - drop_v - drop_t, drop_t, drop_v]
        config["dropout_vt"] = [drop_v * 100.0, drop_t * 100.0]
    if args.modal is not None:
        config["modal"] = args.modal
    if args.contact_method is not None:
        config["contact_method"] = args.contact_method
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)
    if args.lr_schedule is not None:
        config["lr_schedule"] = args.lr_schedule
    for key in ("wandb_mode", "wandb_project", "wandb_entity", "wandb_eval_interval", "wandb_experiment_tag",
                "monitor_backend", "swanlab_workspace"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("checkpoint_policy", "checkpoint_role",
                "checkpoint_gate_v", "checkpoint_gate_t", "checkpoint_gate_vt"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.smpl_support_policy is not None:
        config["smpl_support_policy"] = args.smpl_support_policy
    if config.get("smpl_support_policy", "exclude") not in ("exclude", "legacy_clip"):
        raise ValueError("Invalid SMPL supervision support policy")
    if args.checkpoint_all is not None:
        config["checkpoint_all"] = bool(args.checkpoint_all)
    if config.get("checkpoint_policy", "legacy") not in ("legacy", "multi_metric"):
        raise ValueError("Unknown checkpoint_policy")
    # Registered experiment names identify the model branch; the architecture
    # comes from the model's entry script (anysole/models/<module>.py), which
    # fixes the structure — structure flags are not a CLI choice (2026-09-24
    # independence restructure).  Every run starts from random init: model +
    # hyperparameters is the full identity of a run.
    entry = None
    if args.model_name is not None:
        config["modal"] = MODEL_ANYSOLEV2
        entry = get_builder(args.model_name)
        config["model_name"] = args.model_name
        config.update(entry.STRUCTURE)
        config.update(getattr(entry, "CONFIG_EXTRA", {}))
    v3f_mode = bool(config.get("v3f", False))
    v3h_mode = bool(config.get("v3h", False))
    v5_1_mode = bool(config.get("v5_1", False))
    v4_1_mode = bool(config.get("v4_1", False))
    enum_mode = config.get("condition_sampling", "sample") == "enum"
    if enum_mode and (not bool(config.get("v7")) or bool(config.get("occ_mix", False)) or args.occ_mix):
        raise ValueError("enum currently requires a V7-derived model without occ_mix")
    if bool(config.get("ar_rollout", False)) and not enum_mode:
        # 修改 04：rollout 只接在 backward_v7_enum（枚举路径）上——
        # 非枚举模式静默死开关 = 假实验（2026-11-03）。
        raise ValueError("--ar-rollout requires enum training (condition_sampling=enum)")
    if args.v5_conditioning is not None:
        config["v5_conditioning"] = args.v5_conditioning
    v5_availability = conditioning_policy(config) == "availability_v1"
    if v5_availability and not v5_1_mode:
        raise ValueError("availability_v1 requires the V5 model family")
    if v5_1_mode:
        config["v5_conditioning"] = conditioning_policy(config)
        print("V5 training conditioning: %s" % config["v5_conditioning"])
    if config.get("modal", MODEL_ANYSOLEV1) not in MODEL_NAMES:
        raise ValueError("Unknown modal %r; expected one of %s" % (config.get("modal"), MODEL_NAMES))
    if config.get("modal") == MODEL_ANYSOLEV2 and not v3f_mode and (
        args.tau_max is not None
        or args.tau_fixed is not None
        or bool(config.get("warm_start", False))
        or bool(config.get("continuation", False))
    ):
        # F0b: the regression model has no diffusion chain; these E-era knobs
        # would silently do nothing, so say so instead.
        print("WARNING: --tau-max/--tau-fixed/warm_start/continuation are diffusion-era "
              "knobs and are ignored by modal anysolev2 (regression)")

    session_ids = None
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = load_split_ids(Path(config["split_csv"]), "train")[: args.limit_sessions]

    if args.tau_max is not None:
        if not 1 <= args.tau_max <= int(config["diffusion_train_steps"]):
            raise ValueError("--tau-max must be in [1, diffusion_train_steps]")
        config["tau_max"] = int(args.tau_max)
    if args.tau_fixed is not None:
        if not 0 <= args.tau_fixed < int(config["diffusion_train_steps"]):
            raise ValueError("--tau-fixed must be in [0, diffusion_train_steps)")
        config["tau_fixed"] = int(args.tau_fixed)
    if args.loss_cap is not None:
        if not args.loss_cap > 0.0:
            raise ValueError("--loss-cap must be positive")
        config["loss_cap"] = float(args.loss_cap)
    if args.grad_clip is not None:
        if not args.grad_clip > 0.0:
            raise ValueError("--grad-clip must be positive")
        config["grad_clip"] = float(args.grad_clip)
    if args.lr is not None:
        if not args.lr > 0.0:
            raise ValueError("--lr must be positive")
        config["lr"] = float(args.lr)
    if args.stride is not None:
        if not 1 <= args.stride <= int(config["tw"]):
            raise ValueError("--stride must be in [1, tw]")
        config["stride"] = int(args.stride)
    if args.part_json is not None:
        # V4B: learned hard partition (probe_part_cluster_b.py output or a
        # plain list of joint-index lists).  The partition is structural data
        # for the V4B entry; every other model rejects it.
        if args.model_name != "V4B":
            raise ValueError("--part-json is only valid for --model-name V4B")
        import json as _json
        part_json_path = Path(args.part_json)
        if not part_json_path.is_file():
            raise FileNotFoundError(
                "V4B partition file missing: %s\n"
                "Re-derive it (data-only, no trained model needed):\n"
                "  python z_note/probes/probe_part_cluster_b_data.py\n"
                "  -> results/AnySole/partitions/partitions_kinematic_b_K*.json\n"
                "then pass one of the exported candidates here."
                % part_json_path
            )
        doc = _json.loads(part_json_path.read_text())
        groups = doc["partition"] if isinstance(doc, dict) and "partition" in doc else doc
        if isinstance(groups, dict):
            # probe format: {"slot_xx": ["joint_name", ...]} -> index lists
            from anysole.types import JOINT_NAMES as _JN
            name_to_idx = {n: i for i, n in enumerate(_JN)}
            groups = [[name_to_idx[n] for n in names] for names in groups.values()]
        part_joints = tuple(tuple(int(j) for j in g) for g in groups)
        config["pose_parts"] = len(part_joints)
        config["part_joints"] = [list(g) for g in part_joints]
        print("part-json: %d groups %s" % (len(part_joints), [list(g) for g in part_joints]))
    if args.lambda_assign is not None:
        config["lambda_assign"] = float(args.lambda_assign)
    if args.lambda_sigma is not None:
        config["lambda_sigma"] = float(args.lambda_sigma)
    if args.sigma_freeze_frac is not None:
        config["sigma_freeze_frac"] = float(args.sigma_freeze_frac)
    if args.assign_cluster:
        # assign_cluster is part of the V4A entry's identity (its STRUCTURE
        # already carries it); the CLI flag stays only to pass the runtime
        # annealing/loss hyperparameters.
        if args.model_name != "V4A":
            raise ValueError("--assign-cluster is only valid for --model-name V4A "
                             "(its entry fixes 24 slots + soft A)")
        for key in ("lambda_assign_ent", "lambda_assign_conc", "lambda_assign_dead",
                    "assign_dead_beta", "assign_temp_init", "assign_temp_final",
                    "assign_anneal_frac", "assign_lock_frac", "assign_lr_mult"):
            config[key] = float(getattr(args, key))
        if float(config["lambda_assign"]) != 1.0:
            print("NOTE: V4A clustering expects --lambda-assign 1.0 (the cluster "
                  "terms carry their own --lambda-assign-* weights); got %s"
                  % config["lambda_assign"])
    # V4A: the anneal progress consumed by the cluster L_assign each step;
    # defaults to fully-annealed (1.0) for every non-cluster run.
    config.setdefault("assign_anneal", 1.0)
    if args.lr_warmup_frac is not None:
        config["lr_warmup_frac"] = float(args.lr_warmup_frac)
    for key in ("lambda_pose", "lambda_kp", "lambda_pos", "lambda_traj", "lambda_trec", "lambda_vrec",
                "lambda_con", "lambda_pose_vel", "lambda_bone",
                "lambda_contact", "lambda_nll", "lambda_prior_pose",
                "lambda_prior_nll", "lambda_prior_traj", "lambda_air",
                "lambda_foot_res", "lambda_phase", "lambda_cop",
                "lambda_tcop", "lambda_strike", "lambda_gtcop",
                "lambda_cond", "lambda_s",
                "lambda_dyaw", "lambda_cumyaw", "lambda_dyaw_td",
                "lambda_visyaw", "lambda_yawcorr", "lambda_world_traj",
                "ar2_dyaw_t_w"):
        value = getattr(args, key)
        if value is not None:
            config[key] = float(value)
    if args.g_floor_src is not None:
        config["g_floor_src"] = str(args.g_floor_src)
    if args.carrier_from is not None:
        config["carrier_from"] = str(args.carrier_from)
    if args.carrier_lr_mult is not None:
        config["carrier_lr_mult"] = float(args.carrier_lr_mult)
    if args.beta is not None:
        config["beta"] = float(args.beta)
    if args.occ_mix:
        config["occ_mix"] = True
    if getattr(args, "t_phys_v2", None) is not None:
        config["t_phys_v2"] = bool(args.t_phys_v2)
    if getattr(args, "d8_heads", None) is not None:
        config["d8_heads"] = bool(args.d8_heads)
    if args.occ_a3_monitor:
        config["occ_a3_monitor"] = True
    if args.v7_s_label is not None:
        config["v7_s_label"] = str(args.v7_s_label)
    # G1 / AR 反馈配方键（2026-11-01：ctx_ss 此前缺失映射 = 死开关，已修）
    config["ctx_ss"] = float(args.ctx_ss)
    config["ar_ss"] = float(args.ar_ss)
    config["ar_noise"] = float(args.ar_noise)
    # 修改 04（2026-11-03）：rollout 损失（改动 A）与低维反馈（改动 B）
    # 配置键，默认关（改动全关 → AR2 原样契约）。
    if args.ar_rollout:
        config["ar_rollout"] = True
    if args.ar_rollout_max is not None:
        config["ar_rollout_max"] = int(args.ar_rollout_max)
    if args.ar_rollout_schedule_frac is not None:
        config["ar_rollout_schedule_frac"] = float(args.ar_rollout_schedule_frac)
    if args.ar_rollout_detach:
        config["ar_rollout_detach"] = True
    if args.t_prefix_p is not None:
        if not 0.0 <= float(args.t_prefix_p) <= 1.0:
            raise ValueError("--t-prefix-p must be in [0, 1]")
        config["t_prefix_p"] = float(args.t_prefix_p)
    if float(config.get("t_prefix_p", 0.0)) > 0.0:
        # 死开关防御（2026-11-03）：前缀继承样本只接在 backward_v7_enum 的 T 档 + AR2
        # （无反馈）+ 长观测上下文上；其余路径静默忽略 = 假实验。
        if not (enum_mode and bool(config.get("v7_ar2", False))
                and not bool(config.get("v7_ar2_fb", True))):
            raise ValueError("--t-prefix-p requires enum training on an AR2 model "
                             "without feedback (ar2_fb=False)")
        if int(config.get("obs_context", config["tw"])) <= int(config["tw"]):
            raise ValueError("--t-prefix-p requires obs_context > tw (no history, no prefix)")
        if (bool(config.get("ar_rollout", False)) or float(config.get("ar_ss", 0.0)) > 0
                or float(config.get("ar_noise", 0.0)) > 0
                or float(config.get("ctx_ss", 0.0)) > 0):
            raise ValueError("--t-prefix-p is incompatible with --ar-rollout/--ar-ss/"
                             "--ar-noise/--ctx-ss")
    if args.v7_ar2_lowdim:
        config["v7_ar2_lowdim"] = True
    if args.v7_ar2_fb_dim is not None:
        config["v7_ar2_fb_dim"] = int(args.v7_ar2_fb_dim)
    for key, value in (("p_occ", args.p_occ), ("p_nullV", args.p_nullv),
                       ("p_nullT", args.p_nullt)):
        if value is not None:
            config[key] = float(value)
    # V6_Av1 occ_mix (2026-10-19)：训练侧增强，默认关。启用后 train/val
    # dataset 均按标签采样（val 同挂标签供 A3 分层监控）。必须在本 args
    # →config 段之后计算（早于 --occ-mix 映射会恒为 None——2026-10-20
    # 冒烟事故根因）。
    occ_mix_kwargs = None
    if bool(config.get("occ_mix", False)):
        occ_mix_kwargs = {
            "p_occ": float(config.get("p_occ", 0.2)),
            "p_nullV": float(config.get("p_nullV", 0.1)),
            "p_T": float(config.get("p_nullT", 0.1)),
        }
    # 阶段 3 触觉泛化增强（2026-10-31）：训练侧 T_raw 增强，默认关。
    # tactile_aug dict 直接传给 AnySoleDataset（train 侧），val/eval 不传。
    tactile_aug_kwargs = None
    if args.v7_ar2_yaw_fix:
        config["v7_ar2_yaw_fix"] = True
    if args.ar2_dyaw_base is not None:
        config["ar2_dyaw_base"] = str(args.ar2_dyaw_base)
    if args.tactile_gainbias:
        tactile_aug_kwargs = dict(tactile_aug_kwargs or {},
                                  gain_range=[0.7, 1.3], bias_frac=0.05)
    if args.tactile_weight_norm:
        config["tactile_weight_norm"] = True
    if getattr(args, "tactile_frame_dropout", None) is not None:
        tactile_aug_kwargs = dict(tactile_aug_kwargs or {},
                                  frame_dropout=float(args.tactile_frame_dropout))
    config["motion_protocol"] = MOTION_PROTOCOL
    config["motion_n_joints"] = int(N_JOINTS)
    config["motion_pose_dim"] = int(POSE_DIM)
    config["joint_protocol_checksum"] = JOINT_PROTOCOL_CHECKSUM

    from anysole.data.contact_package import VERSIONED_CONTACT_METHODS, corrected_contact_provenance
    if config["contact_method"] in VERSIONED_CONTACT_METHODS:
        config.update(corrected_contact_provenance(config["contact_method"]))

    try:
        dataset = AnySoleDataset(
            mode="train",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=int(config["tw"]),
            session_ids=session_ids,
            contact_method=str(config["contact_method"]),
            stride=config.get("stride"),
            tactile_input=str(config.get("tactile_input", "raw108")),
            tactile_weight_norm=bool(config.get("tactile_weight_norm", False)),
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            f2_repr=bool(config.get("f2_repr", False)),
            smpl_roots=config.get("smpl_roots"),
            cam_id=int(config.get("cam_id", 3)),
            v7_ctx=bool(config.get("v7_prev_ctx", False)),
            v7_ctx3=bool(config.get("v7_ctx3", False) or config.get("v7_ctx4", False) or config.get("v7_ctx5", False)),
            v7_ctx5=bool(config.get("v7_ctx5", False)),
            v7_ar=bool(config.get("v7_ar", False)),
            v7_ar2=bool(config.get("v7_ar2", False)),
            obs_context=config.get("obs_context"),
            load_v5_1_gt=v5_1_mode,
            occ_mix=occ_mix_kwargs,
            t_phys_v2=bool(config.get("t_phys_v2", False)),
            d8_heads=bool(config.get("d8_heads", False)),
            tactile_aug=tactile_aug_kwargs,
            smpl_support_policy=str(config.get("smpl_support_policy", "exclude")),
            v7_s_label=config.get("v7_s_label"),
        )
    except FileNotFoundError as exc:
        if "HRNet cache" in str(exc):
            raise FileNotFoundError(
                "%s Run `python -m anysole.data.extract_hrnet --cam-id %d`; "
                "training never extracts features on demand."
                % (exc, int(config["cam_id"]))
            ) from exc
        raise
    if len(dataset) == 0:
        raise RuntimeError("Training dataset contains no valid windows")

    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    try:
        val_dataset = AnySoleDataset(mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]), cache_root=Path(config["cache_root"]), window_length=int(config["tw"]), contact_method=str(config["contact_method"]), tactile_input=str(config.get("tactile_input", "raw108")), no_imu=bool(config.get("no_imu", False)), v_input=str(config.get("v_input", "hrnet")), f2_repr=bool(config.get("f2_repr", False)), smpl_roots=config.get("smpl_roots"), cam_id=int(config.get("cam_id", 3)), load_v5_1_gt=v5_1_mode, occ_mix=occ_mix_kwargs, t_phys_v2=bool(config.get("t_phys_v2", False)), d8_heads=bool(config.get("d8_heads", False)),
        tactile_weight_norm=bool(config.get("tactile_weight_norm", False)),
        v7_ctx=bool(config.get("v7_prev_ctx", False)),
        v7_ctx3=bool(config.get("v7_ctx3", False) or config.get("v7_ctx4", False) or config.get("v7_ctx5", False)),
        v7_ctx5=bool(config.get("v7_ctx5", False)),
        v7_ar=bool(config.get("v7_ar", False)),
        v7_ar2=bool(config.get("v7_ar2", False)),
        obs_context=config.get("obs_context"),
        smpl_support_policy=str(config.get("smpl_support_policy", "exclude")))
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), collate_fn=collate_windows, pin_memory=device.type == "cuda")
    except (FileNotFoundError, RuntimeError) as exc:
        if v5_1_mode and "V5_Av1 GT CoP cache" in str(exc):
            # V5_Av1：z 通道 GT 缺失时 val 必须失败而非静默禁用——ckpt_best
            # 只由 val 面板选出，静默禁用 = 整轮训练无 best 快照。
            raise
        if occ_mix_kwargs and "occ010 cache" in str(exc):
            # V6_Av1：occ 缓存缺失时 val 同样必须失败（ckpt_best 只由 val
            # 面板选出；occ_mix 增强运行不允许静默退化为普通 val）。
            raise
        print("validation disabled: %s" % exc)
    monitor_run = None
    if str(config.get("wandb_mode", "disabled")) != "disabled":
        # Identity fields consumed by Wandb_Analyzer: experiment_tag groups a
        # training batch, run_kind/model/stage/fold decide the raw/ directory
        # tree. Single-stage, non-cross-validation runs use stage=single,
        # fold=fold0.
        monitor_identity = {
            "experiment_tag": str(config.get("wandb_experiment_tag") or "anysole_v1"),
            "run_kind": "train",
            # run 名称用模型编号（用户裁定 2026-10-25）——model_name 即
            # 注册表编号（如 V7_Av1），不再是笼统的 modal（anysole_v2）。
            "model": str(config.get("model_name")
                         or config.get("modal", MODEL_ANYSOLEV1)),
            "stage": "single",
            "fold": "fold0",
        }
        monitor_name = "%s_%s_%s" % (monitor_identity["model"], monitor_identity["stage"], monitor_identity["fold"])
        monitor_group = "%s_%s" % (monitor_identity["experiment_tag"], monitor_identity["run_kind"])
        monitor_tags = [monitor_identity[k] for k in
                        ("experiment_tag", "run_kind", "model", "stage", "fold")]
        monitor_config = {**dict(config), **monitor_identity}
        results_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results")))
        monitor_backend = str(config.get("monitor_backend") or "wandb")
        if monitor_backend == "swanlab":
            try:
                import swanlab
            except ImportError as exc:
                raise RuntimeError("SwanLab monitoring requested but swanlab is not installed") from exc
            # Local run data lives under results/logs/swanlab (not a repo-root
            # swanlog/). SwanLab 0.7.x（touch_gait 是 Python 3.8）只有 mode 名
            # 一处差异（online→cloud）；tags 在云端落成 run 的 labels。
            monitor_mode = {"online": "cloud"}.get(
                str(config["wandb_mode"]), str(config["wandb_mode"]))
            monitor_run = swanlab.init(
                mode=monitor_mode,
                project=str(config.get("wandb_project") or "Anysole"),
                workspace=config.get("swanlab_workspace") or None,
                experiment_name=monitor_name,
                group=monitor_group,
                job_type="train",
                tags=monitor_tags,
                config=monitor_config,
                logdir=str(results_root / "logs" / "swanlab"),
            )
        elif monitor_backend == "wandb":
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("W&B monitoring requested but wandb is not installed") from exc
            os.environ.setdefault("WANDB_DIR", str(results_root / "logs" / "wandb"))
            monitor_run = wandb.init(
                mode=str(config["wandb_mode"]),
                project=str(config.get("wandb_project") or "Anysole"),
                entity=config.get("wandb_entity"),
                name=monitor_name,
                group=monitor_group,
                job_type="train",
                tags=monitor_tags,
                config=monitor_config,
            )
        else:
            raise ValueError("unknown monitor_backend: %s" % monitor_backend)
    model_kw = {}
    modal = str(config.get("modal", MODEL_ANYSOLEV1))
    regress_mode = modal == MODEL_ANYSOLEV2 and not v3f_mode
    # Keep model variants independent in the centralized results tree. An
    # explicit --out-dir remains authoritative for custom experiments.
    # Otherwise the dir is INFERRED (2026-09-22): model-name + contact-method
    # + stacked hyperparameter fields built from the effective config (or
    # pinned by --variant); every run trains from random init — no warm-start
    # source is ever inferred (--init-from = debug override only).
    if args.model_name is not None:
        contact_method = str(config["contact_method"])
        # Uniform hierarchy + complete hyperparameter record (2026-09-24):
        # the variant name = the full stacked fields of the effective config
        # (fixed order, no default omission) — every run gets its own dir,
        # including all-default ones; --variant pins it explicitly instead.
        variant = args.variant
        if variant is None:
            variant = variant_from_config(config)
        if variant is not None:
            config["variant"] = str(variant)
        inferred = infer_anysole_paths(args.model_name, variant=variant, contact=contact_method)
        if args.out_dir is None:
            config["out_dir"] = str(inferred["model_dir"] / "checkpoints")
        # Independence (2026-09-24): no warm-start is ever inferred — every
        # run starts from random init; --init-from remains a debug override.
    else:
        if args.out_dir is None:
            raise ValueError(
                "--out-dir omitted: pass --model-name (dir identifier, e.g. V3_4a) so the "
                "output dir is inferred as <model-name>_<contact>[/<stacked fields>]/checkpoints, "
                "or pass --out-dir explicitly for a custom experiment."
            )
    if v5_availability:
        require_fresh_output(config["out_dir"])
    if modal == MODEL_ANYSOLEV2:
        # Registered models are built from their entry script — the structure
        # is fixed by the model name (anysole/models/<module>.py), and the
        # entry's STRUCTURE/CONFIG_EXTRA were recorded into config above.
        # Regression pose head, no diffusion pair in forward.
        if entry is None:
            raise ValueError("anysolev2 requires --model-name (entry script defines the structure)")
        if bool(config.get("use_insole_drift", False)):
            raise ValueError("anysolev2 does not support insole drift compensation")
        part_joints = tuple(tuple(int(j) for j in g) for g in config["part_joints"]) \
            if config.get("part_joints") else None
        if str(config.get("pressure_head", "")) in ("query_nonnegative", "query_factorized") \
                and config.get("pressure_cell_mean") is None:
            cell_mean = fit_pressure_cell_means(loader, device)
            config["pressure_cell_mean"] = cell_mean
            print("pressure cell means: min %.2f max %.2f g (frozen in ckpt config)"
                  % (min(cell_mean), max(cell_mean)))
        model = entry.build(config, part_joints=part_joints).to(device)
        # Entries that compute build-time provenance (V3_Gv1 carrier sha256)
        # stash it in CONFIG_EXTRA during build — fold it into the config so
        # every checkpoint records it.
        config.update(getattr(entry, "CONFIG_EXTRA", {}) or {})
    else:
        if modal == "anysolev1_insole_drift" or bool(config.get("use_insole_drift", False)):
            templates, subject_map = load_template_bank(config["template_path"])
            model_kw.update(templates=templates, subject_to_index=subject_map)
        model = AnySoleModel(
            d=int(config["d_model"]),
            tw=int(config["tw"]),
            dropout=float(config.get("dropout", 0.1)),
            modal=modal,
            use_insole_drift=bool(config.get("use_insole_drift", False)),
            pose_layers=int(config.get("pose_layers", 6)),
            tactile_input=str(config.get("tactile_input", "raw108")),
            tactile_direct=bool(config.get("tactile_direct", False)),
            no_imu=bool(config.get("no_imu", False)),
            v_input=str(config.get("v_input", "hrnet")),
            t_encoder=str(config.get("t_encoder", "linear")),
            f2_repr=bool(config.get("f2_repr", False)),
            **model_kw,
        ).to(device)
    if args.init_from is not None:
        # Debug-only warm-start (independent-models rule: never inferred).
        # Loaded before fitting pose stats, so the normalization buffers are
        # refit on the current dataset regardless of what the source
        # checkpoint carried.
        init_from_checkpoint(
            model, args.init_from,
            drop_prefixes=tuple(p for p in args.init_drop.split(",") if p),
        )
    # Frozen pose-normalization stats (recorded so eval/infer rebuild the same depth).
    config["pose_layers"] = int(config.get("pose_layers", 6))
    # E6.1: pose_repr follows the modal so losses/eval agree with the head.
    # F0b: anysolev2 has no position-space variant — the regression head is 6D only.
    if modal == MODEL_ANYSOLEV2:
        config["pose_repr"] = "6d"
    else:
        config["pose_repr"] = "pos" if modal == MODEL_ANYSOLEV1_POS else str(config.get("pose_repr", "6d"))
    pos_mode = config["pose_repr"] == "pos"
    if pos_mode and float(config.get("lambda_pose_vel", 0.0)) == 0.0 and float(config.get("lambda_bone", 0.0)) == 0.0:
        # Guard against repeating the 2026-09-16 misconfigured E6.2 launches:
        # three runs went out with the E6.2 losses at 0 (yaml default) and
        # were indistinguishable from E6.1.  E6.2 without these losses is
        # E6.1 - make that impossible to miss.
        print("=" * 79)
        print("WARNING: pos modal with lambda_pose_vel=0 AND lambda_bone=0")
        print("         this trains plain E6.1, not E6.2 (smooth/rigidity off).")
        print("         pass --lambda-pose-vel 10.0 --lambda-bone 1.0 for E6.2.")
        print("=" * 79)
    # noise_scaled is a config knob now: E3 base = False (unscaled N(0,1)
    # noise, the regime whose output was verified good), E4+ = True.  Saved in
    # every checkpoint so eval/infer match the sampling init scale.
    # V3_Fv1: the state is normalized before diffusing, so unit noise is the
    # matching scale (the E4 noise_scaled machinery applies to raw-pose
    # diffusion only).
    if v3f_mode:
        config["noise_scaled"] = False
    else:
        config["noise_scaled"] = bool(config.get("noise_scaled", True))
    model.noise_scaled = config["noise_scaled"]
    # Pose stats live on the pose head; guard kept defensively (the V3
    # structure always has one, but a missing head must fail loudly elsewhere,
    # not here).
    if getattr(model, "pose_head", None) is not None:
        _stats_res_w = 0.0
        _stats_first_abs = None
        if getattr(model, "ctx4", False):
            _probs = [float(x) for x in config["config_probs"]]
            _abs = _probs[CONFIG_VT] + _probs[CONFIG_V]
            _stats_res_w = _probs[CONFIG_T] / _abs if _abs > 0 else 0.0
        elif getattr(model, "ctx4b", False):
            # v4-b：残差全窗 + 绝对仅首窗（p_T 权重，与枚举损失配比一致）。
            _probs = [float(x) for x in config["config_probs"]]
            _stats_first_abs = _probs[CONFIG_T]
        pose_mean, pose_std = fit_pose_stats(
            loader, device, key="pose_gt_pos" if pos_mode else "pose_gt",
            residual_weight=_stats_res_w, first_abs_share=_stats_first_abs
        )
        model.pose_head.set_stats(pose_mean, pose_std)
        print("pose stats: mean %.4f +/- %.4f, std %.4f +/- %.4f (dims floored at 1e-2: %d/%d)"
              % (pose_mean.mean().item(), pose_mean.std().item(), pose_std.mean().item(),
                 pose_std.std().item(), int((pose_std <= 1e-2).sum().item()), pose_mean.shape[0]))
    if v3f_mode:
        state_mean, state_std = fit_state_stats(loader, device)
        model.denoiser.set_stats(state_mean, state_std)
        print("v3f state stats: mean %.4f +/- %.4f, std %.4f +/- %.4f (dims floored at 1e-2: %d/%d)"
              % (state_mean.mean().item(), state_mean.std().item(), state_std.mean().item(),
                 state_std.std().item(), int((state_std <= 1e-2).sum().item()), state_mean.shape[0]))
    if bool(config.get("f2_repr", False)):
        # F2a: per-dim traj stats balance psi_dot / v_h / h inside the
        # trajectory loss; saved with the checkpoint for eval reference.
        config["traj_f2_stats"] = fit_traj_stats(loader, device)
        print("traj f2 stats: mean %s std %s"
              % (config["traj_f2_stats"]["mean"], config["traj_f2_stats"]["std"]))
    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    # V4A clustering: part_logits (A) gets its own optimizer group with an lr
    # multiplier, so the assignment can move faster than the heads while they
    # co-train (soft EM).  The per-epoch/per-step lr setters below multiply
    # by each group's "assign_mult".
    # 双阶段精调（V4_Av4, 2026-10-25）：carrier_lr_mult != 1.0 时骨架
    # （B backbone，模型暴露 carrier_param_ids）与其他模块分组——骨架组
    # assign_mult = carrier_lr_mult（0.1），其余 1.0。与 V4A 的 assign_lr_mult
    # 分组互斥（后者只对 soft_parts 的 A 矩阵生效；两者不同时出现）。
    assign_lr_mult = float(config.get("assign_lr_mult", 1.0))
    carrier_lr_mult = float(config.get("carrier_lr_mult", 1.0))
    if carrier_lr_mult != 1.0 and hasattr(model, "carrier_param_ids"):
        carrier_ids = set(model.carrier_param_ids())
        carrier = [p for p in model.parameters() if id(p) in carrier_ids]
        rest = [p for p in model.parameters() if id(p) not in carrier_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": rest, "assign_mult": 1.0},
                {"params": carrier, "assign_mult": carrier_lr_mult},
            ],
            lr=float(config["lr"]),
        )
    elif assign_lr_mult != 1.0 and bool(config.get("soft_parts", False)):
        part_logits = model.pose_head.part_logits
        rest = [p for p in model.parameters() if p is not part_logits]
        optimizer = torch.optim.Adam(
            [
                {"params": rest, "assign_mult": 1.0},
                {"params": [part_logits], "assign_mult": assign_lr_mult},
            ],
            lr=float(config["lr"]),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    warn_legacy_output_checkpoint(out_dir)
    # Optional additive candidate pool. Default legacy runs execute the
    # unchanged best/last blocks below and create no additional pool/files.
    checkpoint_archive = None
    if config.get("checkpoint_policy", "legacy") == "multi_metric":
        if (config.get("modal") != MODEL_ANYSOLEV2 or bool(config.get("v3f", False))
                or str(config.get("pose_repr", "6d")) != "6d"):
            raise ValueError("multi_metric currently requires a 6D regression model with validation PA")
        if val_loader is None:
            raise ValueError("multi_metric requires a validation loader")
        checkpoint_archive = CandidateArchive(
            out_dir / "multi_metric", infer_role(config),
            v_pa_max=float(config.get("checkpoint_gate_v", 40.0)),
            t_pa_max=float(config.get("checkpoint_gate_t", 90.0)),
            vt_pa_max=float(config.get("checkpoint_gate_vt", 40.0)),
            keep_latest=not bool(config.get("checkpoint_all", False)),
        )
        print("additive candidate pool: role=%s, mode=%s, path=%s" %
              (checkpoint_archive.role,
               "latest" if checkpoint_archive.keep_latest else "full",
               checkpoint_archive.root))

    # V3 structural steps (fix_plan_v3 §2.2): linear LR warmup over the first
    # lr_warmup_frac of the total planned steps, multiplying the epoch cosine
    # LR. 0/absent = no warmup (F0b/F4a behavior).
    lr_warmup_frac = float(config.get("lr_warmup_frac", 0.0))
    warmup_steps = 0
    if lr_warmup_frac > 0:
        warmup_steps = int(lr_warmup_frac * int(config["epochs"]) * len(loader))
        print("lr warmup: %.4f x %d epochs x %d steps/epoch = %d steps"
              % (lr_warmup_frac, int(config["epochs"]), len(loader), warmup_steps))

    global_step = 0
    # V4A clustering schedule (方案 A): temperature/entropy anneal linearly
    # over the first assign_anneal_frac of total steps, then A is frozen for
    # the last (1 - assign_lock_frac) of the run so the heads settle against
    # a fixed grouping.  Non-cluster runs keep the defaults (temp 1.0, no
    # lock) so V3-3 behavior is byte-identical.
    cluster_mode = bool(config.get("assign_cluster", False))
    ctx_ss_cache = None  # G1 scheduled sampling 的模型预测 ctx 缓存（enum 分支也引用，须无条件初始化）
    ar_ss_cache = None   # AR 反馈 scheduled sampling 缓存（同 ctx_ss_cache 生命周期）
    if cluster_mode:
        total_steps = int(config["epochs"]) * len(loader)
        anneal_steps = int(float(config.get("assign_anneal_frac", 0.7)) * total_steps)
        lock_steps = int(float(config.get("assign_lock_frac", 0.7)) * total_steps)
        temp_init = float(config.get("assign_temp_init", 1.0))
        temp_final = float(config.get("assign_temp_final", 0.2))
        print("V4A cluster schedule: anneal %d steps (temp %.2f -> %.2f, entropy -> 0), "
              "lock A after step %d, A lr mult %.1f"
              % (anneal_steps, temp_init, temp_final, lock_steps, assign_lr_mult))
    else:
        anneal_steps = lock_steps = 0
        temp_init = temp_final = 1.0
    # Best-ckpt tracking (loss-explosion protection): the val eval cadence is
    # the only visibility into val quality, so the best-so-far model by
    # val/tau0_mpjpe/VT is snapshotted to ckpt_best.pt — a finite-gradient
    # explosion (F0b_warm ep370 / V3_2_part9 ep271, no nonfinite guard fires)
    # can destroy ckpt_last.pt in ~10 epochs; the snapshot survives.
    # V3-3 sigma freeze schedule (plan §3, "first sigma_freeze_frac sigma
    # fixed"): sigma stays at 0 while the tau_p prior bias shapes the gate,
    # then comes online for the remainder of the run.
    if str(config.get("gate", "none")) != "none":
        sigma_freeze_steps = int(
            float(config.get("sigma_freeze_frac", 0.1))
            * len(loader) * int(config["epochs"])
        )
    else:
        sigma_freeze_steps = 0
    best_vt = None
    for epoch in range(int(config["epochs"])):
        model.train()
        # E4 cosine LR decay over the whole run; E3 base uses lr_schedule
        # "constant" (E3's regime, whose output was verified good).
        if str(config.get("lr_schedule", "cosine")) == "constant":
            epoch_lr = float(config["lr"])
        else:
            progress = epoch / float(config["epochs"])
            epoch_lr = float(config["lr"]) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr * float(group.get("assign_mult", 1.0))
        epoch_start_step = global_step
        nonfinite_skips = 0
        grad_probe = _grad_probe_groups(model)
        grad_probe_counts = {name: sum(param.numel() for param in params)
                             for name, params in grad_probe.items()}
        epoch_sums = {"loss_total": 0.0, "loss_pose": 0.0, "loss_traj": 0.0, "loss_con": 0.0, "loss_kp": 0.0, "loss_pos": 0.0, "loss_pose_vel": 0.0, "loss_bone": 0.0, "loss_assign": 0.0, "loss_sigma": 0.0, "loss_contact": 0.0, "loss_phase": 0.0, "loss_cop": 0.0, "loss_Trec_Tmissing": 0.0, "loss_Vrec_Vmissing": 0.0, "grad_norm": 0.0, "n": 0, "n_tmissing": 0, "n_vmissing": 0}
        # 梯度探针在 GPU 上累加，每个 epoch 只回读一次：逐 batch 的
        # float() 会强制同步、把异步流水线打散（实测 7 步/epoch 慢 ~10 倍）。
        grad_probe_acc = {}
        for _group in grad_probe:
            grad_probe_acc[_group] = torch.zeros((), device=device)
            grad_probe_acc[_group + "_rms"] = torch.zeros((), device=device)
        availability_counts = {k: 0 for k in ("windows", "V", "T", "VT", "neither")}
        # 条件契约运行阈值（V5_Av5/V4_Av2，2026-10-25）：V-only 窗口误差的
        # 批中位数 EMA（只统计 V 在场窗口；occ 窗口误差自然抬高、中位数
        # 稳健）。首批直接取中位数，其后 α=0.05 慢跟踪。
        cond_tau = None
        # V6_Av1 A1/A2 窗源分层累计（occ_mix 启用时；key -> [sum, count]）。
        occ_acc: Dict[str, tuple] = {}
        # epoch_sums 的损失键映射：L_corr_prior 只在 v4_1_mode 存在
        # （compute_losses_v4_1 独有）——按模式隔离，其余模式损失 dict
        # 无此键，混入会 KeyError（2026-10-20 V5_Av1 队列竞态事故）。
        loss_pairs = [("loss", "loss_total"), ("L_pose", "loss_pose"),
                      ("L_traj", "loss_traj"), ("L_con", "loss_con"),
                      ("L_kp", "loss_kp"), ("L_pos", "loss_pos"),
                      ("L_pose_vel", "loss_pose_vel"), ("L_bone", "loss_bone"),
                      ("L_assign", "loss_assign"), ("L_sigma", "loss_sigma"),
                      ("L_contact", "loss_contact"), ("L_phase", "loss_phase"),
                      ("L_cop", "loss_cop"),
                      ("L_tcop", "loss_tcop"), ("L_strike", "loss_strike"),
                      ("L_gtcop", "loss_gtcop"), ("L_cond", "loss_cond")]
        if v4_1_mode:
            loss_pairs.append(("L_corr_prior", "loss_corr_prior"))
            epoch_sums["loss_corr_prior"] = 0.0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            batch_size = batch["pose_gt"].shape[0]
            config_id = (torch.full((batch_size,), CONFIG_VT, device=device, dtype=torch.long)
                         if enum_mode else sample_config_ids(batch_size, probs=config["config_probs"]).to(device))
            batch["config_id"] = config_id
            assert_batch_shapes(batch, batch_size, tw=int(config["tw"]),
                        t_phys_v2=bool(config.get("t_phys_v2", False)))
            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
            # V6_Av1 occ_mix：condition_inputs 之后叠加整窗置零（occ_src 已
            # 在 dataset 侧选定 V_feat 来源；vz/tz 为窗口级标签）。
            if occ_mix_kwargs and batch.get("occ_v_zeroed") is not None:
                vz = batch["occ_v_zeroed"].view(-1, 1, 1).to(v_feat.device, dtype=torch.bool)
                tz = batch["occ_t_zeroed"].view(-1, 1, 1).to(v_feat.device, dtype=torch.bool)
                v_feat = torch.where(vz, torch.zeros_like(v_feat), v_feat)
                t_raw = torch.where(tz, torch.zeros_like(t_raw), t_raw)
                t_phys = torch.where(tz, torch.zeros_like(t_phys), t_phys)
                t_s2m = torch.where(tz, torch.zeros_like(t_s2m), t_s2m)
            # F1: hmr_gvhmr checkpoints consume the GVHMR channel; V1 models
            # have no V_hmr parameter.
            v_kw = {"V_hmr": batch.get("V_hmr")} if str(config.get("v_input", "hrnet")) == "hmr_gvhmr" else {}

            if cluster_mode:
                # V4A per-step schedule: anneal the assignment temperature
                # (soft -> sharp) and the L_assign entropy scale, and freeze
                # part_logits (A) after the lock point (the M step).
                anneal_prog = min(1.0, global_step / float(max(anneal_steps, 1)))
                model.pose_head.set_assign_temp(temp_init + (temp_final - temp_init) * anneal_prog)
                config["assign_anneal"] = 1.0 - anneal_prog
                if lock_steps > 0:
                    model.pose_head.part_logits.requires_grad_(global_step < lock_steps)

            optimizer.zero_grad(set_to_none=True)
            if sigma_freeze_steps:
                model.pose_head.set_sigma_frozen(global_step < sigma_freeze_steps)
            if enum_mode:
                if ctx_ss_cache is None:
                    ctx_ss_cache = [None]
                if ar_ss_cache is None:
                    ar_ss_cache = [None]
                # 2026-11-01 修复：total_steps 仅 V4A 聚类分支定义——
                # SS 概率斜坡改用本处局部总量（ARSS 首跑因此崩过）。
                _total = int(config["epochs"]) * len(loader)
                ctx_ss_target = float(config.get("ctx_ss", 0.0))
                p_cur = (ctx_ss_target * min(1.0, global_step / max(1, _total))
                         if ctx_ss_target > 0 else 0.0)
                ar_ss_target = float(config.get("ar_ss", 0.0))
                ar_p_cur = (ar_ss_target * min(1.0, global_step / max(1, _total))
                            if ar_ss_target > 0 else 0.0)
                ar_noise = float(config.get("ar_noise", 0.0))
                # 修改 04 改动 A（2026-11-03）：rollout 课程——L 从 1 线性
                # 升到 L_max（斜坡占 ar_rollout_schedule_frac 总步数，其后
                # 固定 L_max）；与 SS/噪声互斥（backward_v7_enum 内硬门）。
                ar_rollout_len = 0
                if bool(config.get("ar_rollout", False)):
                    if ar_ss_target > 0 or ar_noise > 0:
                        raise ValueError(
                            "--ar-rollout is incompatible with --ar-ss/--ar-noise")
                    ar_rollout_len = _ar_rollout_curriculum(
                        global_step, _total, int(config["tw"]),
                        l_max=int(config.get("ar_rollout_max", 8)),
                        schedule_frac=float(
                            config.get("ar_rollout_schedule_frac", 0.6)))
                out, losses = backward_v7_enum(model, batch, config,
                                               ctx_ss_p=p_cur,
                                               ctx_cache=ctx_ss_cache,
                                               ar_ss_p=ar_p_cur,
                                               ar_noise=ar_noise,
                                               ar_cache=ar_ss_cache,
                                               ar_rollout_len=ar_rollout_len)
            elif v3f_mode:
                # V3_Fv1: diffuse the full normalized state [pose | trans |
                # contact]; 10% V zeroing is the unconditional prior pass
                # (no learned null tokens — 方案 §2.1).
                x0_n = model.denoiser.normalize(torch.cat(
                    [batch["pose_gt"], batch["trans_gt"], batch["contact_gt"]], dim=-1))
                tau = torch.randint(
                    0, int(config.get("tau_max", config["diffusion_train_steps"])),
                    (batch_size,), device=device, dtype=torch.long)
                x_tau = diffusion.q_sample(x0_n, tau)
                v_drop = torch.rand(batch_size, device=device) < float(config.get("v_cond_dropout", 0.1))
                v_feat_d = v_feat.clone()
                v_feat_d[v_drop] = 0.0
                out = model(v_feat_d, t_raw, t_phys, x_tau, tau, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            elif regress_mode:
                # F0b: regression head — no tau sampling, no q_sample, no
                # x_tau. L_pose applies directly to the regression output
                # (losses.py unchanged).
                if v3h_mode and float(config.get("v_cond_dropout", 0.0)) > 0.0:
                    # V3_Hv1: V zeroing trains the null-V prior (the T2M
                    # protocol row — H has no T tower, 方案 §3; no learned
                    # null tokens).
                    v_drop = torch.rand(batch_size, device=device) < float(config["v_cond_dropout"])
                    v_feat = torch.where(v_drop.view(-1, 1, 1),
                                         torch.zeros_like(v_feat), v_feat)
                if v5_1_mode:
                    # legacy keeps the historical random-null-V-only mask.
                    # availability_v1 also records config/occ missing inputs.
                    v_drop = torch.rand(batch_size, device=device) < float(
                        config.get("v_cond_dropout", 0.1))
                    if v5_availability:
                        (v_feat, t_raw, t_phys, t_s2m), masks = apply_availability(
                            (v_feat, t_raw, t_phys, t_s2m), config_id, batch, v_drop)
                        batch.update(masks)
                        if v_kw.get("V_hmr") is not None:
                            v_kw["V_hmr"] = torch.where(masks["v_available"][:, None, None],
                                                       v_kw["V_hmr"], torch.zeros_like(v_kw["V_hmr"]))
                        va, ta = masks["v_available"], masks["t_available"]
                        availability_counts["windows"] += batch_size
                        for key, mask in (("V", va), ("T", ta), ("VT", va & ta), ("neither", ~(va | ta))):
                            availability_counts[key] += int(mask.sum())
                        if not bool((va | ta).any()):
                            # No observed input can supervise a branch. Skip
                            # the optimizer too (including any old momentum).
                            continue
                    else:
                        if bool(v_drop.any()):
                            v_feat = torch.where(v_drop.view(-1, 1, 1),
                                                 torch.zeros_like(v_feat), v_feat)
                        batch["v_zero"] = v_drop
                z_kw = {"z_src": "all"} if v5_1_mode else {}
                out = model(v_feat, t_raw, t_phys, config_id, batch.get("session_id"), T_s2m=t_s2m, **v_kw, **z_kw,
                            **model_observation_kwargs(batch))
            else:
                # Test11 τ training-regime knobs: --tau-fixed pins every step to one
                # noise level (0 = identity-mapping training), --tau-max narrows the
                # sampling band to U(0, tau_max) (low-noise region training).
                if "tau_fixed" in config:
                    tau = torch.full(
                        (batch_size,),
                        int(config["tau_fixed"]),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    tau = torch.randint(
                        0,
                        int(config.get("tau_max", config["diffusion_train_steps"])),
                        (batch_size,),
                        device=device,
                        dtype=torch.long,
                    )
                # E4: diffuse with per-dim noise scaled by the pose std (Step2Motion
                # normalizes the data before diffusing; scaling the noise is the
                # same thing here since the head normalizes its input internally).
                # Without this, N(0,1) noise lands on a signal of std ~0.118 (8.5x
                # SNR mismatch) and the DDIM chain injects per-frame jitter.
                pose_target = batch["pose_gt_pos"] if pos_mode else batch["pose_gt"]
                noise = torch.randn_like(pose_target)
                if model.noise_scaled:
                    noise = noise * model.pose_head.pose_std.view(1, 1, -1)
                x_tau = diffusion.q_sample(pose_target, tau, noise=noise)
                out = model(v_feat, t_raw, t_phys, x_tau, tau, config_id,
                            batch.get("session_id"), T_s2m=t_s2m, **v_kw)
            if float(config.get("lambda_cond", 0.0)) > 0.0 and (v5_1_mode or v4_1_mode):
                # 契约触发器：V-only 误差（v5 = x0_v 行；v4 = V 读出
                # x0_hat）批中位数更新 τ。T-only 窗口 V 已零化 → 剔除。
                with torch.no_grad():
                    pred_v = out["x0_v"] if v5_1_mode else out["x0_hat"]
                    e_v = (pred_v - batch["pose_gt"]).square().mean(dim=(-1, -2))
                    v_present = config_id != CONFIG_T
                    if v5_availability:
                        v_present = batch["v_available"] & batch["t_available"]
                    if bool(v_present.any()):
                        e_v = e_v[v_present]
                    if e_v.numel():
                        med = float(e_v.median())
                        cond_tau = med if cond_tau is None else 0.95 * cond_tau + 0.05 * med

            if enum_mode:
                pass  # Losses and backward were accumulated one condition at a time.
            elif v3f_mode:
                losses = compute_losses_v3f(out, x0_n, batch, config)
            elif v5_1_mode:
                losses = compute_losses_v5_1(out, batch, config_id, config, cond_tau=cond_tau)
            elif v4_1_mode:
                losses = compute_losses_v4_1(out, batch, config_id, config, cond_tau=cond_tau)
            else:
                losses = compute_losses(out, batch, config_id, config)

            finite = torch.isfinite(losses["loss"]) and torch.isfinite(out["F"]).all()
            if not bool(finite):
                for group in optimizer.param_groups:
                    group["lr"] = 1.0e-4
                optimizer.zero_grad(set_to_none=True)
                print("non-finite fusion/loss at step %d; lr set to 1e-4, batch skipped" % global_step)
                nonfinite_skips += 1
                global_step += 1
                continue

            # V6_Av1 A1/A2 训练窗分层统计（occ_mix 启用时）：T 分支
            # per-sample 损失（接触 BCE + 压力重建，未加权）与 L_pose
            # per-sample 按窗源标签分组（occ vs cam3）。train.py 侧重算
            # （losses.py 的 per_sample_contact_bce 同公式，注明）。
            if occ_mix_kwargs and batch.get("occ_src") is not None:
                with torch.no_grad():
                    gt = batch["contact_gt"]
                    pos = gt.float().mean().clamp(1e-4, 1.0 - 1e-4)
                    pos_weight = ((1.0 - pos) / pos).clamp(0.05, 20.0)
                    air_w = float(config.get("lambda_air", 0.0))
                    tbranch = per_sample_contact_bce(
                        out["contact_logits"], gt, pos_weight, air_w)
                    if out.get("pressure_hat") is not None:
                        tbranch = tbranch + (out["pressure_hat"] - batch["T_raw"]).square().flatten(1).mean(dim=1)
                    lpose_ps = (out["x0_hat"] - batch["pose_gt"]).square().flatten(1).mean(dim=1)
                    src = batch["occ_src"].to(device)
                    vz_l = batch["occ_v_zeroed"].to(device)
                    tz_l = batch["occ_t_zeroed"].to(device)
                    t_stats = grouped_mean(tbranch, src, vz_l, tz_l)
                    p_stats = grouped_mean(lpose_ps, src, vz_l, tz_l)
                    for name, groups in (("tbranch", t_stats), ("lpose", p_stats)):
                        for src_name in ("cam3", "occ"):
                            cell = groups[src_name]
                            if cell["mean"] is not None:
                                key = "%s%s" % (name, src_name)
                                occ_acc[key] = (occ_acc.get(key, (0.0, 0))[0] + float(cell["sum"]),
                                               occ_acc.get(key, (0.0, 0))[1] + int(cell["n"]))
                    occ_acc["frac_occ"] = (occ_acc.get("frac_occ", (0.0, 0))[0]
                                           + float((src == SRC_OCC).sum().item()), batch_size)
                    occ_acc["frac_nullV"] = (occ_acc.get("frac_nullV", (0.0, 0))[0]
                                             + float(vz_l.sum().item()), batch_size)
                    occ_acc["frac_nullT"] = (occ_acc.get("frac_nullT", (0.0, 0))[0]
                                             + float(tz_l.sum().item()), batch_size)
                    occ_acc["frac_undisturbed"] = (occ_acc.get("frac_undisturbed", (0.0, 0))[0]
                                                   + float(((src == SRC_CAM3) & (vz_l == 0) & (tz_l == 0)).sum().item()),
                                                   batch_size)
            # V5_Av1 z 通道监控（batch 级累计，无梯度）：专家分歧三通道 +
            # PoE 接触读数塌缩信号。监控键，不入损失、不进 ckpt。
            if v5_1_mode:
                with torch.no_grad():
                    zs = out["z"]
                    mu_t, mu_v = zs["mu_gT"], zs["mu_gV"]
                    prob_t = torch.sigmoid(zs["logit_gT"])
                    prob_v = torch.sigmoid(zs["logit_gV"])
                    prob_p = torch.sigmoid(zs["logit_poe"])
                    pm = batch["phase_mask"].to(mu_t.device).float()
                    cm = batch["cop_mask"].to(mu_t.device).float()
                    d_phase = (mu_t[..., 0:4] - mu_v[..., 0:4]).abs().reshape(
                        *mu_t.shape[:2], 2, 2).mean(dim=-1)     # (B,tw,2)
                    d_cop = (mu_t[..., 4:8] - mu_v[..., 4:8]).abs().reshape(
                        *mu_t.shape[:2], 2, 2).mean(dim=-1)
                    gt_b = batch["contact_gt"].to(prob_p.device) > 0.5
                    pred = prob_p > 0.5
                    tp = (pred & gt_b).sum()
                    fp = (pred & ~gt_b).sum()
                    tn = (~pred & ~gt_b).sum()
                    z_now = {
                        "z_div_contact": (prob_t - prob_v).abs().mean(),
                        "z_div_phase": (d_phase * pm).sum() / pm.sum().clamp(min=1),
                        "z_div_cop": (d_cop * cm).sum() / cm.sum().clamp(min=1),
                        "z_poe_prob_std": prob_p.std(),
                        "z_air_recall": tn / (tn + fp).clamp(min=1),
                        "z_logsig_gap": (zs["log_sig_gT"] - zs["log_sig_gV"]).mean(),
                    }
                    for key, value in z_now.items():
                        epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value) * batch_size
            if not enum_mode:
                losses["loss"].backward()
            # NaN-grad guard (fix_plan_v3 §2.2): check every parameter BEFORE
            # the clip, because clip_grad_norm_ mixes one non-finite gradient
            # into every parameter (F5B_gated failure mode).  Same treatment
            # as the loss-level guard above: skip the step, drop the lr.
            grads_list = [
                p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None
            ]
            grads_flat = torch.cat(grads_list) if grads_list else torch.zeros(0, device=device)
            grads_ok = bool(torch.isfinite(grads_flat).all())
            if not grads_ok:
                for group in optimizer.param_groups:
                    group["lr"] = 1.0e-4
                optimizer.zero_grad(set_to_none=True)
                print("non-finite gradient at step %d; lr set to 1e-4, batch skipped" % global_step)
                nonfinite_skips += 1
                global_step += 1
                continue
            # 分支挤压探针：在 clip（就地缩放）之前取原始梯度范数，
            # 否则所有组都被压到同一个上限，看不出相对量级。
            for _name, _params in grad_probe.items():
                _grads = [param.grad for param in _params if param.grad is not None]
                if not _grads:
                    continue
                _norm = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(_grads)))
                grad_probe_acc[_name] += _norm * batch_size
                grad_probe_acc[_name + "_rms"] += (
                    _norm / math.sqrt(max(grad_probe_counts[_name], 1)) * batch_size)
            grad_clip = config.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            # clip 就地缩放梯度，须重建扁平列表才能读到缩放后的值。
            grads_list = [
                p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None
            ]
            grad_sq = torch.cat(grads_list).square().sum() if grads_list else torch.zeros((), device=device)
            epoch_sums["grad_norm"] += float(torch.sqrt(grad_sq).item()) * batch_size
            if warmup_steps > 0:
                # Per-step linear ramp over the cosine epoch LR (the epoch
                # value is set on the groups at the top of the loop).
                ramp = min(1.0, global_step / float(warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = epoch_lr * ramp * float(group.get("assign_mult", 1.0))
            optimizer.step()
            n = batch_size
            epoch_sums["n"] += n
            present_losses = [(key, name) for key, name in loss_pairs if key in losses]
            if present_losses:
                loss_values = torch.stack(
                    [losses[key].detach().float() for key, _ in present_losses]
                ).cpu().tolist()
                for (key, name), value in zip(present_losses, loss_values):
                    epoch_sums[name] = epoch_sums.get(name, 0.0) + value * n
            n_tmissing = int((config_id == CONFIG_T).sum().item())
            n_vmissing = int((config_id == CONFIG_V).sum().item())
            if enum_mode:
                n_tmissing = batch_size if config["config_probs"][CONFIG_T] > 0 else 0
                n_vmissing = batch_size if config["config_probs"][CONFIG_V] > 0 else 0
            if n_tmissing:
                epoch_sums["loss_Trec_Tmissing"] += float(losses["L_Trec"].detach().item()) * n_tmissing
                epoch_sums["n_tmissing"] += n_tmissing
            if n_vmissing:
                epoch_sums["loss_Vrec_Vmissing"] += float(losses["L_Vrec"].detach().item()) * n_vmissing
                epoch_sums["n_vmissing"] += n_vmissing
            if global_step % 50 == 0:
                values = " ".join("%s=%.6f" % (key, value.detach().item()) for key, value in losses.items())
                print("epoch=%d step=%d %s" % (epoch + 1, global_step, values))
            global_step += 1
        if v5_availability:
            if epoch_sums["n"] == 0:
                raise RuntimeError("availability_v1 epoch has no usable optimizer updates; check input/dropout settings")
            print("v5 availability epoch %d: %s" % (epoch + 1, availability_counts))
        if occ_mix_kwargs and occ_acc:
            parts = ["%s=%.4f(n=%d)" % (key, total / max(count, 1), count)
                     for key, (total, count) in occ_acc.items()]
            print("occ_mix epoch %d: %s" % (epoch + 1, " ".join(parts)))
        # V5_Av1 机制探针（epoch 级）：film/* = FiLM 偏离恒等幅度与梯度
        # （兑现瓶颈定位——FiLM 是否收到梯度、是否被推回恒等）；poe/w_* =
        # PoE 接触合成权重（精度加权是否分化）。z_* 均值见 wandb 块。
        v5_1_probes = {}
        if v5_1_mode:
            with torch.no_grad():
                film = model.film_layer.mlp
                v5_1_probes = {
                    "film/gamma_dev": float((film.bias[: model.d] - 1.0).abs().mean()),
                    "film/beta_dev": float(film.bias[model.d:].abs().mean()),
                    "film/w_norm": float(film.weight.norm()),
                    "film/w_grad_norm": float(film.weight.grad.norm())
                    if film.weight.grad is not None else 0.0,
                    "poe/w_V": float(F.softplus(model.logit_w)[0]),
                    "poe/w_T": float(F.softplus(model.logit_w)[1]),
                }
            denom_z = max(epoch_sums["n"], 1)
            print("v5_1 epoch %d: %s  %s" % (epoch + 1,
                  "  ".join("%s=%.4f" % (key, epoch_sums[key] / denom_z)
                            for key in _V5_Av1_Z_KEYS),
                  "  ".join("%s=%.4f" % kv for kv in v5_1_probes.items())))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": dict(config),
        }
        # V4A: assign_anneal is per-step scratch for the L_assign entropy
        # scale, not a static config value — don't persist it.
        checkpoint["config"].pop("assign_anneal", None)
        # Atomic replace: a reader (auto-eval torch.load) must never see a
        # half-written file when several runs share one out_dir.  Each process
        # writes its own temp file, so a concurrent save cannot corrupt it.
        # Re-mkdir so an externally deleted results dir cannot kill a long run.
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "ckpt_last.pt"
        tmp_path = out_dir / ("ckpt_last.pt.tmp%d" % os.getpid())
        ckpt_interval = max(int(config.get("ckpt_interval", 5)), 1)
        if (epoch + 1) % ckpt_interval == 0 or (epoch + 1) == int(config["epochs"]):
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print("saved %s (epoch %d)" % (ckpt_path, epoch + 1))
        # Val eval cadence is the wall-clock knob: one _evaluate pass (3
        # condition configs x full val set, DDIM sampling) costs ~16 normal
        # epochs of wall-clock.  Triggered by --wandb_eval_interval but runs
        # regardless of wandb mode; results go to the console and, when a
        # wandb run is active, to its log.
        eval_results = None
        eval_interval = max(int(config.get("wandb_eval_interval", 5)), 1)
        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            model.eval()
            eval_results = _evaluate(model, diffusion, val_loader, config, device)
            if checkpoint_archive is not None:
                # Compatibility aliases remain intact. The explicit new name
                # describes the actual world-coordinate dashboard definition.
                for mode in ("V", "T", "VT"):
                    if "val/mpjpe/" + mode in eval_results:
                        eval_results["val/global_mpjpe/" + mode] = eval_results["val/mpjpe/" + mode]
            if occ_mix_kwargs and bool(config.get("occ_a3_monitor", False)):
                # V6_Av1 A3 分层监控（默认关）：窗口级 VT2M ≤ V2M 占比
                # occ vs cam3 对照（曲线分离方向）。
                eval_results.update(_occ_a3_monitor(model, val_loader, config, device))
            print("epoch %d eval: %s" % (epoch + 1,
                  "  ".join("%s=%.4f" % (key, value) for key, value in sorted(eval_results.items()))))
            vt_now = eval_results.get("val/tau0_mpjpe/VT")
            if vt_now is not None and (best_vt is None or vt_now < best_vt):
                best_vt = vt_now
                best_ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "config": dict(config),
                    "best_vt": best_vt,
                }
                best_path = out_dir / "ckpt_best.pt"
                best_tmp = out_dir / ("ckpt_best.pt.tmp%d" % os.getpid())
                torch.save(best_ckpt, best_tmp)
                os.replace(best_tmp, best_path)
                print("saved best %s (epoch %d, val/tau0 VT %.4f)" % (best_path, epoch + 1, best_vt))
            if checkpoint_archive is not None:
                try:
                    state = checkpoint_archive.update(checkpoint, eval_results)
                    print("candidate pool epoch %d: %d unique candidates; final selection pending full validation" %
                          (epoch + 1, len(state["candidates"])))
                except DashboardUnavailable as exc:
                    # Degenerate PA is undefined, not zero. Legacy saves have
                    # already completed; record the skip without stopping training.
                    checkpoint_archive.root.mkdir(parents=True, exist_ok=True)
                    with (checkpoint_archive.root / "skipped_validation.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"epoch": epoch + 1, "reason": str(exc)}) + "\n")
                    print("candidate pool skip epoch %d: %s" % (epoch + 1, exc))
        for _name in grad_probe:
            epoch_sums["grad_" + _name] = float(grad_probe_acc[_name])
            epoch_sums["grad_rms_" + _name] = float(grad_probe_acc[_name + "_rms"])
        if monitor_run is not None:
            denom = max(epoch_sums["n"], 1)
            log = {"epoch": epoch + 1}
            log["train/global_step"] = global_step
            log["train/steps_per_epoch"] = global_step - epoch_start_step
            log["train/nonfinite_skips"] = nonfinite_skips
            if v5_availability:
                for key, count in availability_counts.items():
                    log["train/v5_availability/" + key] = (count if key == "windows" else
                        count / max(availability_counts["windows"], 1))
            for key, value in epoch_sums.items():
                if key in ("n", "n_tmissing", "n_vmissing") or key.startswith("z_"):
                    continue
                metric_denom = denom
                if key == "loss_Trec_Tmissing": metric_denom = max(epoch_sums["n_tmissing"], 1)
                if key == "loss_Vrec_Vmissing": metric_denom = max(epoch_sums["n_vmissing"], 1)
                log[f"train/{key}"] = value / metric_denom
            log["train/lr"] = optimizer.param_groups[0]["lr"]
            if occ_mix_kwargs and occ_acc:
                # V6_Av1 A1/A2 分层日志键（epoch 级）：
                #   train/occ_a1/tbranch_{cam3,occ}  = T 分支损失（接触 +
                #       压力重建，未加权）per-sample 组均值；
                #   train/occ_a2/lpose_{cam3,occ}    = L_pose per-sample
                #       组均值（操纵检查：occ 窗 V 削弱应 > cam3 窗）；
                #   train/occ/frac_{occ,nullV,nullT,undisturbed} = 组合占比。
                for name, prefix in (("tbranch", "train/occ_a1/"), ("lpose", "train/occ_a2/")):
                    for src_name in ("cam3", "occ"):
                        key = "%s%s" % (name, src_name)
                        total, count = occ_acc.get(key, (0.0, 0))
                        log[prefix + key] = total / max(count, 1)
                for frac_key in ("frac_occ", "frac_nullV", "frac_nullT", "frac_undisturbed"):
                    total, count = occ_acc.get(frac_key, (0.0, 0))
                    log["train/occ/" + frac_key] = total / max(count, 1)
            # V3-3 gate probes (plan §2.3 F4): per-slot mean [g_V, g_T, g_∅]
            # from the last decoder layer, under wandb prefix "gate/".  The
            # assignment matrix itself stays auditable via the smoke/probe
            # scripts off the checkpoint's part_logits parameter.
            if str(config.get("gate", "none")) != "none":
                summary = model.pose_head.gate_summary()  # (N_PARTS, 3)
                for p, pname in enumerate(PART_NAMES):
                    log["gate/%s_gV" % pname] = float(summary[p, 0])
                    log["gate/%s_gT" % pname] = float(summary[p, 1])
                    log["gate/%s_gE" % pname] = float(summary[p, 2])
            # V4A cluster probes: per-slot loads + effective K so the grouping
            # emergence is watchable live (wandb prefix "assign/").  The final
            # grouping itself is audited off the checkpoint via
            # z_note/probes/probe_v4a_readout.py.
            if cluster_mode:
                assign = model.pose_head.assignment().detach()
                loads = assign.sum(dim=1)
                log["assign/effective_K_ge0.5"] = float((loads >= 0.5).sum())
                log["assign/effective_K_ge1.0"] = float((loads >= 1.0).sum())
                log["assign/max_load"] = float(loads.max())
                log["assign/mean_entropy"] = float(
                    -(assign * (assign + 1e-12).log()).sum(dim=0).mean()
                )
                for k in range(int(loads.shape[0])):
                    log["assign/load_%02d" % k] = float(loads[k])
            # V5_Av1 z 通道监控（wandb z/*）：专家分歧 + PoE 接触读数；
            # film/poe 探针来自 epoch 末（v5_1_probes）。
            if v5_1_mode:
                for key in _V5_Av1_Z_KEYS:
                    log["z/" + key[2:]] = epoch_sums[key] / denom
                log.update(v5_1_probes)
            # Optional capped logging (--loss-cap): the first epochs spike far
            # above the converged range (loss_total 30 -> 0.1, loss_pose 1.5 ->
            # 0.01), which stretches the raw charts' y-axis and hides
            # convergence.  When set, clip the train losses in place under
            # their original names so the wandb view stays within 0-cap and
            # Wandb_Analyzer picks up the clipped series directly.
            loss_cap = config.get("loss_cap")
            if loss_cap is not None:
                for key in ("loss_total", "loss_pose", "loss_traj", "loss_con", "loss_kp", "loss_pos",
                            "loss_Trec_Tmissing", "loss_Vrec_Vmissing", "grad_norm"):
                    full_key = "train/%s" % key
                    if full_key in log:
                        log[full_key] = min(float(log[full_key]), float(loss_cap))
            if eval_results is not None:
                log.update(eval_results)
            monitor_run.log(log, step=epoch + 1)
    if monitor_run is not None:
        monitor_run.finish()
    final_split = str(config.get("final_eval_split", "test"))
    if final_split == "none":
        return 0
    try:
        from anysole.eval import main as eval_main
        # Training-time auto-eval stays lean: metrics only and no protocol pass.
        eval_main(["--config", str(args.config), "--ckpt", str(out_dir / "ckpt_last.pt"),
                   "--split", final_split, "--device", str(device),
                   "--contact-method", str(config["contact_method"]), "--no-write-motion"])
    except Exception as exc:
        print("automatic test evaluation failed: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
