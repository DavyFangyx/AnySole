"""V3_Hv1 entry: single-generator regression + residual channels (2026-10-17).

Fresh architecture (2026-09-24 independence restructure): the structure is
fixed here; everything trains from random init except the frozen vonly
visual encoder.

Structure summary (model_v3h.py + constraints_h.py):
  V: v_enc (frozen vonly specialist) -> v_temporal -> 9-part pose head
     (144-dim 6D, no sigma branch) + traj head.  The only generator.
  T: foot_conv -> contact head (f6_soft, air-weighted BCE).  No pose
     tower — the T side emits evidence only.
  Inference: constraints_h.apply_v3h_constraints — E-v2.1 residual root
     lift + contact-gated ankle alignment (flat foot) + ankle-yaw support
     damping; floor_src "gt" main / "t" diagnostic.

The frozen vonly visual encoder (sanctioned reuse, 方案 §1): the entry
loads the per-config V-only specialist's ``encoders.v_enc.*`` projection
and freezes it (V3_Hv1 -> V3_Bv1_vonly, V3_Hv1_cam3_occ010 ->
V3_Bv1_cam3_occ010_vonly).  Embeddings stay fresh: time_pe is sinusoidal by
construction and the modality row trains with the T stream.  Everything
else — the temporal trunk, the heads, the contact stream — trains from
random init.
"""

from __future__ import annotations

import hashlib

import torch

from anysole.models.common.model_v3h import AnySoleModelV3H
from anysole.types import anysole_model_dir

STRUCTURE = {
    "t_encoder": "foot_conv",
    "v_input": "hrnet",
    "tactile_input": "raw108",
    "v3h": True,
}

# Provenance-only flags recorded into the ckpt config (not constructor args).
CONFIG_EXTRA: dict = {}


def _vonly_candidates(model_name: str) -> list:
    """Per-config frozen V specialist candidates (first existing wins):
    V3_Hv5_cam1_occ010 -> [V3_Bv1_cam1_occ010_vonly, V3_Bv1_cam1_vonly] —
    遮挡组合无专才时回落到该相机无遮挡 vonly（v_enc 是冻结特征提取器，
    遮挡条件由从零训练的主干适应）。"""
    name = str(model_name or "")
    suffix = ""
    for prefix in ("V3_Hv6", "V3_Hv5", "V3_Hv3", "V3_Hv2", "V3_Hv1"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            break
    candidates = ["V3_Bv1" + suffix + "_vonly"]
    import re as _re
    m = _re.match(r"^(_cam[1-4])_occ\d{3}$", suffix)
    if m:
        candidates.append("V3_Bv1" + m.group(1) + "_vonly")
    candidates.append("V3_Bv1_vonly")
    return candidates


def build(config: dict, part_joints=None):
    """Construct V3_Hv1, then load + freeze the vonly visual encoder.

    V3_Hv2 = v1 + T 流 aux 压力重建（t_rec，λ_trec=0.1；命题 2 载体依赖
    修复——T 编码器监督从「仅接触」恢复为「接触 + 重建」）。
    V3_Hv3 = v2 + 一阶足部残差头（foot_res，λ_foot_res；命题 2 的兑现
    通道——接触软门控、输出饱和、只写足关节 6D）。v1/v2 无对应头，state
    dict 逐键兼容。
    V3_Hv5 = v3 结构 − 穿透通道（无锚版：T 提供不了绝对高度，地面通道
    整体删除，推理层只跑支撑通道；Pen/Global/ATE 为纯观测值）。
    V3_Hv6 = v5 + 残差头落点改踝 7/8（错误一修复：足关节是叶节点，改自身
    旋转不移动任何关键点——位置效应恒 0；踝旋转经 127 mm 杠杆 ≈ 平移足
    关键点；V3_H设计错误裁定.md §1）。state dict 与 v5 逐键一致（落点是
    应用层 + 损失目标槽位，非参数变化）。
    """
    name = str(config.get("model_name", ""))
    aux = name.startswith("V3_Hv2") or name.startswith("V3_Hv3") or name.startswith("V3_Hv5") or name.startswith("V3_Hv6")
    fres = name.startswith("V3_Hv3") or name.startswith("V3_Hv5") or name.startswith("V3_Hv6")
    noanch = name.startswith("V3_Hv5") or name.startswith("V3_Hv6")
    landing = "ankle" if name.startswith("V3_Hv6") else "foot"
    model = AnySoleModelV3H(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(config.get("dropout", 0.1)),
        pose_layers=int(config.get("pose_layers", 6)),
        t_rec=aux,
        foot_res=fres,
        no_anchor=noanch,
        res_landing=landing,
    )
    # The vonly specialists are scheduler-derived names, not registry
    # entries — the path is built directly (no registry validation).
    vonly_ckpt = None
    for vonly_name in _vonly_candidates(str(config.get("model_name", ""))):
        cand = anysole_model_dir(vonly_name, "joint_and") / "checkpoints" / "ckpt_best.pt"
        if cand.is_file():
            vonly_ckpt, vonly_name = cand, vonly_name
            break
    if vonly_ckpt is None:
        raise FileNotFoundError(
            "V3_Hv1 needs a frozen vonly visual encoder but none of the "
            "candidates exist (tried full-suffix / camera-only / cam3 vonly); "
            "train the V-only specialist first.")
    ckpt = vonly_ckpt
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    src = state.get("model", {}) if isinstance(state, dict) else {}
    dst = model.state_dict()
    loaded = []
    for key, value in src.items():
        if key.startswith("encoders.v_enc.") and key in dst and dst[key].shape == value.shape:
            dst[key] = value.to(dst[key].dtype)
            loaded.append(key)
    if not loaded:
        raise RuntimeError("no encoders.v_enc.* keys copied from %s" % ckpt)
    model.load_state_dict(dst)
    for p in model.encoders.v_enc.parameters():
        p.requires_grad_(False)
    digest = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    CONFIG_EXTRA["vonly_enc"] = {
        "source": str(ckpt), "sha256": digest, "n_keys": len(loaded),
    }
    print("v3_3h: frozen vonly V encoder from %s (%d keys)" % (ckpt, len(loaded)))
    return model
