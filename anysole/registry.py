"""Model registry + centralized address inference for the whole AnySole pipeline.

Pattern adapted from SurvPGC's dataset_deployment/registry.py
(``DATASET_CONFIGS`` + ``infer_standard_paths``): every real path derives
from a model identifier plus semantic knobs — the CLI never writes raw
paths.  train / eval / infer / ridge / tune all resolve their addresses
through :func:`infer_anysole_paths`.

**Independence (2026-09-24 restructure)**: each registered model maps to an
entry script in ``anysole/models/`` (``f0b.py`` … ``v4b.py``) that fixes the
model's structure; ``get_builder`` imports that entry.  Every training run —
base dir or variant dir — starts from random init: there is no warm-start
lineage, no parent/child relation, and ``infer_anysole_paths`` never infers
an ``--init-from`` (it stays None; ``--init-from`` in train.py remains a
debug-only override).  Model + hyperparameters is the full identity of a run.

Variant rule (fixed, needs no registration): a variant dir (``tw40``,
``t0003_tw40_lr3e4``, ...) is just a hyperparameter subdir under the base
model dir — same entry, same from-scratch rule, different address.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Optional

from anysole.types import anysole_model_dir


@dataclass(frozen=True)
class ModelEntry:
    name: str
    module: str  # entry script under anysole/models/ (e.g. "v3_4b")
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "F0b": ModelEntry("F0b", "f0b", "V2 regression base (linear t-encoder)"),
    "F4a": ModelEntry("F4a", "f4a", "foot_conv tactile encoder"),
    "F2": ModelEntry("F2", "f2", "f2 representation"),
    "F2p4": ModelEntry("F2p4", "f2p4", "f2 + foot_conv"),
    "V3_2": ModelEntry("V3_2", "v3_2", "9-part pose head"),
    "V3_3A": ModelEntry("V3_3A", "v3_3a", "9-part + soft A"),
    "V3_Bv1": ModelEntry("V3_Bv1", "v3_3b", "9-part + soft A + sigma gate"),
    "V3_Bv2": ModelEntry("V3_Bv2", "v3_3b_query", "Bv1 + query pressure readout (Q-DIRECT / Q-NN)"),
    "V3_Bv3": ModelEntry("V3_Bv3", "v3_3b_query", "Bv2 + Q-FACT factored readout"),
    "V3_Cv1": ModelEntry("V3_Cv1", "v3_3c", "frozen-V (V3_Bv1_vonly base) + zero-init T-residual"),
    "V3_Dv1": ModelEntry("V3_Dv1", "v3_3d", "frozen-V + joint position head (position-space T channel)"),
    "V3_Ev1": ModelEntry("V3_Ev1", "v3_3e", "constraint fusion: pure-V pose + T contact head + T pose prior"),
    "V3_Fv1": ModelEntry("V3_Fv1", "v3_3f", "diffusion prior (V cond) + T contact/ground guidance + leg-chain refinement"),
    "V3_Gv1": ModelEntry("V3_Gv1", "v3_3g", "frozen E carrier + T contact stream + part-wise observation layer"),
    "V3_Hv1": ModelEntry("V3_Hv1", "v3_3h", "single-generator regression + T physical state + residual channels"),
    "V3_Hv2": ModelEntry("V3_Hv2", "v3_3h",
                           "V3_Hv1 + T-stream aux pressure reconstruction (命题 2 载体依赖修复)"),
    "V3_Hv3": ModelEntry("V3_Hv3", "v3_3h",
                           "V3_Hv2 + first-order foot residual head (命题 2 兑现通道)"),
    "V3_Hv5": ModelEntry("V3_Hv5", "v3_3h",
                           "V3_Hv3 − 穿透通道（无锚版：不消费任何 floor 输入）"),
    "V3_Hv6": ModelEntry("V3_Hv6", "v3_3h",
                           "V3_Hv5 + 残差头落点改踝 7/8（错误一修复：足关节是叶节点、"
                           "改自身旋转不移动任何关键点；踝旋转经 127 mm 杠杆 ≈ 平移足）"),
    "V5_Av1": ModelEntry("V5_Av1", "v5_1",
                       "V5_Av1: frozen E carrier + V/T dual z experts (PoE) + "
                       "FiLM lower-limb injection (2026-10-19)"),
    "V4_Av1": ModelEntry("V4_Av1", "v4_1",
                       "V4_Av1: frozen E carrier + part-token gated position "
                       "correction (2026-10-19)"),
    "V5_Av1_cam3_offfix_tphys": ModelEntry(
        "V5_Av1_cam3_offfix_tphys", "v5_1",
        "V5_Av1 + 维度 2 显式物理特征（T_phys 12→20，2026-10-21）"),
    "V5_Av1_cam3_offfix_d8": ModelEntry(
        "V5_Av1_cam3_offfix_d8", "v5_1",
        "V5_Av1 + 维度 3 足-地监督三头（tcop/着地模式/GT CoP，2026-10-21）"),
    "V5_Av1_occmix": ModelEntry("V5_Av1_occmix", "v5_1",
                              "V5_Av1 + V6_Av1 occ_mix 对照（训练侧 V 失效态混合，2026-10-20）"),
    "V4_Av1_occmix": ModelEntry("V4_Av1_occmix", "v4_1",
                              "V4_Av1 + V6_Av1 occ_mix 对照（训练侧 V 失效态混合，2026-10-20）"),
    "V3_4a": ModelEntry("V3_4a", "v3_4a", "9-part on f2"),
    "V3_4b": ModelEntry("V3_4b", "v3_4b", "9-part soft A on f2"),
    "V3_4c": ModelEntry("V3_4c", "v3_4c", "9-part soft A + sigma gate on f2"),
    "V4A": ModelEntry("V4A", "v4a", "24-slot gradient clustering"),
    "V4B": ModelEntry("V4B", "v4b", "learned hard partition (--part-json)"),
}

# 四相机扩展（D2，2026-10-12 裁定：相机进模型名）。每相机一份 V3_Bv1，
# 共用 v3_3b 结构入口；数据根差异由 v1_cam{N}.yaml 携带（types.camera_roots）。
MODEL_REGISTRY.update({
    f"V3_Bv1_cam{cam}": ModelEntry(f"V3_Bv1_cam{cam}", "v3_3b",
                                  f"V3_Bv1 on camera {cam} (four-camera experiment)")
    for cam in (1, 2, 3, 4)
})

# V-degradation 遮挡组（2026-10-13）：cam3 + 人物 bbox 底部遮挡，共用 v3_3b
# 结构入口；数据根差异由 v1_cam3_occ{N}.yaml 携带（occ010 = 底部 10%，f=0.10）。
MODEL_REGISTRY.update({
    f"V3_Bv1_cam3_occ{occ}": ModelEntry(f"V3_Bv1_cam3_occ{occ}", "v3_3b",
                                       f"V3_Bv1 on cam3 with bottom {int(occ) / 10:g}% of "
                                       f"the person bbox occluded (V-degradation experiment)")
    for occ in ("010", "020", "030", "050")
})

# 新数据验证（2026-10-22）：facts 已固化为 v3+offset，默认口径训练即新数据。
MODEL_REGISTRY["V3_Bv1_cam3_v3off"] = ModelEntry(
    "V3_Bv1_cam3_v3off", "v3_3b", "V3_Bv1 cam3 on production v3+offset facts (replacement validation)")

# E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型。
# 统一 = dropout [0.5,0.25,0.25]（3B 基名默认 / 其余 _drop 尾缀）；
# vonly/tonly = 单模态专家；基名（E/V4_Av1/V5_Av1）或 _nodrop（3B）= [1,0,0] VT 专家。
for _suf, _mod, _note in (("_vonly", "v3_3b", "E1: 3B V-only expert"),
                          ("_tonly", "v3_3b", "E1: 3B T-only expert"),
                          ("_nodrop", "v3_3b", "E1: 3B always-dual expert ([1,0,0])"),
                          ("", "v3_3e", "E1: E always-dual expert ([1,0,0]; new carrier for V4_Av1/V5_Av1)"),
                          ("_drop", "v3_3e", "E1: E unified (dropout)"),
                          ("_vonly", "v3_3e", "E1: E V-only expert"),
                          ("_tonly", "v3_3e", "E1: E T-only expert"),
                          ("", "v4_1", "E1: V4_Av1 always-dual expert ([1,0,0])"),
                          ("_drop", "v4_1", "E1: V4_Av1 unified (dropout)"),
                          ("_vonly", "v4_1", "E1: V4_Av1 V-only expert"),
                          ("_tonly", "v4_1", "E1: V4_Av1 T-only expert"),
                          ("", "v5_1", "E1: V5_Av1 always-dual expert ([1,0,0])"),
                          ("_drop", "v5_1", "E1: V5_Av1 unified (dropout)"),
                          ("_vonly", "v5_1", "E1: V5_Av1 V-only expert"),
                          ("_tonly", "v5_1", "E1: V5_Av1 T-only expert")):
    _pre = {"v3_3b": "V3_Bv1_cam3_v3off", "v3_3e": "V3_Ev1_cam3_v3off",
            "v4_1": "V4_Av1_cam3_v3off", "v5_1": "V5_Av1_cam3_v3off"}[_mod]
    MODEL_REGISTRY[_pre + _suf] = ModelEntry(_pre + _suf, _mod, _note)

# E1 同架构专家对比（2026-10-22，V5_Av4 修复迭代）：V5_Av1 结构 + V 流
# 从零训练（carrier_scratch，去 E 继承）+ 先验塔 E 配方。四变体 × v3off。
for _suf, _note in (("", "E1: V5_Av4 always-dual expert ([1,0,0])"),
                    ("_drop", "E1: V5_Av4 unified (dropout)"),
                    ("_vonly", "E1: V5_Av4 V-only expert"),
                    ("_tonly", "E1: V5_Av4 T-only expert")):
    MODEL_REGISTRY["V5_Av4_cam3_v3off" + _suf] = ModelEntry(
        "V5_Av4_cam3_v3off" + _suf, "v5_1", _note)

# 5-fold 动作覆盖实验（2026-10-22）：受试者轮换 leave-subjects-out。
# fold 命名 = 统一编号 plain 名 + fold 尾缀（结构与基座逐项一致：
# V3_Ev1→v3_3e / V3_Bv1→v3_3b / V5_Av4 与 V5_Av1→v5_1 / V4_Av1→v4_1）；
# fold 尾缀与 split 文件见 AnysoleWorkspace/protocol/splits/default/。
for _fold in ("fS13S14", "fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8"):
    for _pre, _mod in (("V3_Ev1", "v3_3e"), ("V3_Bv1", "v3_3b"),
                       ("V5_Av4", "v5_1"),
                       ("V5_Av1", "v5_1"), ("V4_Av1", "v4_1")):
        MODEL_REGISTRY[_pre + "_" + _fold] = ModelEntry(
            _pre + "_" + _fold, _mod,
            "5-fold per-action coverage, fold %s (2026-10-22)" % _fold)
# 5-fold occ050 档（2026-10-23）：四评估模型 × 5 fold。
for _fold in ("fS13S14", "fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8"):
    for _pre, _mod in (("V5_Av1", "v5_1"), ("V4_Av1", "v4_1"),
                       ("V3_Bv1", "v3_3b"), ("V5_Av4", "v5_1")):
        MODEL_REGISTRY[_pre + "_occ050_" + _fold] = ModelEntry(
            _pre + "_occ050_" + _fold, _mod,
            "5-fold per-action coverage occ050, fold %s (2026-10-23)" % _fold)
# 5-fold occ010 档（2026-10-23）：四评估模型 × 5 fold。
for _fold in ("fS13S14", "fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8"):
    for _pre, _mod in (("V5_Av1", "v5_1"), ("V4_Av1", "v4_1"),
                       ("V3_Bv1", "v3_3b"), ("V5_Av4", "v5_1")):
        MODEL_REGISTRY[_pre + "_occ010_" + _fold] = ModelEntry(
            _pre + "_occ010_" + _fold, _mod,
            "5-fold per-action coverage occ010, fold %s (2026-10-23)" % _fold)

# V3_Ev1 削弱首组（2026-10-17）：cam3 + occ010 两配置自训自测。
MODEL_REGISTRY.update({
    f"V3_Ev1_cam3_occ{occ}": ModelEntry(f"V3_Ev1_cam3_occ{occ}", "v3_3e",
                                       f"V3_Ev1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_Fv1 削弱首组（2026-10-17）：同一 v3_3f 结构入口；冻结 vonly V 编码器由
# 入口按名称尾缀推导（V3_Bv1_vonly / V3_Bv1_cam3_occ010_vonly），数据根差异
# 由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_Fv1_cam3_occ{occ}": ModelEntry(f"V3_Fv1_cam3_occ{occ}", "v3_3f",
                                       f"V3_Fv1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_Gv1 削弱首组（2026-10-17）：同一 v3_3g 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_Gv1_cam3_occ{occ}": ModelEntry(f"V3_Gv1_cam3_occ{occ}", "v3_3g",
                                       f"V3_Gv1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_Hv1 削弱首组（2026-10-17）：同一 v3_3h 结构入口；冻结 vonly V 编码器由
# 入口按名称尾缀推导（V3_Bv1_vonly / V3_Bv1_cam3_occ010_vonly），数据根差异
# 由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_Hv1_cam3_occ{occ}": ModelEntry(f"V3_Hv1_cam3_occ{occ}", "v3_3h",
                                       f"V3_Hv1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_Hv2 削弱首组（2026-10-19）：同入口，名称尾缀推导同 V3_Hv1。
MODEL_REGISTRY.update({
    f"V3_Hv2_cam3_occ{occ}": ModelEntry(f"V3_Hv2_cam3_occ{occ}", "v3_3h",
                                          f"V3_Hv2 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_Hv3 削弱首组（2026-10-19）：同入口。
MODEL_REGISTRY.update({
    f"V3_Hv3_cam3_occ{occ}": ModelEntry(f"V3_Hv3_cam3_occ{occ}", "v3_3h",
                                          f"V3_Hv3 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_Hv5 削弱首组（2026-10-19）：同入口。
MODEL_REGISTRY.update({
    f"V3_Hv5_cam3_occ{occ}": ModelEntry(f"V3_Hv5_cam3_occ{occ}", "v3_3h",
                                          f"V3_Hv5 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_Hv6 削弱首组（2026-10-20）：同入口（错误一修复的 occ010 对照）。
MODEL_REGISTRY.update({
    f"V3_Hv6_cam3_occ{occ}": ModelEntry(f"V3_Hv6_cam3_occ{occ}", "v3_3h",
                                          f"V3_Hv6 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_Hv5 难度轴扩展（2026-10-19）：cam1/cam2/cam4 + occ020 无锚配置
# （T 价值区 = V 失效条件的无锚对照实验）。
MODEL_REGISTRY.update({
    f"V3_Hv5_cam{cam}": ModelEntry(f"V3_Hv5_cam{cam}", "v3_3h",
                                     f"V3_Hv5 (no-anchor) on camera {cam}")
    for cam in (1, 2, 4)
})
MODEL_REGISTRY["V3_Hv5_cam3_occ020"] = ModelEntry(
    "V3_Hv5_cam3_occ020", "v3_3h",
    "V3_Hv5 (no-anchor) on cam3 with bottom 2% occluded")

# V3_Hv5 全网格扩展（2026-10-20）：cam{1,2,4} × occ{010,020}。
MODEL_REGISTRY.update({
    f"V3_Hv5_cam{cam}_occ{occ}": ModelEntry(f"V3_Hv5_cam{cam}_occ{occ}", "v3_3h",
                                              f"V3_Hv5 (no-anchor) on cam{cam} with bottom "
                                              f"{int(occ) / 10:g}% occluded")
    for cam in (1, 2, 4) for occ in ("010", "020")
})

# V5_Av1 削弱首组（2026-10-19）：同一 v5_1 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V5_Av1_cam3_occ{occ}": ModelEntry(f"V5_Av1_cam3_occ{occ}", "v5_1",
                                      f"V5_Av1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                      f"(V-degradation suite)")
    for occ in ("010",)
})

# V4_Av1/V5_Av1 V 失效梯度组（2026-10-20）：occ020/030/050，载体 = cam3 E v2.1
#（无 E 各档 ckpt；V5_Av1 §3 occ050 裁定口径），数据根由 runner 按尾缀推导。
MODEL_REGISTRY.update({
    f"{_grad_base}_cam3_occ{_occ}": ModelEntry(
        f"{_grad_base}_cam3_occ{_occ}", _grad_mod,
        f"{_grad_base} on cam3 occ{_occ}（载体 = cam3 E v2.1，V 失效梯度档）")
    for _grad_base, _grad_mod in (("V5_Av1", "v5_1"), ("V4_Av1", "v4_1"))
    for _occ in ("020", "030", "050")
})

# V4_Av1 削弱首组（2026-10-19）：同一 v4_1 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V4_Av1_cam3_occ{occ}": ModelEntry(f"V4_Av1_cam3_occ{occ}", "v4_1",
                                      f"V4_Av1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                      f"(V-degradation suite)")
    for occ in ("010",)
})


# V3_Dv1 削弱套件（2026-10-15）：同一 v3_3d 结构入口，每配置一份冻结基底
# （该配置 vonly 专才 + 该配置主模型受过训 foot_conv，build_v3_3c_base.py
# 逐配置合成）；数据根差异由 v1_cam{N}/v1_cam3_occ{N}.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_Dv1_cam{cam}": ModelEntry(f"V3_Dv1_cam{cam}", "v3_3d",
                                  f"V3_Dv1 on camera {cam} (V-degradation suite)")
    for cam in (1, 2, 4)
})
MODEL_REGISTRY.update({
    f"V3_Dv1_cam3_occ{occ}": ModelEntry(f"V3_Dv1_cam3_occ{occ}", "v3_3d",
                                       f"V3_Dv1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010", "020", "030", "050")
})


def list_registered_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_model_entry(model_name: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown model %r. Expected one of: %s" % (model_name, sorted(MODEL_REGISTRY))
        ) from exc


def get_builder(model_name: str):
    """Import and return the entry module (STRUCTURE / CONFIG_EXTRA / build)
    that defines ``model_name``'s architecture."""
    entry = get_model_entry(model_name)
    return import_module("anysole.models.%s" % entry.module)


def infer_anysole_paths(
    model_name: str,
    *,
    variant: Optional[str] = None,
    contact: str = "joint_and",
    which: str = "last",
) -> dict:
    """Central address inference (the only place real paths are built).

    ``model_name``: registry identifier (e.g. V3_4a).  ``variant``: stacked
    hyperparameter fields under the model dir (tw40 / t0003_tw40_lr3e4),
    built automatically by train from its effective config and passed
    explicitly by eval/infer/ridge.  ``contact``: contact-label scheme in the
    dir name.  ``which``: "last" | "best" — which checkpoint to address.
    Default "last": ckpt_last is written at the end of every run, while
    ckpt_best is the val-metric-best INTERMEDIATE and can be several epochs
    older — use last for the current model, switch to best via --which best
    when producing a final report.
    """
    if which not in ("last", "best"):
        raise ValueError("which must be 'last' or 'best', got %r" % which)
    get_model_entry(model_name)
    model_dir = anysole_model_dir(model_name, contact, variant)
    ckpt_name = "ckpt_last.pt" if which == "last" else "ckpt_best.pt"
    return {
        "model_dir": model_dir,
        "ckpt": model_dir / "checkpoints" / ckpt_name,
        "init_from": None,  # independence: never inferred (train --init-from = debug only)
        "predictions_dir": model_dir / "predictions" / "eval_motion",
        "metrics_dir": model_dir / "metrics",
    }

# 时间窗扫描（2026-10-22，z_note/multimodal_fusion/时间窗扫描实验.md）：
# 五家族 x tw{5,10,15,20} x cam3/cam3_occ010（v3off 生产轴）。tw 是训练期
# 协变量（eval 从 ckpt 读 tw）；stride 随 tw 非重叠。E 格 = V4_Av1/V5_Av1
# 的同 (tw, 视觉) 载体。波 1 = E + V3_Bv1 + V5_Av4；波 2 = V4_Av1 + V5_Av1。
for _mod, _cam3, _occ010 in (
        ("v3_3b", "V3_Bv1_cam3_v3off", "V3_Bv1_cam3_occ010_v3off"),
        ("v3_3e", "V3_Ev1_cam3_v3off", "V3_Ev1_cam3_occ010_v3off"),
        ("v4_1", "V4_Av1_cam3_v3off", "V4_Av1_cam3_occ010_v3off"),
        ("v5_1", "V5_Av1_cam3_v3off", "V5_Av1_cam3_occ010_v3off"),
        ("v5_1", "V5_Av4_cam3_v3off", "V5_Av4_cam3_occ010_v3off")):
    MODEL_REGISTRY.update({
        f"{_cam3}_tw{tw}": ModelEntry(f"{_cam3}_tw{tw}", _mod,
                                      "tw scan cam3 cell (2026-10-22)")
        for tw in (5, 10, 15)
    })
    MODEL_REGISTRY.update({
        _occ010: ModelEntry(_occ010, _mod, "tw scan occ010 cell (2026-10-22)"),
        **{
            f"{_occ010}_tw{tw}": ModelEntry(f"{_occ010}_tw{tw}", _mod,
                                            "tw scan occ010 cell (2026-10-22)")
            for tw in (5, 10, 15)
        },
    })

# wo05 实验轴（2026-10-24，AnysoleWorkspace/protocol/splits/default/
# splits_wo05.csv）：canonical 去动作 05（绕圈跑）的 12 模型（3 家族 ×
# E1 四变体）；结构与 v3off 条目逐项一致（v3_3e / v4_1 / v5_1 入口）。
for _suf, _mod, _note in (
        ("", "v3_3e", "wo05: E always-dual expert ([1,0,0]; V4_Av1 wo05 carrier)"),
        ("_drop", "v3_3e", "wo05: E unified (dropout)"),
        ("_vonly", "v3_3e", "wo05: E V-only expert"),
        ("_tonly", "v3_3e", "wo05: E T-only expert"),
        ("", "v4_1", "wo05: V4_Av1 always-dual expert ([1,0,0])"),
        ("_drop", "v4_1", "wo05: V4_Av1 unified (dropout)"),
        ("_vonly", "v4_1", "wo05: V4_Av1 V-only expert"),
        ("_tonly", "v4_1", "wo05: V4_Av1 T-only expert"),
        ("", "v5_1", "wo05: V5_Av4 always-dual expert ([1,0,0])"),
        ("_drop", "v5_1", "wo05: V5_Av4 unified (dropout)"),
        ("_vonly", "v5_1", "wo05: V5_Av4 V-only expert"),
        ("_tonly", "v5_1", "wo05: V5_Av4 T-only expert")):
    _pre = {"v3_3e": "V3_Ev1_wo05", "v4_1": "V4_Av1_wo05",
            "v5_1": "V5_Av4_wo05"}[_mod]
    MODEL_REGISTRY[_pre + _suf] = ModelEntry(_pre + _suf, _mod, _note)

# λ 扫描（2026-10-24，用户裁定「先扫描一下这几个参数点」）：V5_Av1 / V5_Av4 /
# V4_Av1 × λ_prior_pose {1,2,3,4} × λ_prior_traj {0,1} 共 24 模型；cam3 +
# v1_wo05.yaml 纯扫描。结构与 v3off 条目逐项一致（v4_1 / v5_1 入口）。
for _m, _mod, _fam in (("V5_Av1", "v5_1", "V5_Av1_wo05"),
                       ("V5_Av4", "v5_1", "V5_Av4_wo05"),
                       ("V4_Av1", "v4_1", "V4_Av1_wo05")):
    for _lp in (1, 2, 3, 4):
        for _lt in (0, 1):
            _n = "%s_lp%d_lt%d" % (_fam, _lp, _lt)
            MODEL_REGISTRY[_n] = ModelEntry(
                _n, _mod, "prior-lambda scan %s lp%s lt%s (2026-10-24)"
                % (_m, _lp, _lt))
