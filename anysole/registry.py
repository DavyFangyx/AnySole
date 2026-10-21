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
    "V3_3B": ModelEntry("V3_3B", "v3_3b", "9-part + soft A + sigma gate"),
    "V3_3B_denoised": ModelEntry("V3_3B_denoised", "v3_3b_denoised",
                                 "V3_3B on handoff denoised pressure (2026-10-17)"),
    "V3_3C": ModelEntry("V3_3C", "v3_3c", "frozen-V (V3_3B_vonly base) + zero-init T-residual"),
    "V3_3D": ModelEntry("V3_3D", "v3_3d", "frozen-V + joint position head (position-space T channel)"),
    "V3_3E": ModelEntry("V3_3E", "v3_3e", "constraint fusion: pure-V pose + T contact head + T pose prior"),
    "V3_3E_denoised": ModelEntry("V3_3E_denoised", "v3_3e_denoised",
                                 "V3_3E on handoff denoised pressure (2026-10-17)"),
    "V3_3F": ModelEntry("V3_3F", "v3_3f", "diffusion prior (V cond) + T contact/ground guidance + leg-chain refinement"),
    "V3_3F_denoised": ModelEntry("V3_3F_denoised", "v3_3f_denoised",
                                 "V3_3F on handoff denoised pressure (2026-10-17)"),
    "V3_3G": ModelEntry("V3_3G", "v3_3g", "frozen E carrier + T contact stream + part-wise observation layer"),
    "V3_3H": ModelEntry("V3_3H", "v3_3h", "single-generator regression + T physical state + residual channels"),
    "V3_3H_v2": ModelEntry("V3_3H_v2", "v3_3h",
                           "V3_3H + T-stream aux pressure reconstruction (命题 2 载体依赖修复)"),
    "V3_3H_v3": ModelEntry("V3_3H_v3", "v3_3h",
                           "V3_3H_v2 + first-order foot residual head (命题 2 兑现通道)"),
    "V3_3H_v5": ModelEntry("V3_3H_v5", "v3_3h",
                           "V3_3H_v3 − 穿透通道（无锚版：不消费任何 floor 输入）"),
    "V3_3H_v6": ModelEntry("V3_3H_v6", "v3_3h",
                           "V3_3H_v5 + 残差头落点改踝 7/8（错误一修复：足关节是叶节点、"
                           "改自身旋转不移动任何关键点；踝旋转经 127 mm 杠杆 ≈ 平移足）"),
    "V5_1": ModelEntry("V5_1", "v5_1",
                       "V5_1A: frozen E carrier + V/T dual z experts (PoE) + "
                       "FiLM lower-limb injection (2026-10-19)"),
    "V4_1": ModelEntry("V4_1", "v4_1",
                       "V4_1A: frozen E carrier + part-token gated position "
                       "correction (2026-10-19)"),
    "V5_1_cam3_offfix_tphys": ModelEntry(
        "V5_1_cam3_offfix_tphys", "v5_1",
        "V5_1A + 维度 2 显式物理特征（T_phys 12→20，2026-10-21）"),
    "V5_1_cam3_offfix_d8": ModelEntry(
        "V5_1_cam3_offfix_d8", "v5_1",
        "V5_1A + 维度 3 足-地监督三头（tcop/着地模式/GT CoP，2026-10-21）"),
    "V5_1_occmix": ModelEntry("V5_1_occmix", "v5_1",
                              "V5_1A + V6_1A occ_mix 对照（训练侧 V 失效态混合，2026-10-20）"),
    "V4_1_occmix": ModelEntry("V4_1_occmix", "v4_1",
                              "V4_1A + V6_1A occ_mix 对照（训练侧 V 失效态混合，2026-10-20）"),
    "V3_4a": ModelEntry("V3_4a", "v3_4a", "9-part on f2"),
    "V3_4b": ModelEntry("V3_4b", "v3_4b", "9-part soft A on f2"),
    "V3_4c": ModelEntry("V3_4c", "v3_4c", "9-part soft A + sigma gate on f2"),
    "V4A": ModelEntry("V4A", "v4a", "24-slot gradient clustering"),
    "V4B": ModelEntry("V4B", "v4b", "learned hard partition (--part-json)"),
}

# 四相机扩展（D2，2026-10-12 裁定：相机进模型名）。每相机一份 V3_3B，
# 共用 v3_3b 结构入口；数据根差异由 v1_cam{N}.yaml 携带（types.camera_roots）。
MODEL_REGISTRY.update({
    f"V3_3B_cam{cam}": ModelEntry(f"V3_3B_cam{cam}", "v3_3b",
                                  f"V3_3B on camera {cam} (four-camera experiment)")
    for cam in (1, 2, 3, 4)
})

# V-degradation 遮挡组（2026-10-13）：cam3 + 人物 bbox 底部遮挡，共用 v3_3b
# 结构入口；数据根差异由 v1_cam3_occ{N}.yaml 携带（occ010 = 底部 10%，f=0.10）。
MODEL_REGISTRY.update({
    f"V3_3B_cam3_occ{occ}": ModelEntry(f"V3_3B_cam3_occ{occ}", "v3_3b",
                                       f"V3_3B on cam3 with bottom {int(occ) / 10:g}% of "
                                       f"the person bbox occluded (V-degradation experiment)")
    for occ in ("010", "020", "030", "050")
})

# offset 重同步组（2026-10-20）：V3_3B 同结构同规格，唯一差异 =
# corrected_offset=True（B 类 session GT 采样按 z_note/data_offset_fixpipe
# 修正表平移，A/C/D 不动）。数据根推导同 V3_3B_cam3 / _cam3_occ{N}。
MODEL_REGISTRY.update({
    "V3_3B_cam3_offfix": ModelEntry("V3_3B_cam3_offfix", "v3_3b",
                                    "V3_3B on cam3 with corrected mocap offsets "
                                    "(offset-resync experiment)"),
    **{
        f"V3_3B_cam3_occ{occ}_offfix": ModelEntry(
            f"V3_3B_cam3_occ{occ}_offfix", "v3_3b",
            f"V3_3B on cam3 occ{occ} with corrected mocap offsets "
            f"(offset-resync experiment)")
        for occ in ("010", "020", "030", "050")
    },
})
# v2/v3 + offset 组（2026-10-20）：结构 = v3_3b_v2 / v3_3b_v3（tactile_input
# 分别固定 denoised108_v2 / denoised108_v3），其余同 offset 组。
for _entry_name, _struct in (("v2", "v3_3b_v2"), ("v3", "v3_3b_v3")):
    MODEL_REGISTRY.update({
        f"V3_3B_cam3_offfix_{_entry_name}": ModelEntry(
            f"V3_3B_cam3_offfix_{_entry_name}", _struct,
            f"V3_3B on cam3, corrected offsets, tactile {_entry_name}"),
        **{
            f"V3_3B_cam3_occ{occ}_offfix_{_entry_name}": ModelEntry(
                f"V3_3B_cam3_occ{occ}_offfix_{_entry_name}", _struct,
                f"V3_3B on cam3 occ{occ}, corrected offsets, tactile {_entry_name}")
            for occ in ("010", "020", "030", "050")
        },
    })



# offset 设计组（2026-10-21）：V3_3E（E v2.1 载体）/ V5_1A / V4_1A / V3_3H
# 的 corrected_offset 版。E_offfix 为 V5_1/V4_1 的新载体（v1 载体的
# offset 重训版）；V3_3H_offfix 的 vonly 载体按名称尾缀推导
# （V3_3B_cam3_offfix_vonly / V3_3B_cam3_occ010_offfix_vonly，专才由
# offsetfix_vonly 实验组训练）。
for _base, _struct, _note in (("V3_3E_cam3_offfix", "v3_3e", "E v2.1 offset 重训（V5_1/V4_1 新载体）"),
                              ("V3_3E_cam3_occ010_offfix", "v3_3e", "E v2.1 occ010 offset 重训"),
                              ("V5_1_cam3_offfix", "v5_1", "V5_1A offset"),
                              ("V5_1_cam3_occ010_offfix", "v5_1", "V5_1A occ010 offset"),
                              ("V5_1_cam3_occ020_offfix", "v5_1", "V5_1A occ020 offset"),
                              ("V5_1_cam3_occ030_offfix", "v5_1", "V5_1A occ030 offset"),
                              ("V5_1_cam3_occ050_offfix", "v5_1", "V5_1A occ050 offset"),
                              ("V4_1_cam3_offfix", "v4_1", "V4_1A offset"),
                              ("V4_1_cam3_occ010_offfix", "v4_1", "V4_1A occ010 offset"),
                              ("V3_3H_cam3_offfix", "v3_3h", "V3_3H offset"),
                              ("V3_3H_cam3_occ010_offfix", "v3_3h", "V3_3H occ010 offset")):
    MODEL_REGISTRY[_base] = ModelEntry(_base, _struct, _note + " (2026-10-21)")

# v2 + offset 设计组（2026-10-21）：V5_1A / V4_1A × cam3/cam3_occ010 的
# denoised108_v2 版。与 offset 设计组逐项相同（corrected_offset=True、
# 载体 = E_offfix 系），唯一差异 = 结构固定 tactile_input=denoised108_v2
#（v5_1_v2 / v4_1_v2）。
for _base, _struct, _note in (("V5_1_cam3_offfix_v2", "v5_1_v2", "V5_1A offset + v2 压力包"),
                              ("V5_1_cam3_occ010_offfix_v2", "v5_1_v2", "V5_1A occ010 offset + v2 压力包"),
                              ("V4_1_cam3_offfix_v2", "v4_1_v2", "V4_1A offset + v2 压力包"),
                              ("V4_1_cam3_occ010_offfix_v2", "v4_1_v2", "V4_1A occ010 offset + v2 压力包")):
    MODEL_REGISTRY[_base] = ModelEntry(_base, _struct, _note + " (2026-10-21)")

# 新数据验证（2026-10-22）：facts 已固化为 v3+offset，默认口径训练即新数据。
MODEL_REGISTRY["V3_3B_cam3_v3off"] = ModelEntry(
    "V3_3B_cam3_v3off", "v3_3b", "V3_3B cam3 on production v3+offset facts (replacement validation)")

# E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型。
# 统一 = dropout [0.5,0.25,0.25]（3B 基名默认 / 其余 _drop 尾缀）；
# vonly/tonly = 单模态专家；基名（E/V4_1/V5_1）或 _nodrop（3B）= [1,0,0] VT 专家。
for _suf, _mod, _note in (("_vonly", "v3_3b", "E1: 3B V-only expert"),
                          ("_tonly", "v3_3b", "E1: 3B T-only expert"),
                          ("_nodrop", "v3_3b", "E1: 3B always-dual expert ([1,0,0])"),
                          ("", "v3_3e", "E1: E always-dual expert ([1,0,0]; new carrier for V4_1/V5_1)"),
                          ("_drop", "v3_3e", "E1: E unified (dropout)"),
                          ("_vonly", "v3_3e", "E1: E V-only expert"),
                          ("_tonly", "v3_3e", "E1: E T-only expert"),
                          ("", "v4_1", "E1: V4_1A always-dual expert ([1,0,0])"),
                          ("_drop", "v4_1", "E1: V4_1A unified (dropout)"),
                          ("_vonly", "v4_1", "E1: V4_1A V-only expert"),
                          ("_tonly", "v4_1", "E1: V4_1A T-only expert"),
                          ("", "v5_1", "E1: V5_1A always-dual expert ([1,0,0])"),
                          ("_drop", "v5_1", "E1: V5_1A unified (dropout)"),
                          ("_vonly", "v5_1", "E1: V5_1A V-only expert"),
                          ("_tonly", "v5_1", "E1: V5_1A T-only expert")):
    _pre = {"v3_3b": "V3_3B_cam3_v3off", "v3_3e": "V3_3E_cam3_v3off",
            "v4_1": "V4_1_cam3_v3off", "v5_1": "V5_1_cam3_v3off"}[_mod]
    MODEL_REGISTRY[_pre + _suf] = ModelEntry(_pre + _suf, _mod, _note)

# V3_3E 削弱首组（2026-10-17）：cam3 + occ010 两配置自训自测。
MODEL_REGISTRY.update({
    f"V3_3E_cam3_occ{occ}": ModelEntry(f"V3_3E_cam3_occ{occ}", "v3_3e",
                                       f"V3_3E on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_3F 削弱首组（2026-10-17）：同一 v3_3f 结构入口；冻结 vonly V 编码器由
# 入口按名称尾缀推导（V3_3B_vonly / V3_3B_cam3_occ010_vonly），数据根差异
# 由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_3F_cam3_occ{occ}": ModelEntry(f"V3_3F_cam3_occ{occ}", "v3_3f",
                                       f"V3_3F on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_3G 削弱首组（2026-10-17）：同一 v3_3g 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_3G_cam3_occ{occ}": ModelEntry(f"V3_3G_cam3_occ{occ}", "v3_3g",
                                       f"V3_3G on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_3H 削弱首组（2026-10-17）：同一 v3_3h 结构入口；冻结 vonly V 编码器由
# 入口按名称尾缀推导（V3_3B_vonly / V3_3B_cam3_occ010_vonly），数据根差异
# 由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_3H_cam3_occ{occ}": ModelEntry(f"V3_3H_cam3_occ{occ}", "v3_3h",
                                       f"V3_3H on cam3 with bottom {int(occ) / 10:g}% occluded "
                                       f"(V-degradation suite)")
    for occ in ("010",)
})

# V3_3H_v2 削弱首组（2026-10-19）：同入口，名称尾缀推导同 V3_3H。
MODEL_REGISTRY.update({
    f"V3_3H_v2_cam3_occ{occ}": ModelEntry(f"V3_3H_v2_cam3_occ{occ}", "v3_3h",
                                          f"V3_3H_v2 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_3H_v3 削弱首组（2026-10-19）：同入口。
MODEL_REGISTRY.update({
    f"V3_3H_v3_cam3_occ{occ}": ModelEntry(f"V3_3H_v3_cam3_occ{occ}", "v3_3h",
                                          f"V3_3H_v3 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_3H_v5 削弱首组（2026-10-19）：同入口。
MODEL_REGISTRY.update({
    f"V3_3H_v5_cam3_occ{occ}": ModelEntry(f"V3_3H_v5_cam3_occ{occ}", "v3_3h",
                                          f"V3_3H_v5 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_3H_v6 削弱首组（2026-10-20）：同入口（错误一修复的 occ010 对照）。
MODEL_REGISTRY.update({
    f"V3_3H_v6_cam3_occ{occ}": ModelEntry(f"V3_3H_v6_cam3_occ{occ}", "v3_3h",
                                          f"V3_3H_v6 on cam3 with bottom {int(occ) / 10:g}% occluded")
    for occ in ("010",)
})

# V3_3H_v5 难度轴扩展（2026-10-19）：cam1/cam2/cam4 + occ020 无锚配置
# （T 价值区 = V 失效条件的无锚对照实验）。
MODEL_REGISTRY.update({
    f"V3_3H_v5_cam{cam}": ModelEntry(f"V3_3H_v5_cam{cam}", "v3_3h",
                                     f"V3_3H_v5 (no-anchor) on camera {cam}")
    for cam in (1, 2, 4)
})
MODEL_REGISTRY["V3_3H_v5_cam3_occ020"] = ModelEntry(
    "V3_3H_v5_cam3_occ020", "v3_3h",
    "V3_3H_v5 (no-anchor) on cam3 with bottom 2% occluded")

# V3_3H_v5 全网格扩展（2026-10-20）：cam{1,2,4} × occ{010,020}。
MODEL_REGISTRY.update({
    f"V3_3H_v5_cam{cam}_occ{occ}": ModelEntry(f"V3_3H_v5_cam{cam}_occ{occ}", "v3_3h",
                                              f"V3_3H_v5 (no-anchor) on cam{cam} with bottom "
                                              f"{int(occ) / 10:g}% occluded")
    for cam in (1, 2, 4) for occ in ("010", "020")
})

# V5_1 削弱首组（2026-10-19）：同一 v5_1 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V5_1_cam3_occ{occ}": ModelEntry(f"V5_1_cam3_occ{occ}", "v5_1",
                                      f"V5_1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                      f"(V-degradation suite)")
    for occ in ("010",)
})

# V4_1/V5_1 V 失效梯度组（2026-10-20）：occ020/030/050，载体 = cam3 E v2.1
#（无 E 各档 ckpt；V5_1A §3 occ050 裁定口径），数据根由 runner 按尾缀推导。
MODEL_REGISTRY.update({
    f"{_grad_base}_cam3_occ{_occ}": ModelEntry(
        f"{_grad_base}_cam3_occ{_occ}", _grad_mod,
        f"{_grad_base} on cam3 occ{_occ}（载体 = cam3 E v2.1，V 失效梯度档）")
    for _grad_base, _grad_mod in (("V5_1", "v5_1"), ("V4_1", "v4_1"))
    for _occ in ("020", "030", "050")
})

# V4_1 削弱首组（2026-10-19）：同一 v4_1 结构入口；carrier_from 按名称
# 尾缀指向该配置的 E v2.1 ckpt_best（registry 的 train_args 携带），数据根
# 差异由 v1_cam3_occ010.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V4_1_cam3_occ{occ}": ModelEntry(f"V4_1_cam3_occ{occ}", "v4_1",
                                      f"V4_1 on cam3 with bottom {int(occ) / 10:g}% occluded "
                                      f"(V-degradation suite)")
    for occ in ("010",)
})


# V3_3D 削弱套件（2026-10-15）：同一 v3_3d 结构入口，每配置一份冻结基底
# （该配置 vonly 专才 + 该配置主模型受过训 foot_conv，build_v3_3c_base.py
# 逐配置合成）；数据根差异由 v1_cam{N}/v1_cam3_occ{N}.yaml 按名尾缀推导。
MODEL_REGISTRY.update({
    f"V3_3D_cam{cam}": ModelEntry(f"V3_3D_cam{cam}", "v3_3d",
                                  f"V3_3D on camera {cam} (V-degradation suite)")
    for cam in (1, 2, 4)
})
MODEL_REGISTRY.update({
    f"V3_3D_cam3_occ{occ}": ModelEntry(f"V3_3D_cam3_occ{occ}", "v3_3d",
                                       f"V3_3D on cam3 with bottom {int(occ) / 10:g}% occluded "
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
