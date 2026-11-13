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
    "V3_Bv1_IMU": ModelEntry(
        "V3_Bv1_IMU", "v3_3b_imu",
        "V3_Bv1 + S2M-style IMU: 12ch MLP embedding concatenated with the "
        "FootConv tactile token (zero-init concat projection, 2026-11-12)"),
    "V3_Bv1_PhysIMU": ModelEntry(
        "V3_Bv1_PhysIMU", "v3_3b_physimu",
        "V3_Bv1 + fusion 2: T_raw FootConv + T_phys MLP + IMU MLP, "
        "two-stage [T_phys,IMU] -> [T_raw,.] 2d->d fusion (2026-11-12)"),
    "V3_Bv1_IMU_NG": ModelEntry(
        "V3_Bv1_IMU_NG", "v3_3b_imu_nogate",
        "V3_Bv1_IMU with pose-head gate='none' (no-gate control)"),
    "V3_Bv1_PhysIMU_NG": ModelEntry(
        "V3_Bv1_PhysIMU_NG", "v3_3b_physimu_nogate",
        "V3_Bv1_PhysIMU with pose-head gate='none' (no-gate control)"),
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
    # 条件合作线（2026-10-25）：V5_Av5 / V4_Av2 = scratch 载体（E 式双流
    # 同训）+ occ_mix + 条件契约损失（λ_cond——V 健康窗口罚修正量）。
    # _nocond = 同配方去契约的对照（隔离契约贡献）。
    "V5_Av5": ModelEntry("V5_Av5", "v5_1",
                       "V5_Av5: V5 结构 + scratch V + λ3/1 + occ_mix + 条件契约（2026-10-25）"),
    "V5_Av5_nocond": ModelEntry("V5_Av5_nocond", "v5_1",
                              "V5_Av5 − 条件契约（对照，2026-10-25）"),
    "V4_Av2": ModelEntry("V4_Av2", "v4_1",
                       "V4_Av2: V4 结构 + scratch V + occ_mix + 条件契约（2026-10-25）"),
    "V4_Av3": ModelEntry("V4_Av3", "v4b",
                       "V4_Av3: V4 修正分支挂冻结 B 骨架（两阶段，T 解毒，2026-10-25）"),
    "V4_Av3_cam3_occ010": ModelEntry("V4_Av3_cam3_occ010", "v4b",
                                   "V4_Av3 occ010 轴（2026-10-25；yaml 按名尾缀推导）"),
    "V4_Av4": ModelEntry("V4_Av4", "v4b",
                       "V4_Av4: V4_Av3 + 骨架双阶段精调（carrier_lr_mult=0.1，不冻结，2026-10-25）"),
    "V4_Av4_cam3_occ010": ModelEntry("V4_Av4_cam3_occ010", "v4b",
                                   "V4_Av4 occ010 轴（2026-10-25；yaml 按名尾缀推导）"),
    "V4_Av4_drop": ModelEntry("V4_Av4_drop", "v4b",
                            "V4_Av4 drop 对照（0.5/0.25/0.25；drop 毒性 vs 可训载体，2026-10-25）"),
    "V4_Av4_drop_cam3_occ010": ModelEntry("V4_Av4_drop_cam3_occ010", "v4b",
                                        "V4_Av4 drop 对照 occ010 轴（2026-10-25）"),
    # V7 候选 A/B/C（2026-10-25，模型设计思路.md §3）：独立证据 + 共享生成。
    "V7_Av1": ModelEntry("V7_Av1", "v7_a", "候选 A：独立证据 + 共享生成（2026-10-25）"),
    "V7_Bv1": ModelEntry("V7_Bv1", "v7_b", "候选 B：A + 接触参与根运动（2026-10-25）"),
    "V7_Cv1": ModelEntry("V7_Cv1", "v7_c", "候选 C：A + 窗口级可靠性路由（2026-10-25）"),
    "V7_Av1_cam3_occ010": ModelEntry("V7_Av1_cam3_occ010", "v7_a", "候选 A occ010 轴"),
    "V7_Bv1_cam3_occ010": ModelEntry("V7_Bv1_cam3_occ010", "v7_b", "候选 B occ010 轴"),
    "V7_Cv1_cam3_occ010": ModelEntry("V7_Cv1_cam3_occ010", "v7_c", "候选 C occ010 轴"),
    "V7_Av1_precheck": ModelEntry("V7_Av1_precheck", "v7_a", "候选 A 端到端预检（epochs=6）"),
    "V7_Bv1_precheck": ModelEntry("V7_Bv1_precheck", "v7_b", "候选 B 端到端预检（epochs=6）"),
    "V7_Cv1_precheck": ModelEntry("V7_Cv1_precheck", "v7_c", "候选 C 端到端预检（epochs=6）"),
    "V7_Cv2": ModelEntry("V7_Cv2", "v7_c2", "V7_Cv1 + 根/姿态双独立 T 门（2026-10-26）"),
    "V7_Cv3": ModelEntry("V7_Cv3", "v7_c3", "V7_Cv1 + T-only 私有残差适配器（2026-10-26）"),
    "V7_Cv4": ModelEntry("V7_Cv4", "v7_c", "V7_Cv1 + 枚举三条件训练（2026-10-26）"),
    "V7_Cv5": ModelEntry("V7_Cv5", "v7_c2", "V7_Cv5: 双独立 T 门 + 枚举三条件训练（2026-10-26）"),
    "V7_Cv5_bs512": ModelEntry("V7_Cv5_bs512", "v7_c2", "V7_Cv5 训练提速实验：batch 256→512（2026-11-01）"),
    "V7_Cv5_precheck": ModelEntry("V7_Cv5_precheck", "v7_c2", "V7_Cv5 六轮预检"),
    # 修改 11 A3（2026-11-05）：Cv5 + T_phys v2（12→20 维，hull 着地几何 +
    # 双足关系）。Cv5 既有 12 维 base 不动；4 折 LSO 测净兑现量。
    "V7_Cv5_tphys": ModelEntry("V7_Cv5_tphys", "v7_c2tphys",
                               "V7_Cv5 + T_phys v2（修改 11 A3，2026-11-05）"),
    "V7_Cv5_tphys_precheck": ModelEntry("V7_Cv5_tphys_precheck", "v7_c2tphys",
                                        "V7_Cv5_tphys 六轮预检"),
    "V7_Cv5_tphys_fS5S6S11": ModelEntry("V7_Cv5_tphys_fS5S6S11", "v7_c2tphys",
                                        "V7_Cv5_tphys 折 fS5S6S11（LSO）"),
    "V7_Cv5_tphys_fS5S11S12": ModelEntry("V7_Cv5_tphys_fS5S11S12", "v7_c2tphys",
                                         "V7_Cv5_tphys 折 fS5S11S12（LSO）"),
    "V7_Cv5_tphys_fS6S7S13": ModelEntry("V7_Cv5_tphys_fS6S7S13", "v7_c2tphys",
                                        "V7_Cv5_tphys 折 fS6S7S13（LSO）"),
    "V7_Cv5_tphys_fS5S6S8": ModelEntry("V7_Cv5_tphys_fS5S6S8", "v7_c2tphys",
                                       "V7_Cv5_tphys 折 fS5S6S8（LSO）"),
    # 修改 11 §9.3（2026-11-05）：hull_mcol 逐会话 p5–p95 重定心；T_phys 仍 20 维。
    "V7_Cv5_tphysRC": ModelEntry("V7_Cv5_tphysRC", "v7_c2tphysrc",
                                 "V7_Cv5 + T_phys v2 重定心 hull_mcol（修改 11 §9.3）"),
    "V7_Cv5_tphysRC_precheck": ModelEntry("V7_Cv5_tphysRC_precheck", "v7_c2tphysrc",
                                          "V7_Cv5_tphysRC 六轮预检"),
    "V7_Cv5_tphysRC_fS5S6S11": ModelEntry("V7_Cv5_tphysRC_fS5S6S11", "v7_c2tphysrc",
                                          "V7_Cv5_tphysRC 折 fS5S6S11（LSO）"),
    "V7_Cv5_tphysRC_fS5S11S12": ModelEntry("V7_Cv5_tphysRC_fS5S11S12", "v7_c2tphysrc",
                                           "V7_Cv5_tphysRC 折 fS5S11S12（LSO）"),
    "V7_Cv5_tphysRC_fS6S7S13": ModelEntry("V7_Cv5_tphysRC_fS6S7S13", "v7_c2tphysrc",
                                          "V7_Cv5_tphysRC 折 fS6S7S13（LSO）"),
    "V7_Cv5_tphysRC_fS5S6S8": ModelEntry("V7_Cv5_tphysRC_fS5S6S8", "v7_c2tphysrc",
                                         "V7_Cv5_tphysRC 折 fS5S6S8（LSO）"),
    "V7_Cv2_precheck": ModelEntry("V7_Cv2_precheck", "v7_c2", "V7_Cv2 六轮预检"),
    "V7_Cv3_precheck": ModelEntry("V7_Cv3_precheck", "v7_c3", "V7_Cv3 六轮预检"),
    "V7_Cv4_precheck": ModelEntry("V7_Cv4_precheck", "v7_c", "V7_Cv4 六轮预检"),
    "V7_Cv6": ModelEntry("V7_Cv6", "v7_c2", "V7_Cv5 双门 + 无模态缺失训练（始终 VT；2026-10-26）"),
    "V7_Cv6_precheck": ModelEntry("V7_Cv6_precheck", "v7_c2", "V7_Cv6 六轮预检"),
    "V7_Cv7": ModelEntry("V7_Cv7", "v7_a", "无门控 A + 枚举三条件训练（2026-10-26）"),
    # V7 C 门控修改方案（2026-10-30，z_note/multimodal_fusion/V7_方案/
    # V7_C门控修改方案.md §3）：T/LR/TLR 三实验组，独立模型与结果目录，
    # 不占用新的 V7_Cv 编号、不覆盖 Cv5/Cv7 历史产物。Q/S 按方案 §8.3
    # 于 TLR 完成后另立。
    "V7_Cv5T": ModelEntry("V7_Cv5T", "v7_c2t", "V7_Cv5T: Cv5 双门逐帧化（T 时间门，2026-10-30）"),
    "V7_Cv5LR": ModelEntry("V7_Cv5LR", "v7_c2lr", "V7_Cv5LR: Cv5 双门 + 左右部位门（LR，2026-10-30）"),
    "V7_Cv5TLR": ModelEntry("V7_Cv5TLR", "v7_c2tlr", "V7_Cv5TLR: 逐帧 + 左右部位组合门（TLR，2026-10-30）"),
    "V7_Cv5Ctx": ModelEntry("V7_Cv5Ctx", "v7_c2ctx", "V7_Cv5Ctx: Cv5 + 上一窗姿态上下文续写（2026-11-01）"),
    "V7_Cv5Ctx2": ModelEntry("V7_Cv5Ctx2", "v7_c2ctx2", "V7_Cv5Ctx2: Cv5 + 续写 v2（ctx 直连根朝向：部位 query 注入 + 输出残差，2026-11-01）"),
    "V7_Cv5Ctx3": ModelEntry("V7_Cv5Ctx3", "v7_c2ctx3", "V7_Cv5Ctx3: Cv5 + 续写 v3（V/VT 回归 + T-only 硬填式重叠，2026-11-01）"),
    "V7_Cv5Ctx4": ModelEntry("V7_Cv5Ctx4", "v7_c2ctx4", "V7_Cv5Ctx4: Cv5 + 续写 v4（锚点系残差分解，2026-11-01）"),
    "V7_Cv5Ctx5": ModelEntry("V7_Cv5Ctx5", "v7_c2ctx5", "V7_Cv5Ctx5: Cv5 + 续写 v5（块自回归：全姿态硬填，2026-11-01）"),
    "V7_Cv5AR": ModelEntry("V7_Cv5AR", "v7_c2ar", "V7_Cv5AR: Cv5 + 帧级双模式 AR（V/VT 回归 + T 自回归，2026-11-01）"),
    "V7_Cv5ARM": ModelEntry("V7_Cv5ARM", "v7_c2arm", "V7_Cv5ARM: Cv5 + 因果掩码对照格（AR 无反馈支路，2026-11-01）"),
    "V7_Cv5AR2": ModelEntry("V7_Cv5AR2", "v7_c2ar2", "V7_Cv5AR2: Cv5 + AR 相对运动 + 航向状态分离（2026-11-01）"),
    "V7_Cv5AR2MP": ModelEntry("V7_Cv5AR2MP", "v7_c2ar2mp", "V7_Cv5AR2MP: AR2M + 前缀继承样本（enum T 档里历史 V 真实、当前窗 V 为 null，2026-11-03）"),
    "V7_Cv5AR2M": ModelEntry("V7_Cv5AR2M", "v7_c2ar2m", "V7_Cv5AR2M: AR2 对照格（因果掩码无反馈，2026-11-02）"),
    "V7_Cv5AR2MA": ModelEntry("V7_Cv5AR2MA", "v7_c2ar2ma", "V7_Cv5AR2MA: AR2M + 读出按条件分支（V/VT 绝对 = Cv5 目标空间，T 相对 + 外部 ψ，2026-11-04 修改 09）"),
    "V7_Cv5AR2MF": ModelEntry("V7_Cv5AR2MF", "v7_c2ar2m", "V7_Cv5AR2MF: AR2M + yaw-fix（T 行无航向监督 + 自由生成航向链 + 世界轨迹损失，2026-11-02）"),
    "V7_Cv5AR2MF_gb": ModelEntry("V7_Cv5AR2MF_gb", "v7_c2ar2m", "V7_Cv5AR2MF_gb: AR2MF + 鞋垫增益偏置增强（G-b，2026-11-02）"),
    "V7_Cv5_gb": ModelEntry("V7_Cv5_gb", "v7_c2", "V7_Cv5_gb: Cv5 双门回归 + 鞋垫增益偏置增强（G-b，2026-11-02）"),
    "V7_Cv5AR2M_gb": ModelEntry("V7_Cv5AR2M_gb", "v7_c2ar2m", "V7_Cv5AR2M_gb: AR2M（因果掩码无反馈）+ G-b 鞋垫增强（2026-11-02）"),
    "V7_Cv5AR2R": ModelEntry("V7_Cv5AR2R", "v7_c2ar2r", "V7_Cv5AR2R: Cv5 + AR2 相对读出纯回归（双向无反馈无因果，2026-11-02）"),
    "V7_Cv5AR2RT": ModelEntry("V7_Cv5AR2RT", "v7_c2ar2rt", "V7_Cv5AR2RT: AR2R + VT 零初始化触觉残差（2026-11-02）"),
    "V7_Cv5SC": ModelEntry("V7_Cv5SC", "v7_c2sc", "V7_Cv5SC: Cv5 + 按模态分窗（V 只看当前窗，T 吃 obs_context 长历史，非 AR2，2026-11-03）"),
    # 修改 07（2026-11-04，z_note/修改/07_根头速率方向解耦与T行方向掩码.md）：
    # 根头三通道（水平速率/水平方向/垂直分量）+ 按通道损失 + T 行方向掩码。
    "V7_Cv5SD": ModelEntry("V7_Cv5SD", "v7_c2sd", "V7_Cv5SD: Cv5 + 根头速率/方向解耦 + T 行方向掩码（2026-11-04）"),
    "V7_Cv5SD_precheck": ModelEntry("V7_Cv5SD_precheck", "v7_c2sd", "V7_Cv5SD 六轮预检"),
    # 第二臂 B′（2026-11-04，修改 07）：SD 结构不变，损失改为恒等式配平
    # （全 1.0 且不掩码时 ≡ Cv5 的 l_vel；T 行方向掩码为唯一干预）。
    "V7_Cv5SD2": ModelEntry("V7_Cv5SD2", "v7_c2sd2", "V7_Cv5SD2: SD 结构 + 恒等式配平损失（B′，2026-11-04）"),
    "V7_Cv5SD2_precheck": ModelEntry("V7_Cv5SD2_precheck", "v7_c2sd2", "V7_Cv5SD2 六轮预检"),
    # 修改 10 §三 可直接移植 ①（2026-11-05）：SD2 + VT 零初始化触觉残差。
    "V7_Cv5SD2RT": ModelEntry("V7_Cv5SD2RT", "v7_c2sd2rt", "V7_Cv5SD2RT: SD2 + VT 零初始化触觉残差（修改 10，2026-11-05）"),
    "V7_Cv5SD2RT_precheck": ModelEntry("V7_Cv5SD2RT_precheck", "v7_c2sd2rt", "V7_Cv5SD2RT 六轮预检"),
    # V8 系列（2026-11-05）：新基座 = V7_Cv5SD2；消融在 V8 上重建。
    # 第 4 折直接用 splits_test_S6S7_val_S14.csv（train/val 与原折逐位相同，
    # test = S6+S7，S13 退出留出池）；其余实验沿同一折表。
    "V8_Cv5": ModelEntry("V8_Cv5", "v8_cv5", "V8 基座：双门+枚举+SD2+锚头（2026-11-05）"),
    "V8_Cv4": ModelEntry("V8_Cv4", "v8_cv4", "V8 消融：单门+枚举+SD2+锚头"),
    "V8_Cv2": ModelEntry("V8_Cv2", "v8_cv5", "V8 消融：双门+随机 dropout+SD2+锚头"),
    "V8_Cv6": ModelEntry("V8_Cv6", "v8_cv5", "V8 消融：双门+始终 VT+SD2+锚头"),
    "V8_Cv7": ModelEntry("V8_Cv7", "v8_cv7", "V8 消融：无门+枚举+SD2+锚头"),
    # V8 + IMU（2026-11-12）：融合方式 1（S2M 式 IMU MLP + concat）与
    # 融合方式 2（T_raw/T_phys/IMU 三支两级融合），各带 Cv5 门控 / Cv7 无门。
    "V8_Cv5_IMU": ModelEntry(
        "V8_Cv5_IMU", "v8_cv5_imu",
        "V8_Cv5 + S2M-style IMU (fusion 1, raw108_imu)"),
    "V8_Cv5_PhysIMU": ModelEntry(
        "V8_Cv5_PhysIMU", "v8_cv5_physimu",
        "V8_Cv5 + PhysIMU (fusion 2, raw96_phys_imu)"),
    "V8_Cv7_IMU": ModelEntry(
        "V8_Cv7_IMU", "v8_cv7_imu",
        "V8_Cv7 (no gate) + S2M-style IMU (fusion 1)"),
    "V8_Cv7_PhysIMU": ModelEntry(
        "V8_Cv7_PhysIMU", "v8_cv7_physimu",
        "V8_Cv7 (no gate) + PhysIMU (fusion 2)"),
    # V8_Cv8U（2026-11-12）：NLL 校准的 per-slot precision router。
    "V8_Cv8U": ModelEntry(
        "V8_Cv8U", "v8_cv8u",
        "V8_Cv8 + per-slot precision router (s_V/s_T NLL-calibrated, K stop-grad)"),
    "V8_Cv8U_SG": ModelEntry(
        "V8_Cv8U_SG", "v8_cv8u_sg",
        "V8_Cv8U with stop-grad off (VT task loss can close K)"),
    "V8_Cv8U_PhysIMU_SG": ModelEntry(
        "V8_Cv8U_PhysIMU_SG", "v8_cv8u_physimu_sg",
        "V8_Cv8U-SG on PhysIMU (raw96_phys_imu) tactile encoding"),
    "V8_Cv8U_PhysIMU": ModelEntry(
        "V8_Cv8U_PhysIMU", "v8_cv8u_physimu",
        "V8_Cv8U on PhysIMU (raw96_phys_imu), stop-grad on (NLL-only K)"),
    "V8_Cv7_PhysIMU_Gate": ModelEntry(
        "V8_Cv7_PhysIMU_Gate", "v8_cv7_physimu_gate",
        "frozen V8_Cv7_PhysIMU + four-group independent V/T gate (2026-11-12)"),
    "V8_Cv7_PhysIMU_TGate": ModelEntry(
        "V8_Cv7_PhysIMU_TGate", "v8_cv7_physimu_tgate",
        "frozen V8_Cv7_PhysIMU + frozen V gate + independent T gate "
        "(window base + frame correction, 2026-11-13)"),
    # T 运动轴 T1（2026-11-06）：方向载体 = 同帧根 6D 前向轴 + 标量 φ。
    "V8_Cv5T1": ModelEntry("V8_Cv5T1", "v8_cv5t1", "V8 基座 + T1 方向载体（根6D前向轴+φ，2026-11-06）"),
    "V8_Cv5T1_precheck": ModelEntry("V8_Cv5T1_precheck", "v8_cv5t1", "V8_Cv5T1 六轮预检"),
    # 门控条件化波 1（修改 13，2026-11-06）：
    #   RM = 结构同 V8_Cv5，只把可靠性轴放进训练分布（occ_mix 档梯
    #        {010,020,030}，p_occ=0.45，p_nullV=p_nullT=0）—— 单旗在 train_args。
    #   PO = 结构单旗（腿/足 query 门钉 1.0），train_args 同 V8_Cv5。
    "V8_Cv5RM": ModelEntry("V8_Cv5RM", "v8_cv5", "V8 基座 + occ 档梯训练分布（修改 13 D1，2026-11-06）"),
    "V8_Cv5PO": ModelEntry("V8_Cv5PO", "v8_cv5po", "V8 基座 + 腿/足门钉 1.0（修改 13 D3，2026-11-06）"),
    # 双向信任门波 1（修改 14 §5，2026-11-06）：
    #   VG   = V 侧镜像门 + T 门 2σ（结构由入口 v8_cv5vg 携带）；
    #   ST   = 结构同 V8_Cv5，单旗在 train_args（p_stale，同会话窗级错位）；
    #   VGST = 两旗同开（主模型）。
    "V8_Cv5VG": ModelEntry("V8_Cv5VG", "v8_cv5vg", "V8 基座 + 双向信任门（修改 14 §5，2026-11-06）"),
    "V8_Cv5ST": ModelEntry("V8_Cv5ST", "v8_cv5", "V8 基座 + 同会话窗级错位增强（修改 14 §5.3，2026-11-06）"),
    "V8_Cv5VGST": ModelEntry("V8_Cv5VGST", "v8_cv5vg", "V8 基座 + 双向门 + 错位增强（修改 14 主模型，2026-11-06）"),
    # 修改 15（2026-11-07）：Cv5 + V 门（T 门保持 Cv5），两阶段只训 V 门。
    "V8_Cv5VIG": ModelEntry("V8_Cv5VIG", "v8_cv5vig", "V8 基座 + V 门（T 门 Cv5，b* 监督，修改 15）"),
    # 修改 15（2026-11-07）：Cv5 + 不确定性专家驱动的 V 门（IG 监督，运行时零 GT）。
    "V8_Cv5VUG": ModelEntry("V8_Cv5VUG", "v8_cv5vug", "V8 基座 + 不确定性专家 V 门（IG 监督，修改 15）"),
    # 竞争路由门（2026-11-07 会话裁定）：k = σ(s_v − s_t) 逐帧逐组，
    # s_v 读 V token、s_t 读 T token + 分组物理/IMU 特征；V 支路乘
    # (1−k)、T 支路乘 k，V-only 行 k≡0、T-only 行 k≡1。
    "V8_Cv8": ModelEntry("V8_Cv8", "v8_cv8", "V8 基座 + 竞争路由门（k=σ(s_v−s_t)，2026-11-07）"),
    "V8_Cv8_precheck": ModelEntry("V8_Cv8_precheck", "v8_cv8", "V8_Cv8 六轮预检"),
    # 修改 08（2026-11-04，z_note/修改/08_平移锚头.md）：V/VT 行平移锚头
    # （旁路读出，不替换现役 GT 锚）；新增并列自锚列。
    "V7_Cv5TA": ModelEntry("V7_Cv5TA", "v7_c2ta", "V7_Cv5TA: Cv5 + V/VT 平移锚头（旁路读出，2026-11-04）"),
    "V7_Cv5TA_precheck": ModelEntry("V7_Cv5TA_precheck", "v7_c2ta", "V7_Cv5TA 六轮预检"),
    "V7_Cv5AR2MSC": ModelEntry("V7_Cv5AR2MSC", "v7_c2ar2msc", "V7_Cv5AR2MSC: AR2M + 按模态分窗（V 只看当前窗，T 吃 obs_context 长历史，2026-11-03）"),
    # 修改 04（2026-11-03，z_note/修改/04_AR训练目标与自由生成对齐.md）：
    # R = AR2 结构 + 多步自由 rollout 损失（改动 A）；RD = R + 低维状态
    # 反馈（改动 B）。结构 = AR2（ar2_fb=True——开工裁定：AR2M 无反馈
    # 支路 TF≡rollout，改动 A 无作用对象）；两改动配置键默认关。
    "V7_Cv5AR2RL": ModelEntry("V7_Cv5AR2RL", "v7_c2ar2rl", "V7_Cv5AR2RL: AR2 + 多步自由 rollout 损失（修改 04 R 格，2026-11-03）"),
    "V7_Cv5AR2RLD": ModelEntry("V7_Cv5AR2RLD", "v7_c2ar2rl", "V7_Cv5AR2RLD: AR2 + rollout 损失 + 低维状态反馈（修改 04 RD 格，2026-11-03）"),
    "V7_Cv5ARSS": ModelEntry("V7_Cv5ARSS", "v7_c2arss", "V7_Cv5ARSS: AR + 反馈 scheduled sampling + 收缩噪声（2026-11-01）"),
    # 2026-11-02：ARSS 正名结果目录被 huyuxuan 旧目录占用（写权限受阻）——
    # 临时注册名开新目录；旧目录清出后本行可删、结果目录可 rename 回正名。
    "V7_Cv5ARSS2": ModelEntry("V7_Cv5ARSS2", "v7_c2arss", "V7_Cv5ARSS2: ARSS 同名临时注册（结果目录绕过，2026-11-02）"),
    "V7_Cv5Ctx4B": ModelEntry("V7_Cv5Ctx4B", "v7_c2ctx4b", "V7_Cv5Ctx4B: Cv5 + 续写 v4-b（统一残差空间 + V/VT 锚估计头，2026-11-01）"),
    "V7_Cv5Ctx4R": ModelEntry("V7_Cv5Ctx4R", "v7_c2ctx4r", "V7_Cv5Ctx4R: Cv5 + 续写 v4-a + 接触根运动（B 候选挂链，2026-11-01）"),
    "V7_Cv5CtxSS": ModelEntry("V7_Cv5CtxSS", "v7_c2ctx", "V7_Cv5CtxSS: Ctx + scheduled sampling（G1，2026-11-01）"),
    "V7_Cv5CtxVC": ModelEntry("V7_Cv5CtxVC", "v7_c2ctx_vc", "V7_Cv5CtxVC: Ctx + V 在场条件门控（G2，2026-11-01）"),
    "V7_Cv5CtxHP": ModelEntry("V7_Cv5CtxHP", "v7_c2ctx_hp", "V7_Cv5CtxHP: Ctx + 评估硬前缀（G3，2026-11-01）"),
    "V7_Cv5CtxFull": ModelEntry("V7_Cv5CtxFull", "v7_c2ctx_full", "V7_Cv5CtxFull: G1+G2+G3 三合一（G4，2026-11-01）"),
    "V7_Cv5Q": ModelEntry("V7_Cv5Q", "v7_c2q", "V7_Cv5Q: TLR + 第一层解码上下文条件门（Q，2026-10-31）"),
    "V7_Cv5S": ModelEntry("V7_Cv5S", "v7_c2tlr", "V7_Cv5S: TLR + 显式收益监督（S，λ_s=1.0，2026-11-01）"),
    "V7_Cv5S01": ModelEntry("V7_Cv5S01", "v7_c2tlr", "V7_Cv5S01: TLR + 显式收益监督（S，λ_s=0.1，2026-11-01）"),
    "V7_Cv7_precheck": ModelEntry("V7_Cv7_precheck", "v7_a", "V7_Cv7 六轮预检"),
    # 阶段 3 触觉泛化（2026-10-31，E6 门控重设计）：以 Cv7（v7_a 无门
    # 入口）为基座，训练侧数据增强单点实验。结构不变，只有训练配方
    # （G-b 鞋垫增益偏置 / G-c T token dropout）不同，故共用 v7_a 入口，
    # 由新模型名分家到独立结果目录。
    "V7_Cv7_gb": ModelEntry("V7_Cv7_gb", "v7_a", "V7_Cv7 + 鞋垫增益偏置增强 (G-b)"),
    "V7_Cv7_gc": ModelEntry("V7_Cv7_gc", "v7_a", "V7_Cv7 + T token dropout (G-c)"),
    # 阶段 2 受试者交叉泛化（2026-10-31，E6 门控重设计配套）：V7_Cv7
    # leave-subjects-out 4 折，结构不变（v7_a 无门入口），只有 split 配置
    # 不同（v1_fold_*.yaml → splits_test_*.csv），新模型名分家到独立结果目录。
    "V7_Cv7_fS5S6S11": ModelEntry("V7_Cv7_fS5S6S11", "v7_a", "V7_Cv7 折 fS5S6S11（leave-subjects-out）"),
    "V7_Cv7_fS5S11S12": ModelEntry("V7_Cv7_fS5S11S12", "v7_a", "V7_Cv7 折 fS5S11S12（leave-subjects-out）"),
    "V7_Cv7_fS6S7S13": ModelEntry("V7_Cv7_fS6S7S13", "v7_a", "V7_Cv7 折 fS6S7S13（leave-subjects-out）"),
    "V7_Cv7_fS5S6S8": ModelEntry("V7_Cv7_fS5S6S8", "v7_a", "V7_Cv7 折 fS5S6S8（leave-subjects-out）"),
    # 协议 A 主表 4 折（2026-11-05）：V7_Cv5 双门 + 枚举，跨受试者配对读数；
    # 对照 = V7_Cv7 同 fold（无门，已跑）。仅 split 配置不同。
    "V7_Cv5_fS5S6S11": ModelEntry("V7_Cv5_fS5S6S11", "v7_c2", "V7_Cv5 折 fS5S6S11（leave-subjects-out）"),
    "V7_Cv5_fS5S11S12": ModelEntry("V7_Cv5_fS5S11S12", "v7_c2", "V7_Cv5 折 fS5S11S12（leave-subjects-out）"),
    "V7_Cv5_fS6S7S13": ModelEntry("V7_Cv5_fS6S7S13", "v7_c2", "V7_Cv5 折 fS6S7S13（leave-subjects-out）"),
    "V7_Cv5_fS5S6S8": ModelEntry("V7_Cv5_fS5S6S8", "v7_c2", "V7_Cv5 折 fS5S6S8（leave-subjects-out）"),
    "V7_SV1": ModelEntry("V7_SV1", "v7_a", "V7 E1 专家 S_V（仅视觉训练，2026-10-26）"),
    "V7_ST1": ModelEntry("V7_ST1", "v7_a", "V7 E1 专家 S_T（仅触觉训练，2026-10-26）"),
    "V7_Cv5_SV1": ModelEntry("V7_Cv5_SV1", "v7_c2", "V7_Cv5 E1 专家 S_V（双门同架构，2026-10-26）"),
    "V7_Cv5_ST1": ModelEntry("V7_Cv5_ST1", "v7_c2", "V7_Cv5 E1 专家 S_T（双门同架构，2026-10-26）"),
    "V4_Av2_nocond": ModelEntry("V4_Av2_nocond", "v4_1",
                              "V4_Av2 − 条件契约（对照，2026-10-25）"),
    "V5_Av5_cam3_occ010": ModelEntry("V5_Av5_cam3_occ010", "v5_1",
                                   "条件合作线 occ010 轴（2026-10-25；yaml 按名尾缀推导）"),
    "V5_Av5_nocond_cam3_occ010": ModelEntry("V5_Av5_nocond_cam3_occ010", "v5_1",
                                          "条件合作线 occ010 轴去契约对照（2026-10-25）"),
    "V4_Av2_cam3_occ010": ModelEntry("V4_Av2_cam3_occ010", "v4_1",
                                   "条件合作线 occ010 轴（2026-10-25；yaml 按名尾缀推导）"),
    "V4_Av2_nocond_cam3_occ010": ModelEntry("V4_Av2_nocond_cam3_occ010", "v4_1",
                                          "条件合作线 occ010 轴去契约对照（2026-10-25）"),
    "V3_4a": ModelEntry("V3_4a", "v3_4a", "9-part on f2"),
    "V3_4b": ModelEntry("V3_4b", "v3_4b", "9-part soft A on f2"),
    "V3_4c": ModelEntry("V3_4c", "v3_4c", "9-part soft A + sigma gate on f2"),
    "V4A": ModelEntry("V4A", "v4a", "24-slot gradient clustering"),
    "V4B": ModelEntry("V4B", "v4b", "learned hard partition (--part-json)"),
}

# V3_Bv1 IMU / PhysIMU 4 折（2026-11-12，woS13 四折）：每族 4 折，
# fS6S7 = S13-out（与 V8 当前四折同表）。
for _fam, _mod in (("V3_Bv1_IMU", "v3_3b_imu"),
                   ("V3_Bv1_PhysIMU", "v3_3b_physimu"),
                   ("V3_Bv1_IMU_NG", "v3_3b_imu_nogate"),
                   ("V3_Bv1_PhysIMU_NG", "v3_3b_physimu_nogate")):
    for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"):
        _name = "%s_%s" % (_fam, _fold)
        MODEL_REGISTRY[_name] = ModelEntry(
            _name, _mod, "%s 折 %s（woS13 四折，2026-11-12）" % (_fam, _fold))

# V8 + IMU 4 折（2026-11-12，woS13 四折，主模型 V8_Cv5 / 无门 V8_Cv7）。
for _fam, _mod in (("V8_Cv5_IMU", "v8_cv5_imu"),
                   ("V8_Cv5_PhysIMU", "v8_cv5_physimu"),
                   ("V8_Cv7_IMU", "v8_cv7_imu"),
                   ("V8_Cv7_PhysIMU", "v8_cv7_physimu"),
                   ("V8_Cv8U", "v8_cv8u"),
                   ("V8_Cv8U_SG", "v8_cv8u_sg"),
                   ("V8_Cv8U_PhysIMU_SG", "v8_cv8u_physimu_sg"),
                   ("V8_Cv8U_PhysIMU", "v8_cv8u_physimu"),
                   ("V8_Cv7_PhysIMU_Gate", "v8_cv7_physimu_gate")):
    for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"):
        _name = "%s_%s" % (_fam, _fold)
        MODEL_REGISTRY[_name] = ModelEntry(
            _name, _mod, "%s 折 %s（woS13 四折，2026-11-12）" % (_fam, _fold))

# 协议 A 主表 4 折（2026-11-05）：Cv5 折已在上方；其余四族同 split 逐折配对。
for _fam, _mod in (("V7_Cv5SD2", "v7_c2sd2"), ("V7_Cv5SD2RT", "v7_c2sd2rt"),
                   ("V7_Cv4", "v7_c"), ("V7_Cv2", "v7_c2"), ("V7_Cv6", "v7_c2")):
    for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8"):
        _name = "%s_%s" % (_fam, _fold)
        MODEL_REGISTRY[_name] = ModelEntry(
            _name, _mod, "%s 折 %s（leave-subjects-out，协议 A）" % (_fam, _fold))

# V8 系列 4 折（2026-11-05）：第 4 折用 S13-out split（test = S6+S7），
# 其余三折不变；主评估与 E5/E6 等实验沿同一折表。
for _fam, _mod in (("V8_Cv5", "v8_cv5"), ("V8_Cv4", "v8_cv4"),
                   ("V8_Cv2", "v8_cv5"), ("V8_Cv6", "v8_cv5"),
                   ("V8_Cv7", "v8_cv7"), ("V8_Cv5T1", "v8_cv5t1"),
                   ("V8_Cv5RM", "v8_cv5"), ("V8_Cv5PO", "v8_cv5po"),
                   ("V8_Cv5VG", "v8_cv5vg"), ("V8_Cv5ST", "v8_cv5"),
                   ("V8_Cv5VGST", "v8_cv5vg"), ("V8_Cv5VIG", "v8_cv5vig"),
                   ("V8_Cv5VUG", "v8_cv5vug"), ("V8_Cv8", "v8_cv8")):
    for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"):
        _name = "%s_%s" % (_fam, _fold)
        MODEL_REGISTRY[_name] = ModelEntry(
            _name, _mod, "%s 折 %s（协议 A 4 折，S13-out）" % (_fam, _fold))

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

# tw × 遮挡交互（2026-10-24）：occ020/030 × tw{5,10,15,20}（v3off 轴）。
# 模型 = V3_Bv1（v3_3b）/ V5_Av4（v5_1，carrier_from=scratch）。
for _occ in ("020", "030"):
    for _tw in (5, 10, 15):
        MODEL_REGISTRY[f"V3_Bv1_cam3_occ{_occ}_v3off_tw{_tw}"] = ModelEntry(
            f"V3_Bv1_cam3_occ{_occ}_v3off_tw{_tw}", "v3_3b",
            "tw x occlusion interaction cell (2026-10-24)")
        MODEL_REGISTRY[f"V5_Av4_cam3_occ{_occ}_v3off_tw{_tw}"] = ModelEntry(
            f"V5_Av4_cam3_occ{_occ}_v3off_tw{_tw}", "v5_1",
            "tw x occlusion interaction cell (2026-10-24)")
    MODEL_REGISTRY[f"V3_Bv1_cam3_occ{_occ}_v3off"] = ModelEntry(
        f"V3_Bv1_cam3_occ{_occ}_v3off", "v3_3b",
        "tw x occlusion interaction cell (2026-10-24)")
    MODEL_REGISTRY[f"V5_Av4_cam3_occ{_occ}_v3off"] = ModelEntry(
        f"V5_Av4_cam3_occ{_occ}_v3off", "v5_1",
        "tw x occlusion interaction cell (2026-10-24)")
