"""Model registry and shared defaults for the offline experiments.

2026-09-24 independence restructure: structure flags (t_encoder / f2_repr /
pose_parts / soft_parts / gate) are no longer train CLI options — each
model's structure is fixed by its entry script (anysole/models/<name>.py),
and every run trains from random init.  ``train_args`` carries only runtime
hyperparameters.  ``part_json`` (V4B structural input) and
``assign_cluster`` (V4A train protocol) stay.
"""

from __future__ import annotations

from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1]
# V4B's partition is derived from RAW training-data kinematics (2026-09-24,
# model-independent): probe_part_cluster_b_data.py -> results/AnySole/partitions/.
# K9 = the V3_2 对位 comparison; K10 = silhouette-optimal alternative.
_V4B_PART_JSON = "results/AnySole/partitions/partitions_kinematic_b_K9.json"

MODELS = {
    "F0b": dict(label="F0b", name="F0b", which="last", train=True, epochs=400,
                provides=("base",), train_args={"grad_clip": 5.0}),
    "F4a": dict(label="F4a", name="F4a", which="last", train=True, epochs=400,
                provides=("base", "foot_conv"),
                train_args={"grad_clip": 5.0}),
    "F2": dict(label="F2", name="F2", which="last", train=True, epochs=400,
               provides=("base", "f2_repr"),
               train_args={"grad_clip": 5.0}),
    "F2p4": dict(label="F2+4", name="F2p4", which="last", train=True, epochs=400,
                 provides=("base", "foot_conv", "f2_repr"),
                 train_args={"grad_clip": 5.0}),
    "V3_2": dict(label="V3-2", name="V3_2", which="last", train=True, epochs=740,
                 provides=("base", "foot_conv", "pose_parts9"),
                 train_args={"lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_3A": dict(label="V3-3A", name="V3_3A", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts"),
                  train_args={"lambda_assign": 0.05, "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_Bv1": dict(label="V3-3B", name="V3_Bv1", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Cv1: frozen-V + zero-init T-residual fine-tune (VT-only windows,
    # 200ep).  Deliberate, sanctioned exception to the 2026-09-24 no-warm-start
    # independence rule — the base ckpt is the run's provenance (runner writes
    # it into run_manifest.json with sha256).
    # 2026-10-14 correction (z_note/multimodal_fusion/Appendix/T信息量判定实验.md §5.2): the V3_Bv1_vonly
    # specialist's T encoder never received a gradient (bit-identical to
    # random init after 740 epochs), so the original run regressed on random
    # features.  The base is now the derived ckpt built by
    # z_note/probes/build_v3_3c_base.py: vonly frozen base + the MAIN
    # model's TRAINED T encoder (body + modality row 1); the V path is
    # untouched so the VT == V init guarantee still holds.
    "V3_Cv1": dict(label="V3-3C", name="V3_Cv1", which="best", train=True, epochs=200,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts",
                            "gate_sigma", "residual_t"),
                  train_args={"init_from": "results/AnySole/V3_Cv1_joint_and/init_base_ttrained.pt",
                              "config_probs": [1.0, 0.0, 0.0],
                              "lambda_assign": 0.0, "lambda_sigma": 0.0,
                              "sigma_freeze_frac": 0.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Dv1: frozen-V + joint position head (2026-10-15).  Same synthetic
    # base as V3_Cv1 (vonly V path + the main model's TRAINED foot_conv —
    # z_note/multimodal_fusion/Appendix/T信息量判定实验.md §5.2); the head is the only trainable part.
    # Config sampling stays 0.5/0.25/0.25 — the head is active under every
    # condition, and V-only/T-only rows feed the M1/M2/C3 position-level
    # story.  L_pos (lambda_pos=3.0) + the corrected-root L_traj are the
    # head's only supervision channels.
    "V3_Dv1": dict(label="V3-3D", name="V3_Dv1", which="best", train=True, epochs=200,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts",
                            "gate_sigma", "position_head"),
                  train_args={"init_from": "results/AnySole/V3_Cv1_joint_and/init_base_ttrained.pt",
                              "config_probs": [0.5, 0.25, 0.25],
                              "lambda_assign": 0.0, "lambda_sigma": 0.0,
                              "sigma_freeze_frac": 0.0,
                              "lambda_pos": 3.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Ev1 (2026-10-17): constraint-fusion fresh architecture.  No condition
    # sampling (config_probs [1,0,0] — both streams always live, the three
    # eval conditions read one forward); no aux reconstruction; the T contact
    # head, both sigma NLL calibrations, and the T prior are the new terms.
    # lambda_contact=1.0 (pos-weighted BCE); nll weights 0.01 follow the
    # V3-3B sigma scale; the prior is fully supervised (it is the fallback).
    "V3_Ev1": dict(label="V3-3E", name="V3_Ev1", which="last", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3e"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lambda_nll": 0.0,
                              "lambda_prior_pose": 3.0,
                              "lambda_prior_nll": 0.0,
                              "lambda_prior_traj": 1.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Fv1 (2026-10-17): diffusion prior + T guidance, independent model.
    # No condition sampling ([1,0,0] — the three eval conditions are sampler
    # switches); no aux reconstruction; the T contact head is the only
    # supervised T output (pos-weighted f6_soft BCE).  The frozen vonly V
    # encoder is loaded by the entry (no init_from — the ckpt is provenance).
    "V3_Fv1": dict(label="V3-3F", name="V3_Fv1", which="last", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3f"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Gv1 (2026-10-17): regression carrier (frozen E v2.1 V-stream, loaded
    # by the entry from carrier_from) + T contact stream (air-weighted BCE —
    # the observation layer's precondition) + T pose prior + part-wise
    # observation layer (inference-only, no GT floor).  V3_G方案_GT燃料.md.
    "V3_Gv1": dict(label="V3-3G", name="V3_Gv1", which="best", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3g"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lambda_nll": 0.0,
                              "lambda_prior_pose": 3.0,
                              "lambda_prior_nll": 0.0,
                              "lambda_prior_traj": 1.0,
                              "lambda_air": 2.0,
                              "g_floor_src": "gt",
                              "carrier_from": "results/AnySole/V3_Ev1_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Hv1 (2026-10-17): single-generator regression (frozen vonly v_enc +
    # from-zero V tower) + T contact stream (air-weighted BCE — the residual
    # channels' gate) + residual channels (inference-only).  No T pose tower
    # (T2M = null-V prior via 10% V zeroing); no sigma/NLL.  V3_H方案.md.
    # v_cond_dropout 走 DEFAULT_CONFIG 默认 0.1（无 CLI 旗标）；floor_src
    # 开关为代码级默认 "gt"（调用点注释登记，"t" 为诊断）——均不注册为
    # train_args，runner 的 fail-loudly 校验保持零豁免。
    "V3_Hv1": dict(label="V3-3H", name="V3_Hv1", which="best", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3h"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lambda_nll": 0.0,
                              "lambda_air": 2.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Hv2 (2026-10-19): V3_Hv1 + T 流 aux 压力重建（λ_trec=0.1，B 的
    # 已验证配方）——命题 2 载体依赖修复：T 编码器监督从「仅接触」恢复为
    # 「接触 + 重建」。其余与 V3_Hv1 完全一致（z_note/multimodal_fusion/
    # V3_方案/命题2载体依赖.md）。训练完成后复测足部信号再预注册一阶
    # 残差头。
    "V3_Hv2": dict(label="V3-3H-v2", name="V3_Hv2", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Hv3 (2026-10-19): V3_Hv2 + 一阶足部残差头（接触软门控、输出
    # 饱和 0.05、只写足关节 6D；λ_foot_res=1.0，基础部分 detach——V 塔不
    # 被残差损失触碰）。v3 − v2 = 残差头纯贡献，v2 − v1 = 编码器监督贡献
    # （命题 2 兑现实验的消融结构）。
    "V3_Hv3": dict(label="V3-3H-v3", name="V3_Hv3", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Hv5 (2026-10-19): V3_Hv3 − 穿透通道（无锚版：T 提供不了绝对
    # 高度，地面通道整体删除，推理层只跑支撑通道 + 足部残差头，模型不
    # 消费任何 floor 输入；Pen/Global/ATE 为纯观测值）。训练规格与 v3 一致。
    "V3_Hv5": dict(label="V3-3H-v5", name="V3_Hv5", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_Hv6 (2026-10-20): V3_Hv5 + 残差头落点改踝 7/8（错误一修复：
    # 足关节是叶节点，改自身旋转不移动任何关键点 → 位置效应恒 0；踝旋转
    # 经 127 mm 杠杆 ≈ 平移足关键点）。损失目标槽位随之改踝（42:54），
    # 基础 detach 不变；state dict 与 v5 逐键一致（落点 = 应用层 + 损失
    # 槽位，非参数变化）。训练规格与 v5 一致。
    "V3_Hv6": dict(label="V3-3H-v6", name="V3_Hv6", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V5_Av1 (2026-10-19, V5_A方案.md): 冻结 E v2.1 V 载体（carrier_from 装载
    # 286 键）+ g_V/g_T 双 z 专家（接触 2 + 相位 4 + CoP 4，10 通道；相位
    # 保留 = 用户裁定）+ PoE 闭式合成 + FiLM 下肢/根轨迹注入（零初始化
    # 契约 1）+ G 配方接触头 + T 先验塔（T2M 健全性行）。无推理侧约束层。
    # 条件配比 VT 90% / V 零化 10%（v_cond_dropout 走 DEFAULT_CONFIG 默认
    # 0.1，无 CLI 旗标，H 同款）。损失权重单点预注册（λ 不设网格）：
    #   L_pose 3.0（V2M+VT2M 行）/ L_traj 1.0（VT 行）/ L_contact 1.0
    #   （接触头 + z 接触三读数，air 加权 2.0）/ L_phase 1.0 / L_cop 1.0
    #   （各专家 μ，掩码跳过）/ L_prior_pose 1.0（T2M 行健全性，不承诺）。
    "V5_Av1": dict(label="V5-1", name="V5_Av1", which="best", train=True, epochs=740,
                 contact_method="f6_soft",
                 provides=("base", "foot_conv", "v5_1"),
                 train_args={"config_probs": [1.0, 0.0, 0.0],
                             "lambda_trec": 0.0, "lambda_vrec": 0.0,
                             "lambda_contact": 1.0,
                             "lambda_nll": 0.0,
                             "lambda_prior_pose": 1.0,
                             "lambda_prior_traj": 0.0,
                             "lambda_air": 2.0,
                             "lambda_phase": 1.0,
                             "lambda_cop": 1.0,
                             "carrier_from": "results/AnySole/V3_Ev1_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
                             "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V4_Av1 (2026-10-19, V4_A方案.md): 冻结 E v2.1 V 载体（carrier_from
    # 装载 286 键）+ T 接触流（air 加权 BCE，G 配方）+ T 先验流（T2M 行）
    # + 部位 token 门控修正分支（XAttn + kin 软掩码 β + 接触门控 a_p +
    # 位置修正头，零初始化契约 1）。三条件 = 同一前向三种读出，无推理
    # 侧约束层。损失权重单点预注册（λ 不设网格）：
    #   L_pose 3.0（V 流，无梯度常数）/ L_contact 1.0（air 2.0）/
    #   L_prior_pose 3.0 + L_prior_traj 1.0（先验流）/
    #   L_corr 3.0（lambda_pos 复用键，VT 行）+ L_corr_prior 同 λ（T2M 行）。
    "V4_Av1": dict(label="V4-1", name="V4_Av1", which="best", train=True, epochs=740,
                 contact_method="f6_soft",
                 provides=("base", "foot_conv", "v4_1"),
                 train_args={"config_probs": [1.0, 0.0, 0.0],
                             "lambda_trec": 0.0, "lambda_vrec": 0.0,
                             "lambda_contact": 1.0,
                             "lambda_nll": 0.0,
                             "lambda_prior_pose": 3.0,
                             "lambda_prior_nll": 0.0,
                             "lambda_prior_traj": 1.0,
                             "lambda_air": 2.0,
                             "lambda_pos": 3.0,
                             "beta": 1.0,
                             "carrier_from": "results/AnySole/V3_Ev1_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
                             "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4a": dict(label="V3-4a", name="V3_4a", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9"),
                  train_args={"lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4b": dict(label="V3-4b", name="V3_4b", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts"),
                  train_args={"lambda_assign": 0.05, "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4c": dict(label="V3-4c", name="V3_4c", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V4A": dict(label="V4A", name="V4A", which="last", train=True, epochs=740,
                provides=("base", "foot_conv", "pose_parts24", "soft_parts", "assign_cluster"),
                train_args={"assign_cluster": True, "lambda_assign": 1.0,
                            "lambda_assign_ent": 0.05, "lambda_assign_conc": 0.01,
                            "lambda_assign_dead": 0.02, "assign_dead_beta": 2.0,
                            "assign_temp_init": 1.0, "assign_temp_final": 0.2,
                            "assign_anneal_frac": 0.7, "assign_lock_frac": 0.7,
                            "assign_lr_mult": 5.0, "lr_warmup_frac": 0.05,
                            "grad_clip": 5.0, "loss_cap": 10.0}),
    "V4B": dict(label="V4B", name="V4B", which="best", train=True, epochs=740,
                provides=("base", "foot_conv", "pose_parts9", "learned_partition"),
                train_args={"part_json": _V4B_PART_JSON,
                            "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
}


# 四相机扩展（D2，2026-10-12：相机进模型名）。训练规格与 V3_Bv1 完全一致；
# 数据根差异由 create_queue 按名称尾缀推导 v1_cam{N}.yaml。
_V3_Bv1 = MODELS["V3_Bv1"]
for _cam in (1, 2, 3, 4):
    MODELS[f"V3_Bv1_cam{_cam}"] = {
        **_V3_Bv1,
        "label": f"V3-3B-cam{_cam}",
        "name": f"V3_Bv1_cam{_cam}",
    }

# 交接包降噪压力（2026-10-17）：V3_Bv1 同结构同规格，tactile_input 由
# v3_3b_denoised 入口固定为 denoised108（z_note/data_pressure_fixpipe/Tactile_pressure_handoff
# _20261017_v1 的 baseline_step_corrected 值，帧网格映射与 build_shared
# 同语义）。
MODELS["V3_Bv1_denoised"] = {
    **_V3_Bv1,
    "label": "V3-3B-denoised",
    "name": "V3_Bv1_denoised",
}

# E/F 降噪压力变体（2026-10-17）：规格与 E/F 完全一致，仅入口
# tactile_input=denoised108。
MODELS["V3_Ev1_denoised"] = {**MODELS["V3_Ev1"], "label": "V3-3E-denoised",
                            "name": "V3_Ev1_denoised"}
MODELS["V3_Fv1_denoised"] = {**MODELS["V3_Fv1"], "label": "V3-3F-denoised",
                            "name": "V3_Fv1_denoised"}

# V-degradation 遮挡组（2026-10-13：cam3 + bbox 底部遮挡 4 档）。训练规格与
# V3_Bv1 完全一致；数据根差异由 create_queue/runner 按名称尾缀推导
# v1_cam3_occ{N}.yaml（occ010 = 底部 10%，f=0.10）。
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_Bv1_cam3_occ{_occ}"] = {
        **_V3_Bv1,
        "label": f"V3-3B-cam3-occ{_occ}",
        "name": f"V3_Bv1_cam3_occ{_occ}",
    }

# offset 重同步组（2026-10-20）：训练规格与 V3_Bv1 完全一致，唯一差异 =
# corrected_offset=True（B 类 session GT 采样按 z_note/data_offset_fixpipe
# 修正表平移，A/C/D 不动）。yaml 尾缀推导兼容 `_offfix` 结尾
# （_model_v_config_suffix 的 `(?:$|_)`），cam3_offfix → v1_cam3.yaml、
# cam3_occ{N}_offfix → v1_cam3_occ{N}.yaml。
MODELS["V3_Bv1_cam3_offfix"] = {
    **_V3_Bv1,
    "label": "V3-3B-cam3-offfix",
    "name": "V3_Bv1_cam3_offfix",
    "train_args": {**_V3_Bv1["train_args"]},
}
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_Bv1_cam3_occ{_occ}_offfix"] = {
        **_V3_Bv1,
        "label": f"V3-3B-cam3-occ{_occ}-offfix",
        "name": f"V3_Bv1_cam3_occ{_occ}_offfix",
        "train_args": {**_V3_Bv1["train_args"]},
    }

# v2/v3 + offset 组（2026-10-20）：结构 = v3_3b_v2 / v3_3b_v3（tactile_input
# 由结构固定 denoised108_v2 / denoised108_v3），其余同 offset 组。
for _ver in ("v2", "v3"):
    MODELS[f"V3_Bv1_cam3_offfix_{_ver}"] = {
        **_V3_Bv1,
        "label": f"V3-3B-cam3-offfix-{_ver}",
        "name": f"V3_Bv1_cam3_offfix_{_ver}",
        "train_args": {**_V3_Bv1["train_args"]},
    }
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"V3_Bv1_cam3_occ{_occ}_offfix_{_ver}"] = {
            **_V3_Bv1,
            "label": f"V3-3B-cam3-occ{_occ}-offfix-{_ver}",
            "name": f"V3_Bv1_cam3_occ{_occ}_offfix_{_ver}",
            "train_args": {**_V3_Bv1["train_args"]},
        }

# 新数据验证（2026-10-22）：默认口径 = 生产 v3+offset facts。
MODELS["V3_Bv1_cam3_v3off"] = {**_V3_Bv1, "label": "V3-3B-cam3-v3off",
                              "name": "V3_Bv1_cam3_v3off"}

# E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型，
# 生产 v3+offset facts（无 offfix 分支——offset 已整合）。
#   统一 = dropout [0.5,0.25,0.25]；vonly/tonly = 同架构单模态专家；
#   nodrop（3B）/ 基名（E、V4_Av1、V5_Av1）= 始终双模态 [1,0,0]（VT 专家）。
#   V4_Av1/V5_Av1 的 carrier_from = 新轴 E（V3_Ev1_cam3_v3off，随波 1 训练）。
# 波次（实际执行）：波 1 = E x4；波 2 = 3B x3；波 3 = V4_Av1/V5_Av1 x8（闸门 =
# E 基名完成，载体快照一致）。
_E1_V3OFF = ("results/AnySole/V3_Ev1_cam3_v3off_f6_soft/"
             "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
             "checkpoints/ckpt_best.pt")
_V3_Bv1_V3OFF = MODELS["V3_Bv1_cam3_v3off"]
# V3_Dv1 削弱套件（2026-10-15）：训练规格与 V3_Dv1 完全一致；每配置一份合成
for _suf, _cprobs in (("_vonly", [0.0, 1.0, 0.0]), ("_tonly", [0.0, 0.0, 1.0]),
                     ("_nodrop", [1.0, 0.0, 0.0])):
    MODELS["V3_Bv1_cam3_v3off" + _suf] = {
        **_V3_Bv1_V3OFF, "label": "V3-3B-cam3-v3off" + _suf.replace("_", "-"),
        "name": "V3_Bv1_cam3_v3off" + _suf,
        "train_args": {**_V3_Bv1_V3OFF["train_args"], "config_probs": _cprobs}}
for _suf, _cprobs in (("", [1.0, 0.0, 0.0]), ("_drop", [0.5, 0.25, 0.25]),
                     ("_vonly", [0.0, 1.0, 0.0]), ("_tonly", [0.0, 0.0, 1.0])):
    _b = MODELS["V3_Ev1"]
    MODELS["V3_Ev1_cam3_v3off" + _suf] = {
        **_b, "label": "V3-3E-cam3-v3off" + _suf.replace("_", "-"),
        "name": "V3_Ev1_cam3_v3off" + _suf,
        "train_args": {**_b["train_args"], "config_probs": _cprobs}}
for _suf, _cprobs in (("", [1.0, 0.0, 0.0]), ("_drop", [0.5, 0.25, 0.25]),
                     ("_vonly", [0.0, 1.0, 0.0]), ("_tonly", [0.0, 0.0, 1.0])):
    _b = MODELS["V4_Av1"]
    MODELS["V4_Av1_cam3_v3off" + _suf] = {
        **_b, "label": "V4-1-cam3-v3off" + _suf.replace("_", "-"),
        "name": "V4_Av1_cam3_v3off" + _suf,
        "train_args": {**_b["train_args"], "config_probs": _cprobs,
                       "carrier_from": _E1_V3OFF}}
for _suf, _cprobs in (("", [1.0, 0.0, 0.0]), ("_drop", [0.5, 0.25, 0.25]),
                     ("_vonly", [0.0, 1.0, 0.0]), ("_tonly", [0.0, 0.0, 1.0])):
    _b = MODELS["V5_Av1"]
    MODELS["V5_Av1_cam3_v3off" + _suf] = {
        **_b, "label": "V5-1-cam3-v3off" + _suf.replace("_", "-"),
        "name": "V5_Av1_cam3_v3off" + _suf,
        "train_args": {**_b["train_args"], "config_probs": _cprobs,
                       "carrier_from": _E1_V3OFF}}

# V5_Av4（2026-10-22，V5_A方案 修复迭代）：V5_Av1 结构 + V 流从零训练
# （carrier_from="scratch"，去 E 继承）+ 先验塔 E 配方（λ_prior_pose 3.0
# / λ_prior_traj 1.0——Av1 的 1.0/0.0 使 T2M MPJPE 180.7 / ATE 291 远差于
# E 配方的 146.5 / 84）。E1 四变体 × v3off 生产轴。
_V5_AV4_TA = {**_b["train_args"], "lambda_prior_pose": 3.0,
              "lambda_prior_traj": 1.0, "carrier_from": "scratch"}
for _suf, _cprobs in (("", [1.0, 0.0, 0.0]), ("_drop", [0.5, 0.25, 0.25]),
                     ("_vonly", [0.0, 1.0, 0.0]), ("_tonly", [0.0, 0.0, 1.0])):
    MODELS["V5_Av4_cam3_v3off" + _suf] = {
        **_b, "label": "V5-Av4-cam3-v3off" + _suf.replace("_", "-"),
        "name": "V5_Av4_cam3_v3off" + _suf,
        "train_args": {**_V5_AV4_TA, "config_probs": _cprobs}}

# 5-fold 动作覆盖实验（2026-10-22，AnysoleWorkspace/protocol/splits/default/
# splits_test_*.csv）：受试者轮换 leave-subjects-out——5 个 fold 各自一个
# split 文件，test 覆盖全部 11 动作。fold 命名 = 统一编号 plain 名 + fold
# 尾缀（V5_Av4 同理，不带轴标签；offset 已整合进生产 facts、无 offset
# 分支）。fold 条目 = 基座规格逐项继承，差异：
#   config  → 每 fold 专属 yaml（split_csv 指向该 fold 文件；canonical
#             fold fS13S14 = v1.yaml 本身）；
#   载体    → V5_Av1/V4_Av1 的 carrier_from 换接该 fold 的 V3_Ev1
#             ckpt_best（波 1 训练，波 3 闸门 = 波 1 完成）；
#   dump_detail → eval 附 --dump-detail（session_detail 逐会话行 =
#             按动作聚合的 canonical 数据源）。
# 波次：波 1 = V3_Ev1 ×5；波 2 = V3_Bv1/V5_Av4 ×5（无依赖，随波 1 并行）；
# 波 3 = V5_Av1/V4_Av1 ×5（闸门 = 对应 fold E1 完成）。conf 生成 =
# configs/z_gen/per_action_folds.py（--stage 12|3，波 3 带闸门断言）。
_FOLDS = (("fS13S14", "anysole/configs/v1.yaml"),
          ("fS5S6S11", "anysole/configs/v1_fold_S5S6S11.yaml"),
          ("fS5S11S12", "anysole/configs/v1_fold_S5S11S12.yaml"),
          ("fS6S7S13", "anysole/configs/v1_fold_S6S7S13.yaml"),
          ("fS5S6S8", "anysole/configs/v1_fold_S5S6S8.yaml"))
# E1 fold 训练运行目录尾（tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_
# bs256_sd1；基座逐项同参 → 目录名确定）。
_FOLD_E1_CKPT = ("tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                 "checkpoints/ckpt_best.pt")
for _fold, _yaml in _FOLDS:
    MODELS["V3_Ev1_" + _fold] = {
        **MODELS["V3_Ev1"], "label": "V3-3E-" + _fold, "name": "V3_Ev1_" + _fold,
        "config": _yaml, "dump_detail": True}
    # 本轮口径：所有模型接触方法统一 f6_soft（含 V3_Bv1 fold）。
    MODELS["V3_Bv1_" + _fold] = {
        **MODELS["V3_Bv1"], "label": "V3-3B-" + _fold, "name": "V3_Bv1_" + _fold,
        "config": _yaml, "contact_method": "f6_soft", "dump_detail": True}
    # V5_Av4 fold：统一编号 plain 名，配方继承 V5_Av4（= V5_Av4_cam3_v3off
    # 基座条目；该条目保留不动——历史 conf 引用它）。
    MODELS["V5_Av4_" + _fold] = {
        **MODELS["V5_Av4_cam3_v3off"], "label": "V5-Av4-" + _fold,
        "name": "V5_Av4_" + _fold,
        "config": _yaml, "dump_detail": True}
    _carrier = ("results/AnySole/V3_Ev1_%s_f6_soft/%s" % (_fold, _FOLD_E1_CKPT))
    MODELS["V5_Av1_" + _fold] = {
        **MODELS["V5_Av1"], "label": "V5-1-" + _fold, "name": "V5_Av1_" + _fold,
        "config": _yaml, "dump_detail": True,
        "train_args": {**MODELS["V5_Av1"]["train_args"], "carrier_from": _carrier}}
    MODELS["V4_Av1_" + _fold] = {
        **MODELS["V4_Av1"], "label": "V4-1-" + _fold, "name": "V4_Av1_" + _fold,
        "config": _yaml, "dump_detail": True,
        "train_args": {**MODELS["V4_Av1"]["train_args"], "carrier_from": _carrier}}

# 冻结基底（该配置 vonly 专才 + 该配置主模型受过训 foot_conv），init_from
# 指向 results/AnySole/V3_Dv1_<config>_joint_and/init_base_ttrained.pt。
_V3_Dv1 = MODELS["V3_Dv1"]
for _cam in (1, 2, 4):
    MODELS[f"V3_Dv1_cam{_cam}"] = {
        **_V3_Dv1,
        "label": f"V3-3D-cam{_cam}",
        "name": f"V3_Dv1_cam{_cam}",
        "train_args": {**_V3_Dv1["train_args"],
                       "init_from": f"results/AnySole/V3_Dv1_cam{_cam}_joint_and/init_base_ttrained.pt"},
    }
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_Dv1_cam3_occ{_occ}"] = {
        **_V3_Dv1,
        "label": f"V3-3D-cam3-occ{_occ}",
        "name": f"V3_Dv1_cam3_occ{_occ}",
        "train_args": {**_V3_Dv1["train_args"],
                       "init_from": f"results/AnySole/V3_Dv1_cam3_occ{_occ}_joint_and/init_base_ttrained.pt"},
    }

# V3_Ev1 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_Ev1 完全一致；
# 数据根差异由 runner 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_Ev1 = MODELS["V3_Ev1"]
for _occ in ("010",):
    MODELS[f"V3_Ev1_cam3_occ{_occ}"] = {
        **_V3_Ev1,
        "label": f"V3-3E-cam3-occ{_occ}",
        "name": f"V3_Ev1_cam3_occ{_occ}",
    }

# V3_Fv1 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_Fv1 完全一致；
# 数据根差异由 runner 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_Fv1 = MODELS["V3_Fv1"]
for _occ in ("010",):
    MODELS[f"V3_Fv1_cam3_occ{_occ}"] = {
        **_V3_Fv1,
        "label": f"V3-3F-cam3-occ{_occ}",
        "name": f"V3_Fv1_cam3_occ{_occ}",
    }

# V3_Gv1 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_Gv1 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_Gv1 = MODELS["V3_Gv1"]
for _occ in ("010",):
    MODELS[f"V3_Gv1_cam3_occ{_occ}"] = {
        **_V3_Gv1,
        "label": f"V3-3G-cam3-occ{_occ}",
        "name": f"V3_Gv1_cam3_occ{_occ}",
        "train_args": {**_V3_Gv1["train_args"],
                       "carrier_from": f"results/AnySole/V3_Ev1_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }

# V3_Hv1 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_Hv1 完全一致；
# 冻结 vonly V 编码器由入口按名称尾缀推导（V3_Bv1_vonly /
# V3_Bv1_cam3_occ010_vonly）；数据根差异由 runner 按名称尾缀推导
# v1_cam3_occ010.yaml。
_V3_Hv1 = MODELS["V3_Hv1"]
for _occ in ("010",):
    MODELS[f"V3_Hv1_cam3_occ{_occ}"] = {
        **_V3_Hv1,
        "label": f"V3-3H-cam3-occ{_occ}",
        "name": f"V3_Hv1_cam3_occ{_occ}",
    }

# V3_Hv2 削弱首组（2026-10-19）：训练规格与 V3_Hv2 完全一致。
_V3_Hv2 = MODELS["V3_Hv2"]
for _occ in ("010",):
    MODELS[f"V3_Hv2_cam3_occ{_occ}"] = {
        **_V3_Hv2,
        "label": f"V3-3H-v2-cam3-occ{_occ}",
        "name": f"V3_Hv2_cam3_occ{_occ}",
    }

# V3_Hv3 削弱首组（2026-10-19）：训练规格与 V3_Hv3 完全一致。
_V3_Hv3 = MODELS["V3_Hv3"]
for _occ in ("010",):
    MODELS[f"V3_Hv3_cam3_occ{_occ}"] = {
        **_V3_Hv3,
        "label": f"V3-3H-v3-cam3-occ{_occ}",
        "name": f"V3_Hv3_cam3_occ{_occ}",
    }

# V3_Hv5 削弱首组（2026-10-19）：训练规格与 V3_Hv5 完全一致。
_V3_Hv5 = MODELS["V3_Hv5"]
for _occ in ("010",):
    MODELS[f"V3_Hv5_cam3_occ{_occ}"] = {
        **_V3_Hv5,
        "label": f"V3-3H-v5-cam3-occ{_occ}",
        "name": f"V3_Hv5_cam3_occ{_occ}",
    }

# V3_Hv6 削弱首组（2026-10-20）：训练规格与 V3_Hv6 完全一致。
_V3_Hv6 = MODELS["V3_Hv6"]
for _occ in ("010",):
    MODELS[f"V3_Hv6_cam3_occ{_occ}"] = {
        **_V3_Hv6,
        "label": f"V3-3H-v6-cam3-occ{_occ}",
        "name": f"V3_Hv6_cam3_occ{_occ}",
    }

# V3_Hv5 难度轴扩展（2026-10-19）：cam1/cam2/cam4 + occ020，规格同 v5。
_V3_Hv5 = MODELS["V3_Hv5"]
for _cam in (1, 2, 4):
    MODELS[f"V3_Hv5_cam{_cam}"] = {
        **_V3_Hv5,
        "label": f"V3-3H-v5-cam{_cam}",
        "name": f"V3_Hv5_cam{_cam}",
    }
for _occ in ("020",):
    MODELS[f"V3_Hv5_cam3_occ{_occ}"] = {
        **_V3_Hv5,
        "label": f"V3-3H-v5-cam3-occ{_occ}",
        "name": f"V3_Hv5_cam3_occ{_occ}",
    }

# V3_Hv5 全网格扩展（2026-10-20）：cam{1,2,4} × occ{010,020}，规格同 v5。
_V3_Hv5 = MODELS["V3_Hv5"]
for _cam in (1, 2, 4):
    for _occ in ("010", "020"):
        MODELS[f"V3_Hv5_cam{_cam}_occ{_occ}"] = {
            **_V3_Hv5,
            "label": f"V3-3H-v5-cam{_cam}-occ{_occ}",
            "name": f"V3_Hv5_cam{_cam}_occ{_occ}",
        }

# V4_Av1/V5_Av1 V 失效梯度组（2026-10-20）：occ020/030/050；载体 = cam3 E v2.1
#（V5_Av1 §3 occ050 裁定口径——无 E 各档 ckpt），数据根由 runner 按尾缀推导
# v1_cam3_occ{NNN}.yaml。
for _grad_base in ("V5_Av1", "V4_Av1"):
    _grad_spec = MODELS[_grad_base]
    for _occ in ("020", "030", "050"):
        MODELS[f"{_grad_base}_cam3_occ{_occ}"] = {
            **_grad_spec,
            "label": f"{_grad_spec['label']}-cam3-occ{_occ}",
            "name": f"{_grad_base}_cam3_occ{_occ}",
        }


# 5-fold occ050 档（2026-10-23 裁定「加」）：V5_Av1/V4_Av1 × 5 fold ×
# occ050——按动作 T 兑现只能在 occ 档测（cam3 池化干预已证 T 不参与）。
# 基座 = V 失效梯度组条目（同 plain 基座规格，载体换接该 fold 的 E1）；
# config = occ-fold yaml（cache_root = cam3_occ_lower050 + fold split）。
_FOLD_OCC_YAMLS = {"fS13S14": "anysole/configs/v1_cam3_occ050.yaml"}
_FOLD_OCC_YAMLS.update({_f: "anysole/configs/v1_occ050_fold_%s.yaml" % _f[1:]
                        for _f, _ in _FOLDS if _f != "fS13S14"})
for _fold, _occ_yaml in _FOLD_OCC_YAMLS.items():
    _carrier = ("results/AnySole/V3_Ev1_%s_f6_soft/%s" % (_fold, _FOLD_E1_CKPT))
    for _pre in ("V5_Av1", "V4_Av1"):
        MODELS[f"{_pre}_occ050_{_fold}"] = {
            **MODELS[f"{_pre}_cam3_occ050"], "label": f"{_pre}-occ050-" + _fold,
            "name": f"{_pre}_occ050_{_fold}",
            "config": _occ_yaml, "dump_detail": True,
            "train_args": {**MODELS[f"{_pre}_cam3_occ050"]["train_args"],
                           "carrier_from": _carrier}}
    # 全面添加（2026-10-23）：V3_Bv1（无载体；接触 f6_soft 覆盖）与
    # V5_Av4（配方继承 V5_Av4_cam3_v3off，carrier=scratch）。
    MODELS[f"V3_Bv1_occ050_{_fold}"] = {
        **MODELS["V3_Bv1_cam3_occ050"], "label": "V3-3B-occ050-" + _fold,
        "name": f"V3_Bv1_occ050_{_fold}", "contact_method": "f6_soft",
        "config": _occ_yaml, "dump_detail": True}
    MODELS[f"V5_Av4_occ050_{_fold}"] = {
        **MODELS["V5_Av4_cam3_v3off"], "label": "V5-Av4-occ050-" + _fold,
        "name": f"V5_Av4_occ050_{_fold}",
        "config": _occ_yaml, "dump_detail": True}


# V6_Av1 occ_mix 对照（2026-10-20）：与 V5_Av1/V4_Av1 同规格，训练侧增开
# occ_mix 三开关（预注册默认 p_occ=0.2 / p_nullV=0.1 / p_nullT=0.1）。
for _occmix_base in ("V5_Av1", "V4_Av1"):
    _occmix_spec = MODELS[_occmix_base]
    MODELS[f"{_occmix_base}_occmix"] = {
        **_occmix_spec,
        "label": f"{_occmix_spec['label']}-occmix",
        "name": f"{_occmix_base}_occmix",
        "train_args": {**_occmix_spec["train_args"],
                       "occ_mix": True,
                       "p_occ": 0.2, "p_nullV": 0.1, "p_nullT": 0.1},
    }

# V6_Av1 occ_mix 的 occ010 评估（2026-10-20）：eval-only 条目——复用 occmix
# 训练 ckpt（name 字段保持 occmix 基名、which=best → registry_ckpt 落到
# occmix 的 ckpt_best），数据根按 id 尾缀推导 v1_cam3_occ010.yaml；不重训。
for _occmix_base in ("V5_Av1", "V4_Av1"):
    _occ_eval_id = f"{_occmix_base}_occmix_cam3_occ010"
    _occ_eval_spec = MODELS[f"{_occmix_base}_occmix"]
    MODELS[_occ_eval_id] = {
        **_occ_eval_spec,
        "label": f"{_occ_eval_spec['label']}-cam3-occ010",
        "train": False,
    }

# V5_Av1 削弱首组（2026-10-19）：cam3 + occ010，训练规格与 V5_Av1 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V5_Av1 = MODELS["V5_Av1"]
for _occ in ("010",):
    MODELS[f"V5_Av1_cam3_occ{_occ}"] = {
        **_V5_Av1,
        "label": f"V5-1-cam3-occ{_occ}",
        "name": f"V5_Av1_cam3_occ{_occ}",
        "train_args": {**_V5_Av1["train_args"],
                       "carrier_from": f"results/AnySole/V3_Ev1_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }

# V4_Av1 削弱首组（2026-10-19）：cam3 + occ010，训练规格与 V4_Av1 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V4_Av1 = MODELS["V4_Av1"]
for _occ in ("010",):
    MODELS[f"V4_Av1_cam3_occ{_occ}"] = {
        **_V4_Av1,
        "label": f"V4-1-cam3-occ{_occ}",
        "name": f"V4_Av1_cam3_occ{_occ}",
        "train_args": {**_V4_Av1["train_args"],
                       "carrier_from": f"results/AnySole/V3_Ev1_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }

# 5-fold occ010 档（2026-10-23 裁定「加」）：同 occ050 结构。
_FOLD_OCC010_YAMLS = {"fS13S14": "anysole/configs/v1_cam3_occ010.yaml"}
_FOLD_OCC010_YAMLS.update({_f: "anysole/configs/v1_occ010_fold_%s.yaml" % _f[1:]
                           for _f, _ in _FOLDS if _f != "fS13S14"})
for _fold, _occ_yaml in _FOLD_OCC010_YAMLS.items():
    _carrier = ("results/AnySole/V3_Ev1_%s_f6_soft/%s" % (_fold, _FOLD_E1_CKPT))
    for _pre in ("V5_Av1", "V4_Av1"):
        MODELS[f"{_pre}_occ010_{_fold}"] = {
            **MODELS[f"{_pre}_cam3_occ010"], "label": f"{_pre}-occ010-" + _fold,
            "name": f"{_pre}_occ010_{_fold}",
            "config": _occ_yaml, "dump_detail": True,
            "train_args": {**MODELS[f"{_pre}_cam3_occ010"]["train_args"],
                           "carrier_from": _carrier}}
    # 全面添加（2026-10-23）：V3_Bv1（无载体；接触 f6_soft 覆盖）与
    # V5_Av4（配方继承 V5_Av4_cam3_v3off——occ010_v3off 基座同配方）。
    MODELS[f"V3_Bv1_occ010_{_fold}"] = {
        **MODELS["V3_Bv1_cam3_occ010"], "label": "V3-3B-occ010-" + _fold,
        "name": f"V3_Bv1_occ010_{_fold}", "contact_method": "f6_soft",
        "config": _occ_yaml, "dump_detail": True}
    MODELS[f"V5_Av4_occ010_{_fold}"] = {
        **MODELS["V5_Av4_cam3_v3off"], "label": "V5-Av4-occ010-" + _fold,
        "name": f"V5_Av4_occ010_{_fold}",
        "config": _occ_yaml, "dump_detail": True}


_COMMON_FALLBACK = {
    "CONFIG": "anysole/configs/v1.yaml",
    "RESULTS_ROOT": "results",
    "DISPLAY_ROOT": "results_display",
    "MODAL": "anysolev2",
    "CONTACT_METHOD": "joint_and",
    "DEVICE": "cuda",
    "TRAIN_SEED": "1",
    "EPOCHS": "800",
    "BATCH_SIZE": "256",
    "PROTOCOL_SEED": "0",
    "DISPLAY": "false",
}


def _load_defaults() -> dict[str, str]:
    values = dict(_COMMON_FALLBACK)
    path = CONFIG_DIR / "defaults.conf"
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            values[key] = value.split("#", 1)[0].strip()
    return values


COMMON = _load_defaults()

# offset 设计组（2026-10-21）：E v2.1 / V5_Av1 / V4_Av1 / V3_Hv1 的
# corrected_offset 版。载体换接：V5_Av1/V4_Av1 的 carrier_from → E_offfix
# （occ020/030/050 沿用 cam3 载体口径）；V3_Hv1 的 vonly 载体按名推导。
_E_OFF = ("results/AnySole/V3_Ev1_cam3_offfix_f6_soft/"
          "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
          "checkpoints/ckpt_best.pt")
_E_OFF_OCC010 = ("results/AnySole/V3_Ev1_cam3_occ010_offfix_f6_soft/"
                 "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                 "checkpoints/ckpt_best.pt")
for _name, _base in (("V3_Ev1_cam3_offfix", "V3_Ev1"), ("V3_Ev1_cam3_occ010_offfix", "V3_Ev1_cam3_occ010")):
    MODELS[_name] = {**MODELS[_base], "label": _name.replace("_", "-"), "name": _name,
                     "train_args": {**MODELS[_base]["train_args"]}}
for _occ, _base_occ, _carrier in (("", "", _E_OFF), ("_occ010", "_cam3_occ010", _E_OFF_OCC010),
                                  ("_occ020", "_cam3_occ020", _E_OFF), ("_occ030", "_cam3_occ030", _E_OFF),
                                  ("_occ050", "_cam3_occ050", _E_OFF)):
    for _prefix, _base_prefix in (("V5_Av1", "V5_Av1"), ("V4_Av1", "V4_Av1")):
        _n = f"{_prefix}_cam3{_occ}_offfix"
        _b = f"{_base_prefix}_cam3{_occ}" if _occ else _prefix
        if _b not in MODELS:
            continue
        MODELS[_n] = {**MODELS[_b], "label": _n.replace("_", "-"), "name": _n,
                      "train_args": {**MODELS[_b]["train_args"],
                                     "carrier_from": _carrier}}
# T 增强三组（2026-10-21，T增强三组实验方案.md）：
# V5_Av1_cam3_offfix_tphys = 维度 2 显式物理特征（T_phys 12→20）；
# V5_Av1_cam3_offfix_d8   = 维度 3 足-地监督（占位，监督头落地后启用）。
MODELS["V5_Av1_cam3_offfix_tphys"] = {
    **MODELS["V5_Av1_cam3_offfix"], "label": "V5-1-cam3-offfix-tphys",
    "name": "V5_Av1_cam3_offfix_tphys",
    "train_args": {**MODELS["V5_Av1_cam3_offfix"]["train_args"], "t_phys_v2": True}}
MODELS["V5_Av1_cam3_offfix_d8"] = {
    **MODELS["V5_Av1_cam3_offfix"], "label": "V5-1-cam3-offfix-d8",
    "name": "V5_Av1_cam3_offfix_d8",
    "train_args": {**MODELS["V5_Av1_cam3_offfix"]["train_args"],
                   "d8_heads": True, "lambda_tcop": 0.1,
                   "lambda_strike": 0.1, "lambda_gtcop": 0.1}}
# v2 + offset 设计组（2026-10-21）：V5_Av1 / V4_Av1 × cam3/cam3_occ010 的
# denoised108_v2 版。与 offset 设计组逐项相同（corrected_offset=True、
# carrier_from = E_offfix 系），唯一差异 = 结构固定
# tactile_input=denoised108_v2（v5_1_v2 / v4_1_v2）。
for _occ in ("", "_occ010"):
    for _prefix in ("V5_Av1", "V4_Av1"):
        _b = f"{_prefix}_cam3{_occ}_offfix"
        MODELS[f"{_b}_v2"] = {
            **MODELS[_b],
            "label": f"{_b}_v2".replace("_", "-"),
            "name": f"{_b}_v2",
        }
for _occ in ("", "_occ010"):
    _n = f"V3_Hv1_cam3{_occ}_offfix"
    _b = f"V3_Hv1_cam3{_occ}" if _occ else "V3_Hv1"
    MODELS[_n] = {**MODELS[_b], "label": _n.replace("_", "-"), "name": _n,
                  "train_args": {**MODELS[_b]["train_args"]}}

# 时间窗扫描（2026-10-22，z_note/multimodal_fusion/时间窗扫描实验.md）：
# 五家族 x tw{5,10,15,20} x cam3/cam3_occ010（v3off 生产轴）。tw 是训练期
# 协变量——eval 从 ckpt 读 tw（anysole/eval.py），评估期不可换窗，每格独立
# 训练；stride 随 tw 非重叠（stride=tw 全贴片，tw5 = 4× 窗口数 ≈ 4× epoch
# 时长）。E（V3_Ev1）格 = V4_Av1/V5_Av1 的同 (tw, 视觉) 载体（载体冻结自
# 同配置 E ckpt_best；tw≠20 键不匹配、无 E 载体 build 硬失败）。
# 波 1 = E + V3_Bv1 + V5_Av4；波 2 = V4_Av1 + V5_Av1（闸门 = 波 1 E 格完成）。
from anysole.types import variant_from_config

_TW_VALUES = (5, 10, 15)


def _e_carrier_cell(e_name: str, tw: int) -> str:
    """E 载体地址 = 同 (tw, 视觉) 格的 ckpt_best（f6_soft；variant 与 train 一致）。"""
    variant = variant_from_config({
        "tw": tw, "stride": None, "lr": 1e-4, "lambda_pose": 3.0,
        "lambda_traj": 1.0, "lambda_kp": 1.0, "lr_warmup_frac": 0.05,
        "grad_clip": 5.0, "epochs": 740, "batch_size": 256, "seed": 1,
    })
    return f"results/AnySole/{e_name}_f6_soft/{variant}/checkpoints/ckpt_best.pt"


def _tw_args(base_train_args, tw, carrier: str | None = None):
    args = {**base_train_args, "tw": tw, "stride": tw}
    if carrier is not None:
        args["carrier_from"] = carrier
    return args


# --- E 格（V4_Av1/V5_Av1 的载体源；cam3 tw20 已有 = V3_Ev1_cam3_v3off）---
for _tw in _TW_VALUES:
    _b = MODELS["V3_Ev1_cam3_v3off"]
    _n = f"V3_Ev1_cam3_v3off_tw{_tw}"
    MODELS[_n] = {**_b, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b["train_args"], _tw)}
_b_occ = MODELS["V3_Ev1_cam3_occ010"]
_n_occ = "V3_Ev1_cam3_occ010_v3off"
MODELS[_n_occ] = {**_b_occ, "label": _n_occ.replace("_", "-"), "name": _n_occ,
                  "train_args": {**_b_occ["train_args"]}}
for _tw in _TW_VALUES:
    _n = f"V3_Ev1_cam3_occ010_v3off_tw{_tw}"
    MODELS[_n] = {**_b_occ, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b_occ["train_args"], _tw)}

# --- V3_Bv1 格（cam3 tw20 已有 = V3_Bv1_cam3_v3off）---
for _tw in _TW_VALUES:
    _b = MODELS["V3_Bv1_cam3_v3off"]
    _n = f"V3_Bv1_cam3_v3off_tw{_tw}"
    MODELS[_n] = {**_b, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b["train_args"], _tw)}
_b_occ = MODELS["V3_Bv1_cam3_occ010"]
_n_occ = "V3_Bv1_cam3_occ010_v3off"
MODELS[_n_occ] = {**_b_occ, "label": _n_occ.replace("_", "-"), "name": _n_occ,
                  "train_args": {**_b_occ["train_args"]}}
for _tw in _TW_VALUES:
    _n = f"V3_Bv1_cam3_occ010_v3off_tw{_tw}"
    MODELS[_n] = {**_b_occ, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b_occ["train_args"], _tw)}

# --- V5_Av4 格（carrier_from="scratch" 不变；cam3 tw20 已有 = V5_Av4_cam3_v3off）---
for _tw in _TW_VALUES:
    _b = MODELS["V5_Av4_cam3_v3off"]
    _n = f"V5_Av4_cam3_v3off_tw{_tw}"
    MODELS[_n] = {**_b, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b["train_args"], _tw)}
_b_occ = MODELS["V5_Av4_cam3_v3off"]
_n_occ = "V5_Av4_cam3_occ010_v3off"
MODELS[_n_occ] = {**_b_occ, "label": _n_occ.replace("_", "-"), "name": _n_occ,
                  "train_args": {**_b_occ["train_args"]}}
for _tw in _TW_VALUES:
    _n = f"V5_Av4_cam3_occ010_v3off_tw{_tw}"
    MODELS[_n] = {**_b_occ, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": _tw_args(_b_occ["train_args"], _tw)}

# --- V4_Av1 / V5_Av1 格（carrier = 同格 E；cam3 tw20 已有）---
for _prefix, _occ_base in (("V4_Av1", "V4_Av1_cam3_occ010"),
                           ("V5_Av1", "V5_Av1_cam3_occ010")):
    _b_cam3 = MODELS[f"{_prefix}_cam3_v3off"]
    for _tw in _TW_VALUES:
        _n = f"{_prefix}_cam3_v3off_tw{_tw}"
        MODELS[_n] = {**_b_cam3, "label": _n.replace("_", "-"), "name": _n,
                      "train_args": _tw_args(
                          _b_cam3["train_args"], _tw,
                          _e_carrier_cell(f"V3_Ev1_cam3_v3off_tw{_tw}", _tw))}
    _b_occ = MODELS[_occ_base]
    _n_occ = f"{_prefix}_cam3_occ010_v3off"
    MODELS[_n_occ] = {**_b_occ, "label": _n_occ.replace("_", "-"), "name": _n_occ,
                      "train_args": {**_b_occ["train_args"],
                                     "carrier_from": _e_carrier_cell(
                                         "V3_Ev1_cam3_occ010_v3off", 20)}}
    for _tw in _TW_VALUES:
        _n = f"{_prefix}_cam3_occ010_v3off_tw{_tw}"
        MODELS[_n] = {**_b_occ, "label": _n.replace("_", "-"), "name": _n,
                      "train_args": _tw_args(
                          _b_occ["train_args"], _tw,
                          _e_carrier_cell(f"V3_Ev1_cam3_occ010_v3off_tw{_tw}", _tw))}
