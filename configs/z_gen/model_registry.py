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
    "V3_3B": dict(label="V3-3B", name="V3_3B", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3C: frozen-V + zero-init T-residual fine-tune (VT-only windows,
    # 200ep).  Deliberate, sanctioned exception to the 2026-09-24 no-warm-start
    # independence rule — the base ckpt is the run's provenance (runner writes
    # it into run_manifest.json with sha256).
    # 2026-10-14 correction (T信息量判定实验.md §5.2): the V3_3B_vonly
    # specialist's T encoder never received a gradient (bit-identical to
    # random init after 740 epochs), so the original run regressed on random
    # features.  The base is now the derived ckpt built by
    # z_note/probes/build_v3_3c_base.py: vonly frozen base + the MAIN
    # model's TRAINED T encoder (body + modality row 1); the V path is
    # untouched so the VT == V init guarantee still holds.
    "V3_3C": dict(label="V3-3C", name="V3_3C", which="best", train=True, epochs=200,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts",
                            "gate_sigma", "residual_t"),
                  train_args={"init_from": "results/AnySole/V3_3C_joint_and/init_base_ttrained.pt",
                              "config_probs": [1.0, 0.0, 0.0],
                              "lambda_assign": 0.0, "lambda_sigma": 0.0,
                              "sigma_freeze_frac": 0.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3D: frozen-V + joint position head (2026-10-15).  Same synthetic
    # base as V3_3C (vonly V path + the main model's TRAINED foot_conv —
    # T信息量判定实验.md §5.2); the head is the only trainable part.
    # Config sampling stays 0.5/0.25/0.25 — the head is active under every
    # condition, and V-only/T-only rows feed the M1/M2/C3 position-level
    # story.  L_pos (lambda_pos=3.0) + the corrected-root L_traj are the
    # head's only supervision channels.
    "V3_3D": dict(label="V3-3D", name="V3_3D", which="best", train=True, epochs=200,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts",
                            "gate_sigma", "position_head"),
                  train_args={"init_from": "results/AnySole/V3_3C_joint_and/init_base_ttrained.pt",
                              "config_probs": [0.5, 0.25, 0.25],
                              "lambda_assign": 0.0, "lambda_sigma": 0.0,
                              "sigma_freeze_frac": 0.0,
                              "lambda_pos": 3.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3E (2026-10-17): constraint-fusion fresh architecture.  No condition
    # sampling (config_probs [1,0,0] — both streams always live, the three
    # eval conditions read one forward); no aux reconstruction; the T contact
    # head, both sigma NLL calibrations, and the T prior are the new terms.
    # lambda_contact=1.0 (pos-weighted BCE); nll weights 0.01 follow the
    # V3-3B sigma scale; the prior is fully supervised (it is the fallback).
    "V3_3E": dict(label="V3-3E", name="V3_3E", which="last", train=True, epochs=740,
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
    # V3_3F (2026-10-17): diffusion prior + T guidance, independent model.
    # No condition sampling ([1,0,0] — the three eval conditions are sampler
    # switches); no aux reconstruction; the T contact head is the only
    # supervised T output (pos-weighted f6_soft BCE).  The frozen vonly V
    # encoder is loaded by the entry (no init_from — the ckpt is provenance).
    "V3_3F": dict(label="V3-3F", name="V3_3F", which="last", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3f"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3G (2026-10-17): regression carrier (frozen E v2.1 V-stream, loaded
    # by the entry from carrier_from) + T contact stream (air-weighted BCE —
    # the observation layer's precondition) + T pose prior + part-wise
    # observation layer (inference-only, no GT floor).  V3_3G方案.md.
    "V3_3G": dict(label="V3-3G", name="V3_3G", which="best", train=True, epochs=740,
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
                              "carrier_from": "results/AnySole/V3_3E_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3H (2026-10-17): single-generator regression (frozen vonly v_enc +
    # from-zero V tower) + T contact stream (air-weighted BCE — the residual
    # channels' gate) + residual channels (inference-only).  No T pose tower
    # (T2M = null-V prior via 10% V zeroing); no sigma/NLL.  V3_3H方案.md.
    # v_cond_dropout 走 DEFAULT_CONFIG 默认 0.1（无 CLI 旗标）；floor_src
    # 开关为代码级默认 "gt"（调用点注释登记，"t" 为诊断）——均不注册为
    # train_args，runner 的 fail-loudly 校验保持零豁免。
    "V3_3H": dict(label="V3-3H", name="V3_3H", which="best", train=True, epochs=740,
                  contact_method="f6_soft",
                  provides=("base", "foot_conv", "v3h"),
                  train_args={"config_probs": [1.0, 0.0, 0.0],
                              "lambda_trec": 0.0, "lambda_vrec": 0.0,
                              "lambda_contact": 1.0,
                              "lambda_nll": 0.0,
                              "lambda_air": 2.0,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3H_v2 (2026-10-19): V3_3H + T 流 aux 压力重建（λ_trec=0.1，B 的
    # 已验证配方）——命题 2 载体依赖修复：T 编码器监督从「仅接触」恢复为
    # 「接触 + 重建」。其余与 V3_3H 完全一致（z_note/multimodal_fusion/
    # V3_3方案/命题2载体依赖.md）。训练完成后复测足部信号再预注册一阶
    # 残差头。
    "V3_3H_v2": dict(label="V3-3H-v2", name="V3_3H_v2", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3H_v3 (2026-10-19): V3_3H_v2 + 一阶足部残差头（接触软门控、输出
    # 饱和 0.05、只写足关节 6D；λ_foot_res=1.0，基础部分 detach——V 塔不
    # 被残差损失触碰）。v3 − v2 = 残差头纯贡献，v2 − v1 = 编码器监督贡献
    # （命题 2 兑现实验的消融结构）。
    "V3_3H_v3": dict(label="V3-3H-v3", name="V3_3H_v3", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3H_v5 (2026-10-19): V3_3H_v3 − 穿透通道（无锚版：T 提供不了绝对
    # 高度，地面通道整体删除，推理层只跑支撑通道 + 足部残差头，模型不
    # 消费任何 floor 输入；Pen/Global/ATE 为纯观测值）。训练规格与 v3 一致。
    "V3_3H_v5": dict(label="V3-3H-v5", name="V3_3H_v5", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V3_3H_v6 (2026-10-20): V3_3H_v5 + 残差头落点改踝 7/8（错误一修复：
    # 足关节是叶节点，改自身旋转不移动任何关键点 → 位置效应恒 0；踝旋转
    # 经 127 mm 杠杆 ≈ 平移足关键点）。损失目标槽位随之改踝（42:54），
    # 基础 detach 不变；state dict 与 v5 逐键一致（落点 = 应用层 + 损失
    # 槽位，非参数变化）。训练规格与 v5 一致。
    "V3_3H_v6": dict(label="V3-3H-v6", name="V3_3H_v6", which="best", train=True, epochs=740,
                     contact_method="f6_soft",
                     provides=("base", "foot_conv", "v3h"),
                     train_args={"config_probs": [1.0, 0.0, 0.0],
                                 "lambda_trec": 0.1, "lambda_vrec": 0.0,
                                 "lambda_contact": 1.0,
                                 "lambda_nll": 0.0,
                                 "lambda_air": 2.0,
                                 "lambda_foot_res": 1.0,
                                 "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V5_1 (2026-10-19, V5_1A方案.md): 冻结 E v2.1 V 载体（carrier_from 装载
    # 286 键）+ g_V/g_T 双 z 专家（接触 2 + 相位 4 + CoP 4，10 通道；相位
    # 保留 = 用户裁定）+ PoE 闭式合成 + FiLM 下肢/根轨迹注入（零初始化
    # 契约 1）+ G 配方接触头 + T 先验塔（T2M 健全性行）。无推理侧约束层。
    # 条件配比 VT 90% / V 零化 10%（v_cond_dropout 走 DEFAULT_CONFIG 默认
    # 0.1，无 CLI 旗标，H 同款）。损失权重单点预注册（λ 不设网格）：
    #   L_pose 3.0（V2M+VT2M 行）/ L_traj 1.0（VT 行）/ L_contact 1.0
    #   （接触头 + z 接触三读数，air 加权 2.0）/ L_phase 1.0 / L_cop 1.0
    #   （各专家 μ，掩码跳过）/ L_prior_pose 1.0（T2M 行健全性，不承诺）。
    "V5_1": dict(label="V5-1", name="V5_1", which="best", train=True, epochs=740,
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
                             "carrier_from": "results/AnySole/V3_3E_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
                             "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    # V4_1 (2026-10-19, V4_1A方案.md): 冻结 E v2.1 V 载体（carrier_from
    # 装载 286 键）+ T 接触流（air 加权 BCE，G 配方）+ T 先验流（T2M 行）
    # + 部位 token 门控修正分支（XAttn + kin 软掩码 β + 接触门控 a_p +
    # 位置修正头，零初始化契约 1）。三条件 = 同一前向三种读出，无推理
    # 侧约束层。损失权重单点预注册（λ 不设网格）：
    #   L_pose 3.0（V 流，无梯度常数）/ L_contact 1.0（air 2.0）/
    #   L_prior_pose 3.0 + L_prior_traj 1.0（先验流）/
    #   L_corr 3.0（lambda_pos 复用键，VT 行）+ L_corr_prior 同 λ（T2M 行）。
    "V4_1": dict(label="V4-1", name="V4_1", which="best", train=True, epochs=740,
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
                             "carrier_from": "results/AnySole/V3_3E_f6_soft/tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/checkpoints/ckpt_best.pt",
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


# 四相机扩展（D2，2026-10-12：相机进模型名）。训练规格与 V3_3B 完全一致；
# 数据根差异由 create_queue 按名称尾缀推导 v1_cam{N}.yaml。
_V3_3B = MODELS["V3_3B"]
for _cam in (1, 2, 3, 4):
    MODELS[f"V3_3B_cam{_cam}"] = {
        **_V3_3B,
        "label": f"V3-3B-cam{_cam}",
        "name": f"V3_3B_cam{_cam}",
    }

# 交接包降噪压力（2026-10-17）：V3_3B 同结构同规格，tactile_input 由
# v3_3b_denoised 入口固定为 denoised108（z_note/Tactile_pressure_handoff
# _20261017_v1 的 baseline_step_corrected 值，帧网格映射与 build_shared
# 同语义）。
MODELS["V3_3B_denoised"] = {
    **_V3_3B,
    "label": "V3-3B-denoised",
    "name": "V3_3B_denoised",
}

# E/F 降噪压力变体（2026-10-17）：规格与 E/F 完全一致，仅入口
# tactile_input=denoised108。
MODELS["V3_3E_denoised"] = {**MODELS["V3_3E"], "label": "V3-3E-denoised",
                            "name": "V3_3E_denoised"}
MODELS["V3_3F_denoised"] = {**MODELS["V3_3F"], "label": "V3-3F-denoised",
                            "name": "V3_3F_denoised"}

# V-degradation 遮挡组（2026-10-13：cam3 + bbox 底部遮挡 4 档）。训练规格与
# V3_3B 完全一致；数据根差异由 create_queue/runner 按名称尾缀推导
# v1_cam3_occ{N}.yaml（occ010 = 底部 10%，f=0.10）。
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_3B_cam3_occ{_occ}"] = {
        **_V3_3B,
        "label": f"V3-3B-cam3-occ{_occ}",
        "name": f"V3_3B_cam3_occ{_occ}",
    }

# V3_3D 削弱套件（2026-10-15）：训练规格与 V3_3D 完全一致；每配置一份合成
# 冻结基底（该配置 vonly 专才 + 该配置主模型受过训 foot_conv），init_from
# 指向 results/AnySole/V3_3D_<config>_joint_and/init_base_ttrained.pt。
_V3_3D = MODELS["V3_3D"]
for _cam in (1, 2, 4):
    MODELS[f"V3_3D_cam{_cam}"] = {
        **_V3_3D,
        "label": f"V3-3D-cam{_cam}",
        "name": f"V3_3D_cam{_cam}",
        "train_args": {**_V3_3D["train_args"],
                       "init_from": f"results/AnySole/V3_3D_cam{_cam}_joint_and/init_base_ttrained.pt"},
    }
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_3D_cam3_occ{_occ}"] = {
        **_V3_3D,
        "label": f"V3-3D-cam3-occ{_occ}",
        "name": f"V3_3D_cam3_occ{_occ}",
        "train_args": {**_V3_3D["train_args"],
                       "init_from": f"results/AnySole/V3_3D_cam3_occ{_occ}_joint_and/init_base_ttrained.pt"},
    }

# V3_3E 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_3E 完全一致；
# 数据根差异由 runner 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_3E = MODELS["V3_3E"]
for _occ in ("010",):
    MODELS[f"V3_3E_cam3_occ{_occ}"] = {
        **_V3_3E,
        "label": f"V3-3E-cam3-occ{_occ}",
        "name": f"V3_3E_cam3_occ{_occ}",
    }

# V3_3F 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_3F 完全一致；
# 数据根差异由 runner 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_3F = MODELS["V3_3F"]
for _occ in ("010",):
    MODELS[f"V3_3F_cam3_occ{_occ}"] = {
        **_V3_3F,
        "label": f"V3-3F-cam3-occ{_occ}",
        "name": f"V3_3F_cam3_occ{_occ}",
    }

# V3_3G 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_3G 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V3_3G = MODELS["V3_3G"]
for _occ in ("010",):
    MODELS[f"V3_3G_cam3_occ{_occ}"] = {
        **_V3_3G,
        "label": f"V3-3G-cam3-occ{_occ}",
        "name": f"V3_3G_cam3_occ{_occ}",
        "train_args": {**_V3_3G["train_args"],
                       "carrier_from": f"results/AnySole/V3_3E_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }

# V3_3H 削弱首组（2026-10-17）：cam3 + occ010，训练规格与 V3_3H 完全一致；
# 冻结 vonly V 编码器由入口按名称尾缀推导（V3_3B_vonly /
# V3_3B_cam3_occ010_vonly）；数据根差异由 runner 按名称尾缀推导
# v1_cam3_occ010.yaml。
_V3_3H = MODELS["V3_3H"]
for _occ in ("010",):
    MODELS[f"V3_3H_cam3_occ{_occ}"] = {
        **_V3_3H,
        "label": f"V3-3H-cam3-occ{_occ}",
        "name": f"V3_3H_cam3_occ{_occ}",
    }

# V3_3H_v2 削弱首组（2026-10-19）：训练规格与 V3_3H_v2 完全一致。
_V3_3H_v2 = MODELS["V3_3H_v2"]
for _occ in ("010",):
    MODELS[f"V3_3H_v2_cam3_occ{_occ}"] = {
        **_V3_3H_v2,
        "label": f"V3-3H-v2-cam3-occ{_occ}",
        "name": f"V3_3H_v2_cam3_occ{_occ}",
    }

# V3_3H_v3 削弱首组（2026-10-19）：训练规格与 V3_3H_v3 完全一致。
_V3_3H_v3 = MODELS["V3_3H_v3"]
for _occ in ("010",):
    MODELS[f"V3_3H_v3_cam3_occ{_occ}"] = {
        **_V3_3H_v3,
        "label": f"V3-3H-v3-cam3-occ{_occ}",
        "name": f"V3_3H_v3_cam3_occ{_occ}",
    }

# V3_3H_v5 削弱首组（2026-10-19）：训练规格与 V3_3H_v5 完全一致。
_V3_3H_v5 = MODELS["V3_3H_v5"]
for _occ in ("010",):
    MODELS[f"V3_3H_v5_cam3_occ{_occ}"] = {
        **_V3_3H_v5,
        "label": f"V3-3H-v5-cam3-occ{_occ}",
        "name": f"V3_3H_v5_cam3_occ{_occ}",
    }

# V3_3H_v6 削弱首组（2026-10-20）：训练规格与 V3_3H_v6 完全一致。
_V3_3H_v6 = MODELS["V3_3H_v6"]
for _occ in ("010",):
    MODELS[f"V3_3H_v6_cam3_occ{_occ}"] = {
        **_V3_3H_v6,
        "label": f"V3-3H-v6-cam3-occ{_occ}",
        "name": f"V3_3H_v6_cam3_occ{_occ}",
    }

# V3_3H_v5 难度轴扩展（2026-10-19）：cam1/cam2/cam4 + occ020，规格同 v5。
_V3_3H_v5 = MODELS["V3_3H_v5"]
for _cam in (1, 2, 4):
    MODELS[f"V3_3H_v5_cam{_cam}"] = {
        **_V3_3H_v5,
        "label": f"V3-3H-v5-cam{_cam}",
        "name": f"V3_3H_v5_cam{_cam}",
    }
for _occ in ("020",):
    MODELS[f"V3_3H_v5_cam3_occ{_occ}"] = {
        **_V3_3H_v5,
        "label": f"V3-3H-v5-cam3-occ{_occ}",
        "name": f"V3_3H_v5_cam3_occ{_occ}",
    }

# V3_3H_v5 全网格扩展（2026-10-20）：cam{1,2,4} × occ{010,020}，规格同 v5。
_V3_3H_v5 = MODELS["V3_3H_v5"]
for _cam in (1, 2, 4):
    for _occ in ("010", "020"):
        MODELS[f"V3_3H_v5_cam{_cam}_occ{_occ}"] = {
            **_V3_3H_v5,
            "label": f"V3-3H-v5-cam{_cam}-occ{_occ}",
            "name": f"V3_3H_v5_cam{_cam}_occ{_occ}",
        }

# V4_1/V5_1 V 失效梯度组（2026-10-20）：occ020/030/050；载体 = cam3 E v2.1
#（V5_1A §3 occ050 裁定口径——无 E 各档 ckpt），数据根由 runner 按尾缀推导
# v1_cam3_occ{NNN}.yaml。
for _grad_base in ("V5_1", "V4_1"):
    _grad_spec = MODELS[_grad_base]
    for _occ in ("020", "030", "050"):
        MODELS[f"{_grad_base}_cam3_occ{_occ}"] = {
            **_grad_spec,
            "label": f"{_grad_spec['label']}-cam3-occ{_occ}",
            "name": f"{_grad_base}_cam3_occ{_occ}",
        }

# V6_1A occ_mix 对照（2026-10-20）：与 V5_1/V4_1 同规格，训练侧增开
# occ_mix 三开关（预注册默认 p_occ=0.2 / p_nullV=0.1 / p_nullT=0.1）。
for _occmix_base in ("V5_1", "V4_1"):
    _occmix_spec = MODELS[_occmix_base]
    MODELS[f"{_occmix_base}_occmix"] = {
        **_occmix_spec,
        "label": f"{_occmix_spec['label']}-occmix",
        "name": f"{_occmix_base}_occmix",
        "train_args": {**_occmix_spec["train_args"],
                       "occ_mix": True,
                       "p_occ": 0.2, "p_nullV": 0.1, "p_nullT": 0.1},
    }

# V6_1A occ_mix 的 occ010 评估（2026-10-20）：eval-only 条目——复用 occmix
# 训练 ckpt（name 字段保持 occmix 基名、which=best → registry_ckpt 落到
# occmix 的 ckpt_best），数据根按 id 尾缀推导 v1_cam3_occ010.yaml；不重训。
for _occmix_base in ("V5_1", "V4_1"):
    _occ_eval_id = f"{_occmix_base}_occmix_cam3_occ010"
    _occ_eval_spec = MODELS[f"{_occmix_base}_occmix"]
    MODELS[_occ_eval_id] = {
        **_occ_eval_spec,
        "label": f"{_occ_eval_spec['label']}-cam3-occ010",
        "train": False,
    }

# V5_1 削弱首组（2026-10-19）：cam3 + occ010，训练规格与 V5_1 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V5_1 = MODELS["V5_1"]
for _occ in ("010",):
    MODELS[f"V5_1_cam3_occ{_occ}"] = {
        **_V5_1,
        "label": f"V5-1-cam3-occ{_occ}",
        "name": f"V5_1_cam3_occ{_occ}",
        "train_args": {**_V5_1["train_args"],
                       "carrier_from": f"results/AnySole/V3_3E_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }

# V4_1 削弱首组（2026-10-19）：cam3 + occ010，训练规格与 V4_1 完全一致；
# carrier_from 按配置指向该配置的 E v2.1 ckpt_best；数据根差异由 runner
# 按名称尾缀推导 v1_cam3_occ010.yaml。
_V4_1 = MODELS["V4_1"]
for _occ in ("010",):
    MODELS[f"V4_1_cam3_occ{_occ}"] = {
        **_V4_1,
        "label": f"V4-1-cam3-occ{_occ}",
        "name": f"V4_1_cam3_occ{_occ}",
        "train_args": {**_V4_1["train_args"],
                       "carrier_from": f"results/AnySole/V3_3E_cam3_occ{_occ}_f6_soft/"
                                       f"tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                                       f"checkpoints/ckpt_best.pt"},
    }


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
