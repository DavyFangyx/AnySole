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

from anysole.types import variant_from_config


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
    # V3_Bv2/V3_Bv3（2026-10-24）：Bv1 + 查询压力读出头（Q-DIRECT/Q-NN、
    # Q-FACT 一个机制三表示，pressure_head 为唯一切换参数）；cell 均值
    # 先验由 train 启动时一遍拟合冻结进 ckpt config，yaml 只冻结表示选择。
    "V3_Bv2": dict(label="V3-3B-v2", name="V3_Bv2", which="last", train=True, epochs=740,
                  config="anysole/configs/v1_bv2.yaml",
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_Bv3": dict(label="V3-3B-v3", name="V3_Bv3", which="last", train=True, epochs=740,
                  config="anysole/configs/v1_bv3.yaml",
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

# V-degradation 遮挡组（2026-10-13：cam3 + bbox 底部遮挡 4 档）。训练规格与
# V3_Bv1 完全一致；数据根差异由 create_queue/runner 按名称尾缀推导
# v1_cam3_occ{N}.yaml（occ010 = 底部 10%，f=0.10）。
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V3_Bv1_cam3_occ{_occ}"] = {
        **_V3_Bv1,
        "label": f"V3-3B-cam3-occ{_occ}",
        "name": f"V3_Bv1_cam3_occ{_occ}",
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

# 条件合作线（2026-10-25，用户裁定「做几个新模型，cam3 + cam3_occ010 实验」）：
# 框架保留 V4/V5 融合读出（门控/FiLM），通路换 E 式双流同训
# （carrier_from="scratch"——V 流可训、不再压倒 T）+ occ_mix（V6 配方：
# 训练侧 V 失效窗口 p_occ=0.2 / V 零化 0.1 / T 零化 0.1）+ 条件契约
# λ_cond=3.0（V 健康窗口罚修正量、V 失效窗口自由——让位规则写进目标
# 函数，不靠初始化）。_nocond = 同配方去契约（隔离契约贡献）。先验塔
# λ 沿用各家族现有配方（V5 = 3/1；V4 = 3/1 + L_corr 3.0）。which=last
# （避免 ep5 选择假象重演；occ010 轴无 occ_mix——窗口天然 V 退化）。
_V5_AV5_TA = {**_V5_AV4_TA, "occ_mix": True,
              "p_occ": 0.2, "p_nullV": 0.1, "p_nullT": 0.1,
              "lambda_cond": 3.0}
_V4_AV2_TA = {**MODELS["V4_Av1"]["train_args"], "carrier_from": "scratch",
              "occ_mix": True, "p_occ": 0.2, "p_nullV": 0.1, "p_nullT": 0.1,
              "lambda_cond": 3.0}
_OCC_MIX_KEYS = ("occ_mix", "p_occ", "p_nullV", "p_nullT")
for _name, _ta in (("V5_Av5", _V5_AV5_TA), ("V4_Av2", _V4_AV2_TA)):
    for _suf, _suf_ta in (("", _ta),
                          ("_nocond", {k: v for k, v in _ta.items()
                                       if k != "lambda_cond"})):
        _id = _name + _suf
        MODELS[_id] = dict(label=_id.replace("_", "-"), name=_id, which="last",
                           train=True, epochs=740, contact_method="f6_soft",
                           provides=("base",), train_args=_suf_ta)
        _occ_id = _id + "_cam3_occ010"
        MODELS[_occ_id] = dict(
            label=_occ_id.replace("_", "-"), name=_occ_id, which="last",
            train=True, epochs=740, contact_method="f6_soft",
            provides=("base",),
            train_args={k: v for k, v in _suf_ta.items() if k not in _OCC_MIX_KEYS})

# V4_Av3（2026-10-25，用户裁定「B 的 V 部分作为通路、T 解毒」折中方案）：
# V4 修正分支挂冻结 B 骨架（两阶段：阶段 1 = V3_Bv1 生产 ckpt_best 冻结
# ——其 V2M 读出带融合税，修正分支在健康窗口学解毒、失效窗口学地面帮助；
# 阶段 2 = 只训修正分支 + 接触头）。无条件契约（解毒要求健康窗口也出手，
# 契约会压制它）；occ_mix 提供 V 失效窗口训练分布（cam3 轴）。λ_pose/
# λ_prior 置 0（骨架冻结、其损失恒为常数）。
_V4_AV3_CARRIER = ("results/AnySole/V3_Bv1_cam3_v3off_joint_and/"
                   "tw20_st20_lr0.0001_lp3_lt1_lk1_la0.05_ls0.01_sf0.1_wu0.05_gc5_ep740_bs256_sd1/"
                   "checkpoints/ckpt_best.pt")
_V4_AV3_TA = {
    "config_probs": [1.0, 0.0, 0.0],
    "lambda_trec": 0.0, "lambda_vrec": 0.0,
    "lambda_pose": 0.0,
    "lambda_contact": 1.0, "lambda_nll": 0.0,
    "lambda_prior_pose": 0.0, "lambda_prior_nll": 0.0, "lambda_prior_traj": 0.0,
    "lambda_air": 2.0, "lambda_pos": 3.0, "beta": 1.0,
    "carrier_from": _V4_AV3_CARRIER,
    "lr_warmup_frac": 0.05, "grad_clip": 5.0}
MODELS["V4_Av3"] = dict(label="V4-Av3", name="V4_Av3", which="last",
                        train=True, epochs=740, contact_method="f6_soft",
                        provides=("base",), train_args={**_V4_AV3_TA,
                        "occ_mix": True, "p_occ": 0.2,
                        "p_nullV": 0.1, "p_nullT": 0.1})
MODELS["V4_Av3_cam3_occ010"] = dict(label="V4-Av3-cam3-occ010",
                                    name="V4_Av3_cam3_occ010", which="last",
                                    train=True, epochs=740,
                                    contact_method="f6_soft",
                                    provides=("base",), train_args=_V4_AV3_TA)

# V4_Av3 ckpt_last 重读（2026-10-25）：正式评估落 ckpt_best=ep5（骨架冻结
# → val/tau0 VT 平坦 → 选择假象重演）——补 eval-only last 重读。
for _b in ("V4_Av3", "V4_Av3_cam3_occ010"):
    MODELS[_b + "_last_eval"] = {
        **MODELS[_b], "label": MODELS[_b]["label"] + "-last-eval",
        "train": False, "which": "last"}

# V4_Av4（2026-10-25，用户裁定「冻结并非合适选择」→ 双阶段精调）：V4_Av3
# 结构 + 骨架不冻结、carrier_lr_mult=0.1（骨架组学习率 0.1×、修正分支与
# 接触头 1.0——阶段 1 = B 生产 ckpt 作初始化，阶段 2 = 全模型微调，上限
# 不锁死但骨架有锚）。V4_Av3（冻结）保留为对照。
for _n, _ta in (("V4_Av4", {**_V4_AV3_TA, "carrier_lr_mult": 0.1}),):
    MODELS[_n] = dict(label=_n.replace("_", "-"), name=_n, which="last",
                      train=True, epochs=740, contact_method="f6_soft",
                      provides=("base",), train_args={**_ta,
                      "occ_mix": True, "p_occ": 0.2,
                      "p_nullV": 0.1, "p_nullT": 0.1})
    _occ_id = _n + "_cam3_occ010"
    MODELS[_occ_id] = dict(label=_occ_id.replace("_", "-"), name=_occ_id,
                           which="last", train=True, epochs=740,
                           contact_method="f6_soft", provides=("base",),
                           train_args=_ta)

# V4_Av4_drop 对照（2026-10-25 用户追加）：同配方 + config_probs 0.5/0.25/0.25
# ——检验 drop 的毒性是否随「载体可训」消失（vs 冻结载体 drop = T 死/有害；
# vs V4_Av4 [1,0,0] = 无 drop 基准）。cam3 轴继承 occ_mix、occ010 轴无。
_V4_AV4_DROP_TA = {**MODELS["V4_Av4"]["train_args"],
                   "config_probs": [0.5, 0.25, 0.25]}
MODELS["V4_Av4_drop"] = {**MODELS["V4_Av4"], "label": "V4-Av4-drop",
                         "name": "V4_Av4_drop",
                         "train_args": _V4_AV4_DROP_TA}
MODELS["V4_Av4_drop_cam3_occ010"] = {
    **MODELS["V4_Av4_cam3_occ010"], "label": "V4-Av4-drop-cam3-occ010",
    "name": "V4_Av4_drop_cam3_occ010",
    "train_args": {k: v for k, v in _V4_AV4_DROP_TA.items()
                   if k not in _OCC_MIX_KEYS}}

# V7 候选 A/B/C（2026-10-25，模型设计思路.md §3，用户裁定「编号 + 实现 +
# 接入管线 + 冒烟」）：独立证据流 + 共享运动解码 + 三条件硬关闭分支；
# 从零训练（无载体）；config_probs [0.5,0.25,0.25]（ckpt 规则 §3：明确
# 三条件采样）；checkpoint_policy = multi_metric（ckpt 规则：角色按
# config_probs 推断为 U）。B = A + 足-地一致性（λ_foot 1.0）；C = A +
# 窗口级 T 信任门（v1 端到端，λ_route 0）。
_V7_TA = {
    "config_probs": [0.5, 0.25, 0.25],
    "lambda_pose": 3.0, "lambda_traj": 1.0,
    "lambda_contact": 1.0, "lambda_air": 2.0,
    "lambda_trec": 0.0, "lambda_vrec": 0.0, "lambda_nll": 0.0,
    "lr_warmup_frac": 0.05, "grad_clip": 5.0,
    "checkpoint_policy": "multi_metric", "checkpoint_role": "auto",
    # E2/E5 选点需要完整选域：四阈值带内取 VT 最优要求全留协议
    # （2026-10-26 裁定；其余实验默认 checkpoint_all=false 只滚存最近过线者）。
    "checkpoint_all": True,
    "wandb_mode": "online", "wandb_experiment_tag": "v7_first_round",
}
# B/C 的机制由 STRUCTURE 标志驱动（v7_contact_root / v7_reliability_route，
# 随 config 注入）——train.py 无 --lambda-foot/--lambda-route CLI（曾被
# argparse 前缀匹配静默接到 --lambda-foot-res），损失侧改用结构开关。
for _name, _extra in (("V7_Av1", {}), ("V7_Bv1", {}), ("V7_Cv1", {})):
    MODELS[_name] = dict(label=_name.replace("_", "-"), name=_name, which="last",
                         train=True, epochs=740, contact_method="f6_soft",
                         provides=("base",),
                         config="anysole/configs/v1_wo05.yaml",
                         train_args={**_V7_TA, **_extra})

# V7 端到端预检条目（2026-10-25 计划步骤 1）：同名结构、epochs=6 进 spec
# ——EPOCHS conf 覆盖不可行（变体推断/完成度检查/multi_metric 存档都按
# spec epochs 走），预检必须用独立条目（独立 run 目录、新鲜存档）。
# V7 E1 专家（2026-10-26 用户裁定「补 E1 缺口」）：同架构 S_V / S_T——
# 骨架 = 候选 A（v7_a 入口），仅训练配比不同；checkpoint_role auto 按
# config_probs 推断为对应专家（不参与 U 的选点判定，固定各自完整权重）。
for _exp, _cprobs in (("V7_SV1", [0.0, 1.0, 0.0]), ("V7_ST1", [0.0, 0.0, 1.0])):
    MODELS[_exp] = {**MODELS["V7_Av1"], "label": _exp.replace("_", "-"),
                    "name": _exp,
                    "train_args": {**MODELS["V7_Av1"]["train_args"],
                                   "config_probs": _cprobs,
                                   "wandb_experiment_tag": "v7_experts"}}

# V7 E5 occ050 筛选（2026-10-25 计划步骤 3）：eval-only 零重训——C = 保留
# 候选（val E2 零税筛查通过）；A/B = 无门对照行（同 ckpt 的 occ050 读数
# 用于解释门在 V 失效下的价值）。which=last（正式 ckpt 口径）。
# V7_Cv1 候选复评（2026-10-25）：multi_metric 保留权重 + last 的 val 复评
# ——回答「dPA 最优点在其他 Δ 上是否也最优」。注意：ep730 未被档案保留
# （权重不存在），晚期候选用 670/680 替代。name 保持 V7_Cv1（入口与
# ckpt 口径），explicit checkpoint 指向档案权重。
_V7_CX_TA = {**_V7_TA, "checkpoint_all": True, "final_eval_split": "none",
             "wandb_experiment_tag": "v7_variants"}
for _new_name in ("V7_Cv2", "V7_Cv3", "V7_Cv4"):
    _new_ta = {**_V7_CX_TA, "condition_sampling": "enum" if _new_name == "V7_Cv4" else "sample"}
    MODELS[_new_name] = dict(
        label=_new_name.replace("_", "-"), name=_new_name, which="last",
        train=True, epochs=740, contact_method="f6_soft", provides=("base",),
        config="anysole/configs/v1_wo05.yaml",
        eval_splits=["val"], dump_detail=True, train_args=_new_ta)
    _check_name = _new_name + "_precheck"
    MODELS[_check_name] = {
        **MODELS[_new_name], "label": _check_name.replace("_", "-"),
        "name": _check_name, "epochs": 6,
        "train_args": {**_new_ta, "wandb_experiment_tag": "v7_variants_precheck"}}

# V7_Cv5（2026-10-26 用户裁定落地）：V7_Cv2 双独立 T 门 + 枚举三条件
# 训练（Cv4 的 condition_sampling=enum）——E2 侧（枚举）与 E5 侧（双门）
# 两个已兑现的改进合并。
MODELS["V7_Cv5"] = {**MODELS["V7_Cv2"], "label": "V7-Cv5", "name": "V7_Cv5",
                    "train_args": {**MODELS["V7_Cv2"]["train_args"],
                                   "condition_sampling": "enum"}}
MODELS["V7_Cv5_precheck"] = {**MODELS["V7_Cv5"], "label": "V7-Cv5-precheck",
                             "name": "V7_Cv5_precheck", "epochs": 6,
                             "train_args": {**MODELS["V7_Cv5"]["train_args"],
                                            "wandb_experiment_tag": "v7_variants_precheck"}}

# V7_Cv6（2026-10-26 用户裁定，消融组「V5 无模态缺失训练版」）：双门结构
# （v7_c2 入口）同 Cv5，训练始终输入 VT（config_probs [1,0,0] + sample——
# 不做模态 mask/dropout，也不枚举）——验证三条件训练是否有必要。
# checkpoint_role auto → S_VT（仅 VT 单条件选点）。
MODELS["V7_Cv6"] = {**MODELS["V7_Cv2"], "label": "V7-Cv6", "name": "V7_Cv6",
                    "train_args": {**MODELS["V7_Cv2"]["train_args"],
                                   "config_probs": [1.0, 0.0, 0.0],
                                   "condition_sampling": "sample"}}
MODELS["V7_Cv6_precheck"] = {**MODELS["V7_Cv6"], "label": "V7-Cv6-precheck",
                             "name": "V7_Cv6_precheck", "epochs": 6,
                             "train_args": {**MODELS["V7_Cv6"]["train_args"],
                                            "wandb_experiment_tag": "v7_variants_precheck"}}

# V7_Cv7（2026-10-26 用户裁定，消融组「A＋枚举训练版」）：无门控（v7_a
# 入口）+ 枚举三条件训练，其余配方与 V7_Cv5 相同——验证可靠性门控是否
# 有必要。checkpoint_role auto → U（三条件同 Cv5）。
MODELS["V7_Cv7"] = {**MODELS["V7_Cv5"], "label": "V7-Cv7", "name": "V7_Cv7"}
MODELS["V7_Cv7_precheck"] = {**MODELS["V7_Cv7"], "label": "V7-Cv7-precheck",
                             "name": "V7_Cv7_precheck", "epochs": 6,
                             "train_args": {**MODELS["V7_Cv7"]["train_args"],
                                            "wandb_experiment_tag": "v7_variants_precheck"}}

# V7_Cv5 全量套餐专家（2026-10-26 用户裁定）：同架构 = v7_c2（双门结构
# 本体），仅训练配比不同——与 U 逐参数同构（E1 的同架构要求）。对比：
# 早期的 V7_SV1/V7_ST1 是 v7_a 骨架，与 V7_Cv1（v7_c）非同架构。
for _exp, _cprobs in (("V7_Cv5_SV1", [0.0, 1.0, 0.0]), ("V7_Cv5_ST1", [0.0, 0.0, 1.0])):
    MODELS[_exp] = {**MODELS["V7_Cv5"], "label": _exp.replace("_", "-"),
                    "name": _exp,
                    "eval_splits": ["val", "test"],
                    "train_args": {**MODELS["V7_Cv5"]["train_args"],
                                   "config_probs": _cprobs,
                                   "wandb_experiment_tag": "v7_variants_experts"}}



# V1-last 统计（2026-10-26 用户裁定「统一口径」）：V1 = V7_Cv1 基准——
# name 落 V7_Cv1 的 run 目录，which=last（与 V2/V3/V1_enum 同口径）。
MODELS["V7_Cv1_statistics"] = {**MODELS["V7_Cv1"], "label": "V7-Cv1-stat",
                           "name": "V7_Cv1", "train": False, "which": "last",
                           "eval_splits": ["val", "test"]}
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V7_Cv1_statistics_evalocc{_occ}"] = {
        **MODELS["V7_Cv1"], "label": f"V7-Cv1-stat-evalocc{_occ}",
        "name": "V7_Cv1", "train": False, "which": "last",
        "eval_splits": ["val", "test"],
        "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

# V1–V3 统计评估（2026-10-26 用户裁定「进行统计」）：各变体四门选点 =
# ckpt_last（V2/V3/V1_enum 各仅 1 份四门通过者 = ep740）。cam3 test 面板 +
# occ 四档交叉评估（V1 = V7_Cv1 基准，数字已有）。
# 专家 test 补跑（2026-10-26：clean 条目继承 eval_splits=["val"] 导致
# 专家 test 缺失——eval-only 格补，不重训）。
for _exp in ("V7_Cv5_SV1", "V7_Cv5_ST1"):
    MODELS[f"{_exp}_statistics"] = {**MODELS[_exp], "label": _exp + "-stat",
                              "name": _exp, "train": False, "which": "last",
                              "eval_splits": ["val", "test"]}

for _v in ("V7_Cv2", "V7_Cv3", "V7_Cv4"):
    MODELS[f"{_v}_statistics"] = {**MODELS[_v], "label": _v.replace("_", "-") + "-stat",
                            "name": _v, "train": False, "which": "last",
                            "eval_splits": ["val", "test"]}
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"{_v}_statistics_evalocc{_occ}"] = {
            **MODELS[_v], "label": _v.replace("_", "-") + f"-stat-evalocc{_occ}",
            "name": _v, "train": False, "which": "last",
            "eval_splits": ["val", "test"],
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

# V7_Cv5 统计（2026-10-26）：ckpt_last 固定口径（选点规则第三版）。
MODELS["V7_Cv5_statistics"] = {**MODELS["V7_Cv5"], "label": "V7-Cv5-stat",
                         "name": "V7_Cv5", "train": False, "which": "last",
                         "eval_splits": ["val", "test"]}
for _occ in ("010", "020", "030", "050"):
    MODELS[f"V7_Cv5_statistics_evalocc{_occ}"] = {
        **MODELS["V7_Cv5"], "label": f"V7-Cv5-stat-evalocc{_occ}",
        "name": "V7_Cv5", "train": False, "which": "last",
        "eval_splits": ["val", "test"],
        "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

# V7_Cv6/Cv7 统计条目（2026-10-27 接入，消融组 E4 用）：eval-only 零重训，
# 口径与 Cv5 stat 一致——cam3 test 面板 + occ 四档交叉评估（name 落训练
# 条目名、which=last）。训练（00139/00140）完成后由 gen 接线发射。
for _v in ("V7_Cv6", "V7_Cv7"):
    MODELS[f"{_v}_statistics"] = {**MODELS[_v], "label": _v.replace("_", "-") + "-stat",
                            "name": _v, "train": False, "which": "last",
                            "eval_splits": ["val", "test"]}
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"{_v}_statistics_evalocc{_occ}"] = {
            **MODELS[_v], "label": _v.replace("_", "-") + f"-stat-evalocc{_occ}",
            "name": _v, "train": False, "which": "last",
            "eval_splits": ["val", "test"],
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}



_V7_C_ARCHIVE = ("results/AnySole/V7_Cv1_f6_soft/"
                 "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                 "checkpoints/multi_metric/weights/epoch_000%s.pt")
for _ep in ("600", "665", "670", "680"):
    MODELS[f"V7_Cv1_cand{_ep}"] = {
        **MODELS["V7_Cv1"], "label": f"V7-Cv1-cand{_ep}", "name": "V7_Cv1",
        "train": False, "which": "last",
        "checkpoint": _V7_C_ARCHIVE % _ep}
MODELS["V7_Cv1_cand740"] = {**MODELS["V7_Cv1"], "label": "V7-Cv1-cand740",
                            "name": "V7_Cv1", "train": False, "which": "last"}

# 第二轮选点（2026-10-26，ckpt 规则 §2：四门全过 → VT PA 最小）：
# A=ep210（VT 36.33）/ B=ep205（37.02）/ C=ep665（31.49，dPA −0.24）。
# E2/E5 评估 = 固定该权重（explicit checkpoint + 哈希登记）。
_V7_R2_SEL = {
    "V7_Av1": ("epoch_000210.pt", "V7-Av1-sel"),
    "V7_Bv1": ("epoch_000205.pt", "V7-Bv1-sel"),
    "V7_Cv1": ("epoch_000665.pt", "V7-Cv1-sel"),
}
for _v7, (_w, _lab) in _V7_R2_SEL.items():
    _base = f"results/AnySole/{_v7}_f6_soft/"
    _base += ("tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
              "checkpoints/multi_metric/weights/%s" % _w)
    MODELS[f"{_v7}_sel"] = {**MODELS[_v7], "label": _lab, "name": _v7,
                            "train": False, "which": "last",
                            "checkpoint": _base}
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"{_v7}_sel_evalocc{_occ}"] = {
            **MODELS[_v7], "label": _lab + f"-evalocc{_occ}", "name": _v7,
            "train": False, "which": "last", "checkpoint": _base,
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

# V7_Cv1 第一轮调参 18 格（2026-10-26 用户裁定）：固定 λ_pose 3 / λ_traj 1 /
# λ_air 2（基座不变），网格 = lr {5e-5,1e-4,2e-4} × λ_contact {0.1,0.3,1.0}
# × lr_schedule {constant,cosine}。名称编码参数——λ_contact 与 lr_schedule
# 不在 VARIANT_FIELDS 内，同名会撞运行目录（λ 扫描先例）。
_V7C_LR = {"5e-05": 5e-5, "0.0001": 1e-4, "0.0002": 2e-4}
_V7C_LC = {"0.1": 0.1, "0.3": 0.3, "1": 1.0}
_V7C_CS = {"const": "constant", "cos": "cosine"}
for _lrk, _lr in _V7C_LR.items():
    for _lck, _lc in _V7C_LC.items():
        for _csk, _cs in _V7C_CS.items():
            _n = "V7_Cv1_lr%s_lc%s_%s" % (_lrk, _lck, _csk)
            MODELS[_n] = {
                **MODELS["V7_Cv1"], "label": _n.replace("_", "-"),
                "name": _n,
                "train_args": {**MODELS["V7_Cv1"]["train_args"],
                               "lr": _lr, "lambda_contact": _lc,
                               "lr_schedule": _cs}}

for _v7 in ("V7_Av1", "V7_Bv1", "V7_Cv1"):
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"{_v7}_evalocc{_occ}"] = {
            **MODELS[_v7], "label": MODELS[_v7]["label"] + f"-evalocc{_occ}",
            "train": False, "which": "last",
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

# ckpt 口径对照（2026-10-26 用户裁定）：E2/E5 对照补 best 口径行
# （V4/V5 时代对照曾用 best；V7 统一 last 前，best 行作对照参考）。
for _v7 in ("V7_Av1", "V7_Bv1", "V7_Cv1"):
    MODELS[f"{_v7}_evalbest"] = {
        **MODELS[_v7], "label": MODELS[_v7]["label"] + "-evalbest",
        "train": False, "which": "best"}
    for _occ in ("010", "020", "030", "050"):
        MODELS[f"{_v7}_evalbest_occ{_occ}"] = {
            **MODELS[_v7], "label": MODELS[_v7]["label"] + f"-evalbest-occ{_occ}",
            "train": False, "which": "best",
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml"}

for _name in ("V7_Av1", "V7_Bv1", "V7_Cv1"):
    _pid = _name + "_precheck"
    MODELS[_pid] = {**MODELS[_name], "label": _pid.replace("_", "-"),
                    "name": _pid, "epochs": 6}
    _occ_id = _name + "_cam3_occ010"
    MODELS[_occ_id] = dict(label=_occ_id.replace("_", "-"), name=_occ_id,
                           which="last", train=True, epochs=740,
                           contact_method="f6_soft", provides=("base",),
                           config="anysole/configs/v1_cam3_occ010_wo05.yaml",
                           train_args={**_V7_TA, **_extra})

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
# configs/z_gen/gen.py per_action_folds --wave 12|3（波 3 带闸门断言）。
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
MODELS["V5_Av1_cam3_offfix_tphys"] = {
    **MODELS["V5_Av1_cam3_v3off"], "label": "V5-1-cam3-offfix-tphys",
    "name": "V5_Av1_cam3_offfix_tphys",
    "train_args": {**MODELS["V5_Av1_cam3_v3off"]["train_args"], "t_phys_v2": True}}
MODELS["V5_Av1_cam3_offfix_d8"] = {
    **MODELS["V5_Av1_cam3_v3off"], "label": "V5-1-cam3-offfix-d8",
    "name": "V5_Av1_cam3_offfix_d8",
    "train_args": {**MODELS["V5_Av1_cam3_v3off"]["train_args"],
                   "d8_heads": True, "lambda_tcop": 0.1,
                   "lambda_strike": 0.1, "lambda_gtcop": 0.1}}

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

# session_detail 重评（2026-10-24）：全部 35 个 tw_scan 格 dump_detail=True——
# eval 同源同公式落 session_detail_<split>.json（与 test.json 同目录），
# 供逐会话配对检验；仅影响评估输出，不动训练。
for _mid in list(MODELS):
    if "_v3off" in _mid and _mid.startswith(("V3_Ev1", "V3_Bv1", "V4_Av1", "V5_Av1", "V5_Av4")):
        if "_tw" in _mid or _mid.endswith(("cam3_v3off", "cam3_occ010_v3off")):
            MODELS[_mid] = {**MODELS[_mid], "dump_detail": True}

_WO05_CONFIG = "anysole/configs/v1_wo05.yaml"
_WO05_E1_CKPT = ("results/AnySole/V3_Ev1_wo05_f6_soft/"
                 "tw20_st20_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_ep740_bs256_sd1/"
                 "checkpoints/ckpt_best.pt")
for _suf in ("", "_drop", "_vonly", "_tonly"):
    _b = MODELS["V3_Ev1_cam3_v3off" + _suf]
    MODELS["V3_Ev1_wo05" + _suf] = {
        **_b, "label": "V3-3E-wo05" + _suf.replace("_", "-"),
        "name": "V3_Ev1_wo05" + _suf, "config": _WO05_CONFIG}
for _suf in ("", "_drop", "_vonly", "_tonly"):
    _b = MODELS["V5_Av4_cam3_v3off" + _suf]
    MODELS["V5_Av4_wo05" + _suf] = {
        **_b, "label": "V5-Av4-wo05" + _suf.replace("_", "-"),
        "name": "V5_Av4_wo05" + _suf, "config": _WO05_CONFIG}
# V5_Av1_wo05 平条目（2026-10-26）：wo05 轴调参基座（wo05 批次不含 V5_Av1；
# λ 扫描条目名带 λ 尾缀不可作基座）。基座 = V5_Av1_cam3_v3off（v3off E
# 载体、config_probs [1,0,0]），唯一差异 = config → v1_wo05.yaml。
MODELS["V5_Av1_wo05"] = {
    **MODELS["V5_Av1_cam3_v3off"], "label": "V5-1-wo05",
    "name": "V5_Av1_wo05", "config": _WO05_CONFIG}
for _suf in ("", "_drop", "_vonly", "_tonly"):
    _b = MODELS["V4_Av1_cam3_v3off" + _suf]
    MODELS["V4_Av1_wo05" + _suf] = {
        **_b, "label": "V4-1-wo05" + _suf.replace("_", "-"),
        "name": "V4_Av1_wo05" + _suf, "config": _WO05_CONFIG,
        "train_args": {**_b["train_args"], "carrier_from": _WO05_E1_CKPT}}
# wo05 E5 遮挡交叉评估（2026-10-24）：eval-only 条目——复用各家族 _drop 统一
# 模型的 ckpt（name 保持 _drop 基名；which=best → ckpt_best，对齐 E5 口径与
# canonical occ json 的 checkpoint 字段），数据根 = v1_cam3_occ{N}_wo05.yaml
# （显式 config 键）。零重训；入队闸门 = 对应家族 _drop conf 进 done/
# （configs/z_gen/gen.py wo05 --wave 3 按家族把控）。
for _f in ("V3_Ev1", "V4_Av1", "V5_Av4"):
    for _occ in ("010", "020", "030", "050"):
        _eval_id = f"{_f}_wo05_drop_evalocc{_occ}"
        _src = MODELS[f"{_f}_wo05_drop"]
        MODELS[_eval_id] = {
            **_src,
            "label": f"{_src['label']}-evalocc{_occ}",
            "train": False,
            "which": "best",
            "config": f"anysole/configs/v1_cam3_occ{_occ}_wo05.yaml",
        }


# ckpt_last 重读（2026-10-24，用户裁定）：选择假象检查——V4_Av1_wo05_drop 的
# ckpt_best 是 T 死的（val/tau0 VT 选择器在 T 的 val 边际为负时选走 T），
# ckpt_last 门控活着（cam3 面板已有）；E5 的 occ 四档与 cam3 用 ckpt_last

for _occ in ("010", "020", "030", "050"):
    _src = MODELS["V4_Av1_wo05_drop"]
    _rid = "V4_Av1_wo05_drop_last_evalocc" + _occ
    MODELS[_rid] = {
        **_src, "label": _src["label"] + "-last-evalocc" + _occ,
        "train": False, "which": "last",
        "config": "anysole/configs/v1_cam3_occ%s_wo05.yaml" % _occ}
MODELS["V4_Av1_wo05_drop_last_evalcam3"] = {
    **MODELS["V4_Av1_wo05_drop"], "label": "V4-1-wo05-drop-last-evalcam3",
    "train": False, "which": "last", "config": _WO05_CONFIG}
MODELS["V4_Av1_wo05_drop_last_evalcam3"] = {
    **MODELS["V4_Av1_wo05_drop"], "label": "V4-1-wo05-drop-last-evalcam3",
    "train": False, "which": "last", "config": _WO05_CONFIG}


# λ 扫描（2026-10-24，用户裁定「先扫描一下这几个参数点」）：V5_Av1 / V5_Av4 /
# V4_Av1 × λ_prior_pose {1,2,3,4} × λ_prior_traj {0,1} 共 24 格。cam3 训练 +
# 评估、splits_wo05.csv（v1_wo05.yaml）；纯扫描、无调参目标——每格产出三条件
# 面板（V2M/VT2M/T2M），终表由各格 test.json 汇总。
# 基座 = v3off 生产轴条目（与 wo05 批次同数据 facts）：V5_Av1_cam3_v3off /
# V5_Av4_cam3_v3off / V4_Av1_cam3_v3off，config_probs 全 [1,0,0]。载体 =
# 各基座现役 carrier（v3off E / scratch）——wo05 批次的 wo05-E 载体因 GPU
# OOM 未产出，扫描不依赖（occ 档先例：冻结载体与 split 无关）。
# conf 生成 = configs/z_gen/prior_lambda_scan.py。
# 2026-10-25 重跑：追加 multi_metric 候选机制（ckpt规则.md §3；role auto
# 按 config_probs [1,0,0] 推断 = 始终双模态专家）。
for _m in ("V5_Av1", "V5_Av4", "V4_Av1"):
    for _lp in (1.0, 2.0, 3.0, 4.0):
        for _lt in (0.0, 1.0):
            _n = "%s_wo05_lp%s_lt%s" % (_m, "%g" % _lp, "%g" % _lt)
            _b = MODELS["%s_cam3_v3off" % _m]
            MODELS[_n] = {
                **_b, "label": _n.replace("_", "-"), "name": _n,
                "config": _WO05_CONFIG,
                "train_args": {**_b["train_args"],
                               "lambda_prior_pose": _lp,
                               "lambda_prior_traj": _lt,
                               "checkpoint_policy": "multi_metric",
                               "checkpoint_role": "auto"}}

# ckpt_last 全量重定基线（2026-10-24，用户裁定）：三批 eval-only 零重训——
#   批 1：E1 生产轴 V5_Av1/V4_Av1 的 base/drop/vonly/tonly 8 格（T2M 选择污染）；
#   批 2：tw_scan 28 格（「零兑现 tw 不变式」是否选择假象——V3_Bv1/V4_Av1/
#         V5_Av1/V5_Av4 各 7 格，E 载体 7 格不在重读范围）；
#   批 3：λ 扫描 18 格（绝对值 last 复核）。
# name 保持源模型名 → ckpt 落源 run 目录的 ckpt_last.pt。
_E1_LAST_CELLS = ("V5_Av1_cam3_v3off", "V5_Av1_cam3_v3off_drop",
                  "V5_Av1_cam3_v3off_vonly", "V5_Av1_cam3_v3off_tonly",
                  "V4_Av1_cam3_v3off", "V4_Av1_cam3_v3off_drop",
                  "V4_Av1_cam3_v3off_vonly", "V4_Av1_cam3_v3off_tonly")
for _src_id in _E1_LAST_CELLS:
    MODELS[_src_id + "_last_eval"] = {
        **MODELS[_src_id], "label": MODELS[_src_id]["label"] + "-last-eval",
        "train": False, "which": "last"}
for _f in ("V3_Bv1", "V4_Av1", "V5_Av1", "V5_Av4"):
    for _tw in (5, 10, 15):
        for _ax in ("cam3", "cam3_occ010"):
            _src_id = "%s_%s_v3off_tw%d" % (_f, _ax, _tw)
            MODELS[_src_id + "_last_eval"] = {
                **MODELS[_src_id], "label": MODELS[_src_id]["label"] + "-last-eval",
                "train": False, "which": "last"}
    _src_id = "%s_cam3_occ010_v3off" % _f
    MODELS[_src_id + "_last_eval"] = {
        **MODELS[_src_id], "label": MODELS[_src_id]["label"] + "-last-eval",
        "train": False, "which": "last"}
for _m in ("V5_Av1", "V5_Av4", "V4_Av1"):
    for _lp in (1.0, 2.0, 3.0, 4.0):
        for _lt in (0.0, 1.0):
            if _m == "V5_Av4" and not (_lp == 1.0):
                continue
            _src_id = "%s_wo05_lp%s_lt%s" % (_m, "%g" % _lp, "%g" % _lt)
            MODELS[_src_id + "_last_eval"] = {
                **MODELS[_src_id], "label": MODELS[_src_id]["label"] + "-last-eval",
                "train": False, "which": "last"}


# tw × 遮挡交互（2026-10-24，z_note/multimodal_fusion/Time_Windows/时间窗扫描实验.md §9）：
# occ020/030 × tw{5,10,15,20}（v3off 轴）。问题 = V2M 行（单 V）的遮挡惩罚
# 是否随 tw 缩短增大（相位知识假说）。模型 = V3_Bv1 / V5_Av4（从零 V 流，
# 与 tw_scan occ010 腿同规格同口径）；参照 = tw_scan 的 cam3（occ0）与
# occ010 腿。occ050 用户裁定不试。dump_detail 同 tw_scan 政策。
for _occ in ("020", "030"):
    for _b3 in (MODELS["V3_Bv1"],):
        _n = f"V3_Bv1_cam3_occ{_occ}_v3off"
        MODELS[_n] = {**_b3, "label": _n.replace("_", "-"), "name": _n,
                      "train_args": {**_b3["train_args"]}, "dump_detail": True}
        for _tw in (5, 10, 15):
            _n = f"V3_Bv1_cam3_occ{_occ}_v3off_tw{_tw}"
            MODELS[_n] = {**_b3, "label": _n.replace("_", "-"), "name": _n,
                          "train_args": _tw_args(_b3["train_args"], _tw),
                          "dump_detail": True}
    _b4 = MODELS["V5_Av4_cam3_v3off"]
    _n = f"V5_Av4_cam3_occ{_occ}_v3off"
    MODELS[_n] = {**_b4, "label": _n.replace("_", "-"), "name": _n,
                  "train_args": {**_b4["train_args"]}, "dump_detail": True}
    for _tw in (5, 10, 15):
        _n = f"V5_Av4_cam3_occ{_occ}_v3off_tw{_tw}"
        MODELS[_n] = {**_b4, "label": _n.replace("_", "-"), "name": _n,
                      "train_args": _tw_args(_b4["train_args"], _tw),
                      "dump_detail": True}


# 2026-10-27: corrected contact supervision, separate run/cell namespace.
# Historical IDs remain unchanged. These IDs reuse the same registered graphs;
# the explicit contact version creates a different training result directory.
_CORRECTED_CONTACT = "f6_soft_time_accel_v1"
_CORRECTED_BASES = ["V7_Cv%d" % v for v in range(1, 8)] + ["V7_Cv5_SV1", "V7_Cv5_ST1"]
for _base_id in _CORRECTED_BASES:
    _corrected_id = _base_id + "_tafix"
    MODELS[_corrected_id] = {
        **MODELS[_base_id], "name": _base_id,
        "label": _base_id.replace("_", "-") + "-time-accel-fixed",
        "contact_method": _CORRECTED_CONTACT,
        "eval_splits": ["val"], "which": "last",
        "train_args": {**MODELS[_base_id]["train_args"], "final_eval_split": "none"},
    }
    MODELS[_corrected_id + "_statistics"] = {
        **MODELS[_corrected_id], "train": False, "which": "last"}
    for _occ in ("010", "020", "030", "050"):
        MODELS[_corrected_id + "_evalocc" + _occ] = {
            **MODELS[_corrected_id], "train": False, "which": "last",
            "config": "anysole/configs/v1_cam3_occ%s_wo05.yaml" % _occ}

# Reviewed state-machine correction. Historical f6_soft and *_tafix remain frozen.
for _base_id in _CORRECTED_BASES:
    _state_id = _base_id + "_statefix"
    MODELS[_state_id] = {
        **MODELS[_base_id], "name": _base_id,
        "label": _base_id.replace("_", "-") + "-contact-state-v2",
        "contact_method": "f6_soft_time_accel_state_v2",
        "eval_splits": ["val"], "which": "last",
        "train_args": {**MODELS[_base_id]["train_args"], "final_eval_split": "none"},
    }
    MODELS[_state_id + "_statistics"] = {**MODELS[_state_id], "train": False}
    for _occ in ("010", "020", "030", "050"):
        MODELS[_state_id + "_evalocc" + _occ] = {
            **MODELS[_state_id], "train": False,
            "config": "anysole/configs/v1_cam3_occ%s_wo05.yaml" % _occ}


# Current SMPL24 motion source on the inherited frame grid (2026-10-29).
# Historical identifiers and checkpoints keep their old package bindings.
# The current architecture screen contains seven unified models, no experts.
_SMPL24_PACKAGES = {
    "tafix_smpl24": "f6_soft_time_accel_smpl24_v1",
    "statefix_smpl24": "f6_soft_time_accel_state_smpl24_v1",
}
for _suffix, _contact in _SMPL24_PACKAGES.items():
    for _i in range(1, 8):
        _base = "V7_Cv%d" % _i
        _new = _base + "_" + _suffix
        MODELS[_new] = {
            **MODELS[_base], "name": _base,
            "label": _base + "-" + _suffix,
            "contact_method": _contact, "eval_splits": ["val", "test"], "which": "last",
            # canonical split（splits.csv，2026-10-29 用户裁定；旧线保持 wo05）
            "config": "anysole/configs/v1.yaml",
            "train_args": {**MODELS[_base]["train_args"],
                           "smpl_support_policy": "exclude", "final_eval_split": "none"},
        }
        # 训练 cell 即正式统计 cell（2026-10-29 用户裁定：_statistics 后缀作废，
        # 评估本身就是要 test——train 格 val+test 一次跑齐，test 逐帧预测落
        # predictions/eval_motion，E2/E5/E6/E7 直接读训练 cell）
        for _occ in ("010", "020", "030", "050"):
            MODELS[_new + "_evalocc" + _occ] = {
                **MODELS[_new], "train": False,
                "eval_splits": ["val", "test"],
                "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ,
            }

# ---- 种子派生条目（2026-10-29 用户裁定 --seed 1,2,3 多种子；方案 A = cell id 带 _sN）----
# seed 为 spec 顶层键（train 命令 --seed N，variant 串 sdN → ckpt 目录自动分家）。
# ---- 后缀清理（2026-10-29 用户裁定）：当前默认线占短名 V7_CvN ----
# 数据包/来源差异不进模型名（方案 §6 版本登记只有 V7_CvN）：
#   V7_CvN_statefix_smpl24 → V7_CvN（当前默认）
#   V7_CvN_tafix_smpl24   → V7_CvN_tafix
#   V7_CvN（旧 f6_soft）   → V7_CvN_f6
#   V7_CvN_statefix/_tafix（旧 BVH 来源）→ 加 _bvh23
def _apply_suffix_cleanup() -> None:
    rename = {}
    for _i in range(1, 8):
        _b = "V7_Cv%d" % _i
        rename["%s_statefix_smpl24" % _b] = _b
        rename["%s_tafix_smpl24" % _b] = "%s_tafix" % _b
        rename["%s_statefix" % _b] = "%s_statefix_bvh23" % _b
        rename["%s_tafix" % _b] = "%s_tafix_bvh23" % _b
        rename[_b] = "%s_f6" % _b
    for _old_key in list(MODELS):
        for _old_base, _new_base in rename.items():
            if _old_key == _old_base or _old_key.startswith(_old_base + "_"):
                _new_key = _old_key.replace(_old_base, _new_base, 1)
                if _new_key not in MODELS:
                    MODELS[_new_key] = MODELS.pop(_old_key)
                break


_apply_suffix_cleanup()


_SEED_BASES = ["V7_Cv%d" % i for i in range(1, 8)] + ["V7_Cv%d_tafix" % i for i in range(1, 8)]
for _base in _SEED_BASES:
    for _n in (1, 2, 3):
        MODELS["%s_s%d" % (_base, _n)] = {
            **MODELS[_base],
            "label": "%s-s%d" % (MODELS[_base]["label"], _n),
            "name": MODELS[_base]["name"], "seed": _n}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalocc%s" % (_base, _n, _occ)] = {
                **MODELS["%s_evalocc%s" % (_base, _occ)],
                "label": "%s-s%d-evalocc%s" % (MODELS[_base]["label"], _n, _occ),
                "name": MODELS[_base]["name"], "seed": _n}


# ---- best 口径评估条目（2026-10-29 用户裁定 --ckpt last/best；Cv1–7）----
# eval-only：which=best、name=基名；cam3 + occ 四档。
_CKPT_BASES = ["V7_Cv%d" % i for i in range(1, 8)]
for _base in _CKPT_BASES:
    MODELS["%s_evalbest" % _base] = {
        **MODELS[_base], "label": MODELS[_base]["label"] + "-evalbest",
        "name": MODELS[_base]["name"], "train": False, "which": "best",
        "eval_splits": ["val", "test"]}
    for _occ in ("010", "020", "030", "050"):
        MODELS["%s_evalbest_occ%s" % (_base, _occ)] = {
            **MODELS[_base], "label": "%s-evalbest-occ%s" % (MODELS[_base]["label"], _occ),
            "name": MODELS[_base]["name"], "train": False, "which": "best",
            "eval_splits": ["val", "test"],
            "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}
    # 种子派生（--ckpt best × --seed N）
    for _n in (1, 2, 3):
        MODELS["%s_s%d_evalbest" % (_base, _n)] = {
            **MODELS["%s_evalbest" % _base],
            "label": "%s-s%d-evalbest" % (MODELS[_base]["label"], _n),
            "name": MODELS[_base]["name"], "seed": _n}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalbest_occ%s" % (_base, _n, _occ)] = {
                **MODELS["%s_evalbest_occ%s" % (_base, _occ)],
                "label": "%s-s%d-evalbest-occ%s" % (MODELS[_base]["label"], _n, _occ),
                "name": MODELS[_base]["name"], "seed": _n}

# ---- 历史遗留清理（2026-10-29 用户裁定：不用了的分支删除，不重建）----
# wo05 实验轴（旧 splits 已废弃）、T 增强三组、_f6 旧标签线、
# _tafix_bvh23/_statefix_bvh23 旧 BVH 修正线。
def _drop_legacy_entries() -> None:
    drop = set()
    for _key in MODELS:
        if ("_wo05" in _key or "wo05" in _key or "offfix" in _key
                or "_f6" in _key or "_tafix_bvh23" in _key
                or "_statefix_bvh23" in _key):
            drop.add(_key)
    for _key in sorted(drop):
        MODELS.pop(_key, None)
    print("dropped legacy entries:", len(drop))


_drop_legacy_entries()

# ---- 调参格（2026-10-29 重建于当前线）：V7_Cv1 × lr × λ_contact × schedule ----
for _lrk in ("5e-05", "0.0001", "0.0002"):
    for _lck in ("0.1", "0.3", "1"):
        for _csk in ("const", "cos"):
            _tid = "V7_Cv1_lr%s_lc%s_%s" % (_lrk, _lck, _csk)
            MODELS[_tid] = {
                **MODELS["V7_Cv1"], "label": _tid.replace("_", "-"),
                "name": _tid,
                "train_args": {**MODELS["V7_Cv1"]["train_args"],
                               "lr": float(_lrk), "lambda_contact": float(_lck),
                               "lr_schedule": _csk}}

# ---- 第二轮清理（2026-10-29 用户裁定「一并清」）：白名单保留当前线 ----
# 保留 = V7_Cv1..7 短名/tafix 参照线 × seed{1,2,3} × 统计/遮挡/best 派生
# + 调参 18 格；其余全部删除（V7_Av1/Bv1 候选、V3/V4/V5/F 旧家族、
# tw/λ/e1 等历史线）。
def _drop_all_but_current() -> None:
    keep = set()
    for _i in range(1, 8):
        for _base in ("V7_Cv%d" % _i, "V7_Cv%d_tafix" % _i):
            # seed 1 显式 _s1（2026-10-29 用户裁定）；裸名行已无实验发射
            # （训练 cell 即评估 cell），不进白名单
            for _sd in ("_s1", "_s2", "_s3"):
                keep.add(_base + _sd)
                keep.add(_base + _sd + "_evalbest")
                for _o in ("010", "020", "030", "050"):
                    keep.add(_base + _sd + "_evalocc" + _o)
                    keep.add(_base + _sd + "_evalbest_occ" + _o)
    for _lr in ("5e-05", "0.0001", "0.0002"):
        for _lc in ("0.1", "0.3", "1"):
            for _cs in ("const", "cos"):
                keep.add("V7_Cv1_lr%s_lc%s_%s" % (_lr, _lc, _cs))
    drop = [k for k in MODELS if k not in keep]
    for k in drop:
        MODELS.pop(k, None)
    print("dropped second-round entries:", len(drop), "| kept:", len(MODELS))


_drop_all_but_current()

# V7 C 门控修改方案（2026-10-30，z_note/multimodal_fusion/V7_方案/
# V7_C门控修改方案.md §3/§8）：T（逐帧双门）/ LR（窗口四组门）/ TLR
# （逐帧四组门）三实验组。训练配方与 V7_Cv5 当前线逐项一致（contact =
# f6_soft_time_accel_state_smpl24_v1、v1.yaml canonical split、三条件
# 枚举、ckpt_last 口径、740ep；B 基线 = 现有 V7_Cv5_s1 ckpt_last——
# 生产代码自其训练后未变，不补跑）；结构差异由入口 STRUCTURE 携带
# （v7_route_temporal / v7_route_parts）。seed 2/3 按方案 §8.4 于 val
# 确认后补发（条目先注册不发射）；Q/S 按 §8.3 于 TLR 完成后另立。
_V7_CV5_S1_CELL = MODELS["V7_Cv5_s1"]
for _gate in ("V7_Cv5T", "V7_Cv5LR", "V7_Cv5TLR"):
    for _n in (1, 2, 3):
        MODELS["%s_s%d" % (_gate, _n)] = {
            **_V7_CV5_S1_CELL,
            "label": "%s-s%d" % (_gate.replace("_", "-"), _n),
            "name": _gate, "seed": _n}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalocc%s" % (_gate, _n, _occ)] = {
                **_V7_CV5_S1_CELL,
                "label": "%s-s%d-evalocc%s" % (_gate.replace("_", "-"), _n, _occ),
                "name": _gate, "seed": _n, "train": False,
                "eval_splits": ["val", "test"],
                "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}
