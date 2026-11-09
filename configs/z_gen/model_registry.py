"""Model registry and shared defaults for the offline experiments.

2026-09-24 independence restructure: structure flags (t_encoder / f2_repr /
pose_parts / soft_parts / gate) are no longer train CLI options — each
model's structure is fixed by its entry script (anysole/models/<name>.py),
and every run trains from random init.  ``train_args`` carries only runtime
hyperparameters.  ``part_json`` (V4B structural input) and
``assign_cluster`` (V4A train protocol) stay.
"""

from __future__ import annotations

# E5 遮挡十档（2026-11-02 用户裁定：occ010–100 步长 10，occ100 = 视觉全无）
OCC_LEVELS_10 = ("010", "020", "030", "040", "050", "060", "070", "080", "090", "100")

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
        for _occ in OCC_LEVELS_10:
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
        for _occ in OCC_LEVELS_10:
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
    for _occ in OCC_LEVELS_10:
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
                for _o in OCC_LEVELS_10:
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
# （逐帧四组门）三实验组；Q（§6，TLR 完成后的条件输入组，2026-10-31）。
# 训练配方与 V7_Cv5 当前线逐项一致（contact =
# f6_soft_time_accel_state_smpl24_v1、v1.yaml canonical split、三条件
# 枚举、ckpt_last 口径、740ep；B 基线 = 现有 V7_Cv5_s1 ckpt_last——
# 生产代码自其训练后未变，不补跑）；结构差异由入口 STRUCTURE 携带
# （v7_route_temporal / v7_route_parts / v7_route_conditional）。
# seed 2/3 按方案 §8.4 于 val 确认后补发（条目先注册不发射）；
# S 按 §7 硬门槛通过后另立。
_V7_CV5_S1_CELL = MODELS["V7_Cv5_s1"]
# V7_Cv5Ctx / V7_Cv5Ctx2（2026-11-01）：Cv5 + 上一窗姿态上下文续写
# （续写协议方案 §1-§4）。配方与 V7_Cv5_s1 逐项一致；ctx 训练 = teacher
# forcing（GT 窗首帧根状态），评估 = stride tw//2 重叠链式（上一窗预测）。
# Ctx2 = 续写 v2：ctx 直连根朝向（部位 query 注入 + 输出残差，方案 §9）。
for _gate in ("V7_Cv5Ctx", "V7_Cv5Ctx2"):
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
# 续写 v2 实验矩阵（2026-11-01，G1-G4）：G1 = scheduled sampling
# （train_args ctx_ss=0.5，训练时线性增至 0.5 概率用模型预测 ctx）；
# G2 = V 在场条件门控（入口 v7_c2ctx_vc）；G3 = 评估硬前缀（入口
# v7_c2ctx_hp）；G4 = 三合一（v7_c2ctx_full + ctx_ss）。
for _gate, _ta in (("V7_Cv5CtxSS", {"ctx_ss": 0.5}),
                   ("V7_Cv5CtxVC", {}),
                   ("V7_Cv5CtxHP", {}),
                   ("V7_Cv5CtxFull", {"ctx_ss": 0.5})):
    for _n in (1, 2, 3):
        MODELS["%s_s%d" % (_gate, _n)] = {
            **_V7_CV5_S1_CELL,
            "label": "%s-s%d" % (_gate.replace("_", "-"), _n),
            "name": _gate, "seed": _n,
            "train_args": {**_V7_CV5_S1_CELL["train_args"], **_ta}}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalocc%s" % (_gate, _n, _occ)] = {
                **_V7_CV5_S1_CELL,
                "label": "%s-s%d-evalocc%s" % (_gate.replace("_", "-"), _n, _occ),
                "name": _gate, "seed": _n, "train": False,
                "eval_splits": ["val", "test"],
                "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}

# v2 训练 GPU6 直发（2026-11-01 用户指定）：s1 条目注入 device=cuda:6；
# s2/s3 不设（按常规调度器分配）。
# 续写 v3（2026-11-01，V7_续写协议方案 §10）：T-only 硬填式重叠
# （ctx = 重叠区 nf 帧 × 10 维，替换式注入前 nf 帧根 query 槽位；V 在场
# 门控不注入 → V/VT 行回归 Cv5 行为）。入口 v7_c2ctx3，配方同 Cv5。
for _gate in ("V7_Cv5Ctx3",):
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

# 续写 v4（2026-11-01，V7_续写v4残差分解方案）：ctx3 + 锚点系残差分解
# （T 行目标/注入/出口全残差化，R_a 链式携带；V/VT 行 R_a=I 回归 Cv5）。
# 入口 v7_c2ctx4，配方同 Cv5。
for _gate in ("V7_Cv5Ctx4",):
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

# 续写 v5（2026-11-01）：块自回归——ctx 块扩全姿态（nf × 148），全部
# query 槽位逐部位硬填（发射区从完整历史出发）。入口 v7_c2ctx5，配方同 Cv5。
for _gate in ("V7_Cv5Ctx5",):
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

# 帧级双模式 AR（2026-11-01，V7 方案 §11）：V/VT 回归 + T 因果自回归，
# 共享网络。入口 v7_c2ar，配方同 Cv5。
# 续写 v4-b（2026-11-01，v4 方案 §10）：统一残差空间——全条件目标 =
# 锚点系残差（锚 = GT 窗首帧，首窗绝对）；V/VT 行锚 = 锚估计头
# （λ_anchor=1.0，CONFIG_EXTRA），T 行锚 = ctx 链。入口 v7_c2ctx4b。
# 续写 v4-R（2026-11-01）：ctx4 + 接触根运动（B 候选挂链，λ_foot=1.0，
# CONFIG_EXTRA）。入口 v7_c2ctx4r。
# AR 对照格（2026-11-01）：因果掩码、无反馈支路。入口 v7_c2arm。
for _gate in ("V7_Cv5Ctx4B", "V7_Cv5Ctx4R", "V7_Cv5ARM"):
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

# 帧级双模式 AR（2026-11-01，V7 方案 §11）：V/VT 回归 + T 因果自回归，
# 共享网络。入口 v7_c2ar，配方同 Cv5。
# ARSS = AR + 反馈 scheduled sampling（ar_ss 线性升至 0.3）+ 反馈收缩噪声
# （ar_noise=0.1，只扰姿态分量）——2026-11-01 裁定：暴露偏差与 v1 ctx 同构，
# G1 机制平移到 prev_state。
MODELS["V7_Cv5ARSS_s1"] = {
    **_V7_CV5_S1_CELL, "label": "V7-Cv5ARSS-s1",
    "name": "V7_Cv5ARSS", "seed": 1,
    "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                   "ar_ss": 0.3, "ar_noise": 0.1}}

# ARSS2 = 结果目录绕过的同名临时注册（2026-11-02，见 anysole/registry.py 注释）
MODELS["V7_Cv5ARSS2_s1"] = {
    **_V7_CV5_S1_CELL, "label": "V7-Cv5ARSS2-s1",
    "name": "V7_Cv5ARSS2", "seed": 1,
    "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                   "ar_ss": 0.3, "ar_noise": 0.1}}
for _gate in ("V7_Cv5AR",):
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

# V7_Cv5AR2（2026-11-01，V7_AR相对运动与航向状态方案）：Cv5 + AR 相对
# 运动（Δyaw/tilt/body/v_head）+ 航向状态分离 + 视觉航向观测校正。
# 入口 v7_c2ar2，配方同 Cv5。AR2 损失默认权重随 train_args 发射：
# λ_dyaw/λ_cumyaw = 1.0（mrad 口径）、λ_dyaw_td/λ_pose_vel/λ_visyaw/
# λ_yawcorr = 0.5（v1.yaml 的 lambda_pose_vel=0.0 会覆盖函数默认，须显式带出）。
for _gate in ("V7_Cv5AR2",):
    for _n in (1, 2, 3):
        MODELS["%s_s%d" % (_gate, _n)] = {
            **_V7_CV5_S1_CELL,
            "label": "%s-s%d" % (_gate.replace("_", "-"), _n),
            "name": _gate, "seed": _n,
            "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                           "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                           "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                           "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5}}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalocc%s" % (_gate, _n, _occ)] = {
                **_V7_CV5_S1_CELL,
                "label": "%s-s%d-evalocc%s" % (_gate.replace("_", "-"), _n, _occ),
                "name": _gate, "seed": _n, "train": False,
                "eval_splits": ["val", "test"],
                "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}

# Strict context matrix: observed 20/40/80, output/stride 20, batch 64.
# Keep the existing *_tw40/*_tw80 cell IDs; they now label observation length.
# Existing AR2_s1-3 runs remain separate references with batch 256.
for _gate, _context in (("V7_Cv5AR2M", 20), ("V7_Cv5AR2M", 40),
                        ("V7_Cv5AR2M", 80), ("V7_Cv5AR2", 20),
                        ("V7_Cv5AR2", 40), ("V7_Cv5AR2", 80)):
    _suf = ("_ctx20" if _gate == "V7_Cv5AR2" and _context == 20
            else "" if _context == 20 else "_tw%d" % _context)
    MODELS["%s%s_s1" % (_gate, _suf)] = {
        **_V7_CV5_S1_CELL,
        "label": "%s-obs%d-out20-s1" % (_gate.replace("_", "-"), _context),
        "name": _gate, "seed": 1, "batch_size": 64,
        "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                       "tw": 20, "stride": 20, "obs_context": _context,
                       "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                       "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                       "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5}}

MODELS["V7_Cv5Ctx2_s1"]["device"] = "cuda:6"

for _gate in ("V7_Cv5T", "V7_Cv5LR", "V7_Cv5TLR", "V7_Cv5Q"):
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

# S 显式收益监督（2026-11-01，V7_C 门控修改方案 §7）：TLR 结构（v7_c2tlr）
# + l_s_gate BCE（四组门 vel/other/left/right 对离线相对收益标签，§7 硬门槛
# 已全过 S_ok=true）。λ_s 扫 1.0 / 0.1（entry V7_Cv5S / V7_Cv5S01）；标签目录
# 固定 results_display/Etest/S_GateDiag（s_gate_diag.py 产出，τ 固定，组序
# vel/other/left/right）。种子 1/2/3 注册；发射集按卡位裁定（见发射脚本）。
_S_LABEL_DIR = "/data/fangyuxuan/projects/gait/results_display/Etest/S_GateDiag"
for _gate, _ls in (("V7_Cv5S", 1.0), ("V7_Cv5S01", 0.1)):
    for _n in (1, 2, 3):
        MODELS["%s_s%d" % (_gate, _n)] = {
            **_V7_CV5_S1_CELL,
            "label": "%s-s%d" % (_gate.replace("_", "-"), _n),
            "name": _gate, "seed": _n,
            "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                           "lambda_s": _ls, "v7_s_label": _S_LABEL_DIR}}
        for _occ in ("010", "020", "030", "050"):
            MODELS["%s_s%d_evalocc%s" % (_gate, _n, _occ)] = {
                **_V7_CV5_S1_CELL,
                "label": "%s-s%d-evalocc%s" % (_gate.replace("_", "-"), _n, _occ),
                "name": _gate, "seed": _n, "train": False,
                "eval_splits": ["val", "test"],
                "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}

# V3_Bv1（V3-3B）重注册（2026-10-31 用户要求「只想要运行 V3B」）：
# 架构入口 = anysole MODEL_REGISTRY["V3_Bv1"] → module v3_3b（未删）；
# 旧 BVH-23 ckpt 已不存在（results/AnySole 无 V3_Bv1* 目录），新轴
# （v1.yaml SMPL24 + 当前接触包）从零重训。种子 1/2/3 注册。
_V3B_TA = {"lambda_assign": 0.05, "lambda_sigma": 0.01,
           "sigma_freeze_frac": 0.1, "lr_warmup_frac": 0.05,
           "grad_clip": 5.0, "final_eval_split": "none",
           "smpl_support_policy": "exclude",
           "checkpoint_policy": "multi_metric", "checkpoint_role": "auto"}
for _n in (1, 2, 3):
    MODELS["V3_Bv1_s%d" % _n] = dict(
        label="V3-3B-s%d" % _n, name="V3_Bv1", which="last", train=True,
        epochs=740, seed=_n,
        contact_method="f6_soft_time_accel_state_smpl24_v1",
        provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
        config="anysole/configs/v1.yaml",
        eval_splits=["val", "test"],
        train_args=_V3B_TA)

# 阶段 3 触觉泛化（2026-10-31，E6 门控重设计）：以 V7_Cv7（无门 v7_a 入口）
# 为基座，训练侧单点数据增强。配方与 V7_Cv7_s1 逐项一致（contact =
# f6_soft_time_accel_state_smpl24_v1、v1.yaml canonical split、三条件枚举、
# ckpt_last、740ep），唯一差异是 train_args 里的增强开关：
#   G-b = tactile_gainbias（鞋垫增益偏置，每窗每足 gain[0.7,1.3]+bias±5%）
#   G-c = tactile_frame_dropout 0.2（整帧 T token 置零）
# 结构不变 → 共用 v7_a 入口；新模型名分家到独立结果目录。
_V7_CV7_S1_CELL = MODELS["V7_Cv7_s1"]
for _name, _aug in (("V7_Cv7_gb", {"tactile_gainbias": True}),
                    ("V7_Cv7_gc", {"tactile_frame_dropout": 0.2})):
    MODELS["%s_s1" % _name] = {
        **_V7_CV7_S1_CELL,
        "label": "%s-s1" % _name.replace("_", "-"),
        "name": _name,
        "train_args": {**_V7_CV7_S1_CELL["train_args"], **_aug}}
    for _occ in ("010", "020", "030", "050"):
        MODELS["%s_s1_evalocc%s" % (_name, _occ)] = {
            **_V7_CV7_S1_CELL,
            "label": "%s-s1-evalocc%s" % (_name.replace("_", "-"), _occ),
            "name": _name, "seed": 1, "train": False,
            "eval_splits": ["val", "test"],
            "config": "anysole/configs/v1_cam3_occ%s.yaml" % _occ}

# 阶段 2 受试者交叉泛化（2026-10-31，E6 门控重设计配套）：leave-subjects-out
# 4 折，复用 2026-10-22 的 5-fold 动作覆盖轮换 split（splits_test_*.csv，
# 已恢复到 protocol/splits/default/）。基座 = V7_Cv7_s1（无门 v7_a 入口、
# contact f6_soft_time_accel_state_smpl24_v1、三条件枚举、ckpt_last、740ep）。
# 差异：config → 每 fold 专属 yaml（split_csv 指向该 fold 文件，val/test
# 口径随 fold 改变）；name → V7_Cv7_f<testSubjects> 分家到独立结果目录，
# 避免与 V7_Cv7_s1 主模型 ckpt 冲突。eval_splits 含 test（fold-out 受试者
# 的泛化读数）。
_V7_CV7_FOLDS = (
    ("V7_Cv7_fS5S6S11",  "anysole/configs/v1_fold_S5S6S11.yaml"),
    ("V7_Cv7_fS5S11S12", "anysole/configs/v1_fold_S5S11S12.yaml"),
    ("V7_Cv7_fS6S7S13",  "anysole/configs/v1_fold_S6S7S13.yaml"),
    ("V7_Cv7_fS5S6S8",   "anysole/configs/v1_fold_S5S6S8.yaml"),
)
for _fname, _fconfig in _V7_CV7_FOLDS:
    MODELS["%s_s1" % _fname] = {
        **_V7_CV7_S1_CELL,
        "label": "%s-s1" % _fname.replace("_", "-"),
        "name": _fname,
        "config": _fconfig,
        "eval_splits": ["val", "test"]}

# V7_Cv5_bs512（2026-11-01 训练提速实验）：同 V7_Cv5_s1 配方、batch 256→512。
# 确定性步时基准：1.53s→1.88s/步（1.23×），步数 7→4/epoch，训练墙钟省 ~30%。
# batch_size 走 spec 顶层（runner.py _train_command 直读），不进 train_args。
MODELS["V7_Cv5_bs512_s1"] = {**_V7_CV5_S1_CELL, "label": "V7-Cv5-bs512-s1",
                             "name": "V7_Cv5_bs512", "seed": 1,
                             "batch_size": 512}

# V7_Cv5AR2MF（2026-11-02，AR2 yaw-fix 损失）：结构同 V7_Cv5AR2M（因果掩码
# 无反馈，v7_c2ar2m），配方差异只在损失（v7_ar2_yaw_fix=True）：
#   ① T 行航向监督移除（probe_dyaw_observability_20261101 §4 执行；
#      转向只从 V/VT 行学）；
#   ② L_yawcorr 自由生成口径（ψ 从 0 起 + 推理同款 EMA 校正链——旧口径
#      从 GT psi_base 起，gain 最优解 = 0，视觉校正从未学会启用）；
#   ③ L_world_traj 世界系轨迹（V/VT 行，组装后位移多尺度比对）。
# 对照 = V7_Cv5AR2M_s1（同 obs20/batch64/种子，仅损失配方差异）。
MODELS["V7_Cv5AR2MF_s1"] = {
    **_V7_CV5_S1_CELL,
    "label": "V7-Cv5AR2MF-obs20-out20-s1",
    "name": "V7_Cv5AR2MF", "seed": 1, "batch_size": 64,
    "train_args": {**_V7_CV5_S1_CELL.get("train_args", {}),
                   "tw": 20, "stride": 20, "obs_context": 20,
                   "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                   "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                   "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5,
                   "v7_ar2_yaw_fix": True, "lambda_world_traj": 1.0,
                   "ar2_dyaw_base": "history", "ar2_dyaw_t_w": 0.1}}

# V7_Cv5AR2M_s1_evalgfl（2026-11-02，AR2 视觉校正增益下限诊断）：
# 冻结 V7_Cv5AR2M_s1 ckpt_last，评估侧 g←clamp(g, min=0.5) 重评——
# 量化「视觉校正若真生效」的 drift/PA 上限（零重训）。gain 下限经 eval
# yaml 单键 ar2_vis_gain_floor 传入（run_protocol / 导出同键同默认）。
MODELS["V7_Cv5AR2M_s1_evalgfl"] = {
    **_V7_CV5_S1_CELL,
    "label": "V7-Cv5AR2M-s1-gainfloor0.5",
    "name": "V7_Cv5AR2M", "seed": 1, "train": False,
    # 口径必须与 V7_Cv5AR2M_s1（obs20 格）逐项一致：epochs/batch 参与
    # variant 目录派生（effective_variant），错位即找不到 ckpt。
    "epochs": 740, "batch_size": 64,
    "eval_splits": ["val", "test"],
    "config": "anysole/configs/v1_ar2_gainfloor.yaml"}

# V7_Cv5AR2MF_gb / V7_Cv5_gb（2026-11-02）：触觉增益偏置增强（G-b）上线。
# 证据：V7_Cv7_gb（无门+gb）T2M 83.6 vs V7_Cv7 97.2（−13.6mm，ckpt_last
# 740ep 完整训练）；LSO 裁定杠杆 = 触觉表征/跨受试者泛化，非门控。
# AR2MF_gb = yaw-fix（00271 机制修复）× gb（表征增强）合体，对照
# V7_Cv5AR2MF_s1；Cv5_gb = 回归骨架 + gb，隔离增强收益是否与 AR2 骨架
# 正交（对照 V7_Cv5_s1 89.4 与 V7_Cv7_gb 83.6）。
for _base, _name in (("V7_Cv5AR2MF_s1", "V7_Cv5AR2MF_gb"),
                     ("V7_Cv5_s1", "V7_Cv5_gb")):
    MODELS["%s_s1" % _name] = {
        **MODELS[_base],
        "label": "%s-s1" % _name.replace("_", "-"),
        "name": _name,
        "train_args": {**MODELS[_base].get("train_args", {}),
                       "tactile_gainbias": True}}

# 2026-11-02 补：V7_Cv5AR2R_s1（00264 因 z_gen 缺条目 KeyError 两秒失败——
# 最小机制格：Cv5 双门回归 + AR2 相对读出，双向注意力无掩码无反馈，
# 每窗并行前向 + 外部航向状态跨窗积分）；V7_Cv5AR2M_gb_s1（AR2M + G-b，
# oracle 探针裁定航向非 T2M PA 瓶颈后，表征增强直接上稳定性已证骨架）。
MODELS["V7_Cv5AR2R_s1"] = {
    **MODELS["V7_Cv5_s1"],
    "label": "V7-Cv5AR2R-s1", "name": "V7_Cv5AR2R", "seed": 1,
    "train_args": {**MODELS["V7_Cv5_s1"].get("train_args", {}),
                   "lambda_dyaw": 1.0, "lambda_cumyaw": 1.0,
                   "lambda_dyaw_td": 0.5, "lambda_pose_vel": 0.5,
                   "lambda_visyaw": 0.5, "lambda_yawcorr": 0.5}}
# 2026-11-02 论文模型补格：AR2RT（AR2R + VT 零初始化触觉残差）、
# AR2R_tw40/80（长观察窗轴，obs 40/80 输出 20 步长 20 batch 64）、
# AR2R_wn（T_raw 会话体重代理归一，LSO 泛化杠杆）。
MODELS["V7_Cv5AR2RT_s1"] = {
    **MODELS["V7_Cv5AR2R_s1"],
    "label": "V7-Cv5AR2RT-s1", "name": "V7_Cv5AR2RT", "seed": 1}
for _suf, _context in (("_tw40", 40), ("_tw80", 80)):
    MODELS["V7_Cv5AR2R%s_s1" % _suf] = {
        **MODELS["V7_Cv5AR2R_s1"],
        "label": "V7-Cv5AR2R-obs%d-out20-s1" % _context,
        "name": "V7_Cv5AR2R", "seed": 1, "batch_size": 64,
        "train_args": {**MODELS["V7_Cv5AR2R_s1"].get("train_args", {}),
                       "tw": 20, "stride": 20, "obs_context": _context}}
MODELS["V7_Cv5AR2R_wn_s1"] = {
    **MODELS["V7_Cv5AR2R_s1"],
    "label": "V7-Cv5AR2R-wn-s1", "name": "V7_Cv5AR2R", "seed": 1,
    "train_args": {**MODELS["V7_Cv5AR2R_s1"].get("train_args", {}),
                   "tactile_weight_norm": True}}

MODELS["V7_Cv5AR2M_gb_s1"] = {
    **MODELS["V7_Cv5AR2M_s1"],
    "label": "V7-Cv5AR2M-gb-s1", "name": "V7_Cv5AR2M_gb", "seed": 1,
    "train_args": {**MODELS["V7_Cv5AR2M_s1"].get("train_args", {}),
                   "tactile_gainbias": True}}

# 修改 04（2026-11-03，z_note/修改/04_AR训练目标与自由生成对齐.md）：
# R = AR2 结构 + 多步自由 rollout 损失（改动 A，train_args ar_rollout 族）；
# RD = R + 低维状态反馈（改动 B，v7_ar2_lowdim + v7_ar2_fb_dim）。
# 基线 = V7_Cv5AR2M_s1（obs20/batch64 格，不重跑）；对照 = V7_Cv5AR2_ctx20_s1
# （AR2 纯 TF，隔离 rollout 损失单独贡献——04 开工裁定：AR2M 无反馈支路
# TF≡rollout，改动 A 必须伴随反馈支路开启，故 R/RD 结构基座 = AR2）。
# 配方继承 V7_Cv5AR2M_s1（obs20 格：tw20/stride20/obs_context20/batch64 +
# AR2 损失权重），唯一差异 = 下方 train_args 追加键。注册但未授权发射
# （conf 未入队，待用户核验——见 z_note/修改/04_核验清单.md）。
_RL_TA = {**MODELS["V7_Cv5AR2M_s1"].get("train_args", {}),
          "ar_rollout": True, "ar_rollout_max": 8,
          "ar_rollout_schedule_frac": 0.6, "ar_rollout_detach": True}
MODELS["V7_Cv5AR2RL_s1"] = {
    **MODELS["V7_Cv5AR2M_s1"],
    "label": "V7-Cv5AR2RL-s1", "name": "V7_Cv5AR2RL", "seed": 1,
    "train_args": dict(_RL_TA)}
MODELS["V7_Cv5AR2RLD_s1"] = {
    **MODELS["V7_Cv5AR2M_s1"],
    "label": "V7-Cv5AR2RLD-s1", "name": "V7_Cv5AR2RLD", "seed": 1,
    "train_args": {**_RL_TA, "v7_ar2_lowdim": True, "v7_ar2_fb_dim": 16}}

# 继承路线（2026-11-03，z_note/multimodal_fusion/V7_方案/V7_继承路线实验定义.md）：
# 结构 = AR2M（因果掩码无反馈）；配方继承 V7_Cv5AR2M_tw80_s1（tw20/stride20/obs80/batch64 +
# AR2 损失权重），唯一差异 = t_prefix_p：enum 的 T 档里 30% 有历史的样本改成「历史窗口 V 真实、
# 当前窗 V 为 null、T 全程真实」。对照 = V7_Cv5AR2M_tw80_s1（不重跑）。从头训练，不走 --init-from。
MODELS["V7_Cv5AR2MP_tw80_s1"] = {
    **MODELS["V7_Cv5AR2M_tw80_s1"],
    "label": "V7-Cv5AR2MP-obs80-out20-s1",
    "name": "V7_Cv5AR2MP", "seed": 1,
    "train_args": {**MODELS["V7_Cv5AR2M_tw80_s1"].get("train_args", {}),
                   "t_prefix_p": 0.3}}

# 按模态分窗（2026-11-03）：V 只看当前窗（20 帧），T 吃长历史（80/160/320）。
# 结构 = AR2M + v7_vcur（V 流在模型内裁到当前窗，T 信任门同步只池化当前窗）；
# 配方继承 V7_Cv5AR2M_tw80_s1（tw20/stride20/batch64 + AR2 损失权重），唯一差异 =
# 结构标志 + obs_context 取值。对照 = V7_Cv5AR2M_tw80_s1（V80/T80，不重跑）。
for _context in (80, 160, 320):
    MODELS["V7_Cv5AR2MSC_tw%d_s1" % _context] = {
        **MODELS["V7_Cv5AR2M_tw80_s1"],
        "label": "V7-Cv5AR2MSC-v20-t%d-out20-s1" % _context,
        "name": "V7_Cv5AR2MSC", "seed": 1,
        "train_args": {**MODELS["V7_Cv5AR2M_tw80_s1"].get("train_args", {}),
                       "obs_context": _context}}

# Cv5 按模态分窗（2026-11-03）：V 只看当前窗，T 吃长历史（80 帧），非 AR2。
# 结构 = V7_Cv5 + v7_vcur（v7_c2sc 入口）；配方继承 V7_Cv5_s1（绝对目标、枚举三条件、740ep），
# 唯一差异 = obs_context=80 + 结构标志。对照 = V7_Cv5_s1（V20/T20），用于回答 T 长窗在非 AR2 架构里能否兑现。
MODELS["V7_Cv5SC_tw80_s1"] = {
    **MODELS["V7_Cv5_s1"],
    "label": "V7-Cv5SC-v20-t80-s1",
    "name": "V7_Cv5SC", "seed": 1,
    "train_args": {**MODELS["V7_Cv5_s1"].get("train_args", {}),
                   "obs_context": 80}}

# 根头速率/方向解耦（2026-11-04，修改 07 → z_note/修改/07_根头速率方向解耦与
# T行方向掩码.md）：结构 = V7_Cv5（v7_c2 双门、枚举三条件、非 AR2）+ v7_root_sd
# （入口 v7_c2sd）。判据列 = 改动 C 的 WinDispMag / WinDispAngle；新 split 口径
# （_ACT8020，seed=101）与 V7_Cv5_act8020_s1 / V7_Cv7_act8020_s1 同配方可比。
# 基座用 _s1（裸名 V7_Cv5 已在第二轮白名单清理中删除）；seed 从裸条目移除，
# 与既有 V7_Cv5/_s1 的「裸名 = 结构基座、_sN = 训练格」约定一致。
MODELS["V7_Cv5SD"] = {**MODELS["V7_Cv5_s1"], "label": "V7-Cv5SD", "name": "V7_Cv5SD"}
MODELS["V7_Cv5SD"].pop("seed", None)
MODELS["V7_Cv5SD_precheck"] = {**MODELS["V7_Cv5SD"], "label": "V7-Cv5SD-precheck",
                               "name": "V7_Cv5SD_precheck", "epochs": 6,
                               "train_args": {**MODELS["V7_Cv5SD"]["train_args"],
                                              "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5SD_s1"] = {**MODELS["V7_Cv5SD"], "label": "V7-Cv5SD-s1", "seed": 1}

# 第二臂 B′（2026-11-04，修改 07「改动 B 的配平形式」）：结构逐参数 =
# V7_Cv5SD（入口 v7_c2sd2），唯一差异 = 损失恒等式配平（全 1.0 且不掩码
# 时 L_speed+L_vy+L_dir ≡ Cv5 的 l_vel；T 行方向掩码为唯一干预）。
# 对照 = V7_Cv5SD_act8020_s1（默认权重臂）与 V7_Cv5_act8020_s1。
MODELS["V7_Cv5SD2"] = {**MODELS["V7_Cv5SD"], "label": "V7-Cv5SD2", "name": "V7_Cv5SD2"}
MODELS["V7_Cv5SD2_precheck"] = {**MODELS["V7_Cv5SD2"], "label": "V7-Cv5SD2-precheck",
                                "name": "V7_Cv5SD2_precheck", "epochs": 6,
                                "train_args": {**MODELS["V7_Cv5SD2"]["train_args"],
                                               "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5SD2_s1"] = {**MODELS["V7_Cv5SD2"], "label": "V7-Cv5SD2-s1", "seed": 1}

# 修改 10 §三 可直接移植 ①（2026-11-05）：SD2 + VT 零初始化触觉残差
# （入口 v7_c2sd2rt）。控制臂 = V7_Cv5SD2_act8020_s1（同结构、仅差
# vt_t_residual）；判据 = 无遮挡档 Δ(VT−V) 符号 + occ010–100 曲线。
MODELS["V7_Cv5SD2RT"] = {**MODELS["V7_Cv5SD2"], "label": "V7-Cv5SD2RT", "name": "V7_Cv5SD2RT"}
MODELS["V7_Cv5SD2RT_precheck"] = {**MODELS["V7_Cv5SD2RT"], "label": "V7-Cv5SD2RT-precheck",
                                  "name": "V7_Cv5SD2RT_precheck", "epochs": 6,
                                  "train_args": {**MODELS["V7_Cv5SD2RT"]["train_args"],
                                                 "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5SD2RT_s1"] = {**MODELS["V7_Cv5SD2RT"], "label": "V7-Cv5SD2RT-s1", "seed": 1}

# 修改 11 A3（2026-11-05）：Cv5 + T_phys v2（12→20 维，hull 着地几何 +
# 双足关系）。结构逐参数 = Cv5，唯一差异 = t_phys_v2；Cv5/SD2 的 12 维
# base 一行不动（R3）。首轮只做接链，不含量纲修正/A1/A2/A4。
# 4 折 LSO 与 V7_Cv5_f* 基线配对；V5_Av1 老基座已测 T2M −17.9mm。
# 必须放在第二轮白名单清理之后（与 Cv5SD2 同段），否则会被 _drop_all_but_current 删除。
MODELS["V7_Cv5_tphys"] = {**MODELS["V7_Cv5_s1"], "label": "V7-Cv5-tphys",
                          "name": "V7_Cv5_tphys"}
MODELS["V7_Cv5_tphys"].pop("seed", None)
MODELS["V7_Cv5_tphys"]["train_args"] = {
    **MODELS["V7_Cv5_tphys"].get("train_args", {}), "t_phys_v2": True}
MODELS["V7_Cv5_tphys_precheck"] = {
    **MODELS["V7_Cv5_tphys"], "label": "V7-Cv5-tphys-precheck",
    "name": "V7_Cv5_tphys_precheck", "epochs": 6,
    "train_args": {**MODELS["V7_Cv5_tphys"]["train_args"],
                   "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5_tphys_s1"] = {**MODELS["V7_Cv5_tphys"],
                             "label": "V7-Cv5-tphys-s1", "seed": 1}

# 修改 11 §9.3（2026-11-05）：hull_mcol 逐会话 p5–p95 重定心（pressure-only）。
# T_phys 仍 20 维，模型/编码器与 V7_Cv5_tphys 完全同形；唯一差异是数据侧
# hull_mcol 的值。对照 = Cv5（无新增维）与 Cv5_tphys（raw hull_mcol）。
MODELS["V7_Cv5_tphysRC"] = {**MODELS["V7_Cv5_tphys"],
                            "label": "V7-Cv5-tphysRC",
                            "name": "V7_Cv5_tphysRC",
                            "train_args": {**MODELS["V7_Cv5_tphys"]["train_args"],
                                           "t_phys_hull_recenter": True}}
MODELS["V7_Cv5_tphysRC_precheck"] = {
    **MODELS["V7_Cv5_tphysRC"], "label": "V7-Cv5-tphysRC-precheck",
    "name": "V7_Cv5_tphysRC_precheck", "epochs": 6,
    "train_args": {**MODELS["V7_Cv5_tphysRC"]["train_args"],
                   "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5_tphysRC_s1"] = {**MODELS["V7_Cv5_tphysRC"],
                               "label": "V7-Cv5-tphysRC-s1", "seed": 1}

# 平移锚头（2026-11-04，修改 08 → z_note/修改/08_平移锚头.md）：结构 =
# V7_Cv5（v7_c2 双门、枚举三条件、非 AR2）+ v7_trans_anchor_head（入口
# v7_c2ta）。锚头只读 m_v（stopgrad），主干逐位 = Cv5；新列
# GlobalMPJPE_self / W-MPJPE100_self / RootAbs_self 与现役列并列。
# 与 07 相互独立、可并行；基座 / 配方同 V7_Cv5_s1，新 split（_ACT8020，
# seed=101）对照 V7_Cv5_act8020_s1。裸名 = 结构基座、_sN = 训练格。
MODELS["V7_Cv5TA"] = {**MODELS["V7_Cv5_s1"], "label": "V7-Cv5TA", "name": "V7_Cv5TA"}
MODELS["V7_Cv5TA"].pop("seed", None)
MODELS["V7_Cv5TA_precheck"] = {**MODELS["V7_Cv5TA"], "label": "V7-Cv5TA-precheck",
                               "name": "V7_Cv5TA_precheck", "epochs": 6,
                               "train_args": {**MODELS["V7_Cv5TA"]["train_args"],
                                              "wandb_experiment_tag": "v7_variants_precheck"}}
MODELS["V7_Cv5TA_s1"] = {**MODELS["V7_Cv5TA"], "label": "V7-Cv5TA-s1", "seed": 1}

# 动作分层 split 80/20（2026-11-04，用户裁定）：旧 split 按被试切，动作分布在
# train/val/test 间严重偏移（val 缺 03/08/09，test 在难动作 06/07 超配），
# 模型好坏被动作难度偏移淹没。新 split = 按动作分层、切分单位 take 组、
# 04/08 全进 train（方案 A）、无 test 列（109/27，val 19.9%，8 类动作 val 占比
# 18–23%）。csv = splits_action_8020.csv，yaml = v1_act8020.yaml（只改 split_csv）。
# 口径：test 列为空 → eval_splits 必须只含 val（runner 默认 ("val","test") 会失败）。
# 与旧 split 的任何读数都不可比（用户已确认先不管）。
# 修改 09（2026-11-04）：读出按条件分支——V/VT 行绝对（= Cv5 目标空间）、T 行
# 相对 + 外部 ψ。结构 = AR2M + v7_abs_vrows（入口 v7_c2ar2ma）；配方逐项继承
# V7_Cv5AR2M_tw80_s1（obs80/tw20/bs64 + AR2 损失权重），唯一差异 = 结构入口。
# 判据：V/VT 行 PA 回到 Cv5 档（act8020 val 28.9/29.1，AR2M 现为 37.7/37.8），
# 同时 T 行航向保持 AR2 级（5 个 180° 折返会话绝对航向 2–8°、Drift 5.1）。
MODELS["V7_Cv5AR2MA_tw80_s1"] = {
    **MODELS["V7_Cv5AR2M_tw80_s1"],
    "label": "V7-Cv5AR2MA-obs80-out20-s1",
    "name": "V7_Cv5AR2MA", "seed": 1}

_ACT8020 = {"config": "anysole/configs/v1_act8020.yaml", "eval_splits": ["val"]}
for _base in ("V7_Cv5_s1", "V7_Cv2_s1", "V7_Cv5AR2M_tw80_s1", "V7_Cv5AR2MSC_tw160_s1", "V7_Cv7_s1",
              "V7_Cv5SD_s1", "V7_Cv5SD2_s1", "V7_Cv5SD2RT_s1", "V7_Cv5TA_s1", "V7_Cv5AR2MA_tw80_s1"):
    _new = _base.replace("_s1", "_act8020_s1")
    MODELS[_new] = {**MODELS[_base], **_ACT8020,
                    "label": MODELS[_base]["label"] + "-act8020"}
    # 卡位只由 worker 的 CUDA_VISIBLE_DEVICES 决定；cell 写死 device 会在
    # worker 只暴露 1 张卡时触发 "CUDA error: invalid device ordinal"（2026-11-04 实测）。
    MODELS[_new].pop("device", None)
    # 四格共用两个 model_name，监控页默认名 <model>_single_fold0 会重名；
    # 显式 run 名带 cell id，SwanLab 上一眼可分。
    MODELS[_new]["train_args"] = {**MODELS[_new].get("train_args", {}),
                                  "wandb_run_name": _new,
                                  "wandb_experiment_tag": "act8020"}
    # ckpt 路径 = model_name + 超参 variant，不含 split —— 与旧 split 的 _sd1
    # 目录同名会被覆盖（E2–E6 全部表格的来源）。seed 参与 variant，故新 split
    # 统一 seed=101 → 落到 _sd101，与旧 split 物理隔离（2026-11-04 实测）。
    MODELS[_new]["seed"] = 101

# 动作分层 split 的 E5 遮挡交叉评估（2026-11-04）：eval-only，ckpt = Cv5 act8020 格（_sd101）的 ckpt_last，
# 遮挡 yaml = v1_act8020_occNNN.yaml（只换 cache_root），test 列为空 → eval_splits 只含 val。
for _occ in OCC_LEVELS_10:
    MODELS["V7_Cv5_act8020_s1_evalocc%s" % _occ] = {
        **MODELS["V7_Cv5_act8020_s1"],
        "label": "V7-Cv5-act8020-s1-evalocc%s" % _occ,
        "train": False, "which": "last", "eval_splits": ["val"],
        "config": "anysole/configs/v1_act8020_occ%s.yaml" % _occ}
# Cv2 同口径遮挡交叉评估（2026-11-04 用户指定）：ckpt = Cv2 act8020 格（_sd101）。
for _occ in OCC_LEVELS_10:
    MODELS["V7_Cv2_act8020_s1_evalocc%s" % _occ] = {
        **MODELS["V7_Cv2_act8020_s1"],
        "label": "V7-Cv2-act8020-s1-evalocc%s" % _occ,
        "train": False, "which": "last", "eval_splits": ["val"],
        "config": "anysole/configs/v1_act8020_occ%s.yaml" % _occ}

# 门控随遮挡打开实验（2026-11-05 用户指定）：Cv7 = 无门对照，同 Cv5 的
# 遮挡交叉评估口径（eval-only、ckpt = Cv7 act8020 格 _sd101、只评 val）。
# E6 门控表在每个 occ 档用 Cv7（无门）作 ΔE 分解参照，比较 Cv5 双门。
for _occ in OCC_LEVELS_10:
    MODELS["V7_Cv7_act8020_s1_evalocc%s" % _occ] = {
        **MODELS["V7_Cv7_act8020_s1"],
        "label": "V7-Cv7-act8020-s1-evalocc%s" % _occ,
        "train": False, "which": "last", "eval_splits": ["val"],
        "config": "anysole/configs/v1_act8020_occ%s.yaml" % _occ}

# SD2 基座遮挡交叉评估（2026-11-05 用户指定，修改 07 判据 3）：eval-only
# 零重训，ckpt = SD2 act8020 格（_sd101）的 ckpt_last，只评 val。判据 =
# Δ(VT−V) 在 occ010–100 全档为负。
for _occ in OCC_LEVELS_10:
    MODELS["V7_Cv5SD2_act8020_s1_evalocc%s" % _occ] = {
        **MODELS["V7_Cv5SD2_act8020_s1"],
        "label": "V7-Cv5SD2-act8020-s1-evalocc%s" % _occ,
        "train": False, "which": "last", "eval_splits": ["val"],
        "config": "anysole/configs/v1_act8020_occ%s.yaml" % _occ}

# E5 视觉退化扩展（2026-11-04，z_note/E5_视觉退化扩展说明.md §3）：
# 连续断流 5 档，eval-only（train False、which last）。同一 ckpt
# （V7_Cv5 s1）固定 VT 输入，只替换视觉特征缓存。首轮只发 s1。
# 2026-11-08 用户裁定：光度/曝光轴整组删除，不再注册。
# 注意：本块必须留在 _drop_all_but_current() 之后（白名单不登记新格）。
_E5_GAP_SUFFIXES = ("010", "020", "040", "060", "080")
_V7_E5_SPEC = MODELS["V7_Cv5_s1"]
for _suffix in _E5_GAP_SUFFIXES:
    MODELS["V7_Cv5_s1_evalgap%s" % _suffix] = {
        **_V7_E5_SPEC,
        "label": _V7_E5_SPEC["label"] + "-evalgap%s" % _suffix,
        "train": False, "which": "last",
        "eval_splits": ["val", "test"],
        "config": "anysole/configs/v1_cam3_gap_r%s.yaml" % _suffix}

# 协议 A 主表 4 折（2026-11-05，论文设计选择表配套）：每个族一张 4 折表，
# 结构 = 该族 _s1 基座（ckpt_last、740ep），唯一差异 = 每 fold 专属 config
# （v1_fold_<testSubjects>.yaml，split_csv 指向该 fold 文件）。
# Cv7 折已在上方单独注册（无门臂）；本块覆盖 Cv5 / Cv5SD2 / Cv4 / Cv2 / Cv6。
# 必须放在文件末尾：V7_Cv5SD2_s1 等基座在本文件中段才定义。
_V7_FOLD_CONFIGS = (
    ("fS5S6S11",  "anysole/configs/v1_fold_S5S6S11.yaml"),
    ("fS5S11S12", "anysole/configs/v1_fold_S5S11S12.yaml"),
    ("fS6S7S13",  "anysole/configs/v1_fold_S6S7S13.yaml"),
    ("fS5S6S8",   "anysole/configs/v1_fold_S5S6S8.yaml"),
)
_V7_FOLD_FAMILIES = ("V7_Cv5", "V7_Cv5_tphys", "V7_Cv5_tphysRC", "V7_Cv5SD2",
                     "V7_Cv5SD2RT", "V7_Cv4", "V7_Cv2", "V7_Cv6")
for _fam in _V7_FOLD_FAMILIES:
    for _fold, _fconfig in _V7_FOLD_CONFIGS:
        _fname = "%s_%s" % (_fam, _fold)
        MODELS["%s_s1" % _fname] = {
            **MODELS["%s_s1" % _fam],
            "label": "%s-s1" % _fname.replace("_", "-"),
            "name": _fname,
            "config": _fconfig,
            "eval_splits": ["val", "test"]}

# 四折 LSO 的 E5 遮挡交叉评估（2026-11-05，Etest_crossfit 口径）：主线 5 格
# （Cv5 / Cv5SD2 / Cv5SD2RT / Cv5_tphys + Cv7 无门参照）× 4 折 ×
# occ020/040/060/080/100，eval-only、ckpt = 对应 fold cell 的 ckpt_last、
# 只评 test（留出受试者）。fold yaml 与 v1_fold_<fold>.yaml 仅差 cache_root。
# 必须留在 _drop_all_but_current() 与 fold 基座注册之后。
_E5_FOLD_MODELS = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT",
                   "V7_Cv5_tphys", "V7_Cv7")
_E5_FOLD_LEVELS = ("020", "040", "060", "080", "100")
for _fam in _E5_FOLD_MODELS:
    for _fold, _fconfig in _V7_FOLD_CONFIGS:
        _cell = "%s_%s_s1" % (_fam, _fold)
        for _occ in _E5_FOLD_LEVELS:
            MODELS["%s_evalocc%s" % (_cell, _occ)] = {
                **MODELS[_cell],
                "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": "anysole/configs/v1_occ%s_fold_%s.yaml"
                          % (_occ, _fold[1:])}

# S13 退出留出轮换（2026-11-05 用户裁定）：S13 采集时鞋垫硬件异常，压力量程
# 只用掉 71% 值域（单格峰值 915 < clip@1023、饱和 0.000%、活跃格占比 z=−3.01），
# 底层无法弥合，故 S13 只作 train/val 受试者、不作留出 test 受试者。
# 四折里唯一把 S13 放进 test 的是 fS6S7S13 → 本块对该折做 eval-only 重评，
# test 池收缩为 S6/S7（49→31 会话、18180→12540 帧），train/val 列与原折逐位
# 一致（`splits_test_S6S7_val_S14.csv`），故复用同一 ckpt_last、零重训。
# 口径代价：该折 test 动作覆盖 11/11→10/11（动作 08 全库仅 3 条 = S5×2 + S13，
# 本折 S5 在 train）；动作 08 的留出读数仍由 S5 在 test 的另三折提供。
_S13OUT_FAMILIES = _V7_FOLD_FAMILIES + ("V7_Cv7",)
for _fam in _S13OUT_FAMILIES:
    _cell = "%s_fS6S7S13_s1" % _fam
    MODELS["%s_s13out" % _cell] = {
        **MODELS[_cell],
        "label": "%s-s13out" % MODELS[_cell]["label"],
        "train": False, "which": "last",
        "eval_splits": ["test"],
        "config": "anysole/configs/v1_fold_S6S7.yaml"}

# S13-out 折的 E5 遮挡轴（2026-11-05 用户裁定「保留现有 4 折 + _s13out」）：
# 该折的 E5 occ 格同样池化了 S13，用同一 S13-out split + 5 档 occ yaml 做
# eval-only 覆盖；cell 名 = <fam>_fS6S7S13_s1_s13out_evalocc<lv>。
_E5_S13OUT_FAMS = ("V7_Cv5", "V7_Cv5SD2", "V7_Cv5SD2RT",
                   "V7_Cv5_tphys", "V7_Cv7")
for _fam in _E5_S13OUT_FAMS:
    _cell = "%s_fS6S7S13_s1_s13out" % _fam
    for _occ in ("020", "040", "060", "080", "100"):
        MODELS["%s_evalocc%s" % (_cell, _occ)] = {
            **MODELS[_cell],
            "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
            "train": False, "which": "last",
            "eval_splits": ["test"],
            "config": "anysole/configs/v1_occ%s_fold_S6S7.yaml" % _occ}

# V8 系列（2026-11-05）：新基座 = V7_Cv5SD2；门控/条件消融在 V8 上重建。
# 基座结构由入口 v8_cv5（= v7_c2sd2）携带；Cv4/Cv7 用新入口 v8_cv4/v8_cv7，
# Cv2/Cv6 与 Cv5 同结构、只差 train_args（sample / always-VT）。
# 锚头：V8 全族由入口 STRUCTURE 统一置 `v7_trans_anchor_head: True`
# （V8_Cv2/Cv6 复用 v8_cv5，同样带头；不是 train_args 开关）。
# SD2：V8 全族都带 `v7_root_sd` + `v7_root_sd_balance`；无 SD2 的
# 「通道不分离」对照 = V7_Cv5（现成 4 折 + `_s13out`，不重训）。
# 第 4 折直接用 S13-out split（v1_fold_S6S7.yaml）：train/val 与原折逐位
# 相同，test = S6+S7；主评估与 E5/E6 等实验沿同一折表，不再二次 eval-only。
_V8_BASE = {**MODELS["V7_Cv5SD2_s1"], "label": "V8-Cv5", "name": "V8_Cv5"}
MODELS["V8_Cv5"] = _V8_BASE
MODELS["V8_Cv4"] = {**MODELS["V7_Cv4_s1"], "label": "V8-Cv4", "name": "V8_Cv4"}
MODELS["V8_Cv2"] = {**MODELS["V7_Cv2_s1"], "label": "V8-Cv2", "name": "V8_Cv2"}
MODELS["V8_Cv6"] = {**MODELS["V7_Cv6_s1"], "label": "V8-Cv6", "name": "V8_Cv6"}
MODELS["V8_Cv7"] = {**MODELS["V7_Cv7_s1"], "label": "V8-Cv7", "name": "V8_Cv7"}
# T1：V8_Cv5 的方向载体替换（根头 4→3 通道；结构由入口 v8_cv5t1 携带）。
MODELS["V8_Cv5T1"] = {**MODELS["V8_Cv5"], "label": "V8-Cv5T1", "name": "V8_Cv5T1"}
MODELS["V8_Cv5T1_precheck"] = {
    **MODELS["V8_Cv5T1"], "label": "V8-Cv5T1-precheck",
    "name": "V8_Cv5T1_precheck", "epochs": 6,
    "train_args": {**MODELS["V8_Cv5T1"]["train_args"],
                   "wandb_experiment_tag": "v8_precheck"}}
# 门控条件化波 1（修改 13，2026-11-06）。
# D1 RM：结构 = V8_Cv5，单旗在 train_args —— occ_mix 档梯 {010,020,030}
#   逐窗均匀抽档、p_occ=0.45、**p_nullV=p_nullT=0**（enum 本来就在枚举
#   V/T 在场与否，两个 null 开关与它同义重复；置零后与 enum 不冲突）。
#   只训低档 ⇒ E5 的 040–100 保持分布外，OOD 语义不变。
MODELS["V8_Cv5RM"] = {
    **MODELS["V8_Cv5"], "label": "V8-Cv5RM", "name": "V8_Cv5RM",
    "train_args": {**MODELS["V8_Cv5"]["train_args"],
                   "occ_mix": True, "occ_levels": "010,020,030",
                   "p_occ": 0.45, "p_nullV": 0.0, "p_nullT": 0.0}}
# D3 PO：结构单旗（腿/足 query 门钉 1.0，入口 v8_cv5po 携带），train_args
#   与 V8_Cv5 逐项相同。
MODELS["V8_Cv5PO"] = {**MODELS["V8_Cv5"], "label": "V8-Cv5PO",
                      "name": "V8_Cv5PO"}
# 双向信任门波 1（修改 14 §5，2026-11-06）。
#   VG：结构单旗在入口 v8_cv5vg（V 门 + T 门 2σ），train_args 同 V8_Cv5；
#   ST：结构同 V8_Cv5，单旗在 train_args（p_stale=0.5，同会话窗级错位）；
#   VGST：两旗同开，主模型。
MODELS["V8_Cv5VG"] = {**MODELS["V8_Cv5"], "label": "V8-Cv5VG",
                      "name": "V8_Cv5VG"}
MODELS["V8_Cv5ST"] = {
    **MODELS["V8_Cv5"], "label": "V8-Cv5ST", "name": "V8_Cv5ST",
    "train_args": {**MODELS["V8_Cv5"]["train_args"], "p_stale": 0.5}}
MODELS["V8_Cv5VGST"] = {
    **MODELS["V8_Cv5VG"], "label": "V8-Cv5VGST", "name": "V8_Cv5VGST",
    "train_args": {**MODELS["V8_Cv5VG"]["train_args"], "p_stale": 0.5}}
# 修改 15（2026-11-07）：Cv5 + V 门（T 门 Cv5），两阶段只训 V 门。
MODELS["V8_Cv5VIG"] = {**MODELS["V8_Cv5"], "label": "V8-Cv5VIG",
                       "name": "V8_Cv5VIG"}
# 修改 15（2026-11-07）：Cv5 + 不确定性专家 V 门（IG 监督，运行时零 GT）。
MODELS["V8_Cv5VUG"] = {**MODELS["V8_Cv5"], "label": "V8-Cv5VUG",
                       "name": "V8_Cv5VUG"}
# 竞争路由门（2026-11-07 会话裁定）：k = σ(s_v − s_t) 逐帧逐组，s_v 读
# V token、s_t 读 T token + 分组物理/IMU 特征（T_phys + T_s2m 双足
# IMU，只作门控特征）；结构单旗在入口 v8_cv8（v7_route_router），
# train_args 同 V8_Cv5。对照 = V8_Cv5（单侧双门）/ V8_Cv7（无门）。
MODELS["V8_Cv8"] = {**MODELS["V8_Cv5"], "label": "V8-Cv8", "name": "V8_Cv8"}
MODELS["V8_Cv8_precheck"] = {
    **MODELS["V8_Cv8"], "label": "V8-Cv8-precheck",
    "name": "V8_Cv8_precheck", "epochs": 6,
    "train_args": {**MODELS["V8_Cv8"]["train_args"],
                   "wandb_experiment_tag": "v8_precheck"}}
_V8_FOLD_CONFIGS = (
    ("fS5S6S11",  "anysole/configs/v1_fold_S5S6S11.yaml"),
    ("fS5S11S12", "anysole/configs/v1_fold_S5S11S12.yaml"),
    ("fS6S7",     "anysole/configs/v1_fold_S6S7.yaml"),
    ("fS5S6S8",   "anysole/configs/v1_fold_S5S6S8.yaml"),
)
_V8_FOLD_FAMILIES = ("V8_Cv5", "V8_Cv4", "V8_Cv2", "V8_Cv6", "V8_Cv7",
                     "V8_Cv5T1", "V8_Cv5RM", "V8_Cv5PO",
                     "V8_Cv5VG", "V8_Cv5ST", "V8_Cv5VGST", "V8_Cv5VIG",
                     "V8_Cv8")
for _fam in _V8_FOLD_FAMILIES:
    for _fold, _fconfig in _V8_FOLD_CONFIGS:
        _fname = "%s_%s" % (_fam, _fold)
        MODELS["%s_s1" % _fname] = {
            **MODELS[_fam],
            "label": "%s-s1" % _fname.replace("_", "-"),
            "name": _fname,
            "config": _fconfig,
            "eval_splits": ["val", "test"],
            "seed": 1,
        }

# V8 的 E5/E6 评估格（2026-11-06 用户指定）：6 族 × 4 折。
#   E5 = occ020/040/060/080/100，eval-only、只评 test；
#   E6 = 独立 base test eval-only（`_e6`，读同一 fold yaml/ckpt）。
# 已评估的参照（V81 = 旧 V7_Cv5SD2、V7_Cv5）不在此列。
_V8_EVAL_FAMILIES = ("V8_Cv5", "V8_Cv4", "V8_Cv2", "V8_Cv6",
                     "V8_Cv5RM", "V8_Cv5PO")
_V8_EVAL_LEVELS = ("020", "040", "060", "080", "100")
for _fam in _V8_EVAL_FAMILIES:
    for _fold, _fconfig in _V8_FOLD_CONFIGS:
        _cell = "%s_%s_s1" % (_fam, _fold)
        MODELS["%s_e6" % _cell] = {
            **MODELS[_cell],
            "label": "%s-e6" % MODELS[_cell]["label"],
            "train": False, "which": "last",
            "eval_splits": ["test"],
            "config": _fconfig}
        for _occ in _V8_EVAL_LEVELS:
            MODELS["%s_evalocc%s" % (_cell, _occ)] = {
                **MODELS[_cell],
                "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": "anysole/configs/v1_occ%s_fold_%s.yaml"
                          % (_occ, _fold[1:])}

# 无门参照的遮挡轴（14 号文档 §1.9 / 波 0，2026-11-06）：`V8_Cv7` 在
# `_V8_FOLD_FAMILIES` 里、四折 ckpt 与 base test 都已落盘，但不在上面那张
# 用户指定的 6 族 `_V8_EVAL_FAMILIES` 里，所以四折遮挡轴上「有门 vs 无门」
# ——§4.4 的核心对照——一格数据都没有，现在能引的全是 act8020/val 数字。
# 单独补 20 格，不动上面那张表。
for _fold, _fconfig in _V8_FOLD_CONFIGS:
    _cell = "V8_Cv7_%s_s1" % _fold
    for _occ in _V8_EVAL_LEVELS:
        MODELS["%s_evalocc%s" % (_cell, _occ)] = {
            **MODELS[_cell],
            "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
            "train": False, "which": "last",
            "eval_splits": ["test"],
            "config": "anysole/configs/v1_occ%s_fold_%s.yaml"
                      % (_occ, _fold[1:])}

# 竞争路由门的 E5/E6 评估格（2026-11-07）：V8_Cv8 × 4 折，独立块
# （同 VG/Cv7 的做法，不动上面 6 族表的 experiment MODEL_IDS）。
for _fam in ("V8_Cv8",):
    for _fold, _fconfig in _V8_FOLD_CONFIGS:
        _cell = "%s_%s_s1" % (_fam, _fold)
        MODELS["%s_e6" % _cell] = {
            **MODELS[_cell],
            "label": "%s-e6" % MODELS[_cell]["label"],
            "train": False, "which": "last",
            "eval_splits": ["test"],
            "config": _fconfig}
        for _occ in _V8_EVAL_LEVELS:
            MODELS["%s_evalocc%s" % (_cell, _occ)] = {
                **MODELS[_cell],
                "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": "anysole/configs/v1_occ%s_fold_%s.yaml"
                          % (_occ, _fold[1:])}

# 双向信任门波 1 的 E5/E6 评估格（修改 14 §6，2026-11-06）：VG/ST/VGST × 4 折。
# 与上面 6 族表分离，避免改已有 experiment 的 MODEL_IDS（同 Cv7 的做法）。
for _fam in ("V8_Cv5VG", "V8_Cv5ST", "V8_Cv5VGST"):
    for _fold, _fconfig in _V8_FOLD_CONFIGS:
        _cell = "%s_%s_s1" % (_fam, _fold)
        MODELS["%s_e6" % _cell] = {
            **MODELS[_cell],
            "label": "%s-e6" % MODELS[_cell]["label"],
            "train": False, "which": "last",
            "eval_splits": ["test"],
            "config": _fconfig}
        for _occ in _V8_EVAL_LEVELS:
            MODELS["%s_evalocc%s" % (_cell, _occ)] = {
                **MODELS[_cell],
                "label": "%s-evalocc%s" % (MODELS[_cell]["label"], _occ),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": "anysole/configs/v1_occ%s_fold_%s.yaml"
                          % (_occ, _fold[1:])}

# E5 场景遮挡补测格（2026-11-08 用户裁定）：7 个模型 × 4 折 ×（scene, clean）。
# 排除：V7_Cv5SD2 / V7_Cv5SD2RT / V7_Cv5_tphys / V8_Cv5RM / V8_Cv5PO；
# V7_Cv5 已在 2026-11-07 单独跑过正式 8 格，不在此表。
# V7_Cv7 的 fS6S7 折沿用 S13-out 基座 cell（fS6S7S13），config 用 woS13 的
# v1_sceneocc_fS6S7.yaml；V8 各折直接用 v1_sceneocc_<fold>.yaml。
_E5_SCENE_MODELS = {
    "V7_Cv7": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7S13", "fS5S6S8": "fS5S6S8"},
    "V8_Cv5": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
    "V8_Cv4": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
    "V8_Cv2": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
    "V8_Cv6": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
    "V8_Cv7": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
    "V8_Cv8": {"fS5S6S11": "fS5S6S11", "fS5S11S12": "fS5S11S12",
               "fS6S7": "fS6S7", "fS5S6S8": "fS5S6S8"},
}
for _fam, _foldmap in _E5_SCENE_MODELS.items():
    for _fold, _base_fold in _foldmap.items():
        _cell = "%s_%s_s1" % (_fam, _base_fold)
        for _tag, _cfg in (
            ("evalscene", "anysole/configs/v1_sceneocc_%s.yaml" % _fold),
            ("evalsceneclean", "anysole/configs/v1_sceneocc_clean_%s.yaml" % _fold),
        ):
            MODELS["%s_%s" % (_cell, _tag)] = {
                **MODELS[_cell],
                "label": "%s-%s" % (MODELS[_cell]["label"], _tag),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": _cfg,
                "dump_detail": True,
            }

# V8 的 E5 时间退化（连续断流 / gap）评估格（2026-11-08 用户补充）：
# 5 个正文模型 × 4 折 × ρ={10,20,40,60,80}%，eval-only、只评 test。
# 缓存 = cam3_gap_r<rho>（已为 V8 四折 test 的 98 个唯一会话补齐）；
# yaml = v1_gap_r<rho>_fold_<fold>（与 v1_fold_<fold>.yaml 仅差 cache_root）。
_V8_GAP_FAMILIES = ("V8_Cv5", "V8_Cv4", "V8_Cv2", "V8_Cv6", "V8_Cv7")
_V8_GAP_LEVELS = ("010", "020", "040", "060", "080")
for _fam in _V8_GAP_FAMILIES:
    for _fold, _fconfig in _V8_FOLD_CONFIGS:
        _cell = "%s_%s_s1" % (_fam, _fold)
        for _rho in _V8_GAP_LEVELS:
            MODELS["%s_evalgap%s" % (_cell, _rho)] = {
                **MODELS[_cell],
                "label": "%s-evalgap%s" % (MODELS[_cell]["label"], _rho),
                "train": False, "which": "last",
                "eval_splits": ["test"],
                "config": "anysole/configs/v1_gap_r%s_fold_%s.yaml"
                          % (_rho, _fold[1:])}
