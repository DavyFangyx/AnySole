"""Shared registry and queue writer for the two data-producing experiments."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.z_gen.model_registry import COMMON, MODELS


CONFIG_DIR = Path(__file__).resolve().parents[1]
QUEUE_DIR = CONFIG_DIR / "queue"
STATES = ("queue", "running", "done", "failed")

# These are the only experiments that produce model/evaluation data.  Analysis
# and visualization live under results_display and are intentionally absent.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # B 查询压力头线（2026-10-24）：V3_Bv2（Q-NN 先验表示）→ V3_Bv3
    # （Q-FACT 因式化）依次训练；B 的既有配方（740ep/seed1/三条件采样），
    # 唯一变化 = 压力头结构（版本节登记见 V3_B方案.md §6）。
    "bv_query": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Bv2", "V3_Bv3"],
        "REQUIRES": [],
    },
    "all_models": {
        "KIND": "model_suite",
        "MODEL_IDS": list(MODELS),
        "REQUIRES": [],
    },
    "singlemodal_eval": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_Bv1"],
        "SPECIALISTS": ["vonly", "tonly"],
        "REQUIRES": [],
    },
    # 5-fold 动作覆盖实验（2026-10-22）：受试者轮换 leave-subjects-out，
    # test 覆盖 11/11 动作；波 1/2/3 与 fold 条目见 model_registry 同名块。
    # 生成 = configs/z_gen/per_action_folds.py（--stage 12 先入队，
    # 波 3 待波 1 E1 ckpt 完成后生成）。
    "per_action_folds": {
        "KIND": "model_suite",
        "MODEL_IDS": [
            "V3_Ev1_fS13S14", "V3_Ev1_fS5S6S11", "V3_Ev1_fS5S11S12",
            "V3_Ev1_fS6S7S13", "V3_Ev1_fS5S6S8",
            "V3_Bv1_fS13S14", "V3_Bv1_fS5S6S11", "V3_Bv1_fS5S11S12",
            "V3_Bv1_fS6S7S13", "V3_Bv1_fS5S6S8",
            "V5_Av4_fS13S14", "V5_Av4_fS5S6S11",
            "V5_Av4_fS5S11S12", "V5_Av4_fS6S7S13",
            "V5_Av4_fS5S6S8",
            "V5_Av1_fS13S14", "V5_Av1_fS5S6S11", "V5_Av1_fS5S11S12",
            "V5_Av1_fS6S7S13", "V5_Av1_fS5S6S8",
            "V4_Av1_fS13S14", "V4_Av1_fS5S6S11", "V4_Av1_fS5S11S12",
            "V4_Av1_fS6S7S13", "V4_Av1_fS5S6S8",
            # occ050 档（2026-10-23 裁定「加」）
            "V5_Av1_occ050_fS13S14", "V5_Av1_occ050_fS5S6S11",
            "V5_Av1_occ050_fS5S11S12", "V5_Av1_occ050_fS6S7S13",
            "V5_Av1_occ050_fS5S6S8",
            "V4_Av1_occ050_fS13S14", "V4_Av1_occ050_fS5S6S11",
            "V4_Av1_occ050_fS5S11S12", "V4_Av1_occ050_fS6S7S13",
            "V4_Av1_occ050_fS5S6S8",
            # occ010 档（2026-10-23 裁定「加」）
            "V5_Av1_occ010_fS13S14", "V5_Av1_occ010_fS5S6S11",
            "V5_Av1_occ010_fS5S11S12", "V5_Av1_occ010_fS6S7S13",
            "V5_Av1_occ010_fS5S6S8",
            "V4_Av1_occ010_fS13S14", "V4_Av1_occ010_fS5S6S11",
            "V4_Av1_occ010_fS5S11S12", "V4_Av1_occ010_fS6S7S13",
            "V4_Av1_occ010_fS5S6S8",
            # 全面添加（2026-10-23）：V3_Bv1/V5_Av4 × occ010/occ050
            "V3_Bv1_occ050_fS13S14", "V3_Bv1_occ050_fS5S6S11",
            "V3_Bv1_occ050_fS5S11S12", "V3_Bv1_occ050_fS6S7S13",
            "V3_Bv1_occ050_fS5S6S8",
            "V5_Av4_occ050_fS13S14", "V5_Av4_occ050_fS5S6S11",
            "V5_Av4_occ050_fS5S11S12", "V5_Av4_occ050_fS6S7S13",
            "V5_Av4_occ050_fS5S6S8",
            "V3_Bv1_occ010_fS13S14", "V3_Bv1_occ010_fS5S6S11",
            "V3_Bv1_occ010_fS5S11S12", "V3_Bv1_occ010_fS6S7S13",
            "V3_Bv1_occ010_fS5S6S8",
            "V5_Av4_occ010_fS13S14", "V5_Av4_occ010_fS5S6S11",
            "V5_Av4_occ010_fS5S11S12", "V5_Av4_occ010_fS6S7S13",
            "V5_Av4_occ010_fS5S6S8",
        ],
        "REQUIRES": [],
    },
    # 四相机扩展 D2 (2026-10-12): same protocol on the three new cameras.
    # tonly is deliberately absent -- the T stream is camera-independent and
    # the shared V3_Bv1_tonly specialist serves all four cameras.
    "singlemodal_eval_cams": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_Bv1_cam1", "V3_Bv1_cam2", "V3_Bv1_cam4"],
        "SPECIALISTS": ["vonly"],
        "REQUIRES": [],
    },
    # V-degradation 遮挡组 (2026-10-13): cam3 with the bottom fraction of the
    # person bbox occluded.  tonly is absent for the same reason as the camera
    # group -- the T stream is occlusion-independent, V3_Bv1_tonly serves all.
    "singlemodal_eval_occl": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_Bv1_cam3_occ010", "V3_Bv1_cam3_occ020",
                      "V3_Bv1_cam3_occ030", "V3_Bv1_cam3_occ050"],
        "SPECIALISTS": ["vonly"],
        "REQUIRES": [],
    },
    "rho_grid_eval": {
        "KIND": "rho_grid",
        "MODEL_IDS": ["V3_Bv1"],
        "REQUIRES": [],
        "RHOS": "0,20,40,60,80,100",
        "TRAIN_REPR_RHOS": "0,100",
        "VAL_SEEDS": "0,1,2",
        "TEST_SEEDS": "0",
    },
    "residual_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Cv1"],
        "REQUIRES": [],
    },
    "position_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Dv1"],
        "REQUIRES": [],
    },
    # V3_Dv1 削弱套件（2026-10-15）：8 配置 ×（训练 + val + test）事务；
    # cam3 基准行 = position_test/V3_Dv1。每配置自训自测，与 V3_Bv1 套件同协议。
    # V3_Bv1 occ010 生产评估补跑（2026-10-18）：ckpt 已训（10-13），
    # metrics 缺失——只补评估。
    "v3b_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Bv1_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Ev1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3e_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Ev1", "V3_Ev1_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Fv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3f_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Fv1", "V3_Fv1_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Gv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3g_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Gv1", "V3_Gv1_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Hv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3h_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Hv1", "V3_Hv1_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Hv2 首组（2026-10-19）：命题 2 载体依赖修复（T 流 aux 重建）。
    "v3h2_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Hv2", "V3_Hv2_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_Hv3 首组（2026-10-19）：v2 + 一阶足部残差头（命题 2 兑现通道）。
    "v3h3_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Hv3", "V3_Hv3_cam3_occ010"],
        "REQUIRES": [],
    },
    # 新数据验证（2026-10-22）：生产 v3+offset facts 的替换验证。
    "v3off_validation": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Bv1_cam3_v3off"],
        "REQUIRES": [],
    },
    # V3_Hv5 无锚版全网格（2026-10-20）：4 相机 × 3 遮挡 = 12 格
    # （T 价值区 = V 失效条件的无锚对照；cam{1,2,4}_occ{010,020} 的
    # 遮挡缓存由 extract_hrnet 生成）。
    "v3h5_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Hv5", "V3_Hv5_cam3_occ010", "V3_Hv5_cam3_occ020",
                      "V3_Hv5_cam1", "V3_Hv5_cam1_occ010", "V3_Hv5_cam1_occ020",
                      "V3_Hv5_cam2", "V3_Hv5_cam2_occ010", "V3_Hv5_cam2_occ020",
                      "V3_Hv5_cam4", "V3_Hv5_cam4_occ010", "V3_Hv5_cam4_occ020"],
        "REQUIRES": [],
    },
    # V3_Hv6 首组（2026-10-20）：v5 + 残差头落点改踝（错误一修复），
    # cam3 基准 + cam3+occ010，自训自测。
    "v3h6_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Hv6", "V3_Hv6_cam3_occ010"],
        "REQUIRES": [],
    },
    # 新数据验证（2026-10-22，主会话 conf 00618）：生产 v3+offset facts
    # 替换验证——V3_Bv1_cam3_v3off = E1 的 3B 统一模型行。
    "v3off_validation": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Bv1_cam3_v3off"],
        "REQUIRES": [],
    },
    # E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型，
    # 生产 v3+offset facts。波 1 = E 家族（新载体）+ 3B 家族；波 2 = V4_Av1/
    # V5_Av1 家族（carrier 依赖波 1 的 E 基名 ckpt）。
    "e1_experts": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Ev1_cam3_v3off", "V3_Ev1_cam3_v3off_drop",
                      "V3_Ev1_cam3_v3off_vonly", "V3_Ev1_cam3_v3off_tonly",
                      "V3_Bv1_cam3_v3off_vonly", "V3_Bv1_cam3_v3off_tonly",
                      "V3_Bv1_cam3_v3off_nodrop",
                      "V4_Av1_cam3_v3off", "V4_Av1_cam3_v3off_drop",
                      "V4_Av1_cam3_v3off_vonly", "V4_Av1_cam3_v3off_tonly",
                      "V5_Av1_cam3_v3off", "V5_Av1_cam3_v3off_drop",
                      "V5_Av1_cam3_v3off_vonly", "V5_Av1_cam3_v3off_tonly"],
        "REQUIRES": [],
    },
    # E1 修复迭代（2026-10-22）：V5_Av4 四变体（从零 V 流 + 先验塔 E 配方），
    # v3off 生产轴，论文正文模型候选。
    "e1_av4": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_Av4_cam3_v3off", "V5_Av4_cam3_v3off_drop",
                      "V5_Av4_cam3_v3off_vonly", "V5_Av4_cam3_v3off_tonly"],
        "REQUIRES": [],
    },
    # wo05 实验轴（2026-10-24）：canonical split 去掉动作 05（绕圈跑）的
    # 12 模型（3 家族 x E1 四变体，splits_wo05.csv；train 92->88，val/test
    # 不变）。波 1 = V3_Ev1_wo05 + V5_Av4_wo05（V5_Av4 carrier=scratch 无
    # E 依赖）；波 2 = V4_Av1_wo05（carrier 依赖波 1 的 V3_Ev1_wo05
    # ckpt_best）。MODEL_IDS 顺序即 conf 顺序：波 1 排最前（worker 先取件）。
    "wo05": {
        "KIND": "model_suite",
        "MODEL_IDS": [
            "V3_Ev1_wo05", "V3_Ev1_wo05_drop",
            "V3_Ev1_wo05_vonly", "V3_Ev1_wo05_tonly",
            "V5_Av4_wo05", "V5_Av4_wo05_drop",
            "V5_Av4_wo05_vonly", "V5_Av4_wo05_tonly",
            "V4_Av1_wo05", "V4_Av1_wo05_drop",
            "V4_Av1_wo05_vonly", "V4_Av1_wo05_tonly"],
        "REQUIRES": [],
    },
    # 时间窗扫描（2026-10-22，z_note/multimodal_fusion/时间窗扫描实验.md）：
    # 五家族 x tw{5,10,15,20} x cam3/cam3_occ010（v3off 生产轴）。
    # 波 1 = E + V3_Bv1 + V5_Av4；波 2 = V4_Av1 + V5_Av1
    # （闸门 = 波 1 的 E 格完成——同 (tw, 视觉) 的 E 载体就绪后才可训）。
    # MODEL_IDS 顺序即 conf 顺序：波 1 的 E 格排最前（worker 先取件）。
    "tw_scan": {
        "KIND": "model_suite",
        "MODEL_IDS": [
            "V3_Ev1_cam3_v3off_tw5", "V3_Ev1_cam3_v3off_tw10",
            "V3_Ev1_cam3_v3off_tw15", "V3_Ev1_cam3_occ010_v3off",
            "V3_Ev1_cam3_occ010_v3off_tw5", "V3_Ev1_cam3_occ010_v3off_tw10",
            "V3_Ev1_cam3_occ010_v3off_tw15",
            "V3_Bv1_cam3_v3off_tw5", "V3_Bv1_cam3_v3off_tw10",
            "V3_Bv1_cam3_v3off_tw15", "V3_Bv1_cam3_occ010_v3off",
            "V3_Bv1_cam3_occ010_v3off_tw5", "V3_Bv1_cam3_occ010_v3off_tw10",
            "V3_Bv1_cam3_occ010_v3off_tw15",
            "V5_Av4_cam3_v3off_tw5", "V5_Av4_cam3_v3off_tw10",
            "V5_Av4_cam3_v3off_tw15", "V5_Av4_cam3_occ010_v3off",
            "V5_Av4_cam3_occ010_v3off_tw5", "V5_Av4_cam3_occ010_v3off_tw10",
            "V5_Av4_cam3_occ010_v3off_tw15",
            "V4_Av1_cam3_v3off_tw5", "V4_Av1_cam3_v3off_tw10",
            "V4_Av1_cam3_v3off_tw15", "V4_Av1_cam3_occ010_v3off",
            "V4_Av1_cam3_occ010_v3off_tw5", "V4_Av1_cam3_occ010_v3off_tw10",
            "V4_Av1_cam3_occ010_v3off_tw15",
            "V5_Av1_cam3_v3off_tw5", "V5_Av1_cam3_v3off_tw10",
            "V5_Av1_cam3_v3off_tw15", "V5_Av1_cam3_occ010_v3off",
            "V5_Av1_cam3_occ010_v3off_tw5", "V5_Av1_cam3_occ010_v3off_tw10",
            "V5_Av1_cam3_occ010_v3off_tw15",
            # 锚点格（tw20×cam3，E1 波 ckpt）：纳入以便 session_detail 重评
            "V3_Bv1_cam3_v3off", "V4_Av1_cam3_v3off",
            "V5_Av1_cam3_v3off", "V5_Av4_cam3_v3off"],
        "REQUIRES": [],
    },
    # tw × 遮挡交互（2026-10-24）：occ020/030 × tw{5,10,15,20}（v3off 轴），
    # V3_Bv1 / V5_Av4 两模型。参照 = tw_scan 的 cam3 与 occ010 腿。
    "tw_occ_scan": {
        "KIND": "model_suite",
        "MODEL_IDS": [
            "V3_Bv1_cam3_occ020_v3off", "V3_Bv1_cam3_occ020_v3off_tw5",
            "V3_Bv1_cam3_occ020_v3off_tw10", "V3_Bv1_cam3_occ020_v3off_tw15",
            "V3_Bv1_cam3_occ030_v3off", "V3_Bv1_cam3_occ030_v3off_tw5",
            "V3_Bv1_cam3_occ030_v3off_tw10", "V3_Bv1_cam3_occ030_v3off_tw15",
            "V5_Av4_cam3_occ020_v3off", "V5_Av4_cam3_occ020_v3off_tw5",
            "V5_Av4_cam3_occ020_v3off_tw10", "V5_Av4_cam3_occ020_v3off_tw15",
            "V5_Av4_cam3_occ030_v3off", "V5_Av4_cam3_occ030_v3off_tw5",
            "V5_Av4_cam3_occ030_v3off_tw10", "V5_Av4_cam3_occ030_v3off_tw15"],
        "REQUIRES": [],
    },
    # T 增强三组（2026-10-21，T增强三组实验方案.md）：维度 2 显式物理特征
    # （tphys）+ 维度 3 足-地监督（d8，监督头落地后加入）。
    "v5_1_tenhance_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_Av1_cam3_offfix_tphys", "V5_Av1_cam3_offfix_d8"],
        "REQUIRES": [],
    },
    # V5_Av1 首组（2026-10-19，V5_Av1方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。
    "v5_1_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_Av1", "V5_Av1_cam3_occ010", "V5_Av1_occmix_cam3_occ010"],
        "REQUIRES": [],
    },
    # V4_Av1 首组（2026-10-19，V4_Av1方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。V6_Av1 occ_mix
    # 为可选增强开关（conf 直写 OCC_MIX=true/P_OCC/P_NULLV/P_NULLT），
    # 默认关——不为 V6_Av1 单独立项训练 run。
    "v4_1_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V4_Av1", "V4_Av1_cam3_occ010", "V4_Av1_occmix_cam3_occ010"],
        "REQUIRES": [],
    },
    "position_test_vdegrad": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_Dv1_cam1", "V3_Dv1_cam2", "V3_Dv1_cam4",
                      "V3_Dv1_cam3_occ010", "V3_Dv1_cam3_occ020",
                      "V3_Dv1_cam3_occ030", "V3_Dv1_cam3_occ050"],
        "REQUIRES": [],
    },
}


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def build_registry() -> dict[str, Any]:
    defaults = {
        "config": COMMON.get("CONFIG", "anysole/configs/v1.yaml"),
        "results_root": COMMON.get("RESULTS_ROOT", "results"),
        "display_root": COMMON.get("DISPLAY_ROOT", "results_display"),
        "device": COMMON.get("DEVICE", "cuda"),
        "modal": COMMON.get("MODAL", "anysolev2"),
        "contact_method": COMMON.get("CONTACT_METHOD", "joint_and"),
        "train_seed": COMMON.get("TRAIN_SEED", 1),
        "epochs": COMMON.get("EPOCHS", 800),
        "batch_size": COMMON.get("BATCH_SIZE", 256),
        "protocol_seed": COMMON.get("PROTOCOL_SEED", 0),
    }
    registry: dict[str, Any] = {
        "version": 2,
        "defaults": defaults,
        "models": {
            model_id: {**spec, "train_args": dict(spec.get("train_args") or {})}
            for model_id, spec in MODELS.items()
        },
        "experiments": {},
    }
    for experiment_id, raw in EXPERIMENTS.items():
        requires = [str(item) for item in raw.get("REQUIRES", [])]
        models = []
        for model_id in raw["MODEL_IDS"]:
            missing = [cap for cap in requires if cap not in MODELS[model_id].get("provides", ())]
            if not missing:
                models.append(model_id)
        spec = {
            "kind": str(raw["KIND"]),
            "models": models,
            "depends_on": [],
            "display": False,
            "requires": requires,
            "specialists": [str(item) for item in raw.get("SPECIALISTS", [])],
        }
        for source, target in (
            ("RHOS", "rhos"),
            ("TRAIN_REPR_RHOS", "train_repr_rhos"),
            ("VAL_SEEDS", "val_seeds"),
            ("TEST_SEEDS", "test_seeds"),
        ):
            if source in raw:
                spec[target] = _parse_value(str(raw[source]))
        registry["experiments"][experiment_id] = spec
    return registry


def _next_sequence() -> int:
    pattern = re.compile(r"^(\d+)(?:__|_)")
    values: list[int] = []
    for state in STATES:
        directory = CONFIG_DIR / state
        if not directory.is_dir():
            continue
        for path in directory.glob("*.conf"):
            match = pattern.match(path.name)
            if match:
                values.append(int(match.group(1)))
    return max(values, default=0) + 1


def _task_specs(spec: dict[str, Any], models: list[str]) -> list[tuple[str, str | None]]:
    kind = str(spec["KIND"])
    tasks: list[tuple[str, str | None]] = []
    if kind == "model_suite":
        # One train+eval transaction per registered base model (no
        # specialists, no grid cells).
        for model_id in models:
            tasks.append(("model_run", model_id))
    elif kind == "singlemodal_compare":
        for model_id in models:
            tasks.append(("model_run", model_id))
            for specialist in spec.get("SPECIALISTS", []):
                tasks.append(("model_run", f"{model_id}_{specialist}"))
    elif kind == "rho_grid":
        train_rhos = [int(value) for value in str(spec["TRAIN_REPR_RHOS"]).split(",")]
        rhos = [int(value) for value in str(spec["RHOS"]).split(",")]
        val_seeds = [int(value) for value in str(spec["VAL_SEEDS"]).split(",")]
        test_seeds = [int(value) for value in str(spec["TEST_SEEDS"]).split(",")]
        for model_id in models:
            for split, split_rhos, seeds in (
                ("train", train_rhos, [0]),
                ("val", rhos, val_seeds),
                ("test", rhos, test_seeds),
            ):
                for seed in seeds:
                    for rho_v in split_rhos:
                        for rho_t in split_rhos:
                            tasks.append((
                                "rho_grid",
                                f"{model_id}|{split}|{seed}|{rho_v}|{rho_t}",
                            ))
    else:
        raise ValueError(f"unsupported data-producing experiment kind: {kind}")
    return tasks


def emit_experiment(experiment_id: str, selected_models: list[str] | None = None) -> int:
    try:
        raw = EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise SystemExit(f"unknown data-producing experiment: {experiment_id}") from exc

    selected = set(selected_models or [])
    models = [str(item) for item in raw["MODEL_IDS"] if not selected or item in selected]
    if not models:
        raise SystemExit(f"no models selected for {experiment_id}")

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    sequence = _next_sequence()
    count = 0
    for task, model_id in _task_specs(raw, models):
        task_model = model_id or "display"
        if task == "rho_grid":
            base_model, split, seed, rho_v, rho_t = task_model.split("|")
            suffix = f"{base_model}__{split}__s{seed}__rhoV{rho_v}__rhoT{rho_t}"
        else:
            suffix = task_model
        path = QUEUE_DIR / f"{sequence:05d}__{experiment_id}__{suffix}.conf"
        lines = [
            f"# Generated by configs/z_gen/{experiment_id}.py",
            f"EXPERIMENT_ID={experiment_id}",
            f"TASK={task}",
            f"MODEL_ID={task_model.split('|')[0] if task == 'rho_grid' else task_model}" if model_id else "",
            f"GRID_SPLIT={task_model.split('|')[1]}" if task == "rho_grid" else "",
            f"GRID_SEED={task_model.split('|')[2]}" if task == "rho_grid" else "",
            f"RHO_V={task_model.split('|')[3]}" if task == "rho_grid" else "",
            f"RHO_T={task_model.split('|')[4]}" if task == "rho_grid" else "",
            # Generated tasks reuse existing checkpoints by default; the
            # runner only retrains when an explicit FORCE=true is set
            # (it skips the checkpoint/grid reuse check then).
            "FORCE=false",
        ]
        path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")
        print(f"[queue] {path}")
        sequence += 1
        count += 1
    return count


# wo05 E5 遮挡交叉评估（2026-10-24）：eval-only 12 条目（3 家族 × occ010–050），
# 复用 _drop 统一模型 ckpt_best；入队闸门由 wo05.py 波 3 按家族把控。
EXPERIMENTS["wo05_eval"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V3_Ev1_wo05_drop_evalocc010", "V3_Ev1_wo05_drop_evalocc020", "V3_Ev1_wo05_drop_evalocc030", "V3_Ev1_wo05_drop_evalocc050", "V4_Av1_wo05_drop_evalocc010", "V4_Av1_wo05_drop_evalocc020", "V4_Av1_wo05_drop_evalocc030", "V4_Av1_wo05_drop_evalocc050", "V5_Av4_wo05_drop_evalocc010", "V5_Av4_wo05_drop_evalocc020", "V5_Av4_wo05_drop_evalocc030", "V5_Av4_wo05_drop_evalocc050"],
    "REQUIRES": [],
}

# λ 扫描（2026-10-24，用户裁定「先扫描一下这几个参数点」）：V5_Av1 / V5_Av4 /
# V4_Av1 × λ_prior_pose {1,2,3,4} × λ_prior_traj {0,1} 共 24 格；cam3 训练 +
# 评估、splits_wo05.csv；纯扫描、无调参目标——每格产出三条件面板，终表由各格
# test.json 汇总（条目定义见 model_registry 同名块）。
EXPERIMENTS["prior_lambda_scan"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["%s_wo05_lp%s_lt%s" % (m, lp, lt)
                  for m in ("V5_Av1", "V5_Av4", "V4_Av1")
                  for lp in ("1", "2", "3", "4") for lt in ("0", "1")],
    "REQUIRES": [],
}

# V4_Av3 B-基座线（2026-10-25，用户裁定）：V4 修正分支挂冻结 B 骨架
# （T 解毒折中方案）×（cam3 / cam3_occ010）2 格——训练+评估一体事务
# （条目定义见 model_registry 同名块）。
EXPERIMENTS["v4b_carrier"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V4_Av3", "V4_Av3_cam3_occ010"],
    "REQUIRES": [],
}
EXPERIMENTS["v4b_carrier_last"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V4_Av3_last_eval", "V4_Av3_cam3_occ010_last_eval"],
    "REQUIRES": [],
}
# V4_Av4 双阶段精调（2026-10-25，用户裁定「冻结并非合适选择」）：
# B 骨架 0.1× 学习率微调 ×（cam3 / cam3_occ010）2 格。
EXPERIMENTS["v4b_finetune"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V4_Av4", "V4_Av4_cam3_occ010",
                  "V4_Av4_drop", "V4_Av4_drop_cam3_occ010"],
    "REQUIRES": [],
}

EXPERIMENTS["v7_precheck"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Av1_precheck", "V7_Bv1_precheck", "V7_Cv1_precheck"],
    "REQUIRES": [],
}

# V7 候选 A/B/C 第一轮（2026-10-25，模型设计思路.md §7）：三候选 ×
# （cam3 / cam3_occ010）6 格——注册但本轮不发射（用户裁定：先实现 +
# 冒烟，发射时机另定）。
# V7 端到端预检（2026-10-25 计划步骤 1）：三个预检条目（epochs=6 进 spec）。
EXPERIMENTS["v7_precheck"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Av1_precheck", "V7_Bv1_precheck", "V7_Cv1_precheck"],
    "REQUIRES": [],
}

# V7 E5 occ050 筛选（2026-10-25 计划步骤 3）：eval-only 三格（val 口径）。
# V7_Cv1 候选复评（2026-10-25）：五候选 val 复评（Δ 全指标 + 部位分解）。
EXPERIMENTS["v7_cand_review"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Cv1_cand600", "V7_Cv1_cand665", "V7_Cv1_cand670",
                  "V7_Cv1_cand680", "V7_Cv1_cand740"],
    "REQUIRES": [],
}

EXPERIMENTS["v7_e5_screen"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Av1_evalocc010", "V7_Bv1_evalocc010", "V7_Cv1_evalocc010",
                  "V7_Av1_evalocc020", "V7_Bv1_evalocc020", "V7_Cv1_evalocc020",
                  "V7_Av1_evalocc030", "V7_Bv1_evalocc030", "V7_Cv1_evalocc030",
                  "V7_Av1_evalocc050", "V7_Bv1_evalocc050", "V7_Cv1_evalocc050"],
    "REQUIRES": [],
}

EXPERIMENTS["v7_first_round"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Av1", "V7_Av1_cam3_occ010",
                  "V7_Bv1", "V7_Bv1_cam3_occ010",
                  "V7_Cv1", "V7_Cv1_cam3_occ010"],
    "REQUIRES": [],
}

# 条件合作线（2026-10-25，用户裁定）：V5_Av5 / V4_Av2 ×（契约/去契约
# 对照）×（cam3 / cam3_occ010）8 格——训练+评估一体事务（条目定义见
# model_registry 同名块）。
EXPERIMENTS["cond_coop"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V5_Av5", "V5_Av5_nocond", "V4_Av2", "V4_Av2_nocond",
                  "V5_Av5_cam3_occ010", "V5_Av5_nocond_cam3_occ010",
                  "V4_Av2_cam3_occ010", "V4_Av2_nocond_cam3_occ010"],
    "REQUIRES": [],
}

# ckpt_last 重读（2026-10-24，用户裁定）：选择假象检查——V4_Av1_wo05_drop
# 的 ckpt_last 跑 E5 occ 四档 + cam3；V5_Av1_fS13S14 的 ckpt_last cam3。
# V4_Av1/V5_Av1 其余 fold 格（5-fold 按动作表修正口径）。eval-only 零重训
# （条目定义见 model_registry 同名块）。
EXPERIMENTS["ckpt_last_reread"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V4_Av1_wo05_drop_last_evalcam3",
                  "V4_Av1_wo05_drop_last_evalocc010",
                  "V4_Av1_wo05_drop_last_evalocc020",
                  "V4_Av1_wo05_drop_last_evalocc030",
                  "V4_Av1_wo05_drop_last_evalocc050",
                  "V5_Av1_fS13S14_last_evalcam3"]
    + ["V5_Av1_%s_last_evalcam3" % f
       for f in ("fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")]
    + ["V4_Av1_%s_last_evalcam3" % f
       for f in ("fS13S14", "fS5S6S11", "fS5S11S12", "fS6S7S13", "fS5S6S8")],
    "REQUIRES": [],
}

# ckpt_last 全量重定基线（2026-10-24，用户裁定）：三批 eval-only 零重训，
# 条目定义见 model_registry 同名块。顺序 = E1 8 → tw_scan 28 → λ 扫描 18。
EXPERIMENTS["ckpt_last_rebaseline"] = {
    "KIND": "model_suite",
    "MODEL_IDS": (
        ["%s_last_eval" % m for m in (
            "V5_Av1_cam3_v3off", "V5_Av1_cam3_v3off_drop",
            "V5_Av1_cam3_v3off_vonly", "V5_Av1_cam3_v3off_tonly",
            "V4_Av1_cam3_v3off", "V4_Av1_cam3_v3off_drop",
            "V4_Av1_cam3_v3off_vonly", "V4_Av1_cam3_v3off_tonly")]
        + ["%s_%s_v3off_tw%d_last_eval" % (f, ax, tw)
           for f in ("V3_Bv1", "V4_Av1", "V5_Av1", "V5_Av4")
           for tw in (5, 10, 15) for ax in ("cam3", "cam3_occ010")]
        + ["%s_cam3_occ010_v3off_last_eval" % f
           for f in ("V3_Bv1", "V4_Av1", "V5_Av1", "V5_Av4")]
        + ["%s_wo05_lp%s_lt%s_last_eval" % (m, lp, lt)
           for m in ("V5_Av1", "V5_Av4", "V4_Av1")
           for lp in ("1", "2", "3", "4") for lt in ("0", "1")
           if not (m == "V5_Av4" and lp != "1")]),
    "REQUIRES": [],
}
