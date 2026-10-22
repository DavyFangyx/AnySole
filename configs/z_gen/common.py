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
    "all_models": {
        "KIND": "model_suite",
        "MODEL_IDS": list(MODELS),
        "REQUIRES": [],
    },
    "singlemodal_eval": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B"],
        "SPECIALISTS": ["vonly", "tonly"],
        "REQUIRES": [],
    },
    # 交接包降噪压力重训（2026-10-17）：V3_3B 同结构，tactile_input=
    # denoised108（v3_3b_denoised 入口固定）。只训主线——探针实验读主线
    # 三条件，不需要专才；T 流换了数据源，旧 V3_3B_tonly 不再等价。
    "singlemodal_eval_denoised": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B_denoised"],
        "SPECIALISTS": [],
        "REQUIRES": [],
    },
    # 降噪压力 E/F（2026-10-17）：同协议，主线 only。
    "singlemodal_eval_denoised_ef": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3E_denoised", "V3_3F_denoised"],
        "SPECIALISTS": [],
        "REQUIRES": [],
    },
    # 四相机扩展 D2 (2026-10-12): same protocol on the three new cameras.
    # tonly is deliberately absent -- the T stream is camera-independent and
    # the shared V3_3B_tonly specialist serves all four cameras.
    "singlemodal_eval_cams": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B_cam1", "V3_3B_cam2", "V3_3B_cam4"],
        "SPECIALISTS": ["vonly"],
        "REQUIRES": [],
    },
    # V-degradation 遮挡组 (2026-10-13): cam3 with the bottom fraction of the
    # person bbox occluded.  tonly is absent for the same reason as the camera
    # group -- the T stream is occlusion-independent, V3_3B_tonly serves all.
    "singlemodal_eval_occl": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B_cam3_occ010", "V3_3B_cam3_occ020",
                      "V3_3B_cam3_occ030", "V3_3B_cam3_occ050"],
        "SPECIALISTS": ["vonly"],
        "REQUIRES": [],
    },
    "rho_grid_eval": {
        "KIND": "rho_grid",
        "MODEL_IDS": ["V3_3B"],
        "REQUIRES": [],
        "RHOS": "0,20,40,60,80,100",
        "TRAIN_REPR_RHOS": "0,100",
        "VAL_SEEDS": "0,1,2",
        "TEST_SEEDS": "0",
    },
    "residual_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3C"],
        "REQUIRES": [],
    },
    "position_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3D"],
        "REQUIRES": [],
    },
    # V3_3D 削弱套件（2026-10-15）：8 配置 ×（训练 + val + test）事务；
    # cam3 基准行 = position_test/V3_3D。每配置自训自测，与 V3_3B 套件同协议。
    # V3_3B occ010 生产评估补跑（2026-10-18）：ckpt 已训（10-13），
    # metrics 缺失——只补评估。
    "v3b_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3E 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3e_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3E", "V3_3E_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3F 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3f_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3F", "V3_3F_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3G 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3g_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3G", "V3_3G_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3H 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    "v3h_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H", "V3_3H_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3H_v2 首组（2026-10-19）：命题 2 载体依赖修复（T 流 aux 重建）。
    "v3h2_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H_v2", "V3_3H_v2_cam3_occ010"],
        "REQUIRES": [],
    },
    # V3_3H_v3 首组（2026-10-19）：v2 + 一阶足部残差头（命题 2 兑现通道）。
    "v3h3_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H_v3", "V3_3H_v3_cam3_occ010"],
        "REQUIRES": [],
    },
    # offset 重同步组（2026-10-20）：cam3 基准 + occ010/020/030/050，
    # 自训自测（corrected_offset=True）。
    "offsetfix": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_offfix",
                      "V3_3B_cam3_occ010_offfix", "V3_3B_cam3_occ020_offfix",
                      "V3_3B_cam3_occ030_offfix", "V3_3B_cam3_occ050_offfix"],
        "REQUIRES": [],
    },
    # offset 设计组（2026-10-21）：载体先行——E_offfix（V5_1/V4_1 载体）
    # 与 vonly 专才（V3_3H 载体）；设计主体在载体完成后另批入队。
    "offsetfix_e": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3E_cam3_offfix", "V3_3E_cam3_occ010_offfix"],
        "REQUIRES": [],
    },
    "offsetfix_vonly": {
        "KIND": "singlemodal_compare",
        "MODEL_IDS": ["V3_3B_cam3_offfix", "V3_3B_cam3_occ010_offfix"],
        "SPECIALISTS": ["vonly"],
        "REQUIRES": [],
    },
    "offsetfix_v5_1": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_1_cam3_offfix", "V5_1_cam3_occ010_offfix",
                      "V5_1_cam3_occ020_offfix", "V5_1_cam3_occ030_offfix",
                      "V5_1_cam3_occ050_offfix"],
        "REQUIRES": [],
    },
    "offsetfix_v4_1": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V4_1_cam3_offfix", "V4_1_cam3_occ010_offfix"],
        "REQUIRES": [],
    },
    # v2 + offset 设计组（2026-10-21）：V5_1A / V4_1A × cam3/cam3_occ010，
    # 触觉 = denoised108_v2（corrected_offset=True、载体 = E_offfix 系）。
    "offsetfix_v45_v2": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_1_cam3_offfix_v2", "V5_1_cam3_occ010_offfix_v2",
                      "V4_1_cam3_offfix_v2", "V4_1_cam3_occ010_offfix_v2"],
        "REQUIRES": [],
    },
    "offsetfix_v3_3h": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H_cam3_offfix", "V3_3H_cam3_occ010_offfix"],
        "REQUIRES": [],
    },
    # 新数据验证（2026-10-22）：生产 v3+offset facts 的替换验证。
    "v3off_validation": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_v3off"],
        "REQUIRES": [],
    },
    # v2/v3 + offset 组（2026-10-20）：同 5 组合，触觉 = denoised108_v2 / _v3。
    "offsetfix_v2": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_offfix_v2",
                      "V3_3B_cam3_occ010_offfix_v2", "V3_3B_cam3_occ020_offfix_v2",
                      "V3_3B_cam3_occ030_offfix_v2", "V3_3B_cam3_occ050_offfix_v2"],
        "REQUIRES": [],
    },
    "offsetfix_v3": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_offfix_v3",
                      "V3_3B_cam3_occ010_offfix_v3", "V3_3B_cam3_occ020_offfix_v3",
                      "V3_3B_cam3_occ030_offfix_v3", "V3_3B_cam3_occ050_offfix_v3"],
        "REQUIRES": [],
    },
    # V3_3H_v5 无锚版全网格（2026-10-20）：4 相机 × 3 遮挡 = 12 格
    # （T 价值区 = V 失效条件的无锚对照；cam{1,2,4}_occ{010,020} 的
    # 遮挡缓存由 extract_hrnet 生成）。
    "v3h5_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H_v5", "V3_3H_v5_cam3_occ010", "V3_3H_v5_cam3_occ020",
                      "V3_3H_v5_cam1", "V3_3H_v5_cam1_occ010", "V3_3H_v5_cam1_occ020",
                      "V3_3H_v5_cam2", "V3_3H_v5_cam2_occ010", "V3_3H_v5_cam2_occ020",
                      "V3_3H_v5_cam4", "V3_3H_v5_cam4_occ010", "V3_3H_v5_cam4_occ020"],
        "REQUIRES": [],
    },
    # V3_3H_v6 首组（2026-10-20）：v5 + 残差头落点改踝（错误一修复），
    # cam3 基准 + cam3+occ010，自训自测。
    "v3h6_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3H_v6", "V3_3H_v6_cam3_occ010"],
        "REQUIRES": [],
    },
    # 新数据验证（2026-10-22，主会话 conf 00618）：生产 v3+offset facts
    # 替换验证——V3_3B_cam3_v3off = E1 的 3B 统一模型行。
    "v3off_validation": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3B_cam3_v3off"],
        "REQUIRES": [],
    },
    # E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型，
    # 生产 v3+offset facts。波 1 = E 家族（新载体）+ 3B 家族；波 2 = V4_1/
    # V5_1 家族（carrier 依赖波 1 的 E 基名 ckpt）。
    "e1_experts": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3E_cam3_v3off", "V3_3E_cam3_v3off_drop",
                      "V3_3E_cam3_v3off_vonly", "V3_3E_cam3_v3off_tonly",
                      "V3_3B_cam3_v3off_vonly", "V3_3B_cam3_v3off_tonly",
                      "V3_3B_cam3_v3off_nodrop",
                      "V4_1_cam3_v3off", "V4_1_cam3_v3off_drop",
                      "V4_1_cam3_v3off_vonly", "V4_1_cam3_v3off_tonly",
                      "V5_1_cam3_v3off", "V5_1_cam3_v3off_drop",
                      "V5_1_cam3_v3off_vonly", "V5_1_cam3_v3off_tonly"],
        "REQUIRES": [],
    },
    # T 增强三组（2026-10-21，T增强三组实验方案.md）：维度 2 显式物理特征
    # （tphys）+ 维度 3 足-地监督（d8，监督头落地后加入）。
    "v5_1_tenhance_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_1_cam3_offfix_tphys", "V5_1_cam3_offfix_d8"],
        "REQUIRES": [],
    },
    # V5_1 首组（2026-10-19，V5_1A方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。
    "v5_1_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V5_1", "V5_1_cam3_occ010", "V5_1_occmix_cam3_occ010"],
        "REQUIRES": [],
    },
    # V4_1 首组（2026-10-19，V4_1A方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。V6_1A occ_mix
    # 为可选增强开关（conf 直写 OCC_MIX=true/P_OCC/P_NULLV/P_NULLT），
    # 默认关——不为 V6_1A 单独立项训练 run。
    "v4_1_test": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V4_1", "V4_1_cam3_occ010", "V4_1_occmix_cam3_occ010"],
        "REQUIRES": [],
    },
    "position_test_vdegrad": {
        "KIND": "model_suite",
        "MODEL_IDS": ["V3_3D_cam1", "V3_3D_cam2", "V3_3D_cam4",
                      "V3_3D_cam3_occ010", "V3_3D_cam3_occ020",
                      "V3_3D_cam3_occ030", "V3_3D_cam3_occ050"],
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
