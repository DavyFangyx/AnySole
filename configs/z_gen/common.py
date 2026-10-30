"""Shared registry and queue writer for the offline experiments."""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
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
    # V3_Dv1 削弱套件（2026-10-15）：8 配置 ×（训练 + val + test）事务；
    # cam3 基准行 = position_test/V3_Dv1。每配置自训自测，与 V3_Bv1 套件同协议。
    # V3_Bv1 occ010 生产评估补跑（2026-10-18）：ckpt 已训（10-13），
    # metrics 缺失——只补评估。
    # V3_Ev1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    # V3_Fv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    # V3_Gv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    # V3_Hv1 首组（2026-10-17）：cam3 基准 + cam3+occ010，自训自测。
    # V3_Hv2 首组（2026-10-19）：命题 2 载体依赖修复（T 流 aux 重建）。
    # V3_Hv3 首组（2026-10-19）：v2 + 一阶足部残差头（命题 2 兑现通道）。
    # 新数据验证（2026-10-22）：生产 v3+offset facts 的替换验证。
    # V3_Hv5 无锚版全网格（2026-10-20）：4 相机 × 3 遮挡 = 12 格
    # （T 价值区 = V 失效条件的无锚对照；cam{1,2,4}_occ{010,020} 的
    # 遮挡缓存由 extract_hrnet 生成）。
    # V3_Hv6 首组（2026-10-20）：v5 + 残差头落点改踝（错误一修复），
    # cam3 基准 + cam3+occ010，自训自测。
    # 新数据验证（2026-10-22，主会话 conf 00618）：生产 v3+offset facts
    # 替换验证——V3_Bv1_cam3_v3off = E1 的 3B 统一模型行。
    # E1 同架构专家对比（2026-10-22，z_note/实验计划.md）：四家族 x 四模型，
    # 生产 v3+offset facts。波 1 = E 家族（新载体）+ 3B 家族；波 2 = V4_Av1/
    # V5_Av1 家族（carrier 依赖波 1 的 E 基名 ckpt）。
    # E1 修复迭代（2026-10-22）：V5_Av4 四变体（从零 V 流 + 先验塔 E 配方），
    # v3off 生产轴，论文正文模型候选。
    # wo05 实验轴（2026-10-24）：canonical split 去掉动作 05（绕圈跑）的
    # 12 模型（3 家族 x E1 四变体，splits_wo05.csv；train 92->88，val/test
    # 不变）。波 1 = V3_Ev1_wo05 + V5_Av4_wo05（V5_Av4 carrier=scratch 无
    # E 依赖）；波 2 = V4_Av1_wo05（carrier 依赖波 1 的 V3_Ev1_wo05
    # ckpt_best）。MODEL_IDS 顺序即 conf 顺序：波 1 排最前（worker 先取件）。
    # 时间窗扫描（2026-10-22，z_note/multimodal_fusion/时间窗扫描实验.md）：
    # 五家族 x tw{5,10,15,20} x cam3/cam3_occ010（v3off 生产轴）。
    # 波 1 = E + V3_Bv1 + V5_Av4；波 2 = V4_Av1 + V5_Av1
    # （闸门 = 波 1 的 E 格完成——同 (tw, 视觉) 的 E 载体就绪后才可训）。
    # MODEL_IDS 顺序即 conf 顺序：波 1 的 E 格排最前（worker 先取件）。
    # tw × 遮挡交互（2026-10-24）：occ020/030 × tw{5,10,15,20}（v3off 轴），
    # V3_Bv1 / V5_Av4 两模型。参照 = tw_scan 的 cam3 与 occ010 腿。
    # T 增强三组（2026-10-21，T增强三组实验方案.md）：维度 2 显式物理特征
    # （tphys）+ 维度 3 足-地监督（d8，监督头落地后加入）。
    # V5_Av1 首组（2026-10-19，V5_Av1方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。
    # V4_Av1 首组（2026-10-19，V4_Av1方案）：cam3 基准 + cam3+occ010，
    # 自训自测（载体按配置装载该配置的 E v2.1 ckpt_best）。V6_Av1 occ_mix
    # 为可选增强开关（conf 直写 OCC_MIX=true/P_OCC/P_NULLV/P_NULLT），
    # 默认关——不为 V6_Av1 单独立项训练 run。
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
    else:
        raise ValueError(f"unsupported data-producing experiment kind: {kind}")
    return tasks


@lru_cache(maxsize=1)
def _cached_registry() -> dict[str, Any]:
    from configs.tools.common import build_registry
    return build_registry()


def results_complete(model_id: str) -> bool:
    """该 cell（模型×条件）的结果目录是否已有 complete 产物（生成期跳过用）。

    conf 与结果目录一一对应（results/cells/<model_id>/，2026-10-26 裁定）；
    只查 run_manifest.status——ckpt 是否变化由 runner 执行级跳过复核
    （重训后旧结果的 cell 须 --force 重发）。
    """
    try:
        registry = _cached_registry()
        from configs.tools.common import cell_root
        manifest = cell_root(registry, model_id) / "run_manifest.json"
        if not manifest.is_file():
            return False
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return data.get("status") == "complete"
    except Exception:
        return False


def queued_model_ids() -> set[str]:
    """queue/running/done/failed 任一状态已有同名 conf 的 cell（模型）名。

    conf 命名 = {seq}__{model_id}.conf（cell 即命名，2026-10-26 裁定）；
    同时识别旧命名 {seq}__{experiment}__{model}.conf（历史 conf 仍可跳过）。
    """
    names: set[str] = set()
    for state in STATES:
        for path in (CONFIG_DIR / state).glob("*.conf"):
            base = path.name[: -len(".conf")]
            names.add(base.split("__", 1)[-1])
            names.add(base.rsplit("__", 1)[-1])
    return names


def emit_experiment(experiment_id: str, selected_models: list[str] | None = None, *,
                    wandb: bool | None = None,
                    checkpoint_all: bool | None = None,
                    force: bool | None = None,
                    dry_run: bool = False,
                    skip: bool = True) -> int:
    """把 EXPERIMENTS 条目展开成 queue conf（统一生成器 gen.py 的引擎）。

    wandb/checkpoint_all/force 为 None 时取 EXPERIMENTS 级默认
    （缺省 false/false/false）；dry_run 只打印不落盘。

    skip=True（默认，2026-10-26 用户裁定「conf 与结果一一对应」）：跳过
    queue/running/done/failed 已有同名 conf、或结果目录已有 complete 产物
    的模型——先做 E5 后做 E2 时，E2 生成器发现共用格已在队伍/已有结果即
    不生成。force=true 绕过 skip 并写 FORCE=true（强制重跑）。
    """
    try:
        raw = EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise SystemExit(f"unknown data-producing experiment: {experiment_id}") from exc

    selected = set(selected_models or [])
    models = [str(item) for item in raw["MODEL_IDS"] if not selected or item in selected]
    if not models:
        raise SystemExit(f"no models selected for {experiment_id}")

    if skip and not bool(force):
        queued = queued_model_ids()
        kept: list[str] = []
        for item in models:
            if item in queued:
                print("[skip] %s: conf 已在 queue/running/done/failed" % item)
            elif results_complete(item):
                print("[skip] %s: 结果目录已有 complete 产物" % item)
            else:
                kept.append(item)
        models = kept
        if not models:
            print("[skip] %s: nothing to emit（--force 强制重发）" % experiment_id)
            return 0

    wandb_line = "WANDB=%s" % str(
        raw.get("WANDB", False) if wandb is None else wandb).lower()
    ckpt_line = "CHECKPOINT_ALL=%s" % str(
        raw.get("CHECKPOINT_ALL", False) if checkpoint_all is None else checkpoint_all).lower()
    force_line = "FORCE=%s" % str(False if force is None else force).lower()

    sequence = _next_sequence()
    count = 0
    for task, model_id in _task_specs(raw, models):
        task_model = model_id or "display"
        suffix = task_model
        path = QUEUE_DIR / f"{sequence:05d}__{suffix}.conf"
        lines = [
            f"# Generated by configs/z_gen/gen.py {experiment_id}",
            f"EXPERIMENT_ID={experiment_id}",
            f"TASK={task}",
            f"MODEL_ID={task_model}" if model_id else "",
            # Generated tasks reuse existing checkpoints by default; the
            # runner only retrains when an explicit FORCE=true is set
            # (it skips the checkpoint/grid reuse check then).
            force_line,
            # wandb 开关（用户裁定 2026-10-25）：每个 conf 恒带一行，
            # 实验级默认取 EXPERIMENTS[].WANDB（缺省 false）；true 时
            # runner 注入 wandb_mode=online 覆盖 train_args。
            wandb_line,
            # checkpoint_all 开关（2026-10-26 裁定）：true=候选全留协议；
            # false=只滚存最近过线者。实验级默认取 EXPERIMENTS[].CHECKPOINT_ALL。
            ckpt_line,
        ]
        content = "\n".join(line for line in lines if line) + "\n"
        if dry_run:
            print(f"[dry-run] {path}")
            print(content, end="")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"[queue] {path}")
        sequence += 1
        count += 1
    return count


def parse_seeds(value: str | None) -> list[int] | None:
    if value is None:
        return [1]  # 默认 seed 1（用户裁定：默认就是 seed1，cell 带 _s1）
    seeds = [int(x) for x in value.split(",") if x.strip()]
    if not seeds or any(s < 1 for s in seeds):
        raise SystemExit("--seed 需逗号分隔的正整数，如 1,2,3")
    return seeds


def parse_bool(value: str | None, flag: str) -> bool | None:
    """true/false 命令行参数解析（gen.py 与 gen_*.py 共用）。"""
    if value is None:
        return None
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise SystemExit("%s must be true/false, got %r" % (flag, value))


# 接触标签版本 → 注册表条目后缀（2026-10-29 用户裁定：--concat-method 即选接触标签）
CONCAT_VERSIONS = {"f6_soft": "",
                   "f6_soft_time_accel_v1": "_tafix",
                   "f6_soft_time_accel_state_v2": "_statefix"}


def concat_models(models: list[str], method: str) -> list[str]:
    """按接触标签版本映射模型 id：f6_soft 原样；其余版本在后缀位置插入
    _tafix/_statefix（注册表需存在对应条目，否则报错）。"""
    suffix = CONCAT_VERSIONS[method]
    if not suffix:
        return models
    all_suffixes = [v for v in CONCAT_VERSIONS.values() if v]
    mapped = []
    for model in models:
        if model.endswith(suffix) or any(model.endswith(v) for v in all_suffixes):
            # 已带版本后缀（含其他版本）的 id 显式编码了标签版本，不叠加
            mapped.append(model)
            continue
        target = model + suffix
        if target not in MODELS:
            # 子后缀（_stat/_evalocc 等）前插入版本后缀
            for sub in ("_statistics", "_evalocc", "_evalbest", "_precheck", "_sel", "_drop",
                        "_vonly", "_tonly", "_nodrop", "_smpl24"):
                if sub in model:
                    target = model.replace(sub, suffix + sub, 1)
                    break
        if target not in MODELS:
            raise SystemExit("--concat-method %s：%s 无对应注册条目（%s）"
                             % (method, model, target))
        mapped.append(target)
    return mapped


def ckpt_models(models: list[str], ckpt: str) -> list[str]:
    """ckpt 口径映射（--ckpt last/best）：best 时把 _stat/_evalocc 系 id
    映射到 _evalbest/_evalbest_occ 系（2026-10-29 注册的 Cv1–7 条目）；
    last 原样。已含 _evalbest 的 id 不重复映射。"""
    if ckpt == "last":
        return models
    mapped = []
    for model in models:
        if "_evalbest" in model:
            mapped.append(model)
            continue
        # 尾部精确匹配（防 base 名里的 stat/occ 字样被误替换，如 statefix）
        target = None
        pos = model.rfind("_evalocc")
        if pos != -1 and model[pos + len("_evalocc"):].isdigit() \
                and len(model) - (pos + len("_evalocc")) == 3:
            # 尾部 _evaloccNNN / _statistics_evaloccNNN → _evalbest_occNNN
            base = model[:pos]
            if base.endswith("_statistics"):
                base = base[: -len("_statistics")]
            target = base + "_evalbest_occ" + model[pos + len("_evalocc"):]
        elif model.endswith("_statistics"):
            target = model[: -len("_statistics")] + "_evalbest"
        else:
            target = model + "_evalbest"
        if target not in MODELS:
            raise SystemExit("--ckpt best：%s 无对应 best 口径条目（%s）"
                             % (model, target))
        mapped.append(target)
    return mapped


def seed_models(models: list[str], seeds: list[int]) -> list[str]:
    """多种子展开（方案 A，2026-10-29）：seed 1 = _s1 显式（用户裁定「默认
    就是 seed1，名字和目录都要带 _s1」）；seed ≥2 = 追加 _sN。子后缀插入
    （V7_Cv5_evalocc010 → V7_Cv5_s2_evalocc010）；已带 _sN 的 id 原样保留。"""
    expanded = []
    for model in models:
        if any(model.endswith(s) for s in ("_s2", "_s3")):
            expanded.append(model)
            continue
        # 子后缀位置（_sN 插在子后缀前：V7_Cv5_evalocc010 → V7_Cv5_s2_evalocc010）
        sub_pos = None
        for sub in ("_evalocc", "_evalbest", "_precheck", "_sel"):
            pos = model.rfind(sub)
            if pos != -1 and pos > 0:
                sub_pos = pos
                break
        for n in seeds:
            if n == 1:
                if re.search(r"_s\d+($|_)", model):
                    target = model
                elif sub_pos is not None:
                    target = model[:sub_pos] + "_s1" + model[sub_pos:]
                else:
                    target = "%s_s1" % model
            elif sub_pos is not None:
                target = model[:sub_pos] + "_s%d" % n + model[sub_pos:]
            else:
                target = "%s_s%d" % (model, n)
            if target not in MODELS:
                raise SystemExit("--seed %d：%s 无对应注册条目（%s）"
                                 % (n, model, target))
            expanded.append(target)
    return expanded


def add_gen_args(parser) -> None:
    """gen_*.py 公共参数：模型选择与 conf 覆盖开关。"""
    parser.add_argument("--models", default=None,
                        help="comma-separated model IDs（默认：脚本内 DEFAULT_MODELS）")
    parser.add_argument("--seed", default=None,
                        help="多种子（逗号分隔，如 1,2,3；缺省 1；seed 1 也显式 _s1）")
    parser.add_argument("--concat-method", choices=list(CONCAT_VERSIONS),
                        default="f6_soft",
                        help="接触标签版本（默认 f6_soft；tafix/statefix 映射到 *_tafix/*_statefix 条目）")
    parser.add_argument("--wandb", default=None, help="true/false，覆盖实验级默认")
    parser.add_argument("--checkpoint-all", default=None, help="true/false，覆盖实验级默认")
    parser.add_argument("--force", action="store_true",
                        help="绕过生成期 skip 并写 FORCE=true（强制重跑，含重训）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将写入的 conf，不落盘")


DONE_DIR = CONFIG_DIR / "done"


def _wave_key(raw: dict[str, Any], key: str, table: str) -> Any:
    """按 str 匹配 WAVES/WAVE_GATES 的键（兼容 int 键 1/2 与 str 键 "12"/"3"）。"""
    for candidate in raw.get(table, {}):
        if str(candidate) == str(key):
            return candidate
    raise SystemExit("unknown wave %r for %s (waves: %s)"
                     % (key, raw.get("_experiment_id"),
                        ",".join(str(k) for k in sorted(raw.get("WAVES", {}), key=str))))


def _check_wave_gates(raw: dict[str, Any], wave: Any, hard: bool) -> bool:
    gates = raw.get("WAVE_GATES", {})
    patterns = [gates[k] for k in gates if str(k) == str(wave)]
    patterns = [p for group in patterns for p in group]
    missing = [pattern for pattern in patterns if not list(DONE_DIR.glob(pattern))]
    if not missing:
        return True
    if hard:
        raise SystemExit("波 %s 闸门未满足：%s 无 conf 进 done/"
                         % (wave, "、".join(missing)))
    for pattern in missing:
        print("[gate] wave %s skipped: %s 未进 done/（完成后用 --wave %s 补发）"
              % (wave, pattern, wave))
    return False


def _emit_wave(raw: dict[str, Any], wave: Any, selected: list[str],
               wandb: bool | None, checkpoint_all: bool | None,
               force: bool, dry_run: bool) -> int:
    """发射一个训练波：闸门（硬）→ --models 子集 → emit（skip 默认开）。"""
    _check_wave_gates(raw, wave, hard=True)
    selected_set = set(selected)
    wave_list = raw["WAVES"][_wave_key(raw, wave, "WAVES")]
    models = [str(item) for item in wave_list
              if not selected_set or str(item) in selected_set]
    if not models:
        print("[skip] wave %s: no selected model in this wave" % wave)
        return 0
    return emit_experiment(str(raw["_experiment_id"]), models,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def _emit_eval_wave(raw: dict[str, Any], hard: bool,
                    wandb: bool | None, checkpoint_all: bool | None,
                    force: bool, dry_run: bool) -> int:
    """发射 WAVE_EVAL 波（wo05 波 3）：按家族闸门筛 ready 家族后发 eval 条目。"""
    eval_spec = raw["WAVE_EVAL"]
    eval_id = str(eval_spec["experiment"])
    ready: list[str] = []
    for fam, base in eval_spec["families"].items():
        if list(DONE_DIR.glob("*__%s.conf" % base)):
            ready.extend(str(item) for item in EXPERIMENTS[eval_id]["MODEL_IDS"]
                         if str(item).startswith(fam))
        else:
            print("[gate] wave %s skipped: %s 未进 done/" % (eval_spec["wave"], base))
    if not ready:
        if hard:
            raise SystemExit("波 %s 闸门未满足：没有任何家族 _drop conf 进 done/"
                             % eval_spec["wave"])
        return 0
    return emit_experiment(eval_id, ready,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def emit_waved(experiment_id: str, wave: str | None = None,
               models: list[str] | None = None, *,
               wandb: bool | None = None,
               checkpoint_all: bool | None = None,
               force: bool | None = None,
               dry_run: bool = False) -> int:
    """波次感知发射（gen.py 与 gen_*.py 共用）。

    - 无 WAVES 的条目：直接 emit_experiment（models 为全集或子集）
    - 有 WAVES：wave 缺省取 DEFAULT_WAVE；auto = 级联（逐波软闸门）；
      显式 wave = 硬闸门；WAVE_EVAL 波（wo05 波 3）按家族闸门发射
    """
    raw = dict(EXPERIMENTS[experiment_id])
    raw["_experiment_id"] = experiment_id
    waves = raw.get("WAVES")
    wave_eval = raw.get("WAVE_EVAL")

    if waves:
        if wave is None:
            wave = str(raw.get("DEFAULT_WAVE", sorted(waves, key=str)[0]))
        if wave == "auto":
            total = 0
            for w in sorted(waves, key=str):
                if _check_wave_gates(raw, w, hard=False):
                    total += _emit_wave(raw, w, models or [], wandb,
                                        checkpoint_all, bool(force), dry_run)
            if wave_eval:
                total += _emit_eval_wave(raw, hard=False, wandb=wandb,
                                         checkpoint_all=checkpoint_all,
                                         force=bool(force), dry_run=dry_run)
            return total
        if str(wave) in {str(k) for k in waves}:
            total = _emit_wave(raw, wave, models or [], wandb, checkpoint_all,
                               bool(force), dry_run)
            if wave_eval and str(wave_eval.get("wave")) == str(wave):
                total += _emit_eval_wave(raw, hard=True, wandb=wandb,
                                         checkpoint_all=checkpoint_all,
                                         force=bool(force), dry_run=dry_run)
            return total
        if wave_eval and str(wave_eval.get("wave")) == str(wave):
            return _emit_eval_wave(raw, hard=True, wandb=wandb,
                                   checkpoint_all=checkpoint_all,
                                   force=bool(force), dry_run=dry_run)
        raise SystemExit("unknown wave %r for %s (waves: %s)"
                         % (wave, experiment_id,
                            ",".join(str(k) for k in sorted(waves, key=str))))

    return emit_experiment(experiment_id, models,
                           wandb=wandb, checkpoint_all=checkpoint_all,
                           force=force, dry_run=dry_run)


def emit_subset(experiment_id: str, models: list[str], **kwargs) -> int:
    """emit_experiment 的选集版：models 与该条目 MODEL_IDS 无交集时跳过。

    gen_*.py 用——一个脚本发多个底层条目、共用一个 --models 列表时，
    不属于本条目的模型静默跳过（如 gen_full_package 的 V7_Cv5_SV1 不进
    v7_variants_clean）。models 为空 = 发该条目全部 MODEL_IDS。
    """
    if not models:
        return emit_experiment(experiment_id, None, **kwargs)
    allowed = set(str(item) for item in EXPERIMENTS[experiment_id]["MODEL_IDS"])
    subset = [str(item) for item in models if str(item) in allowed]
    if not subset:
        print("[skip] %s: no selected model belongs to this entry" % experiment_id)
        return 0
    return emit_experiment(experiment_id, subset, **kwargs)


# wo05 E5 遮挡交叉评估（2026-10-24）：eval-only 12 条目（3 家族 × occ010–050），
# 复用 _drop 统一模型 ckpt_best；入队闸门由 gen.py wo05 --wave 3 按家族把控。
# λ 扫描（2026-10-24，用户裁定「先扫描一下这几个参数点」）：V5_Av1 / V5_Av4 /
# V4_Av1 × λ_prior_pose {1,2,3,4} × λ_prior_traj {0,1} 共 24 格；cam3 训练 +
# 评估、splits_wo05.csv；纯扫描、无调参目标——每格产出三条件面板，终表由各格
# test.json 汇总（条目定义见 model_registry 同名块）。
# V4_Av3 B-基座线（2026-10-25，用户裁定）：V4 修正分支挂冻结 B 骨架
# （T 解毒折中方案）×（cam3 / cam3_occ010）2 格——训练+评估一体事务
# （条目定义见 model_registry 同名块）。
# V4_Av4 双阶段精调（2026-10-25，用户裁定「冻结并非合适选择」）：
# B 骨架 0.1× 学习率微调 ×（cam3 / cam3_occ010）2 格。
# V7 候选 A/B/C 第一轮（2026-10-25，模型设计思路.md §7）：三候选 ×
# （cam3 / cam3_occ010）6 格——注册但本轮不发射（用户裁定：先实现 +
# 冒烟，发射时机另定）。
# V7 端到端预检（2026-10-25 计划步骤 1）：三个预检条目（epochs=6 进 spec）。
# V7 E5 occ050 筛选（2026-10-25 计划步骤 3）：eval-only 三格（val 口径）。
# V7_Cv1 候选复评（2026-10-25）：五候选 val 复评（Δ 全指标 + 部位分解）。
# ckpt 口径对照（2026-10-26 用户裁定）：E2/E5 对照补 best/last 双口径行。
# last occ 12 格（重跑批次权重）+ best cam3 3 格 + best occ 12 格，
# eval-only 零重训（条目定义见 model_registry 同名块）。
# 第二轮 E2/E5 评估（2026-10-26）：固定选点权重（四门规则）+ occ 四档。
# V7 E1 专家（2026-10-26）：S_V/S_T 两格训练。
# V1–V3 统计（2026-10-26）：cam3 test 面板 + occ 四档交叉评估（15 格）。
# V7_Cv5 全量套餐专家（2026-10-26）。
# 专家 test 补跑（2026-10-26）。
# 条件合作线（2026-10-25，用户裁定）：V5_Av5 / V4_Av2 ×（契约/去契约
# 对照）×（cam3 / cam3_occ010）8 格——训练+评估一体事务（条目定义见
# model_registry 同名块）。
# ckpt_last 重读（2026-10-24，用户裁定）：选择假象检查——V4_Av1_wo05_drop
# 的 ckpt_last 跑 E5 occ 四档 + cam3；V5_Av1_fS13S14 的 ckpt_last cam3。
# V4_Av1/V5_Av1 其余 fold 格（5-fold 按动作表修正口径）。eval-only 零重训
# （条目定义见 model_registry 同名块）。
# fold 系条目已随 per_action_folds 删除（2026-10-29）；仅留 wo05 系重读。
# ckpt_last 全量重定基线（2026-10-24，用户裁定）：三批 eval-only 零重训，
# 条目定义见 model_registry 同名块。顺序 = E1 8 → tw_scan 28 → λ 扫描 18。
# V7_Cv1 第一轮调参 18 格（2026-10-26 用户裁定）：lr × λ_contact ×
# lr_schedule 全网格，λ_pose 3 / λ_traj 1 / λ_air 2 固定（条目定义见
# model_registry 同名块）。每格 train+val+test 走 runner 标准事务，
# multi_metric 候选池随基座继承（checkpoint-all = 三门通过者全留）。
EXPERIMENTS["v7c_tune18"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Cv1_lr%s_lc%s_%s" % (lrk, lck, csk)
                  for lrk in ("5e-05", "0.0001", "0.0002")
                  for lck in ("0.1", "0.3", "1")
                  for csk in ("const", "cos")],
    "REQUIRES": [],
}


# Explicit corrected-label entries; historical experiments remain reproducible.
_CORRECTED_IDS = ["V7_Cv%d_tafix" % v for v in range(1, 8)] + ["V7_Cv5_SV1_tafix", "V7_Cv5_ST1_tafix"]
_STATE_IDS = ["V7_Cv%d_statefix" % v for v in range(1, 8)] + ["V7_Cv5_SV1_statefix", "V7_Cv5_ST1_statefix"]
# best 口径评估（2026-10-29 用户裁定 --ckpt last/best）：Cv1–7_statefix_smpl24
# 的 _evalbest（cam3）+ _evalbest_occ{010-050}，eval-only 零重训。
EXPERIMENTS["v7_statefix_best_eval"] = {
    "KIND": "model_suite",
    "MODEL_IDS": ["V7_Cv%d%s_evalbest" % (i, sd) for i in range(1, 8)
                  for sd in ("_s1", "_s2", "_s3")]
    + ["V7_Cv%d%s_evalbest_occ%s" % (i, sd, o)
       for i in range(1, 8) for sd in ("_s1", "_s2", "_s3")
       for o in ("010", "020", "030", "050")],
    "REQUIRES": [],
}

# Current-source packages, seven unified architectures, validation-only preview.
for _kind in ("tafix", "statefix"):
    _ids = ["V7_Cv%d" % i if _kind == "statefix" else "V7_Cv%d_tafix" % i for i in range(1, 8)]
    _prefix = "v7" if _kind == "statefix" else "v7_tafix"
    _seeded = [x + s for x in _ids for s in ("_s1", "_s2", "_s3")]
    EXPERIMENTS[_prefix + "_train"] = {"KIND": "model_suite",
        "MODEL_IDS": _seeded,
        "REQUIRES": []}
    EXPERIMENTS[_prefix + "_eval_occ"] = {"KIND": "model_suite",
        "MODEL_IDS": [x + "_evalocc" + o for x in _seeded
                      for o in ("010", "020", "030", "050")],
        "REQUIRES": []}

# V7 C 门控修改方案（2026-10-30）：T/LR/TLR 训练格与 occ 四档交叉评估格。
# 首轮只发 s1；s2/s3 按方案 §8.4 于 val 确认后补发（条目已注册）。
_V7_GATE_BASES = ("V7_Cv5T", "V7_Cv5LR", "V7_Cv5TLR")
EXPERIMENTS["v7_train"]["MODEL_IDS"] += [
    x + s for x in _V7_GATE_BASES for s in ("_s1", "_s2", "_s3")]
EXPERIMENTS["v7_eval_occ"]["MODEL_IDS"] += [
    x + s + "_evalocc" + o for x in _V7_GATE_BASES
    for s in ("_s1", "_s2", "_s3") for o in ("010", "020", "030", "050")]
