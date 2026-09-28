"""回填工具：历史 AnySole 指标文件迁移（评估整改任务 01 键名回填）。

两阶段，只做文件变换（读 → 迁移 → 写），不跑模型：

阶段 1（fseries 迁移）：把旧 `*_fseries.json` 明细文件改名落盘为
`<split>.json`，并派生同源简略文件 `<split>_brief.json`；随后删除旧
fseries 文件与已取消的 `*_fseries_seedN.json` 归档。

阶段 2（键名回填）：对已有 `<split>.json` / `<split>_brief.json` 对，
按评估整改任务 01 的锁定口径迁移键名并在 payload 中记录 provenance：

- `yaw_abs_deg` -> `root_orientation_deg`（公式完全一致，纯改名）
- `yaw_drift_deg` -> `root_orientation_drift_deg`（同上）
- 旧 `foot_sliding_mm`（顶点版数值）-> `foot_sliding_vertex_mm`
  （诊断名）；公共关节版 `foot_sliding_mm` 无法从旧文件还原，
  需重跑 eval 后才有，旧值绝不冒充关节版。
- `T_mse` 确定性派生：`T_mse = T_rmse²`（2026-09-27 V2T 层级叶键，
  与 FPP-Net 原生 pressure MSE 同名；纯平方关系，无需重跑 eval）。

阶段 2 同时做展示键名/顺序同步（2026-09-28 裁定）：明细与 brief 的
metrics 块套用 eval_protocol 的 display 映射（`mpjpe_mm` -> `MPJPE`、
`pa_mpjpe_mm` -> `PA-MPJPE` 等）并按固定顺序写出（主指标 PA-MPJPE 排最
前）。已带展示键名的文件会先还原为内部键再重新派生，保证幂等。

brief 由迁移后的明细经 summary_metrics 白名单重新派生：contact 4 键、
shape_vertex_std_mm、foot_sliding_vertex_mm 等诊断不再进入 brief，
V2T 组为层级结构（brief 6 键，叶 4 键只留在明细）。

运行（touch_gait 环境）:
    python -m anysole.backfill_brief                       # results/ 全量
    python -m anysole.backfill_brief --dry-run             # 只看清单不动文件
    python -m anysole.backfill_brief --results-root results/AnySole
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anysole.utils.eval_protocol import (
    DIAGNOSTIC_KEYS,
    INVERSE_DISPLAY_NAMES,
    display_metrics,
    order_payload,
    summary_metrics,
)

# 旧键 -> 新键（纯改名或改标，语义说明见 MIGRATION_NOTES）。
KEY_MIGRATION = {
    "yaw_abs_deg": "root_orientation_deg",
    "yaw_drift_deg": "root_orientation_drift_deg",
    "foot_sliding_mm": "foot_sliding_vertex_mm",
}
MIGRATION_NOTES = {
    "root_orientation_deg": "renamed from yaw_abs_deg (identical formula)",
    "root_orientation_drift_deg": "renamed from yaw_drift_deg (identical formula)",
    "foot_sliding_vertex_mm": (
        "renamed from foot_sliding_mm; the old value was vertex-based, the "
        "public joint-based foot_sliding_mm requires re-evaluation"
    ),
    "T_mse": (
        "derived as T_rmse^2 (2026-09-27 V2T hierarchy leaf key; "
        "deterministic, no re-evaluation needed)"
    ),
}


def migrate_metric_keys(metrics: dict) -> dict[str, list[str]]:
    """Rename legacy metric keys inside every group block (in place).

    Returns ``{new_key: [groups]}`` for provenance recording.  Old values are
    never reused under the public joint-based names: a legacy vertex-based
    ``foot_sliding_mm`` is relabelled as the vertex diagnostic instead.
    """
    record: dict[str, list[str]] = {}
    if not isinstance(metrics, dict):
        return record
    for group, block in metrics.items():
        if not isinstance(block, dict):
            continue
        for old, new in KEY_MIGRATION.items():
            if old in block and new not in block:
                block[new] = block.pop(old)
                record.setdefault(new, []).append(group)
    return record


def restore_display_names(metrics: dict) -> bool:
    """Convert serialized display names back to internal snake_case keys.

    Returns True when any display name was present (the file already uses
    the display naming).  Idempotent: files written by the current protocol
    pass through unchanged in the internal space.
    """
    found = False
    if not isinstance(metrics, dict):
        return found
    for block in metrics.values():
        if not isinstance(block, dict):
            continue
        for display, internal in INVERSE_DISPLAY_NAMES.items():
            if display in block and internal not in block:
                block[internal] = block.pop(display)
                found = True
    return found


def derive_t_mse(metrics: dict) -> list[str]:
    """Derive ``T_mse = T_rmse^2`` into any group block that has T_rmse
    but lacks T_mse (V2T hierarchy leaf key; pure square, no re-run).
    Returns the updated group names."""
    updated: list[str] = []
    if not isinstance(metrics, dict):
        return updated
    for group, block in metrics.items():
        if not isinstance(block, dict):
            continue
        if "T_rmse" in block and "T_mse" not in block:
            try:
                block["T_mse"] = float(block["T_rmse"]) ** 2
            except (TypeError, ValueError):
                continue
            updated.append(group)
    return updated


def _write_json(path: Path, payload: dict, *, sort_keys: bool) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _migrate_pair(detail_path: Path, brief_path: Path, dry_run: bool) -> None:
    try:
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("skip unreadable %s: %s" % (detail_path, exc))
        return
    metrics = detail.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        print("skip %s: no metrics" % detail_path)
        return
    # 已带展示键名的文件先还原内部键；旧式键名迁移只对历史文件生效。
    is_display = restore_display_names(metrics)
    record = {} if is_display else migrate_metric_keys(metrics)
    t_mse_groups = derive_t_mse(metrics)
    if t_mse_groups:
        record.setdefault("T_mse", t_mse_groups)
    # provenance: 已有迁移记录合并，字段级标注诊断键
    if record:
        migration = dict(detail.get("key_migration") or {})
        for new_key, groups in record.items():
            migration.setdefault(new_key, MIGRATION_NOTES[new_key])
        detail["key_migration"] = migration
    present = {key for block in metrics.values() if isinstance(block, dict) for key in block}
    # Existing payloads may already have a diagnostic_fields list from an
    # earlier migration; merge rather than treating it as authoritative so
    # late-added diagnostic keys such as pve_mm are not lost.
    existing_diagnostics = set(detail.get("diagnostic_fields") or ())
    detail["diagnostic_fields"] = sorted(
        existing_diagnostics | (present & set(DIAGNOSTIC_KEYS))
    )
    # 展示键名 + 固定顺序同步（明细与 brief 同一口径，主指标排最前）。
    detail["metrics"] = display_metrics(metrics)
    detail = order_payload(detail)
    brief = {key: value for key, value in detail.items() if key != "metrics"}
    brief["metrics"] = summary_metrics(metrics)
    if dry_run:
        print("would migrate %s (+ %s)%s" % (
            detail_path, brief_path.name,
            " [keys: %s]" % ", ".join(record) if record else " [keys already current]"))
        return
    _write_json(detail_path, detail, sort_keys=False)
    _write_json(brief_path, brief, sort_keys=False)
    print("migrated %s (+ %s)%s" % (
        detail_path, brief_path.name,
        " [keys: %s]" % ", ".join(record) if record else " [keys already current]"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="扫描根目录（默认 results/）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的动作，不落盘")
    args = parser.parse_args(argv)

    written = removed = migrated = 0
    # ---- 阶段 1：fseries 改名迁移（历史遗留） ----
    for detail_path in sorted(args.results_root.rglob("*_fseries.json")):
        if "_fseries_seed" in detail_path.name:
            continue
        try:
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print("skip unreadable %s: %s" % (detail_path, exc))
            continue
        metrics = detail.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            print("skip %s: no metrics" % detail_path)
            continue
        split = str(detail.get("split") or detail_path.stem.replace("_fseries", ""))
        new_detail = detail_path.with_name("%s.json" % split)
        restore_display_names(metrics)
        detail["metrics"] = display_metrics(metrics)
        detail = order_payload(detail)
        brief = {key: value for key, value in detail.items() if key != "metrics"}
        brief["metrics"] = summary_metrics(metrics)
        brief_path = detail_path.with_name("%s_brief.json" % split)

        if args.dry_run:
            print("would write %s (+ %s) from %s"
                  % (new_detail, brief_path.name, detail_path))
        else:
            _write_json(new_detail, detail, sort_keys=False)
            _write_json(brief_path, brief, sort_keys=False)
            detail_path.unlink()
            print("wrote %s (+ %s)" % (new_detail, brief_path.name))
            written += 1
        for seed_path in sorted(detail_path.parent.glob(detail_path.stem + "_seed*.json")):
            if args.dry_run:
                print("would remove %s" % seed_path)
            else:
                seed_path.unlink()
                print("removed %s" % seed_path)
                removed += 1
    # ---- 阶段 2：已有明细/brief 对的键名回填 ----
    for brief_path in sorted(args.results_root.rglob("*_brief.json")):
        stem = brief_path.name[: -len("_brief.json")]
        detail_path = brief_path.with_name("%s.json" % stem)
        if not detail_path.is_file():
            continue
        _migrate_pair(detail_path, brief_path, args.dry_run)
        migrated += 1
    print("done: %d detail renamed + brief written, %d seed archive removed, "
          "%d detail/brief pairs migrated%s"
          % (written, removed, migrated, " (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
