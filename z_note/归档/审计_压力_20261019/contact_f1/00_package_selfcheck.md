# 00 两包自检（前置闸门）

日期：2026-10-19。执行人：Agent 1。

## 执行方式

- 脚本：各包自带 `pipeline/check_package.py`（默认 `--package-root` = 脚本上一级 = 包目录）。
- 环境说明：任务约定统一 touch_gait（Python 3.8.20），但 `check_package.py` 使用
  `Path.is_relative_to`（Python ≥ 3.9 才有），touch_gait 下直接 AttributeError 无法运行。
  两包自检改用 `depthpro` env（Python 3.10.21，numpy 2.2.6；check 只用 stdlib+numpy，
  不依赖包内 requirements.txt 的 numpy==2.5.2 精确版本）。此环境偏差已如实登记。
- 两包均未做任何修改，仅执行校验。

## 结果

### v1 `Tactile_pressure_handoff_20261017_v1` — PASS

```json
{"status": "PASS", "package_manifest_files": 4239, "raw_foot_files": 890,
 "processed_foot_files": 890, "pressure_values": 9888960,
 "baseline_modified_values": 33800, "total_changed_from_raw": 1599431,
 "empty_files": 64, "external_paths_required": false}
```

- MANIFEST.sha256.json 4239 个文件逐一哈希比对通过；inputs 原始 CSV/元数据哈希通过。
- 890 个脚文件（445 会话 × 左右）全部可读，逐文件校验：
  `values48 == max(denoised48 - estimated_baseline48, 0)` 且 row_alignment_eligible 之外
  的行 values48 == denoised48；csv/changes 日志哈希与逐值 change log 一致。
- 关键计数：总压力值 9,888,960；基线修正值 33,800；相对原始值共改动 1,599,431
  （含自适应平滑的大头）；空文件 64。

### v2 `Tactile_pressure_handoff_20261017_v2_no_smoothing` — PASS

```json
{"status": "PASS", "package_manifest_files": 5119, "raw_foot_files": 890,
 "processed_foot_files": 890, "pressure_values": 9888960,
 "baseline_modified_values": 33800, "total_changed_from_raw": 34579,
 "empty_files": 64, "isolated_impulse_modified_values": 783,
 "adaptive_smoothing_enabled": false, "external_paths_required": false}
```

- MANIFEST 5119 文件哈希全过；inputs 原始文件哈希全过。
- 逐文件校验同 v1 口径，另加 v2 特有项：`denoised48 == impulse_only48`、
  impulse change log 逐值一致；`reports/full_rebuild_validation.json` 的计数
  与实测一致（baseline_modified=33800、total_changed=34579、impulses=783），
  且 `adaptive_smoothing_enabled == False`、平滑改动值 == 0。
- 关键计数：基线修正值 33,800（与 v1 相同）；孤立脉冲修正 783；
  相对原始值总改动 34,579（v1 为 1,599,431——差值即平滑处理规模）；
  空文件 64。

## 闸门结论

两包均 PASS，允许进入步骤 1 接线与步骤 2 审计。
