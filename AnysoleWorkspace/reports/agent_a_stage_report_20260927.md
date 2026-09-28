# Agent A Wave 1 交付报告

状态：complete（公共底座与旧树清理）

## 结果

- 新 manifest：`protocol/manifests/session_manifest.{csv,jsonl}`，144 sessions，140 eligible，4 个 `missing_smpl` 保留但不入 shared。
- 冻结 split：`protocol/splits/default/splits.csv`，train/val/test = 92/12/36，subject-disjoint。
- shared facts：140 个完整 session；MMVP 31×11：140 个 session，单一 `v1` 表示树。
- validators：path escape、manifest/split、shared dtype/shape/flags、MMVP 唯一树、工具 import 边界均通过；`workspace.py doctor` 通过。
- 完整逐 session 左右 CSV → shared NPZ provenance：`tactile_provenance.csv`。

## 旧 → 新路径映射（旧树已删除）

| 旧路径 | 新路径/处理 |
|---|---|
| 旧 `AnysoleWorkspace/sources/raw` | `raw://rgb`、`raw://pressure`、`raw://bvh`（直接外部只读链接） |
| 旧 `AnysoleWorkspace/sources/PressureWasher/outputs/*` | `work://data_pipeline/pressure_washer/final_fake_marked`及其过程目录 |
| 旧 `AnysoleWorkspace/manifests/session_manifest.*` | `protocol://manifests/session_manifest.*` |
| 旧 `AnysoleWorkspace/splits/default/splits.csv` | `protocol://splits/default/splits.csv` |
| `derived/MotionPRO/sequences/**/color` | `shared://facts/sessions/.../rgb` |
| `derived/MotionPRO/sequences/**/pressure.npz` | 禁止作为公共源；各模型 adapter 从 `shared pressure_48` 读取 |
| `derived/pressure_toolkit/**/insole` + `derived/VP-MoCap/**/insole` | 唯一 `shared://representations/tactile/mmvp_31x11/v1` |
| 旧 `derived/**`、旧 `dependencies/**`、旧 `calibration/**` | 已删除；依赖迁入 `assets/third_party`，固定资产迁入 `assets/` |

## B–F 冻结接口

B–F 只读 protocol/shared；不得读取其他模型的 model_inputs。`pressure_48.npz` 保留左右 48 格、4×12 物理布局、t_us 来源、valid/fake 和源 hash。MMVP 每帧为 `float32(2,31,11)`，由 pressure_toolkit/FPP-Net 共同消费，不复制。正式 checkpoint/prediction/metric 只能写 `results/`。

## 保留与未处理

保留上游语义：SMPL-24/BVH-23 原生契约、split 数量、PressureWasher fake/valid、D_Test4 已审计转换口径。未处理：B–F adapter、D_Test4 Wave 3 只读收口。

## 交付边界

本轮未修改模型结构/loss/augmentation、公共指标公式或正式 results；用户在开始前已有的代码/文档改动未触碰。
