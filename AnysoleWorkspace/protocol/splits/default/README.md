# splits/default —— 划分、触觉质量标注与排除依据

`splits.csv` 是本工作区唯一的训练/验证/测试划分文件（模型直接读取），列格式 `index,train,val,test`。
自 2026-10-23 起，当前 canonical 文件采用 AnySole 方案A第三版 split：train/val/test = **92/12/36**，共 140 个不同 session，三组两两互斥。它与 `3_Result/AnySole/A3/data/splits.csv` 字节一致，SHA-256 为 `d826dfa7b9ba73cfdcfb5f49707a1bf8b82f013a90aae8f52e771d1ce25cb7b3`。

此前冻结的 split 为 104/36/36，val=test。A3 之前完成的训练和评估仍绑定旧版；历史结果不因当前 canonical 文件更新而改变。`tool/build_manifest.py split` 的旧默认行为会重新生成 val=test 划分，不能用于复现当前文件。

## 触觉质量标注（排除依据）

每个 session 的触觉（压力）质量等级登记在 `protocol/manifests/session_manifest.{csv,jsonl}` 的两列
`pressure_quality_class` / `pressure_quality_reason`，源自压力清洗统计表：

`work/data_pipeline/pressure_washer/stats/pressure_stats_20260814_231054/overall/missing_pressure_objects.csv`

### 等级定义（按 `tool/pressure_washer/analyze_pressure_csv_stats.py`）

| 等级 | 判据 |
|---|---|
| **A** | 双脚数据完整，且 zero-frame 秒数 = 0，且最大采样间隔 < 0.3s |
| **B** | 双脚完整，但存在 zero-frame 秒（>0）或最大间隔 ≥ 0.3s |
| **C** | 单脚数据完全缺失（`single-foot missing: L/R`，该脚 total frames = 0） |
| **D** | 上游采集目录缺失（`session directory missing`；仅见于 2026-08-14 全量统计，现行脚本只输出 A/B/C） |

### 各 subject 等级分布（08-14 全量统计，331 条）

| subject | A | B | C | D | 是否在 manifest |
|---|---|---|---|---|---|
| S5 | 30 | 3 | 0 | 0 | ✅（22 条） |
| S6 | 28 | 5 | 0 | 0 | ✅（21 条） |
| S7 | 14 | 3 | 0 | 16 | ✅（11 条） |
| S8 | 18 | 3 | 12 | 0 | ✅（9 条） |
| **S9** | **0** | **0** | **33** | **0** | ❌ **整组排除** |
| S10 | 13 | 1 | 19 | 0 | ✅（6 条） |
| S11 | 30 | 3 | 0 | 0 | ✅（24 条） |
| S12 | 29 | 3 | 0 | 1 | ✅（14 条） |
| S13 | 33 | 0 | 0 | 0 | ✅（18 条） |
| S14 | 28 | 5 | 0 | 0 | ✅（19 条） |

### 排除依据

1. **manifest 只含 A/B 级 session**（实测 144 行 = 127A + 17B，零 C/D）——触觉质量标注是进入管线的必要门槛。
2. **S9 整组排除**：33/33 条全部 C 级（`single-foot missing: L`，左脚压力数据每个 session 全部缺失），
   是唯一 100% 单脚缺失的 subject，历史上即整组排除。
3. **S1–S4 排除**：不完善数据集（早期采集），导入属历史遗留，不再导入（上游 20260419/20260422 保留但管线不引用）。
4. A/B 级内部仍有个别 session 未入 manifest（如 S8 的 21 条 A/B 只入 9 条），系 mocap 解算失败
   （`Mocap/0808/0808smpl/fail.txt` 等）与 final_fake_marked 可用性等进一步门槛，详见 `tool/build_manifest.py`。

## 旧版生成命令（仅作历史记录；会生成 val=test 并覆盖当前 A3 split，勿用于复现当前文件）

```bash
python3 AnysoleWorkspace/tool/build_manifest.py split \
  --manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl \
  --test S13,S14
```
