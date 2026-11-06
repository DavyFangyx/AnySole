# splits/default —— 划分、触觉质量标注与排除依据

`splits.csv` 是本工作区唯一的训练/验证/测试划分文件（模型直接读取），列格式 `index,train,val,test`，
由 `tool/build_manifest.py split` 生成。**train/val/test = 92/12/36，val 与 test 分离（2026-10-22 裁定：val=S12、test=S13+S14、train 为其余 6 人）。**

## 触觉质量标注（排除依据）

每个 session 的触觉（压力）质量等级登记在 `protocol/manifests/session_manifest.{csv,jsonl}` 的两列
`pressure_quality_class` / `pressure_quality_reason`，源自压力清洗统计表：

`work/data_pipeline/pressure_washer/stats/pressure_stats_20260814_231054/overall/missing_pressure_objects.csv`

### 等级定义（2026-08-14 压力清洗统计口径，统计表见上）

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

## 生成命令（记录用；重跑会改变本文件内容）

```bash
python3 AnysoleWorkspace/tool/build_manifest.py split \
  --manifest AnysoleWorkspace/protocol/manifests/session_manifest.jsonl \
  --test S13,S14 --val S12
```

## 5-fold 动作覆盖实验（2026-10-22）

受试者轮换 leave-subjects-out 的 4 份轮换 split（`splits_test_<testSubjects>_val_<valSubject>.csv`），
与 `splits.csv`（canonical，val=S12 / test=S13,S14）一起构成 5 fold；轮换 fold 的 test
覆盖全部 11 动作（canonical 缺 04/05 系有意保留）。生成 = `configs/z_gen/per_action_folds.py`
（模型 fold 条目见 `configs/z_gen/model_registry.py` 同名块；每 fold 专属 yaml =
`anysole/configs/v1_fold_<testSubjects>.yaml`，仅 `split_csv` 键不同）。

| 文件 | val | test | test 动作 |
|---|---|---|---|
| `splits.csv` | S12 | S13,S14 | 9/11（缺 04/05） |
| `splits_test_S5S6S11_val_S12.csv` | S12 | S5,S6,S11 | 11/11 |
| `splits_test_S5S11S12_val_S13.csv` | S13 | S5,S11,S12 | 11/11 |
| `splits_test_S6S7S13_val_S14.csv` | S14 | S6,S7,S13 | 11/11 |
| `splits_test_S5S6S8_val_S7.csv` | S7 | S5,S6,S8 | 11/11 |
| `splits_test_S6S7_val_S14.csv` | S14 | S6,S7 | 10/11（缺 08） |

数据硬约束：04（折返走路快）全库 2 条（S6、S12）、05（绕圈跑）4 条
（S11×2、S7、S8）、08（折返跑）3 条（S5×2、S13）——稀有动作的 test 每条
split 最多 1~2 条，且每 fold 训练集里 0~2 条（近似未见动作泛化读）。

## splits_test_S6S7_val_S14.csv —— S13 退出留出池（2026-11-05 裁定）

**裁定：S13 只作 train/val 受试者，不作留出 test 受试者。** 理由是采集侧的鞋垫量程
异常，不是模型或表征问题：S13 的压力单格峰值 **915**（到不了 `PRESSURE_CLIP=1023`）、
p99.9 = 691、活跃格饱和率严格 **0.000%**、只用掉 **71%** 值域，而训练受试者约 2%
的活跃格顶死在 1.0；跨受试者看 S13 的活跃格占比 z(train) = **−3.01**、
`hull_mcol`（活跃格沿足长质心）z(train) = **+2.21**，两项都在 train 范围之外。
表征侧两条修法都已实测失败（固定阈 → 逐会话 p99 相对阈：z 从 +2.21 变 +3.11 更差；
逐会话重定心 RC 变体 × 4 折：最差折再升 +1.02），**底层不可弥合**。

> 注意：上面「触觉质量标注」那张表把 S13 记为 **33 A / 0 B / 0 C / 0 D**。
> A/B/C/D 等级判的是**完整性**（缺脚、zero-frame、采样间隔），不判**量程/标定**，
> 所以现行门槛抓不到 S13 这一类。量程是独立的一条轴。

本文件 = `splits_test_S6S7S13_val_S14.csv` 去掉 test 列里 S13 的 18 条会话
（S13011/013/021/023/061/062/063/071/072/073/082/092/093/101/102/103/112/113）：
**train（73）与 val（18）两列与原折逐位一致**，test 49 → 31 会话（18180 → 12540 帧）。
train/val 不变意味着该折**复用同一 ckpt_last、零重训**，只做 eval-only 重评
（cell = `<族>_fS6S7S13_s1_s13out`，注册见 `configs/z_gen/model_registry.py` 末尾
`_S13OUT_FAMILIES`；专属 yaml = `anysole/configs/v1_fold_S6S7.yaml`，仅 `split_csv` 不同）。

代价与口径：

- 该折 test 动作覆盖 **11/11 → 10/11**，掉的是动作 08（折返跑，全库仅 3 条 =
  S5×2 + S13，而本折 S5 在 train）。动作 08 的留出读数仍由 S5 进 test 的另三折提供。
- 四折合起来的 **test 留出池 = S5/S6/S7/S8/S11/S12 六人**（S10、S14 本来也从未入 test；
  S13 在另三折本就已在 train/val：`fS5S6S8`/`fS5S6S11` 为 train、`fS5S11S12` 为 val）。
- eval-only 口径下 S13 的 18 条会话在本折**不进任何列**（不是挪进 train——那要重训）。
  若要严格做成「S13 进 train」，需重训该折 1 格，本轮未做。

## splits_wo05.csv（2026-10-24）

canonical 去掉动作 05（绕圈跑）全部 4 条会话（S11051/S11053/S7053/S8053，均原 train 列）的版本：
train 92→88，val/test 不变（05 本无 held-out 会话）。用于「去绕圈跑」实验组。
