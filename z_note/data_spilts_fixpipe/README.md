# data_spilts_fixpipe —— 5 组受试者级 train/val/test 划分

2026-10-22 用户裁定：val 与 test 分离（旧 val=test 冻结作废）。canonical 划分恢复任务书 92/12/36
（`AnysoleWorkspace/protocol/splits/default/splits.csv`）；本目录 4 组备选划分专为**动作间效果对比**设计：
test 覆盖全部 11 个动作（canonical test = S13+S14 缺 04/05，只覆盖 9 个动作）。

全部由 `AnysoleWorkspace/tool/build_manifest.py split` 生成（列格式 `index,train,val,test`），
subject 级划分、val≠test、train/val/test 两两不相交、并集 140 session。

## 划分总表

| # | 文件 | val | test | train | 规模 (train/val/test) | test 动作覆盖 |
|---|---|---|---|---|---|---|
| 1 | `AnysoleWorkspace/protocol/splits/default/splits.csv` | S12 | S13, S14 | S5,S6,S7,S8,S10,S11 | 92/12/36 | 9/11（缺 04/05） |
| 2 | `splits_test_S5S6S11_val_S12.csv` | S12 | S5, S6, S11 | S7,S8,S10,S13,S14 | 62/12/66 | **11/11** |
| 3 | `splits_test_S5S11S12_val_S13.csv` | S13 | S5, S11, S12 | S6,S7,S8,S10,S14 | 64/18/58 | **11/11** |
| 4 | `splits_test_S6S7S13_val_S14.csv` | S14 | S6, S7, S13 | S5,S8,S10,S11,S12 | 73/18/49 | **11/11** |
| 5 | `splits_test_S5S6S8_val_S7.csv` | S7 | S5, S6, S8 | S10,S11,S12,S13,S14 | 78/11/51 | **11/11** |

## test 每动作会话数（判读每动作指标时参考）

| # | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 canonical | 5 | 3 | 1 | **0** | **0** | 6 | 6 | 1 | 5 | 5 | 4 |
| 2 | 9 | 6 | 5 | 1 | 2 | 9 | 8 | 2 | 8 | 8 | 8 |
| 3 | 9 | 4 | 4 | 1 | 2 | 8 | 7 | 2 | 5 | 9 | 7 |
| 4 | 8 | 6 | 4 | 1 | 1 | 9 | 5 | 1 | 5 | 5 | 4 |
| 5 | 9 | 5 | 4 | 1 | 1 | 8 | 5 | 2 | 6 | 5 | 5 |

## 数据硬约束（为什么 04/05/08 在 test 里不可能多）

- 动作 **04**（折返走路快）全库仅 2 条：S6×1、S12×1
- 动作 **05**（绕圈跑）全库仅 4 条：S11×2、S7×1、S8×1
- 动作 **08**（折返跑）全库仅 3 条：S5×2、S13×1

04/05/08 分属三个不相交 subject 组（{S6,S12} / {S11,S7,S8} / {S5,S13}），
因此 2 受试者的 test 不可能覆盖全部动作；#2–5 均用 3 受试者 test（5/1/3 结构）凑齐 11 动作。
04 在任何 test 里最多 1 条、05 最多 2 条、08 最多 2 条，这几类的每动作指标样本极小，判读需谨慎。

## 生成命令（记录用）

```bash
M=AnysoleWorkspace/protocol/manifests/session_manifest.jsonl
O=z_note/data_spilts_fixpipe
python3 AnysoleWorkspace/tool/build_manifest.py split --manifest $M --output $O/splits_test_S5S6S11_val_S12.csv  --test S5,S6,S11 --val S12
python3 AnysoleWorkspace/tool/build_manifest.py split --manifest $M --output $O/splits_test_S5S11S12_val_S13.csv --test S5,S11,S12 --val S13
python3 AnysoleWorkspace/tool/build_manifest.py split --manifest $M --output $O/splits_test_S6S7S13_val_S14.csv  --test S6,S7,S13 --val S14
python3 AnysoleWorkspace/tool/build_manifest.py split --manifest $M --output $O/splits_test_S5S6S8_val_S7.csv   --test S5,S6,S8 --val S7
```

## 消费方式

管线默认只读 `protocol/splits/default/splits.csv`。备选划分需显式传入（如 `extract_hrnet.py --split-csv`、
`Baselines/utils/gt_loading.py` 的 `split_csv` 参数、`MotionPRO/data_prep/make_splits.py --split-csv`）；
消费时注意 val/test 口径随之改变，#2–5 的 test 指标与历史数字不可比。
