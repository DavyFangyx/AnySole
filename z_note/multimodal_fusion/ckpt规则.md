# checkpoint 保存、候选复评与正式选点规则

日期：2026-10-25。适用任务：一个 U checkpoint 支持 V / T / VT；先检查 E2、E5，再进行 E1。本文件不修改实验计划的达标判据。

## 1 当前机制与本轮修改

仓库已有 `anysole/utils/checkpoint_policy.py`、`anysole/checkpoint_candidates.py`、`anysole/checkpoint_review.py`。复用这套 multi_metric 机制，不再新增第二套加权选点或覆盖原 best 的规则。原说明见 [已有机制说明](../../anysole/model_fix_note/checkpoint_policy_20261025.md)。

| 产物 | 含义 | 使用限制 |
| --- | --- | --- |
| `ckpt_last.pt` | 最近完成 epoch 的训练状态 | 文件存在不等于训练完成；必须核对 epoch 与 config.epochs |
| `ckpt_best.pt` | 原 `val/tau0_mpjpe/VT` 最优点 | 保留历史含义；不能代表 T 专家或统一模型的全条件最优 |
| `survey/survey_ep*.pt` | 原双条件满足时的留档 | 旧运行还可能位于 `checkpoints/checkpoints/survey/`，两处都检查，不搬迁旧文件 |
| `multi_metric/manifest.json` 及权重 | 按角色保护 PA/GlobalMPJPE 最优点，并保留有限数量的互补候选 | 是候选集合，不是最终模型；新训练需显式开启 |
| 原生 val 十指标明细及 receipt | 候选正式比较依据 | 需同会话、同帧轴、同窗口、同评估协议，并核对 checkpoint 哈希 |

本轮执行层修改：`configs/tools/common.py` 严格使用指定 `which` 或显式 `checkpoint`，缺失即报错；`configs/tools/runner.py` 检查源运行完成度与候选配置，拒绝把 OOM 后的中途权重当完成结果，并检查 val/test 期间 checkpoint 是否改变；`survey_eval_emit.py` 与 runner 的候选复评默认使用 val，test 必须显式指定。已注册的单模态条目与动态派生专家按各自实际目录解析。

新增 multi_metric 参数已接入 conf → runner → train 的参数映射。原历史配置、权重、指标不回写，也不把旧 best 重命名为新规则的 best。

## 2 防守条件与选点约束

原 survey 防守条件继续保留：val T-only PA ≤ 90 mm，且 val PA(VT) − PA(V) ≤ 0.5 mm。它们只防止明显欠训与局部姿态退化：0.5 mm 不等于逐位恒等，T-only 健康也不代表触觉帮助了融合。不得用提高阈值让失败模型通过。

正式比较必须列出完整十指标：PA-MPJPE、MPJPE、GlobalMPJPE、W-MPJPE100、MPJRE、traj_ATE、RootOrientationDrift、FootSliding、GroundPen、Jitter_pred。局部、世界系、物理指标分别解释，不用加权总分抵消某一侧的明显退化。训练监控的旧 `val/mpjpe` 是世界坐标误差，不能抄到正式骨盆对齐 MPJPE 一列。

候选优先检查：

- E2：同一 U 的 V/T/VT 三条件都可用；不同骨架、训练集或有效帧的外部基线不能直接作严格排名。
- E5：同一 U 在 cam3 与 occ010/020/030/050 下的 VT−V；重点看根轨迹、全局误差、足部位置、滑步和穿透，同时报告局部姿态代价及触觉干预。遮挡严重度不能成为换 checkpoint 的理由。
- E1：先固定 U，再与同数据、同架构、同监督、可比训练预算的三个专家比较。专家也固定完整权重，不能逐指标挑选专家。

核心指标及所有允许退化幅度尚未完整预注册，所以现有工具只输出候选和十指标差值，不自动宣布最终赢家。当前 shortlist 的各指标极值来自验证监控，不是测试排名，也不是已通过验收的模型。

## 3 新训练如何启用

在新运行的原训练命令后追加：

```bash
--checkpoint-policy multi_metric --checkpoint-role auto --checkpoint-pareto-keep 8
```

通过配置表执行时，在该新运行的 `train_args` 中设置相同三个参数。使用新模型名或新输出目录，不把新规则混入已经完成的旧运行。

角色按 `config_probs=[VT,V,T]` 推断：混合输入为 U，纯 V / T / VT 分别为对应专家。新 U 建议使用明确的三条件采样接口，并让它统一管理缺失掩码；不要同时用多套 null 增强却仍将角色登记为“始终双模态专家”。现有角色推断不能识别所有额外增强，需人工核对 `occ_mix`、`v_cond_dropout`。

## 4 验证与测试的顺序

1. 核对源运行完整结束。候选可以来自早期 epoch，但其所属训练必须已完成；epoch=3/740 的 last 不合格。
2. 只在 val 上重评可用候选，保留逐会话、逐窗口明细以及权重、配置、划分与评估代码版本。
3. 比较 E2、E5 的验证面板，用户固定一份 U 权重与其哈希。没有合格候选就报告失败，不从三个不同 checkpoint 拼接 V/T/VT 行。
4. 用固定权重执行 test E2 与 test E5，之后再执行 E1。历史 test 已被反复查看，旧数据上的再次评估属于历史重评，不能宣称为新的盲测。

已有 multi_metric 候选的验证计划入口：

```bash
python -m anysole.checkpoint_candidates --archive /path/to/checkpoints/multi_metric/manifest.json --config /path/to/run.yaml --out /path/to/new_validation_directory --device cuda:0
```

默认只生成计划，`--execute` 才运行验证。执行前必须确认源运行完整结束；目前该独立工具不能代替 runner 的全部完成度检查。历史 best/last/survey 的重评使用显式 `--ckpt`，命令见 [重执行清单与候选](实验重执行清单与候选checkpoint.md)。不伪造 multi_metric manifest 来绕过真实存档情况。

本轮尝试给独立入口追加同一完成度守卫，但 `anysole/checkpoint_candidates.py` 属于另一会话创建的不可写文件，普通写入与沙箱外写入均因文件权限失败，文件未改动。runner 与 survey 入口的守卫已经实现；使用独立入口前仍须先人工核对完成度。

```bash
python -m anysole.checkpoint_review --manifest /path/to/review_manifest.json --out /path/to/new_comparison_directory
```

## 5 历史结果与数据边界

保存规则只影响后续能够保留哪些权重，不能恢复已经删除或从未保存的 epoch。仅缺选点口径时，优先复用现存权重重评；仅当必须取得缺失的关键权重、修正训练监督或更换训练划分时，才重新训练。

同一 U 的正式表必须从同一个 checkpoint 的完整评估 JSON 取所有输入行。旧报告把 T-only 的 last 修正与 V/VT 的 best 并列，可作为诊断记录，不能原样当作新规则的统一模型主表。

GlobalMPJPE / ATE 仍受当前逐窗 GT `trans_anchor` 评估协议影响，不等于长序列自由积分。GT floor 只用于评分与用于后处理是两种不同的实验，后者不能算触觉贡献。S12102 无有效窗口、不同外部基线有效帧不一致等情况必须原样报告，不补齐、截断、填零或静默删除。

## 6 文件保留与清理

本轮不删除、不搬迁任何已有 checkpoint。现有 legacy survey 和 multi_metric 内部仍有自动 `unlink` 淘汰代码，与本项目“先列清单、确认后移动到 /_trash/<日期>/”的规则冲突；该项已单独提出，尚未执行清理。处理方式未确认前，不启动会触发自动淘汰的新训练。候选复评是只读权重操作，可独立准备。

## 7 验收命令

```bash
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -B -m unittest discover -s tests -p 'test_checkpoint_*.py' -v
```

覆盖已有候选保存与十指标比较，以及严格选取、训练完成度、配置一致性、训练失败传播和 val 默认复评。验收原始输出由本轮最终汇报给出。
