# 检查点保留和十指标验证比较

> 2026-10-25 第二版更新：双条件 survey 与 Pareto 采样已弃用，选点规则改为「三门（V≤40/T≤90/VT≤40）+ 零税带（ΔPA≤0.5）+ 带内 VT PA 最优」，候选池改为三门通过者全留。现行规则见 `z_note/multimodal_fusion/ckpt规则.md`；本文档保留作第一版实现记录。

原有 best、last 和双条件 survey 保留原规则。新增机制额外保护关键权重，并对候选做完整十指标验证比较。目前没有约定正文核心指标和允许退化幅度，因此只输出比较与候选清单，不自动确定最终检查点，不生成替代原生 best 的文件。

## 两种运行方式

沿用原训练指令时，默认仍是 legacy。显式追加 `--checkpoint-policy legacy` 也使用原方案。原来的 `--survey-save` 及两个阈值、滚动数量参数照常使用。

下一次新训练如要同时启用新增机制，在原指令后追加：

```bash
--checkpoint-policy multi_metric --checkpoint-role auto --checkpoint-pareto-keep 8
```

也可以在新配置中添加下列字段。不要修改已完成训练的历史配置；选择新的输出目录。

```yaml
checkpoint_policy: multi_metric
checkpoint_role: auto
checkpoint_pareto_keep: 8
```

新增机制适用于 B、V4、V5_Av1、V5_Av4 这类具有验证 PA 指标的 6D 回归模型，当前不支持扩散或位置表示模型。角色由实际 config_probs 按 `[VT,V,T]` 推断：混合输入是 U；单一输入分别是 S_V、S_T、S_VT。手动指定的角色若与输入配置冲突，会报错。

## 训练时保留哪些权重

| 原有或新增 | 保留规则 | 是否改动原方案 |
| --- | --- | --- |
| ckpt_last | 每轮保存 | 否 |
| ckpt_best | 原来的 val/tau0_mpjpe/VT 最低 | 否 |
| survey | T PA≤90 mm 且 VT PA−V PA≤0.5 mm；仍可配置，仍按原数量滚动 | 否 |
| 新增关键权重 | 专家分别保留对应输入的 PA 和 GlobalMPJPE 最优点；U 分别保留 V、T、VT 的最优点，并另保留通过原双条件的最优点 | 单独新增 |
| 新增互补候选 | 在仍可用候选和当前轮之间保留最多 8 个互不完全优劣的候选；数量可配置 | 单独新增 |

关键最优点不受 survey 最近十个的限制。同一轮命中多个目标只保存一个文件。默认额外文件数上限为 U 20 个、每个专家 10 个，通常更少；这不包含原来的 best、last、survey。新增候选只保存模型、配置和轮次，**不含优化器，只供推理复评**。文件及引用清单提交后才删除不再需要的旧候选，不会删除原方案的任何文件。已有候选目录拒绝被新训练复用。

新增文件位于 `checkpoints/multi_metric/`，`manifest.json` 记录候选、保存理由和轮次，`history.jsonl` 保留每次验证的轨迹。PA 未定义或指标不是有限数时，跳过该轮新增保留并写入 `skipped_validation.jsonl`；原来的保存已经完成，训练继续。

训练日志的 `val/mpjpe` 实际是未做骨盆对齐的世界坐标误差。新机制内部明确按 GlobalMPJPE 使用，并在启用时增加 `val/global_mpjpe` 别名；旧键及 best 的选择规则保留。正式十指标中的 MPJPE 仍按骨盆对齐定义，二者不混用。

这些训练监控值只用于控制候选保存成本，**不是完整十指标**。有些在其他指标上优秀的历史轮次仍可能没有被保存。互补候选为有数量上限的在线样本，也不等于全部历史轮次的完整 Pareto 集。

## 候选的正式验证

原生评估已经提供完整十指标，不在每次训练验证时重复计算昂贵的人体表面指标。训练结束后，使用下列入口生成验证计划：

```bash
python -m anysole.checkpoint_candidates \
  --archive /path/to/checkpoints/multi_metric/manifest.json \
  --config /path/to/run.yaml \
  --out /path/to/new_validation_directory \
  --device cuda:0
```

默认只生成计划。追加 `--execute` 才执行，逐个调用原生 `anysole.eval`，显式指定 val、对应输入、完整明细，并关闭预测导出及额外鲁棒性面板。U 每个候选计算 V、T、VT；专家只计算自己的输入。原 best、last、survey 也纳入计划，除非指定 `--no-legacy`。同一文件内容只评一次；不同文件中相同模型权重但优化器或容器字节不同的情况可能仍重复评估。

执行入口检查 train/val/test 会话不交叉，校验配置及检查点是否变化，记录检查点、划分和协议代码哈希。它不启动训练、不改写检查点，也不运行测试集。

## 生成十指标比较与候选清单

比较入口读取原生 val.json、session_detail_val.json 和 window_detail_val.json，检查元数据和实际评分会话、窗口网格一致；有 receipt 时还核对检查点哈希。评估输入必须使用同一冻结数据和协议，网格核查不能替代数据版本冻结。

示例清单：

```json
{
  "split": "val",
  "experiment": "E1",
  "rules": null,
  "candidates": [
    {"id": "U_epoch130", "role": "U", "epoch": 130, "result": "/path/U_epoch130/val.json"}
  ],
  "references": {
    "V2M": {"result": "/path/S_V/val.json", "mode": "V2M"},
    "T2M": {"result": "/path/S_T/val.json", "mode": "T2M"},
    "VT2M": {"result": "/path/S_VT/val.json", "mode": "VT2M"}
  }
}
```

示例轮次是格式说明，不代表本次已恢复该权重。对照专家必须事先固定其完整检查点，不能按指标拼接。

```bash
python -m anysole.checkpoint_review \
  --manifest /path/to/review_manifest.json \
  --out /path/to/new_comparison_directory
```

输出 comparison.json、ten_metrics.csv、comparison.md。全部十项为 PA-MPJPE、MPJPE、GlobalMPJPE、W-MPJPE100、MPJRE、traj_ATE、RootOrientationDrift、FootSliding、GroundPen、Jitter_pred。位置类用 mm，角度用度，抖动用 m/s³，逐项列候选、对照和差值。正文先列输入条件与数值下降项数，附录列完整十项；下降项数不代表有意义或显著优势。

候选清单分别列各输入的互不完全优劣候选，以及 U 的三个输入合在一起、共 30 个维度的候选；不会把 T 的优势用加权求和抵消 VT 的损失。所有输入始终对应同一份 U 权重。若 12 个候选全部保留在清单里，表示当前规则不能进一步缩小范围，不能解释成 12 个都已合格。

E1、E2、E5 分开生成报告。E2 也必须先具备同协议、同网格的原生格式结果，不能直接混入 BVH-23 与 SMPL-24 的旧汇总表。E5 的参考必须用 `same_checkpoint_mode` 指向同一份 U 的对应输入，例如 VT2M 对 V2M；不同检查点不能伪装成同模型的输入对比。

当前 `rules` 必须为空。没有固定的核心指标和容忍带，工具不会通过隐藏阈值筛选，更不会选最终检查点。后续标准明确后再增加达标判断；测试集不用于选择指标或阈值。

## 本次验证结果

14 项检查均通过，覆盖真实训练保存块、新旧保存并行、输入角色、早期保护、写入失败、缺失指标、时间网格、统一权重、十指标明细、E5 同权重约束和正式验证计划。

用本轮四组真实训练日志各 148 次验证做保留模拟：U 第130轮、T 专家第25轮均能保留到训练结束，U 最多额外保存11份，低于20份上限。T 专家的 PA 最优点是第25轮，GlobalMPJPE 最优点是第50轮，两者会分别保留。**这只是历史保存机制模拟，删除的权重没有恢复，也没有据此生成新的模型成绩。**

已有 U 的12个可用检查点另做只读完整比较，共12×3×10=360项，使用此前固定的专家对照。实际评分仍是11个有有效窗口的验证会话、3360帧；名册中的 S12102 没有有效窗口，不补零。没有新增训练或测试评估，没有自动选最终检查点。

另做真实权重验收：将 T 专家第445轮写入新候选格式，模型参数与原文件逐项一致；两份权重在相同 CPU、相同验证配置下完成11会话3360帧的完整评估，十项指标逐项完全相同。该检查确认新保存与加载路径不改变输出，不代表模型精度提升。

CPU 验证建议单独使用 `num_workers: 0` 的评估配置。本次4读取进程的 CPU 验收出现读取等待，停止后用单进程完成；等待原因尚未确定，原训练配置不改写。跨CPU/GPU结果存在浮点数值差异，应使用同设备对照检查保存格式。

本次完整数值与候选：[十指标比较](/data/huyuxuan/checkpoint_policy_20261025/installed_validation_comparison/comparison.md)；[实施与验收](/data/huyuxuan/checkpoint_policy_20261025/实施说明.md)。
