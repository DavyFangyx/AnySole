# Fake-Aware Dataloader Design Guide

## 1. 目标

不裁剪视频与动捕数据，只在 `dataloader` 里过滤 `fake` 帧对应的训练窗口。

## 2. 当前数据约束

- 重建触觉结果位于 `AnysoleWorkspace/tools/PressureWasher/outputs/...`
- 标注后的触觉 CSV 含 `fake` 列
- MotionPRO 当前训练代码按帧索引直接对齐：
  - `feature_hrnet.pth`
  - `pressure.npz`
  - `smpl.npy`
  - `keypoints.npy`
  - `contact.npy`

## 3. 核心设计

### 3.1 `fake` 的编码

- `fake` 保持为帧级二值标记：`0 = 可训练`, `1 = 禁用`
- 建议编码成与序列等长的 `fake_mask`
- 若左右脚都参与同一训练样本，样本级 `fake_mask` 取 `left OR right`

### 3.2 过滤粒度

- 不删除原始序列
- 不裁剪视频/动捕文件
- 只过滤训练窗口
- 规则：窗口内任意帧 `fake=1`，该窗口直接丢弃

## 4. 推荐落地方式

### 4.1 预处理输出

建议从 CSV 生成：

- `fake_mask_left.npy`
- `fake_mask_right.npy`
- 可选 `fake_mask.npy`（左右合并后的样本级 mask）

### 4.2 Dataloader 改造

修改 `Baselines/MotionPRO/lib/dataset/image_pressure.py`：

- 读取 `fake_mask`
- 预先扫描所有窗口
- 生成 `valid_window_indices`
- `__len__()` 只返回有效窗口数
- `__getitem__()` 只访问有效窗口

## 5. 关键规则

- `fake=1` 的帧永不进入训练
- 末尾 padding 允许保留，但前提是真实帧都不是 `fake`
- 如果一个序列被 `fake` 切碎，仍保留原序列，窗口级跳过即可

## 6. 建议实现顺序

1. 先把 `fake` 编码成帧级 mask
2. 再在 `ImagePressureDataset` 里做窗口级过滤
3. 最后检查 `pressure / smpl / feature / contact` 的帧长是否一致

## 7. 结论

最稳妥的方案是：**保留全量数据，`fake` 只做窗口过滤，不做文件裁剪。**
