# 任务 05 · README + 最小可运行入口

分配给一个 Codex。等 01–04 代码落盘后做。验证与对拍由监督者做；本任务不要堆测试用例。

先读 [00_总控.md](00_总控.md) 第 1 节。

---

## 写哪些文件

1. `README.md`
2. 可选：`tests/test_smoke.py` 只留一个 CPU dummy forward。不要写 geometry/dataset 的长测试套。

---

## README.md

短。必须写明 V1 是 **23 关节 BVH 6D**，不是 SMPL。命令用仓库根、touch_gait python。

内容顺序：

1. 环境：`/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python`，`PYTHONPATH=.`
2. 提特征：`python -m anysole.data.extract_hrnet --cam-id 3`（可加 `--session S12021`）
3. 训练：`python -m anysole.train --config configs/v1.yaml`
4. 评估：`python -m anysole.eval --config configs/v1.yaml --ckpt outputs/v1/ckpt_last.pt`
5. 推理：`python -m anysole.infer --ckpt ... --session S12021 --config-id 0`
6. 数据路径各一行：cam3 序列、splits.csv、fake_marked 压力、HRNet cache 目录

不要写原理长文。不要复制「V+T → M 多模态动作重建 · V1 模型实现说明.md」。

---

## 可选 smoke（能不写测试文件就不写）

若需要一个文件让 import 链可见，最多：

```python
# tests/test_smoke.py
def test_forward_cpu():
    import torch
    from anysole.models import AnySoleModel
    m = AnySoleModel()
    out = m(
        torch.randn(2, 20, 2048),
        torch.randn(2, 20, 96),
        torch.randn(2, 20, 12),
        torch.randn(2, 20, 138),
        torch.randint(0, 1000, (2,)),
        torch.tensor([0, 2]),
    )
    assert out["x0_hat"].shape == (2, 20, 138)
    out["x0_hat"].sum().backward()
```

禁止再加 BVH 往返、数据集扫盘、DDIM 多步、HRNet 权重加载等测试。那些归监督者。

---

## 完成标准

`README.md` 存在，四条命令能从 README 复制。`python -m anysole.train --help` 已由 04 保证。不要为了测试去改模型维数。
