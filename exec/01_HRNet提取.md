# 任务 01 · HRNet 特征提取

分配给一个 Codex。只写下面两个文件。不要动模型、损失、训练。

先读 [00_总控.md](00_总控.md) 第 1 节冻结接口，再读：

- [anysole/types.py](/data/fangyuxuan/projects/gait/AnySole/anysole/types.py) 里的 `CLIFF_CKPT` `HRNET_YAML` `HRNET_CACHE_ROOT` `IMG_NORM_MEAN` `IMG_NORM_STD` `V_FEAT_DIM` `SEQ_ROOT`
- [anysole/data/dataset.py](/data/fangyuxuan/projects/gait/AnySole/anysole/data/dataset.py) 的 `hrnet_cache_path` `find_session_dir`
- MotionPRO [lib/util/gen_image_feature.py](/data/fangyuxuan/projects/gait/Baselines/MotionPRO/lib/util/gen_image_feature.py)
- MotionPRO [lib/model/cliff/cliff_hr48.py](/data/fangyuxuan/projects/gait/Baselines/MotionPRO/lib/model/cliff/cliff_hr48.py)（只用 `self.encoder`，forward 已直接返回 `xf`）

---

## 写哪些文件

1. `anysole/data/crop.py`
2. `anysole/data/extract_hrnet.py`

不要改 `types.py` 里的 `CROP_SIZE=256`。提取器自己用 height=256、width=192。

---

## crop.py

从 MotionPRO `gen_image_feature.py` 原样搬这些函数，不要重写几何：

- `CROP_IMG_HEIGHT = 256`
- `CROP_IMG_WIDTH = 192`
- `CROP_ASPECT_RATIO = 256 / 192`
- `get_transform` `transform` `bbox_from_detector` `crop` `process_image`

`bbox_from_detector` 输入 `[min_x, min_y, max_x, max_y]`，`rescale=1.1`。

`process_image(orig_img_rgb, bbox, crop_height=256, crop_width=192)` 必须输出：

- 裁剪图 `(3, 256, 192)` float32，ImageNet mean/std 归一化，值来自 `types.IMG_NORM_MEAN/STD`
- 不要改成正方形 256x256

读图用 cv2 BGR→RGB。bbox 来源：序列目录 `bbox.npy` 的列 `[1:5]` = minx,miny,maxx,maxy。不要用 `feature_hrnet.pth`。

---

## extract_hrnet.py

CLI：

`python -m anysole.data.extract_hrnet --cam-id 3 [--session S12021] [--overwrite] [--batch-size 32]`

行为：

1. 枚举 `SEQ_ROOT` 下 `*/*/{sid}`。若给了 `--session` 只跑该 sid。
2. cache 路径：`HRNET_CACHE_ROOT / f"{sid}.pt"`。已存在且无 `--overwrite` 则跳过。
3. 读 `align_meta.json["n_frames"]`。`color/` 下应有 `n_frames` 张 JPG symlink，按 `000000.jpg` 排序。
4. 同目录 `bbox.npy`。每一帧一行，取 `[1:5]`。
5. 把 MotionPRO 仓库根插入 `sys.path`，import `lib.model.backbone.hrnet.cls_hrnet.HighResolutionNet` 和 `hrnet_config`。用 `HRNET_YAML` `update_config`。
6. 加载 `CLIFF_CKPT`。state_dict 键带 `module.encoder.` 前缀。剥掉该前缀后 load 进 `HighResolutionNet`。`eval()` + `requires_grad_(False)`。
7. 按 batch 前向，得到每帧 HRNet 2048 维；按 MotionPRO/CLIFF 公式计算归一化 `bbox_info=[cx,cy,b]` 3 维，并拼成 `(T, 2051)` float32。
8. `T` 必须等于 `n_frames`。`torch.save` 到 cache。CPU 也能跑。

不要在训练脚本里提特征。训练只读 cache。

---

## 完成标准

```bash
PYTHONPATH=. /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.data.extract_hrnet --help
```

可选冒烟（有 GPU 再跑全量）：

```bash
PYTHONPATH=. /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m anysole.data.extract_hrnet --cam-id 3 --session S12021
```

成功后 `AnysoleWorkspace/derived/AnySole/hrnet_cache/cam3/S12021.pt` 形状是 `(339, 2051)`（S12021 的 n_frames=339）。没有 GPU 时至少 `--help` 和模块 import 必须通。

不要写测试文件（归 05）。不要改 dataset.py，除非 `hrnet_cache_path` 对不上——若改了，在本文件末尾加「骨架修正」一行。
