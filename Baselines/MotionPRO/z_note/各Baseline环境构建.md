# Baseline 环境构建 CLI

按当前机器来写，不另起一套官方理想环境。

- 仓库根目录：`/data/fangyuxuan/projects/gait/Baselines`
- Conda：`/data/fangyuxuan/miniconda3`
- 现成共享环境：`touch_gait`（Python 3.8.20，`torch 2.4.0+cu121`）
- GPU：8 x NVIDIA L40，驱动 580.159.03

原则：

- `touch_gait` 只增量适配 **CLIFF / PhysPT / SMPLest-X / ROMP+BEV 推理**
- **不要动** 现有 `torch==2.4.0`、`torchvision==0.19.0`、`numpy==1.23.0`
- **不要** 往 `touch_gait` 里装 `mmcv`、`tensorflow`、`pytorch3d`
- **VIBE / WHAM / TRACE** 各自单独建环境，互不复用

如果 `conda activate` 找不到命令，先执行：

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
```

pip 缓存目录当前不可写时，所有 `pip install` 都加 `--no-cache-dir`。

---

## 0. 先确认基座

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate touch_gait
which python
python -V
python -c "import torch, numpy, cv2, smplx, yacs; print('torch', torch.__version__); print('cuda', torch.version.cuda, torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
nvidia-smi -L
```

期望：

- python 路径含 `envs/touch_gait`
- Python `3.8.x`
- `torch 2.4.0+cu121`
- `numpy 1.23.0`
- `torch.cuda.is_available()` 为 `True`
- GPU 名是 `NVIDIA L40`

如果这里 CUDA 已经是 `False`，先不要继续装包。沙箱/远程会话里偶发读不到 `/dev/nvidia*`，换正常终端再测。

---

## 1. 改造 touch_gait

这一步只补共享推理依赖，不升级 PyTorch，不装 OpenMMLab CUDA 扩展。

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate touch_gait

python -m pip install --no-cache-dir \
  trimesh==4.6.2 \
  pyrender==0.1.45 \
  torchgeometry==0.1.2 \
  ultralytics==8.3.75 \
  pyqtgraph \
  einops==0.8.1 \
  timm==1.0.14 \
  json_tricks==3.17.3 \
  cython \
  lapx \
  scipy==1.10.1
```

`pyrender==0.1.45` 会把 `PyOpenGL` 钉死在 `3.1.0`。3.1.0 在无头渲染上容易出问题，所以再单独升到 3.1.4。pip 可能会报 incompatible，可以忽略。这是 SMPLest-X 官方安装脚本的做法。

```bash
python -m pip install --no-cache-dir pyopengl==3.1.4
```

再装 ROMP/BEV 的 `simple_romp`（不要用 `setup_trace.py`）：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp
python setup.py install
```

无头渲染时加上：

```bash
export PYOPENGL_PLATFORM=egl
# 若 egl 不行再改：
# export PYOPENGL_PLATFORM=osmesa
```

系统里已经有 `libosmesa6`，一般够用。

### 1.1 验证 touch_gait 是否装对

```bash
conda activate touch_gait
python - <<'PY'
import torch, numpy, cv2, smplx, yacs, trimesh, pyrender, torchgeometry, ultralytics, einops, timm, json_tricks, lapx
import romp, bev
print('torch', torch.__version__, torch.cuda.is_available())
print('numpy', numpy.__version__)
print('cv2', cv2.__version__)
print('smplx', smplx.__version__ if hasattr(smplx, '__version__') else 'ok')
print('ultralytics', ultralytics.__version__)
print('timm', timm.__version__)
print('einops', einops.__version__)
print('simple_romp', romp.__file__)
print('bev', bev.__file__)
assert torch.__version__.startswith('2.4.0')
assert numpy.__version__ == '1.23.0'
print('touch_gait shared env OK')
PY
```

期望：`torch` 仍是 `2.4.0+cu121`，`numpy` 仍是 `1.23.0`，`romp` / `bev` 能 import。

### 1.2 四个项目分别怎么进

CLIFF：

```bash
conda activate touch_gait
cd /data/fangyuxuan/projects/gait/Baselines/noah-research/CLIFF
python -c "import torch, trimesh, pyrender, torchgeometry, smplx, yacs; print('CLIFF imports OK')"
```

PhysPT：

```bash
conda activate touch_gait
cd /data/fangyuxuan/projects/gait/Baselines/PhysPT
python -c "import torch, trimesh, cv2, smplx, ultralytics, pyqtgraph; print('PhysPT imports OK')"
```

SMPLest-X：

```bash
conda activate touch_gait
cd /data/fangyuxuan/projects/gait/Baselines/SMPLest-X
python -c "import torch, numpy, cv2, smplx, trimesh, pyrender, einops, timm, ultralytics, json_tricks; print('SMPLest-X imports OK')"
```

ROMP / BEV：

```bash
conda activate touch_gait
cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp
python -c "import romp, bev; print('ROMP/BEV imports OK')"
romp -h
bev -h
```

注意：`touch_gait` **不要**再执行下面这些，会把共享环境打坏：

```bash
# 不要执行
pip install mmcv
pip install tensorflow==1.15.4
python setup_trace.py install
pip install -r /data/fangyuxuan/projects/gait/Baselines/VIBE/requirements.txt
pip install -r /data/fangyuxuan/projects/gait/Baselines/WHAM/requirements.txt
```

环境能 import，不代表能出结果。这四个仓库目前还缺各自的 SMPL/SMPL-X 和预训练权重，那是数据准备，不是环境问题。

---

## 2. 独立环境：TRACE

`trace/` 训练和 `simple_romp/trace2/` 推理共用这一个环境。必须现场编译 `deform_conv`，所以不要复用 `touch_gait`。

系统 `/usr/local/cuda` 是 13.3，`CUDA_HOME` 还可能指向不存在的 `/usr/local/cuda-12.2`。编译时改用 conda 里的 CUDA 12.1。

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh

conda create -n trace python=3.8 -y
conda activate trace

python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
conda install -n trace -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 -y

python -m pip install --no-cache-dir --upgrade setuptools numpy==1.23.0 cython scipy opencv-python lap
```

指定 CUDA 后再装 TRACE：

```bash
conda activate trace
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
which nvcc
nvcc --version

cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp
python setup_trace.py install

cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp/trace2/models/deform_conv
python setup.py develop
```

如果还要跑 `ROMP/trace` 训练代码，同一环境里再编一份：

```bash
conda activate trace
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

cd /data/fangyuxuan/projects/gait/Baselines/ROMP/trace/lib/models/deform_conv
python setup.py develop

cd /data/fangyuxuan/projects/gait/Baselines/ROMP/trace/lib/tracker/cython_bbox
python setup.py install
```

验证：

```bash
conda activate trace
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
import trace2
print('trace2', trace2.__file__)
PY
python -c "from models.deform_conv.modules import DeformConv; print('deform_conv OK')" || true
```

当前仓库里现成的 `deform_conv_cuda*.so` 是 **Python 3.9** 编的，这个 `trace` 环境是 3.8，必须重编，不要直接拿来用。

---

## 3. 独立环境：WHAM

官方栈是 Python 3.9 + torch 1.11 + CUDA 11.3 + mmcv 1.3.9。这和 `touch_gait` 的 mmdet 3.x / torch 2.4 完全不兼容。

先把空的 submodule 拉下来：

```bash
cd /data/fangyuxuan/projects/gait/Baselines/WHAM
git submodule update --init --recursive
ls third-party/ViTPose third-party/DPVO
```

建环境：

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
conda create -n wham python=3.9 -y
conda activate wham

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
python -m pip install --no-cache-dir -r /data/fangyuxuan/projects/gait/Baselines/WHAM/requirements.txt
python -m pip install --no-cache-dir -v -e /data/fangyuxuan/projects/gait/Baselines/WHAM/third-party/ViTPose
```

DPVO 要编 CUDA 扩展。本机 gcc 是 11.4，按官方要求降到 9.5，并用 conda 的 CUDA 11.3，不要用系统 CUDA 13.3：

```bash
conda activate wham
cd /data/fangyuxuan/projects/gait/Baselines/WHAM/third-party/DPVO

wget -O /tmp/eigen-3.4.0.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip -o /tmp/eigen-3.4.0.zip -d thirdparty

conda install pytorch-scatter=2.0.9 -c rusty1s -y
conda install cudatoolkit-dev=11.3.1 -c conda-forge -y
conda install -c conda-forge gxx=9.5 -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
nvcc --version
$CC --version

pip install --no-cache-dir .
```

可视化可选，不影响主推理：

```bash
conda activate wham
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
python -m pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

验证：

```bash
conda activate wham
python - <<'PY'
import torch, mmcv, timm
print('torch', torch.__version__, torch.cuda.is_available())
print('mmcv', mmcv.__version__)
print('timm', timm.__version__)
from mmpose.apis import init_pose_model
print('mmpose OK')
PY
```

期望：`torch 1.11.0`，`mmcv 1.3.9`。L40 是 sm_89，torch 1.11 没有原生 kernel，第一次跑可能会 PTX JIT，偏慢；如果直接报 architecture 不支持，就不要再往上升 torch，否则 mmcv 1.3.9 会跟着炸。

---

## 4. 独立环境：VIBE

官方写的是 Python 3.7 + torch 1.4.0 + tensorflow 1.15.4。这套在 L40 上基本不能用：torch 1.4 不支持 sm_89。

下面分两套。日常推理用 4.1；只有对照官方安装脚本时才走 4.2。

### 4.1 推荐：能在这台 L40 上做 demo 的环境

VIBE 的 `demo.py` 不依赖 TensorFlow。TF 1.15 只出现在训练数据预处理 `insta_utils.py`。

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
conda create -n vibe python=3.8 -y
conda activate vibe

python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-cache-dir \
  tqdm yacs h5py numpy==1.23.0 scipy numba smplx gdown PyYAML joblib pillow \
  trimesh pyrender progress filterpy matplotlib scikit-image scikit-video opencv-python llvmlite chumpy

python -m pip install --no-cache-dir git+https://github.com/mkocabas/yolov3-pytorch.git
python -m pip install --no-cache-dir git+https://github.com/mkocabas/multi-person-tracker.git
```

torchvision 0.19 已经没有 `torchvision.models.utils.load_state_dict_from_url`，需要改一行，否则 `demo.py` 会在 import 时挂：

```bash
sed -n '1,15p' /data/fangyuxuan/projects/gait/Baselines/VIBE/lib/models/resnet.py
```

把

```python
from torchvision.models.utils import load_state_dict_from_url
```

改成

```python
from torch.hub import load_state_dict_from_url
```

验证：

```bash
conda activate vibe
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from lib.models.vibe import VIBE
print('VIBE import OK')
PY
```

这条要在 `VIBE/` 目录下跑，因为 `lib` 是相对导入。

### 4.2 官方原版（不建议在这台机器上用）

```bash
source /data/fangyuxuan/miniconda3/etc/profile.d/conda.sh
conda create -n vibe_official python=3.7 -y
conda activate vibe_official
cd /data/fangyuxuan/projects/gait/Baselines/VIBE
bash scripts/install_conda.sh
```

这套会装 `torch==1.4.0` 和 `tensorflow==1.15.4`。环境能建起来，但 L40 上 GPU 推理大概率直接失败。只留给对照官方依赖，不拿来跑这台机器。

---

## 5. 四个环境怎么分开用

```bash
conda env list
```

期望至少有：

```text
touch_gait    /data/fangyuxuan/miniconda3/envs/touch_gait
trace         /data/fangyuxuan/miniconda3/envs/trace
wham          /data/fangyuxuan/miniconda3/envs/wham
vibe          /data/fangyuxuan/miniconda3/envs/vibe
```

对应关系：

```text
touch_gait  ->  CLIFF, PhysPT, SMPLest-X, ROMP/BEV, 以及现有 MotionPRO 主体
trace       ->  ROMP/trace 训练, ROMP/simple_romp/trace2 推理
wham        ->  WHAM
vibe        ->  VIBE
bbox_scan   ->  继续只给 MotionPRO 的 gen_bbox 用，不要混进上面这些
```

进项目前先切环境：

```bash
# 共享推理
conda activate touch_gait
cd /data/fangyuxuan/projects/gait/Baselines/noah-research/CLIFF
cd /data/fangyuxuan/projects/gait/Baselines/PhysPT
cd /data/fangyuxuan/projects/gait/Baselines/SMPLest-X
cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp

# TRACE
conda activate trace
cd /data/fangyuxuan/projects/gait/Baselines/ROMP/simple_romp/trace2
# 或
cd /data/fangyuxuan/projects/gait/Baselines/ROMP/trace

# WHAM
conda activate wham
cd /data/fangyuxuan/projects/gait/Baselines/WHAM

# VIBE
conda activate vibe
cd /data/fangyuxuan/projects/gait/Baselines/VIBE
```

---

## 6. 编译 CUDA 扩展时的公共坑

这台机器上最容易踩的是 CUDA 工具链不一致：

- 驱动可见 CUDA：13.0
- 系统 toolkit：`/usr/local/cuda-13.3`
- `touch_gait` / 推荐 TRACE / 推荐 VIBE：PyTorch **cu121**
- 官方 WHAM：PyTorch **cu113**
- 当前 shell 里 `CUDA_HOME` 可能是不存在的 `/usr/local/cuda-12.2`

所以：

- 共享环境 `touch_gait` **不要编译** CUDA 扩展
- TRACE / WHAM 编译前必须 `export CUDA_HOME="$CONDA_PREFIX"`
- 不要用系统 `nvcc 13.3` 去给 cu121 / cu113 的 PyTorch 编扩展

快速自检：

```bash
echo "CUDA_HOME=$CUDA_HOME"
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```
