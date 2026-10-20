"""V6_1A occ_mix —— 窗口级结构化丢弃采样器（训练侧数据增强 + 窗源标签）。

权威设计：``z_note/multimodal_fusion/V6_1方案/V6_1A方案.md`` §2.3 腿②。
范围规则（§1）：本模块只做训练侧增强与标签——不新增模型组件、不写训练
循环、不改评估。共享文件的注册集成（dataset.py / train.py / losses.py /
configs）由主会话统一串行完成；本文件独立可导入，文件末尾
「集成挂钩点」注释节给出各挂钩函数的精确改法与日志键。

三开关（逐窗独立采样、允许叠加；采样粒度 = 窗口整窗一致，在 crop 装配
之前决定）：

- ``v_src`` ∈ {SRC_CAM3=0, SRC_OCC=1}，P(occ) = ``p_occ``（默认 0.2）。
  occ010 = bbox 底部 10% 涂黑后重跑 HRNet 的独立特征缓存
  （``.../visual/hrnet/cam3_occ_lower010/``，与 cam3 同帧轴、同窗口网格——
  闸门 ``smoke_occ_cache_align.py`` 已核实 140/140 会话零缺失、帧数全等、
  特征确有差异）。occ 窗的 V_feat 整窗来自 occ 缓存（档位 010 = 足部档，
  足部特征级丢弃由此承载；可扩展档位 020/030/050 缓存已齐，登记不进 v1）。
- ``v_zeroed`` ∈ {0,1}，P = ``p_nullV``（默认 0.1）。整窗 V 输入置零——
  train.py ``condition_inputs`` 层既有机制（H 口径，无学习参数，默认输入
  置零而非 learned null token）。
- ``t_zeroed`` ∈ {0,1}，P = ``p_T``（默认 0.1）。整窗 T 输入置零（同层）。
  T-drop 样本上接触头照常前向、接触损失照常计算（null 输入被训成标签先验）。

与 config_id 行的组合语义（train.py ``condition_inputs`` 既有零化是另一
维度）：VT 行 + t_zeroed → T 零化；T 行 + v_zeroed → V 已零化（冗余无害）；
V 行 + t_zeroed → T 已零化（冗余无害）。occ 标志在其上叠加，不改变
config_id 采样（config_probs 不变）。

无干扰 cam3 窗占比（静态核对函数 ``undisturbed_ratio``）：三开关独立，
精确值 exact = (1−p_occ)·(1−p_nullV)·(1−p_T) = 0.8×0.9×0.9 = 0.648；
设计文档口径 ≈ (1−p_occ−p_nullV)·(1−p_T) = 0.63（近似忽略了 occ∧nullV
双计项 p_occ·p_nullV·(1−p_T) = 0.018，两值差 < 2%）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from anysole.types import HRNET_CACHE_ROOT

__all__ = [
    "SRC_CAM3",
    "SRC_OCC",
    "SRC_NAMES",
    "OCC_DIR_DEFAULT",
    "WindowAug",
    "OccMixSampler",
    "undisturbed_ratio",
    "default_occ_cache_root",
    "occ_cache_path",
    "load_hrnet_cache",
    "load_occ_session",
    "check_dual_cache",
    "resolve_v_feat",
    "grouped_mean",
]

SRC_CAM3 = 0
SRC_OCC = 1
SRC_NAMES = {SRC_CAM3: "cam3", SRC_OCC: "occ"}

# 档位口径（010 = 足部档）：occ010 缓存目录名（相对 hrnet 根）。
# 可扩展档位 occ_lower020/030/050 缓存已齐，登记不进 v1（方案 §2.3a）。
OCC_DIR_DEFAULT = "cam3_occ_lower010"

# 预注册参数（方案 §2.3/§3；0.3 作 p_occ 扫描记录，不设网格）。
P_OCC_DEFAULT = 0.2
P_NULLV_DEFAULT = 0.1
P_T_DEFAULT = 0.1


# ---------------------------------------------------------------------------
# 窗源标签
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowAug:
    """一个窗口的采样标签（整窗一致；``__getitem__`` crop 之前决定）。"""

    v_src: int  # SRC_CAM3 / SRC_OCC —— 决定 V_feat crop 来自哪个缓存
    v_zeroed: bool  # null-V 开关（train.py 侧应用置零）
    t_zeroed: bool  # null-T 开关（train.py 侧应用置零）

    @property
    def src_name(self) -> str:
        return SRC_NAMES[int(self.v_src)]

    @property
    def undisturbed(self) -> bool:
        """无干扰 cam3 窗：cam3 源 + 两个置零开关全关。"""
        return (
            int(self.v_src) == SRC_CAM3
            and not self.v_zeroed
            and not self.t_zeroed
        )

    def batch_fields(self) -> Dict[str, torch.Tensor]:
        """进 batch 的标签键（scalar long，``collate_windows`` 直接 stack）。

        ``assert_batch_shapes`` 只校验白名单键，这三个新键不需要改 types.py。
        """
        return {
            "occ_src": torch.tensor(int(self.v_src), dtype=torch.long),
            "occ_v_zeroed": torch.tensor(int(self.v_zeroed), dtype=torch.long),
            "occ_t_zeroed": torch.tensor(int(self.t_zeroed), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 窗口级采样器
# ---------------------------------------------------------------------------

class OccMixSampler:
    """V6_1A 三开关窗口级采样器（逐窗独立伯努利，允许叠加）。

    RNG 约定：
    - ``seed`` 给定 → 内部 ``torch.Generator``（确定性，单进程用途：
      探针/冒烟/复现）；``reseed`` 可换 epoch 种子。
    - ``seed`` 为 None → torch 全局 RNG。DataLoader 默认按 worker 播种，
      多 worker 流独立；代价 = 同一窗口跨 epoch 重访时标签重采样
      （增强语义上等价，无需窗口级固定）。
    """

    def __init__(
        self,
        p_occ: float = P_OCC_DEFAULT,
        p_nullV: float = P_NULLV_DEFAULT,
        p_T: float = P_T_DEFAULT,
        seed: Optional[int] = None,
    ):
        for name, value in (("p_occ", p_occ), ("p_nullV", p_nullV), ("p_T", p_T)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("%s must be in [0, 1], got %s" % (name, value))
        self.p_occ = float(p_occ)
        self.p_nullV = float(p_nullV)
        self.p_T = float(p_T)
        self._gen = torch.Generator() if seed is not None else None
        if self._gen is not None:
            self._gen.manual_seed(int(seed))

    @property
    def enabled(self) -> bool:
        return bool(self.p_occ or self.p_nullV or self.p_T)

    def reseed(self, seed: int) -> None:
        if self._gen is None:
            self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """向量化采样 n 个窗口 → (v_src long (n,), v_zeroed bool (n,), t_zeroed bool (n,))。"""
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        draws = torch.rand(3, n, generator=self._gen)
        v_src = (draws[0] < self.p_occ).long()
        v_zeroed = draws[1] < self.p_nullV
        t_zeroed = draws[2] < self.p_T
        return v_src, v_zeroed, t_zeroed

    def sample_one(self) -> WindowAug:
        """单窗口采样（``__getitem__`` 挂钩用；crop 装配之前调用）。"""
        v_src, v_zeroed, t_zeroed = self.sample(1)
        return WindowAug(
            v_src=int(v_src.item()),
            v_zeroed=bool(v_zeroed.item()),
            t_zeroed=bool(t_zeroed.item()),
        )


@dataclass(frozen=True)
class _ExpectedMix:
    exact: float
    doc_formula: float
    double_count: float


def undisturbed_ratio(
    p_occ: float = P_OCC_DEFAULT,
    p_nullV: float = P_NULLV_DEFAULT,
    p_T: float = P_T_DEFAULT,
) -> _ExpectedMix:
    """无干扰 cam3 窗占比的静态核对函数。

    三开关逐窗独立 → exact = (1−p_occ)·(1−p_nullV)·(1−p_T)。
    设计文档口径 (1−p_occ−p_nullV)·(1−p_T)（≈0.63）为近似：它把 occ 与
    nullV 当互斥，忽略双计项 double_count = p_occ·p_nullV·(1−p_T) = 0.018。
    默认参数下 exact = 0.648；两者差 < 2%，冒烟按 exact 核对（±1%）。
    """
    p_occ, p_nullV, p_T = float(p_occ), float(p_nullV), float(p_T)
    exact = (1.0 - p_occ) * (1.0 - p_nullV) * (1.0 - p_T)
    doc_formula = (1.0 - p_occ - p_nullV) * (1.0 - p_T)
    return _ExpectedMix(
        exact=exact,
        doc_formula=doc_formula,
        double_count=p_occ * p_nullV * (1.0 - p_T),
    )


# ---------------------------------------------------------------------------
# 双缓存装载与对齐闸门（同帧轴/同窗格 = 形状一致 + 特征确有差异）
# ---------------------------------------------------------------------------

def default_occ_cache_root(cam3_root: Path = HRNET_CACHE_ROOT) -> Path:
    """由 cam3 缓存根推导 occ010 缓存根（兄弟目录，v1 档位固定 010）。"""
    return Path(cam3_root).parent / OCC_DIR_DEFAULT


def occ_cache_path(session_id: str, occ_root: Optional[Path] = None) -> Path:
    root = Path(occ_root) if occ_root is not None else default_occ_cache_root()
    return root / ("%s.pt" % session_id)


def load_hrnet_cache(path: Path) -> np.ndarray:
    """HRNet 特征缓存 → float32 (n_frames, V_FEAT_DIM) numpy（同 dataset.py 口径）。"""
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if torch.is_tensor(value):
        return value.float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_occ_session(
    session_id: str,
    occ_root: Optional[Path] = None,
    cam3_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """单会话 occ010 缓存装载；给 ``cam3_array`` 则强制同形状（同帧轴）。"""
    path = occ_cache_path(session_id, occ_root)
    if not path.is_file():
        raise FileNotFoundError(
            "Missing occ010 cache %s (V6_1A occ_mix requires cam3_occ_lower010 "
            "built for every training session)" % path
        )
    occ = load_hrnet_cache(path)
    if cam3_array is not None and occ.shape != np.asarray(cam3_array).shape:
        raise ValueError(
            "%s occ cache shape %s != cam3 %s (帧轴不一致，不可同窗替换)"
            % (session_id, occ.shape, np.asarray(cam3_array).shape)
        )
    return occ


def check_dual_cache(
    session_id: str,
    cam3_root: Path = HRNET_CACHE_ROOT,
    occ_root: Optional[Path] = None,
    cam3_array: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """单会话双缓存对齐闸门（与 ``smoke_occ_cache_align.py`` 同口径）。

    返回 {"shape", "max_abs_diff", "mean_abs_diff", "identical"}；形状不符
    抛 ValueError，特征全等抛 ValueError（occ 缓存不得是 cam3 复制品）。
    """
    if cam3_array is None:
        cam3_array = load_hrnet_cache(Path(cam3_root) / ("%s.pt" % session_id))
    cam3_array = np.asarray(cam3_array, dtype=np.float32)
    occ = load_occ_session(session_id, occ_root=occ_root, cam3_array=cam3_array)
    diff = np.abs(occ - cam3_array)
    max_abs = float(diff.max().item()) if diff.size else 0.0
    mean_abs = float(diff.mean().item()) if diff.size else 0.0
    if max_abs == 0.0:
        raise ValueError(
            "%s occ cache is identical to cam3 (max_abs_diff=0); occ_mix 无法注入遮挡态"
            % session_id
        )
    return {
        "shape": tuple(cam3_array.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "identical": False,
    }


def resolve_v_feat(session: Dict[str, np.ndarray], v_src: int) -> np.ndarray:
    """按窗源标签取该窗口应 crop 的特征数组（occ_mix 双缓存装载后的会话）。

    ``session`` 需含 ``"V_feat"``（cam3，dataset 现装载）与 ``"V_feat_occ"``
    （occ_mix 挂钩时由 ``load_occ_session`` 补入）。
    """
    if int(v_src) == SRC_OCC:
        if "V_feat_occ" not in session:
            raise KeyError("v_src=occ but session has no 'V_feat_occ' (occ_mix 未装载)")
        return session["V_feat_occ"]
    return session["V_feat"]


# ---------------------------------------------------------------------------
# A1–A3 分层统计辅助（train.py 日志侧消费）
# ---------------------------------------------------------------------------

def grouped_mean(
    values: torch.Tensor,
    occ_src: torch.Tensor,
    v_zeroed: Optional[torch.Tensor] = None,
    t_zeroed: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, object]]:
    """per-sample 值按窗源标签分组 → 组统计（{n, sum, mean}，空组 mean=None）。

    A1/A2 判据的口径入口（方案 §4.1）：
    - A1：``values`` = 该步 T 分支 per-sample 损失（L_contact + L_trec，未加权），
      比对 ``groups["occ"].mean`` vs ``groups["cam3"].mean``（配对差 CI）。
    - A2：``values`` = L_pose per-sample 损失，同组比对（操纵检查）。
    键：边际组 "cam3"/"occ"/"vz"/"tz"/"not_vz"/"not_tz" + 8 个组合键
    ``"<src>|<vz>|<tz>"``（如 "occ|0|1"）。
    """
    values = torch.as_tensor(values).reshape(-1).float()
    occ_src = torch.as_tensor(occ_src).reshape(-1).long()
    if v_zeroed is None:
        v_zeroed = torch.zeros_like(occ_src)
    if t_zeroed is None:
        t_zeroed = torch.zeros_like(occ_src)
    v_zeroed = torch.as_tensor(v_zeroed).reshape(-1).long()
    t_zeroed = torch.as_tensor(t_zeroed).reshape(-1).long()
    if not (values.shape[0] == occ_src.shape[0] == v_zeroed.shape[0] == t_zeroed.shape[0]):
        raise ValueError("values/occ_src/v_zeroed/t_zeroed 长度必须一致")

    def _sum(v: torch.Tensor) -> Dict[str, object]:
        if v.numel() == 0:
            return {"n": 0, "sum": 0.0, "mean": None}
        return {
            "n": int(v.numel()),
            "sum": float(v.sum().item()),
            "mean": float(v.mean().item()),
        }

    groups: Dict[str, Dict[str, object]] = {}
    for src in (SRC_CAM3, SRC_OCC):
        mask = occ_src == src
        groups[SRC_NAMES[src]] = _sum(values[mask])
    for key, mask in (
        ("vz", v_zeroed == 1),
        ("not_vz", v_zeroed == 0),
        ("tz", t_zeroed == 1),
        ("not_tz", t_zeroed == 0),
    ):
        groups[key] = _sum(values[mask])
    for src in (SRC_CAM3, SRC_OCC):
        for vz in (0, 1):
            for tz in (0, 1):
                mask = (occ_src == src) & (v_zeroed == vz) & (t_zeroed == tz)
                groups["%s|%d|%d" % (SRC_NAMES[src], vz, tz)] = _sum(values[mask])
    return groups


# ---------------------------------------------------------------------------
# 集成挂钩点（主会话注册集成时按此改共享文件；本文件不触碰它们）
# ---------------------------------------------------------------------------
#
# 1. anysole/data/dataset.py —— AnySoleDataset：
#    a) ``__init__`` 新参数（从 config 流入，默认关闭）：
#           occ_mix: Optional[dict] = None   # {"p_occ": 0.2, "p_nullV": 0.1, "p_T": 0.1}
#       启用后 ``self.occ_sampler = OccMixSampler(**occ_mix)``。
#    b) 逐会话循环（V_feat 装载之后、session dict 组装处）：
#           if self.occ_sampler is not None:
#               session["V_feat_occ"] = load_occ_session(
#                   session_id, occ_root=default_occ_cache_root(self.cache_root),
#                   cam3_array=v_feat)
#       （occ 缓存缺失/帧轴不符 → 立即报错；``allow_missing_video`` 回退路径
#        不挂 occ_mix——零 V 会话无真实遮挡缓存语义。）
#    c) ``__getitem__``（crop 装配之前，窗口级采样粒度）：
#           aug = self.occ_sampler.sample_one() if self.occ_sampler is not None else None
#           if aug is None:
#               item["V_feat"] = crop("V_feat")
#           else:
#               item["V_feat"] = crop("V_feat_occ") if aug.v_src == SRC_OCC else crop("V_feat")
#               item.update(aug.batch_fields())
#       ``collate_windows`` 与 ``assert_batch_shapes`` 均无需改动（新键为
#       scalar long，白名单校验跳过未知键）。
#    d) 增强全关 = 基线逐位一致：occ_mix=None 时路径与现行为完全相同。
#
# 2. anysole/train.py —— 训练循环（每个 batch，condition_inputs 之后）：
#       v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
#       if batch.get("occ_v_zeroed") is not None:
#           vz = batch["occ_v_zeroed"].view(-1, 1, 1).to(v_feat.device, dtype=torch.bool)
#           tz = batch["occ_t_zeroed"].view(-1, 1, 1).to(v_feat.device, dtype=torch.bool)
#           v_feat = torch.where(vz, torch.zeros_like(v_feat), v_feat)
#           t_raw  = torch.where(tz, torch.zeros_like(t_raw), t_raw)
#           t_phys = torch.where(tz, torch.zeros_like(t_phys), t_phys)
#           t_s2m  = torch.where(tz, torch.zeros_like(t_s2m), t_s2m)
#       （V5_1 模式下先恢复 batch T 再套 tz——"T 流三条件恒定"恢复在零化之前。）
#       v3f 分支在其 clone 的 v_feat 上再做 v_cond_dropout——occ 零化已在
#       v_feat 上，clone 自然继承，无需改 v3f 分支。
#    A1/A2 日志：compute_losses 现只回 batch 均值；需 per-sample 损失
#    （losses.py 加 reduction="none" 键，或 train.py 侧同公式重算），再以
#       stats = grouped_mean(l_tbranch_ps, batch["occ_src"],
#                            batch["occ_v_zeroed"], batch["occ_t_zeroed"])
#    wandb 键（epoch 级）：
#       train/occ_a1/tbranch_cam3  train/occ_a1/tbranch_occ   # L_contact+L_trec（未加权）
#       train/occ_a2/lpose_cam3    train/occ_a2/lpose_occ      # L_pose per-sample
#       train/occ/frac_occ         train/occ/frac_nullV
#       train/occ/frac_nullT       train/occ/frac_undisturbed
#    A3（窗口级条件分离，方案 §4.1）：val loader 同挂 occ_mix 标签（输入按
#    标签来源喂）；单次前向同时得 V2M 与 VT2M 读出（回归载体），per-window
#    误差按 (session_id, frame_start, occ_src) 三元组记账，
#    占比例 = mean(VT2M ≤ V2M per window)，occ 组 vs cam3 组对照。
#    注意区分：A3 是 val 分层监控；test 三条件面板不受增强开关影响
#    （增强全开关训练侧），面板自身保持 cam3 + occ010 自测双行。
#
# 3. configs（z_gen 生成 / queue conf 开关，主会话加）：
#       OCC_MIX=true   P_OCC=0.2   P_NULLV=0.1   P_NULLT=0.1
#    config_probs 不变（VT/V/T 行机制与 occ_mix 正交）；损失权重零新增
#    （lambda_contact 1.0 / lambda_air 2.0 / lambda_trec 0.1 / lambda_pose 3.0
#    / lambda_traj 1.0 均为既有权重键）。对照 = 同配方 p 全 0（增强全关）。
