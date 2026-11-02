#!/usr/bin/env python3
"""E0 参照地板（修改 01「评估标尺补全」，2026-11-03）：三类参照行之一。

常数均值姿态地板（train 拟合 / test 拟合）与平凡时序基线（复制 lag 帧前
GT）——让每个模型的数字可以被判读为「相对完全不看输入拿到了多少」。

口径（与评估协议严格一致）：
- GT 来源：``AnySoleDataset(mode="eval", session_ids=test 列,
  contact_method=f6_soft_time_accel_state_smpl24_v1)`` 的 ``kp_gt``
  （24 关节、世界系 native SMPL Y-up）；``seq_root/split_csv/cache_root``
  取 ``anysole.types`` 默认（cam3）。``allow_missing_video=True`` 只影响
  V_feat 装载，kp_gt/floor_y 逐位不变（不读视频缓存，CPU 数分钟）。
- 指标：``anysole.utils.metrics`` 的 ``pa_mpjpe``（逐帧相似对齐含尺度）
  与 ``pelvis_align + mean_point_error``（骨盆对齐 MPJPE），与
  ``eval_protocol._session_metrics`` 同函数、同常量，不自写对齐。
  池化 = 逐会话帧数加权（同 accum.add_scalar 权重 T）。
- 常数均值姿态：``(kp_gt − kp_gt[:, :1])``（逐帧骨盆相对）在 train 上求
  均值 → 作为每帧预测（train 拟合 = 地板）；另出 test 拟合版（乐观地板）。
- 平凡时序基线：lag ∈ {1, 5, 20, 40} 帧，逐会话 ``pa_mpjpe(gt[lag:],
  gt[:-lag])`` 等，帧数加权。
- 可定义指标五枚：PA-MPJPE / MPJPE / Jitter_pred / FootSliding /
  GroundPen；其余（traj_ATE、RootOrientationDrift 等）留空并在
  ``metric_reasons`` 写 ``not_applicable``（Step2Motion 行能力门控同款写法）。
  GroundPen 口径：常数姿态 = 静态预测，逐会话冻结在该会话帧 0 骨盆位置
  （(kp − kp[:, :1]) 的表达原点），FootSliding/Jitter_pred 恒 0；
  lag 基线直接用平移后的 GT。

产物（幂等，固定顺序输出）：
- results_display/Etest/E0_ReferenceFloors/Full_Table1_E0_Floors.csv（根层）
- results_display/Etest/E0_ReferenceFloors/detail/e0_reference_floors.json
- results_display/Etest/E0_ReferenceFloors/detail/E0_参照地板说明.md
- results_display/Etest/receipts/e0_reference_floors_receipt.json（sha256）

用法（touch_gait env）：
  python results_display/script/e0_reference_floors.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, load_split_ids  # noqa: E402
from anysole.types import (  # noqa: E402
    ANKLE_FOOT_JOINTS, FPS, HRNET_CACHE_ROOT, SEQ_ROOT, SPLIT_CSV,
)
from anysole.utils.metrics import (  # noqa: E402
    foot_sliding_joints,
    ground_penetration,
    mean_point_error,
    pa_mpjpe,
    pelvis_align,
    temporal_metrics,
)

ETEST = REPO / "results_display" / "Etest"
OUT_DIR = ETEST / "E0_ReferenceFloors"
DETAIL = OUT_DIR / "detail"
RECEIPTS = ETEST / "receipts"
CONTACT_METHOD = "f6_soft_time_accel_state_smpl24_v1"
CONTACT_VERSION = "state_smpl24"
LAGS = (1, 5, 20, 40)
# 展示层十指标列序（与 e_results_export.METRICS10 一致）；只填可定义五枚。
METRICS10 = ["PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
             "traj_ATE", "RootOrientationDrift", "FootSliding", "GroundPen",
             "Jitter_pred"]
DISPLAY_TO_KEY = {
    "PA-MPJPE": "pa_mpjpe_mm",
    "MPJPE": "mpjpe_mm",
    "FootSliding": "foot_sliding_mm",
    "GroundPen": "ground_pen_mm",
    "Jitter_pred": "jitter_pred_m_s3",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(column: str) -> AnySoleDataset:
    """test/train 会话的 kp_gt 数据集（native 世界系，不读视频缓存）。"""
    ids = load_split_ids(SPLIT_CSV, column)
    return AnySoleDataset(
        mode="eval", seq_root=SEQ_ROOT, split_csv=SPLIT_CSV,
        cache_root=HRNET_CACHE_ROOT, window_length=20, session_ids=ids,
        contact_method=CONTACT_METHOD, allow_missing_video=True)


class _Accum:
    """帧数加权池化（同 eval_protocol._Accum 的标量语义）。"""

    def __init__(self) -> None:
        self.sums: dict = {}
        self.counts: dict = {}

    def add(self, name: str, value: float, weight: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * weight
        self.counts[name] = self.counts.get(name, 0.0) + weight

    def means(self) -> dict:
        return {k: self.sums[k] / max(self.counts[k], 1e-12)
                for k in self.sums}


def _session_metrics(pred: np.ndarray, gt: np.ndarray, floor_y: float,
                     times: np.ndarray) -> dict:
    """五枚可定义指标的单会话均值 + 帧权重（metrics.py 同函数口径）。

    pred/gt 同长 (T,24,3) 世界系（native Y-up）；GroundPen 用 pred 原值
    （常数姿态调用方已做骨盆锚定、lag 调用方为平移后的 GT）。
    """
    T = pred.shape[0]
    out = {
        "pa_mpjpe_mm": float(pa_mpjpe(pred, gt).mean()),
        "frames": T,
    }
    pred_al, _ = pelvis_align(pred)
    gt_al, _ = pelvis_align(gt)
    out["mpjpe_mm"] = float(mean_point_error(pred_al, gt_al,
                                             scale=1000.0).mean())
    temporal = temporal_metrics(pred_al, gt_al, times)
    out["jitter_pred_m_s3"] = float(temporal["jitter_pred_m_s3"])
    value, count = foot_sliding_joints(
        pred[:, ANKLE_FOOT_JOINTS], gt[:, ANKLE_FOOT_JOINTS], times)
    out["foot_sliding_mm"] = (float(value), int(count))
    out["ground_pen_mm"] = float(
        ground_penetration(pred, floor_y)["ground_pen_mean_mm"])
    return out


def _pool(sessions: list) -> tuple[dict, dict]:
    """逐会话帧数加权池化；foot_sliding 按接触帧数加权（同 eval 口径）。"""
    accum = _Accum()
    fs_sum = fs_cnt = 0.0
    per_session = []
    for sess in sessions:
        m = sess["metrics"]
        per_session.append({
            "session_id": sess["session_id"],
            "frames": int(m["frames"]),
            "pa_mpjpe_mm": m["pa_mpjpe_mm"],
            "mpjpe_mm": m["mpjpe_mm"],
            "jitter_pred_m_s3": m["jitter_pred_m_s3"],
            "foot_sliding_mm": m["foot_sliding_mm"][0],
            "foot_sliding_count": m["foot_sliding_mm"][1],
            "ground_pen_mm": m["ground_pen_mm"],
        })
        accum.add("pa_mpjpe_mm", m["pa_mpjpe_mm"], m["frames"])
        accum.add("mpjpe_mm", m["mpjpe_mm"], m["frames"])
        accum.add("jitter_pred_m_s3", m["jitter_pred_m_s3"], m["frames"])
        accum.add("ground_pen_mm", m["ground_pen_mm"], m["frames"])
        fs_sum += m["foot_sliding_mm"][0] * m["foot_sliding_mm"][1]
        fs_cnt += m["foot_sliding_mm"][1]
    pooled = accum.means()
    pooled["foot_sliding_mm"] = fs_sum / max(fs_cnt, 1.0)
    pooled["n_sessions"] = len(sessions)
    pooled["n_frames"] = int(accum.counts["pa_mpjpe_mm"])
    pooled["foot_sliding_count"] = int(fs_cnt)
    return pooled, per_session


def reference_rows(ds_test: AnySoleDataset, ds_train: AnySoleDataset) -> dict:
    """六行参照：常数均值姿态（train/test 拟合）×2 + lag {1,5,20,40} ×4。"""
    # ---- 常数均值姿态 ----
    def mean_pose(ds):
        rel = [s["kp_gt"].astype(np.float64) - s["kp_gt"].astype(np.float64)[:, :1]
               for s in ds.sessions]
        n = sum(len(r) for r in rel)
        return (sum(r.sum(axis=0) for r in rel) / n), n

    rows = {}
    for fit, ds in (("train", ds_train), ("test", ds_test)):
        pose, n_fit_frames = mean_pose(ds)
        sessions = []
        for s in ds_test.sessions:
            T = s["kp_gt"].shape[0]
            kp = s["kp_gt"].astype(np.float64)
            pred = np.broadcast_to(pose, (T, 24, 3))
            # 世界系放置：(kp − kp[:, :1]) 表达在会话帧 0 骨盆系 → 常数姿态
            # 逐会话冻结在帧 0 骨盆位置（完全不看输入的静态预测；FootSliding
            # = 0、GroundPen = 冻结姿态对会话地面的穿透）。注意 broadcast
            # 视图下 pred[0:1] == 整个 pose，必须先做 pose − pose[0:1]。
            anchored = np.broadcast_to((pose - pose[0:1])[None], (T, 24, 3)) \
                + kp[0:1, 0:1]
            times = np.arange(T, dtype=np.float64) / float(FPS)
            sessions.append({"session_id": str(s["session_id"]),
                             "metrics": _session_metrics(
                                 anchored, kp, float(s["floor_y"]), times)})
        pooled, per_session = _pool(sessions)
        pooled["reference"] = "常数均值姿态（%s 拟合）" % (
            "train" if fit == "train" else "test")
        pooled["kind"] = "constant_pose_%s_fit" % fit
        pooled["fit_frames"] = int(n_fit_frames)
        rows[pooled["kind"]] = {"summary": pooled,
                                "per_session": per_session}
    # ---- 平凡时序基线 ----
    for lag in LAGS:
        sessions = []
        for s in ds_test.sessions:
            T = s["kp_gt"].shape[0]
            if T <= lag:
                continue
            kp = s["kp_gt"].astype(np.float64)
            times = np.arange(T - lag, dtype=np.float64) / float(FPS)
            sessions.append({"session_id": str(s["session_id"]),
                             "metrics": _session_metrics(
                                 kp[lag:], kp[:-lag], float(s["floor_y"]),
                                 times)})
        pooled, per_session = _pool(sessions)
        pooled["reference"] = "复制 %d 帧前 GT（%.3f s）" % (
            lag, lag / float(FPS))
        pooled["kind"] = "lag_%d" % lag
        pooled["lag_frames"] = int(lag)
        rows[pooled["kind"]] = {"summary": pooled,
                                "per_session": per_session}
    return rows


def metric_reasons() -> dict:
    reasons = {}
    for key in ("global_mpjpe_mm", "w_mpjpe100_mm", "mpjae_deg",
                "root_ate_mm", "root_ate_seg20_mm", "root_orientation_deg",
                "root_orientation_drift_deg", "pve_mm", "pve_t_mm"):
        reasons[key] = "not_applicable"
    reasons["global_mpjpe_mm"] = ("not_applicable: 参照行只定义对齐指标与"
                                  "逐帧指标（标尺口径五枚）")
    reasons["w_mpjpe100_mm"] = reasons["global_mpjpe_mm"]
    reasons["mpjae_deg"] = ("not_applicable: kp_gt 无关节旋转（角度类指标"
                            "不可定义）")
    reasons["root_ate_mm"] = ("not_applicable: 常数姿态无轨迹；lag 基线"
                              "按标尺口径只出五枚")
    reasons["root_ate_seg20_mm"] = reasons["root_ate_mm"]
    reasons["root_orientation_deg"] = reasons["mpjae_deg"]
    reasons["root_orientation_drift_deg"] = reasons["mpjae_deg"]
    reasons["pve_mm"] = "not_applicable: 无顶点/形状输出"
    reasons["pve_t_mm"] = reasons["pve_mm"]
    return reasons


def write_table(rows: dict) -> Path:
    header = ["参照", "口径"] + METRICS10
    out_rows = []
    order = ["constant_pose_train_fit", "constant_pose_test_fit",
             "lag_1", "lag_5", "lag_20", "lag_40"]
    for kind in order:
        s = rows[kind]["summary"]
        cells = [s["reference"], "test"]
        for display in METRICS10:
            key = DISPLAY_TO_KEY.get(display)
            v = s.get(key) if key else None
            if v is None:
                cells.append("")
            else:
                cells.append(round(float(v), 3))
        out_rows.append(cells)
    path = OUT_DIR / "Full_Table1_E0_Floors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)
    print("[csv] %s (%d rows)" % (path, len(out_rows)))
    return path


def write_json(rows: dict) -> Path:
    payload = {
        "schema_version": "e0_reference_floors_v1",
        "split": "test",
        "contact_version": CONTACT_VERSION,
        "contact_method": CONTACT_METHOD,
        "fps": float(FPS),
        "gt_source": ("AnySoleDataset(mode=eval, session_ids=test 列) kp_gt，"
                      "24 关节世界系 native SMPL Y-up；seq_root/split_csv/"
                      "cache_root = anysole.types 默认（cam3）"),
        "capabilities": {
            "contact_prediction": False,
            "direct_joint_rotations": False,
            "joint_positions": True,
            "predicted_shape": False,
            "predicted_surface": False,
            "pressure_prediction": False,
            "root_rotation": False,
            "root_translation": False,
        },
        "applicable_metrics": ["pa_mpjpe_mm", "mpjpe_mm",
                               "jitter_pred_m_s3", "foot_sliding_mm",
                               "ground_pen_mm"],
        "metric_reasons": metric_reasons(),
        "references": {kind: {"display_name": v["summary"]["reference"],
                              "summary": v["summary"],
                              "per_session": v["per_session"]}
                       for kind, v in rows.items()},
    }
    path = DETAIL / "e0_reference_floors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("[json] %s" % path)
    return path


def write_detail_md(rows: dict) -> Path:
    lines = ["# E0 参照地板说明（修改 01）", ""]
    lines.append("口径与判读见脚本 docstring；本 md 只记关键口径与数字。")
    lines.append("")
    lines.append("| 参照 | PA-MPJPE | MPJPE | FootSliding | GroundPen | Jitter_pred |")
    lines.append("|---|---|---|---|---|---|")
    for kind, v in rows.items():
        s = v["summary"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            s["reference"], s["pa_mpjpe_mm"], s["mpjpe_mm"],
            s["foot_sliding_mm"], s["ground_pen_mm"],
            s["jitter_pred_m_s3"]))
    lines.append("")
    lines.append("判读：train 拟合常数姿态 PA-MPJPE %.1f mm = 完全不看输入的"
                 "地板；lag=20 行 %.1f mm = 一个窗长的外推难度（oracle）；"
                 "lag=1 行 %.1f mm = 40 Hz 下抄上一帧的回报。" % (
                     rows["constant_pose_train_fit"]["summary"]["pa_mpjpe_mm"],
                     rows["lag_20"]["summary"]["pa_mpjpe_mm"],
                     rows["lag_1"]["summary"]["pa_mpjpe_mm"]))
    lines.append("")
    lines.append("其余十指标（GlobalMPJPE / W-MPJPE100 / MPJRE / traj_ATE / "
                 "RootOrientationDrift）对参照行不可定义或不在标尺口径内，"
                 "留空（metric_reasons=not_applicable，见 json）。")
    lines.append("")
    path = DETAIL / "E0_参照地板说明.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[md] %s" % path)
    return path


def write_receipt(artifacts: list[Path]) -> Path:
    sources = [{"cell": "splits.csv", "path": str(SPLIT_CSV),
                "sha256": _sha256(SPLIT_CSV)}]
    payload = {
        "split": "test",
        "contact_version": CONTACT_VERSION,
        "contact_method": CONTACT_METHOD,
        "n_sessions": 36,
        "sources": sources,
        "artifacts": [{"path": str(p), "sha256": _sha256(p)}
                      for p in artifacts],
    }
    path = RECEIPTS / "e0_reference_floors_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("[receipt] %s" % path)
    return path


def main() -> int:
    print("[E0-floors] build datasets ...")
    ds_test = build_dataset("test")
    ds_train = build_dataset("train")
    print("[E0-floors] test sessions=%d frames=%d; train sessions=%d" % (
        len(ds_test.sessions),
        sum(len(s["kp_gt"]) for s in ds_test.sessions),
        len(ds_train.sessions)))
    rows = reference_rows(ds_test, ds_train)
    for kind, v in rows.items():
        s = v["summary"]
        print("[E0-floors] %-28s PA=%.6f MPJPE=%.3f FS=%.3f GP=%.3f J=%.6f"
              % (s["reference"], s["pa_mpjpe_mm"], s["mpjpe_mm"],
                 s["foot_sliding_mm"], s["ground_pen_mm"],
                 s["jitter_pred_m_s3"]))
    artifacts = [write_table(rows), write_json(rows), write_detail_md(rows)]
    write_receipt(artifacts)
    print("[E0-floors] done -> %s" % OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
