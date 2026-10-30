"""E6/E7 共用件（numpy only）：接触标签、相位状态机、GT 关节、逐帧踝足误差。

- 接触标签：f6_soft（z_gen 现口径），逐会话 npz（contact (T,2) 二值 0/1）
- 相位状态机：每脚 rising edge=落脚、falling edge=离地；±margin 近窗
  （重叠取最近决胜）→ 稳定支撑(0)/摆动(1)/落脚附近(2)/离地附近(3)
- GT 关节：facts + load_smpl + 项目 FK（与 prep_kp2d24/calib_baseline 同路径）
- 踝足误差：骨盆对齐后 ANKLE_FOOT_JOINTS 逐帧 L2 → mm/帧/侧
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
from hmr2_tta.paths import PROJECT_ROOT, facts_session_dir, resolve_uri  # noqa: E402
from hmr2_tta.data.assemble_npz import load_manifest_rows  # noqa: E402
from anysole.types import ANKLE_FOOT_JOINTS  # noqa: E402

LABELS_ROOT = (PROJECT_ROOT / "AnysoleWorkspace/model_inputs/AnySole/"
               "adapter_v1/labels/f6_soft_time_accel_state_smpl24_v1/labels")
CELL_ROOT = PROJECT_ROOT / "results/cells"

PHASE_STANCE, PHASE_SWING, PHASE_STRIKE, PHASE_OFF = 0, 1, 2, 3
PHASE_NAMES = {PHASE_STANCE: "stance", PHASE_SWING: "swing",
               PHASE_STRIKE: "strike", PHASE_OFF: "off"}
PHASE_NAMES_ZH = {PHASE_STANCE: "稳定支撑", PHASE_SWING: "摆动",
                  PHASE_STRIKE: "落脚附近", PHASE_OFF: "离地附近"}
LEFT_ANKLE_FOOT = (ANKLE_FOOT_JOINTS[0], ANKLE_FOOT_JOINTS[2])   # 7, 10
RIGHT_ANKLE_FOOT = (ANKLE_FOOT_JOINTS[1], ANKLE_FOOT_JOINTS[3])  # 8, 11


CONTACT_METHOD = "f6_soft_time_accel_state_smpl24_v1"


def load_contact(sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (contact (T,2) float 0/1, frame_id (T,), valid (T,))。"""
    if CONTACT_METHOD != "f6_soft_time_accel_state_smpl24_v1":
        from anysole.data.workspace_adapter import load_shared_session, load_contact_label
        session = load_shared_session(sid)
        values = load_contact_label(session, CONTACT_METHOD)
        frames = session["frames"]
        return values, frames["frame_id"], (frames["valid"] != 0) & (frames["fake"] == 0)
    d = np.load(LABELS_ROOT / f"{sid}.npz")
    # v2 标签包无 fake 字段；source_time_supported=False（会话头部源时间不
    # 支持的帧）即旧 fake 语义，与 valid 交集后剔除
    return d["contact"], d["frame_id"], (d["valid"] != 0) & (d["source_time_supported"] != 0)


def phase_labels(contact: np.ndarray, fps: float = 40.0,
                 margin: float = 0.1) -> np.ndarray:
    """(T,2) 接触 → (T,2) 相位（每脚独立）。

    事件 = 接触标签变化处；margin 近窗内标落脚附近/离地附近，多个事件
    近窗重叠时按最近事件决胜；其余按接触值标稳定支撑/摆动。
    """
    t = len(contact)
    m = max(1, int(round(margin * fps)))
    out = np.empty((t, 2), dtype=np.int8)
    for foot in range(2):
        c = contact[:, foot] > 0.5
        out[:, foot] = np.where(c, PHASE_STANCE, PHASE_SWING)
        events = []  # (idx, phase)
        for i in range(1, t):
            if c[i] and not c[i - 1]:
                events.append((i, PHASE_STRIKE))
            elif not c[i] and c[i - 1]:
                events.append((i, PHASE_OFF))
        for j in range(t):
            best = None
            for idx, kind in events:
                dist = abs(j - idx)
                if dist <= m and (best is None or dist < best[0]):
                    best = (dist, kind)
            if best is not None:
                out[j, foot] = best[1]
    return out


def load_gt_joints(sid: str) -> np.ndarray:
    """facts + load_smpl + 项目 FK → GT 世界关节 (T,24,3)[m]。"""
    from anysole.data.smpl_io import load_smpl
    from anysole.utils.geometry import fk_pose6d_np
    rows = load_manifest_rows(1)
    facts = np.load(facts_session_dir(sid, 3) / "frames.npz")
    smpl = load_smpl(resolve_uri(rows[sid]["smpl_path"]),
                     query_t=facts["mocap_time_s"])
    n = len(facts["frame_id"])
    pose = smpl["pose_6d"].reshape(n, 24, 6)
    return fk_pose6d_np(pose, smpl["trans_m"], smpl["offsets_m"], smpl["parents"])


def load_pred_joints(cell: str, sid: str, cond: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cell 预测 npz → (joints (T,24,3) native SMPL 系, frame_indices, valid_mask)。

    npz 的 joint_xyz_world 是 display 帧（+Z up，eval.py _smpl_yup_to_display
    口径）；评估与 GT（facts FK）都在 native 帧（+Y up）——此处逆变换
    display→native：(x, z, −y)。
    """
    d = np.load(CELL_ROOT / cell / "predictions/eval_motion" / f"{sid}_{cond}.npz")
    disp = d["joint_xyz_world"]
    native = np.stack((disp[..., 0], disp[..., 2], -disp[..., 1]), axis=-1)
    return native, d["frame_indices"], d["valid_mask"]


def foot_mpjpe_per_frame(pred: np.ndarray, gt: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(T,24,3)×2 → 左右踝足逐帧误差 (mm)：骨盆对齐后 7/10 与 8/11 均值。"""
    from anysole.utils.metrics import pelvis_align
    p, _ = pelvis_align(pred)
    g, _ = pelvis_align(gt)
    err = np.linalg.norm(p - g, axis=-1)  # (T,24) mm 前
    left = err[:, LEFT_ANKLE_FOOT].mean(axis=1) * 1000.0
    right = err[:, RIGHT_ANKLE_FOOT].mean(axis=1) * 1000.0
    return left, right


def action_of(sid: str) -> str:
    import csv
    with open(PROJECT_ROOT / "AnysoleWorkspace/protocol/manifests/"
              "session_manifest.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["session_id"] == sid:
                return row["action"]
    raise KeyError(sid)


def action_group(action: str) -> str:
    """01..11 → walk/run/squat/other（走路=03/04、跑步=05/08、深蹲=06）。"""
    return {"03": "walk", "04": "walk", "05": "run", "08": "run",
            "06": "squat"}.get(action, "other")
