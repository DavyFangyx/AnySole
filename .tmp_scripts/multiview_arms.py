"""临时：用其他相机（cam1/2/4）的既有预测，在 cam3 遮挡事件内替代 IMU 链。

装配：事件外 = V8_Cv5 cam3 VT；事件内 = VT 最后可见帧位置 + 其他相机模型
相对位移（按两模型交接帧根朝向差旋转）。指标 = 遮挡区间内 ATE。
"""
import json
import os
import sys

import numpy as np

ROOT = "/data/fangyuxuan/projects/gait"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/results/diagnostics/20261107_e9_blackout_ladder")
sys.path.insert(0, ROOT + "/results/diagnostics/20261108_traj_sceneocc")
from blackout_ladder import aa_yaw, rot2  # noqa: E402

MANI = ROOT + "/results/diagnostics/20261107_e5_sceneocc_full/manifests"
SIDS = ["S14033", "S14091", "S14092", "S14093", "S13082", "S13092", "S13093"]
CELL3 = ROOT + "/results/diagnostics/20261108_traj_sceneocc/scene7/eval_motion"
ARMS = {
    "cam1-VT": "V3_Bv1_cam1", "cam2-VT": "V3_Bv1_cam2", "cam4-VT": "V3_Bv1_cam4",
    "cam2-V": "V3_Bv1_cam2_vonly", "cam4-V": "V3_Bv1_cam4_vonly",
    "Hv5-cam2-VT": "V3_Hv5_cam2", "Hv5-cam4-VT": "V3_Hv5_cam4",
}


def load_pred(cell, sid, kind="VT2M"):
    p = "%s/%s/predictions/eval_motion/%s_%s.npz" % (ROOT, "results/cells/" + cell, sid, kind)
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    return (z["pred_pelvis_trans"].astype(np.float64),
            np.asarray(z["root_orient"]),
            np.asarray(z["valid_mask"], bool))


def evaluate():
    out = {}
    for sid in SIDS:
        man = json.load(open(os.path.join(MANI, sid + ".json")))
        ev = man["full_events"]
        zvt = np.load("%s/%s_VT2M.npz" % (CELL3, sid))
        gt = zvt["gt_pelvis_trans"].astype(np.float64)
        vt = zvt["pred_pelvis_trans"].astype(np.float64)
        ro_vt = np.asarray(zvt["root_orient"])
        n = len(gt)
        vm = zvt["valid_mask"].astype(bool)
        occ = np.zeros(n, bool)
        for e in ev:
            occ[max(int(e["start"]), 0):min(int(e["end"]), n)] = True
        out.setdefault("VT(cam3)", []).append(_occ_err(vt, gt, vm, occ))
        for name, cell in ARMS.items():
            kind = "V2M" if name.endswith("-V") else "VT2M"
            q = load_pred(cell, sid, kind)
            if q is None:
                continue
            other, ro_o, vmo = q
            m = min(n, len(other))
            traj = vt.copy()
            for e in ev:
                a = max(int(e["start"]), 1); b = min(int(e["end"]), n - 1, m - 1)
                n0 = a - 1
                psi_vt = np.radians(aa_yaw(ro_vt[n0]))
                psi_o = np.radians(aa_yaw(ro_o[n0]))
                th = psi_vt - psi_o
                p0 = vt[n0, [0, 2]]
                d = (other[a:b + 1, [0, 2]] - other[n0, [0, 2]])
                d = rot2(d, th)
                traj[a:b + 1, 0] = p0[0] + d[:, 0]
                traj[a:b + 1, 2] = p0[1] + d[:, 1]
            out.setdefault(name, []).append(_occ_err(traj, gt, vm, occ))
    return out


def _occ_err(traj, gt, vm, occ):
    d = np.linalg.norm(traj[:, [0, 2]] - gt[:, [0, 2]], axis=-1)
    ok = occ & vm
    return float(np.sqrt(np.mean(np.square(d[ok]))) * 1000) if ok.any() else None


if __name__ == "__main__":
    res = evaluate()
    # 按动作分组
    acts = {}
    for sid in SIDS:
        acts[sid] = json.load(open(os.path.join(MANI, sid + ".json")))["action"]
    for arm, vals in res.items():
        by = {}
        for sid, v in zip(SIDS, vals):
            by.setdefault(acts[sid], []).append(v)
        line = "%-14s" % arm
        for a in sorted(by):
            vv = [x for x in by[a] if x is not None]
            line += "  act%s %.1f (n=%d)" % (a, np.mean(vv) if vv else -1, len(vv))
        print(line)
