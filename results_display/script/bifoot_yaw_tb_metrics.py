#!/usr/bin/env python3
"""T+BiFoot with the new BiFoot yaw branch (2026-11-15).

Translation: frozen BiFoot displacement branch.
Yaw: per-frame dyaw from the combined BiFoot-Yaw checkpoint, integrated from
the GT first-frame yaw; replaces the T2M root yaw, tilt preserved.
Outputs E2 clean and E5 scene/gap T-channel metrics.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
VAR = REPO / "results/diagnostics/20261111_variant_sweep"
YAW = REPO / "results/diagnostics/20261115_imu_yaw_branch"
sys.path.insert(0, str(VAR)); sys.path.insert(0, str(YAW)); sys.path.insert(0, str(REPO))
from bifoot_standalone import load_standalone  # noqa: E402
from bifoot_yaw_train import clip_arrays, infer_starts, _pad  # noqa: E402
from bifoot_clean_e2_row import metrics_for_session  # noqa: E402
import anysole.phase_common as PC  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ADAPTER = REPO / "AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3_folds"
MOTION = REPO / "results/diagnostics/20261114_imu_eval_v2_motion"
CACHE = VAR / "clean_e2_t_bifoot/_cache"
PROTO = REPO / "results/diagnostics/20261114_imu_eval_v2_all/V8_IMU_Residual"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
KEYS = ("PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "traj_ATE",
        "GroundPen", "FootSliding", "Jitter_pred", "RootOrientationDrift",
        "MPJPE_anklefoot")
SCEN = ("sceneocc", "gap010", "gap020", "gap040", "gap060")
T = 100

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def yaw_of_axisangle(aa):
    R = Rotation.from_rotvec(aa).as_matrix()
    return np.unwrap(np.arctan2(R[:, 0, 2], R[:, 2, 2]))

@torch.no_grad()
def predict_yaw(model, norm, clip, device):
    n = clip["n"]
    disp = np.zeros((n, 3), np.float64); dyaw = np.zeros(n, np.float64)
    cnt = np.zeros(n, np.float64)
    for si, a in enumerate(infer_starts(n)):
        b = min(a + T, n); L = b - a
        cc = _pad(clip["c"][a:b], L, T)
        tl = _pad(clip["tl"][a:b], L, T)
        tr = _pad(clip["tr"][a:b], L, T)
        y_d, y_w = model(cc[None].to(device), tl[None].to(device),
                         tr[None].to(device), return_yaw=True)
        d = y_d[0, :L].cpu().numpy().astype(np.float64)
        w = y_w[0, :L].cpu().numpy().astype(np.float64)
        ea = 0 if si == 0 else min(T // 2, L)
        if ea < L:
            disp[a + ea:b] += d[ea:L]
            dyaw[a + ea:b] += w[ea:L]
            cnt[a + ea:b] += 1
    ok = cnt > 0
    disp[ok] /= cnt[ok, None]; dyaw[ok] /= cnt[ok]
    if not ok.all():
        idx = np.where(ok)[0]
        for t in range(n):
            if not ok[t]:
                disp[t] = disp[idx[-1]]; dyaw[t] = dyaw[idx[-1]]
    dyaw[0] = 0.0
    yaw_hat = clip["yaw"][0] + np.cumsum(dyaw)
    return yaw_hat, clip["yaw"]

def load_yaw(fold, device):
    model, norm, _ = load_standalone(str(YAW / ("bifoot_standalone_yaw_%s.pt" % fold)), device)
    norm.to("cpu")
    te = torch.load(ADAPTER / fold / "gait/gait_test.pt", map_location="cpu", weights_only=False)
    out = {}
    for i in range(len(te.poses)):
        sid = str(te.session_ids[i])
        if int(te.poses[i].shape[0]) < 20:
            continue
        clip = clip_arrays(te, i, norm)
        out[sid] = predict_yaw(model, norm, clip, device)
    return out

def rotate_joints(pred, root_orient, yaw_hat, frames):
    out = pred.copy()
    psi_pose = yaw_of_axisangle(root_orient)
    for t in frames:
        k = min(int(t), len(yaw_hat) - 1)
        p = min(int(t), len(psi_pose) - 1)
        d = float(wrap(yaw_hat[k] - psi_pose[p]))
        c, s = np.cos(d), np.sin(d)
        v = out[t] - out[t, 0]
        out[t, :, 0] = c * v[:, 0] + s * v[:, 2] + out[t, 0, 0]
        out[t, :, 2] = -s * v[:, 0] + c * v[:, 2] + out[t, 0, 2]
    return out

def one_scenario(fold, cond, yaw_hats, device):
    z = np.load(CACHE / ("tr_%s.npz" % fold))
    proto = json.loads((PROTO / cond / ("%s.json" % fold)).read_text())
    rows = []
    for e in proto["evaluation_contract"]["sessions"]:
        sid = str(e["session_id"])
        npz = MOTION / fold / ("%s_T2M.npz" % sid)
        if sid not in z.files or sid not in yaw_hats or not npz.is_file():
            continue
        yaw_hat, gt_yaw = yaw_hats[sid]
        d = np.load(npz)
        disp = d["joint_xyz_world"]
        native = np.stack((disp[..., 0], disp[..., 2], -disp[..., 1]), -1)
        frames = np.asarray(d["frame_indices"], int)
        pred = rotate_joints(native, np.asarray(d["root_orient"]), yaw_hat, frames)
        gt = PC.load_gt_joints(sid)
        m = metrics_for_session(sid, fold, np.asarray(z[sid], np.float64), gt,
                                pred, d["frame_indices"], d["valid_mask"],
                                e["frame_ids"])
        # yaw ROD over the valid protocol frames
        n = min(len(yaw_hat), len(gt_yaw))
        fid = np.asarray(e["frame_ids"], int)
        fid = fid[(fid >= 0) & (fid < n)]
        m["RootOrientationDrift"] = float(
            np.abs(wrap(yaw_hat[fid] - gt_yaw[fid])).mean() * 180.0 / np.pi)
        rows.append(m)
    return rows

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_scen = {}
    for fold in FOLDS:
        yaw_hats = load_yaw(fold, device)
        print(fold, "yaw sessions", len(yaw_hats), flush=True)
        for cond in ("clean",) + SCEN:
            rows = one_scenario(fold, cond, yaw_hats, device)
            if not rows:
                continue
            w = np.asarray([r["n_valid"] for r in rows], float)
            all_scen.setdefault(cond, {})[fold] = {
                k: float(np.average([r[k] for r in rows], weights=w)) for k in KEYS}
    out = {}
    for cond, per in all_scen.items():
        mean = {k: float(np.mean([per[f][k] for f in per])) for k in KEYS}
        std = {k: float(np.std([per[f][k] for f in per])) for k in KEYS}
        out[cond] = {"per_fold": per, "mean": mean, "std": std}
        print(cond, {k: round(mean[k], 3) for k in KEYS})
    (MOTION / "V8_IMU_Residual_TBiFoot_yaw_all.json").write_text(json.dumps(out, indent=1))
    print("wrote", MOTION / "V8_IMU_Residual_TBiFoot_yaw_all.json")

if __name__ == "__main__":
    main()
