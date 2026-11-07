"""临时：多相机 + 标定直接反投影（无学习），cam3 遮挡事件内轨迹。

每帧从 cam2/cam4 的人体 mask 取最低点像素，按相机标定反投影到地面 z=0
（checkerboard 世界，z 负向上），两相机平均 → 水平轨迹；事件起点按 GT
前 20 帧方向对齐（模拟 VT 交接），报遮挡内 ATE。
"""
import json
import os
import sys

import numpy as np
import cv2

ROOT = "/data/fangyuxuan/projects/gait"
MANI = ROOT + "/results/diagnostics/20261107_e5_sceneocc_full/manifests"
SIDS = ["S14033", "S14091", "S14092", "S14093", "S13082", "S13092", "S13093"]
CAL = json.load(open(ROOT + "/AnysoleWorkspace/hf-raw/config/calibration/20260810.json"))


def load_masks(sid, cam):
    out = {}
    for line in open(ROOT + "/AnysoleWorkspace/shared/frontends/human_masks/sam31/v1/index.jsonl"):
        r = json.loads(line)
        if r["session"] == sid and r["camera"] == cam:
            out[int(r["frame_id"])] = r["mask_path"]
    return out


def bottom_pixel(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    v = int(np.percentile(ys, 99.5))
    sel = ys >= v - 3
    if sel.sum() == 0:
        sel = ys == ys.max()
    return float(xs[sel].mean()), float(ys[sel].mean())


def ground_point(u, v, cam):
    c = CAL["cameras"][cam]
    K = np.array(c["K"]); D = np.array(c["D"]); R = np.array(c["R"])
    t = np.array(c["t"]) / 100.0  # 标定 t 为 cm；世界为 m
    pts = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, D).reshape(2)
    ray_c = np.array([pts[0], pts[1], 1.0])
    ray_w = R.T @ ray_c
    C = -R.T @ t
    if abs(ray_w[2]) < 1e-9:
        return None
    lam = -C[2] / ray_w[2]
    P = C + lam * ray_w
    return P


def traj_from_cams(sid, frames, cams=("cam2", "cam4"), smooth=21):
    masks = {c: load_masks(sid, c) for c in cams}
    out = []
    for f in frames:
        ps = []
        for c in cams:
            mp = masks[c].get(int(f))
            if not mp or not os.path.isfile(mp):
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
            bp = bottom_pixel(m)
            if bp is None:
                continue
            P = ground_point(bp[0], bp[1], c)
            if P is not None:
                ps.append(P[:2])
        out.append(np.mean(ps, 0) if ps else np.array([np.nan, np.nan]))
    out = np.asarray(out)
    if smooth and len(out) > smooth:
        for k in range(2):
            v = out[:, k].copy()
            ok = np.isfinite(v)
            if ok.sum() > smooth:
                v = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
                out[:, k] = np.array([np.median(v[max(0, i - smooth // 2):i + smooth // 2 + 1])
                                      for i in range(len(v))])
    return out


def align_start_err(tr, gt2, n_align=20):
    """用前 n_align 帧的位移方向 + 首点位置对齐，返回逐帧误差。"""
    ok = np.isfinite(tr).all(1)
    if ok.sum() < n_align + 2:
        return None
    i0 = int(np.where(ok)[0][0])
    a = tr[min(i0 + n_align, len(tr) - 1)] - tr[i0]
    b = gt2[min(i0 + n_align, len(gt2) - 1)] - gt2[i0]
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return None
    th = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]])
    z = (tr - tr[i0]) @ rot.T + gt2[i0]
    d = np.linalg.norm(z - gt2, axis=1)
    return d[ok]


def main():
    for sid in SIDS:
        man = json.load(open(os.path.join(MANI, sid + ".json")))
        n = man["n_frames"]
        zvt = np.load(ROOT + "/results/diagnostics/20261108_traj_sceneocc/scene7/eval_motion/%s_VT2M.npz" % sid)
        gt = zvt["gt_pelvis_trans"].astype(np.float64)[:, [0, 2]]
        for ev in man["full_events"]:
            a, b = int(ev["start"]), int(ev["end"])
            frames = list(range(a, b + 1))
            tr = traj_from_cams(sid, frames)
            err = align_start_err(tr, gt[a:b + 1])
            if err is None:
                print(sid, ev, "align fail")
                continue
            print("%s act%s [%d,%d] n=%d  geom ATE %.1f mm  (median %.1f)"
                  % (sid, man["action"], a, b, len(frames),
                     float(np.sqrt(np.mean(err ** 2)) * 1000),
                     float(np.median(err) * 1000)))


if __name__ == "__main__":
    main()
