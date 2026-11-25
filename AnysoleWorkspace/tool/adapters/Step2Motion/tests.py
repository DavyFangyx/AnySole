"""Current Step2Motion four-fold data acceptance checks (SMPL-24 / 50 channels).

Run in touch_gait from the repository root. --skip-heavy omits the single
training-batch smoke; dataset, split, normalizer and metric checks still run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
STEP2MOTION_SRC = ROOT / "Baselines/Step2Motion/src"
for path in (ROOT, STEP2MOTION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AnysoleWorkspace.tool.adapters.Step2Motion import build_gait as bg

FOLDS = {
    "fS5S6S11": ("splits_test_S5S6S11_val_S12.csv", (62, 12, 66)),
    "fS5S11S12": ("splits_test_S5S11S12_val_S13.csv", (64, 18, 58)),
    "fS6S7": ("splits_test_S6S7_val_S14.csv", (73, 18, 31)),
    "fS5S6S8": ("splits_test_S5S6S8_val_S7.csv", (78, 11, 51)),
}
ADAPTER_BASE = ROOT / "AnysoleWorkspace/model_inputs/Step2Motion/adapter_v3_folds"
MANIFEST = ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl"
_PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED
    if not condition:
        raise AssertionError(name + " FAILED: " + detail)
    _PASSED += 1


def _load_native():
    """Import the pristine ``dataset`` module (module name ``dataset``).

    Python 3.8 cannot import it as-is (``list[int]`` annotations without
    ``from __future__ import annotations``; ~/.local pymotion shadow), so the
    module source is compiled with the future import and the user site is
    filtered out.  The produced .pt files pickle against ``dataset.MotionDataset``,
    so this module entry must carry that name.
    """
    kept = []
    for path in sys.path:
        normalized = path.replace("\\", "/")
        if "/.local/lib/python" in normalized and normalized.rstrip("/").endswith("site-packages"):
            continue
        kept.append(path)
    sys.path[:] = kept
    for name in list(sys.modules):
        if name == "pymotion" or name.startswith("pymotion."):
            del sys.modules[name]
    if "dataset" in sys.modules:
        return sys.modules["dataset"].MotionDataset
    src = STEP2MOTION_SRC / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset", str(src))
    mod = importlib.util.module_from_spec(spec)
    code = src.read_text(encoding="utf-8")
    if not code.startswith("from __future__ import annotations"):
        code = "from __future__ import annotations\n" + code
    exec(compile(code, str(src), "exec"), mod.__dict__)
    sys.modules["dataset"] = mod
    return mod.MotionDataset


def test_fold(fold: str, filename: str, expected_counts: tuple) -> None:
    import torch
    MotionDataset = _load_native()
    root = ADAPTER_BASE / fold
    splits = bg.read_split(ROOT / "AnysoleWorkspace/protocol/splits/default" / filename)
    payload = json.loads((root / "gait/split_sessions.json").read_text())
    artifact = json.loads((root / "artifact.json").read_text())
    manifest = {r["session_id"]: r for r in map(json.loads, MANIFEST.read_text().splitlines()) if r}
    check(fold + ": declared counts", tuple(len(splits[k]) for k in ("train", "val", "test")) == expected_counts)
    check(fold + ": sessions recorded", payload["sessions"] == splits)
    subjects = {k: {manifest[s]["subject_id"] for s in splits[k]} for k in splits}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(fold + ": disjoint " + a + "/" + b, not subjects[a].intersection(subjects[b]))
    info = artifact["parameters"]["normalizers"]["gait"]
    train_path = root / "gait/gait_train.pt"
    check(fold + ": train-only normalizer source", Path(info["source"]).resolve() == train_path.resolve())
    normalizer_path = root / "gait/normalizer_gait.pth"
    check(fold + ": normalizer output", Path(info["output"]).resolve() == normalizer_path.resolve())
    normalizer = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    check(fold + ": pose normalizer dimension", normalizer.poses_mean.numel() == 72)
    check(fold + ": condition normalizer dimension", normalizer.insoles_mean.numel() == 50)
    skipped = set(payload.get("empty_or_skipped", {}))
    for split in ("train", "val", "test"):
        db = MotionDataset.load(str(root / "gait" / ("gait_" + split + ".pt")), torch.device("cpu"))
        actual = set(db.session_ids)
        declared = set(splits[split])
        check(fold + "/" + split + ": no undeclared sessions", actual <= declared)
        check(fold + "/" + split + ": missing sessions explicitly recorded", declared - actual == declared.intersection(skipped))
        check(fold + "/" + split + ": SMPL24 names", list(db.joint_names) == bg.SRC_JOINTS)
        check(fold + "/" + split + ": SMPL24 parents", np.array_equal(db.parents.numpy(), bg.SRC_PARENTS))
        check(fold + "/" + split + ": offsets", tuple(db.offsets.shape)[1:] == (24, 3))
        check(fold + "/" + split + ": distances", tuple(db.distances.shape)[1:] == (23,))
        check(fold + "/" + split + ": rate", float(db.sample_rate) == 40.0)
        check(fold + "/" + split + ": IMU", bool(db.has_imu))
        if split == "train":
            check(fold + ": no validation/test normalizer samples", not actual.intersection(set(splits["val"]) | set(splits["test"])))
        for i in range(db.n_clips()):
            pose, cond, quat = db.poses[i], db.insole[i], db.quats[i]
            ids = np.asarray(db.clip_frame_ids[i])
            valid = np.asarray(db.clip_valid[i])
            check(fold + "/" + split + ": pose shape", pose.ndim == 2 and pose.shape[-1] == 72)
            check(fold + "/" + split + ": condition shape", cond.ndim == 2 and cond.shape == (len(pose), 50))
            check(fold + "/" + split + ": quaternion shape", tuple(quat.shape) == (len(pose), 24, 4))
            check(fold + "/" + split + ": row counts", len(ids) == len(valid) == len(pose))
            check(fold + "/" + split + ": contiguous frame IDs", ids.size > 0 and np.all(np.diff(ids) == 1))
            check(fold + "/" + split + ": validity", set(np.unique(valid)) <= {0, 1})
            check(fold + "/" + split + ": finite data", bool(torch.isfinite(pose).all() and torch.isfinite(cond).all() and torch.isfinite(quat).all()))
        # Pressure values remain tied to raw facts through the original frame IDs.
        sid = db.session_ids[0]
        session = bg.load_shared_session(sid)
        ids = np.asarray(db.clip_frame_ids[0], dtype=np.int64)
        observed = db.insole[0].numpy()
        left = bg.pool_48_to_16(np.asarray(session["pressure"]["left48"], dtype=np.float32))
        right = bg.pool_48_to_16(np.asarray(session["pressure"]["right48"], dtype=np.float32))
        check(fold + "/" + split + ": left pressure parity", np.allclose(observed[:, :16], left[ids], atol=1e-5))
        check(fold + "/" + split + ": right pressure parity", np.allclose(observed[:, 25:41], right[ids], atol=1e-5))
        print("DATASET_OK", fold, split, "declared=" + str(len(declared)), "scored=" + str(len(actual)), "clips=" + str(db.n_clips()))


def test_public_metrics_zero() -> None:
    print("== pred=GT SMPL-24 public metrics zero + name-based foot groups ==")
    # 公共评估层 2026-10-12 起为 Baselines 库内真实目录 Baselines/utils（H13 执行，
    # 符号链接桥与 Baselines_old 移除）；直接导入，不再需要别名 shim。
    # 2026-11-24：bvh23 历史协议退役，本测试改走 SMPL-24 口径（不变量不变）。
    from Baselines.utils.protocols import SMPL_JOINT_NAMES, PROTOCOL_FORWARD_AXIS, PROTOCOL_FOOT_NAMES
    from Baselines.utils.solver import metrics as public_metrics

    n = 200
    fps = 40.0
    t = np.arange(n, dtype=np.float64) / fps
    names = SMPL_JOINT_NAMES
    n_joints = len(names)
    rots = np.tile(np.eye(3), (n, n_joints, 1, 1))

    def run(root: np.ndarray) -> dict:
        jitter = np.random.default_rng(7).normal(0, 1e-6, (n, n_joints, 3))
        joints = root[:, None, :] + np.zeros((n, n_joints, 3)) + jitter
        return public_metrics(
            joints.copy(), joints.copy(), np.ones(n, dtype=bool), fps,
            protocol="smpl24", names=names,
            pred_rotations=rots, gt_rotations=rots,
            times=t, forward_axis=PROTOCOL_FORWARD_AXIS["smpl24"],
        )

    # Static trajectory: every metric that compares motion must be zero.
    static = run(np.zeros((n, 3)))
    for key, tol in (("mpjpe_mm", 1e-3), ("pa_mpjpe_mm", 1e-3), ("w_mpjpe100_mm", 1e-3),
                     ("wa_mpjpe100_mm", 1e-3), ("root_ate_mm", 1e-3),
                     ("root_orientation_deg", 1e-6),
                     ("root_orientation_drift_deg", 1e-6), ("mpjae_deg", 1e-6),
                     ("foot_sliding_mm", 1e-2), ("accel_error_m_s2", 1e-3)):
        check(f"pred=GT static {key} zero", abs(float(static[key])) <= tol, f"{static[key]}")
    # jitter is a finite-difference measure of the synthetic noise itself:
    # with identical trajectories the two sides must agree exactly.
    check("pred=GT static jitter_pred == jitter_gt",
          float(static["jitter_pred_m_s3"]) == float(static["jitter_gt_m_s3"]),
          f"{static['jitter_pred_m_s3']} vs {static['jitter_gt_m_s3']}")
    # Moving trajectory: root-relative errors must still be zero; foot
    # sliding then equals the real per-frame motion (2 mm/frame), so only the
    # similarity-fit RTE is checked here.
    moving = run(np.stack([np.linspace(0, 0.4, n), np.zeros(n),
                           0.9 + 0.05 * np.sin(t * 2)], axis=1))
    check("pred=GT moving root_rte_percent zero",
          abs(float(moving["root_rte_percent"])) <= 1e-3, f"{moving['root_rte_percent']}")
    check("pred=GT moving root_ate_mm zero",
          abs(float(moving["root_ate_mm"])) <= 1e-3, f"{moving['root_ate_mm']}")
    # foot groups are selected by SMPL-24 joint names: shuffled names must give
    # the same result (selection by name), missing names must raise.
    foot_ids = [names.index(j) for j in PROTOCOL_FOOT_NAMES["smpl24"]]
    check("smpl24 foot names resolve", foot_ids == [7, 8, 10, 11], str(foot_ids))
    from Baselines.utils.solver import _foot_joint_indices
    shuffled = list(names)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]  # swap pelvis/left_hip positions
    ids = _foot_joint_indices(tuple(shuffled), "smpl24")
    check("foot selection follows names, not positions",
          ids.tolist() == foot_ids, str(ids.tolist()))
    bad_names = tuple(name + "_x" for name in names)
    try:
        _foot_joint_indices(bad_names, "smpl24")
        raise AssertionError("expected ValueError for unknown SMPL-24 foot names")
    except ValueError:
        check("unknown foot names raise", True)

def test_train_smoke() -> None:
    print("== one-batch train smoke (gait, canonical config) ==")
    import torch
    import prefer_env_site  # noqa: F401  # must run before pymotion imports
    from config import load_config
    from dataset import MotionDataset
    from forward_diffusion import ForwardDiffusion
    from backward_diffusion import model_from_config
    from losses import PoseLoss
    from utils import data_augmentation
    from workspace import normalizer_path

    torch.manual_seed(0)
    config = load_config(str(ROOT / "Baselines/Step2Motion/configs/config_gait.json"))
    normalizer = torch.load(normalizer_path(config), weights_only=False)
    db = MotionDataset.load(config["train_data"], torch.device("cpu"))
    config["skeleton_joint_names"] = list(db.joint_names)
    config["pose_layout"] = "semantic_v1"
    db.set_temporality(config["input_T"])
    db.set_stride(config["input_stride"])
    loader = torch.utils.data.DataLoader(db, batch_size=16, shuffle=True)
    forward_diffusion = ForwardDiffusion(T=config["diffusion_T"])
    bw_pose, model_trans = model_from_config(config, is_prior=False)
    bw_prior, _ = model_from_config(config, is_prior=True)
    assert model_trans is not None
    optimizer = torch.optim.Adam(list(bw_prior.parameters()) + list(bw_pose.parameters()),
                                 lr=config["lr"])
    loss_fn = PoseLoss(torch.device("cpu"))
    for step, (x_0, c, distances, parents) in enumerate(loader):
        x_0, c = data_augmentation(x_0, c, db)
        x_0 = normalizer.normalize_poses(x_0)
        c = normalizer.normalize_insole(c)
        distances = normalizer.normalize_distances(distances)
        x_0 = x_0[..., 3:]
        t = torch.randint(0, forward_diffusion.T + 1, (x_0.shape[0],))
        c_mask = (torch.rand(x_0.shape[0]) > config["p_uncond"]).float()
        x_t, noise = forward_diffusion(x_0, t)
        prediction, _ = bw_prior(x=x_t, distances=distances, t=t, control=bw_pose, c=c, c_mask=c_mask)
        loss = loss_fn(prediction, x_0, c)
        check("one-batch loss finite", bool(torch.isfinite(loss)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bw_prior.parameters(), config["clip_grad_value"])
        torch.nn.utils.clip_grad_norm_(bw_pose.parameters(), config["clip_grad_value"])
        optimizer.step()
        check("one-batch backward+step ok", True)
        break
    check("train dataset temporality window == 100", db.temporality == 100)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true", help="Skip the one-batch training smoke.")
    args = parser.parse_args()
    for fold, (filename, counts) in FOLDS.items():
        test_fold(fold, filename, counts)
    test_public_metrics_zero()
    if not args.skip_heavy:
        test_train_smoke()
    print("current_fourfold_data_checks_passed=" + str(_PASSED))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
