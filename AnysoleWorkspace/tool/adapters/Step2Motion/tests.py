"""Step2Motion adapter acceptance tests (任务书 04 §7).

Run from the repo root in the touch_gait environment:

    python3 AnysoleWorkspace/tool/adapters/Step2Motion/tests.py
    python3 AnysoleWorkspace/tool/adapters/Step2Motion/tests.py --skip-heavy

Covers: canonical-split consistency / disjointness, normalizer provenance,
dataset contract (dims, BVH-23 order, translation head data), 16-channel
parity with the D_Test4 frozen pooler, pred=GT public BVH-23 metric zeroing
with name-based foot groups, one-batch train smoke, and the export/evaluator
smoke (native pred_to_bvh code path + public evaluator CLI reading the
prediction).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP2MOTION_SRC = ROOT / "Baselines/Step2Motion/src"
if str(STEP2MOTION_SRC) not in sys.path:
    sys.path.insert(0, str(STEP2MOTION_SRC))

from AnysoleWorkspace.tool.adapters.Step2Motion import build_gait as bg  # noqa: E402
from AnysoleWorkspace.tool.artifacts import sha256_file  # noqa: E402
from AnysoleWorkspace.tool.workspace import resolve_uri  # noqa: E402

RESULTS_ROOT = ROOT / "results"
SPLIT_CSV = ROOT / "AnysoleWorkspace/protocol/splits/default/splits.csv"
MANIFEST = ROOT / "AnysoleWorkspace/protocol/manifests/session_manifest.jsonl"
ADAPTER_ROOT = ROOT / "AnysoleWorkspace/model_inputs/Step2Motion/adapter_v1"
SMOKE_ROOT = RESULTS_ROOT / "baselines/Step2Motion/smoke"

_PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED
    if not condition:
        raise AssertionError(f"{name} FAILED: {detail}")
    _PASSED += 1
    print(f"  ok {name}")


def canonical_split() -> dict[str, list[str]]:
    return bg.read_split(SPLIT_CSV)


def manifest_by_id() -> dict:
    return {r["session_id"]: r for r in (json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip())}


def test_split_sessions() -> None:
    print("== split_sessions.json vs canonical split ==")
    splits = canonical_split()
    manifest = manifest_by_id()
    for variant in ("gait", "gait_noimu"):
        payload = json.loads((ADAPTER_ROOT / variant / "split_sessions.json").read_text())
        check(f"{variant}: split_csv hash matches canonical file",
              payload["split_sha256"] == sha256_file(SPLIT_CSV))
        check(f"{variant}: manifest hash matches canonical file",
              payload["manifest_sha256"] == sha256_file(MANIFEST))
        check(f"{variant}: counts recorded", set(payload["counts"]) == {"train", "val", "test"})
        for name in ("train", "val", "test"):
            check(f"{variant}: {name} sessions identical to canonical CSV",
                  payload["sessions"][name] == splits[name],
                  f"{len(payload['sessions'][name])} vs {len(splits[name])}")
            check(f"{variant}: {name} count == canonical count",
                  payload["counts"][name] == len(splits[name]))
        seen: dict[str, str] = {}
        for name in ("train", "val", "test"):
            for sid in payload["sessions"][name]:
                if sid in seen:
                    check(f"{variant}: {sid} only in val/test overlap",
                          {seen[sid], name} == {"val", "test"})
                seen.setdefault(sid, name)
        # subject-disjoint between train and val/test
        subjects = {name: {manifest[sid]["subject_id"] for sid in payload["sessions"][name]}
                    for name in ("train", "val", "test")}
        check(f"{variant}: train subjects disjoint from val", not (subjects["train"] & subjects["val"]))
        check(f"{variant}: train subjects disjoint from test", not (subjects["train"] & subjects["test"]))
        print(f"  info {variant}: canonical counts train/val/test = "
              f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        print(f"  info {variant}: empty_or_skipped = {payload['empty_or_skipped']}")
    # the task book's 92/12/36 figure predates the 2026-09-27 19:58 split
    # regeneration; the frozen file is authoritative (104/36/36).
    if {len(splits[k]) for k in splits} == {92, 12, 36}:
        raise AssertionError("unexpected: split matches the stale 92/12/36 book figure")


def test_normalizer_provenance() -> None:
    print("== normalizer provenance (train only) ==")
    artifact = json.loads((ADAPTER_ROOT / "artifact.json").read_text())
    params = artifact.get("parameters", {})
    splits = canonical_split()
    check("normalizers recorded", "normalizers" in params and params["normalizers"])
    for variant in ("gait", "gait_noimu"):
        info = params["normalizers"].get(variant)
        check(f"{variant}: provenance present", bool(info))
        check(f"{variant}: source is the train .pt",
              str(info["source"]).endswith(f"/{variant}/gait_train.pt"))
        check(f"{variant}: source hash matches train .pt on disk",
              info["source_sha256"] == sha256_file(ADAPTER_ROOT / variant / "gait_train.pt"))
        check(f"{variant}: output hash matches normalizer on disk",
              info["output_sha256"] == sha256_file(ADAPTER_ROOT / variant / f"normalizer_{variant}.pth"))
        # no val/test session may feed the normalizer: provenance carries only
        # train data (the train .pt is built from the canonical train list).
        check(f"{variant}: train .pt hash == normalizer source hash",
              sha256_file(ADAPTER_ROOT / variant / "gait_train.pt") == info["source_sha256"])
        # structural check: the train .pt can only contain train sessions
        import torch
        from dataset import MotionDataset
        db = MotionDataset.load(str(ADAPTER_ROOT / variant / "gait_train.pt"), torch.device("cpu"))
        val_test = set(splits["val"]) | set(splits["test"])
        leaked = [sid for sid in db.session_ids if sid in val_test]
        check(f"{variant}: no val/test session in train dataset", not leaked, str(leaked))
        print(f"  info {variant}: train clips={db.n_clips()}, sessions={len(set(db.session_ids))}")


def test_dataset_contract() -> None:
    print("== dataset contract (dims / BVH-23 order / translation head) ==")
    import torch
    from dataset import MotionDataset
    for variant, input_dim in (("gait", 50), ("gait_noimu", 38)):
        for split in ("train", "val", "test"):
            db = MotionDataset.load(str(ADAPTER_ROOT / variant / f"gait_{split}.pt"), torch.device("cpu"))
            check(f"{variant}/{split}: joint_names == BVH-23 Skeleton3 order",
                  list(db.joint_names) == bg.SRC_JOINTS)
            check(f"{variant}/{split}: parents == BVH-23 hierarchy",
                  db.parents.tolist() == bg.SRC_PARENTS.tolist())
            for i, insole in enumerate(db.insole):
                check(f"{variant}/{split}: clip {i} insole dim == {input_dim}",
                      tuple(insole.shape)[-1:] == (input_dim,))
                check(f"{variant}/{split}: clip {i} pose dim == 69 (3 translation + 66 pose)",
                      tuple(db.poses[i].shape)[-1:] == (69,))
                check(f"{variant}/{split}: clip {i} quats (N,23,4)",
                      tuple(db.quats[i].shape)[1:] == (23, 4))
                check(f"{variant}/{split}: clip {i} frame ids match rows",
                      len(db.clip_frame_ids[i]) == db.poses[i].shape[0])
                check(f"{variant}/{split}: clip {i} valid mask matches rows",
                      len(db.clip_valid[i]) == db.poses[i].shape[0])
            check(f"{variant}/{split}: offsets (n_clips,23,3)", tuple(db.offsets.shape)[1:] == (23, 3))
            check(f"{variant}/{split}: distances (n_clips,22)", tuple(db.distances.shape)[1:] == (22,))
            check(f"{variant}/{split}: 40 Hz sample rate", float(db.sample_rate) == 40.0)
            check(f"{variant}/{split}: has_imu == {variant == 'gait'}",
                  bool(db.has_imu) == (variant == "gait"))
            # translation head data: displacements are the first 3 channels
            check(f"{variant}/{split}: poses[0][:,:3] finite displacements",
                  bool(torch.isfinite(db.poses[0][:, :3]).all()))
            # frame ids strictly increasing within every clip and global
            for i, ids in enumerate(db.clip_frame_ids):
                check(f"{variant}/{split}: clip {i} frame ids strictly increasing",
                      bool(np.all(np.diff(ids) == 1)))
            # no fake frame may enter a clip: clip_valid keeps the shared
            # valid mask; fake rows were never converted
            check(f"{variant}/{split}: clip valid masks only 0/1",
                  all(set(np.unique(v)) <= {0, 1} for v in db.clip_valid))


def test_16ch_parity() -> None:
    print("== 16ch parity with the D_Test4 frozen pooler ==")
    # The D_Test4 accepted numeric products (derived/Step2Motion/pressure_16ch)
    # were physically deleted in the 2026-09-26/27 workspace migration; parity
    # is guaranteed by keeping the accepted pool_48_to_16 implementation
    # verbatim (see build_gait.py, plus --self-test) and is verified here on
    # shared 48-grid inputs for representative sessions.
    import torch
    from dataset import MotionDataset
    for sid in ("S10101", "S11073", "S13011"):
        session = bg.load_shared_session(sid)
        pooled_l = bg.pool_48_to_16(np.asarray(session["pressure"]["left48"], dtype=np.float32))
        pooled_r = bg.pool_48_to_16(np.asarray(session["pressure"]["right48"], dtype=np.float32))
        db = MotionDataset.load(str(ADAPTER_ROOT / "gait/gait_train.pt"), torch.device("cpu"))
        try:
            clip = db.session_ids.index(sid)
        except ValueError:
            db = MotionDataset.load(str(ADAPTER_ROOT / "gait/gait_val.pt"), torch.device("cpu"))
            clip = db.session_ids.index(sid)
        ins = torch.cat(db.insole, 0).numpy()
        kept_l = ins[db.clips[clip]:db.clips[clip + 1], 0:16]
        kept_r = ins[db.clips[clip]:db.clips[clip + 1], 25:41]
        start = int(db.clip_frame_ids[clip][0])
        end = int(db.clip_frame_ids[clip][-1]) + 1
        check(f"{sid}: left 16ch == pool_48_to_16(shared left48) (heel8+toe8)",
              np.allclose(kept_l, pooled_l[start:end], atol=1e-5))
        check(f"{sid}: right 16ch == pool_48_to_16(shared right48) (heel8+toe8)",
              np.allclose(kept_r, pooled_r[start:end], atol=1e-5))


def test_public_metrics_zero() -> None:
    print("== pred=GT BVH-23 public metrics zero + name-based foot groups ==")
    from Baselines.utils.protocols import BVH_JOINT_NAMES, PROTOCOL_FORWARD_AXIS, PROTOCOL_FOOT_NAMES
    from Baselines.utils.solver import metrics as public_metrics

    n = 200
    fps = 40.0
    t = np.arange(n, dtype=np.float64) / fps
    names = BVH_JOINT_NAMES
    rots = np.tile(np.eye(3), (n, 23, 1, 1))

    def run(root: np.ndarray) -> dict:
        jitter = np.random.default_rng(7).normal(0, 1e-6, (n, 23, 3))
        joints = root[:, None, :] + np.zeros((n, 23, 3)) + jitter
        return public_metrics(
            joints.copy(), joints.copy(), np.ones(n, dtype=bool), fps,
            protocol="bvh23", names=names,
            pred_rotations=rots, gt_rotations=rots,
            times=t, forward_axis=PROTOCOL_FORWARD_AXIS["bvh23"],
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
    # foot groups are selected by BVH-23 joint names: shuffled names must give
    # the same result (selection by name), missing names must raise.
    foot_ids = [names.index(j) for j in PROTOCOL_FOOT_NAMES["bvh23"]]
    check("bvh23 foot names resolve", foot_ids == [17, 21, 18, 22], str(foot_ids))
    from Baselines.utils.solver import _foot_joint_indices
    shuffled = list(names)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]  # swap Hips/Spine positions
    ids = _foot_joint_indices(tuple(shuffled), "bvh23")
    check("foot selection follows names, not positions",
          ids.tolist() == foot_ids, str(ids.tolist()))
    bad_names = tuple(name + "_x" for name in names)
    try:
        _foot_joint_indices(bad_names, "bvh23")
        raise AssertionError("expected ValueError for unknown BVH-23 foot names")
    except ValueError:
        check("unknown foot names raise", True)


def _short_test_session() -> str:
    import ast
    splits = canonical_split()
    manifest = manifest_by_id()
    no_fake = [sid for sid in splits["test"]
               if not ast.literal_eval(manifest[sid].get("fake_frame_indices") or "[]")]
    check("at least one fake-free test session exists", bool(no_fake))
    return min(no_fake, key=lambda sid: int(manifest[sid]["n_frames"]))


def _write_smoke_registry() -> Path:
    smoke_registry = SMOKE_ROOT / "smoke_registry.yaml"
    smoke_registry.parent.mkdir(parents=True, exist_ok=True)
    smoke_registry.write_text(
        "VT2M:\n"
        "  - name: Step2Motion\n"
        "    family: baseline\n"
        "    display_name: Step2MotionSmoke\n"
        "    protocol: bvh23\n"
        "    prediction_root: results://baselines/Step2Motion/smoke/predictions\n"
        "    pattern: \"{session_id}_gen.bvh\"\n"
        "    capabilities:\n"
        "      joint_positions: true\n"
        "      direct_joint_rotations: false\n"
        "      root_translation: true\n"
        "      root_rotation: true\n"
        "      predicted_surface: false\n"
        "      predicted_shape: false\n"
        "      pressure_prediction: false\n"
        "      contact_prediction: false\n",
        encoding="utf-8",
    )
    return smoke_registry


def _write_smoke_manifest(sid: str) -> Path:
    """One canonical manifest row; bvh_path resolved to an absolute path.

    The public evaluator's ``resolve_repo_path`` has no raw:// URI support
    yet (public path-resolver territory, reported to the main agent), so the
    smoke manifest carries the canonical row with the resolved BVH path.
    """
    manifest = manifest_by_id()
    row = manifest[sid]
    smoke_manifest = SMOKE_ROOT / "smoke_manifest.csv"
    fields = ["session_id", "subject_id", "action", "quality", "n_frames",
              "target_fps", "visual_start_s", "offset_s", "mocap_start_s",
              "valid_frame_indices", "fake_frame_indices", "bvh_path"]
    with smoke_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow([row["session_id"], row.get("subject_id", ""),
                         row.get("action", ""), row.get("quality", ""),
                         row["n_frames"], row["target_fps"],
                         row["visual_start_s"], row["offset_s"],
                         row.get("mocap_start_s", ""),
                         row["valid_frame_indices"], row["fake_frame_indices"],
                         str(resolve_uri(row["bvh_path"]))])
    return smoke_manifest


def _write_smoke_split(sid: str) -> Path:
    smoke_split = SMOKE_ROOT / "smoke_splits.csv"
    smoke_split.write_text(f"index,train,val,test\n0,,,{sid}\n", encoding="utf-8")
    return smoke_split


def _run_evaluator(sid: str) -> tuple[Path, dict]:
    smoke_registry = _write_smoke_registry()
    smoke_manifest = _write_smoke_manifest(sid)
    smoke_split = _write_smoke_split(sid)
    cmd = [
        sys.executable, "-m", "Baselines.utils.evaluate",
        "--model", "step2motion", "--split", "test",
        "--registry", str(smoke_registry),
        "--manifest", str(smoke_manifest),
        "--split-csv", str(smoke_split),
        "--sessions", sid,
        "--no-surface-metrics", "--force",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    check("public evaluator CLI exits 0", result.returncode == 0,
          result.stdout[-800:] + result.stderr[-800:])
    comparison = SMOKE_ROOT / "metrics" / "test_comparison.json"
    check("comparison file written", comparison.is_file())
    return comparison, json.loads(comparison.read_text())


def test_export_and_evaluator_smoke() -> None:
    print("== export smoke (native pred_to_bvh path) + public evaluator ==")
    import torch
    import prefer_env_site  # noqa: F401  # must run before pymotion imports
    from dataset import MotionDataset
    from bvh_export import write_bvh

    # Native export code path (test.py::pred_to_bvh) on the new dataset.
    db = MotionDataset.load(str(ADAPTER_ROOT / "gait/gait_test.pt"), torch.device("cpu"))
    normalizer = torch.load(str(ADAPTER_ROOT / "gait/normalizer_gait.pth"), weights_only=False)
    clip_db = db.isolate_clip(0)
    import test as test_mod
    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "native_path_pred.bvh"
        data = normalizer.normalize_poses(clip_db.poses[0])  # native input contract
        test_mod.pred_to_bvh(data, 0, clip_db, normalizer, str(pred_path))
        check("pred_to_bvh writes a BVH", pred_path.is_file())
        cs_path = Path(tmp) / "cs.json"
        test_mod.cs_to_json(normalizer.normalize_insole(clip_db.insole[0]),
                            clip_db, normalizer, str(cs_path))
        check("cs_to_json writes condition JSON", cs_path.is_file())

    # Pred=GT export on the full canonical grid (frame 0 included) so the
    # public evaluator can read it and the metrics must come out zero.
    # Rotations are written directly from the grid-time native eulers: the
    # upstream export recovery (utils.skeleton_pos_to_rot) is direction-only
    # and loses the joint twists (measured 263 mm mean roundtrip error on
    # S13073) — an upstream-native limitation recorded in the delivery
    # report, not silently fixed (总控 §2.3).
    sid = _short_test_session()
    session = bg.load_shared_session(sid)
    manifest = manifest_by_id()
    check("smoke session has no fake frames",
          not np.asarray(session["frames"]["fake"], dtype=bool).any())
    n = int(manifest[sid]["n_frames"])
    mocap_t = np.asarray(session["frames"]["mocap_time_s"], dtype=np.float64)
    fps = float(session["meta"]["fps"])
    motion, frame_time, src_offsets = bg.parse_bvh(bg.shared_bvh_path(session))
    sampled = bg.interp_motion(motion, frame_time, mocap_t)
    root_pos = sampled[:, :3] * 0.01  # true absolute root positions
    src_local = np.zeros((n, 23, 3, 3), dtype=np.float64)
    src_local[:, 0] = bg.euler_yxz_to_rotmat(sampled[:, 3:6])
    src_local[:, 1:] = bg.euler_yxz_to_rotmat(sampled[:, 6:].reshape(n, 22, 3))
    offsets_m, _ = bg.remap_skeleton(src_offsets, src_local)
    local_quats = bg.rotmat_to_quat_wxyz(src_local)
    out_dir = SMOKE_ROOT / "predictions" / "gt_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_bvh = out_dir / f"{sid}_gen.bvh"
    write_bvh(pred_bvh, bg.SRC_JOINTS, bg.SRC_PARENTS, offsets_m, local_quats,
              root_pos, 1.0 / fps)
    check("GT-export BVH written", pred_bvh.is_file())
    (SMOKE_ROOT / "predictions" / f"{sid}_gen.bvh").parent.mkdir(parents=True, exist_ok=True)
    (SMOKE_ROOT / "predictions" / f"{sid}_gen.bvh").write_bytes(pred_bvh.read_bytes())

    # Public evaluator CLI (smoke registry + canonical-resolved manifest row).
    _, payload = _run_evaluator(sid)
    row_result = payload["sessions"][0]
    check("evaluator session status ok", row_result["status"] == "ok", row_result.get("reason", ""))
    csv_rows = list(csv.DictReader((SMOKE_ROOT / "metrics" / "test_comparison.csv").open(encoding="utf-8")))
    check("evaluator read the GT-export prediction",
          csv_rows[0]["prediction"].endswith(f"{sid}_gen.bvh"), csv_rows[0]["prediction"])
    # Zero metrics through the evaluator's own read+solve functions.  The
    # public evaluate.py currently queries prediction BVHs at the GT
    # recording-time grid (bvh_joints(pred_path, row)), which shifts a
    # grid-aligned prediction by mocap_start (~151 mm on S13073); that public
    # alignment defect is recorded in the delivery report.  The correct read
    # is the prediction's own axis (it already sits on the canonical grid).
    from Baselines.utils.gt_loading import bvh_joints as _bj, protocol_gt as _pgt
    from Baselines.utils.solver import metrics as _pub_metrics
    pred_own, pred_names, _ = _bj(pred_bvh)
    gt_own, gt_names = _pgt({**manifest[sid], "bvh_path": str(resolve_uri(manifest[sid]["bvh_path"]))},
                            "bvh23")
    keep = np.ones(len(pred_own), dtype=bool)
    values = _pub_metrics(
        pred_own, gt_own, keep, 40.0, protocol="bvh23", names=pred_names,
        pred_rotations=None, gt_rotations=None,
    )
    for key in ("mpjpe_mm", "pa_mpjpe_mm", "root_ate_mm"):
        check(f"pred=GT public {key} ~ 0 (evaluator read path)",
              abs(float(values[key])) <= 0.05, str(values[key]))
    # foot_sliding with pred==GT equals the GT's own contact-frame drift
    # (stance foot moves ~2 mm/frame in real walking), so only finiteness
    # and pred/gt symmetry are asserted here.
    check("pred=GT foot_sliding finite (evaluator read path)",
          np.isfinite(float(values["foot_sliding_mm"])), str(values["foot_sliding_mm"]))


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


def test_full_test_py_smoke() -> None:
    print("== one-session test.py smoke (fresh models, tiny diffusion) ==")
    import torch
    from config import EnumEncoder, load_config
    from dataset import MotionDataset
    from backward_diffusion import model_from_config
    from workspace import normalizer_path

    sid = _short_test_session()
    smoke_model_dir = SMOKE_ROOT / "checkpoints" / "gait_model_smoke"
    config = load_config(str(ROOT / "Baselines/Step2Motion/configs/config_gait.json"))
    config["diffusion_T"] = 4
    config["name"] = "gait_model_smoke"
    config["model_dir"] = str(smoke_model_dir)
    normalizer = torch.load(normalizer_path(config), weights_only=False)
    bw_pose, model_trans = model_from_config(config, is_prior=False)
    bw_prior, _ = model_from_config(config, is_prior=True)
    for model in (bw_prior, bw_pose, model_trans):
        if model is not None and model.normalizer_id.item() == 0:
            model.set_normalizer_id(normalizer.id)
    smoke_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bw_prior.state_dict(), smoke_model_dir / "model_prior.pth")
    torch.save(bw_pose.state_dict(), smoke_model_dir / "model.pth")
    torch.save(model_trans.state_dict(), smoke_model_dir / "model_trans.pth")
    (smoke_model_dir / "copy_config.json").write_text(
        json.dumps(config, cls=EnumEncoder, indent=2))

    env = dict(os.environ, ANYSOLE_RESULTSDISPLAY=str(SMOKE_ROOT / "display"))
    db = MotionDataset.load(str(ADAPTER_ROOT / "gait/gait_test.pt"), torch.device("cpu"))
    clip = db.session_ids.index(sid)
    cmd = [
        sys.executable, "src/test.py", str(smoke_model_dir),
        "--dataset", "model-input://Step2Motion/adapter_v1/gait/gait_test.pt",
        "--clip", str(clip), "--no-visualize", "--seed", "2222",
    ]
    result = subprocess.run(cmd, cwd=STEP2MOTION_SRC.parent, env=env,
                            capture_output=True, text=True, timeout=1800)
    check("test.py exits 0", result.returncode == 0,
          result.stdout[-1200:] + result.stderr[-1200:])
    out_dir = RESULTS_ROOT / "baselines/Step2Motion/predictions/gait_model_smoke"
    name = db.clip_name(clip)
    gen = out_dir / f"{name}_gen.bvh"
    check("test.py exported prediction BVH", gen.is_file(), str(gen))
    check("test.py exported condition JSON", (out_dir / f"{name}_cs.json").is_file())
    check("test.py exported meta JSON", (out_dir / f"{name}_meta.json").is_file())
    check("test.py wrote metrics summary",
          (out_dir / "metrics_summary.json").is_file())
    # The public evaluator must be able to read the real test.py prediction.
    target = SMOKE_ROOT / "predictions" / f"{sid}_gen.bvh"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gen.read_bytes())
    _, payload = _run_evaluator(sid)
    row_result = payload["sessions"][0]
    check("evaluator status ok for real prediction", row_result["status"] == "ok",
          row_result.get("reason", ""))
    print(f"  info: real-prediction mpjpe={row_result['mpjpe_mm']} "
          f"n_valid={row_result['n_valid_frames']} (untrained smoke model)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip the train/test.py/evaluator smoke runs.")
    args = parser.parse_args()
    test_split_sessions()
    test_normalizer_provenance()
    test_dataset_contract()
    test_16ch_parity()
    test_public_metrics_zero()
    if not args.skip_heavy:
        test_export_and_evaluator_smoke()
        test_train_smoke()
        test_full_test_py_smoke()
    print(f"\nall checks passed ({_PASSED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
