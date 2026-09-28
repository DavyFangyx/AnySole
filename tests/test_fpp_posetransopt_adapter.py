#!/usr/bin/env python3
"""FPP-Net / PoseTransOpt adapter tests (Agent E, 2026-09-27).

Run under the ``mmvp`` environment from the repository root:

    /data/fangyuxuan/miniconda3/envs/mmvp/bin/python tests/test_fpp_posetransopt_adapter.py [--skip-gpu] [--session S11013] [--split test]

Tests:

  T1  FPP dataset one-batch: GT shape (B,192), native binary vertex contact,
      shared frame id, no f6_soft dependency.
  T2  Native contact intra-foot statistics (four-zone variation, intra-foot
      distinct values, no foot-level broadcast chain).
  T3  Public 31x11 insole parity against the audited 4x12->31x11 mapping.
  T4  No private 31x11 copy under model_inputs/FPP-Net.
  T5  FPP contact loss one-step finite.
  T6  Frame-id join report for gap sessions (zero silent misalignment).
  T7  CLIFF NPZ contract (one row per canonical frame, unique frame ids).
  T8  FPP V2T one-session export passes the standard archive validator.
  T9  PoseTransOpt/V2M one-session smoke (if inputs exist).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "AnysoleWorkspace"
FPP_ROOT = REPO_ROOT / "Baselines/VP-MoCap/FPP-Net"
PTO_ROOT = REPO_ROOT / "Baselines/VP-MoCap/PoseTransOpt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# S14011 is a canonical test session with invalid-frame gaps: the smoke
# session exercises both the dataset and the PoseTransOpt join path.
SMOKE_SESSION = "S14011"
GAP_SESSIONS = (
    "S11013", "S11062", "S12011", "S12102", "S14011", "S14013", "S14022",
    "S14062", "S14063", "S14091", "S5011", "S5012", "S5062", "S5091",
    "S6011", "S6013", "S6022", "S6093", "S7053", "S8021",
)


def manifest_rows() -> dict[str, dict]:
    rows = {}
    for line in (WORKSPACE / "protocol/manifests/session_manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["session_id"]] = row
    return rows


def split_sessions(name: str) -> list[str]:
    cols = ("train", "val", "test") if name == "all" else (name,)
    result = []
    with (WORKSPACE / "protocol/splits/default/splits.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.extend((row.get(col) or "").strip() for col in cols)
    return sorted(set(x for x in result if x))


def session_parts(row: dict) -> tuple[str, str, str]:
    from AnysoleWorkspace.tool.workspace import resolve_uri
    parts = resolve_uri(row["video_path"], must_exist=True).parts
    return parts[-3], parts[-2], row["session_id"]


def make_dataset(phase: str = "test"):
    sys.path.insert(0, str(FPP_ROOT))
    from lib.config.config import config_cont as config
    cfg = config()
    cfg.load(str(FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"))
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.dataset.aug.is_aug = False
    cfg.freeze()
    from lib.Dataset.PressDataset.PED_tempKPCont import ContDataset
    return ContDataset(cfg.dataset, phase)


def import_adapter(relative: str, name: str):
    """Import one adapter module (the adapter dirs contain hyphens)."""
    import importlib.util
    path = WORKSPACE / "tool/adapters" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def t1_dataset_one_batch() -> None:
    dataset = make_dataset("test")
    index = 0
    item = dataset[index]
    contact_smpl = np.asarray(item["contact_smpl"])
    assert contact_smpl.shape == (192,), f"GT shape {contact_smpl.shape} != (192,)"
    assert set(np.unique(contact_smpl)) <= {0.0, 1.0}, \
        f"contact GT not binary: {np.unique(contact_smpl)}"
    assert int(item["frame_id"]) >= 0, "missing shared frame id"
    assert np.asarray(item["insole"]).shape == (484,)
    assert tuple(item["keypoints"].shape) == (5, 26, 3)
    # f6_soft dependency must be gone: the dataset loads a real sample without
    # any contact_f6_soft.npy file existing anywhere on the FPP input tree.
    f6_files = list(Path(str(dataset.basedir)).rglob("contact_f6_soft.npy"))
    assert not f6_files, f"f6_soft dependency not removed: {f6_files}"
    print(f"T1 ok: frame {item['frame_id']} contact GT (192,) binary, "
          f"unique={sorted(np.unique(contact_smpl).tolist())}")


def t2_native_contact_stats() -> None:
    dataset = make_dataset("test")
    sid = SMOKE_SESSION
    indices = [i for i, c in enumerate(dataset.images_fn) if f"/{sid}/" in c]
    assert indices, f"{sid} has no cases in the dataset"
    n_frames = min(120, len(indices))
    zone_change = np.zeros(4, dtype=int)
    intra_foot_diff_frames = 0
    any_two_valued = 0
    for k in indices[:n_frames]:
        item = dataset[k]
        contact = np.asarray(item["contact_smpl"]).reshape(2, 96)
        for foot in range(2):
            toe, heel = contact[foot, :48], contact[foot, 48:]
            if toe.size and np.unique(toe).size > 1:
                zone_change[foot * 2] += 1
            if heel.size and np.unique(heel).size > 1:
                zone_change[foot * 2 + 1] += 1
            if np.unique(contact[foot]).size > 1:
                intra_foot_diff_frames += 1
        if np.unique(contact).size > 1:
            any_two_valued += 1
    print(f"T2 stats over {n_frames} frames of {sid}: zone two-valued counts "
          f"(L toe/L heel/R toe/R heel)={zone_change.tolist()}, "
          f"intra-foot two-valued frames={intra_foot_diff_frames}, "
          f"any-two-valued={any_two_valued}")
    # a broadcast chain would make every intra-foot vertex equal the same
    # foot-level scalar in every frame; the native GT must show intra-foot
    # variation (vertices differing within one foot)
    assert intra_foot_diff_frames > 0, "contact GT looks broadcast (no intra-foot variation)"
    assert any_two_valued > 0, "contact GT is constant across frames"


def t3_insole_parity() -> None:
    from AnysoleWorkspace.tool import build_shared
    rows = manifest_rows()
    date, subject, sid = session_parts(rows[SMOKE_SESSION])
    pressure = np.load(
        WORKSPACE / "shared/facts/sessions/cam3" / date / subject / sid
        / "pressure_48.npz", allow_pickle=True)
    maps = {foot: build_shared.mmvp_map(foot) for foot in ("L", "R")}
    checked = 0
    for index in (0, 1, 2, 100, len(pressure["frame_id"]) - 1):
        feet = []
        for foot, key in (("L", "left48"), ("R", "right48")):
            indices, mask = maps[foot]
            grid = np.zeros((31, 11), dtype=np.float32)
            grid[mask] = pressure[key][index][indices[mask]]
            feet.append(grid)
        expected = np.stack(feet, axis=0)
        actual = np.load(
            WORKSPACE / "shared/representations/tactile/mmvp_31x11/v1"
            / date / subject / sid / "insole" / f"{index:06d}.npy")
        assert actual.shape == (2, 31, 11), f"bad public insole shape {actual.shape}"
        diff = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
        assert np.array_equal(actual, expected), \
            f"insole parity mismatch at frame {index} (max diff {diff})"
        checked += 1
    print(f"T3 ok: {checked} frames of {sid} bitwise-identical to the audited "
          f"4x12->31x11 mapping")


def t4_no_private_insole_copy() -> None:
    fpp_inputs = WORKSPACE / "model_inputs/FPP-Net/adapter_v1"
    private = list(fpp_inputs.rglob("insole/*.npy"))
    assert not private, f"private 31x11 copies exist: {private[:5]}"
    print("T4 ok: model_inputs/FPP-Net contains no private insole tree")


def t5_contact_loss_one_step() -> None:
    sys.path.insert(0, str(FPP_ROOT))
    import torch
    from lib.config.config import config_cont as config
    from lib.Dataset import make_dataset as make_ds
    from lib.Networks import make_network
    from lib.Record.record import ContRecorder
    from lib.Trainer.trainer_tempkpSMPLCont import ContTrainer

    cfg = config()
    cfg.load(str(FPP_ROOT / "configs/temporalKPSMPLCont_series5_mlp.yaml"))
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.dataset.aug.is_aug = False
    # dataset/network/trainer paths in the config are relative to the
    # FPP-Net working directory
    cfg.dataset.path = str(FPP_ROOT / cfg.dataset.path)
    cfg.networks.path = str(FPP_ROOT / cfg.networks.path)
    cfg.trainer.path = str(FPP_ROOT / cfg.trainer.path)
    tmp = Path(tempfile.mkdtemp(prefix="fpp_onestep_"))
    cfg.logdir = str(tmp / "log")
    cfg.checkpoint_path = str(tmp / "ckpt")
    cfg.result_path = str(tmp / "res")
    cfg.freeze()

    dataset = make_ds.make_dataset(cfg.dataset, phase="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                             num_workers=0)
    net = make_network.make_network(cfg.networks)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.trainer.lr)
    recorder = ContRecorder(cfg)
    trainer = ContTrainer(dataloader, net, optimizer, recorder, None, cfg.trainer)
    batch = next(iter(dataloader))
    for key in trainer.to_cuda:
        batch[key] = batch[key].to(device=trainer.device)
    pred_press, pred_cont = net(keypoints=batch["keypoints"])
    pred_press = pred_press.squeeze(-1)
    pred_cont = pred_cont.squeeze(-1)
    loss_press = trainer.w_press * torch.nn.functional.mse_loss(pred_press, batch["insole"])
    loss_cont = trainer.w_cont * torch.nn.functional.binary_cross_entropy(
        pred_cont, batch["contact_smpl"])
    loss = loss_press + loss_cont
    assert torch.isfinite(loss), f"contact loss not finite: {loss}"
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    grads = [p.grad.abs().max().item() for p in net.parameters() if p.grad is not None]
    assert all(np.isfinite(g) for g in grads), "non-finite gradients"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"T5 ok: one-step loss={float(loss):.6f} "
          f"(press {float(loss_press):.6f}, cont {float(loss_cont):.6f}) "
          f"max|grad|={max(grads):.4e} finite")


def t6_join_gap_sessions() -> None:
    pt_build = import_adapter("PoseTransOpt/build_inputs.py", "pt_build_inputs")
    rows = manifest_rows()
    checked = 0
    for sid in GAP_SESSIONS:
        if sid not in rows:
            continue
        date, subject, _ = session_parts(rows[sid])
        session_root = (WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1"
                        / date / subject / sid)
        if not (session_root / "CLIFF_results.npz").is_file():
            continue
        manifest = pt_build.build_join(rows[sid], force=False)
        mismatches = []
        for stream in ("keypoints", "fpp_contact"):
            mismatches += manifest["dropped"]["missing_per_stream"][stream].get(
                "frame_id_mismatch", [])
        assert not mismatches, f"{sid}: silent frame-id mismatches {mismatches}"
        # every joined frame is valid in shared facts and CLIFF
        joined = np.asarray(manifest["joined_frames"], dtype=np.int64)
        frames = np.load(WORKSPACE / "shared/facts/sessions/cam3" / date
                         / subject / sid / "frames.npz")
        valid = np.asarray(frames["valid"], dtype=bool)
        fake = np.asarray(frames["fake"], dtype=bool)
        assert not np.any(fake[joined]) and np.all(valid[joined]), \
            f"{sid}: joined frames contain fake/invalid frames"
        checked += 1
    print(f"T6 ok: {checked} gap sessions join on frame id with zero silent "
          f"misalignment (of {len(GAP_SESSIONS)} candidate sessions)")


def t7_cliff_contract(split: str) -> None:
    rows = manifest_rows()
    sessions = split_sessions(split)
    checked = 0
    for sid in sessions:
        date, subject, _ = session_parts(rows[sid])
        path = (WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1" / date
                / subject / sid / "CLIFF_results.npz")
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=False)
        n = int(rows[sid]["n_frames"])
        frame_id = np.asarray(data["frame_id"])
        assert len(frame_id) == n, f"{sid}: rows {len(frame_id)} != {n}"
        assert len(np.unique(frame_id)) == n, f"{sid}: duplicate frame ids"
        assert np.array_equal(frame_id, np.arange(n, dtype=np.int64)), \
            f"{sid}: frame ids not canonical"
        valid = np.asarray(data["valid"], dtype=np.uint8).astype(bool)
        # CLIFF must never be valid where shared facts mark invalid/fake
        facts = np.load(WORKSPACE / "shared/facts/sessions/cam3" / date
                        / subject / sid / "frames.npz")
        shared_valid = np.asarray(facts["valid"], dtype=np.uint8).astype(bool)
        assert np.all(valid <= shared_valid), \
            f"{sid}: CLIFF valid where shared facts invalid"
        # CLIFF validity must equal its own mask-recompute criteria
        bbox_ok = np.any(np.asarray(data["bbox_xyxy_px"])[:, 2:] > 0, axis=1)
        time_ok = np.asarray(data["time_error_s"]) <= 0.020
        assert np.array_equal(valid, bbox_ok & time_ok), \
            f"{sid}: CLIFF valid != (nonzero mask bbox & <=20ms)"
        for key in ("pose", "shape", "global_t"):
            assert np.asarray(data[key]).shape[0] == n, f"{sid}: {key} rows"
        checked += 1
    print(f"T7 ok: {checked}/{len(sessions)} CLIFF NPZ files satisfy the "
          f"one-row-per-frame single-person contract")


def t8_v2t_export(split: str) -> None:
    fpp_export = import_adapter("FPP-Net/export_v2t.py", "fpp_export_v2t")
    from AnysoleWorkspace.tool.validate_baseline_exports import validate_v2t
    rows = manifest_rows()
    session = SMOKE_SESSION
    if session not in rows:
        print(f"T8 skip: {session} not in manifest")
        return
    result = fpp_export.export_session(rows[session], force=True)
    print(result)
    archive = (REPO_ROOT / "results/baselines/FPP-Net/predictions/v2t"
               / f"{session}.npz")
    assert archive.is_file(), f"archive missing: {archive}"
    reason = validate_v2t(archive, int(rows[session]["n_frames"]))
    assert reason is None, f"archive validation failed: {reason}"
    print(f"T8 ok: {archive.name} passes the standard V2T archive validator")


def t9_v2m_smoke(split: str) -> None:
    rows = manifest_rows()
    session = SMOKE_SESSION
    if session not in rows:
        print(f"T9 skip: {session} not in manifest")
        return
    date, subject, _ = session_parts(rows[session])
    inputs = (WORKSPACE / "model_inputs/PoseTransOpt/adapter_v1" / date
              / subject / session)
    if not (inputs / "join_manifest.json").is_file():
        print(f"T9 skip: no join manifest for {session} yet")
        return
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "Baselines/VP-MoCap/PoseTransOpt"
                             / "run_full_mmvp.py"),
         "--sessions", session, "--gpu", "0", "--max-iter", "3",
         "--max-frames", "30", "--no-visualization", "--force",
         "--hydra-root", "/tmp/fpp_posetransopt_smoke"],
        check=True)
    vp_export = import_adapter("PoseTransOpt/export_v2m.py", "vp_export_v2m")
    from AnysoleWorkspace.tool.validate_baseline_exports import validate_motion
    result = vp_export.export_session(rows[session], force=True)
    print(result)
    archive = (REPO_ROOT / "results/baselines/VP-MoCap/predictions/eval_motion"
               / f"{session}.npz")
    assert archive.is_file(), f"archive missing: {archive}"
    reason = validate_motion(archive, int(rows[session]["n_frames"]))
    assert reason is None, f"archive validation failed: {reason}"
    print(f"T9 ok: {archive.name} passes the standard motion archive validator")


def main() -> int:
    global SMOKE_SESSION
    import functools
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=SMOKE_SESSION)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--only", default="", help="comma-separated test ids, e.g. t1,t3")
    args = parser.parse_args()
    SMOKE_SESSION = args.session
    tests = [
        t1_dataset_one_batch, t2_native_contact_stats, t3_insole_parity,
        t4_no_private_insole_copy, t5_contact_loss_one_step,
        t6_join_gap_sessions,
        functools.partial(t7_cliff_contract, args.split),
        functools.partial(t8_v2t_export, args.split),
        functools.partial(t9_v2m_smoke, args.split),
    ]
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        tests = [fn for fn in tests if any(
            (fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__).startswith(w)
            for w in wanted)]
    failures = 0
    for fn in tests:
        name = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
        print(f"\n== {name} ==")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAILED ({exc})")
    print(f"\n{len(tests)} tests, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
