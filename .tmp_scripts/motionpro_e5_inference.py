#!/usr/bin/env python3
"""MotionPRO E5 per-level, per-fold inference (U@VT).

Fold-specific models (like Step2Motion): predictions are written to

    results/baselines/MotionPRO/predictions/<variant>/<fold>/eval_motion/<sid>.npz

so the four folds never overwrite each other.  Clean variants copy the
existing four-fold clean predictions; degraded variants run test_frappe per
fold with the level feature tree and the fold's LR-floor last checkpoint.

Every finished run is verified (session set, archive loadability, frame/mask
length, checkpoint provenance); terminal failures make the exit code
non-zero so the watcher does not evaluate incomplete output.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/fangyuxuan/projects/gait")
PY = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
RESULTS = ROOT / "results/baselines/MotionPRO"
LOGROOT = ROOT / "results/diagnostics/baselines_fourfold_20261109/motionpro_e5"
GPUS = (2, 3, 6, 7)
MAX_RUNS = 4

FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
FOLD_SPLITS = {
    "fS5S6S11": "splits_test_S5S6S11_val_S12",
    "fS5S11S12": "splits_test_S5S11S12_val_S13",
    "fS6S7": "splits_test_S6S7_val_S14",
    "fS5S6S8": "splits_test_S5S6S8_val_S7",
}
CKPT = {
    "fS5S6S11": "imagepressure2smpl_117.pth",
    "fS5S11S12": "imagepressure2smpl_108.pth",
    "fS6S7": "imagepressure2smpl_126.pth",
    "fS5S6S8": "imagepressure2smpl_142.pth",
}
CKPT_DIR = RESULTS / "checkpoints"

VARIANTS = [
    ("e5_clean", None, "clean"),
    ("e5_occ_lower010", "adapter_v4_anysolebbox_occ010", "occ"),
    ("e5_occ_lower020", "adapter_v4_anysolebbox_occ020", "occ"),
    ("e5_occ_lower030", "adapter_v4_anysolebbox_occ030", "occ"),
    ("e5_occ_lower040", "adapter_v4_anysolebbox_occ040", "occ"),
    ("e5_occ_lower050", "adapter_v4_anysolebbox_occ050", "occ"),
    ("e5_occ_lower060", "adapter_v4_anysolebbox_occ060", "occ"),
    ("e5_occ_lower070", "adapter_v4_anysolebbox_occ070", "occ"),
    ("e5_occ_lower080", "adapter_v4_anysolebbox_occ080", "occ"),
    ("e5_occ_lower090", "adapter_v4_anysolebbox_occ090", "occ"),
    ("e5_gap_r010", "adapter_v4_anysolebbox_gap_r010", "gap"),
    ("e5_gap_r020", "adapter_v4_anysolebbox_gap_r020", "gap"),
    ("e5_gap_r040", "adapter_v4_anysolebbox_gap_r040", "gap"),
    ("e5_gap_r060", "adapter_v4_anysolebbox_gap_r060", "gap"),
    ("e5_scene_clean", None, "scene_clean"),
    ("e5_scene", "adapter_v4_anysolebbox_scene", "scene"),
]


def fold_test_sessions(fold: str, kind: str) -> list[str]:
    if kind in ("scene", "scene_clean"):
        csv_name = f"splits_sceneocc_{fold}"
    else:
        csv_name = FOLD_SPLITS[fold]
    path = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{csv_name}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    sessions = [(r.get("test") or "").strip() for r in rows if (r.get("test") or "").strip()]
    return [s for s in sessions if s != "S12102"]


def out_dir(variant: str, fold: str) -> Path:
    return RESULTS / "predictions" / variant / fold / "eval_motion"


def verify(variant: str, fold: str) -> tuple[bool, str]:
    d = out_dir(variant, fold)
    expected = fold_test_sessions(fold, "scene" if variant.startswith("e5_scene")
                                  else "occ")
    missing = [s for s in expected if not (d / f"{s}.npz").is_file()]
    if missing:
        return False, f"missing {len(missing)}: {missing[:5]}"
    for s in expected:
        p = d / f"{s}.npz"
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            return False, f"{s}: unreadable npz ({exc})"
        if "joint_xyz_world" not in z or "valid_mask" not in z:
            return False, f"{s}: missing contract arrays"
        j = z["joint_xyz_world"]
        m = z["valid_mask"]
        if j.ndim != 3 or len(m) != len(j):
            return False, f"{s}: shape {j.shape} mask {len(m)}"
        src = str(z["source_native_output"]) if "source_native_output" in z else ""
        if f"imagepressure2smpl_v4_{fold}" not in src:
            return False, f"{s}: checkpoint provenance {src[:80]}"
    return True, ""


def copy_variant(variant: str, kind: str) -> None:
    for fold in FOLDS:
        src = RESULTS / "predictions" / f"carpet_v4_anysolebbox_{fold}" / "eval_motion"
        dst = out_dir(variant, fold)
        dst.mkdir(parents=True, exist_ok=True)
        want = set(fold_test_sessions(fold, kind))
        for npz in sorted(src.glob("*.npz")):
            if npz.stem in want:
                target = dst / npz.name
                # Always overwrite: a previous base's (v3) clean predictions may
                # still sit here, and provenance verification would reject them.
                shutil.copy2(npz, target)
        print(f"[copy] {variant} {fold}: {len(list(dst.glob('*.npz')))}")


def filtered_split_csv(fold: str, kind: str) -> str:
    """Fold split without S12102 (the E5 feature trees never include it)."""
    split = (f"splits_sceneocc_{fold}" if kind == "scene" else FOLD_SPLITS[fold])
    src = ROOT / "AnysoleWorkspace/protocol/splits/default" / f"{split}.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig", newline="")))
    if not any((r.get("test") or "").strip() == "S12102" for r in rows):
        return f"protocol://splits/default/{split}.csv"
    out_dir = LOGROOT / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}_woS12102.csv"
    fieldnames = rows[0].keys()
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if (row.get("test") or "").strip() == "S12102":
                row = dict(row)
                row["test"] = ""
            writer.writerow(row)
    return str(out)


def make_cmd(variant: str, tree: str, kind: str, fold: str) -> list[str]:
    split_csv = filtered_split_csv(fold, kind)
    ckpt = CKPT_DIR / f"imagepressure2smpl_v4_{fold}" / "init" / "5e-05" / CKPT[fold]
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    return [PY, "-m", "app.test_frappe",
            "task=imagepressure2smpl_carpet_v4_anysolebbox_cam3",
            f"task.split_csv={split_csv}",
            f"task.seq_root=model-input://MotionPRO/{tree}/cam3",
            f"task.name=imagepressure2smpl_v4_{fold}",
            f"task.checkpoint_path={ckpt}",
            f"task.unified_prediction_dir=results://baselines/MotionPRO/predictions/{variant}/{fold}/eval_motion"]


def free_mem() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True)
    free = {}
    for line in out.splitlines():
        i, m = [x.strip() for x in line.split(",")[:2]]
        free[int(i)] = int(m)
    return free


def main() -> int:
    jobs = []
    terminal: list[str] = []
    for variant, tree, kind in VARIANTS:
        if tree is None:
            copy_variant(variant, kind)
            for fold in FOLDS:
                ok, reason = verify(variant, fold)
                if not ok:
                    print(f"[copy-verify FAIL] {variant} {fold}: {reason}", flush=True)
                    terminal.append(f"{variant}__{fold}")
                else:
                    print(f"[copy-verify ok] {variant} {fold}", flush=True)
            continue
        for fold in FOLDS:
            jobs.append((variant, tree, kind, fold))
    pending = jobs[::-1]
    running: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    while pending or running:
        for name in list(running):
            rec = running[name]
            code = rec["proc"].poll()
            if code is None:
                continue
            del running[name]
            variant, fold = name.split("__", 1)
            ok, reason = verify(variant, fold)
            if code == 0 and ok:
                print(f"[ok] {name}", flush=True)
                continue
            attempts[name] = attempts.get(name, 0) + 1
            print(f"[fail] {name} rc={code} verify={reason} attempts={attempts[name]}",
                  flush=True)
            if attempts[name] < MAX_RUNS:
                for v, t, k, f in jobs:
                    if f"{v}__{f}" == name:
                        pending.append((v, t, k, f))
                        break
            else:
                terminal.append(name)
        free = free_mem()
        busy = {rec["gpu"] for rec in running.values()}
        choices = [g for g in GPUS if g not in busy and free.get(g, 0) >= 6000]
        for gpu in choices:
            if len(running) >= 4 or not pending:
                break
            variant, tree, kind, fold = pending.pop()
            name = f"{variant}__{fold}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            out_dir(variant, fold).mkdir(parents=True, exist_ok=True)
            log = LOGROOT / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(f"\n=== {variant} {fold} gpu={gpu} ===\n")
            handle.flush()
            proc = subprocess.Popen(make_cmd(variant, tree, kind, fold),
                                    cwd=str(ROOT / "Baselines/MotionPRO"),
                                    env=env, stdout=handle,
                                    stderr=subprocess.STDOUT)
            running[name] = {"proc": proc, "gpu": gpu}
            print(f"[start] {name} gpu={gpu}", flush=True)
        time.sleep(10)
    if terminal:
        print(f"TERMINAL FAILURES: {terminal}")
        return 1
    print("inference done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
