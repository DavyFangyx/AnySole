"""GVHMR checkpoint inference -> historical export -> public scoring.

Variants are isolated so old results remain untouched:

* ``baseline``    original pretrained checkpoint, original inference/export
* ``ft_<fold>``   one fine-tuned checkpoint per fold, same original flow

Only the checkpoint changes between the two rows.  Camera intrinsics, per-date
translation and FK conventions stay on the original path.

Run in ``touch_gait`` from the repo root:

    python -m GVHMR.finetune.run_protocol --mode zero-shot \
        --gpus 2,3 --phase all

    python -m GVHMR.finetune.run_protocol --mode finetuned \
        --gpus 2,3 --phase all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import common

ROOT = common.ROOT
WORKSPACE = common.WORKSPACE
TOUCH_GAIT = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
EXPORT_PY = WORKSPACE / "tool/export_baseline_motion.py"
SPLITS = WORKSPACE / "protocol/splits/crossfit_woS13"
UNIT_MANIFEST = (
    WORKSPACE
    / "work/visual_baselines_local/reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl"
)
PROTOCOL_ROOT = WORKSPACE / "work/gvhmr_protocol"
PREDICTION_TEMPLATE = "results/baselines/GVHMR/predictions/{variant}"
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
ALLOWED_GPUS = (1, 2, 3, 4)
_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with _LOG_LOCK:
        print(f"[{stamp}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_sessions(fold: str) -> list[str]:
    path = SPLITS / f"{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing derived fold split {path}")
    values = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get(fold) or "").strip()
            if value and value not in values:
                values.append(value)
    return values


def _load_tools():
    import importlib.util

    generate_path = WORKSPACE / "tool/adapters/visual_baselines/generate.py"
    spec = importlib.util.spec_from_file_location("vb_generate_protocol", generate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    crossfit_path = WORKSPACE / "tool/adapters/visual_baselines/run_crossfit.py"
    spec = importlib.util.spec_from_file_location("vb_crossfit_protocol", crossfit_path)
    crossfit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(crossfit)
    return module, crossfit


def _gpu_pool(gpus: str):
    values = [value.strip() for value in gpus.split(",") if value.strip()]
    bad = [value for value in values if not value.isdigit() or int(value) not in ALLOWED_GPUS]
    if not values or bad:
        raise SystemExit(f"--gpus must be a non-empty subset of {ALLOWED_GPUS}, got {gpus!r}")

    class Pool:
        def __init__(self):
            self._free = list(values)
            self._condition = threading.Condition()

        def acquire(self) -> str:
            with self._condition:
                while not self._free:
                    self._condition.wait()
                return self._free.pop(0)

        def release(self, value: str) -> None:
            with self._condition:
                self._free.append(value)
                self._condition.notify()

    return Pool()


def _row_manifest() -> dict[str, dict]:
    rows = {}
    with (WORKSPACE / "protocol/manifests/session_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["session_id"])] = row
    return rows


def _native_root(variant: str) -> Path:
    return PROTOCOL_ROOT / "native" / variant


def _variant_name(mode: str, fold: str | None, condition_tag: str) -> str:
    suffix = f"_{condition_tag}" if condition_tag else ""
    if mode == "zero-shot":
        return f"baseline{suffix}"
    if fold is None:
        raise ValueError("fine-tuned variant needs a fold")
    return f"ft_{fold}{suffix}"


def _checkpoint_for_fold(fold: str, override: str | None) -> tuple[Path, str]:
    if override:
        path = Path(override).expanduser().resolve()
        return path, "override"
    run_root = common.DEFAULT_OUTPUT / "runs" / fold
    best_path = run_root / "best.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"missing fold best.json: {best_path}")
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    value = payload.get("best_model_path") or payload.get("last_model_path")
    if not value:
        raise ValueError(f"no checkpoint in {best_path}")
    return Path(value).expanduser().resolve(), str(best_path)


def _release_checkpoint() -> Path:
    from .. import config as _config

    return Path(_config.DEFAULT_CHECKPOINT).expanduser().resolve()


def _verify_result(
    result: Path,
    checkpoint: Path | None,
    expected_source: str = "estimate_K(diagonal FOV)",
    expected_K: np.ndarray | None = None,
    cam: int = 3,
) -> None:
    if not result.is_file():
        raise RuntimeError(f"native result missing: {result}")
    metadata = result.parent / "model_metadata.json"
    if not metadata.is_file():
        metadata = result.parent / "result.metadata.json"
    if not metadata.is_file():
        raise RuntimeError(
            f"worker metadata missing: {result.parent}/model_metadata.json or result.metadata.json"
        )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if expected_source and payload.get("intrinsics_source") != expected_source:
        raise RuntimeError(
            f"worker intrinsics source {payload.get('intrinsics_source')!r} != {expected_source!r}: {metadata}"
        )
    if checkpoint is not None:
        recorded = Path(str(payload.get("checkpoint_path", ""))).expanduser().resolve()
        if recorded != checkpoint.resolve():
            raise RuntimeError(f"checkpoint mismatch: worker={recorded} expected={checkpoint}")
        expected_sha = _sha256(checkpoint)
        if payload.get("checkpoint_sha256") != expected_sha:
            raise RuntimeError(
                f"checkpoint SHA mismatch: worker={payload.get('checkpoint_sha256')} expected={expected_sha}"
            )
    if expected_K is not None:
        with np.load(result, allow_pickle=False) as archive:
            K = np.asarray(archive["K_fullimg"], dtype=np.float64)
        if K.shape[1:] != (3, 3) or not np.isfinite(K).all():
            raise RuntimeError(f"result K invalid: {K.shape}")
        delta = np.abs(K - np.asarray(expected_K, dtype=np.float64)[None]).max()
        if delta > 1e-3:
            raise RuntimeError(f"result K does not match calibration cam{cam} K: max delta {delta}")


def _generate_one(gen, crossfit, mode, variant, checkpoint, row, unit, cam, condition,
                  pool, force, retries=2):
    date, subject = crossfit.session_parts(row)
    width, height = [int(value) for value in common.calibration(date)["image_size_px"]]
    focal = (width * width + height * height) ** 0.5
    expected_K = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    work_root = _native_root(variant)
    unit_root = work_root / "reference_results/3_Result/GVHMR_baseline" / (
        f"{date}_m1_{row['session_id']}_c{cam}"
    )
    result = unit_root / "result.npz"
    if result.is_file() and not force:
        try:
            _verify_result(result, checkpoint, expected_K=expected_K, cam=cam)
            return row["session_id"], "native-exists"
        except Exception as exc:
            return row["session_id"], f"native-invalid: {exc}"
    gpu = pool.acquire()
    try:
        for attempt in range(1, retries + 1):
            try:
                gen.generate_gvhmr(
                    row,
                    cam,
                    unit,
                    work_root,
                    int(gpu),
                    condition if condition != "none" else "none",
                    checkpoint=checkpoint,
                )
                _verify_result(result, checkpoint, expected_K=expected_K, cam=cam)
                return row["session_id"], "ok" if attempt == 1 else f"ok-retry{attempt - 1}"
            except Exception as exc:  # noqa: BLE001 — per-session resilience
                log(f"generate FAIL {variant} {row['session_id']} attempt {attempt}/{retries}: {exc}")
                if attempt == retries:
                    return row["session_id"], f"FAILED: {type(exc).__name__}: {exc}"
                time.sleep(5)
        return row["session_id"], "FAILED"
    finally:
        pool.release(gpu)


def _export_one(mode, variant, session, row, unit, cam, condition, force, retries=2):
    output = ROOT / PREDICTION_TEMPLATE.format(variant=variant) / "eval_motion" / f"{session}.npz"
    parts = str(unit["unit_id"]).split("/")
    date = parts[0]
    native = (
        _native_root(variant) / "reference_results/3_Result/GVHMR_baseline"
        / f"{date}_m1_{session}_c{cam}" / "result.npz"
    )
    if not native.is_file():
        return session, f"FAILED: native result missing: {native}"
    calib_sha = common.sha256_file(common.PROTOCOL / "calibration" / f"{date}.json")
    result_sha = _sha256(native)

    def provenance_ok() -> bool:
        if not output.is_file() or not native.is_file():
            return False
        try:
            with np.load(output, allow_pickle=True) as archive:
                return (
                    str(archive["source_native_output"]) == str(native)
                    and str(archive["package_result_sha256"]) == result_sha
                    and str(archive["calibration_sha256"]) == calib_sha
                )
        except Exception:
            return False

    if provenance_ok() and not force:
        return session, "pred-exists"
    command = [
        TOUCH_GAIT, str(EXPORT_PY), "--model", "gvhmr", "--session", session,
        "--package-root", str(_native_root(variant)), "--variant", variant,
        "--unit-manifest", str(UNIT_MANIFEST),
    ]
    if force or output.is_file():
        command.append("--force")
    for attempt in range(1, retries + 1):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode == 0 and provenance_ok():
            return session, "ok" if attempt == 1 else f"ok-retry{attempt - 1}"
        log(f"export FAIL {variant} {session}: {(completed.stderr or completed.stdout)[-600:]}")
        time.sleep(3)
    return session, "FAILED"


def _evaluable_sessions(folds: list[str]) -> list[str]:
    result = set()
    for fold in folds:
        test = set(_fold_sessions(fold))
        with (SPLITS / f"{fold}_manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                session = str(row.get("session_id"))
                value = row.get("valid_frame_indices", [])
                if isinstance(value, str):
                    value = json.loads(value or "[]")
                if session in test and value:
                    result.add(session)
    return sorted(result)


def _fold_expected(fold: str, sessions: list[str], variant: str) -> dict[str, int]:
    from Baselines.utils.gt_loading import valid_mask

    fold_ids = set(_fold_sessions(fold))
    requested = set(sessions) & fold_ids
    expected = {}
    with (SPLITS / f"{fold}_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            session = str(row.get("session_id"))
            if session not in requested:
                continue
            value = row.get("valid_frame_indices", [])
            if isinstance(value, str):
                value = json.loads(value or "[]")
            if value:
                original = ROOT / "results/baselines/GVHMR/predictions/local_gen/eval_motion" / f"{session}.npz"
                prediction = ROOT / PREDICTION_TEMPLATE.format(variant=variant) / "eval_motion" / f"{session}.npz"
                with np.load(original, allow_pickle=False) as archive:
                    original_mask = np.asarray(archive["valid_mask"], dtype=bool)
                with np.load(prediction, allow_pickle=False) as archive:
                    prediction_mask = np.asarray(archive["valid_mask"], dtype=bool)
                frames = int(float(row["n_frames"]))
                if original_mask.shape != (frames,) or prediction_mask.shape != (frames,):
                    raise RuntimeError(f"{session}: prediction mask length differs from the session grid")
                if not np.array_equal(prediction_mask, original_mask):
                    raise RuntimeError(f"{session}: prediction validity differs from the historical baseline")
                expected[session] = int((valid_mask(row, frames) & original_mask).sum())
    return expected


def _evaluate_fold(fold: str, variant: str, sessions: list[str], force: bool) -> tuple[str, bool, str]:
    manifest = SPLITS / f"{fold}_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing fold manifest: {manifest}")
    command = [
        TOUCH_GAIT, "-m", "Baselines.utils.evaluate",
        "--model", "gvhmr", "--split", fold,
        "--split-csv", str(SPLITS / f"{fold}.csv"),
        "--manifest", str(manifest),
        "--prediction-root", PREDICTION_TEMPLATE.format(variant=variant) + "/eval_motion",
        "--result-tag", variant,
        "--sessions", ",".join(sessions),
    ]
    if force:
        command.append("--force")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    metrics = ROOT / "results/baselines/GVHMR/metrics" / f"{fold}_{variant}_comparison.json"
    if completed.returncode != 0 or not metrics.is_file():
        return fold, False, f"FAILED: {(completed.stderr or completed.stdout)[-800:]}"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    expected = _fold_expected(fold, sessions, variant)
    if not expected:
        return fold, False, "FAILED: no evaluable sessions in fold manifest"
    if int(payload.get("n_sessions") or 0) != len(expected):
        return fold, False, f"FAILED: n_sessions={payload.get('n_sessions')} expected={len(expected)}"
    by_id = {str(row.get("session_id")): row for row in payload.get("sessions", [])}
    for session, frames in expected.items():
        row = by_id.get(session)
        if row is None or row.get("status") != "ok":
            return fold, False, f"FAILED: session {session} missing/not ok"
        if int(row.get("n_valid_frames") or 0) != int(frames):
            return fold, False, (
                f"FAILED: {session} n_valid_frames={row.get('n_valid_frames')} expected={frames}"
            )
    return fold, True, f"ok n_sessions={len(expected)}"


def _run_variant(mode, variant, sessions, folds, gen, crossfit, rows, units, cam, condition,
                 pool, workers, phase, force, checkpoint_override=None):
    summary = {"variant": variant, "sessions": {}, "folds": {}, "ok": True}
    missing = [session for session in sessions if session not in rows]
    if missing:
        summary["ok"] = False
        summary["failed_stage"] = "sessions"
        summary["missing_sessions"] = missing
        common.json_dump(PROTOCOL_ROOT / f"summary_{variant}.json", summary)
        log(f"{variant}: missing sessions {missing}")
        return summary
    if phase in ("generate", "all"):
        tasks = []
        for session in sessions:
            row = rows[session]
            date, _subject = crossfit.session_parts(row)
            uid = f"{date}/method_1_offset/{session}/camera_{cam}"
            if uid not in units:
                raise KeyError(f"unit not in manifest: {uid}")
            tasks.append((row, units[uid]))
        if mode == "zero-shot":
            checkpoint = _release_checkpoint()
        else:
            checkpoint, _ = _checkpoint_for_fold(folds[0], checkpoint_override)
        log(f"{variant}: generate {len(tasks)} sessions")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_one, gen, crossfit, mode, variant, checkpoint, row, unit,
                    cam, condition, pool, force,
                )
                for row, unit in tasks
            ]
            for future in as_completed(futures):
                session, status = future.result()
                summary["sessions"][session] = status
                log(f"{variant}: generate {session} -> {status}")
        bad = [s for s, status in summary["sessions"].items() if status.startswith("FAILED") or status.startswith("native-invalid")]
        if bad:
            summary["ok"] = False
            summary["failed_stage"] = "generate"
            common.json_dump(PROTOCOL_ROOT / f"summary_{variant}.json", summary)
            log(f"{variant}: generate failures {bad}; stopping before export/eval")
            return summary
    if phase in ("export", "all"):
        log(f"{variant}: export {len(sessions)} sessions")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            by_id = {row["session_id"]: row for row in rows.values()}
            futures = [
                executor.submit(
                    _export_one, mode, variant, session, by_id[session],
                    units[f"{crossfit.session_parts(by_id[session])[0]}/method_1_offset/{session}/camera_{cam}"],
                    cam, condition, force,
                )
                for session in sessions
            ]
            for future in as_completed(futures):
                session, status = future.result()
                summary["sessions"][session] = status
                log(f"{variant}: export {session} -> {status}")
        bad = [s for s, status in summary["sessions"].items() if status.startswith("FAILED")]
        if bad:
            summary["ok"] = False
            summary["failed_stage"] = "export"
            common.json_dump(PROTOCOL_ROOT / f"summary_{variant}.json", summary)
            log(f"{variant}: export failures {bad}; stopping before eval")
            return summary
    if phase in ("eval", "all"):
        for fold in folds:
            fold_status, fold_ok, detail = _evaluate_fold(fold, variant, sessions, force)
            summary["folds"][fold] = detail
            summary["ok"] = summary["ok"] and fold_ok
            log(f"{variant}: eval {fold} -> {detail}")
            if not fold_ok:
                summary["failed_stage"] = "eval"
                break
    common.json_dump(PROTOCOL_ROOT / f"summary_{variant}.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("zero-shot", "finetuned", "both"), default="zero-shot")
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--condition", default="none",
                        help="clean/none or an E5 degradation tag (passed to generate_gvhmr)")
    parser.add_argument("--sessions", default="", help="comma-separated subset (smoke)")
    parser.add_argument("--folds", default="", help="comma-separated folds (default: all)")
    parser.add_argument("--tag", default="", help="variant suffix for smoke/partial runs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--phase", choices=("generate", "export", "eval", "all"), default="all")
    parser.add_argument("--checkpoint", default="",
                        help="fine-tuned checkpoint override (smoke/debug)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    pool = _gpu_pool(args.gpus)
    gen, crossfit = _load_tools()
    kind, level = gen.parse_degrade(args.condition)
    condition_spec = args.condition if kind != "none" else "none"
    condition_tag = "" if kind == "none" else gen.degrade_tag(kind, level)
    rows = _row_manifest()
    units = crossfit.unit_manifest()
    crossfit.write_fold_manifests()

    modes = ("zero-shot", "finetuned") if args.mode == "both" else (args.mode,)
    selected_folds = [fold.strip() for fold in args.folds.split(",") if fold.strip()]
    bad_folds = [fold for fold in selected_folds if fold not in FOLDS]
    if bad_folds:
        raise SystemExit(f"unknown folds {bad_folds}; expected {FOLDS}")
    if not selected_folds:
        selected_folds = list(FOLDS)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    results = {}
    for mode in modes:
        if mode == "zero-shot":
            if args.sessions:
                sessions = [value.strip() for value in args.sessions.split(",") if value.strip()]
            else:
                sessions = _evaluable_sessions(selected_folds)
            if args.limit:
                sessions = sessions[: args.limit]
            variant = _variant_name(mode, None, condition_tag) + tag_suffix
            results[variant] = _run_variant(
                mode, variant, sessions, selected_folds, gen, crossfit, rows, units,
                args.cam, condition_spec, pool, args.workers, args.phase, args.force,
                args.checkpoint or None,
            )
        else:
            for fold in selected_folds:
                sessions = _fold_sessions(fold)
                if args.sessions:
                    wanted = {value.strip() for value in args.sessions.split(",") if value.strip()}
                    sessions = [session for session in sessions if session in wanted]
                else:
                    evaluable = set(_evaluable_sessions([fold]))
                    sessions = [session for session in sessions if session in evaluable]
                if args.limit:
                    sessions = sessions[: args.limit]
                variant = _variant_name(mode, fold, condition_tag) + tag_suffix
                results[variant] = _run_variant(
                    mode, variant, sessions, [fold], gen, crossfit, rows, units,
                    args.cam, condition_spec, pool, args.workers, args.phase, args.force,
                    args.checkpoint or None,
                )
    common.json_dump(PROTOCOL_ROOT / "protocol_summary.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(bool(item.get("ok")) for item in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
