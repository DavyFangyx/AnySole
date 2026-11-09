#!/usr/bin/env python3
"""Four-fold LSO woS13 driver for the three pure-visual baselines.

Protocol (2026-11-08, user ruling): the paper's main protocol is the four-fold
leave-subject-out pool of ``results_display/Etest_crossfit`` with S13 removed
from the held-out test pool:

    fS5S6S11 (test S5/S6/S11)   fS5S11S12 (test S5/S11/S12)
    fS6S7    (test S6/S7)       fS5S6S8   (test S5/S6/S8)

The baselines are zero-shot (one fixed model per baseline), so the fold
difference is only the test session population; predictions are generated once
per session and evaluated four times.

Two stages, both resumable (skip existing artifacts):

    # native inference (model subprocesses), 1 job per GPU
    python AnysoleWorkspace/tool/adapters/visual_baselines/run_crossfit.py \
        --models sam3db --workers 2 --gpus 3,5

    # export to the unified eval_motion contract (CPU, applies per-date b if present)
    python .../run_crossfit.py --models wham,gvhmr,sam3db --phase export

Generation runs the same model drivers as ``generate.py`` (WHAM demo.py,
SAM3DB worker in vb_tactile, GVHMR worker in vb_gvhmr); the unit manifest is
parsed once per process instead of once per session.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT / "AnysoleWorkspace"
TOUCH_GAIT = "/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python"
GENERATE_PY = WORKSPACE / "tool/adapters/visual_baselines/generate.py"
EXPORT_PY = WORKSPACE / "tool/export_baseline_motion.py"
SPLITS = WORKSPACE / "protocol/splits/crossfit_woS13"
LOCAL_ROOT = WORKSPACE / "work/visual_baselines_local"
RESULTS = ROOT / "results/baselines"
MODEL_DIR = {"wham": "WHAM", "sam3db": "SAM3DB", "gvhmr": "GVHMR"}
NATIVE_SUFFIX = {"wham": "result.pkl", "sam3db": "result.pkl", "gvhmr": "result.npz"}
FOLDS = ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8")
# 主模型 cell（Etest_crossfit 口径）；fS6S7S13 折读 S13-out eval-only 格。
CELL_OF_FOLD = {
    "fS5S6S11": "V7_Cv5_fS5S6S11_s1",
    "fS5S11S12": "V7_Cv5_fS5S11S12_s1",
    "fS6S7": "V7_Cv5_fS6S7S13_s1_s13out",
    "fS5S6S8": "V7_Cv5_fS5S6S8_s1",
}

_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with _LOG_LOCK:
        print("[%s] %s" % (stamp, message), flush=True)


def fold_sessions() -> list[str]:
    sessions: list[str] = []
    for fold in FOLDS:
        path = SPLITS / ("%s.csv" % fold)
        if not path.is_file():
            raise FileNotFoundError(
                "missing derived fold split %s (see protocol/splits/crossfit_woS13)" % path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get(fold) or "").strip()
                if value and value not in sessions:
                    sessions.append(value)
    return sessions


def manifest_rows() -> dict[str, dict]:
    rows = {}
    path = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def unit_manifest() -> dict[str, dict]:
    path = LOCAL_ROOT / "reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl"
    units = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                unit = json.loads(line)
                units[unit["unit_id"]] = unit
    return units


def write_fold_manifests() -> dict[str, Path]:
    """Per-fold manifest overrides with the main-model cell frame set.

    The cell scores only complete 20-frame windows that are also inside the
    SMPL support (frames outside support are dropped at the *start*, and gaps
    are cut).  The generic manifest ``valid_frame_indices`` does not carry
    those exclusions, so using it would score the baselines on a different
    frame population than the main model (e.g. S11102: 300 vs 280).  We copy
    the manifest and replace ``valid_frame_indices`` with the cell's
    ``frame_ids`` for the fold's test sessions; evaluation then uses exactly
    the cell population.
    """
    cells = ROOT / "results/cells"
    base_rows = []
    with (WORKSPACE / "protocol/manifests/session_manifest.jsonl").open(encoding="utf-8") as handle:
        base_rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {row["session_id"]: row for row in base_rows}
    paths = {}
    for fold in FOLDS:
        cell = CELL_OF_FOLD[fold]
        metrics_path = cells / cell / "metrics/test.json"
        if not metrics_path.is_file():
            raise FileNotFoundError("missing main-model cell metrics: %s" % metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        frame_ids = {
            str(entry["session_id"]): [int(v) for v in entry["frame_ids"]]
            for entry in payload["evaluation_contract"]["sessions"]
        }
        with (SPLITS / ("%s.csv" % fold)).open(encoding="utf-8-sig", newline="") as handle:
            fold_sessions = [row[fold].strip() for row in csv.DictReader(handle)
                             if (row.get(fold) or "").strip()]
        missing = [sid for sid in fold_sessions if sid not in frame_ids]
        if missing:
            raise ValueError("cell %s lacks sessions: %s" % (cell, missing[:5]))
        output = SPLITS / ("%s_manifest.jsonl" % fold)
        with output.open("w", encoding="utf-8") as handle:
            for row in base_rows:
                copy = dict(row)
                if copy["session_id"] in frame_ids:
                    copy["valid_frame_indices"] = json.dumps(frame_ids[copy["session_id"]])
                    copy["frame_ids_pinned"] = "true"
                handle.write(json.dumps(copy, ensure_ascii=False) + "\n")
        paths[fold] = output
        log("manifest %s: %d sessions pinned to cell %s frame_ids"
            % (output.name, len(fold_sessions), cell))
    return paths


def load_generate():
    spec = importlib.util.spec_from_file_location("vb_generate", GENERATE_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def session_parts(row: dict) -> tuple[str, str]:
    from AnysoleWorkspace.tool.workspace import resolve_uri
    recording = resolve_uri(row["video_path"], must_exist=True)
    return recording.parts[-3], recording.parts[-2]


def native_result(model: str, date: str, session: str, cam: int) -> Path:
    return (LOCAL_ROOT / "reference_results/3_Result" / ("%s_baseline" % MODEL_DIR[model])
            / ("%s_m1_%s_c%d" % (date, session, cam)) / NATIVE_SUFFIX[model])


def prediction(model: str, session: str) -> Path:
    return RESULTS / MODEL_DIR[model] / "predictions/local_gen/eval_motion" / ("%s.npz" % session)


class GpuPool:
    """One concurrent job per GPU; each task leases a GPU for its duration."""

    def __init__(self, gpus: list[str]):
        self._free = list(gpus)
        self._cond = threading.Condition()

    def acquire(self) -> str:
        with self._cond:
            while not self._free:
                self._cond.wait()
            return self._free.pop(0)

    def release(self, gpu: str) -> None:
        with self._cond:
            self._free.append(gpu)
            self._cond.notify()


def generate_one(gen, model: str, session: str, row: dict, unit: dict,
                 cam: int, gpus: GpuPool, retries: int) -> tuple[str, str, str]:
    date, _subject = session_parts(row)
    result = native_result(model, date, session, cam)
    if result.is_file():
        return model, session, "native-exists"
    gpu = gpus.acquire()
    try:
        for attempt in range(1, retries + 1):
            work = LOCAL_ROOT
            try:
                if model == "wham":
                    gen.generate_wham(row, cam, unit, work, int(gpu))
                elif model == "sam3db":
                    gen.generate_sam3db(row, cam, unit, work, int(gpu))
                elif model == "gvhmr":
                    gen.generate_gvhmr(row, cam, unit, work, int(gpu))
                else:
                    raise ValueError(model)
                if result.is_file():
                    return model, session, "ok" if attempt == 1 else "ok-retry%d" % (attempt - 1)
                raise RuntimeError("model finished but %s missing" % result)
            except Exception as exc:  # noqa: BLE001 — per-session resilience
                log("FAIL %s %s attempt %d/%d gpu%s: %s: %s"
                    % (model, session, attempt, retries, gpu, type(exc).__name__, exc))
                if attempt == retries:
                    return model, session, "FAILED: %s: %s" % (type(exc).__name__, exc)
                time.sleep(10)
        return model, session, "FAILED"
    finally:
        gpus.release(gpu)


def export_one(model: str, session: str, retries: int, force: bool = False) -> tuple[str, str, str]:
    output = prediction(model, session)
    if output.is_file() and not force:
        return model, session, "pred-exists"
    command = [TOUCH_GAIT, str(EXPORT_PY), "--model", model, "--session", session,
               "--variant", "local_gen"]
    if force:
        command.append("--force")
    for attempt in range(1, retries + 1):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode == 0 and output.is_file():
            return model, session, "ok" if attempt == 1 else "ok-retry%d" % (attempt - 1)
        log("EXPORT FAIL %s %s attempt %d/%d: %s"
            % (model, session, attempt, retries,
               (completed.stderr or completed.stdout or "")[-800:]))
        time.sleep(5)
    return model, session, "FAILED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="wham,gvhmr,sam3db",
                        help="comma-separated: wham / gvhmr / sam3db")
    parser.add_argument("--sessions", default="",
                        help="comma-separated session IDs (default: union of the four fold test sets)")
    parser.add_argument("--phase", choices=("generate", "export", "eval", "both", "all"),
                        default="generate")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="5,3")
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force-export", action="store_true",
                        help="re-export sessions whose eval_motion npz already exists")
    args = parser.parse_args()

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = [m for m in models if m not in MODEL_DIR]
    if unknown:
        raise SystemExit("unknown models: %s" % unknown)
    sessions = ([s.strip() for s in args.sessions.split(",") if s.strip()]
                if args.sessions else fold_sessions())
    rows = manifest_rows()
    missing = [s for s in sessions if s not in rows]
    if missing:
        raise SystemExit("sessions absent from session manifest: %s" % missing)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log("models=%s sessions=%d gpus=%s workers=%d phase=%s"
        % (models, len(sessions), gpus, args.workers, args.phase))

    if args.phase in ("generate", "both", "all"):
        gen = load_generate()
        units = unit_manifest()
        tasks = []
        for model in models:
            for session in sessions:
                date, _subject = session_parts(rows[session])
                uid = "%s/method_1_offset/%s/camera_%d" % (date, session, args.cam)
                if uid not in units:
                    log("SKIP %s %s: unit not in manifest" % (model, session))
                    continue
                if native_result(model, date, session, args.cam).is_file():
                    continue
                tasks.append((model, session, rows[session], units[uid]))
        log("generate tasks=%d" % len(tasks))
        pool = GpuPool(gpus)
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_one, gen, model, session, row, unit,
                                       args.cam, pool, args.retries)
                       for model, session, row, unit in tasks]
            done = 0
            for future in as_completed(futures):
                model, session, status = future.result()
                done += 1
                counts[status] = counts.get(status, 0) + 1
                log("generate %d/%d %s %s -> %s" % (done, len(tasks), model, session, status))
        log("generate summary: %s" % counts)

    if args.phase in ("export", "both", "all"):
        tasks = []
        skipped = []
        for model in models:
            for session in sessions:
                output = prediction(model, session)
                if output.is_file() and not args.force_export:
                    skipped.append((model, session))
                else:
                    tasks.append((model, session))
        log("export tasks=%d (existing %d)" % (len(tasks), len(skipped)))
        counts = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(export_one, model, session, args.retries, args.force_export)
                       for model, session in tasks]
            done = 0
            for future in as_completed(futures):
                model, session, status = future.result()
                done += 1
                counts[status] = counts.get(status, 0) + 1
                log("export %d/%d %s %s -> %s" % (done, len(tasks), model, session, status))
        log("export summary: %s" % counts)

    if args.phase in ("eval", "all"):
        fold_manifests = write_fold_manifests()
        expected = {}
        for fold in FOLDS:
            with (SPLITS / ("%s.csv" % fold)).open(encoding="utf-8-sig", newline="") as handle:
                fold_ids = {row[fold].strip() for row in csv.DictReader(handle)
                            if (row.get(fold) or "").strip()}
            evaluable = set()
            with fold_manifests[fold].open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get("valid_frame_indices")
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = []
                    if value:
                        evaluable.add(row["session_id"])
            # S12102 是全 fake 会话（valid_frame_indices=[]）：主模型 cell 同样
            # 只评 57/58，read_manifest(evaluable_only) 会把它排除。
            expected[fold] = len(fold_ids & evaluable)
        failures = []
        for model in models:
            for fold in FOLDS:
                cmd = [TOUCH_GAIT, "-m", "Baselines.utils.evaluate", "--model", model,
                       "--split", fold, "--split-csv", str(SPLITS / ("%s.csv" % fold)),
                       "--manifest", str(fold_manifests[fold]), "--force"]
                completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                metrics_path = (RESULTS / MODEL_DIR[model] / "metrics"
                                / ("%s_comparison.json" % fold))
                if completed.returncode != 0 or not metrics_path.is_file():
                    failures.append((model, fold, (completed.stderr or "")[-400:]))
                    log("EVAL FAIL %s %s: %s" % (model, fold, failures[-1][2]))
                    continue
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                n_ok = int(payload.get("n_sessions") or 0)
                status = "ok" if n_ok == expected[fold] else "SHORT"
                if status != "ok":
                    failures.append((model, fold, "n_sessions=%d expected=%d"
                                     % (n_ok, expected[fold])))
                log("eval %s %s -> %s (n=%d/%d)"
                    % (model, fold, status, n_ok, expected[fold]))
        log("eval summary: %d failure(s)" % len(failures))
        if failures:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
