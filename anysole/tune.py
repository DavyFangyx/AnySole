"""Optuna hyperparameter tuning for AnySole (command_manual.md · Optuna 调参).

Pattern adapted from SurvPGC's utils/optuna_utils.py: TPE/random sampler,
median/none pruner, sqlite study storage under ``results/AnySole/optuna/``,
per-trial SUBPROCESS training (``python -m anysole.train``) + val evaluation
(``python -m anysole.eval --no-write-motion --no-protocol``), objective =
weighted mean of the three configs' val MPJPE (minimize), plus an analysis
log and study artifacts (optuna_trials.csv / best_trial.txt).

Base model = **V3_4a** (9 部位 + f2): the simplest V3 structure, chain-best
V2M and T2M 121.2 — the window sweep is its main growth axis, and ``tw`` is
therefore a search dimension here.  Every trial warm-starts from the base
checkpoint (the tw=20 -> tw=N transfer is key-verified: only time_pe/query
reinitialize, 330/336 keys inherit), so trials start near-converged.

No pruning in v1: each trial reports its objective once at completion, so a
MedianPruner has no intermediate values to act on.  Budget control = short
``--epochs`` per trial.  The study is stored in sqlite, so an interrupted
run resumes via the same ``--study-name``.

Usage (touch_gait env):
  python -m anysole.tune --study-name v34a_tw_lr --n-trials 30 --epochs 150 \
      --device cuda:4 [--dry-run] [--sampler tpe] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from anysole.registry import infer_anysole_paths
from anysole.types import GAIT_ROOT, stacked_variant_name

# Fixed base flags: every trial is V3_4a + sampled knobs, warm-started from
# the base checkpoint (the tw-shaped tensors reinitialize per sampled tw).
BASE_TRAIN_FLAGS = (
    "--modal", "anysolev2",
    "--contact-method", "joint_and",
    "--t-encoder", "foot_conv",
    "--f2-repr",
    "--pose-parts", "9",
    "--lr-warmup-frac", "0.05",
    "--grad-clip", "5.0",
)
BASE_EVAL_FLAGS = (
    "--modal", "anysolev2",
    "--contact-method", "joint_and",
    "--split", "val",
    "--no-write-motion",
    "--no-protocol",
)
DEFAULT_BASE_MODEL = "V3_4a"  # registry identifier (infer_anysole_paths resolves the ckpt)
CONFIGS = ("VT2M", "V2M", "T2M")


def ensure_optuna_available():
    try:
        import optuna  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna is not installed in the current environment. "
            "Install it first, e.g. `python -m pip install optuna`."
        ) from exc


def build_optuna_components(args):
    import optuna

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        raise ValueError("Unsupported sampler %r; expected 'tpe' or 'random'" % args.sampler)
    pruner = optuna.pruners.NopPruner()  # v1: objectives report once at completion
    return sampler, pruner


def resolve_optuna_storage(study_name: str) -> str:
    storage_dir = GAIT_ROOT / "results" / "AnySole" / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str((storage_dir / ("%s.db" % study_name)).resolve())


def stacked_name(trial_number: int, params: dict) -> str:
    """t{NNNN} + stacked hyperparameter fields, built by the SAME rule train
    inference uses (types.stacked_variant_name) so a trial dir name always
    matches the out-dir the subprocess derives."""
    variant = stacked_variant_name(
        tw=params["tw"],
        stride=params["tw"] if params["stride"] == "tw" else params["tw"] // 2,
        lr=params["lr"],
        lambda_pose=params["lambda_pose"],
        lambda_traj=params["lambda_traj"],
        lambda_kp=params["lambda_kp"],
    )
    return "t%04d_%s" % (trial_number, variant)


def trial_dir(base_model: Path, trial_number: int, params: dict) -> Path:
    """Trial dir = the base model's dir + stacked trial name (model+contact+
    variant == the ckpt address, so eval/infer/probes resolve it unchanged)."""
    return base_model / stacked_name(trial_number, params)


def sample_params(trial) -> dict:
    """V3-4a search space (command_manual.md · Optuna 调参).  Structure flags
    (f2 / foot_conv / 9 parts) are fixed; only knobs with CLI coverage vary."""
    tw = trial.suggest_categorical("tw", [30, 40, 50, 80])
    return {
        "tw": tw,
        "lr": trial.suggest_categorical("lr", [3e-5, 1e-4, 3e-4, 1e-3]),
        "lambda_pose": trial.suggest_categorical("lambda_pose", [2.0, 3.0, 5.0]),
        "lambda_traj": trial.suggest_categorical("lambda_traj", [0.5, 1.0, 2.0]),
        "lambda_kp": trial.suggest_categorical("lambda_kp", [0.5, 1.0, 2.0]),
        "stride": trial.suggest_categorical("stride", ["tw", "half"]),
    }


def train_command(args, params: dict, out_dir: Path) -> list:
    stride = params["tw"] if params["stride"] == "tw" else max(1, params["tw"] // 2)
    return [
        sys.executable, "-m", "anysole.train",
        *BASE_TRAIN_FLAGS,
        "--tw", str(params["tw"]),
        "--stride", str(stride),
        "--lr", "%.6g" % params["lr"],
        "--lambda-pose", "%.6g" % params["lambda_pose"],
        "--lambda-traj", "%.6g" % params["lambda_traj"],
        "--lambda-kp", "%.6g" % params["lambda_kp"],
        "--epochs", str(args.epochs),
        "--init-from", str(args.base_ckpt),
        "--out-dir", str(out_dir / "checkpoints"),
        "--device", args.device,
    ]


def eval_command(args, out_dir: Path) -> list:
    return [
        sys.executable, "-m", "anysole.eval",
        "--ckpt", str(out_dir / "checkpoints" / "ckpt_last.pt"),
        *BASE_EVAL_FLAGS,
        "--device", args.device,
    ]


def parse_val_metrics(out_dir: Path) -> dict:
    """{'VT2M': mpjpe, ...} from the eval output metrics/<split>.json."""
    path = out_dir / "metrics" / "val.json"
    if not path.is_file():
        raise FileNotFoundError("trial eval wrote no metrics: %s" % path)
    metrics = json.loads(path.read_text())["metrics"]
    return {config: float(metrics[config]["MPJPE"]) for config in CONFIGS}


def run_trial(args, study_name: str, trial_number: int, params: dict) -> dict:
    out_dir = trial_dir(args.base_model_dir, trial_number, params)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "trial.log"
    with log.open("w") as fh:
        fh.write("params: %s\n" % json.dumps(params))
        fh.flush()
        print("[tune] trial %04d: %s" % (trial_number, json.dumps(params)))
        proc = subprocess.run(
            train_command(args, params, out_dir), cwd=str(GAIT_ROOT),
            stdout=fh, stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError("train failed (rc=%d) for trial %04d; log: %s" %
                               (proc.returncode, trial_number, log))
        proc = subprocess.run(
            eval_command(args, out_dir), cwd=str(GAIT_ROOT),
            stdout=fh, stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError("eval failed (rc=%d) for trial %04d; log: %s" %
                               (proc.returncode, trial_number, log))
    metrics = parse_val_metrics(out_dir)
    return metrics


def objective(args, study_name: str):
    def _objective(trial):
        params = sample_params(trial)
        if args.dry_run:
            print("[dry-run] trial %04d: %s" % (trial.number, json.dumps(params)))
            print("[dry-run]   train:", " ".join(train_command(args, params, trial_dir(args.base_model_dir, trial.number, params))))
            return 0.0
        metrics = run_trial(args, study_name, trial.number, params)
        weights = args.objective_weights
        value = sum(w * metrics[c] for w, c in zip(weights, CONFIGS)) / sum(weights)
        for config in CONFIGS:
            trial.set_user_attr(config, metrics[config])
        return value

    return _objective


def save_study_artifacts(study, study_dir: Path, study_name: str) -> None:
    import pandas as pd

    study_dir.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(study_dir / "optuna_trials.csv", index=False)
    best = study.best_trial
    with (study_dir / "best_trial.txt").open("w") as fh:
        fh.write("study: %s\n" % study_name)
        fh.write("best_value: %.4f\n" % study.best_value)
        fh.write("best_trial_number: %d\n" % best.number)
        fh.write("best_params:\n")
        for key, value in best.params.items():
            fh.write("  %s: %s\n" % (key, value))
        for config in CONFIGS:
            fh.write("%s_MPJPE: %.2f\n" % (config, best.user_attrs.get(config, float("nan"))))
    print("artifacts -> %s" % study_dir)


def main(argv=None) -> int:
    ensure_optuna_available()
    import optuna

    ap = argparse.ArgumentParser(description="AnySole Optuna tuning (base = V3_4a)")
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=150, help="Short per-trial budget (v1 has no pruning).")
    ap.add_argument("--model-name", default=DEFAULT_BASE_MODEL,
                    help="Registry identifier of the base model each trial warm-starts from "
                    "(e.g. V3_4a); its ckpt resolves via infer_anysole_paths.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-startup-trials", type=int, default=10, help="TPE random startup trials.")
    ap.add_argument("--objective-weights", default="1,1,1",
                    help="Comma VT2M,V2M,T2M weights (normalized); default mean of the three.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sample and print trials without training/eval.")
    args = ap.parse_args(argv)

    weights = [float(w) for w in args.objective_weights.split(",")]
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("--objective-weights must be three non-negative weights")
    args.objective_weights = weights
    # Base addresses resolve through the registry (no user-written paths):
    # every trial warm-starts from the base model's ckpt and lands under the
    # base model's dir with a stacked trial name.
    base_paths = infer_anysole_paths(args.model_name)
    args.base_ckpt = base_paths["ckpt"]
    args.base_model_dir = base_paths["model_dir"]

    storage = resolve_optuna_storage(args.study_name)
    sampler, pruner = build_optuna_components(args)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage,
        sampler=sampler, pruner=pruner, direction="minimize", load_if_exists=True,
    )
    study.optimize(objective(args, args.study_name), n_trials=args.n_trials)
    if not args.dry_run:
        study_dir = GAIT_ROOT / "results" / "AnySole" / "optuna" / args.study_name
        save_study_artifacts(study, study_dir, args.study_name)
    print("study %s: best %.4f @ trial %d" % (args.study_name, study.best_value, study.best_trial.number))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
