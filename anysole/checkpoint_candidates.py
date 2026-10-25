"""Plan (or explicitly execute) native validation of a new candidate archive.

This does not start training or choose a final checkpoint. Existing output
directories are refused. By default, it only writes an evaluation plan.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from anysole.checkpoint_review import sha256
from anysole.utils.checkpoint_policy import atomic_json, FORMAL_MODES, ROLE_MODES


def prepare(archive_path, config, out, device="cuda:0", include_legacy=True):
    archive_path, config, out = Path(archive_path).resolve(), Path(config).resolve(), Path(out).resolve()
    manifest = json.loads(archive_path.read_text())
    role = manifest["role"]
    if role not in ROLE_MODES or manifest.get("version") not in (1, 2):
        raise ValueError("Unsupported checkpoint pool manifest")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation directory")
    if not config.is_file():
        raise ValueError("Missing evaluation config")
    archive_root = archive_path.parent
    files = [(ident, (archive_root / row["file"]).resolve(), row["epoch"])
             for ident, row in manifest["candidates"].items()]
    if any(archive_root not in p.parents for _, p, _ in files):
        raise ValueError("Archive weights must stay inside their archive")
    if include_legacy:
        ckpts = archive_root.parent
        files += [(p.stem, p, None) for p in [ckpts / "ckpt_best.pt", ckpts / "ckpt_last.pt"] if p.is_file()]
        files += [(p.stem, p, None) for p in sorted((ckpts / "survey").glob("survey_ep*.pt"))]
    modes = [FORMAL_MODES[m] for m in ROLE_MODES[role]]
    jobs = []
    hashes = {}
    for ident, checkpoint, epoch in files:
        digest = sha256(checkpoint)
        if digest in hashes:
            hashes[digest]["aliases"].append(str(checkpoint))
            continue
        folder = out / "results" / ident
        command = [sys.executable, "-m", "anysole.eval", "--config", str(config), "--ckpt", str(checkpoint),
                   "--split", "val", "--config-id", ",".join(modes), "--device", device,
                   "--no-robustness", "--no-write-motion", "--dump-detail",
                   "--protocol-out", str(folder / "val.json"), "--metrics-out", str(folder / "val_brief.json")]
        job = {"id": ident, "role": role, "epoch": epoch, "checkpoint": str(checkpoint),
               "checkpoint_sha256": digest, "aliases": [str(checkpoint)], "modes": modes,
               "output_dir": str(folder), "command": command}
        hashes[digest] = job
        jobs.append(job)
    if not jobs:
        raise ValueError("Archive contains no available checkpoints")
    plan = {"version": 1, "split": "val", "status": "planned", "test_used": False,
            "archive_manifest": str(archive_path), "archive_sha256": sha256(archive_path),
            "config": str(config), "config_sha256": sha256(config), "jobs": jobs,
            "deduplication": "identical file bytes only; optimizer/container differences may retain equivalent model weights"}
    atomic_json(out / "evaluation_plan.json", plan)
    return plan


def execute(plan, out):
    # Use the saved explicit val split, not eval.py's default (test).
    from anysole.train import load_config
    from anysole.data.dataset import load_split_ids
    config = load_config(Path(plan["config"]))
    ids = {split: set(load_split_ids(Path(config["split_csv"]), split)) for split in ("train", "val", "test")}
    if not ids["val"] or any(ids[a] & ids[b] for a, b in (("train", "val"), ("val", "test"), ("train", "test"))):
        raise ValueError("Full validation review requires disjoint train/val/test session lists")
    cwd = Path(__file__).resolve().parents[1]
    for job in plan["jobs"]:
        if sha256(plan["config"]) != plan["config_sha256"] or sha256(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("Config/checkpoint changed since planning")
        folder = Path(job["output_dir"])
        folder.mkdir(parents=True)
        with (folder / "eval.log").open("w") as stream:
            subprocess.run(job["command"], cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT, check=True)
        if sha256(job["checkpoint"]) != job["checkpoint_sha256"] or sha256(plan["config"]) != plan["config_sha256"]:
            raise ValueError("Config/checkpoint changed during evaluation")
        receipt = {"status": "complete", "split": "val", "checkpoint": job["checkpoint"],
                   "checkpoint_sha256": job["checkpoint_sha256"], "test_evaluated": False,
                   "config_sha256": plan["config_sha256"], "split_csv_sha256": sha256(config["split_csv"]),
                   "listed_sessions": sorted(ids["val"]), "modes": job["modes"],
                   "protocol_source_sha256": sha256(cwd / "anysole/utils/eval_protocol.py")}
        atomic_json(folder / "receipt.json", receipt)
        job["status"] = "complete"
        atomic_json(Path(out) / "evaluation_plan.json", plan)
    plan["status"] = "complete"
    atomic_json(Path(out) / "evaluation_plan.json", plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="checkpoints/multi_metric/manifest.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run validation after writing the plan; default is plan only")
    args = parser.parse_args(argv)
    plan = prepare(args.archive, args.config, args.out, args.device, not args.no_legacy)
    if args.execute:
        execute(plan, args.out)
    print("%s: %d validation jobs; no final checkpoint selected." % (plan["status"], len(plan["jobs"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
