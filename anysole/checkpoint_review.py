"""Read-only ten-metric validation comparison; never promotes a final checkpoint.

Run: python -m anysole.checkpoint_review --manifest review.json --out new_dir
The manifest fixes candidates and reference weights. Native run_protocol outputs
and their session/window detail files are required. Test results are rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from anysole.utils.checkpoint_policy import METRICS, UNITS, ROLE_MODES, FORMAL_MODES
from anysole.utils.checkpoint_policy import atomic_json, finite, pareto_ids


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(path, sources):
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("split") != "val":
        raise ValueError("Only validation results may enter checkpoint review: %s" % path)
    if obj.get("robustness"):
        raise ValueError("Use separate, explicit condition panels for robustness results")
    required = ("modal", "contact_method", "protocol_seed", "tw", "sample_steps", "forward_axis")
    if any(key not in obj for key in required):
        raise ValueError("Incomplete native evaluation metadata: %s" % path)
    checkpoint = Path(obj["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise ValueError("Referenced checkpoint is unavailable: %s" % checkpoint)
    sources[str(path)] = sha256(path)
    # Cache checkpoint hashes: U has three input panels, but one weight file.
    if str(checkpoint) not in sources:
        sources[str(checkpoint)] = sha256(checkpoint)
    obj["checkpoint_sha256"] = sources[str(checkpoint)]
    obj["checkpoint"] = str(checkpoint)
    grids = {}
    for kind in ("session", "window"):
        detail_path = path.with_name(kind + "_detail_val.json")
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if (detail.get("split") != "val" or Path(detail["checkpoint"]).resolve() != checkpoint
                or detail.get("tw") != obj["tw"] or detail.get("contact_method") != obj["contact_method"]):
            raise ValueError("Detail metadata disagrees with result: %s" % detail_path)
        sources[str(detail_path)] = sha256(detail_path)
        for mode, values in detail["conditions"].items():
            grids.setdefault(mode, {})
            if kind == "session":
                rows = sorted((v["session_id"], int(v["n_frames"])) for v in values)
                if not rows or any(n <= 0 for _, n in rows) or len(set(x for x, _ in rows)) != len(rows):
                    raise ValueError("Empty or duplicated scored session set")
                grids[mode][kind] = rows
            else:
                ids, starts, counts = values["session_ids"], values["frame_starts"], values["n_frames"]
                if not ids or len(ids) != len(starts) or len(ids) != len(counts):
                    raise ValueError("Invalid validation window grid")
                rows = sorted(zip(ids, starts, counts))
                if len(set((s, i) for s, i, n in rows)) != len(rows) or any(n <= 0 for s, i, n in rows):
                    raise ValueError("Repeated or invalid validation windows")
                grids[mode][kind] = rows
    for mode, block in obj["metrics"].items():
        if mode not in ("V2M", "T2M", "VT2M"):
            continue
        if any(not finite(block.get(metric)) for metric in METRICS):
            raise ValueError("Missing/nonfinite formal metric in %s %s" % (path, mode))
        if set(grids.get(mode, {})) != {"session", "window"}:
            raise ValueError("Missing frame coverage details for %s %s" % (path, mode))
    receipt_path = path.with_name("receipt.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("split") != "val" or receipt.get("status") != "complete"
                or receipt.get("checkpoint_sha256") != obj["checkpoint_sha256"]):
            raise ValueError("Validation receipt failed checkpoint provenance check: %s" % receipt_path)
        sources[str(receipt_path)] = sha256(receipt_path)
    obj["_signature"] = {key: obj[key] for key in required}
    obj["_grids"] = grids
    return obj


def compare(candidate, mode, reference, reference_mode):
    if candidate["_signature"] != reference["_signature"]:
        raise ValueError("Evaluation protocols differ; do not mix model/metric protocols")
    if candidate["_grids"][mode] != reference["_grids"][reference_mode]:
        raise ValueError("Validation sessions/windows differ; compare identical frame grids")
    values, ref = candidate["metrics"][mode], reference["metrics"][reference_mode]
    return {metric: {"candidate": values[metric], "reference": ref[metric],
                     "delta": values[metric] - ref[metric], "unit": UNITS[metric]}
            for metric in METRICS}


def review(manifest_path, out_dir):
    manifest_path = Path(manifest_path).resolve()
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.get("split") != "val" or spec.get("experiment") not in ("E1", "E2", "E5"):
        raise ValueError("Declare validation split and one experiment E1/E2/E5 explicitly")
    if spec.get("rules") is not None:
        raise ValueError("Acceptance tolerances are not fixed. This version produces comparisons only.")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Write a new review directory; previous results are immutable")
    sources = {str(manifest_path): sha256(manifest_path)}
    cache = {}

    def get(value):
        p = Path(value)
        if not p.is_absolute():
            p = manifest_path.parent / p
        p = p.resolve()
        if p not in cache:
            cache[p] = load_result(p, sources)
        return cache[p]

    rows, states, ids, checkpoints = [], {}, set(), set()
    for item in spec["candidates"]:
        ident, role = item["id"], item["role"]
        if ident in ids or role not in ROLE_MODES:
            raise ValueError("Repeated candidate ID or unsupported role")
        ids.add(ident)
        candidate = get(item["result"])
        if candidate["checkpoint"] in checkpoints:
            raise ValueError("Deduplicate checkpoint aliases before comparison")
        checkpoints.add(candidate["checkpoint"])
        modes = [FORMAL_MODES[m] for m in ROLE_MODES[role]]
        if not set(modes) <= set(candidate["metrics"]):
            raise ValueError("A candidate must contain every required input panel from ONE checkpoint")
        panels = {}
        for mode in modes:
            rule = spec["references"][mode]
            if "same_checkpoint_mode" in rule:
                if spec["experiment"] != "E5":
                    raise ValueError("Within-checkpoint references are reserved for E5")
                reference, ref_mode = candidate, rule["same_checkpoint_mode"]
            else:
                reference, ref_mode = get(rule["result"]), rule.get("mode", mode)
                if spec["experiment"] == "E5":
                    raise ValueError("E5 must compare VT and V from the same checkpoint")
            differences = compare(candidate, mode, reference, ref_mode)
            panels[mode] = {"metrics": differences,
                            "reference_checkpoint": reference["checkpoint"], "reference_mode": ref_mode,
                            "numerically_lower_count": sum(x["delta"] < 0 for x in differences.values())}
            for metric, cell in differences.items():
                rows.append({"candidate_id": ident, "role": role, "mode": mode,
                             "checkpoint": candidate["checkpoint"],
                             "reference_checkpoint": reference["checkpoint"], "reference_mode": ref_mode,
                             "metric": metric, **cell})
        pa = {m: candidate["metrics"].get(m, {}).get("PA-MPJPE") for m in ("V2M", "T2M", "VT2M")}
        defense = spec.get("defense", {"dpa_max_mm": 0.5, "t2m_pa_max_mm": 90.0})
        if any(not finite(defense.get(k)) for k in ("dpa_max_mm", "t2m_pa_max_mm")):
            raise ValueError("Nonfinite defensive thresholds")
        defense_pass = (all(finite(x) for x in pa.values())
                        and pa["VT2M"] - pa["V2M"] <= defense["dpa_max_mm"]
                        and pa["T2M"] <= defense["t2m_pa_max_mm"]) if role == "U" else None
        scores = {mode + "/" + metric: candidate["metrics"][mode][metric]
                  for mode in modes for metric in METRICS}
        states[ident] = {"role": role, "epoch": item.get("epoch", 0),
                         "checkpoint": candidate["checkpoint"],
                         "checkpoint_sha256": candidate["checkpoint_sha256"],
                         "scores": scores, "panels": panels, "defensive_pass": defense_pass}
    if not states:
        raise ValueError("No candidates")
    groups = {}
    for role in ROLE_MODES:
        subset = {i: r for i, r in states.items() if r["role"] == role}
        if not subset:
            continue
        keys = tuple(next(iter(subset.values()))["scores"])
        # All candidates must be evaluated on the exact same validation grid,
        # even if E5 references each candidate against itself.
        anchor = get(next(x["result"] for x in spec["candidates"] if x["role"] == role))
        for item in spec["candidates"]:
            if item["role"] == role:
                current = get(item["result"])
                for mode in (FORMAL_MODES[m] for m in ROLE_MODES[role]):
                    compare(current, mode, anchor, mode)
        groups[role] = {"all_input_ten_metric_pareto": pareto_ids(subset, keys),
                        "by_input": {mode: pareto_ids(subset, [mode + "/" + m for m in METRICS])
                                     for mode in (FORMAL_MODES[m] for m in ROLE_MODES[role])}}
        if role == "U":
            guarded = {i: r for i, r in subset.items() if r["defensive_pass"]}
            groups[role]["defensive_pass_candidates"] = sorted(guarded)
            groups[role]["defensive_pass_pareto"] = pareto_ids(guarded, keys)
    payload = {"version": 1, "experiment": spec["experiment"], "split": "val",
               "status": "comparison_only_thresholds_not_defined", "selected_checkpoint": None,
               "test_used_for_selection": False, "metric_units": UNITS,
               "candidates": states, "candidate_lists": groups, "source_sha256": sources,
               "limitations": ["Numerically lower does not establish meaningful or significant improvement.",
                               "Pareto membership does not establish acceptable regression or final suitability.",
                               "Only evaluated retained weights are compared; deleted weights cannot be recovered."]}
    atomic_json(out_dir / "comparison.json", payload)
    with (out_dir / "ten_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 检查点十指标验证比较", "",
             "尚未确定核心指标及允许退化幅度，因此只列候选，不选最终检查点。负差值表示误差下降；下降项数不代表有意义或显著优势。", "",
             "| 候选 | 输入 | 数值下降项数 | 原双条件 |", "| --- | --- | ---: | --- |"]
    for ident, state in states.items():
        for mode, panel in state["panels"].items():
            lines.append("| %s | %s | %d/10 | %s |" % (
                ident, mode, panel["numerically_lower_count"],
                "通过" if state["defensive_pass"] else "未通过" if state["defensive_pass"] is False else "专家不适用"))
    lines += ["", "每个统一模型的所有输入共用一份权重。原双条件是防守检查，不代表十项指标都没有明显退化。候选列表保存于 comparison.json；完整数值及差值保存于 ten_metrics.csv。", "", "## 附录 完整十指标", ""]
    for ident, state in states.items():
        for mode, panel in state["panels"].items():
            lines += ["### %s %s" % (ident, mode), "", "| 指标 | 单位 | 候选 | 对照 | 差值 |",
                      "| --- | --- | ---: | ---: | ---: |"]
            for metric, cell in panel["metrics"].items():
                lines.append("| %s | %s | %.3f | %.3f | %+.3f |" % (
                    metric, cell["unit"], cell["candidate"], cell["reference"], cell["delta"]))
            lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = review(args.manifest, args.out)
    print("Compared %d checkpoints; final checkpoint intentionally unselected." % len(payload["candidates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
