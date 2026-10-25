"""Additive validation checkpoint retention; legacy best/last/survey stay independent.

Training dashboards do NOT contain the formal ten metrics. This archive protects
PA/GlobalMPJPE champions and a bounded dashboard Pareto sample for later formal
validation. It makes no claim to find the ten-metric optimum over every epoch.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


METRICS = (
    "PA-MPJPE", "MPJPE", "GlobalMPJPE", "W-MPJPE100", "MPJRE",
    "traj_ATE", "RootOrientationDrift", "FootSliding", "GroundPen", "Jitter_pred",
)
UNITS = {key: ("deg" if key in ("MPJRE", "RootOrientationDrift") else
               "m/s^3" if key == "Jitter_pred" else "mm") for key in METRICS}
ROLE_MODES = {"U": ("V", "T", "VT"), "S_V": ("V",),
              "S_T": ("T",), "S_VT": ("VT",)}
FORMAL_MODES = {"V": "V2M", "T": "T2M", "VT": "VT2M"}


class DashboardUnavailable(ValueError):
    """An undefined/nonfinite dashboard must not corrupt retention or kill training."""


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(str(tmp), str(path))


def infer_role(config):
    probs = [float(x) for x in config["config_probs"]]
    if len(probs) != 3 or not all(math.isfinite(x) and x >= 0 for x in probs) or sum(probs) <= 0:
        raise ValueError("config_probs must contain three nonnegative finite probabilities [VT,V,T]")
    present = [i for i, x in enumerate(probs) if x > 0]
    inferred = {0: "S_VT", 1: "S_V", 2: "S_T"}[present[0]] if len(present) == 1 else "U"
    requested = config.get("checkpoint_role", "auto")
    if requested not in ("auto", inferred):
        raise ValueError("checkpoint_role %s conflicts with config_probs (%s)" % (requested, inferred))
    return inferred


def dashboard_scores(metrics, modes):
    scores = {}
    for mode in modes:
        pa = metrics.get("val/pa_mpjpe/" + mode)
        # Legacy dashboard mpjpe is a WORLD-coordinate error, not the formal
        # pelvis-aligned MPJPE. Do not silently label it as formal MPJPE.
        glob = metrics.get("val/global_mpjpe/" + mode, metrics.get("val/mpjpe/" + mode))
        if not finite(pa) or not finite(glob):
            raise DashboardUnavailable("Missing/nonfinite validation PA or GlobalMPJPE for " + mode)
        scores[mode + "/PA-MPJPE"] = float(pa)
        scores[mode + "/GlobalMPJPE"] = float(glob)
    return scores


def dominates(a, b):
    """All axes are errors: no worse everywhere and strictly better somewhere."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_ids(rows, keys):
    ids = sorted(rows, key=lambda x: (rows[x]["epoch"], x))
    vectors = {i: tuple(rows[i]["scores"][key] for key in keys) for i in ids}
    return [i for n, i in enumerate(ids)
            if not any(dominates(vectors[j], vectors[i]) or
                       (vectors[j] == vectors[i] and m < n)
                       for m, j in enumerate(ids) if j != i)]


def diverse_ids(rows, ids, keys, keep):
    """Deterministic farthest-point sample after per-axis range normalization.

    This controls storage ONLY; it is not a weighted score or a scientific
    winner selection. Objective champions are separately protected.
    """
    if len(ids) <= keep:
        return list(ids)
    if keep == 0:
        return []
    lo = {k: min(rows[i]["scores"][k] for i in ids) for k in keys}
    hi = {k: max(rows[i]["scores"][k] for i in ids) for k in keys}
    vec = {i: [(rows[i]["scores"][k] - lo[k]) / (hi[k] - lo[k] or 1.0)
               for k in keys] for i in ids}
    chosen = [min(ids, key=lambda i: (rows[i]["epoch"], i))]
    while len(chosen) < keep:
        remaining = [i for i in ids if i not in chosen]
        best = max(remaining, key=lambda i: (
            min(sum((a - b) ** 2 for a, b in zip(vec[i], vec[j])) for j in chosen),
            -rows[i]["epoch"], i))
        chosen.append(best)
    return chosen


class CandidateArchive:
    """One writer per fresh training run; own files only, atomic manifest commit."""

    def __init__(self, root, role, pareto_keep=8, dpa_max=0.5, t2m_pa_max=90.0):
        if role not in ROLE_MODES:
            raise ValueError("Unknown checkpoint role: " + str(role))
        if int(pareto_keep) != pareto_keep or pareto_keep < 0:
            raise ValueError("checkpoint_pareto_keep must be a nonnegative integer")
        if not finite(dpa_max) or not finite(t2m_pa_max):
            raise ValueError("Nonfinite defensive thresholds")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Use a fresh run directory; candidate archive already exists: %s" % self.root)
        self.role, self.keep = role, int(pareto_keep)
        self.dpa_max, self.t2m_pa_max = dpa_max, t2m_pa_max
        self.keys = tuple(mode + "/" + metric for mode in ROLE_MODES[role]
                          for metric in ("PA-MPJPE", "GlobalMPJPE"))
        self.rows, self.champions, self.pool = {}, {}, []
        self.last_epoch = 0

    def update(self, checkpoint, metrics, save=None):
        epoch = int(checkpoint["epoch"])
        if epoch <= self.last_epoch:
            raise ValueError("Checkpoint epochs must increase")
        scores = dashboard_scores(metrics, ROLE_MODES[self.role])
        required = [metrics.get("val/pa_mpjpe/" + x) for x in ("V", "T", "VT")]
        guarded = (all(finite(x) for x in required) and required[2] - required[0] <= self.dpa_max
                   and required[1] <= self.t2m_pa_max)
        ident = "epoch_%06d" % epoch
        row = {"epoch": epoch, "scores": scores, "defensive_pass": guarded,
               "file": "weights/" + ident + ".pt"}
        rows = dict(self.rows, **{ident: row})
        champions = dict(self.champions)
        for key in self.keys:
            for guarded_only in (False, True) if self.role == "U" else (False,):
                if guarded_only and not guarded:
                    continue
                label = ("guarded/" if guarded_only else "all/") + key
                old = champions.get(label)
                if old is None or scores[key] < rows[old]["scores"][key]:
                    champions[label] = ident
        # Previously discarded epochs are unavailable. This is a bounded
        # online Pareto sample over retained candidates + the current epoch.
        front = pareto_ids(rows, self.keys)
        pool = diverse_ids(rows, front, self.keys, self.keep)
        kept = set(champions.values()) | set(pool)
        self.root.mkdir(parents=True, exist_ok=True)
        if ident in kept:
            if save is None:
                import torch
                save = torch.save
            destination = self.root / row["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp%d" % os.getpid())
            payload = {key: checkpoint[key] for key in ("model", "epoch", "config")}
            payload["checkpoint_selection"] = {"policy": "multi_metric_v1", "role": self.role,
                                               "inference_only": True, "dashboard": row}
            save(payload, tmp)
            os.replace(str(tmp), str(destination))
        retained = {i: rows[i] for i in sorted(kept)}
        state = {"version": 1, "role": self.role, "last_epoch": epoch,
                 "dashboard_metrics": list(self.keys), "pareto_keep": self.keep,
                 "max_weight_files": len(self.keys) * (2 if self.role == "U" else 1) + self.keep,
                 "defense": {"dpa_max_mm": self.dpa_max, "t2m_pa_max_mm": self.t2m_pa_max},
                 "champions": champions, "pareto_sample": pool, "candidates": retained,
                 "formal_ten_metric_selection": "pending_validation_review"}
        atomic_json(self.root / "manifest.json", state)
        # The manifest always references committed files before old files go.
        for old in set(self.rows) - kept:
            (self.root / self.rows[old]["file"]).unlink()
        log = {"epoch": epoch, "dashboard": row, "retained": sorted(kept), "champions": champions}
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        self.rows, self.champions, self.pool, self.last_epoch = retained, champions, pool, epoch
        return state
