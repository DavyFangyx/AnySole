"""Training-time candidate pool: every gate-passing epoch is retained.

Training dashboards do NOT contain the formal ten metrics. The pool keeps
every epoch whose three-condition val PA passes the gates, plus per-axis
PA/GlobalMPJPE champions as a bounded fallback for gate-failing epochs, so
the post-training formal validation review sees the complete selection
domain. Selection itself (gates + zero-tax band + best VT PA) happens in
checkpoint_review.py on formal ten-metric results.
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
# val PA-MPJPE gates per condition (2026-10-25 裁定): a healthy candidate
# must pass all of its role's conditions; U needs V <= 40, T <= 90, VT <= 40.
DEFAULT_GATES = {"V": 40.0, "T": 90.0, "VT": 40.0}


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


class CandidateArchive:
    """One writer per fresh training run; own files only, atomic manifest commit."""

    def __init__(self, root, role, v_pa_max=40.0, t_pa_max=90.0, vt_pa_max=40.0):
        if role not in ROLE_MODES:
            raise ValueError("Unknown checkpoint role: " + str(role))
        gates = {"V": v_pa_max, "T": t_pa_max, "VT": vt_pa_max}
        if any(not finite(x) or x <= 0 for x in gates.values()):
            raise ValueError("Checkpoint gates must be positive finite numbers")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Use a fresh run directory; candidate archive already exists: %s" % self.root)
        self.role, self.gates = role, gates
        self.keys = tuple(mode + "/" + metric for mode in ROLE_MODES[role]
                          for metric in ("PA-MPJPE", "GlobalMPJPE"))
        self.rows, self.champions, self.last_epoch = {}, {}, 0

    def _gate_pass(self, metrics):
        for mode in ROLE_MODES[self.role]:
            value = metrics.get("val/pa_mpjpe/" + mode)
            if not finite(value) or value > self.gates[mode]:
                return False
        return True

    def update(self, checkpoint, metrics, save=None):
        epoch = int(checkpoint["epoch"])
        if epoch <= self.last_epoch:
            raise ValueError("Checkpoint epochs must increase")
        scores = dashboard_scores(metrics, ROLE_MODES[self.role])
        gate_pass = self._gate_pass(metrics)
        ident = "epoch_%06d" % epoch
        row = {"epoch": epoch, "scores": scores, "gate_pass": gate_pass,
               "file": "weights/" + ident + ".pt"}
        rows = dict(self.rows, **{ident: row})
        champions = dict(self.champions)
        for key in self.keys:
            old = champions.get(key)
            if old is None or scores[key] < rows[old]["scores"][key]:
                champions[key] = ident
        # Gate passers are retained in full — the formal review needs the
        # complete selection domain. Gate failures survive only as champions.
        # No cap, no sampling, no filtering: eviction only touches epochs
        # that never passed the gates and lost their champion status.
        kept = {i for i, r in rows.items() if r["gate_pass"]} | set(champions.values())
        self.root.mkdir(parents=True, exist_ok=True)
        if ident in kept:
            if save is None:
                import torch
                save = torch.save
            destination = self.root / row["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp%d" % os.getpid())
            payload = {key: checkpoint[key] for key in ("model", "epoch", "config")}
            payload["checkpoint_selection"] = {"policy": "candidate_pool_v2", "role": self.role,
                                               "inference_only": True, "dashboard": row}
            save(payload, tmp)
            os.replace(str(tmp), str(destination))
        retained = {i: rows[i] for i in sorted(kept)}
        state = {"version": 2, "role": self.role, "last_epoch": epoch,
                 "dashboard_metrics": list(self.keys),
                 "gates": {"val_pa_max_mm": dict(self.gates)},
                 "champions": champions, "candidates": retained,
                 "formal_ten_metric_selection": "pending_validation_review"}
        atomic_json(self.root / "manifest.json", state)
        # The manifest always references committed files before old files go.
        for old in set(self.rows) - kept:
            (self.root / self.rows[old]["file"]).unlink()
        log = {"epoch": epoch, "dashboard": row, "gate_pass": gate_pass, "retained": sorted(kept)}
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        self.rows, self.champions, self.last_epoch = retained, champions, epoch
        return state
