"""Test9.1: dissect the Test9 tau0-vs-DDIM gap on the SAME fixed batch.

Test9 (results_display/Test9_overfit) overfits 8 windows and ends with
tau0 ~ 58mm but DDIM ~ 220mm, and the DDIM number *worsens* during training
while tau0 improves.  If the model truly only output g(F) (ignoring x_tau),
the DDIM endpoint must equal the tau0 output: with x0_hat ≡ c the update
x_{t-1} = √ᾱ_{t-1}·c + √(1-ᾱ_{t-1})·(x_t - √ᾱ_t·c)/√(1-ᾱ_t) keeps
u_t = (x_t - √ᾱ_t·c)/√(1-ᾱ_t) invariant, so x_0 → c.  Hence 220mm is a
second, independent fault.  This script runs the five ordered checks:

  1. Fake oracle: run the production DDIM loop but replace the model output
     with GT x0 at every step.  The endpoint must be ≈0mm, and the residual
     noise norm u_t must stay constant across steps — otherwise the sampler
     itself is broken and the model is irrelevant.
  2. Same batch: verify the tau0 probe and the DDIM run consume the exact
     same tensors (pose_gt / V / T / config_id / session_id), and measure
     tau0 on a held-out batch to quantify how much "8 windows don't
     generalize" could explain (Test9's own CSV is same-batch; the train.py
     dashboard probe mixes scopes: tau0 on the first eval batch, DDIM on the
     whole eval split).
  3. Tau grid: print the actual sampling τ schedule, ᾱ at the first/last
     step, per-step ‖x_τ‖ of the DDIM trace vs the training-time q_sample
     envelope at the same τ.
  4. Last-step x0_hat vs tau0 output, element-wise: if the model really
     ignores x_tau these must be almost equal and the per-step trace must be
     constant; also measures the model's τ=0 sensitivity to the final-call
     input x_penultimate (the chain error would enter exactly here).
  5. 6D→SO(3): confirm both eval paths share fk_pose6d/rot6d_to_rotmat and
     the same trans anchoring, check orthonormality of GT/tau0/DDIM outputs,
     and torch-vs-numpy conversion consistency.

Checks 1-3 are model-free (--skip-train runs them instantly); checks 2b/4/5
need the overfit model (default: retrain one lr, 3000 steps, like Test9, or
--ckpt to load a saved overfit checkpoint).

Outputs under results_display/Test9_1_sampler/:
    train_log.csv       single-lr overfit log (same schema as Test9)
    check3_tau_grid.csv tau schedule + per-step norms vs q_sample envelope
    check4_trace.csv    per-step model trace (tau, |x_in|, |x0_hat-GT|, ...)
    test91_report.json  per-check numbers + verdicts
    overfit_ckpt_<lr>.pt  only with --save-ckpt (reusable via --ckpt)

Usage (run from the repository root):
    python results_display/script/r_test7_legacy_sampler_checks.py
    python results_display/script/r_test7_legacy_sampler_checks.py --skip-train
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import cli_common  # noqa: E402
from loguru import logger as log  # noqa: E402

from anysole.data.dataset import AnySoleDataset, collate_windows  # noqa: E402
from anysole.diffusion import GaussianDiffusion, _extract, _timestep_schedule  # noqa: E402
from anysole.geometry import fk_pose6d, rot6d_to_rotmat, rot6d_to_rotmat_np  # noqa: E402
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1  # noqa: E402
from anysole.train import condition_inputs, load_config, move_batch, resolve_device  # noqa: E402
from anysole.types import CONFIG_VT, POSE_DIM, assert_batch_shapes  # noqa: E402


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# --------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------

class FakeOracleModel:
    """Returns GT x0 at every step, recording the tau sequence and states."""

    def __init__(self, x0: torch.Tensor):
        self.x0 = x0
        self.taus: list = []
        self.x_inputs: list = []

    def __call__(self, V_feat, T_raw, T_phys, x_tau, tau, config_id, session_id=None):
        self.taus.append(int(tau[0].item()))
        self.x_inputs.append(x_tau.detach().clone())
        return {"x0_hat": self.x0}


class RecordingModel:
    """Wraps the trained model, recording per-step (tau, x_in, x0_hat)."""

    def __init__(self, model):
        self.model = model
        self.trace: list = []

    def __call__(self, V_feat, T_raw, T_phys, x_tau, tau, config_id, session_id=None):
        out = self.model(V_feat, T_raw, T_phys, x_tau, tau, config_id, session_id)
        self.trace.append(
            {
                "tau": int(tau[0].item()),
                "x_in": x_tau.detach().clone(),
                "x0_hat": out["x0_hat"].detach().clone(),
            }
        )
        return out


# --------------------------------------------------------------------------
# Probe helpers (mirror Test9's two probes exactly)
# --------------------------------------------------------------------------

def _fk_mpjpe_mm(pred_pose: torch.Tensor, batch: dict) -> float:
    """FK with GT trajectory + anchor — the exact Test9 metric for both probes."""
    anchor = batch["trans_anchor"][:, None, :]
    kp = fk_pose6d(pred_pose, batch["trans_gt"] + anchor, batch["offsets"], batch["parents"])
    return float(torch.linalg.vector_norm(kp - batch["kp_gt"], dim=-1).mean().item()) * 1000.0


def _tau0_probe(model, v_feat, t_raw, t_phys, batch, device):
    """Test9's tau0 probe: clean GT pose input, tau=0. Returns (mpjpe_mm, x0_hat)."""
    with torch.inference_mode():
        zero = torch.zeros(batch["pose_gt"].shape[0], device=device, dtype=torch.long)
        out0 = model(v_feat, t_raw, t_phys, batch["pose_gt"], zero, batch["config_id"], batch.get("session_id"))
        return _fk_mpjpe_mm(out0["x0_hat"], batch), out0["x0_hat"].detach().clone()


def _rms(x: torch.Tensor) -> float:
    return float(x.double().pow(2).mean().sqrt().item())


def _same_tensor(a, b) -> bool:
    return bool(
        a is b
        or (torch.is_tensor(a) and torch.is_tensor(b) and a.data_ptr() == b.data_ptr())
    )


def _q_sample_envelope(diffusion, x0: torch.Tensor, tau: int, draws: int = 16) -> tuple:
    """Mean/std of ‖q_sample(x0, tau)‖ over fresh noise draws (training distribution)."""
    t = torch.full((x0.shape[0],), tau, device=x0.device, dtype=torch.long)
    norms = [float(diffusion.q_sample(x0, t).norm().item()) for _ in range(draws)]
    return float(np.mean(norms)), float(np.std(norms))


def _residual_noise_rms(diffusion, x: torch.Tensor, tau: int, x0: torch.Tensor) -> float:
    """Per-element RMS of u_tau = (x - √ᾱ_τ·x0)/√(1-ᾱ_τ); invariant under DDIM when x0_hat is constant."""
    t = torch.full((x.shape[0],), tau, device=x.device, dtype=torch.long)
    sqrt_abar = _extract(diffusion.sqrt_alphas_cumprod, t, x.shape)
    sqrt_1m = _extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
    return _rms((x - sqrt_abar * x0) / sqrt_1m)


def _ortho_stats(rotmats: torch.Tensor) -> dict:
    """Orthonormality of a batch of 3x3 rotation matrices."""
    eye = torch.eye(3, device=rotmats.device, dtype=rotmats.dtype).expand_as(rotmats)
    err = torch.matmul(rotmats.transpose(-1, -2), rotmats) - eye
    det = torch.linalg.det(rotmats)
    return {
        "max_abs_RtR_minus_I": float(err.abs().max().item()),
        "mean_abs_RtR_minus_I": float(err.abs().mean().item()),
        "max_abs_det_minus_1": float((det - 1.0).abs().max().item()),
        "min_det": float(det.min().item()),
    }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check1_fake_oracle(diffusion, cond, batch, sample_steps, device) -> tuple:
    """Production DDIM loop with the model output replaced by GT x0 every step.

    Returns (result, oracle) — the oracle's recorded tau/states are reused by check 3.
    """
    pose_gt = batch["pose_gt"]
    bsz, tw = pose_gt.shape[0], pose_gt.shape[1]
    oracle = FakeOracleModel(pose_gt)
    with torch.inference_mode():
        pred = diffusion.ddim_sample_loop(
            oracle,
            tau_related_kwargs=cond,
            shape=(bsz, tw, POSE_DIM),
            steps=sample_steps,
            eta=0.0,
            device=device,
        )
    endpoint_mpjpe = _fk_mpjpe_mm(pred, batch)
    endpoint_pose_rms = _rms(pred - pose_gt)
    schedule = _timestep_schedule(diffusion.n_train_steps, int(sample_steps))
    taus = oracle.taus
    # Residual-noise norm per step: must be ~constant (the u invariant).
    u_rms = [
        _residual_noise_rms(diffusion, x_in, tau, pose_gt)
        for x_in, tau in zip(oracle.x_inputs, taus)
    ]
    u_ratio = (max(u_rms) / min(u_rms)) if min(u_rms) > 1e-12 else float("inf")
    # Endpoint must be ≈0mm.  The floor is ~3e-5 mm = the 6D↔FK representation
    # roundtrip (dataset kp_gt comes from the original motion, not from
    # FK(pose_gt)), so 1e-3 mm is "exactly zero" for any real fault.
    ok_endpoint = endpoint_mpjpe < 1e-3 and endpoint_pose_rms < 1e-6
    ok_schedule = (
        taus == schedule
        and all(0 <= t < diffusion.n_train_steps for t in taus)
        and all(a > b for a, b in zip(taus, taus[1:]))
        and taus[-1] == 0
    )
    # u stays within 2% of its first-step value (catches wrong dir_coeff / noise damping).
    ok_invariant = abs(u_rms[-1] - u_rms[0]) / max(u_rms[0], 1e-12) < 0.02
    result = {
        "endpoint_mpjpe_mm": endpoint_mpjpe,
        "endpoint_pose_rms": endpoint_pose_rms,
        "taus_seen": taus,
        "schedule_matches_and_valid": ok_schedule,
        "residual_noise_rms_first": u_rms[0],
        "residual_noise_rms_last": u_rms[-1],
        "residual_noise_rms_ratio": u_ratio,
        "noise_invariant_holds": ok_invariant,
        "verdict": "PASS" if (ok_endpoint and ok_schedule and ok_invariant) else "FAIL",
    }
    print("== Check 1: fake oracle (GT x0 at every step) ==")
    print("  taus seen: %s ... %s (%d steps)" % (taus[:5], taus[-3:], len(taus)))
    print("  endpoint pose RMS vs GT: %.3g   endpoint MPJPE: %.6f mm" % (endpoint_pose_rms, endpoint_mpjpe))
    print("  residual noise RMS  first=%.6f  last=%.6f  (invariant: %s)"
          % (u_rms[0], u_rms[-1], ok_invariant))
    print("  => %s" % result["verdict"])
    return result, oracle


def check2a_same_batch(batch, cond) -> dict:
    """The two Test9 probes receive the same batch dict; verify tensor identity here.

    ``condition_inputs`` returns a fresh tensor (torch.where copies even when
    nothing is dropped), so V/T compare by VALUE equality; config_id/session_id
    can compare by buffer identity.  Either way, a mismatch would mean the two
    probes were built from different windows.
    """
    def value_or_identity(a, b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return a == b
        return bool(torch.equal(a, b))

    checks = {
        "pose_gt_same_values": value_or_identity(batch["pose_gt"], batch["pose_gt"]),
        "cond_V_feat_same_values": value_or_identity(cond["V_feat"], batch["V_feat"]),
        "cond_T_raw_same_values": value_or_identity(cond["T_raw"], batch["T_raw"]),
        "cond_T_phys_same_values": value_or_identity(cond["T_phys"], batch["T_phys"]),
        "cond_config_id_same_buffer": _same_tensor(cond["config_id"], batch["config_id"]),
        "cond_session_id_same_buffer": _same_tensor(cond["session_id"], batch["session_id"]),
    }
    # Test9's code inspection: _tau0_mpjpe_mm and _ddim_mpjpe_mm both read from the
    # same `batch` dict (r_test7_legacy_overfit.py), config_id forced to CONFIG_VT for both.
    all_shared = all(checks.values())
    result = {
        "tensor_identity": checks,
        "test9_probes_share_batch": True,
        "note": "In r_test7_legacy_overfit.py both probes receive the same fixed `batch`; "
                "the 169->220 DDIM series and the tau0 series are same-batch numbers. "
                "Cross-batch tau0 (check 2b) quantifies what a batch mismatch would cost.",
        "verdict": "PASS" if all_shared else "FAIL",
    }
    print("== Check 2a: same batch for tau0 and DDIM ==")
    for key, ok in checks.items():
        print("  %-34s %s" % (key, "OK" if ok else "MISMATCH"))
    print("  => %s" % result["verdict"])
    return result


def check2b_heldout_tau0(model, batch, train_loader, eval_dataset, device) -> dict:
    """tau0 MPJPE on the fixed train batch vs held-out windows vs the eval split."""
    def tau0_on(b):
        cid = torch.full((b["pose_gt"].shape[0],), CONFIG_VT, device=device, dtype=torch.long)
        b["config_id"] = cid
        vf, tr, tp = condition_inputs(b, cid)
        with torch.inference_mode():
            zero = torch.zeros(b["pose_gt"].shape[0], device=device, dtype=torch.long)
            out0 = model(vf, tr, tp, b["pose_gt"], zero, cid, b.get("session_id"))
        return _fk_mpjpe_mm(out0["x0_hat"], b)

    train_batch_mm = tau0_on(batch)
    heldout_mm = None
    it = iter(train_loader)
    next(it)
    try:
        held = move_batch(next(it), device)
        held["config_id"] = torch.full((held["pose_gt"].shape[0],), CONFIG_VT, device=device, dtype=torch.long)
        assert_batch_shapes(held, held["pose_gt"].shape[0])
        heldout_mm = tau0_on(held)
    except StopIteration:
        log.warning("train loader has no second batch; held-out tau0 skipped")
    eval_batch_mm = None
    if eval_dataset is not None and len(eval_dataset):
        eval_loader = DataLoader(eval_dataset, batch_size=batch["pose_gt"].shape[0],
                                 shuffle=False, num_workers=0, collate_fn=collate_windows)
        ev = move_batch(next(iter(eval_loader)), device)
        ev["config_id"] = torch.full((ev["pose_gt"].shape[0],), CONFIG_VT, device=device, dtype=torch.long)
        assert_batch_shapes(ev, ev["pose_gt"].shape[0])
        eval_batch_mm = tau0_on(ev)
    result = {
        "tau0_fixed_train_batch_mm": train_batch_mm,
        "tau0_next_train_batch_mm": heldout_mm,
        "tau0_first_eval_batch_mm": eval_batch_mm,
    }
    print("== Check 2b: cross-batch tau0 (how much does '8 windows don't generalize' cost) ==")
    print("  tau0 fixed train batch : %.1f mm" % train_batch_mm)
    print("  tau0 next train batch  : %s" % ("%.1f mm" % heldout_mm if heldout_mm is not None else "n/a"))
    print("  tau0 first eval batch  : %s" % ("%.1f mm" % eval_batch_mm if eval_batch_mm is not None else "n/a"))
    if heldout_mm is not None:
        print("  gap (held-out - train): %.1f mm" % (heldout_mm - train_batch_mm))
    return result


def check3_tau_grid(diffusion, oracle: FakeOracleModel, batch, sample_steps, out_dir) -> dict:
    """Actual τ schedule, first/last ᾱ, per-step ‖x_τ‖ vs the training q_sample envelope."""
    pose_gt = batch["pose_gt"]
    schedule = _timestep_schedule(diffusion.n_train_steps, int(sample_steps))
    rows = []
    suspicious = []
    for x_in, tau in zip(oracle.x_inputs, oracle.taus):
        abar = float(diffusion.alphas_cumprod[tau])
        mean, std = _q_sample_envelope(diffusion, pose_gt, tau)
        x_norm = float(x_in.norm().item())
        ratio = x_norm / mean if mean > 0 else float("inf")
        within = abs(x_norm - mean) <= 3.0 * std
        if not within:
            suspicious.append(tau)
        rows.append({"tau": tau, "abar": abar, "sqrt_abar": math.sqrt(abar),
                     "x_norm_ddim": x_norm, "q_sample_mean": mean,
                     "q_sample_std": std, "ratio": ratio, "within_3sigma": within})
    first_tau, last_tau = schedule[0], schedule[-1]
    result = {
        "schedule": schedule,
        "n_train_steps": diffusion.n_train_steps,
        "abar_first_step": float(diffusion.alphas_cumprod[first_tau]),
        "sqrt_abar_first_step": float(diffusion.sqrt_alphas_cumprod[first_tau]),
        "abar_last_step": float(diffusion.alphas_cumprod[last_tau]),
        "sqrt_abar_last_step": float(diffusion.sqrt_alphas_cumprod[last_tau]),
        "per_step": rows,
        "steps_outside_q_sample_3sigma": suspicious,
        "training_tau_range": [0, diffusion.n_train_steps - 1],
        "verdict": "PASS" if not suspicious else "FAIL",
    }
    csv_path = out_dir / "check3_tau_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tau", "abar", "sqrt_abar", "x_norm_ddim",
                                                    "q_sample_mean", "q_sample_std", "ratio", "within_3sigma"])
        writer.writeheader()
        writer.writerows(rows)
    print("== Check 3: tau grid / abar / per-step |x_tau| ==")
    print("  schedule: %s ... %s (%d steps), range [0, %d]"
          % (schedule[:5], schedule[-3:], len(schedule), diffusion.n_train_steps - 1))
    print("  first step tau=%d: abar=%.3e sqrt_abar=%.3e" % (first_tau, result["abar_first_step"], result["sqrt_abar_first_step"]))
    print("  last  step tau=%d: abar=%.6f sqrt_abar=%.6f  (note: NOT exactly 1.0 — cosine schedule)" % (last_tau, result["abar_last_step"], result["sqrt_abar_last_step"]))
    for row in rows[:3] + rows[-3:]:
        print("  tau=%4d  |x|=%.3f  q_sample=%.3f±%.3f  ratio=%.3f  %s"
              % (row["tau"], row["x_norm_ddim"], row["q_sample_mean"], row["q_sample_std"], row["ratio"],
                 "OK" if row["within_3sigma"] else "OUT"))
    print("  wrote %s" % csv_path)
    print("  => %s%s" % (result["verdict"],
                        "" if not suspicious else "  (suspicious taus: %s)" % suspicious))
    return result


def check4_x0hat_vs_tau0(model, diffusion, cond, v_feat, t_raw, t_phys, batch, sample_steps, device, out_dir) -> tuple:
    """Last-step x0_hat vs tau0 output element-wise, plus the full per-step trace.

    Returns (result, out0, pred, final) so check 5 can reuse the same outputs.
    """
    pose_gt = batch["pose_gt"]
    with torch.inference_mode():
        tau0_mpjpe, out0 = _tau0_probe(model, v_feat, t_raw, t_phys, batch, device)
        rec = RecordingModel(model)
        pred = diffusion.ddim_sample_loop(
            rec,
            tau_related_kwargs=cond,
            shape=(pose_gt.shape[0], pose_gt.shape[1], POSE_DIM),
            steps=sample_steps,
            eta=0.0,
            device=device,
        )
    final = rec.trace[-1]["x0_hat"]
    final_in = rec.trace[-1]["x_in"]
    loop_returns_final_call = bool(torch.equal(pred, final))
    diff = final - out0
    per_window_rel = (diff.reshape(pose_gt.shape[0], -1).norm(dim=-1)
                      / out0.reshape(pose_gt.shape[0], -1).norm(dim=-1).clamp_min(1e-8)).tolist()
    corr = float(torch.corrcoef(torch.stack([out0.flatten().double(), final.flatten().double()]))[0, 1].item())
    ddim_mpjpe = _fk_mpjpe_mm(pred, batch)
    # Model's tau=0 sensitivity to the final-call input: GT input vs the actual x_penultimate.
    with torch.inference_mode():
        zero = torch.zeros(pose_gt.shape[0], device=device, dtype=torch.long)
        out_pen = model(cond["V_feat"], cond["T_raw"], cond["T_phys"], final_in, zero,
                        cond["config_id"], cond.get("session_id"))["x0_hat"]
    pen_out_diff = _rms(out0 - out_pen)
    pen_in_diff = _rms(final_in - pose_gt)
    echo_factor = pen_out_diff / max(pen_in_diff, 1e-12)  # 1.0 = output copies its input
    trace_rows = []
    prev_x0 = None
    for entry in rec.trace:
        tau = entry["tau"]
        drift = _rms(entry["x0_hat"] - prev_x0) if prev_x0 is not None else None
        trace_rows.append({
            "tau": tau,
            "x_in_rms": _rms(entry["x_in"]),
            "x_in_to_gt_rms": _rms(entry["x_in"] - pose_gt),
            "x0hat_to_gt_rms": _rms(entry["x0_hat"] - pose_gt),
            "x0hat_to_out0_rms": _rms(entry["x0_hat"] - out0),
            "x0hat_drift_from_prev_rms": drift,
        })
        prev_x0 = entry["x0_hat"]
    csv_path = out_dir / "check4_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    result = {
        "tau0_mpjpe_mm": tau0_mpjpe,
        "ddim_mpjpe_mm": ddim_mpjpe,
        "loop_returns_final_call_output": loop_returns_final_call,
        "rms_diff_tau0_vs_final": _rms(diff),
        "rms_tau0_vs_gt": _rms(out0 - pose_gt),
        "rms_final_vs_gt": _rms(final - pose_gt),
        "max_abs_diff_tau0_vs_final": float(diff.abs().max().item()),
        "corr_tau0_final": corr,
        "per_window_relative_diff": [float(v) for v in per_window_rel],
        "final_call_input_to_gt_rms": _rms(final_in - pose_gt),
        "tau0_sensitivity_to_final_input": {
            "rms_out(x=GT)-out(x=x_penultimate)": pen_out_diff,
            "rms_out(x=x_penultimate)_vs_gt": _rms(out_pen - pose_gt),
            "echo_factor": echo_factor,
        },
        "per_step": trace_rows,
        "verdict": None,  # informational: PASS/FAIL decided by interpretation below
    }
    print("== Check 4: last-step x0_hat vs tau0 output ==")
    print("  tau0 MPJPE %.1f mm   DDIM MPJPE %.1f mm   (loop returns final call: %s)"
          % (tau0_mpjpe, ddim_mpjpe, loop_returns_final_call))
    print("  RMS(tau0 - final) = %.4g   RMS(tau0 - GT) = %.4g   RMS(final - GT) = %.4g"
          % (result["rms_diff_tau0_vs_final"], result["rms_tau0_vs_gt"], result["rms_final_vs_gt"]))
    print("  max abs diff = %.4g   corr(tau0, final) = %.4f" % (result["max_abs_diff_tau0_vs_final"], corr))
    print("  per-window rel diff: %s" % ", ".join("%.3f" % v for v in per_window_rel))
    print("  final-call input x_penultimate: RMS to GT = %.4g" % result["final_call_input_to_gt_rms"])
    print("  model tau=0 x_tau sensitivity: RMS(out(x=GT)-out(x=pen)) = %.4g  (echo factor = %.3f, 1.0 = pure input copy)"
          % (pen_out_diff, echo_factor))
    print("  trace (first / last 2 steps):")
    for row in trace_rows[:2] + trace_rows[-2:]:
        print("    tau=%4d |x_in|=%.3f  |x_in-GT|=%.4g  |x0_hat-GT|=%.4g  |x0_hat-out0|=%.4g  drift=%s"
              % (row["tau"], row["x_in_rms"], row["x_in_to_gt_rms"], row["x0hat_to_gt_rms"],
                 row["x0hat_to_out0_rms"], "%.4g" % row["x0hat_drift_from_prev_rms"] if row["x0hat_drift_from_prev_rms"] is not None else "-"))
    print("  wrote %s" % csv_path)
    if result["rms_diff_tau0_vs_final"] < 0.1 * result["rms_tau0_vs_gt"]:
        print("  => tau0 ≈ final (model ignores x_tau): the 220mm gap is NOT explained by the model; suspect the sampler/conditioning path (see checks 1-3, 5)")
    else:
        print("  => tau0 != final: the model DOES use x_tau at tau=0, so the chain error enters through x_penultimate — the 'g(F) only' assumption fails")
    return result, out0, pred, final


def check5_rotations(batch, out0, pred, final, device) -> dict:
    """6D→SO(3) path identity + orthonormality on both eval paths."""
    def to_rot(pose):
        return rot6d_to_rotmat(pose.reshape(*pose.shape[:2], pose.shape[-1] // 6, 6))

    with torch.inference_mode():
        stats = {
            "gt_pose": _ortho_stats(to_rot(batch["pose_gt"])),
            "tau0_output": _ortho_stats(to_rot(out0)),
            "ddim_output": _ortho_stats(to_rot(pred)),
            "ddim_final_step_x0hat": _ortho_stats(to_rot(final)),
        }
    # torch vs numpy 6D->SO(3) conversion consistency on the same input
    # (rot6d_to_rotmat_np is element-wise: reshape to joints like fk_pose6d_np does).
    pose_np = batch["pose_gt"].detach().cpu().numpy()
    gt_np = rot6d_to_rotmat_np(pose_np.reshape(-1, pose_np.shape[-1] // 6, 6))
    gt_torch = to_rot(batch["pose_gt"]).detach().cpu().numpy()
    torch_np_max_diff = float(np.abs(gt_np.reshape(gt_torch.shape) - gt_torch).max())
    result = {
        "fk_path_shared": True,
        "fk_path_note": "r_test7_legacy_overfit.py: both _tau0_mpjpe_mm and _ddim_mpjpe_mm call the same "
                        "anysole.geometry.fk_pose6d -> rot6d_to_rotmat (torch Gram-Schmidt); "
                        "both anchor with batch['trans_gt'] + trans_anchor.",
        "orthonormality": stats,
        "torch_vs_numpy_6d_to_so3_max_diff": torch_np_max_diff,
        "verdict": "PASS" if all(s["max_abs_RtR_minus_I"] < 1e-4 and s["max_abs_det_minus_1"] < 1e-4
                                for s in stats.values()) else "FAIL",
    }
    print("== Check 5: 6D -> SO(3) on both eval paths ==")
    print("  both probes: fk_pose6d -> rot6d_to_rotmat (torch), trans = trans_gt + anchor (shared)")
    for name, s in stats.items():
        print("  %-24s max|R'R-I|=%.2e  mean=%.2e  max|det-1|=%.2e"
              % (name, s["max_abs_RtR_minus_I"], s["mean_abs_RtR_minus_I"], s["max_abs_det_minus_1"]))
    print("  torch vs numpy 6D->SO3 max abs diff: %.3e" % torch_np_max_diff)
    print("  => %s" % result["verdict"])
    return result


# --------------------------------------------------------------------------
# Training (reproduces Test9 for one lr)
# --------------------------------------------------------------------------

def train_overfit(args, config, batch, v_feat, t_raw, t_phys, diffusion, sample_steps, device, out_dir) -> dict:
    seed_everything(args.seed)
    model = AnySoleModel(
        d=int(config["d_model"]),
        tw=int(config["tw"]),
        dropout=float(args.dropout),
        modal=str(args.modal),
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train = int(config["diffusion_train_steps"])
    rows = []
    log.info("overfit: lr={} steps={} dropout={}", args.lr, args.steps, args.dropout)
    for step in range(int(args.steps)):
        tau = torch.randint(0, n_train, (batch["pose_gt"].shape[0],), device=device, dtype=torch.long)
        x_tau = diffusion.q_sample(batch["pose_gt"], tau)
        optimizer.zero_grad(set_to_none=True)
        out = model(v_feat, t_raw, t_phys, x_tau, tau, batch["config_id"], batch.get("session_id"))
        loss = F.mse_loss(out["x0_hat"], batch["pose_gt"])
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            log.warning("non-finite at step {}; skipped", step)
            continue
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % args.log_every == 0:
            want_sample = step == 0 or (step + 1) % args.sample_every == 0
            row = {
                "lr": args.lr,
                "step": step + 1,
                "L_pose": float(loss.detach().item()),
                "tau0_mpjpe_mm": _tau0_probe(model, v_feat, t_raw, t_phys, batch, device)[0],
                "ddim_mpjpe_mm": None,
            }
            if want_sample:
                with torch.inference_mode():
                    cond = {
                        "V_feat": v_feat,
                        "T_raw": t_raw,
                        "T_phys": t_phys,
                        "config_id": batch["config_id"],
                        "session_id": batch.get("session_id"),
                    }
                    pred = diffusion.ddim_sample_loop(
                        model,
                        tau_related_kwargs=cond,
                        shape=(batch["pose_gt"].shape[0], batch["pose_gt"].shape[1], POSE_DIM),
                        steps=sample_steps,
                        eta=0.0,
                        device=device,
                    )
                row["ddim_mpjpe_mm"] = _fk_mpjpe_mm(pred, batch)
            rows.append(row)
            print("lr=%g step=%d L_pose=%.6f tau0=%.1fmm%s"
                  % (args.lr, row["step"], row["L_pose"], row["tau0_mpjpe_mm"],
                     (" ddim=%.1fmm" % row["ddim_mpjpe_mm"]) if row["ddim_mpjpe_mm"] is not None else ""))
    csv_path = out_dir / "train_log.csv"
    fieldnames = ["lr", "step", "L_pose", "tau0_mpjpe_mm", "ddim_mpjpe_mm"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in fieldnames})
    log.info("wrote {}", csv_path)
    if args.save_ckpt:
        torch.save({"model": model.state_dict(), "config": dict(config)}, out_dir / ("overfit_ckpt_%g.pt" % args.lr))
        log.info("wrote {}", out_dir / ("overfit_ckpt_%g.pt" % args.lr))
    final = rows[-1]
    ddim_final = final["ddim_mpjpe_mm"] if final["ddim_mpjpe_mm"] is not None else [r["ddim_mpjpe_mm"] for r in rows if r["ddim_mpjpe_mm"] is not None][-1]
    return {
        "model": model,
        "final_L_pose": final["L_pose"],
        "final_tau0_mpjpe_mm": final["tau0_mpjpe_mm"],
        "final_ddim_mpjpe_mm": ddim_final,
    }


def load_overfit_ckpt(args, config, device) -> AnySoleModel:
    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    saved = checkpoint.get("config", {})
    model = AnySoleModel(
        d=int(saved.get("d_model", config["d_model"])),
        tw=int(saved.get("tw", config["tw"])),
        dropout=float(saved.get("dropout", args.dropout)),
        modal=str(saved.get("modal", args.modal)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    log.info("loaded overfit checkpoint {}", args.ckpt)
    return model


# --------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test9.1: dissect the Test9 tau0-vs-DDIM gap (fake oracle, batch identity, tau grid, x0_hat trace, 6D->SO3).")
    cli_common.add_common_args(
        parser,
        session=False,
        split_csv=False,
        split=False,
        gen=False,
        fps=False,
        stride=False,
        max_frames=False,
        out_dir=True,
        out_dir_default=cli_common.DISPLAY_ROOT / "Test9_1_sampler",
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "anysole" / "configs" / "v1.yaml"))
    parser.add_argument("--contact-method", type=str, default="", help="Contact-label scheme (default: config's).")
    parser.add_argument("--modal", type=str, default=MODEL_ANYSOLEV1)
    parser.add_argument("--batch-size", type=int, default=8, help="Windows in the fixed batch.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Single overfit lr (Test9's best tau0/DDIM row).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dropout", type=float, default=0.0, help="Model nn.Dropout rate (modality dropout is off by construction: VT-only).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=None, help="DDIM steps (default: config diffusion_sample_steps).")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--skip-train", action="store_true", help="Run only the model-free checks 1-3 (no training, no model).")
    parser.add_argument("--ckpt", type=str, default="", help="Load an overfit checkpoint instead of retraining (skips training).")
    parser.add_argument("--save-ckpt", action="store_true", help="Also save the overfit checkpoint.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.contact_method:
        config["contact_method"] = str(args.contact_method)
    device = resolve_device(args.device)
    out_dir = Path(cli_common.resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli_common.outputs_ready([out_dir / "test91_report.json"]) and not args.force:
        log.info("Skip Test9.1: {} already exists (use --force to overwrite)", out_dir / "test91_report.json")
        return 0

    dataset = AnySoleDataset(
        mode="train",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=int(config["tw"]),
        contact_method=str(config["contact_method"]),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_windows)
    batch = move_batch(next(iter(loader)), device)
    bsz = batch["pose_gt"].shape[0]
    batch["config_id"] = torch.full((bsz,), CONFIG_VT, device=device, dtype=torch.long)
    assert_batch_shapes(batch, bsz)
    v_feat, t_raw, t_phys = condition_inputs(batch, batch["config_id"])
    cond = {
        "V_feat": v_feat,
        "T_raw": t_raw,
        "T_phys": t_phys,
        "config_id": batch["config_id"],
        "session_id": batch.get("session_id"),
    }
    log.info("fixed batch: {} windows (same first-batch draw as Test9)", bsz)

    diffusion = GaussianDiffusion(n_train_steps=int(config["diffusion_train_steps"]))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])

    report: dict = {
        "config": {"batch_windows": bsz, "steps": int(args.steps), "lr": float(args.lr),
                   "dropout": float(args.dropout), "modal": str(args.modal),
                   "contact_method": str(config["contact_method"]), "seed": int(args.seed),
                   "n_train_steps": diffusion.n_train_steps, "sample_steps": sample_steps,
                   "loss": "pose-only (L_pose; same as Test9)"},
    }

    # --- model-free checks -------------------------------------------------
    check1, oracle = check1_fake_oracle(diffusion, cond, batch, sample_steps, device)
    check2a = check2a_same_batch(batch, cond)
    check3 = check3_tau_grid(diffusion, oracle, batch, sample_steps, out_dir)
    report["check1_fake_oracle"] = check1
    report["check2_same_batch"] = check2a
    report["check3_tau_grid"] = check3

    # --- model-dependent checks -------------------------------------------
    if args.skip_train:
        log.info("--skip-train: checks 2b/4/5 need a model; skipping")
        report["check2b_heldout_tau0"] = None
        report["check4_x0hat_vs_tau0"] = None
        report["check5_rotations"] = None
    else:
        if args.ckpt:
            model = load_overfit_ckpt(args, config, device)
        else:
            trained = train_overfit(args, config, batch, v_feat, t_raw, t_phys, diffusion, sample_steps, device, out_dir)
            model = trained["model"]
            report["overfit"] = {"final_L_pose": trained["final_L_pose"],
                                 "final_tau0_mpjpe_mm": trained["final_tau0_mpjpe_mm"],
                                 "final_ddim_mpjpe_mm": trained["final_ddim_mpjpe_mm"]}
        model.eval()
        eval_dataset = AnySoleDataset(
            mode="eval",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=int(config["tw"]),
            contact_method=str(config["contact_method"]),
        )
        check2b = check2b_heldout_tau0(model, batch, loader, eval_dataset, device)
        check4, out0, pred, final = check4_x0hat_vs_tau0(model, diffusion, cond, v_feat, t_raw, t_phys, batch, sample_steps, device, out_dir)
        check5 = check5_rotations(batch, out0, pred, final, device)
        report["check2b_heldout_tau0"] = check2b
        report["check4_x0hat_vs_tau0"] = check4
        report["check5_rotations"] = check5

    report_path = out_dir / "test91_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    log.info("wrote {}", report_path)

    print("\n== Test9.1 summary ==")
    for name in ("check1_fake_oracle", "check2_same_batch", "check3_tau_grid", "check5_rotations"):
        entry = report.get(name)
        if entry and entry.get("verdict"):
            print("  %-22s %s" % (name, entry["verdict"]))
    if report.get("check4_x0hat_vs_tau0"):
        print("  %-22s see Check 4 printout above (no hard threshold)" % "check4_x0hat_vs_tau0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
