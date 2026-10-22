"""A2 pressure-only losses with masks-only full-update denominators."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _layout(batch, index):
    g, c = index.unbind(-1)
    frame = batch["visual"]["view_valid"][g, c]
    valid = frame[..., None] & batch["masks"]["pressure_valid"][g]
    foot_valid = valid.reshape(*valid.shape[:2], 2, 48).all(-1)
    times = batch["visual"]["target_time_s"][g]
    dt = times[:, 1:]-times[:, :-1]
    edges = (valid[:, 1:] & valid[:, :-1] & batch["masks"]["pressure_edges"][g]
             & batch["visual"]["view_edges"][g, c, :, None]
             & torch.isclose(dt, dt.new_tensor(.025), rtol=0, atol=1e-9)[..., None])
    return g, valid, foot_valid, edges


def a2_pressure_counts(batch, index, config):
    _, _, foot, edges = _layout(batch, index)
    counts = {}
    if config.get("pressure_head") in ("query_nonnegative", "query_factorized"):
        counts["pressure_contact"] = int(foot.sum())
    loss = config.get("pressure_loss", "mse")
    if loss == "total":
        counts["pressure_total"] = int(foot.sum())
    if loss == "diff":
        counts["pressure_diff"] = int(edges.sum())
    return counts


def _huber(residual, delta):
    # Ordinary Huber, not smooth_l1 (which divides by delta).
    a = residual.abs()
    return torch.where(a <= delta, .5*residual.square(), delta*(a-.5*delta))


def modify_pressure_terms(terms, out, batch, index, config):
    group, valid, foot_valid, edges = _layout(batch, index)
    pred, truth = out["pressure_hat"].float(), batch["labels"]["pressure"][group].float()
    zero = pred.reshape(-1)[:0].sum()
    residual = pred[valid]-truth[valid]
    loss = config.get("pressure_loss", "mse")
    if loss == "active":
        threshold = pred.new_tensor(config["pressure_active_threshold"])
        active = truth > threshold
        weight = (1 + float(config["pressure_active_alpha"])*active) / float(config["pressure_active_normalizer"])
        terms["pressure"]["raw_sum"] = (residual.square()*weight[valid]).sum()
    if loss == "huber":
        delta = pred.new_tensor(config["pressure_huber_delta"]).repeat_interleave(48)
        expanded = delta[None, None].expand_as(pred)
        terms["pressure"]["raw_sum"] = _huber(residual, expanded[valid]).sum()
        terms["pressure"]["unit"] = "normalized_reading_huber"
    counts = a2_pressure_counts(batch, index, config)
    if "pressure_total" in counts:
        # Select all-valid foot rows before summing, so masked NaNs stay out.
        a = pred.reshape(*pred.shape[:2], 2, 48)[foot_valid].sum(-1)/48
        b = truth.reshape(*truth.shape[:2], 2, 48)[foot_valid].sum(-1)/48
        terms["pressure_total"] = {"raw_sum": (a-b).square().sum(), "valid_count": counts["pressure_total"],
                                    "unit": "mean_foot_reading_squared"}
    if "pressure_diff" in counts:
        a = pred[:, 1:][edges]-pred[:, :-1][edges]
        b = truth[:, 1:][edges]-truth[:, :-1][edges]
        delta = pred.new_tensor(config["pressure_diff_delta"]).repeat_interleave(48)[None, None].expand_as(edges)
        terms["pressure_diff"] = {"raw_sum": _huber(a-b, delta[edges]).sum(), "valid_count": counts["pressure_diff"],
                                   "unit": "adjacent_reading_difference_huber"}
    if "pressure_contact" in counts:
        threshold = pred.new_tensor(config["pressure_contact_threshold"])
        # Same sum-of-48 reading threshold as the audited primary metric.
        target = (truth.reshape(*truth.shape[:2], 2, 48).sum(-1) > threshold).to(pred.dtype)
        logits = out["pressure_contact_logits"].float()
        pos = pred.new_tensor(config["pressure_contact_pos_weight"]).expand_as(logits)
        raw = F.binary_cross_entropy_with_logits(logits[foot_valid], target[foot_valid],
                                                  pos_weight=pos[foot_valid], reduction="sum") if counts["pressure_contact"] else zero
        terms["pressure_contact"] = {"raw_sum": raw, "valid_count": counts["pressure_contact"], "unit": "contact_bce"}
    return terms


def calibrate_auxiliary_coefficient(mse, auxiliary, model, *, minimum_norm=1e-12):
    """Once-only train probe: H_Q+F, target .1 of already .1-weighted MSE."""
    from anysole.utils.a2_gradients import parameter_groups
    if not hasattr(model, "pressure_readout"):
        raise ValueError("auxiliary calibration requires the actual new pressure readout")
    groups = parameter_groups(model)
    parameters = groups["H"] + groups["F"]
    def norm(value):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum((g.float().square().sum() for g in gradients if g is not None), value.detach()*0))
    base, aux = float(norm(.1*mse)), float(norm(auxiliary))
    if not (torch.isfinite(torch.tensor([base, aux])).all() and base > 0 and aux > minimum_norm):
        raise ValueError("auxiliary calibration gate failed: no finite nondegenerate train support")
    return {"lambda_pressure_aux": .1*base/aux, "base_weighted_gradient_norm": base,
            "auxiliary_gradient_norm": aux, "parameter_group": "H_Q+F", "fit_split": "train"}
