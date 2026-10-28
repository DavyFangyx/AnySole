"""Per-loss same-graph gradient routing, grouped by actual Parameter identity."""
from __future__ import annotations

import torch


def pressure_module(model):
    return model.pressure_readout if hasattr(model, "pressure_readout") else model.aux_heads.t_rec


def parameter_groups(model):
    """Disjoint groups: shared aliases are never admitted by prefix matching."""
    shared = list(model.embeddings.parameters()) if hasattr(model, "embeddings") else []
    shared_ids = {id(p) for p in shared}
    h = [p for p in pressure_module(model).parameters() if p.requires_grad and id(p) not in shared_ids]
    f = [p for p in model.fusion.parameters() if p.requires_grad and id(p) not in shared_ids]
    enc = getattr(getattr(model, "encoders", None), "v_enc", None)
    r = [p for p in enc.parameters() if p.requires_grad and id(p) not in shared_ids] if enc else []
    raw = {"H": h, "F": f, "R": r, "E_shared": [p for p in shared if p.requires_grad]}
    seen = set()
    result = {}
    for name, values in raw.items():
        result[name] = []
        for p in values:
            if id(p) not in seen:
                result[name].append(p)
                seen.add(id(p))
    result["other"] = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    return result


def loss_route(name, config):
    if name in ("kd_pressure", "kd_total", "kd_allocation", "kd_contact"):
        return config.get("pressure_kd_route", "HF")
    if name == "pressure_pair":
        return config.get("pressure_pair_route", "all")
    if name in ("pressure", "pressure_total", "pressure_diff", "pressure_contact"):
        return config.get("supervised_pressure_route", "all")
    return "all"


def routed_backward(contributions, model, config, scaler):
    """Accumulate scaled per-loss gradients; one later unscale/clip/step."""
    groups = parameter_groups(model)
    free, restricted = [], []
    for name, value in contributions.items():
        route = loss_route(name, config)
        if route == "all":
            free.append(value)
        elif route in ("H", "HF"):
            allowed = groups["H"] + (groups["F"] if route == "HF" else [])
            restricted.append((value, allowed))
        else:
            raise ValueError("unknown pressure gradient route")
    for value, parameters in restricted:
        if not parameters:
            raise ValueError("restricted loss has no allowed trainable parameters")
        grads = torch.autograd.grad(scaler.scale(value), parameters, retain_graph=True, allow_unused=True)
        for p, grad in zip(parameters, grads):
            if grad is not None:
                p.grad = grad.detach() if p.grad is None else p.grad + grad.detach()
    if free:
        # Keep insertion order identical to legacy summation.
        objective = next(iter(contributions.values())).reshape(-1)[:0].sum()
        for value in free:
            objective = objective + value
        scaler.scale(objective).backward()


def gradient_statistics(losses, model, *, weights=None):
    """No optimizer/RNG changes; float64 CPU reductions of original derivatives.

    CPU vectors avoid retaining eight diagnostic vectors on the training GPU.
    Derivatives themselves retain the original training dtype and graph.
    """
    groups = {k: v for k, v in parameter_groups(model).items() if k != "other"}
    parameters = [p for values in groups.values() for p in values]
    locations = {id(p): i for i, p in enumerate(parameters)}
    vectors, norms = {}, {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name], norms[name] = {}, {}
        for group, values in groups.items():
            pieces = [(grads[locations[id(p)]].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                       if grads[locations[id(p)]] is not None else torch.zeros(p.numel(), dtype=torch.float64)) for p in values]
            vector = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)
            vectors[name][group] = vector
            norm = float(torch.linalg.vector_norm(vector))
            norms[name][group] = {"raw": norm, "weighted": norm*abs(float((weights or {}).get(name, 1.)))}
    cosines = {}
    names = list(losses)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            cosines[f"{left}:{right}"] = {}
            for group in groups:
                a, b = vectors[left][group], vectors[right][group]
                denom = norms[left][group]["raw"] * norms[right][group]["raw"]
                value = float(torch.dot(a, b))/denom if denom > 0 else None
                if value is not None:
                    if abs(value) > 1. + 1.e-10:
                        raise FloatingPointError("diagnostic cosine outside rounding tolerance")
                    value = max(-1., min(1., value))
                cosines[f"{left}:{right}"][group] = value
    return {"norms": norms, "cosines": cosines,
            "parameter_counts": {k: sum(p.numel() for p in v) for k, v in groups.items()}}
