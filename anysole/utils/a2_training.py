"""A2 checkpointed exposure accounting, initialization identity and fixed probes."""
from __future__ import annotations

import hashlib

import torch


def state_tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_identity(model):
    state = model.state_dict()
    common = {k: v for k, v in state.items() if not k.startswith(("pressure_readout.", "aux_heads.t_rec."))}
    readout = {k: v for k, v in state.items() if k.startswith(("pressure_readout.", "aux_heads.t_rec."))}
    query = {k: v for k, v in readout.items() if k.startswith(("pressure_readout.foot_embedding", "pressure_readout.attention.", "pressure_readout.norm.", "pressure_readout.hidden."))}
    return {"schema": "anysole.a2.initialization.v1", "common_state_hash": state_tensor_hash(common),
            "pressure_state_hash": state_tensor_hash(readout), "query_state_hash": state_tensor_hash(query),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "pressure_parameters": sum(p.numel() for p in (model.pressure_readout if hasattr(model, "pressure_readout") else model.aux_heads.t_rec).parameters()),
            "head_rng_isolated": hasattr(model, "pressure_readout")}


def record_exposure(state, selected, dataset, keys):
    ledger = state.setdefault("exposure_ledger", {"schema": "anysole.a2.exposure.v1", "per_key": {}, "per_camera": {}, "per_camera_key": {}})
    for index in selected:
        key = keys[index]
        ledger["per_key"][key] = ledger["per_key"].get(key, 0)+1
        for camera in dataset.windows[index][3]:
            camera = str(camera)
            ledger["per_camera"][camera] = ledger["per_camera"].get(camera, 0)+1
            by_key = ledger["per_camera_key"].setdefault(camera, {})
            by_key[key] = by_key.get(key, 0)+1
    return ledger


def validate_exposure(state, config, *, terminal=False, keys=None, weights=None):
    if config.get("b1_version") is not None:
        from anysole.utils.b1_training import validate_exposure as validate_b1_exposure
        return validate_b1_exposure(state, config, terminal=terminal, keys=keys, weights=weights)
    ledger = state.get("exposure_ledger", {})
    if sum(ledger.get("per_camera", {}).values()) != state["rgb_windows"] or sum(ledger.get("per_key", {}).values()) != state["physical_group_windows"]:
        raise ValueError("checkpoint exposure ledger disagrees with successful budget")
    if keys is not None:
        expected = dict(zip(keys, weights))
        if any(key not in expected or type(count) is not int or count < 0 for key, count in ledger.get("per_key", {}).items()):
            raise ValueError("checkpoint exposure ledger contains unknown keys/counts")
        if sum(expected[key]*count for key, count in ledger.get("per_key", {}).items()) != state["rgb_windows"]:
            raise ValueError("checkpoint key exposure weights disagree with RGB budget")
    for camera, count in ledger.get("per_camera", {}).items():
        if sum(ledger["per_camera_key"].get(camera, {}).values()) != count:
            raise ValueError("per-camera physical-window exposure ledger mismatch")
    if terminal and config["budget_mode"] == "cam3_exact":
        if ledger["per_camera"].get("3") != config["cam3_target"]:
            raise ValueError("B-CAM3 terminal cam3 exposure is not exactly matched")
        if len(ledger["per_key"]) != 1765 or sorted(ledger["per_key"].values()) != [751]*1200+[752]*565:
            raise ValueError("B-CAM3 physical-window multiset differs from the reference")
        if any(values != ledger["per_key"] for values in ledger["per_camera_key"].values()) or set(ledger["per_camera_key"]) != {"1", "2", "3", "4"}:
            raise ValueError("B-CAM3 per-camera physical-window multisets disagree")


def fixed_gradient_probe(model, batch, config, device, *, update, horizon):
    from anysole.train import (capture_training_rng, restore_training_rng, _move_grouped,
                               grouped_view_index, visual_forward)
    from anysole.utils.losses import compute_visual_loss_terms, compute_pair_loss_terms, loss_term_weights
    from anysole.utils.a2_gradients import gradient_statistics
    rng, training = capture_training_rng(device), model.training
    gradients = {id(p): None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    try:
        # Eval makes probes comparable across checkpoints. Preserve all caller
        # RNG/mode/gradients; no optimizer, scheduler or sampler consumption.
        model.eval()
        batch = _move_grouped(batch, device)
        index = grouped_view_index(batch)
        out = visual_forward(model, batch, index)
        terms = compute_visual_loss_terms(out, batch, index, config)
        terms.update(compute_pair_loss_terms(out, batch, index, config))
        ramp = max(1, int(float(config.get("mv_ramp_frac", .1))*horizon))
        lam = float(config.get("lambda_mv") or 0)*min(update/ramp, 1.)
        weights = loss_term_weights(config, int(config["tw"]), lambda_mv=lam)
        classes = {"supervised_motion": [], "supervised_pressure": [], "pose_pair": ["local23_pair"], "pressure_pair": ["pressure_pair"]}
        for name in terms:
            if name in ("local23_pair", "pressure_pair"):
                continue
            classes["supervised_pressure" if name.startswith("pressure") else "supervised_motion"].append(name)
        losses = {}
        zero = out["pressure_hat"].reshape(-1)[:0].sum()
        for name, members in classes.items():
            raw, weighted = zero, zero
            for member in members:
                term = terms[member]
                value = term["raw_sum"]/term["valid_count"] if term["valid_count"] else term["raw_sum"]*0
                raw = raw + value
                weighted = weighted + value*weights.get(member, 0.)
            losses[name+"/raw"], losses[name+"/weighted"] = raw, weighted
        result = gradient_statistics(losses, model)
        routes = {"supervised_motion": "all", "pose_pair": "all",
                  "supervised_pressure": config.get("supervised_pressure_route", "all"),
                  "pressure_pair": config.get("pressure_pair_route", "all")}
        result["route_matrix"] = routes
        result["norm_definition"] = "norms/cosines are pre-route derivatives; routed_norms/routed_cosines apply exact per-loss whitelists"
        def admitted(name, group):
            route = routes[name.split("/")[0]]
            return route == "all" or group == "H" or (route == "HF" and group == "F")
        result["routed_norms"] = {name: {group: values if admitted(name, group) else {"raw": 0., "weighted": 0.}
                                                       for group, values in by_group.items()}
                                  for name, by_group in result["norms"].items()}
        result["routed_cosines"] = {pair: {group: value if admitted(pair.split(":")[0], group) and admitted(pair.split(":")[1], group) else None
                                                       for group, value in by_group.items()}
                                    for pair, by_group in result["cosines"].items()}
        result.update(update=int(update), lambda_mv=lam, mode="eval_fixed_train_probe", fit_split="train",
                      sample_keys=list(batch["meta"]["sample_key"]),
                      counts={k: v["valid_count"] for k, v in terms.items()},
                      losses={k: float(v.detach()) for k, v in losses.items()})
        return result
    finally:
        model.train(training)
        restore_training_rng(rng, device)
        # autograd.grad must never mutate accumulated .grad buffers.
        for p in model.parameters():
            previous = gradients[id(p)]
            if previous is None and p.grad is not None or previous is not None and (p.grad is None or not torch.equal(previous, p.grad)):
                raise RuntimeError("gradient probe changed a training gradient")
