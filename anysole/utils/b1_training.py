"""B1 coordinate adapters and full-update KD on the shared visual lifecycle."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from anysole.utils.geometry import rot6d_to_rotmat, rotmat_to_6d


def cached_fingerprints(config):
    """Fingerprint the consumed cached frontend, without pretending to run it.

    Migrated training needs real cached tensors and SMPL, not the unused HRNet
    or detector executables. Their original content identities are retained in
    each cache receipt and explicitly distinguished from current validation.
    """
    import json
    from pathlib import Path
    from anysole.train import canonical_hash, visual_source_hash
    from anysole.data.workspace_adapter import load_grouped_manifest, load_grouped_stats, sha256_file
    records, meta = load_grouped_manifest(Path(config["manifest_path"]))
    stats = load_grouped_stats(Path(config["stats_path"]), manifest_sha256=meta["manifest_sha256"])
    path = Path(config["feature_cache_root"]) / "index.json"
    index = json.loads(path.read_text())
    first = next(iter(index.get("entries", {}).values()), {}).get("provenance", {})
    keys = ("asset_hashes", "crop_source_sha256", "extractor_sha256", "frontend_versions",
            "hrnet_source_commit", "hrnet_source_files", "numeric_backend", "crop_shape")
    frontend = {key: first.get(key) for key in keys}
    frontend["mode"] = "cached_provenance_only; original frontend assets not rerun/revalidated"
    values = {"manifest_hash": meta["manifest_sha256"], "split_hash": sha256_file(Path(config["split_csv"])),
              "stats_hash": stats["stats_sha256"], "stats_file_hash": sha256_file(Path(config["stats_path"])),
              "cache_index_hash": sha256_file(path), "frontend_hash": index.get("frontend_sha256"),
              "frontend_runtime_hash": canonical_hash(frontend), "source_hash": visual_source_hash(),
              "smpl_runtime_model_hash": sha256_file(Path(config["smpl_model_path"]))}
    return values, records, stats, index


def cached_preflight(config):
    from pathlib import Path
    from anysole.data.workspace_adapter import sha256_file
    errors, values, missing = [], None, []
    try:
        values, records, stats, index = cached_fingerprints(config)
        if not index.get("complete") or index.get("manifest_sha256") != values["manifest_hash"]:
            errors.append("B1 cache index incomplete or manifest identity changed")
        for record in records:
            if not record["eligible"] or not set(record["split_roles"]) & {"train", "val"}:
                continue
            for view in record["views"]:
                if view["camera_id"] not in (1, 2, 4) or not view["sources"]:
                    continue
                key = f"{record['recording_id']}/cam{view['camera_id']}"
                entry = index.get("entries", {}).get(key, {})
                path = Path(config["feature_cache_root"]) / entry.get("path", "MISSING")
                if not path.is_file():
                    missing.append(key)
                elif sha256_file(path) != entry.get("sha256"):
                    errors.append("B1 feature tensor changed: "+key)
                if entry.get("provenance", {}).get("frontend_sha256") != index.get("frontend_sha256"):
                    errors.append("B1 mixed cached frontend: "+key)
        if missing:
            errors.append(f"B1 missing{len(missing)} train/val feature caches")
        for key, value in values.items():
            if config.get(key) is not None and config[key] != value:
                errors.append("B1 configured fingerprint changed: "+key)
        if stats.get("pressure_train_statistics", {}).get("thresholds_sha256") != config.get("pressure_train_statistics_hash"):
            errors.append("B1 pressure train statistics identity changed")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(type(exc).__name__+": "+str(exc))
    return {"schema": "anysole.b1.cached-preflight.v1", "ready": not errors, "errors": errors,
            "fingerprints": values, "missing_caches": missing,
            "frontend_validation": "actual train/val tensor bytes + retained extraction provenance; original unused frontend assets not revalidated",
            "cam3_inputs_read": False}


def initialize_models(model, config, device, *, resume=None):
    """Fresh stage optimizer; exact B0 weights and separately frozen teacher."""
    from pathlib import Path
    from anysole.registry import get_builder
    from anysole.data.workspace_adapter import sha256_file
    from anysole.train import restore_visual_runtime_state, atomic_json
    receipt = {"schema": "anysole.b1.initialization.v1", "run": config["b1_run_id"], "seed": config["seed"]}
    if config["b1_run_id"] != "B0":
        path = Path(config["baseline_checkpoint_path"])
        if sha256_file(path) != config["baseline_checkpoint_hash"]:
            raise ValueError("B1 baseline checkpoint changed")
        base = torch.load(path, map_location="cpu", weights_only=False)
        if (base["config"].get("b1_run_id") != "B0" or base["config"]["seed"] != config["seed"] or
                base["config"].get("stats_hash") != config.get("stats_hash")):
            raise ValueError("B1 baseline lineage/seed/statistics mismatch")
        if resume is None:
            if config["b1_role"] == "teacher":
                receipt.update(model.initialize_from_baseline(base["model"]))
            else:
                model.load_state_dict(base["model"], strict=True)
        receipt["baseline_checkpoint_hash"] = config["baseline_checkpoint_hash"]
    if float(config.get("kd_pose_local23", 0)) > 0:
        path = Path(config["teacher_checkpoint_path"])
        if sha256_file(path) != config["teacher_checkpoint_hash"]:
            raise ValueError("B1 frozen teacher checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        teacher_config = checkpoint["config"]
        if (teacher_config.get("b1_role") != "teacher" or teacher_config["seed"] != config["seed"] or
                teacher_config.get("baseline_checkpoint_hash") != config.get("baseline_checkpoint_hash") or
                teacher_config.get("stats_hash") != config.get("stats_hash")):
            raise ValueError("B1 teacher does not descend from this exact B0")
        teacher = get_builder("visual").build(teacher_config)
        teacher.load_state_dict(checkpoint["model"], strict=True)
        restore_visual_runtime_state(teacher, checkpoint)
        teacher.requires_grad_(False).eval().to(device)
        # Deliberately not registered: no teacher parameters in student
        # optimizer or student-only checkpoint/deployment artifacts.
        object.__setattr__(model, "_b1_teacher", teacher)
        receipt["teacher_checkpoint_hash"] = config["teacher_checkpoint_hash"]
        receipt["teacher_frozen"] = True
    if resume is None:
        atomic_json(Path(config["run_dir"]) / "initialization.json", receipt)
    return receipt


def record_exposure(state, selected, keys, cameras):
    ledger = state["exposure_ledger"]
    if len(selected) != len(cameras):
        raise ValueError("B1 dropout exposure/group count mismatch")
    for index, used in zip(selected, cameras):
        key = keys[index]
        ledger["per_key"][key] = ledger["per_key"].get(key, 0) + 1
        for camera in used:
            camera = str(camera)
            if camera not in ("1", "2", "4"):
                raise ValueError("B1 exposure contains a forbidden camera")
            ledger["per_camera"][camera] = ledger["per_camera"].get(camera, 0) + 1
            by_key = ledger["per_camera_key"].setdefault(camera, {})
            by_key[key] = by_key.get(key, 0) + 1


def validate_exposure(state, config, *, terminal=False, keys=None, weights=None):
    ledger = state.get("exposure_ledger", {})
    if (sum(ledger.get("per_camera", {}).values()) != state["rgb_windows"] or
            sum(ledger.get("per_key", {}).values()) != state["physical_group_windows"]):
        raise ValueError("B1 actual-view exposure ledger differs from successful budget")
    if keys is not None and set(ledger.get("per_key", {})) - set(keys):
        raise ValueError("B1 exposure contains unknown physical windows")
    for camera, count in ledger.get("per_camera", {}).items():
        if camera not in ("1", "2", "4") or sum(ledger["per_camera_key"].get(camera, {}).values()) != count:
            raise ValueError("B1 per-camera exposure mismatch")
    nominal = state.get("nominal_rgb_windows", 0)
    if nominal != state["successful_updates"] * config["effective_batch_rgb_windows"]:
        raise ValueError("B1 nominal successful-update exposure mismatch")
    if config["b1_role"] == "teacher":
        if state["physical_group_windows"] != 84*state["successful_updates"]:
            raise ValueError("B1 teacher physical group budget mismatch")
    elif state["rgb_windows"] != nominal:
        raise ValueError("B1 independent student RGB budget mismatch")
    if terminal and state["successful_updates"] != config["target_updates"]:
        raise ValueError("B1 incomplete successful update budget")


def camera_supervision(batch, index, *, rotation=None, frame_valid=None, view_edges=None):
    """Flatten selected cameras and rotate only root/global GT, never local23.

    Rotation is composed world->camera from the data adapter. Translation is
    relative to the first valid GT root of this window; no GT enters forward.
    """
    g, c = index.unbind(-1)
    visual = batch["visual"]
    r = visual["camera_R_cw"][g, c] if rotation is None else rotation
    labels = {k: v[g] for k, v in batch["labels"].items()}
    masks = {k: v[g] for k, v in batch["masks"].items()}
    motion = masks["motion_valid"]
    first = motion.long().argmax(-1)
    origin = labels["root"][torch.arange(len(g), device=g.device), first]
    origin = torch.where(motion.any(-1, keepdim=True), origin, torch.zeros_like(origin))
    pose = labels["pose6d"].clone()
    # Invalid poses may contain NaN; they remain masked by the common losses.
    safe_root = torch.where(motion[..., None], pose[..., :6],
                            pose.new_tensor([1., 0, 0, 0, 1, 0]))
    root_r = rot6d_to_rotmat(safe_root)
    pose[..., :6] = rotmat_to_6d(r[:, None] @ root_r)
    labels["pose6d"] = pose
    labels["root"] = torch.einsum("nij,ntj->nti", r, labels["root"] - origin[:, None])
    labels["velocity"] = torch.einsum("nij,ntj->nti", r, labels["velocity"])
    labels["kp_gt"] = torch.einsum("nij,ntkj->ntki", r, labels["kp_gt"] - origin[:, None, None])
    valid = visual["view_valid"][g, c] if frame_valid is None else frame_valid
    edges = (visual["view_edges"][g, c] if frame_valid is None
             else valid[:, :-1] & valid[:, 1:])
    if view_edges is not None:
        edges = view_edges
    result = {"visual": {"V_feat": visual["V_feat"][g, c, None],
                          "view_valid": valid[:, None], "view_edges": edges[:, None],
                          "target_time_s": visual["target_time_s"][g]},
              "labels": labels, "masks": masks, "meta": {}}
    flat = torch.stack([torch.arange(len(g), device=g.device), torch.zeros_like(g)], -1)
    return result, flat


def teacher_supervision(batch, selected_view_valid):
    g = torch.arange(len(selected_view_valid), device=selected_view_valid.device)
    index = torch.stack([g, torch.zeros_like(g)], -1)
    return camera_supervision(batch, index, rotation=batch["visual"]["reference_R_cw"],
                              frame_valid=selected_view_valid.any(1),
                              view_edges=(selected_view_valid[:, :, :-1] & selected_view_valid[:, :, 1:] &
                                          batch["visual"]["view_edges"]).any(1))


def predictions_to_world(out, rotation):
    """Evaluation-only inverse rotation, preserving zero-origin deployment."""
    result = dict(out)
    pose = out["x0_hat"].float().clone()
    pose[..., :6] = rotmat_to_6d(rotation.transpose(-1, -2)[:, None] @ rot6d_to_rotmat(pose[..., :6]))
    result["x0_hat"] = pose
    result["v_hat"] = torch.einsum("nji,ntj->nti", rotation.float(), out["v_hat"].float())
    return result


def kd_masks(batch, index, config):
    g, c = index.unbind(-1)
    teacher = batch["visual"]["view_valid"].sum(1) >= 2
    valid = batch["visual"]["view_valid"][g, c] & teacher[g]
    motion = valid & batch["masks"]["motion_valid"][g]
    pressure = valid[..., None] & batch["masks"]["pressure_valid"][g]
    foot = pressure.reshape(*pressure.shape[:2], 2, 48).all(-1)
    truth = torch.where(pressure, batch["labels"]["pressure"][g], torch.zeros_like(batch["labels"]["pressure"][g]))
    threshold = truth.new_tensor(config["pressure_contact_threshold"])
    active = foot & (truth.reshape(*truth.shape[:2], 2, 48).sum(-1) > threshold)
    return motion, pressure, foot, active


def kd_counts(batch, index, config):
    motion, pressure, foot, active = kd_masks(batch, index, config)
    masks = {"kd_pose_local23": motion.sum() * 23, "kd_pressure": pressure.sum(),
             "kd_total": foot.sum(), "kd_allocation": active.sum(), "kd_contact": foot.sum()}
    return {name: int(count) for name, count in masks.items() if float(config.get(name, 0)) > 0}


def kd_terms(student, teacher, batch, index, config):
    """Raw sums/counts; no microbatch mean and no root/velocity KD."""
    g = index[:, 0]
    motion, pressure, foot, active = kd_masks(batch, index, config)
    counts = kd_counts(batch, index, config)
    zero = student["pressure_hat"].float().reshape(-1)[:0].sum()
    result = {key: {"raw_sum": zero, "valid_count": count, "unit": "dimensionless"}
              for key, count in counts.items()}
    if "kd_pose_local23" in result:
        a = student["x0_hat"].float().reshape(*motion.shape, 24, 6)[motion][:, 1:]
        b = teacher["x0_hat"].detach()[g].float().reshape(*motion.shape, 24, 6)[motion][:, 1:]
        result["kd_pose_local23"]["raw_sum"] = (rot6d_to_rotmat(a) - rot6d_to_rotmat(b)).square().sum()
    if "kd_pressure" in result:
        result["kd_pressure"]["raw_sum"] = (student["pressure_hat"].float()[pressure] -
                                                 teacher["pressure_hat"].detach()[g].float()[pressure]).square().sum()
    if "kd_total" in result:
        a = torch.log1p(student["pressure_total"].float()[foot] / 48.)
        b = torch.log1p(teacher["pressure_total"].detach()[g].float()[foot] / 48.)
        scale = student["pressure_total"].new_tensor(config["pressure_log_total_scale"]).expand_as(foot)[foot]
        result["kd_total"]["raw_sum"] = ((a-b) / scale).square().sum()
    if "kd_allocation" in result:
        a = student["pressure_allocation"].float()[active] + 1.e-8
        b = teacher["pressure_allocation"].detach()[g].float()[active] + 1.e-8
        a, b = a/a.sum(-1, keepdim=True), b/b.sum(-1, keepdim=True)
        result["kd_allocation"]["raw_sum"] = (b * (b.log()-a.log())).sum()
    if "kd_contact" in result:
        a = student["pressure_contact_logits"].float()[foot]
        b = teacher["pressure_contact_logits"].detach()[g].float()[foot].sigmoid()
        result["kd_contact"]["raw_sum"] = F.binary_cross_entropy_with_logits(a, b, reduction="sum")
    return result


def validation_diagnostics(series, config):
    """Deduplicated new-val teacher gate on identical >=2-view frame support.

    Every independent B0 prediction is retained; the fused teacher is repeated
    over the same camera-frame IDs. Metrics therefore use identical weighting.
    """
    import numpy as np
    from anysole.train import canonical_hash
    sums = {name: [0., 0] for name in ("pose_local23", "pressure_rmse", "total_rmse", "active_rmse",
                                       "highload_rmse", "peak_mae")}
    tp = fp = fn = 0
    coverage = []
    active_threshold = np.asarray(config["pressure_active_threshold"], np.float64)
    high_threshold = np.asarray(config["pressure_highload_threshold"], np.float64)
    contact = np.asarray(config["pressure_contact_threshold"], np.float64)
    for key, item in sorted(series.items()):
        eligible = item["valid"] & np.asarray(item.get("teacher_eligible", np.zeros(len(item["valid"]))), bool)
        motion = eligible & np.asarray(item["motion_valid"], bool)
        pressure = eligible[:, None] & np.asarray(item["pressure_valid"], bool)
        coverage.append([*key, item["frame_id"].tolist(), motion.astype(int).tolist(), pressure.astype(int).tolist()])
        if motion.any():
            target = rot6d_to_rotmat(torch.as_tensor(item["gt_pose6d"][motion].reshape(-1, 24, 6))).numpy()
            error = np.square(item["rotation"][motion, 1:] - target[:, 1:]).sum((-1, -2))
            sums["pose_local23"][0] += float(error.sum())
            sums["pose_local23"][1] += error.size
        truth, pred = np.asarray(item["gt_pressure"], np.float64), np.asarray(item["pressure"], np.float64)
        for name, mask in (("pressure_rmse", pressure), ("active_rmse", pressure & (truth > active_threshold)),
                           ("highload_rmse", pressure & (truth > high_threshold))):
            error = pred[mask] - truth[mask]
            sums[name][0] += float(np.square(error).sum())
            sums[name][1] += error.size
        foot = pressure.reshape(-1, 2, 48).all(-1)
        a, b = pred.reshape(-1, 2, 48), truth.reshape(-1, 2, 48)
        totals_a, totals_b = a.sum(-1), b.sum(-1)
        total_error = (totals_a[foot] - totals_b[foot])/48.
        sums["total_rmse"][0] += float(np.square(total_error).sum())
        sums["total_rmse"][1] += total_error.size
        peak_error = np.abs(a.max(-1)[foot] - b.max(-1)[foot])
        sums["peak_mae"][0] += float(peak_error.sum())
        sums["peak_mae"][1] += peak_error.size
        predicted, target = totals_a > contact, totals_b > contact
        tp += int((predicted & target & foot).sum())
        fp += int((predicted & ~target & foot).sum())
        fn += int((~predicted & target & foot).sum())
    result = {}
    for name, (total, count) in sums.items():
        value = total/count if count else None
        if value is not None and name.endswith("rmse"):
            value = float(np.sqrt(value))
        result["val/gate/"+name] = value
        result["val/gate/"+name+"_count"] = count
    result["val/gate/contact_f1"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    result["val/gate_coverage_hash"] = canonical_hash(coverage)
    result["val/gate/support_rule"] = "same recording/camera/frame; >=2 valid seen views; original GT masks"
    return result


def accumulation_step(model, batches, config, optimizer, scaler, device, update, horizon):
    from anysole import train
    from anysole.utils.losses import (compute_visual_loss_counts, compute_visual_loss_terms,
                                     compute_pair_loss_counts, compute_pair_loss_terms, loss_term_weights)
    from anysole.utils.a2_gradients import routed_backward
    teacher_role = config["b1_role"] == "teacher"
    frozen_teacher = getattr(model, "_b1_teacher", None)
    kd = float(config.get("kd_pose_local23", 0)) > 0
    if kd and frozen_teacher is None:
        raise ValueError("B1 KD requires the frozen teacher instance")
    sigma_frozen = update < int(float(config.get("sigma_freeze_frac", .1))*horizon)
    model.pose_head.set_sigma_frozen(sigma_frozen)
    lam = float(config.get("lambda_mv", 0)) * min(update / max(1, int(.1*horizon)), 1.)
    weights = loss_term_weights(config, int(config["tw"]), lambda_mv=lam)
    ramp = min(update / max(1, int(float(config["kd_ramp_frac"])*horizon)), 1.)
    weights.update({key: float(config.get(key, 0))*ramp for key in
                    ("kd_pose_local23", "kd_pressure", "kd_total", "kd_allocation", "kd_contact")})
    prepared, denominators = [], {}
    regularizer_used, actual_views, selected_cameras, teacher_views = False, 0, [], 0
    for original in batches:
        index = train.grouped_view_index(original)
        selected = None
        if teacher_role:
            selected, selected_windows = model.select_views(original["visual"]["view_valid"])
            supervision, flat = teacher_supervision(original, selected)
            actual_views += int(selected_windows.sum())
            cameras = original["meta"]["camera_id"]
            selected_cameras.extend([[int(cameras[g, c]) for c in range(cameras.shape[1]) if selected_windows[g, c]]
                                      for g in range(len(cameras))])
        else:
            supervision, flat = camera_supervision(original, index)
            actual_views += len(index)
            selected_cameras.extend([[int(c) for c in row if c > 0] for row in original["meta"]["camera_id"]])
            if kd:
                teacher_views += int(original["visual"]["view_valid"].any(-1).sum())
        counts = compute_visual_loss_counts(supervision, flat, config, include_assignment=False, sigma_frozen=sigma_frozen)
        include_assignment = not regularizer_used and counts["pose"] + counts["pressure"] > 0
        if include_assignment:
            regularizer_used = True
            counts = compute_visual_loss_counts(supervision, flat, config, include_assignment=True, sigma_frozen=sigma_frozen)
        if lam:
            counts.update(compute_pair_loss_counts(original, index, config))
        if kd:
            counts.update(kd_counts(original, index, config))
        for key, count in counts.items():
            denominators[key] = denominators.get(key, 0) + count
        prepared.append((original, index, selected, include_assignment, counts))
    if denominators.get("pose", 0) + denominators.get("pressure", 0) == 0:
        return {"stepped": False, "reason": "zero_supervision", "counts": denominators,
                "lambda_mv": lam, "sigma_frozen": sigma_frozen}
    optimizer.zero_grad(set_to_none=True)
    raw, weighted = {}, {}
    amp = bool(config.get("amp")) and device.type == "cuda"
    for original, index, selected, include_assignment, expected in prepared:
        batch = train._move_grouped(original, device)
        index = index.to(device)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            visual = batch["visual"]
            if teacher_role:
                selected = selected.to(device)
                out = model.forward_multiview(visual["V_feat"], visual["view_valid"], visual["target_time_s"],
                                              visual["camera_geometry"], selected_view_valid=selected)
                supervision, flat = teacher_supervision(batch, selected)
            else:
                out = train.visual_forward(model, batch, index)
                supervision, flat = camera_supervision(batch, index)
            terms = compute_visual_loss_terms(out, supervision, flat, config, include_assignment=include_assignment)
            if lam:
                terms.update(compute_pair_loss_terms(out, batch, index, config))
            if kd:
                frozen_teacher.eval()
                with torch.no_grad():
                    target = frozen_teacher.forward_multiview(visual["V_feat"], visual["view_valid"],
                                                               visual["target_time_s"], visual["camera_geometry"])
                terms.update(kd_terms(out, target, batch, index, config))
            contributions = {}
            for key, term in terms.items():
                numerator, count = term["raw_sum"], int(term["valid_count"])
                if not bool(torch.isfinite(numerator)) or count != expected.get(key, 0):
                    raise ValueError(f"B1 nonfinite/count mismatch for {key}")
                raw[key] = raw.get(key, 0.) + float(numerator.detach())
                value = numerator / denominators[key] if denominators.get(key, 0) else numerator*0
                contributions[key] = value * weights.get(key, 0.)
                weighted[key] = weighted.get(key, 0.) + float(contributions[key].detach())
        routed_backward(contributions, model, config, scaler)
    result = train.finish_visual_optimizer_step(model, optimizer, scaler, config, raw, weighted, denominators,
                                                lam, sigma_frozen, [])
    result.update(actual_rgb_windows=actual_views, selected_cameras=selected_cameras,
                  teacher_rgb_windows=teacher_views, kd_ramp=ramp)
    return result
