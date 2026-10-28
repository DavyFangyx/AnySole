"""Explicit A2/A3 contracts; absent version fields never alter legacy runs."""
from __future__ import annotations

import math

A2_VERSION = "anysole.a2.v1"
A3_VERSION = "anysole.a3.v1"
B1_VERSION = "anysole.b1.v1"
B1_RUNS = ("B0", "Teacher", "B0-Continue", "A-Pressure", "B-Pose",
           "B-Pressure-HF", "B-Pressure-ALL", "B-Load-HF", "B-Load-ALL")
A2_HEADS = ("time_pe", "capacity_time_pe", "query_direct", "query_nonnegative", "query_factorized")
# run: (base experiment, head, effective pose peak, pressure peak, pair route, supervised route)
A2_RUNS = {
    "W005": ("view_consist", "time_pe", .2, .05, "all", "all"),
    "W010": ("view_consist", "time_pe", .2, .1, "all", "all"),
    "W030": ("view_consist", "time_pe", .2, .3, "all", "all"),
    "W050": ("view_consist", "time_pe", .2, .5, "all", "all"),
    "G-H": ("view_consist", "time_pe", .2, .2, "H", "all"),
    "G-HF": ("view_consist", "time_pe", .2, .2, "HF", "all"),
    "G-SP-HF": ("pose_pair", "time_pe", .2, 0., "all", "HF"),
    "B-CAM3": ("base_pool", "time_pe", 0., 0., "all", "all"),
    "C0": ("base_pool", "capacity_time_pe", 0., 0., "all", "all"),
    "CP": ("pose_pair", "capacity_time_pe", .2, 0., "all", "all"),
    "Q0": ("base_pool", "query_direct", 0., 0., "all", "all"),
    "QP": ("pose_pair", "query_direct", .2, 0., "all", "all"),
    "Q-NN": ("pose_pair", "query_nonnegative", .2, 0., "all", "all"),
    "Q-FACT": ("pose_pair", "query_factorized", .2, 0., "all", "all"),
    **{run: ("pose_pair", "query_direct", .2, 0., "all", "all")
       for run in ("L-TOTAL", "L-ACTIVE", "L-HUBER", "L-DIFF")},
}
A2_CONDITIONAL = ("L-TOTAL", "L-ACTIVE", "L-HUBER", "L-DIFF", "ROI", "ROI-CTRL", "C-FINAL")
A3_RUNS = {
    f"{family}-{route}-{memory}": (experiment, "query_factorized", pose, pressure, "HF", supervised)
    for family, experiment, pose, pressure in (("B", "base_pool", 0., 0.),
                                               ("P", "pose_pair", .2, 0.),
                                               ("R", "press_pair", 0., .2))
    for route, supervised in (("ALL", "all"), ("HF", "HF"))
    for memory in ("T20", "VT40")
}


def is_a2(config):
    if config.get("a2_version") is None:
        if any(k in config for k in ("a2_run_id", "pressure_pair_route", "supervised_pressure_route", "budget_mode")):
            raise ValueError("A2 route/budget fields require explicit a2_version")
        return False
    if config["a2_version"] not in (A2_VERSION, A3_VERSION):
        raise ValueError("unsupported a2_version")
    return True


def effective_pair_peaks(config):
    peak = float(config.get("lambda_mv") or 0.)
    return peak * float(config.get("mv_pose_weight", 1.)), peak * float(config.get("mv_pressure_weight", 1.))


def a2_budget(mode="standard"):
    if mode not in ("standard", "cam3_exact"):
        raise ValueError("unsupported A2 budget_mode")
    factor = 4 if mode == "cam3_exact" else 1
    return {"budget_mode": mode, "target_updates": 5180*factor,
            "target_rgb_windows": 1326080*factor, "effective_batch_rgb_windows": 256,
            "validation_every_updates": 100*factor,
            "lr_warmup_updates": 259*factor, "sigma_freeze_updates": 518*factor,
            "snapshot_updates": [1300, 2600, 3900, 5180] if factor == 1 else [5180, 10360, 15540, 20720],
            "gradient_probe_updates": [100, 500, 1300, 2600, 3900, 5180],
            **({"cam3_target": 1326080} if factor == 4 else {})}


def _finite_array(config, key, length, *, positive=False, probability=False):
    value = config.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"A2 {key} requires {length} frozen train-only values")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           or (positive and x <= 0) or (probability and not 0 < x < 1) for x in value):
        raise ValueError(f"invalid A2 {key}")


def validate_a2_config(config, *, formal=False):
    if config.get("b1_version") is not None:
        validate_b1_config(config, formal=formal)
        return
    if not is_a2(config):
        return
    a3 = config["a2_version"] == A3_VERSION
    runs = A3_RUNS if a3 else A2_RUNS
    run = config.get("a2_run_id")
    if run not in runs and (a3 or run != "C-FINAL"):
        raise ValueError("unknown or not implemented A2 run; ROI requires a separately frozen deployable feature contract")
    if type(config.get("seed")) is not int or config["seed"] not in ((1, 2, 3) if a3 else (1,)):
        raise ValueError("A3 requires seed1/2/3; A2 requires seed1")
    for key in ("pressure_pair_route", "supervised_pressure_route"):
        if config.get(key) not in ("all", "H", "HF"):
            raise ValueError(f"explicit A2 {key} must be all/H/HF")
    if config.get("budget_mode") != ("a3_extended" if a3 else "cam3_exact" if run == "B-CAM3" else "standard"):
        raise ValueError("A2 run and budget_mode disagree")
    pose, pressure = effective_pair_peaks(config)
    if run in runs:
        exp, head, ep, et, pair_route, sup_route = runs[run]
        for key, expected in (("experiment", exp), ("pressure_head", head),
                              ("pressure_pair_route", pair_route), ("supervised_pressure_route", sup_route)):
            if config.get(key) != expected:
                raise ValueError(f"A2 {run} requires {key}={expected!r}")
        if not (math.isclose(pose, ep, rel_tol=0, abs_tol=1e-12) and math.isclose(pressure, et, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("A2 effective pair peaks disagree with run contract")
    else:
        if len(config.get("parent_ids", [])) != 2 or not config.get("gate_evidence_hash"):
            raise ValueError("C-FINAL requires two frozen passing parents and gate evidence")
    if config.get("pair_shuffle"):
        raise ValueError("A2 does not add shuffle experiments")
    for key, expected in (("effective_pose_pair_peak", pose), ("effective_pressure_pair_peak", pressure)):
        if key not in config or not math.isclose(float(config[key]), expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"A2 requires readable {key} without double weighting")
    for key, expected in (("lr", 1e-4), ("lambda_pose", 3.), ("lambda_traj", 1.),
                          ("lambda_trec", .1), ("mv_ramp_frac", .1), ("lr_warmup_frac", .05),
                          ("sigma_freeze_frac", .1)):
        # lambda_trec is the canonical supervised pressure coefficient.
        if key in config and float(config[key]) != expected:
            raise ValueError(f"A2 shared {key} must remain {expected}")
    if config.get("lr_schedule", "constant") != "constant":
        raise ValueError("A2 preserves the constant-after-warmup scheduler")
    head = config.get("pressure_head")
    if head not in A2_HEADS:
        raise ValueError("unsupported A2 pressure head")
    if head != "time_pe":
        if config.get("a2_head_seed") != (config["seed"] if a3 else 1):
            raise ValueError("new heads require isolated a2_head_seed matching the run seed")
        if config.get("query_layers", 1) != 1 or config.get("query_heads", 8) != 8 or config.get("query_hidden", 128) != 128:
            raise ValueError("A2 query dimensions are frozen at one layer/eight heads/128 hidden")
    if head == "capacity_time_pe" and config.get("capacity_width") != 544:
        raise ValueError("A2 capacity width was frozen at 544 before training")
    if head in ("query_nonnegative", "query_factorized"):
        _finite_array(config, "pressure_cell_mean", 96, positive=True)
        _finite_array(config, "pressure_contact_rate", 2, probability=True)
        _finite_array(config, "pressure_contact_threshold", 2, positive=True)
        _finite_array(config, "pressure_contact_pos_weight", 2, positive=True)
        if config.get("lambda_contact") != .01 or not config.get("pressure_train_statistics_hash"):
            raise ValueError("NN/FACT require shared BCE=.01 and frozen train statistics identity")
    elif float(config.get("lambda_contact", 0.)) != 0.:
        raise ValueError("contact auxiliary belongs only to matched NN/FACT heads")
    loss = {"L-TOTAL": "total", "L-ACTIVE": "active", "L-HUBER": "huber", "L-DIFF": "diff"}.get(run, "mse")
    if config.get("pressure_loss", "mse") != loss and run != "C-FINAL":
        raise ValueError("A2 single loss run must change exactly its declared pressure loss")
    if run in A2_CONDITIONAL and not config.get("gate_evidence_hash"):
        raise ValueError("conditional training requires a frozen gate evidence hash")
    if loss == "active":
        _finite_array(config, "pressure_active_threshold", 96)
        if not 0 <= float(config.get("pressure_active_alpha", -1)) <= 4 or not 0 < float(config.get("pressure_active_normalizer", 0)) <= 5:
            raise ValueError("active loss requires frozen train alpha/mean-weight normalization")
    if loss in ("huber", "diff"):
        _finite_array(config, "pressure_huber_delta" if loss == "huber" else "pressure_diff_delta", 2, positive=True)
    if loss in ("total", "diff") and not 0 < float(config.get("lambda_pressure_aux", 0)) < math.inf:
        raise ValueError("pressure auxiliary requires a finite positive train-probe coefficient")
    offset = config.get("pilot_schedule_update_offset", 0)
    if type(offset) is not int or offset < 0 or offset > int(config.get("target_updates", 5180)):
        raise ValueError("pilot_schedule_update_offset requires a bounded nonnegative integer")
    if "measurement_replica_id" in config:
        replica = config["measurement_replica_id"]
        if not isinstance(replica, str) or not replica.strip():
            raise ValueError("measurement_replica_id must be a nonempty pilot-only string")
        if formal or config.get("stage") != "pilot":
            raise ValueError("formal A2 runs prohibit measurement_replica_id")
    if formal and ("pilot_schedule_update_offset" in config or "timing_warmup_updates" in config):
        raise ValueError("formal A2 runs prohibit benchmark schedule/timing overrides")
    if formal:
        if config.get("save_initial_checkpoint") is not True:
            raise ValueError("formal A2 saves initialization/model/optimizer/sampler/RNG")
    if a3:
        memory = "fusion_tactile_slice" if run.endswith("-T20") else "fusion_all"
        if config.get("query_memory") != memory:
            raise ValueError("A3 run and query_memory disagree")
        if config.get("stage") not in ("pilot", "formal") or (formal and config["stage"] != "formal"):
            raise ValueError("A3 stage and formal validation disagree")
        if config.get("train_cameras") != [1, 2, 4] or config.get("validation_cameras") != [1, 2, 4]:
            raise ValueError("A3 train and validation use only cameras1,2,4")
        if config.get("selection_motion_weight") != .8 or config.get("selection_pressure_weight") != .2:
            raise ValueError("A3 motion/pressure selection weights must be .8/.2")
        for key in ("target_updates", "effective_batch_rgb_windows", "microbatch_rgb_windows"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ValueError(f"A3 requires a positive integer {key}")
        batch, micro = config["effective_batch_rgb_windows"], config["microbatch_rgb_windows"]
        if batch % 3 or micro % 3 or micro < 6 or micro > batch:
            raise ValueError("A3 batches must preserve at least two complete three-view groups")
        if config.get("target_rgb_windows") != config["target_updates"] * batch:
            raise ValueError("A3 target RGB windows must equal successful updates times batch")
        if formal:
            if config["target_updates"] < 10360:
                raise ValueError("A3 formal budget must at least double the previous5180 updates")
            for key in ("selection_m_ref", "selection_p_ref"):
                if not isinstance(config.get(key), (int, float)) or not 0 < config[key] < math.inf:
                    raise ValueError("A3 requires finite positive new-val selection references")
            snapshots = config.get("snapshot_updates", [])
            if (not snapshots or any(type(s) is not int or not 0 < s <= config["target_updates"] for s in snapshots)
                    or snapshots != sorted(set(snapshots)) or snapshots[-1] != config["target_updates"]):
                raise ValueError("A3 snapshots must be ordered unique successful updates through the final budget")
        return
    if formal:
        expected = a2_budget(config["budget_mode"])
        for key, value in expected.items():
            if key == "gradient_probe_updates":
                continue
            if config.get(key) != value:
                raise ValueError(f"A2 formal {key} must equal {value!r}")
        if config.get("microbatch_rgb_windows") != 8:
            raise ValueError("A2 formal microbatch remains eight RGB windows")
        if config.get("selection_m_ref") != .1262301249103615 or config.get("selection_p_ref") != .2082712196486636:
            raise ValueError("A2 preserves frozen historical joint selection references")
        if config.get("reference_epochs") != 740 or config.get("n_cam3_train_windows") != 1765:
            raise ValueError("A2 reference budget requires audited 1765 cam3 windows/740 epochs")


def validate_b1_config(config, *, formal=False):
    """Separate B1 scientific identity; A3 is reused only for architecture."""
    if config.get("b1_version") != B1_VERSION or config.get("a2_version") != A3_VERSION:
        raise ValueError("B1 requires explicit B1 and reused A3 architecture versions")
    if config.get("a2_run_id") is not None:
        raise ValueError("B1 cannot impersonate an A3 run ID")
    run = config.get("b1_run_id")
    if run not in B1_RUNS or config.get("b1_role") != (
            "teacher" if run == "Teacher" else "baseline" if run == "B0" else "student"):
        raise ValueError("B1 run/role mismatch")
    fixed = {
        "schema_version": "view_consistency.v1", "protocol": "anysole.b.cam3_holdout.v1",
        "model_name": "visual", "visual_structure_version": "visual_v1", "strict_visual": True,
        "config_probs": [0., 1., 0.], "pressure_head": "query_factorized",
        "query_memory": "fusion_tactile_slice", "query_layers": 1, "query_heads": 8,
        "query_hidden": 128, "d_model": 256, "tw": 20, "window_length": 20,
        "pressure_time_pe": "before_fusion", "time_pe_scale": 40.,
        "train_cameras": [1, 2, 4], "validation_cameras": [1, 2, 4],
        "teacher_cameras": [1, 2, 4], "teacher_reference_camera": 1,
        "student_output_frame": "camera_relative_origin", "teacher_output_frame": "camera1_relative_origin",
        "supervised_pressure_route": "all", "teacher_supervised_pressure_route": "all",
        "pressure_pair_route": "HF", "budget_mode": "b1", "pair_shuffle": False,
        "mv_pose_weight": 0., "mv_pressure_weight": 1.,
        "selection_motion_weight": .8, "selection_pressure_weight": .2,
        "selection_rule": "pareto_weighted", "selection_tie_break": ["motion", "pressure", "earlier_successful_update"],
        "generate_contact_cache": False, "horizontal_flip": False, "time_stretch": False,
        "mix_occlusion_cache": False, "num_workers": 0, "hrnet_frozen": True,
        "lambda_pose": 3., "lambda_traj": 1., "lambda_trec": .1, "lambda_contact": .01,
        "lr": 1e-4, "lr_schedule": "constant", "kd_ramp_frac": .1,
        "wandb_mode": "offline", "wandb_entity": "foundary-f-beijing-institute-of-technology",
        "wandb_project": "Anysole-B1", "save_initial_checkpoint": True,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"B1 requires {key}={expected!r}")
    if type(config.get("seed")) is not int or config["seed"] not in (1, 2, 3):
        raise ValueError("B1 requires seed1/2/3")
    if config.get("a2_head_seed") != config["seed"] or config.get("stage") not in ("pilot", "formal"):
        raise ValueError("B1 head seed/stage mismatch")
    if config.get("experiment") != ("press_pair" if run == "A-Pressure" else "base_pool"):
        raise ValueError("B1 experiment adapter mismatch")
    if float(config.get("lambda_mv", -1)) != (.2 if run == "A-Pressure" else 0.):
        raise ValueError("only A-Pressure enables B1 pressure pair=.2")
    kd = run.startswith("B-")
    expected = {"kd_pose_local23": .2 if kd else 0., "kd_pressure": .02 if run.startswith("B-Pressure-") else 0.,
                "kd_total": .02 if run.startswith("B-Load-") else 0.,
                "kd_allocation": .01 if run.startswith("B-Load-") else 0.,
                "kd_contact": .005 if run.startswith("B-Load-") else 0.}
    for key, value in expected.items():
        if float(config.get(key, -1)) != value:
            raise ValueError(f"B1 {run} requires {key}={value}")
    route = "all" if run.endswith("-ALL") else "HF"
    if config.get("pressure_kd_route") != route or config.get("kd_quality_mode") != "valid_only":
        raise ValueError("B1 pressure KD route/quality mode mismatch")
    for key in ("kd_root", "kd_velocity", "kd_body_latent", "kd_load_latent"):
        if float(config.get(key, 0)) != 0:
            raise ValueError("B1 does not implement trajectory or latent KD")
    for key, length in (("pressure_cell_mean", 96), ("pressure_contact_threshold", 2),
                        ("pressure_contact_pos_weight", 2), ("pressure_log_total_scale", 2)):
        _finite_array(config, key, length, positive=True)
    _finite_array(config, "pressure_contact_rate", 2, probability=True)
    if not config.get("pressure_train_statistics_hash"):
        raise ValueError("B1 requires new train-only pressure statistics")
    for key in ("target_updates", "effective_batch_rgb_windows", "microbatch_rgb_windows",
                "validation_every_updates", "checkpoint_every_updates"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"B1 requires positive integer {key}")
    batch, micro = config["effective_batch_rgb_windows"], config["microbatch_rgb_windows"]
    if batch != 252 or micro < 3 or micro > batch or micro % 3:
        raise ValueError("B1 uses RGB252 and complete-group microbatches divisible by3")
    if config.get("target_rgb_windows") != config["target_updates"] * batch:
        raise ValueError("B1 nominal update budget mismatch")
    if config.get("b1_role") == "teacher" and config.get("teacher_effective_batch_groups") != 84:
        raise ValueError("B1 teacher uses84 physical groups/update and counts actual RGB views")
    if formal:
        if config["target_updates"] != (20720 if run in ("B0", "Teacher") else 10360):
            raise ValueError("B1 formal stage budget mismatch")
        if config.get("plan_frozen") is not True or config.get("resource_frozen") is not True:
            raise ValueError("B1 formal plan/resources must be frozen")
        for key in ("manifest_hash", "split_hash", "stats_hash", "stats_file_hash", "cache_index_hash",
                    "frontend_hash", "frontend_runtime_hash", "source_hash", "structure_hash", "plan_hash",
                    "selection_coverage_hash", "train_sample_keys_hash", "pressure_train_statistics_hash"):
            value = config.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"B1 unresolved or invalid {key}")
        for key in ("selection_m_ref", "selection_p_ref"):
            if not isinstance(config.get(key), (float, int)) or not 0 < config[key] < math.inf:
                raise ValueError("B1 needs new-val selection references")
        for key in ("max_steps", "limit_sessions", "max_batches", "smoke_steps", "stop_after_updates", "pilot_train_sample_keys"):
            if config.get(key) is not None:
                raise ValueError("B1 formal cannot truncate training")
        if run != "B0" and not config.get("baseline_checkpoint_hash"):
            raise ValueError("B1 continuation requires same-seed B0 checkpoint")
        if kd and not config.get("teacher_checkpoint_hash"):
            raise ValueError("B1 KD requires a frozen teacher checkpoint")
