"""Explicit A2 scientific contracts; absent A2 fields never alter legacy runs."""
from __future__ import annotations

import math

A2_VERSION = "anysole.a2.v1"
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


def is_a2(config):
    if config.get("a2_version") is None:
        if any(k in config for k in ("a2_run_id", "pressure_pair_route", "supervised_pressure_route", "budget_mode")):
            raise ValueError("A2 route/budget fields require explicit a2_version")
        return False
    if config["a2_version"] != A2_VERSION:
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
    if not is_a2(config):
        return
    run = config.get("a2_run_id")
    if run not in A2_RUNS and run != "C-FINAL":
        raise ValueError("unknown or not implemented A2 run; ROI requires a separately frozen deployable feature contract")
    if config.get("seed") != 1 or isinstance(config.get("seed"), bool):
        raise ValueError("all A2 scientific runs require seed1")
    for key in ("pressure_pair_route", "supervised_pressure_route"):
        if config.get(key) not in ("all", "H", "HF"):
            raise ValueError(f"explicit A2 {key} must be all/H/HF")
    if config.get("budget_mode") != ("cam3_exact" if run == "B-CAM3" else "standard"):
        raise ValueError("A2 run and budget_mode disagree")
    pose, pressure = effective_pair_peaks(config)
    if run in A2_RUNS:
        exp, head, ep, et, pair_route, sup_route = A2_RUNS[run]
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
        if config.get("a2_head_seed") != 1:
            raise ValueError("A2 new heads require isolated a2_head_seed=1")
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
