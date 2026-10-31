"""Explicit scope and coordinate definitions for native session evaluation."""
from __future__ import annotations

import numpy as np
from itertools import combinations

PART_METRIC_DEFINITION = "pelvis_local_and_world_parts_v2"


def e3_complete_window_starts(n_frames, window_length, stride=None):
    """Include the real tail using a final overlapping complete window."""
    stride = window_length if stride is None else stride
    if window_length < 1 or stride < 1 or stride > window_length:
        raise ValueError("E3 windows need 1 <= stride <= window_length")
    if n_frames < window_length:
        return []
    starts = list(range(0, n_frames - window_length + 1, stride))
    if starts[-1] != n_frames - window_length:
        starts.append(n_frames - window_length)
    return starts


def e3_evaluation_contract(prediction, *, template_sha256):
    ids = np.asarray(prediction["frame_ids"].detach().cpu(), dtype=np.int64)
    segments = np.asarray(prediction["segment_ids"].detach().cpu(), dtype=np.int64)
    if len(ids) == 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("E3 prediction must contain unique sorted source frames")
    return {"schema": "e3_evaluation_contract.v1", "free_integration": True,
            "trajectory": "predicted_velocity_integrated_once_per_unique_source_frame",
            "coordinates": "fixed_training_world_Y_up_metres; body_scale_not_fitted",
            "shape": "fixed_train_mean_offsets", "template_sha256": template_sha256,
            "optimizer_gt_inputs": [], "camera_ground_scope": "one_recording",
            "contact_reduction": prediction.get("contact_reduction"),
            "lambda_contact_raw": prediction.get("lambda_contact_raw"),
            "contact_scale_caution": "training_raw_time_sum_and_TTA_frame_mean_use_different_lambda_scales",
            "fusion": prediction.get("fusion"),
            "scoring_alignment": "one_SE3_per_contiguous_segment; no_scale; scoring_only",
            "global_metric_names": ["GlobalMPJPE_unaligned", "GlobalMPJPE_segment_SE3", "ATE_segment_SE3"],
            "native_compatibility": "native GT_translation_anchor_per_window remains separately labeled",
            "coverage_protocol": "complete_real_windows; overlap_folded; each_source_frame_scored_once",
            "short_recordings": "no_padding; explicitly_unscored_if_shorter_than_model_window",
            "scored_frames": len(ids), "frame_ids": ids.tolist(),
            "segments": [{"segment_id": int(s), "start": int(ids[segments == s][0]),
                          "stop_inclusive": int(ids[segments == s][-1]),
                          "frames": int((segments == s).sum())} for s in np.unique(segments)]}


def e3_evaluation_manifest_schema():
    """Unbound scientific panel rows; this is not a completed evaluation manifest.

    Root/post-training binds immutable checkpoint/observation/config identities.
    Each seed/visual condition enumerates ALL 15 nonempty camera subsets.
    """
    subsets = [list(c) for size in range(1, 5) for c in combinations((1, 2, 3, 4), size)]
    rows = []
    for recipe in ("U0", "UP_G1", "UC_G1"):
        paths = ("direct",) if recipe == "UP_G1" else ("direct", "0step", "C0", "C_original")
        for seed in (1, 2, 3):
            for condition in ("clean", "occ010", "occ030"):
                for subset in subsets:
                    for initialization in ("V", "VT"):
                        for path in paths:
                            tactile = ("initialization_and_contact" if initialization == "VT" else "contact_only") if path == "C_original" else (
                                "initialization" if initialization == "VT" else "absent")
                            rows.append({"recipe": recipe, "seed": seed, "visual_condition": condition,
                                         "camera_subset": subset, "initialization": initialization,
                                         "path": path, "test_t_use": tactile,
                                         "optimizer_steps": 0 if path in ("direct", "0step") else None,
                                         "checkpoint_sha256": None, "observation_sha256": None,
                                         "tta_config_sha256": None, "status": "unbound"})
    return {"schema": "e3_evaluation_panel.v1", "phase": "G1_original", "seeds": [1, 2, 3],
            "visual_conditions": ["clean", "occ010", "occ030"], "camera_subsets": subsets,
            "rows": rows, "external_baselines": {"original_TTA": "availability_audit_required",
                                                     "calibrated_reference": "REQUIRED_PENDING_unit_and_world_frame_audit",
                                                     "EasyMocap": "not_reproduced; local_DLT_fixed_template_FK_reference_implemented"},
            "statistics": "recording/subject units; seeds_and_subsets_are_paired_repeated_measurements"}


def evaluation_contract(dataset, checkpoint_config=None):
    """Describe real evaluated frames, without changing any score formula."""
    checkpoint_config = checkpoint_config or {}
    covered = [np.zeros(len(s["V_feat"]), dtype=bool) for s in dataset.sessions]
    repeated_frames = 0
    for index, start, stop in dataset.valid_windows:
        repeated_frames += int(covered[index][start:stop].sum())
        covered[index][start:stop] = True
    sessions = []
    for session, seen in zip(dataset.sessions, covered):
        n = len(seen)
        observed = np.asarray(session["fake_mask"]) == 0
        support = np.asarray(session.get("smpl_support_valid", np.ones(n)), dtype=bool)
        sessions.append({"session_id": session["session_id"], "grid_frames": n,
            "observed_valid_frames": int(observed.sum()), "smpl_supported_frames": int(support.sum()),
            "observed_valid_outside_smpl": int((observed & ~support).sum()),
            "scored_frames": int(seen.sum()), "scored_outside_smpl": int((seen & ~support).sum()),
            "unscored_observed_valid_frames": int((observed & ~seen).sum()),
            "frame_ids": np.flatnonzero(seen).tolist()})
    keys = ["grid_frames", "observed_valid_frames", "observed_valid_outside_smpl", "scored_frames",
            "scored_outside_smpl", "unscored_observed_valid_frames"]
    return {"schema": "native_evaluation_contract.v2",
        "part_metric_definition": PART_METRIC_DEFINITION,
        "coordinates": {"MPJPE": "pelvis_aligned", "MPJPE_parts": "pelvis_aligned",
                        "GlobalMPJPE": "world", "GlobalMPJPE_parts": "world",
                        "PA-MPJPE_parts": "one_whole_body_per_frame_similarity_alignment"},
        "trajectory": "ground_truth_previous_frame_translation_anchor_per_window",
        "free_integration": False, "floor": "GT_session_joint_height_percentile_5_scoring_only",
        "coverage_protocol": ("overlapping_complete_windows; unique frame coverage reported" if repeated_frames
                              else "non_overlapping_complete_windows_only; no tail padding"),
        "repeated_window_frame_instances": repeated_frames,
        "evaluation_smpl_support_policy": getattr(dataset, "smpl_support_policy", "legacy_clip"),
        "training_smpl_support_policy": checkpoint_config.get("smpl_support_policy", "legacy_clip"),
        "declared_sessions": len(sessions), "scored_sessions": sum(r["scored_frames"] > 0 for r in sessions),
        "coverage_totals": {key: sum(row[key] for row in sessions) for key in keys}, "sessions": sessions}
