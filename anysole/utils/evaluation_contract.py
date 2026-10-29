"""Explicit scope and coordinate definitions for native session evaluation."""
from __future__ import annotations

import numpy as np

PART_METRIC_DEFINITION = "pelvis_local_and_world_parts_v2"


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
