"""Model registry and shared defaults for the offline experiments.

2026-09-24 independence restructure: structure flags (t_encoder / f2_repr /
pose_parts / soft_parts / gate) are no longer train CLI options — each
model's structure is fixed by its entry script (anysole/models/<name>.py),
and every run trains from random init.  ``train_args`` carries only runtime
hyperparameters.  ``part_json`` (V4B structural input) and
``assign_cluster`` (V4A train protocol) stay.
"""

from __future__ import annotations

from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1]
# V4B's partition is derived from RAW training-data kinematics (2026-09-24,
# model-independent): probe_part_cluster_b_data.py -> results/AnySole/partitions/.
# K9 = the V3_2 对位 comparison; K10 = silhouette-optimal alternative.
_V4B_PART_JSON = "results/AnySole/partitions/partitions_kinematic_b_K9.json"

MODELS = {
    "F0b": dict(label="F0b", name="F0b", which="last", train=True, epochs=400,
                provides=("base",), train_args={"grad_clip": 5.0}),
    "F4a": dict(label="F4a", name="F4a", which="last", train=True, epochs=400,
                provides=("base", "foot_conv"),
                train_args={"grad_clip": 5.0}),
    "F2": dict(label="F2", name="F2", which="last", train=True, epochs=400,
               provides=("base", "f2_repr"),
               train_args={"grad_clip": 5.0}),
    "F2p4": dict(label="F2+4", name="F2p4", which="last", train=True, epochs=400,
                 provides=("base", "foot_conv", "f2_repr"),
                 train_args={"grad_clip": 5.0}),
    "V3_2": dict(label="V3-2", name="V3_2", which="last", train=True, epochs=740,
                 provides=("base", "foot_conv", "pose_parts9"),
                 train_args={"lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_3A": dict(label="V3-3A", name="V3_3A", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts"),
                  train_args={"lambda_assign": 0.05, "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_3B": dict(label="V3-3B", name="V3_3B", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4a": dict(label="V3-4a", name="V3_4a", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9"),
                  train_args={"lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4b": dict(label="V3-4b", name="V3_4b", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts"),
                  train_args={"lambda_assign": 0.05, "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V3_4c": dict(label="V3-4c", name="V3_4c", which="last", train=True, epochs=740,
                  provides=("base", "foot_conv", "f2_repr", "pose_parts9", "soft_parts", "gate_sigma"),
                  train_args={"lambda_assign": 0.05, "lambda_sigma": 0.01,
                              "sigma_freeze_frac": 0.1,
                              "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
    "V4A": dict(label="V4A", name="V4A", which="last", train=True, epochs=740,
                provides=("base", "foot_conv", "pose_parts24", "soft_parts", "assign_cluster"),
                train_args={"assign_cluster": True, "lambda_assign": 1.0,
                            "lambda_assign_ent": 0.05, "lambda_assign_conc": 0.01,
                            "lambda_assign_dead": 0.02, "assign_dead_beta": 2.0,
                            "assign_temp_init": 1.0, "assign_temp_final": 0.2,
                            "assign_anneal_frac": 0.7, "assign_lock_frac": 0.7,
                            "assign_lr_mult": 5.0, "lr_warmup_frac": 0.05,
                            "grad_clip": 5.0, "loss_cap": 10.0}),
    "V4B": dict(label="V4B", name="V4B", which="best", train=True, epochs=740,
                provides=("base", "foot_conv", "pose_parts9", "learned_partition"),
                train_args={"part_json": _V4B_PART_JSON,
                            "lr_warmup_frac": 0.05, "grad_clip": 5.0}),
}


_COMMON_FALLBACK = {
    "CONFIG": "anysole/configs/v1.yaml",
    "RESULTS_ROOT": "results",
    "DISPLAY_ROOT": "results_display",
    "MODAL": "anysolev2",
    "CONTACT_METHOD": "joint_and",
    "DEVICE": "cuda",
    "TRAIN_SEED": "1",
    "EPOCHS": "800",
    "BATCH_SIZE": "256",
    "PROTOCOL_SEED": "0",
    "DISPLAY": "false",
}


def _load_defaults() -> dict[str, str]:
    values = dict(_COMMON_FALLBACK)
    path = CONFIG_DIR / "defaults.conf"
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            values[key] = value.split("#", 1)[0].strip()
    return values


COMMON = _load_defaults()
