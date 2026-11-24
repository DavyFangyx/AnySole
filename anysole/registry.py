"""Model registry + centralized address inference for the AnySole pipeline.

Current public surface (2026-11-24, cleanup batch 3): the main-model family
V8_IMU_* (Residual + training ablations + Router/TGate/DualGate), the
no-IMU ablation group V8_Cv5, and the V7 channel representative V7_Cv7.
The V7 implementation chain (v7_a/v7_c/v7_c2/v7_c2sd2 + common/model_v7,
model_v4_1, model_v2) is kept as the V8 implementation dependency; it is
not registered as parallel experiments.

Variant rule (fixed, needs no registration): a variant dir (``tw40``,
``t0003_tw40_lr3e4``, ...) is just a hyperparameter subdir under the base
model dir — same entry, same from-scratch rule, different address.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Optional

from anysole.types import anysole_model_dir


@dataclass(frozen=True)
class ModelEntry:
    name: str
    module: str  # entry script under anysole/models/ (e.g. "v3_4b")
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # V7 通道代表（另一通道选择，2026-11-24 暂存）。
    "V7_Cv7": ModelEntry(
        "V7_Cv7", "v7_a",
        "无门控 A + 枚举三条件训练；另一通道（T2M）代表"),
    # V8_Cv5 = 主模型的无 IMU 消融组（用户裁定保留）。
    "V8_Cv5": ModelEntry("V8_Cv5", "v8_cv5", "V8 基座（双门+枚举+SD2+锚头，无 IMU 消融组）"),
    # ------------------------------------------------------------------
    # V8_IMU_* canonical naming (2026-11-15).  Main model = residual
    # fusion (gV=gT=1) with enum pretraining + occlusion-aware fine-tune
    # (20 ep, best-val).
    # ------------------------------------------------------------------
    "V8_IMU_Residual": ModelEntry(
        "V8_IMU_Residual", "v8_cv7_physimu",
        "Residual fusion (gV=gT=1), enum pretrain + occlusion-aware FT "
        "(20 ep, best-val); main model"),
    "V8_IMU_Residual_woAdapt": ModelEntry(
        "V8_IMU_Residual_woAdapt", "v8_cv7_physimu",
        "Residual fusion enum base (740 ep ckpt_last) without adaptation; "
        "w/o adaptation ablation and start point for all FT arms"),
    "V8_IMU_Router": ModelEntry(
        "V8_IMU_Router", "v8_cv7_physimu_router",
        "Competitive routing, router-only on the frozen residual base; "
        "main-text fusion comparison"),
    "V8_IMU_Router_Full": ModelEntry(
        "V8_IMU_Router_Full", "v8_cv7_physimu_router",
        "Competitive routing with full joint fine-tune; supplementary"),
    "V8_IMU_TGate": ModelEntry(
        "V8_IMU_TGate", "v8_cv5_physimu",
        "Sensing-branch gating: gV=1, gT=sigma(s_T) (Cv5 dual T gate + PhysIMU)"),
    "V8_IMU_DualGate": ModelEntry(
        "V8_IMU_DualGate", "v8_cv7_physimu_gate",
        "Independent dual-branch gating: gV=sigma(s_V), gT=sigma(s_T); "
        "post-stage gate on the frozen base"),
    "V8_IMU_Base_woEnum": ModelEntry(
        "V8_IMU_Base_woEnum", "v8_cv7_physimu",
        "Enum base without V/T/VT pretraining (VT-only); start point for "
        "the no-enum FT ablation"),
    "V8_IMU_Residual_woEnum": ModelEntry(
        "V8_IMU_Residual_woEnum", "v8_cv7_physimu",
        "Residual fusion + enum FT from the no-enum base (training ablation)"),
    "V8_IMU_Residual_DecoderOnly": ModelEntry(
        "V8_IMU_Residual_DecoderOnly", "v8_cv7_physimu",
        "Residual fusion, fine-tune pose_head only (training ablation)"),
    "V8_IMU_Residual_CleanOnly": ModelEntry(
        "V8_IMU_Residual_CleanOnly", "v8_cv7_physimu",
        "Residual fusion, clean-only fine-tune (training ablation)"),
}

# V8_Cv5 无 IMU 消融组 4 折（woS13 四折，与主模型同折表）。
for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"):
    _name = "V8_Cv5_%s" % _fold
    MODEL_REGISTRY[_name] = ModelEntry(
        _name, "v8_cv5", "V8_Cv5 折 %s（woS13 四折，无 IMU 消融组）" % _fold)

# V8_IMU_* canonical four-fold cells (2026-11-15).
for _fam, _mod in (("V8_IMU_Residual", "v8_cv7_physimu"),
                   ("V8_IMU_Residual_woAdapt", "v8_cv7_physimu"),
                   ("V8_IMU_TGate", "v8_cv5_physimu"),
                   ("V8_IMU_DualGate", "v8_cv7_physimu_gate")):
    for _fold in ("fS5S6S11", "fS5S11S12", "fS6S7", "fS5S6S8"):
        _name = "%s_%s" % (_fam, _fold)
        MODEL_REGISTRY[_name] = ModelEntry(
            _name, _mod, "%s 折 %s（V8_IMU 四折 ckpt）" % (_fam, _fold))
# Single-fold pilots (2026-11-15, fS5S6S11 only).
for _fam in ("V8_IMU_Residual_woEnum", "V8_IMU_Residual_DecoderOnly",
             "V8_IMU_Residual_CleanOnly"):
    _name = _fam + "_fS5S6S11"
    MODEL_REGISTRY[_name] = ModelEntry(
        _name, "v8_cv7_physimu", "%s 单折试点 fS5S6S11（2026-11-15）" % _fam)


def list_registered_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_model_entry(model_name: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown model %r. Expected one of: %s" % (model_name, sorted(MODEL_REGISTRY))
        ) from exc


def get_builder(model_name: str):
    """Import and return the entry module (STRUCTURE / CONFIG_EXTRA / build)
    that defines ``model_name``'s architecture."""
    entry = get_model_entry(model_name)
    return import_module("anysole.models.%s" % entry.module)


def infer_anysole_paths(
    model_name: str,
    *,
    variant: Optional[str] = None,
    contact: str = "joint_and",
    which: str = "last",
) -> dict:
    """Central address inference (the only place real paths are built).

    ``model_name``: registry identifier (e.g. V8_IMU_Residual).  ``variant``:
    stacked hyperparameter fields under the model dir (tw40 / t0003_tw40_lr3e4),
    built automatically by train from its effective config and passed
    explicitly by eval/infer/ridge.  ``contact``: contact-label scheme in the
    dir name.  ``which``: "last" | "best" — which checkpoint to address.
    Default "last": ckpt_last is written at the end of every run, while
    ckpt_best is the val-metric-best INTERMEDIATE and can be several epochs
    older — use last for the current model, switch to best via --which best
    when producing a final report.
    """
    if which not in ("last", "best"):
        raise ValueError("which must be 'last' or 'best', got %r" % which)
    get_model_entry(model_name)
    model_dir = anysole_model_dir(model_name, contact, variant)
    ckpt_name = "ckpt_last.pt" if which == "last" else "ckpt_best.pt"
    return {
        "model_dir": model_dir,
        "ckpt": model_dir / "checkpoints" / ckpt_name,
        "init_from": None,  # independence: never inferred (train --init-from = debug only)
        "predictions_dir": model_dir / "predictions" / "eval_motion",
        "metrics_dir": model_dir / "metrics",
    }
