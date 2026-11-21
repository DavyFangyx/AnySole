"""End-to-end smoke test for the GVHMR fine-tune path.

Checks one real training session and one real validation session:

* SMPL -> SMPL-X label conversion and camera placement
* feature extraction and dataset contract
* forward / backward / optimizer step / loss decrease on a fixed batch
* checkpoint save and reload into a fresh model
* camera-space validation metric before and after the smoke updates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import common


def _ensure_support(root: Path, rows: dict, sessions: list[str], args, cam: int = 3) -> dict:
    """Prepare labels/features only for sessions whose artifacts are absent."""
    from . import labels as label_module
    from . import prepare as prepare_module

    os.chdir(common.ROOT / "Baselines" / "GVHMR")
    extractors = None
    status = {}
    try:
        for session in sessions:
            row = rows[session]
            date, _subject = common.session_date_subject(row)
            unit = common.load_unit(date, session, cam)
            label_path = root / "labels" / f"{session}.npz"
            feature_path = root / "features" / f"{session}.pt"
            state = {}
            if not label_path.is_file() or args.force:
                label_module.write_labels(
                    root,
                    session,
                    *label_module.prepare_session_labels(
                        row,
                        unit,
                        cam=cam,
                        fit_steps=args.fit_steps,
                        fit_batch_size=args.fit_batch_size,
                        fit_learning_rate=args.fit_learning_rate,
                        max_vertex_rmse_mm=args.max_vertex_rmse_mm,
                    ),
                )
                state["labels"] = "ok"
            else:
                state["labels"] = "skip"
            if not feature_path.is_file() or args.force:
                if extractors is None:
                    extractors = prepare_module._Extractors()
                state["features"] = prepare_module.prepare_features(
                    root, row, unit, extractors, args, cam
                )
            else:
                state["features"] = "skip"
            status[session] = state
    finally:
        os.chdir(common.ROOT)
    return status


def _aggregate_metrics(model, dataset, device) -> dict[str, float]:
    from torch.utils.data import DataLoader

    from .dataset import collate_fn
    from .train import _to_device, camera_mpjpe

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    abs_values = []
    rel_values = []
    frames = []
    for batch in loader:
        batch = _to_device(batch, device)
        metric = camera_mpjpe(model, batch)
        abs_values.append(metric["abs_mm"] * metric["frames"])
        rel_values.append(metric["rel_mm"] * metric["frames"])
        frames.append(metric["frames"])
    total = max(1, sum(frames))
    return {
        "mpjpe_abs_mm": float(sum(abs_values) / total),
        "mpjpe_rel_mm": float(sum(rel_values) / total),
        "frames": int(total),
    }


def _label_vitpose_error(model, root: Path, session: str, device) -> dict:
    """Project camera SMPL-X labels and compare with the cached ViTPose keypoints."""
    import torch

    with np.load(root / "labels" / f"{session}.npz", allow_pickle=False) as archive:
        labels = {key: np.asarray(archive[key]) for key in archive.files}
    features = torch.load(root / "features" / f"{session}.pt", map_location="cpu", weights_only=False)
    valid = np.asarray(labels["valid"], dtype=bool)
    index = np.flatnonzero(valid)
    with torch.no_grad():
        _, joints = model.smplx(
            body_pose=torch.as_tensor(labels["smplx_body_pose"][valid], dtype=torch.float32, device=device),
            betas=torch.as_tensor(labels["smplx_betas"][valid], dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(labels["smplx_global_orient"][valid], dtype=torch.float32, device=device),
            transl=torch.as_tensor(labels["smplx_transl"][valid], dtype=torch.float32, device=device),
        )
    K = features["K_fullimg"][index].float().to(device)
    keypoints = features["kp2d"][index].float().to(device)
    pixel = torch.stack(
        [
            joints[..., 0] / joints[..., 2] * K[:, 0, 0, None] + K[:, 0, 2, None],
            joints[..., 1] / joints[..., 2] * K[:, 1, 1, None] + K[:, 1, 2, None],
        ],
        dim=-1,
    )
    error = torch.linalg.vector_norm(pixel - keypoints[..., :2], dim=-1).cpu().numpy()
    return {
        "frames": int(len(index)),
        "mean_px": float(error.mean()),
        "median_px": float(np.median(error)),
        "p90_px": float(np.percentile(error, 90)),
    }


def _predict_params(model, batch) -> dict:
    import torch
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
    obs[~batch["mask"]["valid"]] = 0
    inputs = {
        "length": batch["length"],
        "obs": obs,
        "bbx_xys": batch["bbx_xys"],
        "K_fullimg": batch["K_fullimg"],
        "cam_angvel": batch["cam_angvel"],
        "f_imgseq": batch["f_imgseq"],
    }
    with torch.no_grad():
        outputs = model.pipeline.forward(inputs, train=False, postproc=False)
    return {key: value.detach() for key, value in outputs["pred_smpl_params_incam"].items()}


def _max_param_diff(left: dict, right: dict) -> float:
    import torch

    values = []
    for key in left:
        if key in right:
            values.append(float((left[key] - right[key]).abs().max()))
    return max(values) if values else 0.0


def _state_diff(model, fresh) -> dict:
    import torch

    left = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    right = {key: value.detach().cpu() for key, value in fresh.state_dict().items()}
    common_keys = sorted(set(left) & set(right))
    float_diff = 0.0
    int_diff = 0
    shape_mismatch = []
    for key in common_keys:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            shape_mismatch.append(key)
            continue
        if torch.is_floating_point(a):
            float_diff = max(float_diff, float((a.float() - b.float()).abs().max()))
        else:
            int_diff += int((a != b).sum())
    return {
        "keys_missing": sorted(set(right) - set(left)),
        "keys_unexpected": sorted(set(left) - set(right)),
        "shape_mismatch": shape_mismatch,
        "max_float_abs_diff": float_diff,
        "int_value_mismatches": int_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-session", default="S10101")
    parser.add_argument("--val-session", default="S12011")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--support-root", default=str(common.DEFAULT_OUTPUT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-batch-size", type=int, default=24)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--max-vertex-rmse-mm", type=float, default=50.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = common.check_gpu(args.gpu)
    root = Path(args.support_root).expanduser().resolve()
    output = Path(args.output_root or (root / "smoke")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = common.load_session_manifest()
    missing = [name for name in (args.train_session, args.val_session) if name not in rows]
    if missing:
        raise SystemExit(f"sessions absent from the manifest: {missing}")
    prepare_status = _ensure_support(
        root, rows, [args.train_session, args.val_session], args
    )
    print(json.dumps({"prepare": prepare_status}, indent=2), flush=True)

    import torch
    import pytorch_lightning as pl

    pl.seed_everything(0, workers=True)
    from .dataset import TactileMotionDataset, collate_fn
    from .. import config as _config
    from .train import _compose_pipeline, _patch_body_model_path, camera_mpjpe

    _patch_body_model_path()
    cfg = _compose_pipeline()
    from hmr4d.model.gvhmr.gvhmr_pl import GvhmrPL

    class SmokeModel(GvhmrPL):
        def __init__(self):
            super().__init__(pipeline=cfg.pipeline, optimizer=None, scheduler_cfg=None)
            self.smoke_log = []
            self.log = lambda *args, **kwargs: None

        def validation_step(self, batch, batch_idx):
            return None

    model = SmokeModel()
    model.load_pretrained_model(str(_config.DEFAULT_CHECKPOINT))
    device = torch.device("cuda")
    model.train().to(device)
    train_dataset = TactileMotionDataset(
        root, rows, [args.train_session], max_len=120, min_len=48,
        samples_per_session=1, training=True, seed=0,
    )
    val_dataset = TactileMotionDataset(
        root, rows, [args.val_session], max_len=120, min_len=48,
        training=False, seed=0,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("smoke datasets are empty; check valid runs and prepared artifacts")
    from torch.utils.data import DataLoader

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    batch = {key: (value.to(device) if torch.is_tensor(value) else
                   {k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value)
             for key, value in batch.items()}
    model.eval()
    val_before = _aggregate_metrics(model, val_dataset, device)
    model.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    probe = model.pipeline.denoiser3d.embed_noisyobs.fc1.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    grad_norms = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_step(batch, step)
        loss = outputs["loss"]
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None], 0.5
        )
        optimizer.step()
        losses.append(float(loss))
        grad_norms.append(float(grad))
    parameter_delta = float((model.pipeline.denoiser3d.embed_noisyobs.fc1.weight - probe).abs().max())
    model.eval()
    val_after = _aggregate_metrics(model, val_dataset, device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.eval()
    repeat_a = _predict_params(model, batch)
    repeat_b = _predict_params(model, batch)
    inference_repeat_diff = _max_param_diff(repeat_a, repeat_b)

    checkpoint = output / "smoke_after.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    fresh = SmokeModel()
    fresh.load_pretrained_model(str(_config.DEFAULT_CHECKPOINT))
    fresh.load_pretrained_model(str(checkpoint))
    checkpoint_diff = _state_diff(model, fresh)
    fresh.eval().to(device)
    fresh_prediction = _predict_params(fresh, batch)
    saved_prediction = _predict_params(model, batch)
    reload_inference_diff = _max_param_diff(saved_prediction, fresh_prediction)

    label_stats = {}
    label_ok = True
    for session in (args.train_session, args.val_session):
        meta = json.loads((root / "labels" / f"{session}.json").read_text(encoding="utf-8"))
        fit = meta["fit"]
        check = meta["camera_forward_check"]
        projection = meta["projection_check"]
        session_ok = (
            float(fit["vertex_rmse_mm_max"]) <= 50.0
            and int(meta.get("gate_invalid_frames", 0)) == 0
            and max(check["smplx_vertex_max_mm"], check["smpl_vertex_max_mm"]) <= 0.1
            and 0.6
            <= projection["projected_skeleton_over_bbox_height_median"]
            <= 1.3
            and max(
                projection["bbox_center_offset_fraction_x_median"],
                projection["bbox_center_offset_fraction_y_median"],
            )
            <= 0.30
        )
        keypoint_check = _label_vitpose_error(model, root, session, device)
        session_ok = session_ok and keypoint_check["mean_px"] < 80.0
        label_ok = label_ok and session_ok
        label_stats[session] = {
            "vitpose_2d": keypoint_check,
            "valid_frames": meta["valid_frames"],
            "frame_count": meta["frame_count"],
            "vertex_rmse_mm_mean": fit["vertex_rmse_mm_mean"],
            "vertex_rmse_mm_p95": fit["vertex_rmse_mm_p95"],
            "vertex_rmse_mm_max": fit["vertex_rmse_mm_max"],
            "duplicate_frames": fit.get("duplicate_frames", 0),
            "refined_frames": fit.get("refined_frames", 0),
            "gate_invalid_frames": meta.get("gate_invalid_frames", 0),
            "projection_height_ratio_median": projection[
                "projected_skeleton_over_bbox_height_median"
            ],
            "camera_forward_max_mm": max(
                check["smplx_vertex_max_mm"], check["smpl_vertex_max_mm"]
            ),
            "ok": session_ok,
        }
    first3 = float(np.median(losses[: min(3, len(losses))]))
    last3 = float(np.median(losses[-min(3, len(losses)):]))
    loss_decreased = last3 < first3
    checks = {
        "label_ok": label_ok,
        "loss_decreased": loss_decreased,
        "losses_finite": bool(np.isfinite(losses).all()),
        "gradients_finite": bool(np.isfinite(grad_norms).all()),
        "parameter_updated": parameter_delta > 0.0,
        "checkpoint_complete": (
            not checkpoint_diff["keys_missing"]
            and not checkpoint_diff["keys_unexpected"]
            and not checkpoint_diff["shape_mismatch"]
            and checkpoint_diff["max_float_abs_diff"] < 1e-6
            and checkpoint_diff["int_value_mismatches"] == 0
        ),
        "inference_repeatable": inference_repeat_diff < 1e-6,
        "reload_inference_consistent": reload_inference_diff < 1e-6,
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "complete": status == "pass",
        "status": status,
        "checks": checks,
        "elapsed_s": time.time() - started,
        "gpu": args.gpu,
        "train_session": args.train_session,
        "val_session": args.val_session,
        "prepare": prepare_status,
        "label_stats": label_stats,
        "train_item": train_dataset.entries[0],
        "val_windows": len(val_dataset),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first3_median": first3,
        "loss_last3_median": last3,
        "grad_norm_first": grad_norms[0],
        "grad_norm_last": grad_norms[-1],
        "gradients_finite": bool(np.isfinite(grad_norms).all()),
        "parameter_max_abs_delta": parameter_delta,
        "checkpoint": str(checkpoint),
        "checkpoint_diff": checkpoint_diff,
        "inference_repeat_max_abs_diff": inference_repeat_diff,
        "reload_inference_max_abs_diff": reload_inference_diff,
        "val_before": val_before,
        "val_after": val_after,
    }
    common.json_dump(output / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
