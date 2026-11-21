"""Fold-wise WHAM motion-network fine-tuning on the Tactile four-fold split.

Run in the WHAM environment:

    python -m GVHMR.finetune.wham_train --fold fS5S6S11 \
        --labels-root work/gvhmr_wham --gpu 2

The visual front end is frozen because features are precomputed.  Only the
native WHAM motion network is updated; the camera branch is supervised
(pose/2D/3D/vertices), while global trajectory/contact losses remain masked by
the BEDLAM-style target contract (``has_traj=False`` / contact ``-1``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _early_gpu_env() -> None:
    """Set CUDA_VISIBLE_DEVICES before torch import; avoids a stray GPU0 context."""
    argv = sys.argv
    for index, value in enumerate(argv):
        if value == "--gpu" and index + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[index + 1]
        elif value.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = value.split("=", 1)[1]


_early_gpu_env()
from pathlib import Path

import numpy as np
import torch

from . import common

ROOT = common.ROOT
WORKSPACE = common.WORKSPACE
from .wham_env import activate  # noqa: E402

WHAM_ROOT = activate()

from configs.config import get_cfg_defaults  # noqa: E402
from lib.core.loss import WHAMLoss  # noqa: E402
from lib.models import build_body_model, build_network  # noqa: E402
from lib.utils.data_utils import make_collate_fn  # noqa: E402
from lib.utils.utils import prepare_batch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .wham_dataset import build_dataset  # noqa: E402

DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "assets/third_party/visual_baselines/weights/third_party/WHAM/checkpoints"
    / "wham_vit_bedlam_w_3dpw.pth.tar"
)
DEFAULT_OUTPUT = WORKSPACE / "work" / "gvhmr_wham"


def _cfg(batch_size: int) -> object:
    cfg = get_cfg_defaults()
    cfg.DEVICE = "cuda"
    cfg.MODEL.BACKBONE = "vit"
    cfg.MODEL_CONFIG = str(WHAM_ROOT / "configs/yamls/model_base.yaml")
    cfg.DATASET.SEQLEN = 81
    cfg.TRAIN.STAGE = "stage2"
    cfg.TRAIN.CHECKPOINT = ""
    cfg.TRAIN.BATCH_SIZE = int(batch_size)
    cfg.NUM_WORKERS = 0
    cfg.LOSS.SHAPE_LOSS_WEIGHT = 0.0
    cfg.LOSS.JOINT2D_LOSS_WEIGHT = 3.0
    cfg.LOSS.JOINT3D_LOSS_WEIGHT = 6.0
    cfg.LOSS.POSE_LOSS_WEIGHT = 1.0
    cfg.LOSS.CASCADED_LOSS_WEIGHT = 0.05
    cfg.LOSS.SLIDING_LOSS_WEIGHT = 0.5
    cfg.LOSS.CAMERA_LOSS_WEIGHT = 0.01
    cfg.LOSS.ROOT_VEL_LOSS_WEIGHT = 0.001
    cfg.LOSS.ROOT_POSE_LOSS_WEIGHT = 0.4
    cfg.LOSS.CONTACT_LOSS_WEIGHT = 0.04
    cfg.LOSS.VERTS3D_LOSS_WEIGHT = 1.0
    cfg.LOSS.LOSS_WEIGHT = 60.0
    cfg.LOSS.CAMERA_LOSS_SKIP_EPOCH = 0
    return cfg


def _load_full_checkpoint(network, path: str | Path) -> dict:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    ignore = {
        "smpl.body_pose",
        "smpl.betas",
        "smpl.global_orient",
        "smpl.J_regressor_extra",
        "smpl.J_regressor_eval",
    }
    target = network.state_dict()
    state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key not in ignore and key in target
    }
    missing, unexpected = network.load_state_dict(state, strict=False)
    integrator = [key for key in target if key.startswith("integrator.")]
    if not integrator:
        raise RuntimeError("network has no integrator parameters")
    missing_integrator = [key for key in integrator if key not in state]
    if missing_integrator:
        raise RuntimeError(f"checkpoint lacks integrator weights: {missing_integrator}")
    return {
        "checkpoint": str(Path(path).expanduser().resolve()),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_performance": float(checkpoint.get("performance", float("nan"))),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "integrator_keys_loaded": len(integrator),
    }


def _forward(network, batch, device):
    x, inits, features, aux, gt = prepare_batch(batch, device, True)
    pred = network(x, inits, features, **aux)
    return pred, gt


def _camera_mpjpe(network, pred, gt) -> dict:
    regressor = network.smpl.J_regressor_wham
    gt_verts = gt["verts"]
    pred_verts = pred["verts_cam"].reshape(gt_verts.shape)
    pred_j = torch.einsum("jv,bfvc->bfjc", regressor, pred_verts)[..., :17, :]
    gt_j = torch.einsum("jv,bfvc->bfjc", regressor, gt_verts)[..., :17, :]
    pred_root = pred_j[:, :, [11, 12], :].mean(dim=2, keepdim=True)
    gt_root = gt_j[:, :, [11, 12], :].mean(dim=2, keepdim=True)
    error = torch.linalg.vector_norm(
        (pred_j - pred_root) - (gt_j - gt_root), dim=-1
    )
    frames = int(error.shape[0] * error.shape[1])
    return {
        "mpjpe_mm": float(error.mean() * 1000.0),
        "frames": frames,
        "per_frame": error.detach().cpu(),
    }


def _evaluate(network, loader, device) -> dict:
    network.eval()
    total = 0.0
    frames = 0
    with torch.no_grad():
        for batch in loader:
            pred, gt = _forward(network, batch, device)
            metric = _camera_mpjpe(network, pred, gt)
            total += metric["mpjpe_mm"] * metric["frames"]
            frames += metric["frames"]
    return {"mpjpe_mm": total / max(1, frames), "frames": frames}


def _save(network, optimizer, epoch: int, performance: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": network.state_dict(),
            "performance": float(performance),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _state_diff(model, fresh) -> dict:
    left = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    right = {key: value.detach().cpu() for key, value in fresh.state_dict().items()}
    keys = sorted(set(left) & set(right))
    float_diff = 0.0
    int_diff = 0
    for key in keys:
        if left[key].shape != right[key].shape:
            return {"shape_mismatch": key}
        if torch.is_floating_point(left[key]):
            float_diff = max(float_diff, float((left[key] - right[key]).abs().max()))
        else:
            int_diff += int((left[key] != right[key]).sum())
    return {
        "keys_missing": sorted(set(right) - set(left)),
        "keys_unexpected": sorted(set(left) - set(right)),
        "max_float_abs_diff": float_diff,
        "int_value_mismatches": int_diff,
    }


def _label_ok(labels_root: Path, tag: str, support_root: Path) -> tuple[bool, dict]:
    metadata = json.loads((labels_root / "labels" / f"{tag}_metadata.json").read_text())
    sessions = sorted(set(metadata.get("train", {}).get("sessions", [])) |
                      set(metadata.get("val", {}).get("sessions", [])))
    result = {}
    ok = True
    for session in sessions:
        path = support_root / "labels" / f"{session}.json"
        if not path.is_file():
            result[session] = {"ok": False, "reason": "missing GVHMR label metadata"}
            ok = False
            continue
        meta = json.loads(path.read_text())
        fit = meta["fit"]
        check = meta["camera_forward_check"]
        projection = meta["projection_check"]
        session_ok = (
            float(fit["vertex_rmse_mm_max"]) <= 50.0
            and int(meta.get("gate_invalid_frames", 0)) == 0
            and max(check["smplx_vertex_max_mm"], check["smpl_vertex_max_mm"]) <= 0.1
            and 0.6 <= projection["projected_skeleton_over_bbox_height_median"] <= 1.3
        )
        result[session] = {"ok": session_ok}
        ok = ok and session_ok
    return ok, result


def _run_smoke(args, network, train_loader, val_loader, criterion, optimizer, output, labels, support):
    device = "cuda"
    batch = next(iter(train_loader))
    probe = network.integrator.layer1.weight.detach().clone()
    losses = []
    grad_norms = []
    network.train()
    for step in range(args.steps):
        x, inits, features, aux, gt = prepare_batch(batch, device, True)
        pred = network(x, inits, features, **aux)
        loss, _ = criterion(pred, gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))
        grad_norms.append(float(grad))
    parameter_delta = float((network.integrator.layer1.weight - probe).abs().max())
    network.eval()
    val = _evaluate(network, val_loader, device)
    checkpoint = output / "smoke_after.pth.tar"
    _save(network, optimizer, 0, val["mpjpe_mm"], checkpoint)
    fresh = build_network(_cfg(args.batch_size), build_body_model("cuda", args.batch_size * 81)).to(device)
    _load_full_checkpoint(fresh, args.checkpoint or DEFAULT_CHECKPOINT)
    _load_full_checkpoint(fresh, checkpoint)
    diff = _state_diff(network, fresh)
    fresh.eval()
    with torch.no_grad():
        pred_a = network(*prepare_batch(batch, device, True)[:3], **prepare_batch(batch, device, True)[3])["verts_cam"]
        pred_b = network(*prepare_batch(batch, device, True)[:3], **prepare_batch(batch, device, True)[3])["verts_cam"]
        pred_c = fresh(*prepare_batch(batch, device, True)[:3], **prepare_batch(batch, device, True)[3])["verts_cam"]
    repeat_diff = float((pred_a - pred_b).abs().max())
    reload_diff = float((pred_a - pred_c).abs().max())
    label_ok, label_stats = _label_ok(labels, args.tag or args.fold, support)
    first3 = float(np.median(losses[: min(3, len(losses))]))
    last3 = float(np.median(losses[-min(3, len(losses)):]))
    checks = {
        "label_ok": label_ok,
        "losses_finite": bool(np.isfinite(losses).all()),
        "loss_decreased": last3 < first3,
        "gradients_finite": bool(np.isfinite(grad_norms).all()),
        "parameter_updated": parameter_delta > 0.0,
        "checkpoint_complete": (
            not diff.get("keys_missing") and not diff.get("keys_unexpected")
            and "shape_mismatch" not in diff
            and diff.get("max_float_abs_diff", 1.0) < 1e-6
            and diff.get("int_value_mismatches", 1) == 0
        ),
        "inference_repeatable": repeat_diff < 1e-6,
        "reload_inference_consistent": reload_diff < 1e-6,
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "complete": status == "pass",
        "status": status,
        "checks": checks,
        "label_stats": label_stats,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first3_median": first3,
        "loss_last3_median": last3,
        "gradients_finite": bool(np.isfinite(grad_norms).all()),
        "grad_norm_first": grad_norms[0],
        "grad_norm_last": grad_norms[-1],
        "parameter_max_abs_delta": parameter_delta,
        "checkpoint_diff": diff,
        "inference_repeat_max_abs_diff": repeat_diff,
        "reload_inference_max_abs_diff": reload_diff,
        "val": val,
        "checkpoint": str(checkpoint),
    }
    common.json_dump(output / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


def run(args) -> int:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = _cfg(args.batch_size)
    smpl = build_body_model("cuda", args.batch_size * cfg.DATASET.SEQLEN)
    network = build_network(cfg, smpl).to(device)
    load_info = _load_full_checkpoint(network, args.checkpoint or DEFAULT_CHECKPOINT)
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = common.ROOT / path
        return path.resolve()

    output = _resolve(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    labels = _resolve(args.labels_root)
    train_path = labels / "labels" / f"{args.tag or args.fold}_train.pkl"
    val_path = labels / "labels" / f"{args.tag or args.fold}_val.pkl"
    train_dataset = build_dataset(cfg, train_path, validation=False)
    val_dataset = build_dataset(cfg, val_path, validation=True)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit(f"empty WHAM dataset: train={len(train_dataset)} val={len(val_dataset)}")

    integrator = [p for n, p in network.named_parameters() if n.startswith("integrator.")]
    others = [p for n, p in network.named_parameters() if not n.startswith("integrator.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": integrator, "lr": args.integrator_lr},
            {"params": others, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = WHAMLoss(cfg, "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(),
    )
    if args.steps:
        return _run_smoke(
            args, network, train_loader, val_loader, criterion, optimizer, output,
            labels, Path(args.support_root).expanduser().resolve()
            if Path(args.support_root).is_absolute()
            else (common.ROOT / args.support_root).resolve(),
        )
    metadata = {
        "fold": args.fold,
        "tag": args.tag or args.fold,
        "labels_root": str(labels),
        "checkpoint": load_info,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "integrator_lr": args.integrator_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "camera": "calibration K; physical camera SMPL labels",
        "supervision": "camera branch; has_traj=False; contact=-1",
        "seed": args.seed,
    }
    common.json_dump(output / "run.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))

    start = time.time()
    best = float("inf")
    losses = []
    steps = 0
    for epoch in range(args.epochs):
        network.train()
        epoch_losses = []
        for batch in train_loader:
            x, inits, features, aux, gt = prepare_batch(batch, "cuda", True)
            pred = network(x, inits, features, **aux)
            loss, loss_dict = criterion(pred, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss))
            losses.append(float(loss))
            steps += 1
            if args.steps and steps >= args.steps:
                break
        metric = _evaluate(network, val_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={np.mean(epoch_losses):.4f} "
            f"val_camera_mpjpe={metric['mpjpe_mm']:.3f}"
        )
        performance = metric["mpjpe_mm"]
        _save(network, optimizer, epoch, performance, output / "checkpoints" / "last.pth.tar")
        if performance < best:
            best = performance
            _save(network, optimizer, epoch, performance, output / "checkpoints" / "best.pth.tar")
        if args.steps and steps >= args.steps:
            break
        train_dataset.prepare_video_batch()
    best_path = output / "checkpoints" / "best.pth.tar"
    common.json_dump(
        output / "best.json",
        {
            "best_model_path": str(best_path) if best_path.is_file() else "",
            "best_val_camera_mpjpe_mm": best,
            "last_model_path": str(output / "checkpoints" / "last.pth.tar"),
        },
    )
    summary = {
        "complete": True,
        "status": "pass" if steps > 0 and np.isfinite(losses).all() else "fail",
        "steps": steps,
        "elapsed_s": time.time() - start,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "best_val_camera_mpjpe_mm": best,
        "checkpoints": str(output / "checkpoints"),
    }
    common.json_dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", required=True, choices=sorted(common.FOLD_SPLITS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--labels-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=0, help="smoke mode: N steps on one fixed batch")
    parser.add_argument("--support-root", default="",
                        help="GVHMR label support root used by the smoke label check")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--integrator-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = common.check_gpu(args.gpu)
    if not args.output_root:
        args.output_root = str(DEFAULT_OUTPUT / "runs" / (args.tag or args.fold))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
