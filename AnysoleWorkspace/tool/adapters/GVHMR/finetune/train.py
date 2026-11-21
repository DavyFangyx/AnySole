"""Fold-wise fine-tuning of the native GVHMR motion network.

The visual front end is not part of GVHMR's motion network; per-session HMR2
features and ViTPose keypoints are cached by :mod:`prepare`.  This script only
updates the released relative-transformer pipeline, using camera-space SMPL-X
labels built from the fold's *training* subjects.  Validation uses the fold's
held-out subject and never contributes gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from . import common


def _patch_body_model_path() -> None:
    """Point the official SMPL-X lite model at the licensed workspace archive."""
    from hmr4d.utils.body_model import smplx_lite

    from .. import config as _config

    if getattr(smplx_lite.SmplxLite.__init__, "_tactile_patched", False):
        return
    original = smplx_lite.SmplxLite.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("model_path", str(_config.DEFAULT_SMPLX_MODEL))
        return original(self, *args, **kwargs)

    init._tactile_patched = True
    smplx_lite.SmplxLite.__init__ = init


def _patch_short_clip_augmentation() -> None:
    """Disable the 30-90 frame leg-occlusion block only for clips shorter than 91.

    The upstream helper draws ``randint(0, L-90)`` and fails for L <= 90; the
    rest of the training augmentation is length-agnostic.
    """
    import torch
    import hmr4d.model.gvhmr.gvhmr_pl as pl_module

    original = pl_module.get_invisible_legs_mask
    if getattr(original, "_tactile_short_clip", False):
        return

    def helper(shape, s_mask=0.03):
        batch, length = shape
        if length <= 90:
            return torch.zeros(batch, length, 17, dtype=torch.bool)
        return original(shape, s_mask)

    helper._tactile_short_clip = True
    pl_module.get_invisible_legs_mask = helper


def _compose_pipeline():
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  (registers demo config group)
    from omegaconf import OmegaConf

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo")
    # Same training weights as hmr4d/configs/exp/gvhmr/mixed/mixed.yaml.
    cfg.pipeline.args.weights = OmegaConf.create(
        {
            "cr_j3d": 500.0,
            "transl_c": 1.0,
            "cr_verts": 500.0,
            "j2d": 1000.0,
            "verts2d": 1000.0,
            "transl_w": 1.0,
            "static_conf_bce": 1.0,
        }
    )
    cfg.pipeline.args.static_conf = OmegaConf.create({"vel_thr": 0.15})
    return cfg


def build_model(
    checkpoint: str | Path,
    *,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_epochs: int = 50,
):
    import torch

    _patch_body_model_path()
    _patch_short_clip_augmentation()
    cfg = _compose_pipeline()
    model = TactileGvhmrPL(
        pipeline=cfg.pipeline,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    )
    model.load_pretrained_model(str(Path(checkpoint).expanduser().resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("GVHMR fine-tuning requires CUDA")
    return model


class TactileGvhmrPL:
    """Factory shim so PL is imported only after CUDA_VISIBLE_DEVICES is set."""

    def __new__(cls, *args, **kwargs):
        from hmr4d.model.gvhmr.gvhmr_pl import GvhmrPL

        if not hasattr(cls, "_impl"):
            class _Impl(GvhmrPL):
                def __init__(
                    self,
                    pipeline,
                    learning_rate: float = 1e-5,
                    weight_decay: float = 0.01,
                    max_epochs: int = 50,
                ):
                    super().__init__(pipeline=pipeline, optimizer=None, scheduler_cfg=None)
                    self.learning_rate = float(learning_rate)
                    self.weight_decay = float(weight_decay)
                    self.max_epochs = int(max_epochs)

                def configure_optimizers(self):
                    import torch

                    parameters = [
                        parameter
                        for parameter in self.pipeline.parameters()
                        if parameter.requires_grad
                    ]
                    optimizer = torch.optim.AdamW(
                        parameters,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, self.max_epochs),
                        eta_min=self.learning_rate * 0.1,
                    )
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                    }

                def on_validation_epoch_start(self):
                    self._val_abs_sum = 0.0
                    self._val_rel_sum = 0.0
                    self._val_frames = 0

                def validation_step(self, batch, batch_idx):
                    metric = camera_mpjpe(self, batch)
                    self._val_abs_sum += metric["abs_mm"] * metric["frames"]
                    self._val_rel_sum += metric["rel_mm"] * metric["frames"]
                    self._val_frames += metric["frames"]
                    return metric

                def on_validation_epoch_end(self):
                    frames = max(1, int(self._val_frames))
                    self.log(
                        "val/mpjpe_abs_mm",
                        self._val_abs_sum / frames,
                        prog_bar=True,
                        sync_dist=False,
                    )
                    self.log(
                        "val/mpjpe_rel_mm",
                        self._val_rel_sum / frames,
                        prog_bar=True,
                        sync_dist=False,
                    )
                    self.log("val/frames", float(self._val_frames), sync_dist=False)

            cls._impl = _Impl
        return cls._impl(*args, **kwargs)


def camera_mpjpe(model, batch, *, max_frames: int | None = None) -> dict[str, float]:
    """Absolute and hip-root-relative camera-space MPJPE over valid frames."""
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
    length = int(batch["length"].max()) if torch.is_tensor(batch["length"]) else int(batch["length"])
    if max_frames is not None:
        length = min(length, max_frames)
        inputs = {
            key: (value[:, :length] if torch.is_tensor(value) and value.ndim >= 2 else value)
            for key, value in inputs.items()
        }
        inputs["length"] = torch.full_like(batch["length"], length)
        batch_params = {
            key: value[:, :length] for key, value in batch["smpl_params_c"].items()
        }
    else:
        batch_params = batch["smpl_params_c"]
    with torch.no_grad():
        outputs = model.pipeline.forward(inputs, train=False, postproc=False)
        pred = outputs["pred_smpl_params_incam"]
        pred_joints = model.pipeline.endecoder.fk_v2(**pred)
        gt_joints = model.pipeline.endecoder.fk_v2(**batch_params)
        length = pred_joints.shape[1]
        mask = batch["mask"]["valid"][:, :length]
        absolute = torch.linalg.vector_norm(pred_joints - gt_joints, dim=-1)
        pred_root = pred_joints[:, :, [1, 2], :].mean(dim=2, keepdim=True)
        gt_root = gt_joints[:, :, [1, 2], :].mean(dim=2, keepdim=True)
        relative = torch.linalg.vector_norm(
            (pred_joints - pred_root) - (gt_joints - gt_root), dim=-1
        )
        frame_weight = mask.float()
        return {
            "abs_mm": float(
                (absolute.mean(dim=-1) * frame_weight).sum() / frame_weight.sum() * 1000.0
            ),
            "rel_mm": float(
                (relative.mean(dim=-1) * frame_weight).sum() / frame_weight.sum() * 1000.0
            ),
            "frames": int(frame_weight.sum()),
        }


class _DataModule:
    def __new__(cls, *args, **kwargs):
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        from .dataset import collate_fn

        def seed_worker(worker_id: int) -> None:
            import numpy as np
            import torch

            info = torch.utils.data.get_worker_info()
            if info is not None:
                info.dataset.rng = np.random.default_rng(
                    (torch.initial_seed() + worker_id) % (2**32)
                )

        class Impl(pl.LightningDataModule):
            def __init__(self, train_dataset, val_dataset, batch_size: int, num_workers: int):
                super().__init__()
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
                self.batch_size = int(batch_size)
                self.num_workers = int(num_workers)

            def train_dataloader(self):
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=len(self.train_dataset) >= self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0,
                    collate_fn=collate_fn,
                    worker_init_fn=seed_worker,
                )

            def val_dataloader(self):
                return DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn,
                )

        return Impl(*args, **kwargs)


def build_datasets(
    support_root: str | Path,
    fold: str,
    rows: dict[str, dict],
    *,
    max_len: int = 120,
    min_len: int = 48,
    samples_per_session: int = 8,
    seed: int = 0,
    train_sessions: list[str] | None = None,
    val_sessions: list[str] | None = None,
):
    from .dataset import TactileMotionDataset

    pools = common.fold_sessions(fold)
    train = TactileMotionDataset(
        support_root,
        rows,
        train_sessions if train_sessions is not None else pools["train"],
        max_len=max_len,
        min_len=min_len,
        samples_per_session=samples_per_session,
        training=True,
        seed=seed,
    )
    val = TactileMotionDataset(
        support_root,
        rows,
        val_sessions if val_sessions is not None else pools["val"],
        max_len=max_len,
        min_len=min_len,
        training=False,
        seed=seed,
    )
    return pools, train, val


def _to_device(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _logger(output: Path):
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(save_dir=str(output), name="tb")
    except Exception:  # noqa: BLE001 — tensorboard is optional for smoke runs
        return False


def run(args) -> int:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    pl.seed_everything(args.seed, workers=True)
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = common.load_session_manifest()
    pools = common.fold_sessions(args.fold)
    train_sessions = pools["train"][: args.max_train_sessions] if args.max_train_sessions else pools["train"]
    val_sessions = pools["val"][: args.max_val_sessions] if args.max_val_sessions else pools["val"]
    pools = {**pools, "train": train_sessions, "val": val_sessions}
    support_root = Path(args.support_root).expanduser().resolve()
    missing = [
        session
        for session in sorted(set(train_sessions) | set(val_sessions))
        if not (support_root / "labels" / f"{session}.npz").is_file()
        or not (support_root / "features" / f"{session}.pt").is_file()
    ]
    if missing:
        raise SystemExit(
            f"prepared support is incomplete under {support_root}; missing {len(missing)} session(s): "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    _, train_dataset, val_dataset = build_datasets(
        args.support_root,
        args.fold,
        rows,
        max_len=args.max_len,
        min_len=args.min_len,
        samples_per_session=args.samples_per_session,
        seed=args.seed,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
    )
    if len(train_dataset) == 0:
        raise SystemExit(
            f"fold {args.fold}: no training session has a contiguous run >= {args.min_len} frames"
        )
    if len(val_dataset) == 0:
        raise SystemExit(f"fold {args.fold}: no validation window")
    coverage = {
        "fold": args.fold,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_step_s": 0.05,
        "train_sessions_available": len(pools["train"]),
        "train_sessions_usable": len(train_dataset.sessions),
        "train_items": len(train_dataset),
        "train_skipped": train_dataset.skipped,
        "val_sessions": len(val_dataset.sessions),
        "val_windows": len(val_dataset),
    }
    print(json.dumps(coverage, indent=2, sort_keys=True))
    if args.preflight:
        common.json_dump(output / "coverage.json", coverage)
        return 0
    model = build_model(
        args.release_ckpt,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
    )
    datamodule = _DataModule(train_dataset, val_dataset, args.batch_size, args.num_workers)
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        filename="{epoch:03d}",
        monitor="val/mpjpe_abs_mm",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=max(1, args.val_every),
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0],
        precision=args.precision,
        gradient_clip_val=0.5,
        log_every_n_steps=10,
        check_val_every_n_epoch=max(1, args.val_every),
        num_sanity_val_steps=0,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=_logger(output),
        enable_model_summary=False,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
    )
    metadata = {
        "fold": args.fold,
        "support_root": str(Path(args.support_root).expanduser().resolve()),
        "release_checkpoint": str(Path(args.release_ckpt).expanduser().resolve()),
        "release_checkpoint_sha256": common.sha256_file(args.release_ckpt),
        "train_sessions": pools["train"],
        "val_sessions": pools["val"],
        "train_items": len(train_dataset),
        "val_windows": len(val_dataset),
        "skipped_train_sessions": train_dataset.skipped,
        "label_conversion": "SMPL GT -> official smplx2smpl correspondence -> SMPL-X refit",
        "camera_labels": "physical cam3 metres; no per-date native_to_mocap translation",
        "supervision_scope": (
            "in-camera only; global trajectory and static-conf losses are masked "
            "(spv_incam_only=True), matching the camera-branch evaluation"
        ),
        "intrinsics": "calibration cam3 K (distortion ignored), shared by train/val",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_step_s": 0.05,
        "samples_per_session": args.samples_per_session,
        "epochs": args.epochs,
        "precision": args.precision,
        "seed": args.seed,
    }
    common.json_dump(output / "run.json", metadata)
    trainer.fit(model, datamodule=datamodule)
    best = {
        "best_model_path": checkpoint.best_model_path,
        "best_model_score": (
            float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None
        ),
        "last_model_path": checkpoint.last_model_path,
    }
    common.json_dump(output / "best.json", best)
    print(json.dumps(best, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", required=True, choices=sorted(common.FOLD_SPLITS))
    parser.add_argument("--support-root", default=str(common.DEFAULT_OUTPUT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--release-ckpt", default="")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-len", type=int, default=48)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--samples-per-session", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-train-sessions", type=int, default=0)
    parser.add_argument("--max-val-sessions", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=float, default=0.0)
    parser.add_argument("--limit-val-batches", type=float, default=0.0)
    parser.add_argument("--preflight", action="store_true",
                        help="report fold coverage/skipped sessions and exit")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = common.check_gpu(args.gpu)
    from .. import config as _config

    if not args.release_ckpt:
        args.release_ckpt = str(_config.DEFAULT_CHECKPOINT)
    if not args.output_root:
        args.output_root = str(common.DEFAULT_OUTPUT / "runs" / args.fold)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
