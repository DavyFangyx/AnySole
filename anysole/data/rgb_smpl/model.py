"""Chunked neutral-SMPL forward and joint regression."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class SMPLForward:
    def __init__(self, model_path: str | Path, *, device: str = "cuda:0", batch_size: int = 256) -> None:
        import smplx
        import torch

        self.torch = torch
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = smplx.SMPL(model_path=str(self.model_path), gender="neutral").to(self.device).eval()
        self.joint_regressor = self.model.J_regressor[:24].detach().cpu().numpy().astype(np.float64)
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        self.joint_regressor_sha256 = hashlib.sha256(self.joint_regressor.tobytes()).hexdigest()

    def forward(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        translations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(poses, dtype=np.float32)
        betas = np.asarray(betas, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 72:
            raise ValueError(f"poses must be [T,72], got {poses.shape}")
        if betas.shape == (10,):
            betas = np.broadcast_to(betas, (poses.shape[0], 10)).copy()
        if betas.shape != (poses.shape[0], 10):
            raise ValueError(f"betas must be [T,10], got {betas.shape}")
        if translations is not None:
            translations = np.asarray(translations, dtype=np.float32)
            if translations.shape != (poses.shape[0], 3):
                raise ValueError(f"translations must be [T,3], got {translations.shape}")
        vertices: list[np.ndarray] = []
        joints: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, poses.shape[0], self.batch_size):
                end = min(poses.shape[0], start + self.batch_size)
                kwargs = {
                    "global_orient": self.torch.from_numpy(poses[start:end, :3]).to(self.device),
                    "body_pose": self.torch.from_numpy(poses[start:end, 3:]).to(self.device),
                    "betas": self.torch.from_numpy(betas[start:end]).to(self.device),
                }
                if translations is not None:
                    kwargs["transl"] = self.torch.from_numpy(translations[start:end]).to(self.device)
                output = self.model(**kwargs)
                vertices.append(output.vertices.detach().cpu().numpy().astype(np.float32))
                joints.append(output.joints[:, :24].detach().cpu().numpy().astype(np.float32))
        return np.concatenate(vertices), np.concatenate(joints)

    def tpose(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        betas = np.asarray(betas, dtype=np.float32)
        if betas.ndim == 1:
            betas = betas[None]
        poses = np.zeros((betas.shape[0], 72), dtype=np.float32)
        return self.forward(poses, betas)

    def regress_joints(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
            raise ValueError(f"vertices must be [T,6890,3], got {vertices.shape}")
        return np.einsum("jv,tvc->tjc", self.joint_regressor, vertices)
