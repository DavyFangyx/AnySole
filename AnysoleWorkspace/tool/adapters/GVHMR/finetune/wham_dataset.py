"""BEDLAMDataset-compatible view over the prepared WHAM fold labels.

The native dataset is reused for its window construction, SMPL augmentation,
forward model and batch contract.  Only two pieces change:

* camera loading uses the physical calibration K/image size (not naive FOV);
* frame observations use the real frontend boxes/keypoints and WHAM HMR2a
  features, rather than re-projecting GT 3D keypoints.

Run in the WHAM environment (import happens after ``wham_prepare`` put the
upstream repository on ``sys.path``).
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import os
import sys
from pathlib import Path

import numpy as np
import torch

from .wham_env import activate

WHAM_ROOT = activate()

from configs import constants as _C  # noqa: E402
from lib.data.datasets.bedlam import BEDLAMDataset  # noqa: E402
from lib.utils import transforms  # noqa: E402


class TactileWHAMDataset(BEDLAMDataset):
    def __init__(self, cfg, labels_path: str | Path, validation: bool = False):
        self.validation = bool(validation)
        path = Path(labels_path).expanduser().resolve()
        original = _C.PATHS.BEDLAM_LABEL
        _C.PATHS.BEDLAM_LABEL = str(path)
        try:
            super().__init__(cfg)
        finally:
            _C.PATHS.BEDLAM_LABEL = original

    def load_params(self, index, target):
        target = super().load_params(index, target)
        start, _end = self.video_indices[index]
        if "visual_init_pose" in self.labels:
            target["_visual_init_pose"] = self.labels["visual_init_pose"][start].clone()
        if "visual_init_betas" in self.labels:
            target["_visual_init_betas"] = self.labels["visual_init_betas"][start].clone()
        return target

    def augment_data(self, target):
        target = self.SMPLAugmentor(target)
        if self.validation and "_visual_init_pose" in target:
            # Deterministic validation uses the cached HMR2a front-end init,
            # not GT first-frame pose plus random noise.
            target["init_pose"] = target["_visual_init_pose"][None].clone()
            if "_visual_init_betas" in target:
                target["betas"] = target["betas"].clone()
                target["betas"][0] = target["_visual_init_betas"]
        target = self.forward_smpl(target)
        return target

    def load_camera(self, index, target):
        start, end = self.video_indices[index]
        extrinsics = torch.as_tensor(self.labels["extrinsics"][start : end + 1]).float()
        R = extrinsics[:, :3, :3]
        self.cam_intrinsics = torch.as_tensor(
            self.labels["intrinsics"][start : start + 1]
        ).float()
        target["R"] = R
        target["res"] = torch.as_tensor(self.labels["res"][start]).float()
        target["K"] = self.cam_intrinsics[0]
        target["cam_intrinsics"] = self.cam_intrinsics
        cam_angvel = transforms.matrix_to_rotation_6d(
            R[:-1] @ R[1:].transpose(-1, -2)
        )
        cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)
        target["cam_angvel"] = cam_angvel * 3e1
        return target

    def get_inputs(self, index, target, vis_thr=0.3):
        start, end = self.video_indices[index]
        keypoints = torch.as_tensor(self.labels["kp2d"][start : end + 1]).float()
        boxes = torch.as_tensor(self.labels["bbox"][start : end + 1]).float()
        mask = keypoints[..., -1] < vis_thr
        normalized, _ = self.keypoints_normalizer(
            keypoints[..., :-1].clone(),
            target["res"],
            self.cam_intrinsics,
            224,
            224,
            boxes,
        )
        target["bbox"] = boxes[1:]
        target["kp2d"] = normalized
        target["mask"] = mask[1:]
        target["features"] = torch.as_tensor(
            self.labels["features"][start + 1 : end + 1]
        ).float()
        return target


def build_dataset(cfg, labels_path: str | Path, validation: bool = False) -> TactileWHAMDataset:
    return TactileWHAMDataset(cfg, labels_path, validation=validation)
