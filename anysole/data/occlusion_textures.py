"""CC0 texture assets shared by the Q14 texture-occlusion recipes.

Strip occlusion (extract_hrnet.occlude_lower_strip fill="texture:<asset>")
and the scene column (make_scene_occ_cache per-face mapping) both draw from
``AnysoleWorkspace/shared/assets/occlusion_textures``.  The tiled canvas is a
pure function of (asset, H, W, tile_px) so every level and every pipeline
(baselines, MotionPRO) sees the identical texture layout.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from anysole.types import WORKSPACE_ROOT

TEXTURE_ROOT = WORKSPACE_ROOT / "shared" / "assets" / "occlusion_textures"

TEXTURE_FILES = {
    "wood": "wood_cabinet_worn_long_diff_2k.jpg",
    "fabric": "fabric_pattern_07_col_1_2k.jpg",
    "crate": "raw_plank_wall_diff_2k.jpg",
    "concrete": "concrete_wall_003_diff_2k.jpg",
}

# One texture repeat spans this many pixels on the shorter side; constant
# across sessions and levels so the surface has a fixed apparent scale.
TILE_PX = 768


def parse_fill(fill: str) -> tuple[str, str | None]:
    """'black' | 'blur' | 'texture:<asset>' -> (kind, asset)."""
    value = (fill or "black").strip().lower()
    if value in ("black", "blur"):
        return value, None
    if value.startswith("texture:"):
        asset = value.split(":", 1)[1]
        if asset not in TEXTURE_FILES:
            raise ValueError(
                "unknown texture asset %r (choose from %s)"
                % (asset, ", ".join(sorted(TEXTURE_FILES)))
            )
        return "texture", asset
    raise ValueError("bad occlude fill %r: use black | blur | texture:<asset>" % fill)


@lru_cache(maxsize=None)
def _load_rgb(asset: str) -> np.ndarray:
    path = TEXTURE_ROOT / TEXTURE_FILES[asset]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("missing occlusion texture %s" % path)
    rgb = bgr[:, :, ::-1]
    rgb.setflags(write=False)
    return rgb


def load_texture_rgb(asset: str) -> np.ndarray:
    """Full-resolution texture (read-only, cached)."""
    return _load_rgb(asset)


def tile_texture(asset: str, height: int, width: int) -> np.ndarray:
    """Periodic tiling of the asset to exactly (height, width)."""
    img = _load_rgb(asset)
    reps_h = math.ceil(height / img.shape[0])
    reps_w = math.ceil(width / img.shape[1])
    out = np.tile(img, (reps_h, reps_w, 1))[:height, :width]
    out.setflags(write=False)
    return out


@lru_cache(maxsize=32)
def texture_canvas(asset: str, height: int, width: int,
                   tile_px: int = TILE_PX) -> np.ndarray:
    """Deterministic full-frame canvas: tile the asset (aspect preserved,
    shorter side = tile_px) and crop to (height, width)."""
    img = _load_rgb(asset)
    scale = tile_px / float(min(img.shape[:2]))
    if scale != 1.0:
        img = cv2.resize(img, (int(round(img.shape[1] * scale)),
                               int(round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    reps_h = math.ceil(height / img.shape[0])
    reps_w = math.ceil(width / img.shape[1])
    canvas = np.tile(img, (reps_h, reps_w, 1))[:height, :width]
    canvas.setflags(write=False)
    return canvas
