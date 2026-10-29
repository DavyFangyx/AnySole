"""Fail-closed saved-output checks shared by inference, resume and audit."""
from pathlib import Path
import joblib
import numpy as np


def prediction_path(root):
    return Path(root) / 'wham_output' / 'sequence' / 'wham_output.pkl'


def validate_prediction(path, frames):
    result = joblib.load(path)
    if not isinstance(result, dict) or set(result) != {0}:
        raise ValueError('target output must contain only track 0')
    track = result[0]
    ids = np.asarray(track['frame_ids'])
    if not np.array_equal(ids, np.arange(frames)):
        raise ValueError('target output frame_ids incomplete or duplicated')
    vertices = np.asarray(track['verts'])
    if vertices.shape != (frames, 6890, 3) or not np.isfinite(vertices).all():
        raise ValueError('invalid target vertices')
    return dict(track_id=0, frames=frames, size_bytes=Path(path).stat().st_size)


def validate_tracking(root, frames):
    result = joblib.load(Path(root) / 'wham_output' / 'sequence' / 'tracking_results.pth')
    if set(result) != {0} or not np.array_equal(result[0]['frame_id'], np.arange(frames)):
        raise ValueError('target tracking incomplete or contains fallback tracks')
    return True
