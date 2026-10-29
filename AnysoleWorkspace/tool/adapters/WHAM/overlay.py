"""CPU-only original-size target SMPL-24 skeleton MP4, atomically published."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .runner import _atomic_write_json
from .manifest_index import file_sha256

PROJECTION = 'f=sqrt(width^2+height^2), principal_point=(width/2,height/2)'


def video_info(path, decode=True):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('video cannot be opened: ' + str(path))
    result = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  fps=float(cap.get(cv2.CAP_PROP_FPS)), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        if decode:
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (result['height'], result['width']):
                    raise ValueError('decoded video dimensions changed')
                count += 1
            if count != result['frames'] or count == 0:
                raise ValueError('video decode count mismatch')
    finally:
        cap.release()
    return result


def verify_video(path, expected):
    actual = video_info(path)
    if any(actual[k] != expected[k] for k in ('width', 'height', 'frames')) or abs(actual['fps'] - expected['fps']) > 0.001:
        raise ValueError('overlay/source geometry, FPS or frame count mismatch')
    return actual


def render(video, prediction, output_dir, smpl_model, ffmpeg_bin, config_hash, fail_at=None):
    import cv2
    import joblib
    import numpy as np
    from .visualize import load_smpl_geometry, _joint_positions, project_vertices, SMPL_PARENTS
    from .validation import validate_prediction
    cv2.setNumThreads(1)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    status = dest / 'status.json'
    started = time.time()
    record = dict(status='running', terminal=False, config_hash=config_hash,
                  owner_pid=os.getpid(), started_epoch=started)
    _atomic_write_json(status, record)
    partial = dest / f'.smpl_joints_overlay.{os.getpid()}.partial.mp4'
    encoder = None
    cap = None
    try:
        expected = video_info(video, decode=False)
        validate_prediction(prediction, expected['frames'])
        track = joblib.load(prediction)[0]
        _, regressor = load_smpl_geometry(smpl_model)
        command = [str(ffmpeg_bin), '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size',
                   f"{expected['width']}x{expected['height']}", '-framerate', str(expected['fps']),
                   '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', '-movflags', '+faststart', str(partial)]
        with (dest / 'encoder.log').open('ab') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            cap = cv2.VideoCapture(str(video))
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                joints = _joint_positions(regressor, track['verts'][count])
                if joints.shape != (24, 3):
                    raise ValueError('SMPL joint regressor must yield exactly 24 joints')
                pixels, valid = project_vertices(joints, expected['width'], expected['height'])
                points = np.rint(np.nan_to_num(pixels, nan=-10000)).astype(np.int32)
                color = (72, 220, 72)
                for child, parent in enumerate(SMPL_PARENTS):
                    if parent >= 0 and valid[child] and valid[parent]:
                        cv2.line(frame, tuple(points[parent]), tuple(points[child]), color, 3, cv2.LINE_AA)
                for point, present in zip(points, valid):
                    if present and 0 <= point[0] < expected['width'] and 0 <= point[1] < expected['height']:
                        cv2.circle(frame, tuple(point), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)
                encoder.stdin.write(frame.tobytes())
                count += 1
                if fail_at == 'encode' and count == 1:
                    raise RuntimeError('injected encoding failure')
            encoder.stdin.close()
            if encoder.wait() or count != expected['frames']:
                raise ValueError('encoder failed or input frames incomplete')
        verify_video(partial, expected)
        with partial.open('rb') as f:
            os.fsync(f.fileno())
        if fail_at == 'publish':
            raise RuntimeError('injected failure before atomic publish')
        final = dest / 'smpl_joints_overlay.mp4'
        os.replace(partial, final)
        metadata = dict(schema_version=1, input_video=str(video), input_pkl=str(prediction),
                        video_sha256=file_sha256(video), prediction_sha256=file_sha256(prediction),
                        output_sha256=file_sha256(final), projection_policy=PROJECTION,
                        track_id=0, joint_count=24, config_hash=config_hash,
                        encoding='H.264/libx264 CRF18 veryfast yuv420p',
                        video=expected, size_bytes=final.stat().st_size,
                        elapsed_seconds=time.time()-started)
        _atomic_write_json(dest / 'metadata.json', metadata)
        record.update(status='succeeded', terminal=True, finished_epoch=time.time(), output=str(final))
        _atomic_write_json(status, record)
        return metadata
    except Exception as exc:
        record.update(status='failed', terminal=True, finished_epoch=time.time(), error=str(exc))
        _atomic_write_json(status, record)
        raise
    finally:
        if cap is not None:
            cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'prediction', 'output-dir', 'smpl-model', 'ffmpeg-bin', 'config-hash'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv)
    print(json.dumps(render(a.video, a.prediction, a.output_dir, a.smpl_model, a.ffmpeg_bin, a.config_hash)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
