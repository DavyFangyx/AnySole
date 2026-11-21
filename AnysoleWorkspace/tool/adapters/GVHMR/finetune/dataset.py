"""Torch datasets over prepared RGB features and camera-space SMPL-X labels.

Training uses only in-camera supervision (``spv_incam_only=True``), matching
the project's camera-branch GVHMR evaluation; global trajectory and static-conf
losses stay masked.  Clips never bridge a frame gap larger than 50 ms (two
40 Hz steps); larger gaps split the run, and sessions without a >= min_len
valid run are reported as skipped rather than stitched across the gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import common, labels as label_module

MAX_STEP_S = 0.05
from .prepare import load_features


@dataclass(frozen=True)
class Run:
    start: int
    end: int  # exclusive, in support-frame coordinates


class SessionSupport:
    """One session's labels, features and contiguous valid runs."""

    def __init__(self, root: Path, session: str, row: dict[str, Any]):
        self.session = session
        self.row = row
        self.labels, self.label_meta = label_module.load_labels(root, session)
        self.features = load_features(root, session)
        self._check()
        canonical = np.asarray(self.labels["canonical_index"], dtype=np.int64)
        allowed = np.zeros(len(canonical), dtype=bool)
        if "valid_frame_indices" in row:
            raw = row.get("valid_frame_indices", [])
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = []
            if raw:
                allowed[np.isin(canonical, np.asarray(raw, dtype=np.int64))] = True
        else:
            allowed[:] = True
        rejected = np.zeros(len(canonical), dtype=bool)
        raw = row.get("fake_frame_indices", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        if raw:
            rejected[np.isin(canonical, np.asarray(raw, dtype=np.int64))] = True
        self.valid = np.asarray(self.labels["valid"], dtype=bool) & allowed & ~rejected
        self.timestamps = np.asarray(self.labels["timestamp_seconds"], dtype=np.float64)
        self.runs = self._runs()

    def _check(self) -> None:
        label_index = np.asarray(self.labels["canonical_index"], dtype=np.int64)
        frame_index = np.asarray(self.features["canonical_index"], dtype=np.int64)
        if not np.array_equal(label_index, frame_index):
            raise common.FinetuneDataError(f"{self.session}: label/feature canonical grids differ")
        if not np.array_equal(
            np.asarray(self.labels["source_frame_number"], dtype=np.int64),
            np.asarray(self.features["source_frame_number"], dtype=np.int64),
        ):
            raise common.FinetuneDataError(f"{self.session}: label/feature source frames differ")
        count = len(label_index)
        if int(self.features["length"]) != count:
            raise common.FinetuneDataError(f"{self.session}: feature length drift")
        for key in ("smplx_body_pose", "smplx_global_orient", "smplx_transl", "smplx_betas"):
            if self.labels[key].shape[0] != count:
                raise common.FinetuneDataError(f"{self.session}: label {key} length drift")

    def _runs(self) -> list[Run]:
        runs: list[Run] = []
        start = None
        for index, good in enumerate(self.valid):
            continues = (
                start is not None
                and good
                and index > 0
                and self.timestamps[index] - self.timestamps[index - 1] <= MAX_STEP_S
            )
            if good and (start is None or not continues):
                if start is not None:
                    runs.append(Run(start, index))
                start = index
            elif not good and start is not None:
                runs.append(Run(start, index))
                start = None
        if start is not None:
            runs.append(Run(start, len(self.valid)))
        return runs


def _repeat_last(value, max_len: int):
    import torch

    if value.shape[0] == max_len:
        return value
    pad = value[-1:].expand(max_len - value.shape[0], *value.shape[1:])
    return torch.cat([value, pad], dim=0)


class TactileMotionDataset:
    """Random training clips or deterministic validation windows."""

    def __init__(
        self,
        root: str | Path,
        rows: dict[str, dict[str, Any]],
        sessions: Iterable[str],
        *,
        max_len: int = 120,
        min_len: int = 48,
        samples_per_session: int = 8,
        training: bool = True,
        seed: int = 0,
    ):
        if max_len < 60:
            raise ValueError("max_len must be at least 60")
        if min_len < 30 or min_len > max_len:
            raise ValueError("min_len must lie in [30, max_len]")
        self.root = Path(root).expanduser().resolve()
        self.max_len = int(max_len)
        self.min_len = int(min_len)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)
        self.sessions: list[str] = []
        self._support: dict[str, SessionSupport] = {}
        skipped = []
        for session in sorted(dict.fromkeys(sessions)):
            support = SessionSupport(self.root, session, rows[session])
            usable = [run for run in support.runs if run.end - run.start >= min_len]
            if training and not usable:
                skipped.append((session, [run.end - run.start for run in support.runs]))
                continue
            self.sessions.append(session)
            self._support[session] = support
        self.skipped = skipped
        self.entries: list[tuple[str, int]] = []
        self.windows: list[tuple[str, int, int]] = []
        if self.training:
            for session in self.sessions:
                support = self._support[session]
                usable = [
                    index for index, run in enumerate(support.runs)
                    if run.end - run.start >= min_len
                ]
                for sample in range(int(samples_per_session)):
                    self.entries.append((session, usable[sample % len(usable)]))
        else:
            for session in self.sessions:
                for run in self._support[session].runs:
                    start = run.start
                    while start < run.end:
                        length = min(self.max_len, run.end - start)
                        if length >= 2:
                            self.windows.append((session, start, length))
                        start += self.max_len

    def __len__(self) -> int:
        return len(self.entries) if self.training else len(self.windows)

    def _window(self, index: int) -> tuple[str, int, int]:
        session, run_index = self.entries[index]
        run = self._support[session].runs[run_index]
        available = run.end - run.start
        length = int(self.rng.integers(self.min_len, min(self.max_len, available) + 1))
        start = int(self.rng.integers(run.start, run.end - length + 1))
        return session, start, length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            session, start, length = self._window(index)
        else:
            session, start, length = self.windows[index]
        support = self._support[session]
        stop = start + length
        labels = support.labels
        features = support.features
        import torch

        def frame(value):
            return torch.as_tensor(np.asarray(value[start:stop]), dtype=torch.float32)

        smplx = {
            "body_pose": frame(labels["smplx_body_pose"]),
            "global_orient": frame(labels["smplx_global_orient"]),
            "transl": frame(labels["smplx_transl"]),
            "betas": frame(labels["smplx_betas"]),
        }
        padded = {key: _repeat_last(value, self.max_len) for key, value in smplx.items()}
        data: dict[str, Any] = {
            "length": length,
            "smpl_params_c": padded,
            "smpl_params_w": {key: torch.zeros_like(value) for key, value in padded.items()},
            "R_c2gv": torch.zeros(self.max_len, 3, 3),
            "gravity_vec": torch.zeros(3),
            "bbx_xys": _repeat_last(frame(features["bbx_xys"]), self.max_len),
            "K_fullimg": _repeat_last(frame(features["K_fullimg"]), self.max_len),
            "f_imgseq": _repeat_last(
                torch.as_tensor(np.asarray(features["f_imgseq"][start:stop]), dtype=torch.float32),
                self.max_len,
            ),
            "kp2d": _repeat_last(frame(features["kp2d"]), self.max_len),
            "cam_angvel": _repeat_last(frame(features["cam_angvel"]), self.max_len),
            "mask": {
                "valid": torch.cat([
                    torch.ones(length, dtype=torch.bool),
                    torch.zeros(self.max_len - length, dtype=torch.bool),
                ]),
                "vitpose": True,
                "bbx_xys": True,
                "f_imgseq": True,
                # Camera-branch fine-tuning: no global trajectory/static-conf loss.
                "spv_incam_only": True,
            },
            "meta": {
                "session_id": session,
                "unit_id": str(features["unit_id"]),
                "start": int(start),
                "end": int(stop),
                "canonical_start": int(labels["canonical_index"][start]),
                "canonical_end": int(labels["canonical_index"][stop - 1]),
                "length": int(length),
            },
        }
        return data


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    from torch.utils.data import default_collate

    result: dict[str, Any] = {}
    for key in batch[0]:
        if key.startswith("meta"):
            result[key] = [item[key] for item in batch]
        else:
            result[key] = default_collate([item[key] for item in batch])
    result["B"] = len(batch)
    return result
