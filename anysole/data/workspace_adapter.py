"""AnySole's private adapter over the frozen public workspace contract.

This module is deliberately model-owned.  It is the only AnySole data code
which knows how a ``shared.session.v1`` artifact becomes AnySole inputs.
Shared facts remain read-only; labels and visual caches are written below the
AnySole-owned ``model_inputs`` tree.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from AnysoleWorkspace.tool.workspace import canonical_uri, resolve_uri
from anysole.types import VIEW_CACHE_SCHEMA, VIEW_MANIFEST_SCHEMA, VIEW_STATS_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "AnysoleWorkspace"
SHARED_SESSION_ROOT = WORKSPACE_ROOT / "shared/facts/sessions/cam3"
PROTOCOL_SPLIT = WORKSPACE_ROOT / "protocol/splits/default/splits.csv"
ADAPTER_VERSION = "adapter_v1"
ANYSOLE_INPUT_ROOT = WORKSPACE_ROOT / "model_inputs/AnySole" / ADAPTER_VERSION
LABEL_ROOT = ANYSOLE_INPUT_ROOT / "labels"
HRNET_ROOT = ANYSOLE_INPUT_ROOT / "visual/hrnet/cam3"
HMR_ROOT = ANYSOLE_INPUT_ROOT / "visual/hmr"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_session_dir(session_id: str, root: Path = SHARED_SESSION_ROOT) -> Path:
    """Return one public shared session, rejecting ambiguous identities."""
    matches = sorted(
        path.parent
        for path in Path(root).glob(f"*/*/{session_id}/session.json")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"shared session {session_id!r} not found under {Path(root)}"
        )
    if len(matches) > 1:
        raise ValueError(f"session id is ambiguous in shared facts: {session_id}: {matches}")
    return matches[0]


def _check_frame_axis(session_dir: Path, frames: dict, pressure: dict) -> None:
    frame_id = np.asarray(frames["frame_id"])
    if frame_id.ndim != 1 or frame_id.dtype.kind not in "iu":
        raise ValueError(f"{session_dir}: frame_id must be a 1-D integer array")
    if not np.array_equal(frame_id, np.arange(len(frame_id), dtype=frame_id.dtype)):
        raise ValueError(f"{session_dir}: frame_id must be unique and monotonic from zero")
    n = len(frame_id)
    for key in ("visual_time_s", "mocap_time_s", "valid", "fake"):
        if np.asarray(frames[key]).shape != (n,):
            raise ValueError(f"{session_dir}: frames.{key} length does not match frame_id")
    if np.any(np.asarray(frames["fake"]) & np.asarray(frames["valid"])):
        # A fake frame is not a valid observed frame in the public contract.
        raise ValueError(f"{session_dir}: fake and valid masks overlap")
    if not np.all(np.diff(np.asarray(frames["visual_time_s"], dtype=float)) > 0):
        raise ValueError(f"{session_dir}: visual_time_s must be strictly increasing")
    if not np.all(np.diff(np.asarray(frames["mocap_time_s"], dtype=float)) > 0):
        raise ValueError(f"{session_dir}: mocap_time_s must be strictly increasing")
    if not np.array_equal(np.asarray(pressure["frame_id"]), frame_id):
        raise ValueError(f"{session_dir}: pressure frame_id differs from frames.npz")
    for key in ("left48", "right48", "valid", "fake"):
        value = np.asarray(pressure[key])
        if key in ("left48", "right48"):
            expected = (n, 48)
        else:
            expected = (n,)
        if value.shape != expected:
            raise ValueError(f"{session_dir}: pressure.{key} shape {value.shape} != {expected}")


def load_shared_session(session_id: str, root: Path = SHARED_SESSION_ROOT) -> dict:
    """Load and validate one shared session without reading any model tree."""
    session_dir = shared_session_dir(session_id, root)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    frames = dict(np.load(session_dir / "frames.npz", allow_pickle=True))
    pressure = dict(np.load(session_dir / "pressure_48.npz", allow_pickle=True))
    required_frames = {"frame_id", "visual_time_s", "mocap_time_s", "valid", "fake"}
    required_pressure = {"frame_id", "left48", "right48", "valid", "fake"}
    if not required_frames <= frames.keys():
        raise ValueError(f"{session_dir}: missing frame keys {required_frames - frames.keys()}")
    if not required_pressure <= pressure.keys():
        raise ValueError(f"{session_dir}: missing pressure keys {required_pressure - pressure.keys()}")
    _check_frame_axis(session_dir, frames, pressure)
    if str(meta.get("session_id")) != str(session_id):
        raise ValueError(f"{session_dir}: session.json identity mismatch")
    return {
        "session_id": str(session_id),
        "dir": session_dir,
        "meta": meta,
        "frames": frames,
        "pressure": pressure,
        "source_artifact": session_dir / "artifact.json",
    }


def shared_bvh_path(session: dict) -> Path:
    value = session["meta"].get("source_files", {}).get("bvh", "")
    path = resolve_uri(value, must_exist=True)
    return path


def shared_smpl_path(session: dict, roots: Iterable[Path] = ()) -> Path:
    value = session["meta"].get("source_files", {}).get("smpl", "")
    if value:
        try:
            return resolve_uri(value, must_exist=True)
        except (FileNotFoundError, ValueError):
            pass
    sid = session["session_id"]
    for root in roots:
        matches = sorted(Path(root).glob(f"**/{sid}/motion_neutral_smpl.npz"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"SMPL NPZ not found for {sid}; shared metadata={value!r}")


def model_input_root() -> Path:
    return ANYSOLE_INPUT_ROOT


def hrnet_cache_path(session_id: str) -> Path:
    return HRNET_ROOT / f"{session_id}.pt"


def hmr_cache_path(model: str, session_id: str, cam: int = 3) -> Path:
    """四相机扩展 D2: per-camera GVHMR cache dir cam{N} (cam3 = frozen)."""
    return HMR_ROOT / str(model) / ("cam%d" % cam) / f"{session_id}.pt"


def label_path(session_id: str, method: str) -> Path:
    if not method or "/" in method or "\\" in method or method in {".", ".."}:
        raise ValueError(f"invalid AnySole contact method: {method!r}")
    return LABEL_ROOT / method / f"{session_id}.npz"


def load_contact_label(session: dict, method: str) -> np.ndarray:
    path = label_path(session["session_id"], method)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing private AnySole contact labels {path}. Run "
            f"`python -m anysole.data.contact_adapter --session {session['session_id']} "
            f"--methods {method}`."
        )
    data = dict(np.load(path, allow_pickle=True))
    if "contact" not in data or "frame_id" not in data:
        raise ValueError(f"{path}: expected contact and frame_id arrays")
    contact = np.asarray(data["contact"], dtype=np.float32)
    frame_id = np.asarray(data["frame_id"])
    expected = np.asarray(session["frames"]["frame_id"])
    if contact.shape != (len(expected), 2) or not np.array_equal(frame_id, expected):
        raise ValueError(f"{path}: label shape/frame axis does not match shared session")
    return contact


def write_label(path: Path, *, session: dict, method: str, contact: np.ndarray,
                parameters: dict) -> None:
    frame_id = np.asarray(session["frames"]["frame_id"], dtype=np.int64)
    contact = np.asarray(contact, dtype=np.float32)
    if contact.shape != (len(frame_id), 2):
        raise ValueError(f"contact shape {contact.shape} != {(len(frame_id), 2)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        frame_id=frame_id,
        contact=contact,
        valid=np.asarray(session["frames"]["valid"], dtype=np.uint8),
        fake=np.asarray(session["frames"]["fake"], dtype=np.uint8),
        method=np.asarray(method),
        adapter_version=np.asarray(ADAPTER_VERSION),
        parameters_json=np.asarray(json.dumps(parameters, sort_keys=True)),
        source_session=np.asarray(canonical_uri(session["dir"])),
        source_session_hash=np.asarray(
            sha256_file(session["source_artifact"]) if session["source_artifact"].is_file() else ""
        ),
    )
    os.replace(temporary, path)


def write_cache_provenance(cache_path: Path, *, session: dict, kind: str,
                           shape: tuple[int, ...], parameters: dict | None = None) -> None:
    payload = {
        "schema_version": "model_input.anysole.v1",
        "adapter_version": ADAPTER_VERSION,
        "producer": f"anysole.data.{kind}",
        "session_id": session["session_id"],
        "source_artifacts": [canonical_uri(session["dir"] / "frames.npz")],
        "source_hashes": {
            "session": sha256_file(session["source_artifact"])
            if session["source_artifact"].is_file() else "",
        },
        "frame_id_min": 0,
        "frame_id_max": int(shape[0] - 1),
        "frame_count": int(shape[0]),
        "shape": list(shape),
        "parameters": parameters or {},
        "consumers": ["AnySole"],
    }
    sidecar = Path(str(cache_path) + ".artifact.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# The grouped protocol reads imported physical files directly. It never
# materializes shared facts, contact labels, features, or statistics on load.
VIEW_SCHEMA = VIEW_MANIFEST_SCHEMA
_RGB_NAME = re.compile(r"^frame(\d+)_(\d+)_(\d{6}\.\d+)\.jpg$")


def semantic_hash(value: dict | list) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_view_config(path: Path) -> dict:
    import yaml
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError("view configuration must be a mapping")
    return {**obj, **obj.get("data", {})}


def read_rgb_sources(directory: Path, camera: int) -> list[dict]:
    """Numerical frame ordering; timestamps remain physical seconds of day."""
    from anysole.data.rgb_smpl.alignment import unwrap_seconds_of_day
    rows = []
    for path in Path(directory).glob("*.jpg"):
        match = _RGB_NAME.fullmatch(path.name)
        if match is None or int(match[2]) != int(camera):
            raise ValueError(f"malformed RGB frame or camera mismatch: {path}")
        clock = match[3]
        if int(clock[:2]) >= 24 or int(clock[2:4]) >= 60 or float(clock[4:]) >= 60:
            raise ValueError(f"invalid RGB wall clock: {path}")
        wall = int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + float(clock[4:])
        rows.append({"file": path.name, "frame_id": int(match[1]), "time_s": wall})
    rows.sort(key=lambda row: row["frame_id"])
    if rows:
        ids = np.asarray([row["frame_id"] for row in rows])
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f"duplicate RGB frame numbers: {directory}")
        times = unwrap_seconds_of_day(np.asarray([row["time_s"] for row in rows]))
        for row, value in zip(rows, times):
            row["time_s"] = float(value)
    return rows


def match_rgb_sources(target: np.ndarray, sources: list[dict], tolerance_ms: float) -> dict:
    """Nearest real image; equal timestamps choose the smallest numeric frame."""
    target = np.asarray(target, dtype=np.float64)
    n = len(target)
    if not sources:
        return {"source_row": [-1] * n, "source_frame_id": [-1] * n,
                "source_time_s": [0.] * n, "time_residual_ms": [0.] * n,
                "view_valid": [False] * n, "reuse": [False] * n}
    times = np.asarray([r["time_s"] for r in sources], dtype=np.float64)
    first = np.r_[True, np.diff(times) > 0]
    unique_rows = np.flatnonzero(first)
    unique = times[unique_rows]
    upper = np.minimum(np.searchsorted(unique, target), len(unique) - 1)
    lower = np.maximum(upper - 1, 0)
    # Millisecond wall times are not exactly representable at ~70,000s.
    choose = np.where(np.abs(unique[lower] - target) <= np.abs(unique[upper] - target) + 1e-9, lower, upper)
    rows = unique_rows[choose]
    selected = times[rows]
    residual = (selected - target) * 1000
    valid = ((target >= times[0]) & (target <= times[-1]) &
             (np.abs(residual) <= float(tolerance_ms) + 1e-7))
    reuse = np.r_[False, rows[1:] == rows[:-1]]
    return {"source_row": rows.tolist(),
            "source_frame_id": [sources[i]["frame_id"] for i in rows],
            "source_time_s": selected.tolist(), "time_residual_ms": residual.tolist(),
            "view_valid": valid.tolist(), "reuse": reuse.tolist()}


def source_pressure_mask(valid: np.ndarray, fake: np.ndarray, span: dict) -> np.ndarray:
    valid, fake = np.asarray(valid, bool), np.asarray(fake, bool)
    if valid.ndim != 1 or valid.shape != fake.shape or np.any(valid & fake):
        raise ValueError("pressure valid/fake masks are malformed or overlap")
    feet = np.repeat((valid & ~fake)[:, None], 2, axis=1)
    for j, foot in enumerate(("left", "right")):
        counts = span["out_of_span_frames_" + foot]
        if len(counts) != 2 or any(int(x) != x or x < 0 for x in counts) or sum(counts) > len(valid):
            raise ValueError(f"invalid {foot} source prefix/suffix counts")
        start, end = map(int, counts)
        feet[:start, j] = False
        if end:
            feet[len(valid)-end:, j] = False
    return feet


def build_grouped_manifest(config: dict, output: Path, *, overwrite: bool = False) -> dict:
    """Explicit, deterministic stored-pressure-grid index generation."""
    from anysole.data.rgb_smpl.alignment import bracket_times
    output = Path(output)
    meta_path = output.with_suffix(".meta.json")
    if meta_path.exists() and json.loads(meta_path.read_text()).get("frozen"):
        raise ValueError("a manifest consumed by a frozen training plan cannot be overwritten")
    if not overwrite and (output.exists() or meta_path.exists()):
        raise FileExistsError(f"manifest already exists: {output}; use explicit --overwrite before training only")
    root = Path(config["dataset_root"]).resolve()
    split_path = Path(config["split_csv"])
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    splits = {role: {r[role].strip() for r in split_rows if r.get(role, "").strip()}
              for role in ("train", "val", "test")}
    b1 = config.get("b1_version") == "anysole.b1.v1"
    if b1:
        for role, members in splits.items():
            if sum(bool(r.get(role, "").strip()) for r in split_rows) != len(members):
                raise ValueError(f"duplicate B1 {role} recording")
        if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("B1 recording splits must be disjoint")
    if splits["train"] & (splits["val"] | splits["test"]):
        raise ValueError("historical train overlaps heldout")
    metadata_root = Path(config["pressure_metadata_root"])
    metadata_paths = {name: metadata_root / name for name in
                      ("sessions.csv", "alignment_audit.json", "selection_audit.json")}
    metadata_hashes = {name: sha256_file(path) for name, path in metadata_paths.items()}
    with metadata_paths["sessions.csv"].open(encoding="utf-8-sig", newline="") as handle:
        sessions = list(csv.DictReader(handle))
    def key(row):
        parts = row["session"].split("/")
        return parts[0], parts[-1]
    align = {key(r): r for r in json.loads(metadata_paths["alignment_audit.json"].read_text())}
    selected = {key(r): r for r in json.loads(metadata_paths["selection_audit.json"].read_text())["sessions"]}
    method = str(config.get("rgb_method", "method_1_offset"))
    tol = float(config.get("rgb_tolerance_ms", 25))
    source_gap = float(config.get("source_gap_ms", 50)) / 1000
    motion_gap = float(config.get("motion_gap_s", 1.5 / 120))
    anchor_cam = int(config.get("clock_anchor_camera", 1 if b1 else 3))
    cameras = tuple(int(c) for c in config.get("manifest_cameras", (1, 2, 4) if b1 else (1, 2, 3, 4)))
    if not cameras or len(cameras) != len(set(cameras)) or set(cameras) - {1, 2, 3, 4}:
        raise ValueError("invalid manifest cameras")
    if b1 and (anchor_cam != 1 or set(cameras) != {1, 2, 4}):
        raise ValueError("B1 preparation requires cam1 clock and only cameras 1,2,4")
    records, exclusions, train_residuals = [], [], []
    all_ids = set.union(*splits.values())
    for session in sorted(sessions, key=lambda r: (r["date"], r["session"])):
        date, sid = session["date"], session["session"]
        if sid not in all_ids:
            continue
        roles = [r for r in splits if sid in splits[r]]
        mp = root / date / "Mocap" / sid / "motion_neutral_smpl.npz"
        pp = root / date / "Tactile" / sid / "pressure_48.npz"
        rgb_root = root / date / "RGB" / method / sid
        missing = [name for name, path in (("Mocap", mp), ("Tactile", pp), ("RGB", rgb_root)) if not path.exists()]
        if missing:
            exclusions.append({"recording_id": f"{date}/{sid}", "session_id": sid,
                               "split_roles": roles, "reasons": ["missing_" + x for x in missing]})
            continue
        skey = date, sid
        choice = selected[skey]
        camera = choice["selected_camera"]
        pa = next(c for c in align[skey]["cameras"] if c["camera"] == camera)
        candidate = next(c for c in choice["candidates"] if c["camera"] == camera)
        psha = sha256_file(pp)
        with np.load(pp, allow_pickle=False) as archive:
            fid, valid, fake = archive["frame_id"], archive["valid"], archive["fake"]
            pressure = np.concatenate([archive["left48"], archive["right48"]], axis=1)
            unit = str(archive["unit"].item())
            alignment_method = str(archive["alignment_method"].item())
        n = len(fid)
        if not (psha == choice["selected_sha256"] == candidate["sha256"] and
                camera == session["selected_camera"] == align[skey]["selected_camera"] and
                n == pa["frame_count"] == candidate["frame_count"] == int(session["frame_count"]) and
                np.array_equal(fid, np.arange(n)) and float(pa["fps"]) == 40. and
                np.isclose(pa["first_frame_wall_s"], candidate["video_first_wall_s"], rtol=0, atol=1e-7)):
            raise ValueError(f"pressure provenance does not bind to NPZ: {skey}")
        if unit not in {"source_sensor_units", "normalized_sensor_reading"}:
            raise ValueError(f"unrecognized pressure unit {unit!r}: {skey}")
        feet = source_pressure_mask(valid, fake, pa)
        cell_mask = np.repeat(feet, 48, axis=1)
        if pressure.shape != (n, 96) or np.any(cell_mask & ~np.isfinite(pressure)):
            raise ValueError(f"valid pressure labels contain NaN/Inf or wrong shape: {skey}")
        if unit == "normalized_sensor_reading" and np.any((pressure[cell_mask] < 0) | (pressure[cell_mask] > 1)):
            raise ValueError(f"normalized pressure lies outside [0,1]: {skey}")
        target = float(pa["first_frame_wall_s"]) + fid / float(pa["fps"])
        # Filter before directory access: a B1 preparation never even lists cam3.
        sources = {c: read_rgb_sources(rgb_root / f"camera_{c}", c) for c in cameras}
        csv_path = root / date / "RGB_MoCap_time_alignment" / f"{sid}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            anchors = list(csv.DictReader(handle))
        marks = {r["标记"]: r for r in anchors}
        if len(anchors) != 2 or set(marks) != {"start", "end"}:
            raise ValueError(f"ambiguous start/end anchors: {csv_path}")
        by_id = {r["frame_id"]: r for r in sources[anchor_cam]}
        w0, w1 = (by_id[int(marks[mark]["视觉帧"])]["time_s"] for mark in ("start", "end"))
        m0, m1 = (float(marks[mark]["动捕时间(s)"]) for mark in ("start", "end"))
        if not (w1 > w0 and m1 > m0):
            raise ValueError(f"nonpositive anchor span: {csv_path}")
        slope = (m1 - m0) / (w1 - w0)
        mt = m0 + (target - w0) * slope
        with np.load(mp, allow_pickle=False) as archive:
            st = archive["source_frame_times_s"]
            poses, trans = archive["poses"], archive["trans"]
            motion_meta = {k: archive[k].tolist() for k in ("coordinate_system", "model_units",
                           "surface_model_sha256", "root_branch_repair_status", "joint_continuity_status")}
        if poses.shape != (len(st), 72) or trans.shape != (len(st), 3):
            raise ValueError(f"invalid native motion shape: {mp}")
        finite_motion = np.isfinite(poses).all(axis=1) & np.isfinite(trans).all(axis=1)
        if not finite_motion.all():
            raise ValueError(f"source motion labels contain NaN/Inf: {mp}")
        brackets = bracket_times(st, mt)
        mv = (brackets.valid & (target >= w0) & (target <= w1) &
              ((st[brackets.upper] - st[brackets.lower]) <= motion_gap))
        adjacent = (np.diff(fid) == 1) & np.isclose(np.diff(target), 1 / 40, rtol=0, atol=1e-7)
        motion_edges = mv[:-1] & mv[1:] & adjacent
        pressure_edges = feet[:-1] & feet[1:] & adjacent[:, None]
        reason = []
        if alignment_method != "nearest_values_absolute_t_us":
            reason.append("untrusted_pressure_time_basis")
        if not cell_mask.any():
            reason.append("no_trustworthy_pressure")
        if not mv.any():
            reason.append("no_bracketed_motion")
        if motion_meta["root_branch_repair_status"] not in {"passed", "repaired"} or motion_meta["joint_continuity_status"] not in {"passed", "repaired"}:
            reason.append("untrusted_motion_quality")
        views = []
        for c, source in sources.items():
            match = match_rgb_sources(target, source, tol)
            vv = np.asarray(match["view_valid"], bool)
            ids = np.asarray(match["source_frame_id"])
            ts = np.asarray(match["source_time_s"])
            ve = (vv[:-1] & vv[1:] & adjacent & (ids[:-1] != ids[1:]) &
                  (np.diff(ts) > 0) & (np.diff(ts) <= source_gap + 1e-9))
            views.append({"camera_id": c, "directory": str((rgb_root / f"camera_{c}").relative_to(root)),
                          "sources": source, **match, "view_edges": ve.tolist()})
            if "train" in roles and not reason and source:
                domain = mv & cell_mask.any(axis=1) & (target >= source[0]["time_s"]) & (target <= source[-1]["time_s"])
                train_residuals.extend(np.abs(np.asarray(match["time_residual_ms"])[domain]).tolist())
        if not any(any(view["view_valid"]) for view in views):
            reason.append("no_valid_rgb")
        calibration = root / date / f"{date}.json"
        cal = json.loads(calibration.read_text())
        camera_contract = {}
        clock_verification = {}
        if b1:
            from anysole.utils.geometry import calibrated_camera_geometry_np
            calibrated_camera_geometry_np(cal, cameras)
            # Canonical pressure labels retain their original selected grid;
            # its absolute hardware clock is independently shared with cam1.
            cam1_pressure = next(c for c in align[skey]["cameras"] if c["camera"] == "cam1")
            epoch = pa.get("pressure_epoch_wall_s")
            cam1_epoch = cam1_pressure.get("pressure_epoch_wall_s")
            if not (epoch is not None and cam1_epoch is not None and
                    np.isfinite([float(epoch), float(cam1_epoch)]).all() and
                    np.isclose(float(epoch), float(cam1_epoch), rtol=0, atol=1e-7)):
                reason.append("unverified_pressure_cam1_shared_clock")
            visual_span = float(marks["end"]["视觉时间(s)"]) - float(marks["start"]["视觉时间(s)"])
            clock_verification = {
                "basis": "cam1_jpeg_absolute_wall_endpoints_to_mocap_csv",
                "cam1_source_index_sha256": semantic_hash(sources[1]),
                "cam1_pressure_epoch_wall_s": cam1_epoch,
                "pressure_epoch_wall_s": epoch,
                "csv_visual_span_residual_ms": ((w1 - w0) - visual_span) * 1000,
                "cam1_start_vs_acquisition_csv_residual_ms":
                    (w0 - float(cam1_pressure["first_frame_wall_s"]) -
                     float(marks["start"]["视觉时间(s)"])) * 1000,
                "inherited_pressure_grid": True,
                "inherited_pressure_selected_camera": camera,
                "cam3_image_or_feature_access": False,
                "raw_pressure_csv_available": False,
            }
            camera_contract = {"b1_version": "anysole.b1.v1",
                "camera_frame": "camera_relative_origin", "teacher_frame": "camera1_relative_origin",
                "R_cw": "R_camera_checkerboard @ R_checkerboard_mocap",
                "translation_m": "(R_camera_checkerboard @ t_checkerboard_mocap_cm + t_camera_cm)/100",
                "geometry_layout": "relative_R_rowmajor9,reference_camera_center_m3,fx/W,fy/H,cx/W,cy/H",
                "geometry_center_scale_m": 1.0, "reference_camera": 1,
                "deferred_final_test_cameras": [3]}
        subject = session["subject"]
        records.append({"schema_version": VIEW_SCHEMA, "recording_id": f"{date}/{sid}",
            "date": date, "session_id": sid, "subject": subject, "action": sid[len(subject):-1],
            "split_roles": roles, "eligible": not reason, "exclusion_reasons": reason, "method": method,
            "target_frame_id": fid.astype(int).tolist(), "target_time_s": target.tolist(),
            "pressure": {"path": str(pp.relative_to(root)), "sha256": psha, "unit": unit,
                "normalizer": "clip_0_1023_div_1023" if unit == "source_sensor_units" else "identity",
                "row": fid.astype(int).tolist(), "foot_valid": feet.tolist(), "fake": np.asarray(fake, bool).tolist(),
                "selected_camera": camera, "fps": 40., "first_frame_wall_s": pa["first_frame_wall_s"],
                "alignment_method": alignment_method, "source_metadata_hashes": metadata_hashes,
                "source_csv_hashes": {f: pa[f + "_source_sha256"] for f in ("left", "right")},
                "source_span": {f: pa["out_of_span_frames_" + f] for f in ("left", "right")}},
            "motion": {"path": str(mp.relative_to(root)), "sha256": sha256_file(mp), **motion_meta,
                "time_s": mt.tolist(), "lower": brackets.lower.tolist(), "upper": brackets.upper.tolist(),
                "alpha": brackets.alpha.tolist(), "valid": mv.tolist(), "source_gap_s": motion_gap},
            "clock_mapping": {"path": str(csv_path.relative_to(root)), "sha256": sha256_file(csv_path),
                "anchor_camera": anchor_cam, "wall_start_s": w0, "wall_end_s": w1,
                "mocap_start_s": m0, "mocap_end_s": m1, "slope": slope, **clock_verification},
            "calibration": {"path": str(calibration.relative_to(root)), "sha256": sha256_file(calibration),
                            "pixel_coordinates": cal["pixel_coordinates"], **camera_contract},
            "views": views, "motion_edges": motion_edges.tolist(), "pressure_foot_edges": pressure_edges.tolist()})
    # Detect old members omitted from metadata rather than silently losing them.
    represented = {r["session_id"] for r in records + exclusions}
    if represented != all_ids:
        raise ValueError(f"split sessions have no pressure provenance: {sorted(all_ids - represented)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for r in records)
    manifest_hash = hashlib.sha256(payload.encode()).hexdigest()
    values = np.asarray(train_residuals)
    if not len(values):
        raise ValueError("no eligible train RGB clock residuals")
    meta = {"schema_version": VIEW_SCHEMA, "manifest_sha256": manifest_hash, "frozen":False,
            "producer_sha256":sha256_file(Path(__file__)),
            "split_path": str(split_path.resolve()), "split_sha256": sha256_file(split_path),
            "original_split_counts": {k: len(v) for k, v in splits.items()},
            "candidate_split_counts": {k: sum(k in r["split_roles"] for r in records) for k in splits},
            "eligible_split_counts": {k: sum(k in r["split_roles"] and r["eligible"] for r in records) for k in splits},
            "val_test_identical": splits["val"] == splits["test"], "missing_modalities": exclusions,
            "invalid_recordings": [{"recording_id": r["recording_id"], "reasons": r["exclusion_reasons"]} for r in records if not r["eligible"]],
            "protocol": {"rgb_method": method, "clock_anchor_camera": anchor_cam,
                         **({"b1_version": "anysole.b1.v1", "indexed_cameras": list(cameras),
                             "cam3_index_deferred_until_final_test": True,
                             "clock_basis": "verified_cam1_endpoints_with_inherited_absolute_pressure_grid"} if b1 else {}),
                         "rgb_tolerance_ms": tol, "source_gap_ms": source_gap*1000, "motion_gap_s": motion_gap,
                         "timestamp_tie": "smallest_numeric_frame_id; nearest_tie_earlier_time",
                         "pressure_time": "stored_selected_camera_40Hz_grid", "target_time_domain": "date_seconds_of_day",
                         "method_reason": "same measured time coverage as method2; fixed method1 baseline convention, not outcome selection",
                         "threshold_reason": "old train residual p99 near23ms; 25ms avoids 50ms two-frame mismatch"},
            "train_residual_ms": {"n": len(values), "p50": float(np.median(values)),
                                   "p95": float(np.quantile(values,.95)), "p99": float(np.quantile(values,.99)),
                                   "max": float(values.max()),
                                   "retained": {str(ms): int((values <= ms+1e-7).sum()) for ms in (12.5,25,50)}}}
    output.write_text(payload, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return meta


def load_grouped_manifest(path: Path) -> tuple[list[dict], dict]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if meta.get("schema_version") != VIEW_SCHEMA or sha256_file(path) != meta["manifest_sha256"]:
        raise ValueError("grouped manifest schema/hash mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(r.get("schema_version") != VIEW_SCHEMA for r in records):
        raise ValueError("grouped recording schema mismatch")
    if len({r['recording_id'] for r in records})!=len(records):
        raise ValueError("duplicate physical recording in grouped manifest")
    for role in ('train','val','test'):
        if sum(role in r['split_roles'] for r in records)!=meta['candidate_split_counts'][role]:
            raise ValueError("grouped split-role count differs from metadata")
        if sum(role in r['split_roles'] and r['eligible'] for r in records)!=meta['eligible_split_counts'][role]:
            raise ValueError("grouped eligible count differs from metadata")
    return records, meta


def load_grouped_labels(record: dict, dataset_root: Path, smpl_model_path: Path) -> dict:
    """One truth sequence per physical recording; no external Pressure access."""
    from anysole.data.smpl_io import load_smpl
    from anysole.utils.geometry import fk_pose6d_np
    root = Path(dataset_root)
    n = len(record["target_frame_id"])
    for name in ("motion", "pressure"):
        if sha256_file(root / record[name]["path"]) != record[name]["sha256"]:
            raise ValueError(f"{name} source hash changed for {record['recording_id']}")
    mv = np.asarray(record["motion"]["valid"], bool)
    motion = load_smpl(root / record["motion"]["path"],
                       np.asarray(record["motion"]["time_s"])[mv],
                       model_path=smpl_model_path, allow_extrapolation=False)
    pose, anchor, kp = np.zeros((n,144),np.float32), np.zeros((n,3),np.float32), np.zeros((n,24,3),np.float32)
    pose[mv], anchor[mv] = motion["pose_6d"], motion["trans_m"]
    kp[mv] = fk_pose6d_np(motion["pose_6d"], motion["trans_m"], motion["offsets_m"], motion["parents"])
    with np.load(root / record["pressure"]["path"], allow_pickle=False) as archive:
        pressure = np.concatenate([archive["left48"], archive["right48"]], axis=1).astype(np.float32)
    pvalid = np.repeat(np.asarray(record["pressure"]["foot_valid"], bool), 48, axis=1)
    if np.any(pvalid & ~np.isfinite(pressure)):
        raise ValueError("valid pressure labels contain nonfinite values")
    pressure = np.where(pvalid, pressure, 0)
    if record["pressure"]["normalizer"] == "clip_0_1023_div_1023":
        pressure = np.clip(pressure, 0, 1023) / 1023
    velocity = np.zeros((n,3),np.float32)
    edges = np.asarray(record["motion_edges"],bool)
    dt = np.diff(np.asarray(record["target_time_s"],np.float64))
    velocity[1:][edges] = (anchor[1:][edges]-anchor[:-1][edges])/dt[edges,None]
    return {"pose6d":pose, "root":anchor, "velocity":velocity, "pressure":pressure,
            "betas":motion["betas"], "offsets":motion["offsets_m"], "parents":motion["parents"], "kp_gt":kp}


def fit_grouped_stats(manifest: Path, dataset_root: Path, smpl_model_path: Path,
                      output: Path | None = None) -> dict:
    """Train-only moments, with B1 root/velocity in allowed camera frames."""
    records, meta = load_grouped_manifest(manifest)
    b1 = meta.get("protocol", {}).get("b1_version") == "anysole.b1.v1"
    moments = {key: [None, None, 0] for key in ("pose", "velocity")}
    recording_ids, pressure_feet, pressure_support = [], [[], []], []
    for record in records:
        if not record["eligible"] or "train" not in record["split_roles"]:
            continue
        labels = load_grouped_labels(record, dataset_root, smpl_model_path)
        selected_views = [v for v in record["views"] if not b1 or v["camera_id"] in (1, 2, 4)]
        any_view = np.asarray([v["view_valid"] for v in selected_views],bool).any(axis=0)
        pm = np.asarray(record["motion"]["valid"],bool) & any_view
        vm = np.r_[False,np.asarray(record["motion_edges"],bool)] & any_view
        samples = [(labels["pose6d"], labels["velocity"], pm, vm)]
        if b1:
            from anysole.utils.geometry import calibrated_camera_geometry_np, rot6d_to_rotmat_np, rotmat_to_6d_np
            path = Path(dataset_root) / record["calibration"]["path"]
            if sha256_file(path) != record["calibration"]["sha256"]:
                raise ValueError("train calibration source changed")
            rotations = calibrated_camera_geometry_np(json.loads(path.read_text()),
                [v["camera_id"] for v in selected_views])["camera_R_cw"].astype(np.float64)
            samples = []
            root_rotation = rot6d_to_rotmat_np(labels["pose6d"][:, :6])
            for rotation, view in zip(rotations, selected_views):
                pose = labels["pose6d"].copy()
                pose[:, :6] = rotmat_to_6d_np(np.einsum("ij,tjk->tik", rotation, root_rotation))
                velocity = np.einsum("ij,tj->ti", rotation, labels["velocity"])
                seen = np.asarray(view["view_valid"], bool)
                samples.append((pose, velocity, pm & seen, vm & seen))
            foot_mask = np.asarray(record["pressure"]["foot_valid"], bool) & any_view[:, None]
            totals = labels["pressure"].reshape(-1, 2, 48).sum(-1).astype(np.float64)
            for foot in range(2):
                pressure_feet[foot].append(totals[foot_mask[:, foot], foot])
            pressure_support.append({"recording_id": record["recording_id"],
                                     "pressure_sha256": record["pressure"]["sha256"],
                                     "foot_mask_sha256": hashlib.sha256(foot_mask.tobytes()).hexdigest()})
        for name, values, mask in ((name, values, mask) for pose, vel, pmask, vmask in samples
                                  for name, values, mask in (("pose", pose, pmask), ("velocity", vel, vmask))):
            values = values[mask].astype(np.float64)
            total, squares, count = moments[name]
            moments[name] = [values.sum(axis=0) if total is None else total+values.sum(axis=0),
                             (values**2).sum(axis=0) if squares is None else squares+(values**2).sum(axis=0),
                             count+len(values)]
        recording_ids.append(record["recording_id"])
    result = {"schema_version":VIEW_STATS_SCHEMA, "manifest_sha256":meta["manifest_sha256"],
              "split_sha256":meta["split_sha256"], "smpl_model_sha256":sha256_file(smpl_model_path),
              "recordings":recording_ids, "frame_weighting":"unique_physical_target_frame",
              "pressure_normalizer":"clip_0_1023_div_1023"}
    if b1:
        from anysole.utils.a2eval_metrics import fit_train_thresholds, finalize_thresholds
        totals = [np.concatenate(values) for values in pressure_feet]
        threshold = [max(float(np.quantile(values, .2, method="linear")), 1e-6) for values in totals]
        result.update({"b1_version": "anysole.b1.v1", "coordinate_frame": "camera_relative_origin",
                       "statistics_cameras": [1, 2, 4],
                       "frame_weighting": "unique_physical_target_camera_frame; no_window_repetition",
                       "pressure_frame_weighting": "unique_physical_target_frame_with_any_allowed_RGB",
                       "pressure_eval": {"policy_version": "anysole.b1.pressure-eval.q10-q20.v1",
                           "fit_split": "train", "fitted_from_train_sha256": semantic_hash(pressure_support),
                           "manifest_sha256": meta["manifest_sha256"], "split_sha256": meta["split_sha256"],
                           "contact_threshold_normalized": threshold,
                           "low_load_threshold_normalized": [float(np.quantile(v, .1, method="linear")) for v in totals],
                           "fit_algorithm": "per-foot train sum48 NumPy linear Q0.10/Q0.20; contact floor1e-6",
                           "physical_foot_ground_contact": False,
                           "contact_definition": "sum48 > train Q0.20; engineering pressure activation",
                           "threshold_units": "unweighted sum48 normalized sensor reading",
                           "sensor_geometry_status": "N/A; trusted numeric sensor coordinates unavailable"}})
        pressure_stats = fit_train_thresholds(records, dataset_root, result)
        pressure_stats["pressure_log_total_scale"] = [max(float(np.std(np.log1p(v / 48.0))), 1e-3) for v in totals]
        pressure_stats["science_config"]["pressure_log_total_scale"] = pressure_stats["pressure_log_total_scale"]
        result["pressure_train_statistics"] = finalize_thresholds(pressure_stats)
    for name,(total,squares,count) in moments.items():
        if not count:
            raise ValueError(f"no valid old-train {name} statistics")
        mean = total/count
        std = np.maximum(np.sqrt(np.maximum(squares/count-mean**2,0)),1e-6)
        result[name+"_mean"],result[name+"_std"],result[name+"_count"] = mean.tolist(),std.tolist(),count
    result["stats_sha256"] = semantic_hash(result)
    if output is not None:
        output=Path(output)
        if output.exists():
            raise FileExistsError(f"statistics already exist: {output}")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def grouped_window_inventory(manifest: Path, *, window_length: int = 20, stride: int = 20,
                             cameras: tuple[int, ...] | None = None) -> dict:
    """Pre-frontend coverage; a detector failure may only reduce these counts."""
    records,meta=load_grouped_manifest(manifest)
    if cameras is None:
        cameras = tuple(meta.get("protocol", {}).get("indexed_cameras", (1, 2, 3, 4)))
    result={role:{"recordings":0,"group_windows":0,"cam3_windows":0,"rgb_windows":0,
                  "zero_supervision_windows":0,"tail_rows":0,"views_per_window":{}}
            for role in ("train","val","test")}
    for record in records:
        if not record["eligible"]:continue
        n=len(record["target_frame_id"])
        selected_views = [v for v in record["views"] if v["camera_id"] in cameras]
        views=[np.asarray(v["view_valid"],bool) for v in selected_views]
        supervised=(np.asarray(record["motion"]["valid"],bool)|np.asarray(record["pressure"]["foot_valid"],bool).any(axis=1))
        for role in record["split_roles"]:
            count=result[role];count["recordings"]+=1
            starts=list(range(0,max(n-window_length+1,0),stride))
            count["tail_rows"]+=n-(starts[-1]+window_length if starts else 0)
            for start in starts:
                stop=start+window_length
                active=[v[start:stop].any() for v in views]
                has_truth=[(v[start:stop]&supervised[start:stop]).any() for v in views]
                if any(has_truth):
                    c=sum(active);count["group_windows"]+=1;count["rgb_windows"]+=int(c)
                    count["views_per_window"][str(int(c))]=count["views_per_window"].get(str(int(c)),0)+1
                else:count["zero_supervision_windows"]+=1
                count["cam3_windows"] += sum(int(keep) for keep, view in zip(has_truth, selected_views)
                                              if view["camera_id"] == 3)
    return {"manifest_sha256":meta["manifest_sha256"],"window_length":window_length,"stride":stride,
            "cameras": list(cameras),
            "train_tail_policy":"drop_incomplete; no_time_compression", "frontend_checked":False,"splits":result}


def load_grouped_stats(path: Path, *, manifest_sha256: str | None = None) -> dict:
    result=json.loads(Path(path).read_text())
    digest=result.pop("stats_sha256",None)
    if result.get("schema_version")!=VIEW_STATS_SCHEMA or semantic_hash(result)!=digest:
        raise ValueError("statistics schema/hash mismatch")
    if manifest_sha256 is not None and result["manifest_sha256"]!=manifest_sha256:
        raise ValueError("statistics were fitted on another manifest")
    return {**result,"stats_sha256":digest}
