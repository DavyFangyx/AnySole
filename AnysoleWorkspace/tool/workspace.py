#!/usr/bin/env python3
"""Initialize and validate the canonical AnysoleWorkspace only.

The historical ``sources/``, ``derived/``, ``dependencies/``, ``calibration/``,
``manifests/`` and ``splits/`` trees are intentionally not supported here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "AnysoleWorkspace"

CANONICAL_ROOTS = {
    "raw": WORKSPACE / "raw",
    "protocol": WORKSPACE / "protocol",
    "shared": WORKSPACE / "shared",
    "model_inputs": WORKSPACE / "model_inputs",
    "work": WORKSPACE / "work",
    "assets": WORKSPACE / "assets",
    "reports": WORKSPACE / "reports",
    "results": ROOT / "results",
}
URI_ROOTS = {key: CANONICAL_ROOTS[key] for key in ("raw", "protocol", "shared", "work", "results")}
URI_ROOTS.update({"asset": CANONICAL_ROOTS["assets"]})

# raw/{rgb,pressure,bvh,smpl} are modality link farms built by tool/relink_raw.py
# (shape-preserving views over the upstream tree; see reports/raw_relink_plan_20261001.md).
# raw/{calibration,human_masks} are plain read-only symlinks to external sources.
RAW_FARM_ENTRIES = ("rgb", "pressure", "bvh", "smpl")
RAW_SYMLINKS = {
    WORKSPACE / "raw/calibration": Path("/data/lizhe/projects/Tactile/0_Calibration"),
    WORKSPACE / "raw/human_masks": Path("/data/lizhe/projects/Tactile/3_Result/processed/rgb_human_masks"),
}


class WorkspacePathError(ValueError):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return False
    return True


def resolve_uri(value: str | Path, *, for_write: bool = False, must_exist: bool = False) -> Path:
    """Resolve one canonical URI; legacy ``workspace://`` is rejected."""
    text = os.fspath(value)
    if "://" not in text:
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not _inside(candidate, ROOT):
            raise WorkspacePathError(f"path escapes repository: {value}")
    else:
        scheme, remainder = text.split("://", 1)
        if scheme == "model-input":
            model, separator, tail = remainder.partition("/")
            if not separator or not model or model in {".", ".."}:
                raise WorkspacePathError(f"model-input URI requires <model>/: {value}")
            root = CANONICAL_ROOTS["model_inputs"] / model
            candidate = root / tail
        elif scheme in URI_ROOTS:
            root = URI_ROOTS[scheme]
            candidate = root / remainder
        else:
            raise WorkspacePathError(f"unsupported URI scheme: {scheme}://")
        if not _inside(candidate, root):
            raise WorkspacePathError(f"URI escapes its root: {value}")
    if must_exist and not candidate.exists():
        raise WorkspacePathError(f"path does not exist: {value}")
    if for_write and text.startswith("raw://"):
        raise WorkspacePathError("raw:// is read-only")
    return candidate


def canonical_uri(path: str | Path) -> str:
    candidate = Path(path).resolve(strict=False)
    model_root = CANONICAL_ROOTS["model_inputs"].resolve(strict=False)
    if _inside(candidate, model_root):
        return "model-input://" + candidate.relative_to(model_root).as_posix()
    for scheme, root in sorted(URI_ROOTS.items(), key=lambda item: len(str(item[1])), reverse=True):
        resolved_root = root.resolve(strict=False)
        if _inside(candidate, resolved_root):
            return f"{scheme}://{candidate.relative_to(resolved_root).as_posix()}"
    raise WorkspacePathError(f"path is outside canonical roots: {path}")


CANONICAL_DIRS = (
    WORKSPACE / "raw",
    WORKSPACE / "protocol/manifests",
    WORKSPACE / "protocol/splits/default",
    WORKSPACE / "protocol/schemas",
    WORKSPACE / "protocol/calibration",
    WORKSPACE / "shared/facts/sessions",
    WORKSPACE / "shared/representations/tactile/mmvp_31x11",
    WORKSPACE / "shared/frontends",
    WORKSPACE / "model_inputs",
    WORKSPACE / "work/data_pipeline/pressure_washer",
    WORKSPACE / "assets",
    WORKSPACE / "reports",
)


def ensure_link(link: Path, target: Path, dry_run: bool) -> None:
    if link.is_symlink():
        actual = (link.parent / os.readlink(link)).resolve(strict=False)
        if actual != target.resolve(strict=False):
            raise RuntimeError(f"unexpected link target: {link} -> {os.readlink(link)}")
        return
    if link.exists():
        raise RuntimeError(f"cannot create link over existing path: {link}")
    print(f"link {link.relative_to(ROOT)} -> {target}")
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)


def initialize(dry_run: bool = False) -> None:
    for directory in CANONICAL_DIRS:
        print(f"mkdir {directory.relative_to(ROOT)}")
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)
    for link, target in RAW_SYMLINKS.items():
        ensure_link(link, target, dry_run)
    if dry_run:
        print("relink_raw build --dry-run")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("relink_raw", Path(__file__).parent / "relink_raw.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(dry_run=False)


def doctor() -> int:
    errors: list[str] = []
    for link, target in RAW_SYMLINKS.items():
        if not link.is_symlink():
            errors.append(f"missing raw link: {link}")
        elif link.resolve(strict=False) != target.resolve(strict=False):
            errors.append(f"wrong raw link: {link} -> {os.readlink(link)}")
        elif not link.exists():
            errors.append(f"broken raw link: {link}")
    for name in RAW_FARM_ENTRIES:
        entry = WORKSPACE / "raw" / name
        if entry.is_symlink() or not entry.is_dir():
            errors.append(f"missing raw farm entry: {entry}")
    manifest = WORKSPACE / "protocol/manifests/session_manifest.jsonl"
    if manifest.is_file():
        import json as _json
        checked = missing = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            for field in ("video_path", "bvh_path", "smpl_path"):
                uri = row.get(field)
                if not uri:
                    continue
                checked += 1
                try:
                    resolve_uri(uri, must_exist=True)
                except WorkspacePathError as exc:
                    missing += 1
                    errors.append(f"manifest URI missing target: {uri} ({exc})")
        print(f"manifest URIs: {checked - missing}/{checked} resolve")
    required = (
        WORKSPACE / "protocol/manifests/session_manifest.jsonl",
        WORKSPACE / "protocol/splits/default/splits.csv",
        WORKSPACE / "protocol/schemas/shared_session.schema.json",
        WORKSPACE / "shared/facts/sessions/artifact.json",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing required canonical path: {path}")
    for root in (WORKSPACE / "raw", WORKSPACE / "protocol", WORKSPACE / "shared",
                 WORKSPACE / "model_inputs", WORKSPACE / "work", WORKSPACE / "assets",
                 WORKSPACE / "reports"):
        for path in root.rglob("*"):
            if path.is_symlink() and not path.exists():
                errors.append(f"broken link: {path} -> {os.readlink(path)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("workspace ok: canonical tree, raw links and required artifacts are present")
    print(f"workspace={WORKSPACE}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "doctor", "resolve"))
    parser.add_argument("uri", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.dry_run)
        return 0
    if args.command == "doctor":
        return doctor()
    if not args.uri:
        parser.error("resolve requires a URI")
    print(resolve_uri(args.uri))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
