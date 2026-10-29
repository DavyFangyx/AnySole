"""Read the versioned, immutable F6 time/acceleration correction package."""
import hashlib
import json
from pathlib import Path

import numpy as np

CORRECTED_CONTACT_METHOD = "f6_soft_time_accel_v1"
PACKAGE_SHA256 = "cc1e76297f6e55fab1e2b704b10b8228f3964c31999cb7ba74f0e1f6bdfe3df3"


STATE_CONTACT_METHOD = "f6_soft_time_accel_state_v2"
STATE_PACKAGE_SHA256 = "6c76f4b963a7395b099b5b6a8444c6be21f217cc4c19eee49499ced05cff24d8"
SMPL24_CORRECTED_CONTACT_METHOD = "f6_soft_time_accel_smpl24_v1"
SMPL24_STATE_CONTACT_METHOD = "f6_soft_time_accel_state_smpl24_v1"
SMPL24_PACKAGE_SHA256 = "998b5ecf9cbc2f948ef86c9aa394aac1f010d17e3d124c5d44742215a3c99554"
SMPL24_STATE_PACKAGE_SHA256 = "24e2092c077c178380b99b602a06d6b463c93793bbd5c213094464e693c74453"
SMPL24_CONTACT_METHODS = (SMPL24_CORRECTED_CONTACT_METHOD, SMPL24_STATE_CONTACT_METHOD)
LEGACY_VERSIONED_CONTACT_METHODS = (CORRECTED_CONTACT_METHOD, STATE_CONTACT_METHOD)
VERSIONED_CONTACT_METHODS = LEGACY_VERSIONED_CONTACT_METHODS + SMPL24_CONTACT_METHODS


def expected_package_sha256(method):
    if method == CORRECTED_CONTACT_METHOD:
        return PACKAGE_SHA256
    if method == STATE_CONTACT_METHOD:
        return STATE_PACKAGE_SHA256
    if method == SMPL24_CORRECTED_CONTACT_METHOD:
        return SMPL24_PACKAGE_SHA256
    if method == SMPL24_STATE_CONTACT_METHOD:
        return SMPL24_STATE_PACKAGE_SHA256
    raise ContactPackageError("Unknown versioned contact method: " + str(method))


class ContactPackageError(ValueError):
    """A pinned package or its source has changed; never rebuild silently."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContactLabelPackage:
    def __init__(self, root, method=None):
        self.root = Path(root).resolve()
        self.method = method or (self.root.name if self.root.name in VERSIONED_CONTACT_METHODS else CORRECTED_CONTACT_METHOD)
        self.expected_sha256 = expected_package_sha256(self.method)
        path = self.root / "label_package.json"
        try:
            if sha256(path) != self.expected_sha256:
                raise ContactPackageError("Corrected contact manifest hash mismatch: " + str(path))
            self.manifest = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ContactPackageError("Cannot load pinned corrected contact package: " + str(exc)) from exc
        schema = self.manifest.get("schema")
        legacy = schema == "contact_label_package.v1" and self.manifest.get("method") == "f6_soft"
        self.pinned_motion_sources = (schema == "contact_label_package.v2" and
            self.method in SMPL24_CONTACT_METHODS and self.manifest.get("method") == self.method)
        if not (legacy or self.pinned_motion_sources):
            raise ContactPackageError("Unsupported corrected contact package")

    def load(self, session):
        sid = session["session_id"]
        row = self.manifest["sessions"].get(sid)
        if row is None:
            raise ContactPackageError("Session missing from corrected contact package: " + sid)
        for name, expected in row["source_files"].items():
            path = Path(session["dir"]) / name
            if not path.is_file() or sha256(path) != expected:
                raise ContactPackageError("%s changed dependency %s" % (sid, name))
        bvh = Path(row["bvh_path"])
        if not bvh.is_file() or sha256(bvh) != row["bvh_sha256"]:
            raise ContactPackageError(sid + " changed BVH dependency")
        if self.pinned_motion_sources:
            from anysole.data.workspace_adapter import shared_bvh_path, shared_smpl_path
            for kind, getter in (("bvh", shared_bvh_path), ("smpl", shared_smpl_path)):
                try:
                    active = getter(session)
                    correct_target = str(active.resolve()) == row[kind + "_resolved_path"]
                    correct_bytes = sha256(active) == row[kind + "_sha256"]
                except (OSError, ValueError, KeyError) as exc:
                    raise ContactPackageError(sid + " cannot validate active " + kind.upper() + " source") from exc
                if not correct_target or not correct_bytes:
                    raise ContactPackageError(sid + " changed active " + kind.upper() + " dependency")
        target = (self.root / row["label_file"]).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ContactPackageError("Label path escapes corrected package") from exc
        if not target.is_file() or sha256(target) != row["label_sha256"]:
            raise ContactPackageError(sid + " label missing or modified")
        with np.load(target, allow_pickle=False) as data:
            contact = np.asarray(data["contact"], dtype=np.float32)
            expected = np.asarray(session["frames"]["frame_id"])
            if contact.shape != (len(expected), 2) or not np.array_equal(data["frame_id"], expected):
                raise ContactPackageError(sid + " label/frame axis mismatch")
            if not np.isin(contact, [0, 1]).all():
                raise ContactPackageError(sid + " non-binary contact label")
            return contact.copy()


def corrected_contact_provenance(method=CORRECTED_CONTACT_METHOD):
    from anysole.data.workspace_adapter import LABEL_ROOT

    root = LABEL_ROOT / method
    ContactLabelPackage(root, method=method)
    state = method in (STATE_CONTACT_METHOD, SMPL24_STATE_CONTACT_METHOD)
    result = {"contact_label_package": str(root),
              "contact_label_package_sha256": expected_package_sha256(method),
              "contact_label_definition": ("current mocap_time_s; second difference; inherited high-unloaded stance guard; F6 binary proxy" if state else "current mocap_time_s; trailing second difference; F6 binary proxy")}
    if method in SMPL24_CONTACT_METHODS:
        result["contact_label_definition"] = ("inherited frame query; second difference; high-unloaded stance guard; F6 binary proxy" if state else "inherited frame query; second difference; F6 binary proxy")
    if method in SMPL24_CONTACT_METHODS:
        result.update(contact_label_schema="contact_label_package.v2",
                      contact_motion_source_policy="pinned active SMPL and BVH content and targets; inherited frame grid")
    return result
