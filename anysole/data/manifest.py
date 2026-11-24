"""会话 manifest 读取（仅依赖 AnysoleWorkspace.tool.workspace）。"""
from __future__ import annotations

import json

from AnysoleWorkspace.tool.workspace import resolve_uri


def load_manifest_rows(cam: int) -> dict:
    """读 session manifest jsonl → {session_id: row}。

    per-camera manifest（session_manifest_cam{N}.jsonl）在 2026-10-24 清洗中
    收敛为统一 session_manifest.jsonl（cam3 文件已删）；缺 per-cam 文件时
    回退统一 manifest。
    """
    path = resolve_uri(f"protocol://manifests/session_manifest_cam{cam}.jsonl",
                       for_write=False, must_exist=False)
    if not path.is_file():
        path = resolve_uri("protocol://manifests/session_manifest.jsonl",
                           for_write=False, must_exist=True)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["session_id"]] = row
    return rows
