# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Per-document result rows for split-topology workers.

Worker → gateway completion callbacks intentionally omit ``result_data``
to keep POST bodies small. When ``NEMO_RETRIEVER_RESULTS_DIR`` is set, rows
are written atomically to that shared directory so any gateway or worker pod
can consume them by document ID. The in-memory store remains as a fallback
for local and non-Helm deployments that do not configure shared storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_store: dict[str, list[dict[str, Any]]] = {}
_RESULTS_DIR_ENV = "NEMO_RETRIEVER_RESULTS_DIR"


def _results_dir() -> Path | None:
    value = os.environ.get(_RESULTS_DIR_ENV, "").strip()
    return Path(value) if value else None


def _result_path(results_dir: Path, document_id: str) -> Path:
    """Return a traversal-safe, deterministic path for *document_id*."""
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return results_dir / f"{digest}.json"


def _store_on_filesystem(results_dir: Path, document_id: str, result_data: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = _result_path(results_dir, document_id)
    temporary = results_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result_data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _consume_from_filesystem(results_dir: Path, document_id: str) -> list[dict[str, Any]] | None:
    target = _result_path(results_dir, document_id)
    claimed = results_dir / f".{target.name}.{uuid.uuid4().hex}.claim"
    try:
        os.replace(target, claimed)
    except FileNotFoundError:
        return None

    try:
        with claimed.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError(f"Shared result payload for {document_id!r} is not a JSON list")
        return rows
    finally:
        claimed.unlink(missing_ok=True)


def store_result_data(document_id: str, result_data: list[dict[str, Any]] | None) -> None:
    """Retain *result_data* for a completed document."""
    if not document_id or not result_data:
        return
    if results_dir := _results_dir():
        _store_on_filesystem(results_dir, document_id, result_data)
        return
    with _lock:
        _store[document_id] = result_data


def consume_result_data(document_id: str) -> list[dict[str, Any]] | None:
    """Return stored rows for *document_id* and remove them from the store."""
    if results_dir := _results_dir():
        return _consume_from_filesystem(results_dir, document_id)
    with _lock:
        return _store.pop(document_id, None)


def clear_for_tests() -> None:
    """Test helper — drop all cached rows."""
    with _lock:
        _store.clear()
