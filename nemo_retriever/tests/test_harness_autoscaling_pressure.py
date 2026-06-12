# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_retriever.harness.autoscaling_pressure import (
    AutoscalingSampler,
    resolve_workloads,
    select_pressure_files,
    summarize_attempts,
)
from nemo_retriever.harness.config import HarnessConfig


def test_resolve_workloads_uses_default_pressure_shapes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="tiny",
        preset="base",
        run_mode="service",
        run_type="autoscaling_pressure",
        manage_service=True,
    )

    workloads = resolve_workloads(cfg)

    assert [w.name for w in workloads] == ["realtime_burst", "batch_burst", "mixed_burst"]
    assert [w.concurrency for w in workloads] == [32, 32, 64]
    assert [w.kind for w in workloads] == ["realtime_page", "batch_document", "mixed"]


def test_select_pressure_files_respects_file_limit(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (dataset_dir / name).write_bytes(b"%PDF-1.4\n")
    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="tiny",
        preset="base",
        input_type="pdf",
        run_mode="service",
        run_type="autoscaling_pressure",
        manage_service=True,
        autoscaling_file_limit=2,
    )

    files = select_pressure_files(cfg)

    assert [path.name for path in files] == ["a.pdf", "b.pdf"]


def test_summarize_attempts_keeps_retry_429s_visible() -> None:
    attempts = [
        {"file": "a.pdf", "status_code": 429, "terminal": False},
        {"file": "a.pdf", "status_code": 202, "terminal": True},
        {"file": "b.pdf", "status_code": 429, "terminal": True},
        {"file": "c.pdf", "status_code": None, "terminal": True},
    ]

    summary = summarize_attempts(attempts)

    assert summary["attempts_total"] == 4
    assert summary["files_seen"] == 3
    assert summary["files_terminal"] == 3
    assert summary["retryable_429s"] == 1
    assert summary["terminal_successes"] == 1
    assert summary["terminal_failures"] == 2
    assert summary["status_counts"] == {"429": 2, "202": 1, "transport_error": 1}
    assert summary["terminal_status_counts"] == {"202": 1, "429": 1, "transport_error": 1}


def test_sampler_prepares_artifact_outputs_before_start(tmp_path: Path) -> None:
    AutoscalingSampler(
        manager=object(),  # type: ignore[arg-type]
        artifact_dir=tmp_path,
        gateway_url="http://localhost:1",
        worker_urls={},
        interval_s=1.0,
    )

    assert (tmp_path / "kubectl").is_dir()
    assert (tmp_path / "http").is_dir()
    assert (tmp_path / "autoscaling_samples.csv").read_text(encoding="utf-8").startswith("timestamp,index,kind")
