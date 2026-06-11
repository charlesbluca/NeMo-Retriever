# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import yaml

from nemo_retriever.harness.autoscaling_lab import LabConfig, Sampler, dry_run_payload, select_files, summarize_attempts


def _write_config(tmp_path: Path, dataset_dir: Path, **lab_overrides) -> Path:
    payload = {
        "lab": {
            "dataset": "tiny",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "file_limit": 2,
            "concurrency_steps": [2, 4],
            **lab_overrides,
        },
        "helm": {
            "chart": "nemo_retriever/helm",
            "release": "nrl-lab-test",
            "namespace": "nrl-lab-test",
            "values": {
                "serviceConfig": {
                    "pipeline": {
                        "realtimeWorkers": 1,
                        "realtimeQueueSize": 4,
                        "batchWorkers": 1,
                        "batchQueueSize": 4,
                    }
                }
            },
        },
        "datasets": {
            "tiny": {
                "path": str(dataset_dir),
                "input_type": "pdf",
            }
        },
    }
    path = tmp_path / "autoscaling-lab.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_lab_config_parses_disposable_yaml_and_writes_helm_values(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "a.pdf").write_bytes(b"%PDF-1.4\n")
    config_path = _write_config(tmp_path, dataset_dir)

    cfg = LabConfig.from_file(config_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    values_file = cfg.write_generated_helm_values(artifact_dir)
    values = yaml.safe_load(values_file.read_text(encoding="utf-8"))
    harness_cfg = cfg.to_harness_config(values_file)

    assert cfg.dataset_label == "tiny"
    assert cfg.concurrency_steps == (2, 4)
    assert cfg.file_limit == 2
    assert values["topology"]["mode"] == "split"
    assert values["topology"]["batch"]["gpu"] == {"enabled": False, "count": 0}
    assert values["serviceConfig"]["pipeline"]["realtimeQueueSize"] == 4
    assert harness_cfg.run_mode == "service"
    assert harness_cfg.manage_service is True
    assert harness_cfg.helm_values_file == str(values_file)


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


def test_dry_run_payload_lists_selected_files_and_helm_values(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (dataset_dir / name).write_bytes(b"%PDF-1.4\n")
    config_path = _write_config(tmp_path, dataset_dir)

    cfg = LabConfig.from_file(config_path)
    files = select_files(cfg)
    payload = dry_run_payload(cfg, tmp_path / "artifacts" / "run", files)

    assert len(files) == 2
    assert payload["selected_file_count"] == 2
    assert payload["concurrency_steps"] == [2, 4]
    assert payload["helm_values"]["serviceConfig"]["pipeline"]["batchWorkers"] == 1
    json.dumps(payload)


def test_sampler_prepares_artifact_outputs_before_start(tmp_path: Path) -> None:
    Sampler(
        manager=object(),  # type: ignore[arg-type]
        artifact_dir=tmp_path,
        gateway_url="http://localhost:1",
        worker_urls={},
        interval_s=1.0,
    )

    assert (tmp_path / "kubectl").is_dir()
    assert (tmp_path / "http").is_dir()
    assert (tmp_path / "samples.csv").read_text(encoding="utf-8").startswith("timestamp,index,kind")
