# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nemo_retriever.harness import app as harness_app
from nemo_retriever.harness.artifact_writer import ArtifactWriter
from nemo_retriever.harness.contracts import EXIT_HELM_FAILURE, HarnessRunError, RunOutcome
from nemo_retriever.harness.helm_config import load_helm_config
from nemo_retriever.harness.helm_execution import run_helm_benchmark
from nemo_retriever.harness.resolution import build_service_query_request
from nemo_retriever.harness.runfile import load_runfile
from nemo_retriever.harness.slack import build_slack_payload, load_session_report


def _write_helm_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "helm_chart: chart",
                "service_image_repository: registry.example/nrl-service",
                "service_image_tag: main-abc123",
                "helm_release: helm-test",
                "helm_namespace: helm-test",
                "keep_up: false",
            ]
        ),
        encoding="utf-8",
    )


def _patch_helm_execution(monkeypatch, tmp_path: Path, *, start_rc: int = 0, stop_rc: int = 0):
    import nemo_retriever.harness.helm_execution as helm_execution

    resolved = {
        "name": "bo20_smoke",
        "dataset": {"input_type": "pdf"},
        "ingest": {},
        "query": {},
        "evaluation": {"mode": "none"},
        "summary_keys": ["files", "pages", "ingest_secs"],
    }
    ingest_request = SimpleNamespace(
        documents=[str(tmp_path / "one.pdf")],
        input_type="pdf",
        connection=SimpleNamespace(service_url="http://localhost:17670", service_concurrency=8),
    )
    query_request = SimpleNamespace(
        retrieval=SimpleNamespace(top_k=5, candidate_k=None, page_dedup=False, content_types=None)
    )

    monkeypatch.setattr(helm_execution, "resolve_benchmark", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(helm_execution, "validate_dataset_inputs", lambda *_args, **_kwargs: (tmp_path, None))
    monkeypatch.setattr(helm_execution, "build_service_ingest_plan_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(helm_execution, "resolve_service_ingest_request", lambda _request: ingest_request)
    monkeypatch.setattr(helm_execution, "build_service_query_request", lambda *_args, **_kwargs: query_request)
    monkeypatch.setattr(
        helm_execution,
        "service_plan_payload",
        lambda *_args, **_kwargs: {"documents": ingest_request.documents, "query": {"top_k": 5}},
    )

    calls: list[tuple[str, object]] = []

    class FakeManager:
        def __init__(self, _config):
            pass

        def start(self):
            calls.append(("start", None))
            return start_rc

        def stop(self, *, uninstall=True):
            calls.append(("stop", uninstall))
            return stop_rc

        def dump_logs(self, _artifact_dir):
            calls.append(("dump_logs", None))
            return 0

    monkeypatch.setattr(helm_execution, "HelmServiceManager", FakeManager)
    monkeypatch.setattr(
        helm_execution,
        "execute_service_ingest_request",
        lambda _request: SimpleNamespace(to_summary_dict=lambda: {"n_rows": 10}),
    )
    return calls


def test_service_query_request_rejects_agentic_retrieval() -> None:
    resolved = {
        "query": {
            "top_k": 10,
            "agentic": True,
            "agentic_llm_model": "test-model",
        }
    }

    with pytest.raises(HarnessRunError) as exc_info:
        build_service_query_request(
            resolved,
            "",
            service_url="http://localhost:17670",
            service_api_token=None,
        )

    assert exc_info.value.failure.failure_reason == "invalid_benchmark_config"
    assert "not supported for Helm/service runs" in exc_info.value.failure.message


def test_helm_config_requires_and_injects_immutable_image(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    config_path.write_text("helm_chart: ./chart\n", encoding="utf-8")
    (tmp_path / "chart").mkdir()
    monkeypatch.setenv("HARNESS_HELM_SERVICE_IMAGE_REPOSITORY", "registry.example/nrl-service")
    monkeypatch.setenv("HARNESS_HELM_SERVICE_IMAGE_TAG", "main-abc123")

    config = load_helm_config(config_path)

    assert config.helm_chart == str((tmp_path / "chart").resolve())
    assert config.effective_helm_set()["service.image.repository"] == "registry.example/nrl-service"
    assert config.effective_helm_set()["service.image.tag"] == "main-abc123"


def test_runfile_loads_relative_helm_config(tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    runfile = tmp_path / "run.yaml"
    runfile.write_text(
        "schema_version: 1\nbenchmark: jp20_beir\ntarget: helm\nhelm_config: helm.yaml\n",
        encoding="utf-8",
    )

    request = load_runfile(runfile)

    assert request.target == "helm"
    assert request.helm_config == str(config_path.resolve())


def test_helm_execution_starts_and_tears_down_release(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    calls = _patch_helm_execution(monkeypatch, tmp_path)
    writer = ArtifactWriter(artifact_dir=tmp_path / "artifacts", run_id="run", benchmark="bo20_smoke")

    outcome = run_helm_benchmark(
        writer,
        "bo20_smoke",
        helm_config_path=config_path,
        overrides=(),
        requirements=(),
        dry_run=False,
        runfile_payload=None,
        runfile_path=None,
    )

    assert outcome.exit_code == 0
    assert outcome.results["target"] == "helm"
    assert calls == [("start", None), ("stop", True)]


def test_helm_readiness_failure_collects_logs_and_fails(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    calls = _patch_helm_execution(monkeypatch, tmp_path, start_rc=1)
    writer = ArtifactWriter(artifact_dir=tmp_path / "artifacts", run_id="run", benchmark="bo20_smoke")

    outcome = run_helm_benchmark(
        writer,
        "bo20_smoke",
        helm_config_path=config_path,
        overrides=(),
        requirements=(),
        dry_run=False,
        runfile_payload=None,
        runfile_path=None,
    )

    assert outcome.exit_code == EXIT_HELM_FAILURE
    assert outcome.results["failure"]["failure_reason"] == "helm_readiness_failed"
    assert calls == [("start", None), ("dump_logs", None), ("stop", True)]


def test_helm_teardown_failure_fails_successful_benchmark(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "helm.yaml"
    _write_helm_config(config_path)
    calls = _patch_helm_execution(monkeypatch, tmp_path, stop_rc=1)
    writer = ArtifactWriter(artifact_dir=tmp_path / "artifacts", run_id="run", benchmark="bo20_smoke")

    outcome = run_helm_benchmark(
        writer,
        "bo20_smoke",
        helm_config_path=config_path,
        overrides=(),
        requirements=(),
        dry_run=False,
        runfile_payload=None,
        runfile_path=None,
    )

    assert outcome.exit_code == EXIT_HELM_FAILURE
    assert outcome.results["failure"]["failure_reason"] == "helm_teardown_failed"
    assert calls == [("start", None), ("stop", True)]


def test_slack_report_reads_current_session_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "001_jp20_helm"
    run_dir.mkdir()
    (run_dir / "environment.json").write_text(json.dumps({"git_sha": "abc1234", "host": "runner"}), encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": "helm-run",
                "benchmark": "jp20_beir",
                "target": "helm",
                "success": True,
                "exit_code": 0,
                "summary_metrics": {"pages": 1940, "recall_5": 0.91},
                "deployment": {
                    "helm_chart": "/repo/nemo_retriever/helm",
                    "service_image_repository": "registry.example/nrl-service",
                    "service_image_tag": "main-abc123",
                },
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "session_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "session_name": "library_nightly_helm_jp20",
                "success": True,
                "runs": [
                    {
                        "run_name": "jp20_helm",
                        "benchmark": "jp20_beir",
                        "target": "helm",
                        "artifact_dir": str(run_dir),
                        "success": True,
                        "exit_code": 0,
                        "summary_metrics": {"pages": 1940, "recall_5": 0.91},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = load_session_report(summary_path)
    payload = build_slack_payload(
        report,
        {"title": "Helm Nightly", "metric_keys": ["pages", "recall_5"], "post_artifact_paths": True},
    )
    flattened = json.dumps(payload)

    assert report.results[0].target == "helm"
    assert "registry.example/nrl-service:main-abc123" in flattened
    assert "PASS (1/1)" in flattened


def test_nightly_dry_run_writes_current_session_summary(monkeypatch, tmp_path: Path) -> None:
    runfile = tmp_path / "run.yaml"
    runfile.write_text("schema_version: 1\nbenchmark: jp20_beir\n", encoding="utf-8")
    config = tmp_path / "nightly.yaml"
    config.write_text(
        "session_name: test_nightly\nruns:\n  - runfile: run.yaml\nslack:\n  enabled: true\n",
        encoding="utf-8",
    )

    import nemo_retriever.harness.nightly as nightly

    def fake_run(*_args, output_dir=None, **_kwargs):
        artifact_dir = Path(str(output_dir))
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "results.json").write_text("{}", encoding="utf-8")
        return RunOutcome(0, artifact_dir, {"summary_metrics": {"pages": 1940}})

    monkeypatch.setattr(nightly, "run_benchmark", fake_run)
    output_dir = tmp_path / "session"
    result = CliRunner().invoke(
        harness_app,
        ["nightly", "--runs-config", str(config), "--output-dir", str(output_dir), "--dry-run"],
    )

    assert result.exit_code == 0
    payload = json.loads((output_dir / "session_summary.json").read_text(encoding="utf-8"))
    assert payload["session_name"] == "test_nightly"
    assert payload["runs"][0]["benchmark"] == "jp20_beir"
    assert "Slack posting skipped for --dry-run." in result.stdout

def test_nightly_enabled_slack_missing_webhook_is_nonzero(monkeypatch, tmp_path: Path) -> None:
    runfile = tmp_path / "run.yaml"
    runfile.write_text("schema_version: 1\nbenchmark: jp20_beir\n", encoding="utf-8")
    config = tmp_path / "nightly.yaml"
    config.write_text(
        "session_name: test_nightly\nruns:\n  - runfile: run.yaml\nslack:\n  enabled: true\n",
        encoding="utf-8",
    )

    import nemo_retriever.harness.nightly as nightly

    def fake_run(*_args, output_dir=None, **_kwargs):
        artifact_dir = Path(str(output_dir))
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "results.json").write_text("{}", encoding="utf-8")
        return RunOutcome(0, artifact_dir, {"summary_metrics": {"pages": 1940}})

    monkeypatch.setattr(nightly, "run_benchmark", fake_run)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    result = CliRunner().invoke(
        harness_app,
        [
            "nightly",
            "--runs-config",
            str(config),
            "--output-dir",
            str(tmp_path / "session"),
        ],
    )

    assert result.exit_code == 1
    assert "SLACK_WEBHOOK_URL is not set" in result.output
