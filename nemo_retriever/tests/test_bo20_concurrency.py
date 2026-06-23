# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict

import pytest

from nemo_retriever.harness.bo20_concurrency import (
    Bo20Inventory,
    JobRunResult,
    SweepRoundResult,
    apply_latency_saturation,
    build_split_harness_config,
    render_clean_recovery_report,
    inventory_bo20_dataset,
    render_markdown_report,
    render_ux_probe_report,
    resolve_bo20_files,
    summarize_thresholds,
    summarize_ux_probe_samples,
    classify_failure_attribution,
    _effective_return_results_modes,
    _helm_set_for_nim_backend,
    _read_env_value,
    _parse_helm_set,
    _resolve_cli_max_n,
    _return_results_modes_from_option,
    _sweep_n_values,
    _ux_target_n,
)


def test_nvcf_backend_forces_hosted_nim_overrides():
    helm_set = _helm_set_for_nim_backend(
        {"nims.enabled": True, "service.image.tag": "test"},
        "nvcf",
    )

    assert helm_set["service.image.tag"] == "test"
    assert helm_set["nims.enabled"] is False
    assert helm_set["nimOperator.page_elements.enabled"] is False
    assert helm_set["serviceConfig.nimEndpoints.pageElementsInvokeUrl"].startswith(
        "https://ai.api.nvidia.com/"
    )
    assert helm_set["serviceConfig.nimEndpoints.tableStructureInvokeUrl"].endswith(
        "nemotron-table-structure-v1"
    )


def test_proxy_dry_run_backend_disables_local_nims_and_forces_false_mode():
    helm_set = _helm_set_for_nim_backend({"topology.mode": "split"}, "proxy_dry_run")

    assert helm_set["nims.enabled"] is False
    assert helm_set["nimOperator.ocr.enabled"] is False
    assert _effective_return_results_modes("proxy-dry-run", (False, True)) == (False,)


def test_parse_helm_set_accepts_json_values():
    parsed = _parse_helm_set([
        'nimOperator.page_elements.env=[{"name":"NIM_ENGINE_MODEL_NAME","value":"nvidia/nemotron-page-elements-v3"}]'
    ])

    assert parsed["nimOperator.page_elements.env"] == [
        {"name": "NIM_ENGINE_MODEL_NAME", "value": "nvidia/nemotron-page-elements-v3"}
    ]


def test_read_env_value_handles_export_and_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# ignored\nexport NGC_NV_DEVELOPER_NVCF='nvapi-secret'\n", encoding="utf-8")

    assert _read_env_value(env_file, "NGC_NV_DEVELOPER_NVCF") == "nvapi-secret"
    assert _read_env_value(env_file, "MISSING") is None


def test_nvcf_sweep_sequence_starts_at_known_cliffs():
    assert _sweep_n_values(nim_backend="nvcf", return_results=False, max_n=10) == [1, 7, 8, 9, 10]
    assert _sweep_n_values(nim_backend="nvcf", return_results=True, max_n=13) == [1, 10, 11, 12, 13]
    assert _sweep_n_values(nim_backend="local", return_results=False, max_n=3) == [1, 2, 3]


def test_failure_attribution_distinguishes_nim_and_result_fetch():
    local = classify_failure_attribution(
        [
            {
                "failures": [
                    [
                        "doc",
                        "GraphIngestionError: NIM endpoint http://nemotron-table-structure-v1:8000/v1/infer returned 500",
                    ]
                ]
            }
        ],
        ["1 service failure(s)"],
        {"retry_429_count": 0},
        {},
    )
    result_fetch = classify_failure_attribution(
        [{"failures": [["doc", "return_results: failed to fetch/persist abc: timed out"]]}],
        ["1 service failure(s)"],
        {"retry_429_count": 0},
        {},
    )

    assert local["primary"] == "local NIM"
    assert result_fetch["primary"] == "result-fetch/row materialization"



def test_resolve_bo20_files_requires_exact_pdf_count(tmp_path):
    dataset = tmp_path / "bo20"
    dataset.mkdir()
    for idx in range(19):
        (dataset / f"{idx:02d}.pdf").write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ValueError, match="Expected exactly 20 bo20 PDFs"):
        resolve_bo20_files(dataset)

    (dataset / "19.pdf").write_bytes(b"%PDF-1.4\n")

    assert len(resolve_bo20_files(dataset)) == 20


def test_inventory_counts_pages(monkeypatch, tmp_path):
    dataset = tmp_path / "bo20"
    dataset.mkdir()
    for idx in range(20):
        (dataset / f"{idx:02d}.pdf").write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("nemo_retriever.harness.bo20_concurrency._safe_pdf_page_count", lambda _path: 3)

    inventory = inventory_bo20_dataset(dataset)

    assert inventory.pdf_count == 20
    assert inventory.total_pages == 60
    assert inventory.page_counts["00.pdf"] == 3


def test_build_split_harness_config_forces_split(tmp_path):
    cfg = build_split_harness_config(
        dataset_dir=str(tmp_path),
        artifacts_dir=None,
        helm_release="rel",
        helm_namespace=None,
        helm_chart=None,
        helm_chart_version=None,
        helm_set={"topology.mode": "standalone", "service.image.tag": "test"},
        helm_timeout=123,
        readiness_timeout=456,
        helm_service_local_port=17670,
        keep_up=True,
        helm_bin="helm",
        kubectl_bin="kubectl",
        helm_sudo=False,
        kubectl_sudo=False,
        api_token="sekret",
    )

    assert cfg.run_mode == "service"
    assert cfg.manage_service is True
    assert cfg.keep_up is True
    assert cfg.api_key == "sekret"
    assert cfg.helm_set["topology.mode"] == "split"
    assert cfg.helm_set["service.image.tag"] == "test"


def test_summarize_thresholds_includes_confirmation_failures():
    rounds = [
        {"return_results": False, "n": 1, "attempt": "primary", "saturation": False, "hard_failure": False},
        {"return_results": False, "n": 2, "attempt": "primary", "saturation": True, "hard_failure": False},
        {
            "return_results": False,
            "n": 3,
            "attempt": "primary",
            "saturation": True,
            "hard_failure": True,
            "hard_failure_reasons": ["pod restarted during run"],
        },
        {"return_results": False, "n": 2, "attempt": "confirm", "saturation": True, "hard_failure": True},
        {"return_results": True, "n": 1, "attempt": "primary", "saturation": False, "hard_failure": False},
    ]

    summary = summarize_thresholds(rounds)

    assert summary["False"]["first_saturation_n"] == 2
    assert summary["False"]["first_hard_failure_n"] == 2
    assert summary["False"]["first_hard_failure_attempt"] == "confirm"
    assert summary["False"]["primary_first_hard_failure_n"] == 3
    assert summary["True"]["first_saturation_n"] is None
    assert summary["True"]["first_hard_failure_n"] is None


def test_render_markdown_report_states_no_hard_failure_phrase():
    payload = {
        "timestamp": "2026-06-15T00:00:00Z",
        "latest_commit": "abc1234",
        "inventory": asdict(
            Bo20Inventory(
                dataset_dir="/localhome/charlesb/datasets/bo20",
                pdf_count=20,
                total_pages=40,
                page_counts={},
            )
        ),
        "config": {"max_n": 16},
        "thresholds": {
            "False": {"first_saturation_n": None, "first_hard_failure_n": None},
            "True": {"first_saturation_n": None, "first_hard_failure_n": None},
        },
        "rounds": [
            {
                "return_results": False,
                "n": 16,
                "attempt": "primary",
                "hard_failure": False,
                "saturation": False,
                "metrics": {
                    "completed_jobs": 16,
                    "retry_429_count": 0,
                    "job_latency_p95_s": 10.0,
                    "pages_per_sec_wall": 1.0,
                },
            }
        ],
        "artifact_paths": {"json": "/tmp/results.json", "markdown": "/tmp/report.md"},
    }

    report = render_markdown_report(payload)

    assert "No hard failure observed up to 16 simultaneous bo20 jobs" in report


def test_apply_latency_saturation_marks_three_x_p95():
    rounds = [
        SweepRoundResult(
            return_results=False,
            n=1,
            attempt="primary",
            started_at="",
            finished_at="",
            wall_s=10.0,
            success=True,
            hard_failure=False,
            hard_failure_reasons=[],
            saturation=False,
            saturation_reasons=[],
            job_results=[],
            metrics={"job_latency_p95_s": 10.0},
            cluster_before={},
            cluster_after={},
            cluster_delta={},
            samples=[],
            idle_after_run=True,
            idle_wait_s=0.0,
        ),
        SweepRoundResult(
            return_results=False,
            n=2,
            attempt="primary",
            started_at="",
            finished_at="",
            wall_s=31.0,
            success=True,
            hard_failure=False,
            hard_failure_reasons=[],
            saturation=False,
            saturation_reasons=[],
            job_results=[],
            metrics={"job_latency_p95_s": 30.0},
            cluster_before={},
            cluster_after={},
            cluster_delta={},
            samples=[],
            idle_after_run=True,
            idle_wait_s=0.0,
        ),
    ]

    apply_latency_saturation(rounds)

    assert rounds[1].saturation is True
    assert "3.0x single-job baseline" in rounds[1].saturation_reasons[0]


def test_missing_job_status_is_inferred_from_complete_counts():
    from nemo_retriever.harness.bo20_concurrency import (
        _infer_terminal_status,
        _job_hard_failure_reasons,
    )

    job = JobRunResult(
        job_index=0,
        job_id="job-1",
        completed=20,
        failed=0,
        uploaded=20,
        upload_failed=0,
        exit_code=0,
    )

    _infer_terminal_status(job, expected_documents=20)

    assert job.job_status == "completed"
    assert _job_hard_failure_reasons(job, expected_documents=20) == []



def test_resolve_cli_max_n_defaults_clean_mode_to_three_but_honors_explicit_value():
    assert _resolve_cli_max_n(None, clean_page_elements_rerun=False) == 16
    assert _resolve_cli_max_n(None, clean_page_elements_rerun=True) == 3
    assert _resolve_cli_max_n(16, clean_page_elements_rerun=True) == 16



def test_return_results_modes_from_option_accepts_single_modes_and_both():
    assert _return_results_modes_from_option("both") == (False, True)
    assert _return_results_modes_from_option("false") == (False,)
    assert _return_results_modes_from_option("true") == (True,)

    with pytest.raises(ValueError, match="return results mode"):
        _return_results_modes_from_option("maybe")



def test_clean_recovery_helm_set_applies_defaults_with_overrides():
    from nemo_retriever.harness.bo20_concurrency import _clean_recovery_helm_set

    helm_set = _clean_recovery_helm_set({"persistence.enabled": True, "service.image.tag": "local"})

    assert helm_set["serviceMonitor.autoEnableInSplitMode"] is False
    assert helm_set["autoscaling.queueDepth.backend"] == "cpu"
    assert helm_set["topology.batch.gpu.enabled"] is False
    assert helm_set["persistence.enabled"] is True
    assert helm_set["retrieverResults.enabled"] is False
    assert helm_set["service.image.tag"] == "local"


def test_render_clean_recovery_report_includes_phase_logs():
    payload = {
        "timestamp": "2026-06-15T00:00:00Z",
        "latest_commit": "abc1234",
        "inventory": {"dataset_dir": "/localhome/charlesb/datasets/bo20"},
        "phases": [
            {
                "name": "phase_a_return_results_false",
                "return_results": False,
                "thresholds": {"False": {"first_hard_failure_n": 3}},
                "health_smoke": {
                    "attempt": "health_smoke",
                    "return_results": False,
                    "n": 1,
                    "hard_failure": False,
                    "metrics": {"completed_jobs": 1, "documents_completed": 20, "documents_failed": 0},
                },
                "rounds": [],
                "page_elements_evidence": {"page_elements_failure_observed": True},
                "service_logs": {"path": "/tmp/phase_a/service_logs"},
            }
        ],
        "interpretation": "clean redeploy reproduced page-elements failure",
        "artifact_paths": {"json": "/tmp/results.json", "markdown": "/tmp/report.md"},
    }

    report = render_clean_recovery_report(payload)

    assert "Clean Page-Elements Recovery Rerun" in report
    assert "Phase A" in report
    assert "page-elements failure" in report
    assert "/tmp/phase_a/service_logs" in report



def test_ux_target_n_uses_known_nvcf_failure_scales():
    assert _ux_target_n(False, false_n=6, true_n=8) == 6
    assert _ux_target_n(True, false_n=6, true_n=8) == 8


def test_summarize_ux_probe_samples_counts_endpoint_health():
    samples = [
        {
            "known_documents": 2,
            "known_completed_documents": 1,
            "known_failed_documents": 0,
            "result_fetch_success": 1,
            "result_fetch_failed": 0,
            "health": {"ok": True, "elapsed_ms": 10.0},
            "batch_status": {"ok": True, "elapsed_ms": 20.0},
            "sse": {"ok": False, "elapsed_ms": 30.0, "error": "ReadTimeout"},
        },
        {
            "known_documents": 4,
            "known_completed_documents": 3,
            "known_failed_documents": 1,
            "result_fetch_success": 2,
            "result_fetch_failed": 1,
            "health": {"ok": True, "elapsed_ms": 15.0},
            "batch_status": {"ok": False, "elapsed_ms": 25.0, "status_code": 503},
            "sse": {"ok": True, "elapsed_ms": 35.0},
        },
    ]

    summary = summarize_ux_probe_samples(samples)

    assert summary["health"]["ok"] == 2
    assert summary["batch_status"]["failed"] == 1
    assert summary["sse"]["ok"] == 1
    assert summary["max_known_documents"] == 4
    assert summary["result_fetch_failed_max"] == 1


def test_render_ux_probe_report_includes_user_visible_probe_columns():
    payload = {
        "timestamp": "2026-06-22T00:00:00Z",
        "latest_commit": "abc1234",
        "inventory": {"dataset_dir": "/localhome/charlesb/datasets/bo20"},
        "phases": [
            {
                "return_results": True,
                "round": {
                    "n": 8,
                    "hard_failure": True,
                    "hard_failure_reasons": ["1 service failure(s)"],
                    "idle_after_run": True,
                    "metrics": {
                        "failure_attribution": "result-fetch/row materialization",
                        "completed_jobs": 7,
                        "documents_completed": 160,
                        "documents_failed": 0,
                        "result_fetch_attempts": 160,
                        "result_fetch_success": 150,
                        "result_fetch_failed": 10,
                        "ux_probe_summary": {
                            "health": {"ok": 3, "failed": 0, "p95_ms": 12.0},
                            "batch_status": {"ok": 3, "failed": 0, "p95_ms": 22.0},
                            "job_aggregate": {"ok": 3, "failed": 0},
                            "sse": {"ok": 2, "failed": 1},
                        },
                    },
                    "cluster_delta": {"restart_delta_by_component": {"gateway": 0}, "oom_events_after": []},
                },
                "endpoint_evidence": {"pattern_counts": {"ai.api.nvidia.com": 10, "returned 429": 2}},
                "service_logs": {"path": "/tmp/logs"},
            }
        ],
        "interpretation": "result fetch failed while status stayed up",
        "artifact_paths": {"json": "/tmp/results.json", "markdown": "/tmp/report.md"},
    }

    report = render_ux_probe_report(payload)

    assert "bo20 User-Experience Failure Probe" in report
    assert "batch_status" in report
    assert "result_fetch_failed" in report
    assert "result fetch failed" in report
