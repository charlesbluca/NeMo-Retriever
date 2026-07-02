# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from nemo_retriever.cli.shared import silence_noisy_libraries
from nemo_retriever.harness.artifact_writer import artifact_paths, ArtifactWriter, capture_output_to_log, redact
from nemo_retriever.harness.beir_runner import run_service_beir_queries
from nemo_retriever.harness.contracts import (
    EXIT_HELM_FAILURE,
    EXIT_INGEST_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    PHASE_VALUES,
    RunOutcome,
)
from nemo_retriever.harness.environment import collect_environment
from nemo_retriever.harness.helm_config import load_helm_config
from nemo_retriever.harness.helm_manager import HelmServiceManager
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.harness.metric_gates import enforce_metric_gates, parse_metric_gates
from nemo_retriever.harness.metrics import build_summary_metrics
from nemo_retriever.harness.resolution import (
    build_service_ingest_plan_request,
    build_service_query_request,
    service_plan_payload,
    resolve_benchmark,
    validate_dataset_inputs,
)
from nemo_retriever.ingest.service import execute_service_ingest_request, resolve_service_ingest_request


def _result_payload(
    writer: ArtifactWriter,
    *,
    status: str,
    success: bool,
    exit_code: int,
    dry_run: bool,
    resolved: dict[str, Any] | None,
    summary_metrics: dict[str, Any],
    failure: FailurePayload | None,
    deployment: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "run_id": writer.run_id,
        "benchmark": writer.benchmark,
        "target": "helm",
        "status": status,
        "success": success,
        "exit_code": exit_code,
        "dry_run": bool(dry_run),
        "resolved_benchmark": resolved,
        "deployment": deployment,
        "summary_metrics": summary_metrics,
        "failure": failure.to_dict() if failure is not None else None,
        "artifacts": artifact_paths(writer),
    }
    payload.update(extra)
    return payload


def _write_failure(
    writer: ArtifactWriter,
    *,
    failure: FailurePayload,
    exit_code: int,
    dry_run: bool,
    resolved: dict[str, Any] | None,
    summary_metrics: dict[str, Any] | None,
    deployment: dict[str, Any] | None,
    cleanup_error: str | None = None,
) -> RunOutcome:
    metrics = summary_metrics or {}
    try:
        write_json(writer.path("summary_metrics.json"), metrics)
    except Exception:
        pass
    status_payload = writer.status(
        status="failed",
        phase=failure.failed_phase if failure.failed_phase in PHASE_VALUES else "write_artifacts",
        failure=failure,
        summary_metrics_path=writer.path("summary_metrics.json"),
    )
    result = _result_payload(
        writer,
        status="failed",
        success=False,
        exit_code=exit_code,
        dry_run=dry_run,
        resolved=resolved,
        summary_metrics=metrics,
        failure=failure,
        deployment=deployment,
        status_payload=status_payload,
        cleanup_error=cleanup_error,
    )
    write_json(writer.path("results.json"), result)
    return RunOutcome(exit_code=exit_code, artifact_dir=writer.artifact_dir, results=result)


def _helm_failure(phase: str, reason: str, message: str) -> HarnessRunError:
    return HarnessRunError(
        EXIT_HELM_FAILURE,
        FailurePayload(
            failed_phase=phase,
            failure_reason=reason,
            retryable=True,
            message=message,
            debug_artifacts=("run.log", "service_logs"),
        ),
    )


def _cleanup_manager(
    manager: HelmServiceManager | None,
    writer: ArtifactWriter,
    *,
    keep_up: bool,
    collect_logs: bool,
) -> str | None:
    if manager is None:
        return None
    if collect_logs:
        try:
            manager.dump_logs(writer.artifact_dir)
        except Exception as exc:
            writer.event("teardown", "log_collection_failed", str(exc))
    try:
        writer.status(status="running", phase="teardown")
        rc = manager.stop(uninstall=not keep_up)
        if rc != 0:
            return f"Helm teardown failed with exit code {rc}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def run_helm_benchmark(
    writer: ArtifactWriter,
    benchmark: str,
    *,
    helm_config_path: Path | None,
    overrides: Sequence[str],
    requirements: Sequence[str],
    dry_run: bool,
    runfile_payload: dict[str, Any] | None,
    runfile_path: str | None,
) -> RunOutcome:
    silence_noisy_libraries()
    resolved: dict[str, Any] | None = None
    summary_metrics: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    manager: HelmServiceManager | None = None
    config = None
    try:
        writer.status(status="planned", phase="resolve")
        if helm_config_path is None:
            raise HarnessRunError(
                EXIT_INVALID,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="invalid_helm_config",
                    retryable=False,
                    message="--helm-config is required when --target helm is selected.",
                ),
            )
        try:
            config = load_helm_config(helm_config_path)
        except ValueError as exc:
            raise HarnessRunError(
                EXIT_INVALID,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="invalid_helm_config",
                    retryable=False,
                    message=str(exc),
                ),
            ) from exc
        deployment = redact(config.public_payload())
        parse_metric_gates(requirements)
        resolved = resolve_benchmark(benchmark, mode="local", overrides=overrides)
        resolved["target"] = "helm"
        resolved["deployment"] = deployment
        dataset_path, _query_path = validate_dataset_inputs(resolved, dry_run=dry_run)
        write_json(writer.path("environment.json"), collect_environment())
        if runfile_payload is not None:
            write_json(writer.path("runfile.json"), {"source_path": runfile_path, "payload": runfile_payload})

        service_url = f"http://localhost:{config.helm_service_local_port}"
        writer.status(status="running", phase="ingest_plan")
        plan_request = build_service_ingest_plan_request(
            resolved,
            dataset_path,
            service_url=service_url,
            service_concurrency=config.service_max_concurrency,
            service_api_token=config.service_api_token,
        )
        ingest_request = resolve_service_ingest_request(plan_request)
        writer.status(status="running", phase="query_plan")
        query_request = build_service_query_request(
            resolved,
            "",
            service_url=service_url,
            service_api_token=config.service_api_token,
        )
        plan_payload = service_plan_payload(ingest_request, query_request)
        write_json(writer.path("ingest_plan.json"), plan_payload)
        write_json(writer.path("query_plan.json"), plan_payload["query"])
        write_json(writer.path("resolved_benchmark.json"), redact(resolved))

        if dry_run:
            summary_metrics = build_summary_metrics(resolved, documents=ingest_request.documents)
            for key in list(summary_metrics):
                if key not in {"files", "pages"}:
                    summary_metrics[key] = None
            skipped = enforce_metric_gates(summary_metrics, requirements, skip_missing=True)
            write_json(writer.path("summary_metrics.json"), summary_metrics)
            result = _result_payload(
                writer,
                status="complete",
                success=True,
                exit_code=EXIT_SUCCESS,
                dry_run=True,
                resolved=resolved,
                summary_metrics=summary_metrics,
                failure=None,
                deployment=deployment,
                metric_gates=list(requirements),
                skipped_metric_gates=list(skipped),
            )
            write_json(writer.path("results.json"), result)
            writer.status(
                status="complete",
                phase="write_artifacts",
                summary_metrics_path=writer.path("summary_metrics.json"),
            )
            return RunOutcome(EXIT_SUCCESS, writer.artifact_dir, result)

        manager = HelmServiceManager(config)
        writer.status(status="running", phase="deploy")
        writer.event("deploy", "helm_start", "Installing or upgrading the managed Helm release")
        if manager.start() != 0:
            raise _helm_failure("readiness", "helm_readiness_failed", "Managed Helm service failed to become ready.")

        writer.status(status="running", phase="ingest")
        writer.event("ingest", "ingest_start", f"Ingesting {len(ingest_request.documents)} document(s) through Helm")
        ingest_start = time.perf_counter()
        try:
            with capture_output_to_log(writer.path("run.log"), label="helm_service_ingest"):
                ingest_result = execute_service_ingest_request(ingest_request)
        except Exception as exc:
            raise HarnessRunError(
                EXIT_INGEST_FAILURE,
                FailurePayload(
                    failed_phase="ingest",
                    failure_reason="ingest_failed",
                    retryable=False,
                    message=str(exc),
                    debug_artifacts=("ingest_plan.json", "run.log", "service_logs"),
                ),
            ) from exc
        ingest_secs = round(time.perf_counter() - ingest_start, 3)
        ingest_summary = ingest_result.to_summary_dict()

        query_latencies_ms: list[float] = []
        beir_metrics: dict[str, float] = {}
        query_count = 0
        if (resolved.get("evaluation") or {}).get("mode") == "beir":
            with capture_output_to_log(writer.path("run.log"), label="helm_service_query_evaluate"):
                query_latencies_ms, beir_metrics, query_count = run_service_beir_queries(
                    writer,
                    resolved,
                    query_request,
                )

        summary_metrics = build_summary_metrics(
            resolved,
            documents=ingest_request.documents,
            ingest_summary=ingest_summary,
            ingest_secs=ingest_secs,
            query_latencies_ms=query_latencies_ms,
            beir_metrics=beir_metrics,
        )
        if query_count and summary_metrics.get("query_count") == 0:
            summary_metrics["query_count"] = query_count
        enforce_metric_gates(summary_metrics, requirements, skip_missing=False)

        cleanup_error = _cleanup_manager(manager, writer, keep_up=config.keep_up, collect_logs=False)
        manager = None
        if cleanup_error:
            raise _helm_failure("teardown", "helm_teardown_failed", cleanup_error)

        writer.status(status="running", phase="write_artifacts")
        write_json(writer.path("summary_metrics.json"), summary_metrics)
        result = _result_payload(
            writer,
            status="complete",
            success=True,
            exit_code=EXIT_SUCCESS,
            dry_run=False,
            resolved=resolved,
            summary_metrics=summary_metrics,
            failure=None,
            deployment=deployment,
            ingest_summary=ingest_summary,
            beir_metrics=beir_metrics,
            metric_gates=list(requirements),
            skipped_metric_gates=[],
            runfile={"source_path": runfile_path, "payload": runfile_payload} if runfile_payload is not None else None,
        )
        write_json(writer.path("results.json"), result)
        writer.status(
            status="complete",
            phase="write_artifacts",
            summary_metrics_path=writer.path("summary_metrics.json"),
        )
        return RunOutcome(EXIT_SUCCESS, writer.artifact_dir, result)
    except HarnessRunError as exc:
        cleanup_error = _cleanup_manager(
            manager,
            writer,
            keep_up=bool(config and config.keep_up),
            collect_logs=True,
        )
        return _write_failure(
            writer,
            failure=exc.failure,
            exit_code=exc.exit_code,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
            deployment=deployment,
            cleanup_error=cleanup_error,
        )
    except Exception as exc:
        cleanup_error = _cleanup_manager(
            manager,
            writer,
            keep_up=bool(config and config.keep_up),
            collect_logs=True,
        )
        failure = FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="unexpected_internal_error",
            retryable=False,
            message=str(exc),
            debug_artifacts=("status.json", "events.jsonl", "run.log", "service_logs"),
        )
        return _write_failure(
            writer,
            failure=failure,
            exit_code=EXIT_INTERNAL_ERROR,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
            deployment=deployment,
            cleanup_error=cleanup_error,
        )
