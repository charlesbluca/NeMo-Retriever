# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed Helm autoscaling pressure harness.

This is the permanent stage-1 harness path for making split-topology demand
visible. It intentionally measures the current routing/ownership behavior;
it does not change gateway routing or recover pod-local backlog.

Keep this focused on durable pressure diagnostics. The experiment branch can
remain the place for broad scenario machinery and one-off lab iteration so this
supported harness does not accumulate avoidable operational maintenance burden.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import typer

from nemo_retriever.harness.artifacts import last_commit, now_timestr, write_json
from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness.helm_manager import HelmServiceManager
from nemo_retriever.utils.input_files import resolve_input_files

logger = logging.getLogger(__name__)

DEFAULT_AUTOSCALING_WORKLOADS: list[dict[str, Any]] = [
    {"name": "realtime_burst", "kind": "realtime_page", "count": 32, "concurrency": 32},
    {"name": "batch_burst", "kind": "batch_document", "count": 32, "concurrency": 32},
    {"name": "mixed_burst", "kind": "mixed", "count": 64, "concurrency": 64},
]
_TERMINAL_JOB_STATUSES = {"completed", "failed", "partial_success"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PressureWorkload:
    name: str
    kind: str
    count: int
    concurrency: int


@dataclass
class AttemptRecord:
    run_id: str
    workload: str
    kind: str
    concurrency: int
    file: str
    attempt: int
    started_at: str
    elapsed_s: float
    status_code: int | None
    retry_after_s: float | None = None
    document_id: str | None = None
    job_id: str | None = None
    error: str | None = None
    body_preview: str | None = None
    terminal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workload": self.workload,
            "kind": self.kind,
            "concurrency": self.concurrency,
            "file": self.file,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 4),
            "status_code": self.status_code,
            "retry_after_s": self.retry_after_s,
            "document_id": self.document_id,
            "job_id": self.job_id,
            "error": self.error,
            "body_preview": self.body_preview,
            "terminal": self.terminal,
        }


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=False) + "\n")


def resolve_workloads(cfg: HarnessConfig) -> list[PressureWorkload]:
    raw_workloads = cfg.autoscaling_workloads or DEFAULT_AUTOSCALING_WORKLOADS
    workloads: list[PressureWorkload] = []
    for idx, raw in enumerate(raw_workloads):
        if not isinstance(raw, dict):
            raise ValueError(f"autoscaling_workloads[{idx}] must be a mapping")
        name = str(raw.get("name") or f"workload_{idx + 1}")
        kind = str(raw.get("kind") or "batch_document")
        if kind not in {"realtime_page", "batch_document", "mixed"}:
            raise ValueError(
                f"autoscaling_workloads[{idx}].kind must be one of "
                "realtime_page, batch_document, mixed"
            )
        count = int(raw.get("count", 0))
        concurrency = int(raw.get("concurrency", 0))
        if count < 1 or concurrency < 1:
            raise ValueError(f"autoscaling_workloads[{idx}] count and concurrency must be >= 1")
        workloads.append(PressureWorkload(name=name, kind=kind, count=count, concurrency=concurrency))
    return workloads


def select_pressure_files(cfg: HarnessConfig) -> list[Path]:
    files = resolve_input_files(Path(cfg.dataset_dir), cfg.input_type)
    return files[: int(cfg.autoscaling_file_limit)]


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    terminal_status_counts: dict[str, int] = {}
    retryable_429s = 0
    terminal_successes = 0
    terminal_failures = 0
    files_seen: set[str] = set()
    terminal_files: set[str] = set()

    for attempt in attempts:
        status = attempt.get("status_code")
        key = "transport_error" if status is None else str(status)
        status_counts[key] = status_counts.get(key, 0) + 1
        file_name = str(attempt.get("file") or "")
        if file_name:
            files_seen.add(file_name)
        if status == 429 and not attempt.get("terminal"):
            retryable_429s += 1
        if attempt.get("terminal"):
            terminal_status_counts[key] = terminal_status_counts.get(key, 0) + 1
            if file_name:
                terminal_files.add(file_name)
            if status is not None and 200 <= int(status) < 300:
                terminal_successes += 1
            else:
                terminal_failures += 1

    return {
        "files_seen": len(files_seen),
        "files_terminal": len(terminal_files),
        "attempts_total": len(attempts),
        "status_counts": status_counts,
        "terminal_status_counts": terminal_status_counts,
        "retryable_429s": retryable_429s,
        "terminal_successes": terminal_successes,
        "terminal_failures": terminal_failures,
    }


def _run_kubectl(manager: HelmServiceManager, args: list[str], *, timeout_s: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        manager.kubectl_cmd + args + ["-n", manager.namespace],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _run_kubectl_to_file(manager: HelmServiceManager, args: list[str], path: Path, *, timeout_s: int = 30) -> int:
    try:
        result = _run_kubectl(manager, args, timeout_s=timeout_s)
        path.write_text(result.stdout, encoding="utf-8")
        if result.stderr:
            path.with_suffix(path.suffix + ".err").write_text(result.stderr, encoding="utf-8")
        return result.returncode
    except Exception as exc:
        path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1


def _http_get_to_file(url: str, path: Path) -> int:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
        path.write_text(resp.text, encoding="utf-8")
        return resp.status_code
    except Exception as exc:
        path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1


class AutoscalingSampler:
    def __init__(
        self,
        *,
        manager: HelmServiceManager,
        artifact_dir: Path,
        gateway_url: str,
        worker_urls: dict[str, str],
        interval_s: float,
    ) -> None:
        self.manager = manager
        self.artifact_dir = artifact_dir
        self.gateway_url = gateway_url.rstrip("/")
        self.worker_urls = {role: url.rstrip("/") for role, url in worker_urls.items()}
        self.interval_s = interval_s
        self.samples_csv = artifact_dir / "autoscaling_samples.csv"
        self.kubectl_dir = artifact_dir / "kubectl"
        self.http_dir = artifact_dir / "http"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._prepare_outputs()

    def _prepare_outputs(self) -> None:
        self.kubectl_dir.mkdir(parents=True, exist_ok=True)
        self.http_dir.mkdir(parents=True, exist_ok=True)
        if not self.samples_csv.exists():
            with self.samples_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "index", "kind", "name", "return_code", "path"])
                writer.writeheader()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="autoscaling-pressure-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2, 5.0))
            self._thread = None

    def collect_once(self, label: str) -> None:
        self._collect(label)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._collect("sample")
            self._stop.wait(self.interval_s)

    def _record(self, *, timestamp: str, kind: str, name: str, return_code: int | None, path: Path) -> None:
        with self.samples_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "index", "kind", "name", "return_code", "path"])
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "index": self._index,
                    "kind": kind,
                    "name": name,
                    "return_code": "" if return_code is None else return_code,
                    "path": str(path.relative_to(self.artifact_dir)),
                }
            )

    def _collect(self, label: str) -> None:
        timestamp = _utc_now()
        self._index += 1
        safe_label = f"{self._index:05d}_{label}"
        commands = {
            "pods_wide": ["get", "pods", "-o", "wide"],
            "pods_json": ["get", "pods", "-o", "json"],
            "hpa_wide": ["get", "hpa", "-o", "wide"],
            "hpa_json": ["get", "hpa", "-o", "json"],
            "events": ["get", "events", "--sort-by=.lastTimestamp"],
        }
        for name, args in commands.items():
            path = self.kubectl_dir / f"{safe_label}_{name}.txt"
            rc = _run_kubectl_to_file(self.manager, args, path)
            self._record(timestamp=timestamp, kind="kubectl", name=name, return_code=rc, path=path)

        http_targets = {
            "gateway_health": f"{self.gateway_url}/v1/health",
            "gateway_metrics": f"{self.gateway_url}/metrics",
        }
        for role, url in self.worker_urls.items():
            http_targets[f"{role}_pool_stats"] = f"{url}/v1/admin/pool_stats"
            http_targets[f"{role}_metrics"] = f"{url}/metrics"
        for name, url in http_targets.items():
            path = self.http_dir / f"{safe_label}_{name}.txt"
            rc = _http_get_to_file(url, path)
            self._record(timestamp=timestamp, kind="http", name=name, return_code=rc, path=path)


def _kubectl_json(manager: HelmServiceManager, args: list[str]) -> dict[str, Any] | None:
    try:
        result = _run_kubectl(manager, args + ["-o", "json"], timeout_s=30)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def inspect_cluster_state(manager: HelmServiceManager) -> dict[str, Any]:
    pods_json = _kubectl_json(manager, ["get", "pods"]) or {}
    hpa_json = _kubectl_json(manager, ["get", "hpa"]) or {}
    pods = []
    for item in pods_json.get("items", []):
        conditions = {
            cond.get("type"): cond.get("status")
            for cond in item.get("status", {}).get("conditions", [])
            if isinstance(cond, dict)
        }
        pods.append(
            {
                "name": item.get("metadata", {}).get("name"),
                "phase": item.get("status", {}).get("phase"),
                "ready": conditions.get("Ready"),
                "component": item.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component"),
            }
        )

    hpas = []
    for item in hpa_json.get("items", []):
        conditions = {
            cond.get("type"): cond.get("status")
            for cond in item.get("status", {}).get("conditions", [])
            if isinstance(cond, dict)
        }
        hpas.append(
            {
                "name": item.get("metadata", {}).get("name"),
                "min_replicas": item.get("spec", {}).get("minReplicas"),
                "max_replicas": item.get("spec", {}).get("maxReplicas"),
                "current_replicas": item.get("status", {}).get("currentReplicas"),
                "desired_replicas": item.get("status", {}).get("desiredReplicas"),
                "scaling_active": conditions.get("ScalingActive"),
                "conditions": conditions,
            }
        )

    external_metrics_api_available = False
    try:
        result = subprocess.run(
            manager.kubectl_cmd + ["get", "--raw", "/apis/external.metrics.k8s.io/v1beta1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        external_metrics_api_available = result.returncode == 0
    except Exception:
        external_metrics_api_available = False

    return {
        "services": {
            role: manager.find_services_by_component(role)
            for role in ("gateway", "realtime", "batch", "service")
        },
        "pods": pods,
        "hpas": hpas,
        "external_metrics_api_available": external_metrics_api_available,
    }


def _start_worker_service_forwards(manager: HelmServiceManager) -> dict[str, str]:
    worker_urls: dict[str, str] = {}
    for offset, role in enumerate(("realtime", "batch"), start=10):
        services = manager.find_services_by_component(role)
        if not services:
            continue
        local_port = manager.local_port + offset
        try:
            manager.start_port_forward(services[0], local_port=local_port, remote_port=7670)
        except RuntimeError as exc:
            logger.warning("Could not port-forward %s worker service: %s", role, exc)
            continue
        worker_urls[role] = f"http://localhost:{local_port}"
    return worker_urls


async def _create_job(client: httpx.AsyncClient, base_url: str, *, expected_documents: int, label: str) -> str:
    resp = await client.post(
        f"{base_url}/v1/ingest/job",
        json={
            "expected_documents": expected_documents,
            "label": label,
            "metadata": {"source": "autoscaling_pressure", "created_at": _utc_now()},
            "retain_results": False,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise RuntimeError(f"job creation returned no job_id: {body!r}")
    return str(job_id)


def _upload_target(kind: str, index: int, job_id: str, file_path: Path) -> tuple[str, dict[str, str]]:
    effective_kind = kind
    if kind == "mixed":
        effective_kind = "realtime_page" if index % 2 == 0 else "batch_document"
    if effective_kind == "realtime_page":
        return (
            f"/v1/ingest/job/{job_id}/page",
            {
                "document_id": f"{file_path.stem}-{index}",
                "page_number": "1",
                "filename": file_path.name,
            },
        )
    return (
        f"/v1/ingest/job/{job_id}/whole",
        {"metadata": json.dumps({"filename": file_path.name})},
    )


async def _upload_file_with_attempts(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    file_path: Path,
    file_index: int,
    run_id: str,
    workload: PressureWorkload,
    attempts_writer: JsonlWriter,
    max_upload_attempts: int,
    default_retry_after_s: float,
) -> list[AttemptRecord]:
    file_bytes = file_path.read_bytes()
    target_path, form_data = _upload_target(workload.kind, file_index, job_id, file_path)
    url = f"{base_url}{target_path}"
    records: list[AttemptRecord] = []

    for attempt in range(1, max_upload_attempts + 1):
        started = time.perf_counter()
        started_at = _utc_now()
        status_code: int | None = None
        retry_after: float | None = None
        document_id: str | None = None
        error: str | None = None
        body_preview: str | None = None
        try:
            resp = await client.post(
                url,
                files={"file": (file_path.name, file_bytes, "application/pdf")},
                data=form_data,
            )
            status_code = resp.status_code
            retry_after = float(resp.headers.get("retry-after", default_retry_after_s))
            body_preview = resp.text[:500] if resp.text else None
            if 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                    document_id = body.get("document_id") or body.get("page_id")
                except Exception:
                    document_id = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        should_retry = status_code == 429 and attempt < max_upload_attempts
        record = AttemptRecord(
            run_id=run_id,
            workload=workload.name,
            kind=workload.kind,
            concurrency=workload.concurrency,
            file=str(file_path),
            attempt=attempt,
            started_at=started_at,
            elapsed_s=time.perf_counter() - started,
            status_code=status_code,
            retry_after_s=retry_after,
            document_id=document_id,
            job_id=job_id,
            error=error,
            body_preview=body_preview,
            terminal=not should_retry,
        )
        records.append(record)
        attempts_writer.write(record.as_dict())
        if should_retry:
            await asyncio.sleep(retry_after or default_retry_after_s)
            continue
        return records

    return records


async def _poll_job_until_terminal_or_timeout(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    *,
    timeout_s: float,
    artifact_dir: Path,
    workload_name: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_body: dict[str, Any] = {}
    consecutive_errors = 0
    while time.monotonic() < deadline:
        try:
            resp = await client.get(
                f"{base_url}/v1/ingest/job/{job_id}",
                params={"include_documents": "true"},
                timeout=10.0,
            )
            body = resp.json()
            body["_http_status"] = resp.status_code
            last_body = body
            consecutive_errors = 0
            if body.get("status") in _TERMINAL_JOB_STATUSES:
                break
        except Exception as exc:
            consecutive_errors += 1
            last_body = {"error": f"{type(exc).__name__}: {exc}", "consecutive_errors": consecutive_errors}
            if consecutive_errors >= 5:
                break
        await asyncio.sleep(2.0)

    jobs_dir = artifact_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    write_json(jobs_dir / f"{workload_name}_{job_id}.json", last_body)
    return last_body


async def run_traffic_workload(
    *,
    base_url: str,
    files: list[Path],
    workload: PressureWorkload,
    cfg: HarnessConfig,
    artifact_dir: Path,
    attempts_writer: JsonlWriter,
    run_id: str,
) -> dict[str, Any]:
    timeout = httpx.Timeout(float(cfg.autoscaling_request_timeout_s))
    limits = httpx.Limits(
        max_connections=max(workload.concurrency * 2, 20),
        max_keepalive_connections=max(workload.concurrency, 10),
    )
    selected = [files[idx % len(files)] for idx in range(workload.count)]
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        job_id = await _create_job(
            client,
            base_url,
            expected_documents=workload.count,
            label=f"autoscaling-{run_id}-{workload.name}",
        )
        sem = asyncio.Semaphore(workload.concurrency)

        async def _one(idx: int, path: Path) -> list[AttemptRecord]:
            async with sem:
                return await _upload_file_with_attempts(
                    client=client,
                    base_url=base_url,
                    job_id=job_id,
                    file_path=path,
                    file_index=idx,
                    run_id=run_id,
                    workload=workload,
                    attempts_writer=attempts_writer,
                    max_upload_attempts=int(cfg.autoscaling_max_upload_attempts),
                    default_retry_after_s=float(cfg.autoscaling_retry_after_s),
                )

        started = time.perf_counter()
        nested = await asyncio.gather(*[_one(idx, path) for idx, path in enumerate(selected)])
        attempts = [record.as_dict() for records in nested for record in records]
        job_snapshot = await _poll_job_until_terminal_or_timeout(
            client,
            base_url,
            job_id,
            timeout_s=float(cfg.autoscaling_job_timeout_s),
            artifact_dir=artifact_dir,
            workload_name=workload.name,
        )
        elapsed_s = time.perf_counter() - started

    summary = summarize_attempts(attempts)
    summary.update(
        {
            "workload": workload.name,
            "kind": workload.kind,
            "concurrency": workload.concurrency,
            "count": workload.count,
            "job_id": job_id,
            "elapsed_s": round(elapsed_s, 3),
            "job_status": job_snapshot.get("status"),
            "job_counts": job_snapshot.get("counts"),
        }
    )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _base_result(cfg: HarnessConfig, artifact_dir: Path, run_id: str, tags: list[str] | None) -> dict[str, Any]:
    result = {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "run_id": run_id,
        "test_config": {
            "dataset_label": cfg.dataset_label,
            "dataset_dir": cfg.dataset_dir,
            "preset": cfg.preset,
            "run_mode": cfg.run_mode,
            "run_type": cfg.run_type,
            "manage_service": cfg.manage_service,
            "helm_chart": cfg.helm_chart,
            "helm_chart_version": cfg.helm_chart_version,
            "helm_release": cfg.helm_release,
            "helm_namespace": cfg.helm_namespace or cfg.helm_release,
            "helm_values_file": cfg.helm_values_file,
            "input_type": cfg.input_type,
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir.resolve()),
            "attempts": str((artifact_dir / "autoscaling_attempts.jsonl").resolve()),
            "samples": str((artifact_dir / "autoscaling_samples.csv").resolve()),
            "kubectl": str((artifact_dir / "kubectl").resolve()),
            "http": str((artifact_dir / "http").resolve()),
            "jobs": str((artifact_dir / "jobs").resolve()),
            "service_logs": str((artifact_dir / "service_logs").resolve()),
        },
    }
    if tags:
        result["tags"] = list(tags)
    return result


def run_autoscaling_pressure(
    cfg: HarnessConfig,
    artifact_dir: Path,
    *,
    run_id: str,
    tags: list[str] | None = None,
    skip_local_history: bool = False,
) -> dict[str, Any]:
    files = select_pressure_files(cfg)
    if not files:
        result = _base_result(cfg, artifact_dir, run_id, tags)
        result.update(
            {
                "success": False,
                "return_code": 1,
                "failure_reason": f"No {cfg.input_type} files found in {cfg.dataset_dir}",
            }
        )
        write_json(artifact_dir / "results.json", result)
        return result

    workloads = resolve_workloads(cfg)
    manager = HelmServiceManager(cfg)
    attempts_path = artifact_dir / "autoscaling_attempts.jsonl"
    attempts_writer = JsonlWriter(attempts_path)
    sampler: AutoscalingSampler | None = None
    step_summaries: list[dict[str, Any]] = []
    preflight: dict[str, Any] = {}
    start_rc = 1
    start_error: str | None = None

    typer.echo(f"\n=== Running autoscaling pressure harness: {run_id} ===")
    typer.echo(f"  Dataset       : {cfg.dataset_dir}")
    typer.echo(f"  Selected files: {len(files)}")
    typer.echo(f"  Workloads     : {', '.join(w.name for w in workloads)}")

    try:
        try:
            start_rc = manager.start()
        except Exception as exc:
            start_error = f"{type(exc).__name__}: {exc}"
            start_rc = 1
        if start_rc != 0:
            raise RuntimeError(start_error or f"managed Helm service failed to become ready (exit {start_rc})")

        worker_urls = _start_worker_service_forwards(manager)
        preflight = inspect_cluster_state(manager)
        write_json(artifact_dir / "autoscaling_preflight.json", preflight)

        sampler = AutoscalingSampler(
            manager=manager,
            artifact_dir=artifact_dir,
            gateway_url=manager.get_service_url(),
            worker_urls=worker_urls,
            interval_s=float(cfg.autoscaling_sample_interval_s),
        )
        sampler.collect_once("preflight")
        sampler.start()

        for workload in workloads:
            typer.echo(f"  Running {workload.name}: count={workload.count} concurrency={workload.concurrency}")
            summary = asyncio.run(
                run_traffic_workload(
                    base_url=manager.get_service_url(),
                    files=files,
                    workload=workload,
                    cfg=cfg,
                    artifact_dir=artifact_dir,
                    attempts_writer=attempts_writer,
                    run_id=run_id,
                )
            )
            step_summaries.append(summary)
            write_json(artifact_dir / f"autoscaling_summary_{workload.name}.json", summary)
            if sampler is not None:
                sampler.collect_once(f"post_{workload.name}")

        attempts = _read_jsonl(attempts_path)
        result = _base_result(cfg, artifact_dir, run_id, tags)
        result.update(
            {
                "success": True,
                "return_code": 0,
                "failure_reason": None,
                "selected_file_count": len(files),
                "workloads": [w.__dict__ for w in workloads],
                "preflight": preflight,
                "overall_attempts": summarize_attempts(attempts),
                "steps": step_summaries,
                "acceptance_signals": {
                    "raw_attempts_recorded": bool(attempts),
                    "retryable_429s": summarize_attempts(attempts).get("retryable_429s", 0),
                    "hpa_snapshots_recorded": bool(preflight.get("hpas")),
                    "pod_readiness_recorded": bool(preflight.get("pods")),
                    "worker_pool_stats_requested": bool(worker_urls),
                },
            }
        )
        write_json(artifact_dir / "autoscaling_pressure_summary.json", result)
        write_json(artifact_dir / "results.json", result)
        if not skip_local_history:
            try:
                from nemo_retriever.harness.history import record_run as _record_history

                _record_history(result, artifact_dir)
            except Exception:
                pass
        return result
    except Exception as exc:
        result = _base_result(cfg, artifact_dir, run_id, tags)
        result.update(
            {
                "success": False,
                "return_code": start_rc if start_rc != 0 else 1,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "selected_file_count": len(files),
                "workloads": [w.__dict__ for w in workloads],
                "preflight": preflight,
                "steps": step_summaries,
                "overall_attempts": summarize_attempts(_read_jsonl(attempts_path)),
            }
        )
        write_json(artifact_dir / "autoscaling_pressure_summary.json", result)
        write_json(artifact_dir / "results.json", result)
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        try:
            manager.dump_logs(artifact_dir)
        except Exception as exc:
            logger.warning("Could not dump managed service logs: %s", exc)
        try:
            manager.stop(uninstall=not cfg.keep_up)
        except Exception as exc:
            logger.warning("Managed Helm cleanup failed: %s", exc)
