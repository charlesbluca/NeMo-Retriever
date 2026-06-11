# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable Helm autoscaling pressure lab.

This module is intentionally separate from the stable ``retriever harness``
CLI. It is a branch-local experiment runner for collecting raw evidence about
split-topology HPA behavior, gateway routing, backend 429s, and pod-local
queue skew.

Run from the repo root with:

    python -m nemo_retriever.harness.autoscaling_lab --config nemo_retriever/harness/autoscaling-lab.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from nemo_retriever.harness.artifacts import create_run_artifact_dir, last_commit, now_timestr, write_json
from nemo_retriever.harness.config import HarnessConfig, REPO_ROOT
from nemo_retriever.harness.helm_manager import HelmServiceManager
from nemo_retriever.utils.input_files import resolve_input_files

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_STEPS = (8, 16, 32, 64)
DEFAULT_SAMPLE_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DEFAULT_JOB_TIMEOUT_S = 1800.0
DEFAULT_MAX_UPLOAD_ATTEMPTS = 5
DEFAULT_RETRY_AFTER_S = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_pressure_helm_values() -> dict[str, Any]:
    """Return the intentionally small split-topology lab values.

    The default uses CPU HPA because it is the least surprising way to get
    a runnable lab on clusters without prometheus-adapter. Swap the
    ``autoscaling.queueDepth`` and ``topology.*.hpa.metrics`` blocks in the
    YAML config when validating the chart's prometheus-adapter path.
    """

    return {
        "topology": {
            "mode": "split",
            "gateway": {"replicas": 1},
            "realtime": {
                "replicas": 1,
                "hpa": {
                    "enabled": True,
                    "minReplicas": 1,
                    "maxReplicas": 4,
                    "metrics": {
                        "queueDepthRatio": {"enabled": False},
                        "processingLatencyP95": {"enabled": False},
                        "cpu": {"enabled": True, "targetUtilizationPercentage": 60},
                    },
                },
            },
            "batch": {
                "replicas": 1,
                "gpu": {"enabled": False, "count": 0},
                "hpa": {
                    "enabled": True,
                    "minReplicas": 1,
                    "maxReplicas": 4,
                    "metrics": {
                        "queueDepthRatio": {"enabled": False},
                        "processingLatencyP95": {"enabled": False},
                        "cpu": {"enabled": True, "targetUtilizationPercentage": 80},
                    },
                },
            },
        },
        "autoscaling": {"queueDepth": {"backend": "cpu"}},
        "serviceConfig": {
            "pipeline": {
                "realtimeWorkers": 1,
                "realtimeQueueSize": 4,
                "batchWorkers": 1,
                "batchQueueSize": 4,
            },
            "vectordb": {"enabled": True},
        },
        "nims": {"enabled": True},
        "nimOperator": {
            "page_elements": {"enabled": True},
            "table_structure": {"enabled": True},
            "ocr": {"enabled": True},
            "vlm_embed": {"enabled": True},
            "rerankqa": {"enabled": False},
            "nemotron_parse": {"enabled": False},
            "nemotron_3_nano_omni_30b_a3b_reasoning": {"enabled": False},
            "audio": {"enabled": False},
        },
    }


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    input_type: str = "pdf"


@dataclass(frozen=True)
class LabConfig:
    config_path: Path
    dataset_label: str
    datasets: dict[str, DatasetSpec]
    concurrency_steps: tuple[int, ...] = DEFAULT_CONCURRENCY_STEPS
    file_limit: int | None = 64
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S
    max_upload_attempts: int = DEFAULT_MAX_UPLOAD_ATTEMPTS
    retry_after_s: float = DEFAULT_RETRY_AFTER_S
    run_name: str = "autoscaling-lab"
    artifacts_dir: str | None = None
    keep_up: bool = False

    helm_chart: str = str(REPO_ROOT / "nemo_retriever" / "helm")
    helm_release: str = "nrl-autoscale-lab"
    helm_namespace: str = "nrl-autoscale-lab"
    helm_timeout: int = 1800
    readiness_timeout: int = 1800
    helm_service_local_port: int = 17670
    helm_bin: str = "helm"
    kubectl_bin: str = "kubectl"
    helm_sudo: bool = False
    kubectl_sudo: bool = False
    helm_set: dict[str, Any] = field(default_factory=dict)
    helm_values: dict[str, Any] = field(default_factory=default_pressure_helm_values)

    @property
    def dataset(self) -> DatasetSpec:
        return self.datasets[self.dataset_label]

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        dataset_override: str | None = None,
        file_limit_override: int | None = None,
        artifacts_dir_override: str | None = None,
        keep_up_override: bool | None = None,
    ) -> "LabConfig":
        config_path = path.expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Autoscaling lab config must be a mapping: {config_path}")

        base_dir = config_path.parent
        datasets_raw = raw.get("datasets") or {}
        if not isinstance(datasets_raw, dict) or not datasets_raw:
            raise ValueError("autoscaling lab config requires at least one dataset")

        datasets: dict[str, DatasetSpec] = {}
        for label, entry in datasets_raw.items():
            if not isinstance(entry, dict):
                raise ValueError(f"dataset {label!r} must be a mapping")
            raw_path = entry.get("path")
            if not raw_path:
                raise ValueError(f"dataset {label!r} requires path")
            ds_path = Path(str(raw_path)).expanduser()
            if not ds_path.is_absolute():
                ds_path = (base_dir / ds_path).resolve()
            else:
                ds_path = ds_path.resolve()
            datasets[str(label)] = DatasetSpec(
                label=str(label),
                path=ds_path,
                input_type=str(entry.get("input_type") or "pdf"),
            )

        lab_raw = raw.get("lab") or {}
        helm_raw = raw.get("helm") or {}
        if not isinstance(lab_raw, dict):
            raise ValueError("lab section must be a mapping")
        if not isinstance(helm_raw, dict):
            raise ValueError("helm section must be a mapping")

        dataset_label = dataset_override or str(lab_raw.get("dataset") or next(iter(datasets)))
        if dataset_label not in datasets:
            raise ValueError(f"unknown dataset {dataset_label!r}; available={sorted(datasets)}")

        concurrency = lab_raw.get("concurrency_steps", DEFAULT_CONCURRENCY_STEPS)
        if not isinstance(concurrency, (list, tuple)) or not concurrency:
            raise ValueError("lab.concurrency_steps must be a non-empty list")
        concurrency_steps = tuple(int(v) for v in concurrency)
        if any(v < 1 for v in concurrency_steps):
            raise ValueError("all concurrency steps must be >= 1")

        file_limit = lab_raw.get("file_limit", 64)
        if file_limit_override is not None:
            file_limit = file_limit_override
        parsed_file_limit = None if file_limit is None else int(file_limit)
        if parsed_file_limit is not None and parsed_file_limit < 1:
            raise ValueError("file_limit must be >= 1 when set")

        default_values = default_pressure_helm_values()
        user_values = helm_raw.get("values") or {}
        if not isinstance(user_values, dict):
            raise ValueError("helm.values must be a mapping")

        return cls(
            config_path=config_path,
            dataset_label=dataset_label,
            datasets=datasets,
            concurrency_steps=concurrency_steps,
            file_limit=parsed_file_limit,
            sample_interval_s=float(lab_raw.get("sample_interval_s", DEFAULT_SAMPLE_INTERVAL_S)),
            request_timeout_s=float(lab_raw.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)),
            job_timeout_s=float(lab_raw.get("job_timeout_s", DEFAULT_JOB_TIMEOUT_S)),
            max_upload_attempts=int(lab_raw.get("max_upload_attempts", DEFAULT_MAX_UPLOAD_ATTEMPTS)),
            retry_after_s=float(lab_raw.get("retry_after_s", DEFAULT_RETRY_AFTER_S)),
            run_name=str(lab_raw.get("run_name") or "autoscaling-lab"),
            artifacts_dir=artifacts_dir_override or lab_raw.get("artifacts_dir"),
            keep_up=bool(keep_up_override if keep_up_override is not None else lab_raw.get("keep_up", False)),
            helm_chart=str(helm_raw.get("chart") or (REPO_ROOT / "nemo_retriever" / "helm")),
            helm_release=str(helm_raw.get("release") or "nrl-autoscale-lab"),
            helm_namespace=str(helm_raw.get("namespace") or helm_raw.get("release") or "nrl-autoscale-lab"),
            helm_timeout=int(helm_raw.get("timeout_s", 1800)),
            readiness_timeout=int(helm_raw.get("readiness_timeout_s", 1800)),
            helm_service_local_port=int(helm_raw.get("service_local_port", 17670)),
            helm_bin=str(helm_raw.get("helm_bin") or "helm"),
            kubectl_bin=str(helm_raw.get("kubectl_bin") or "kubectl"),
            helm_sudo=bool(helm_raw.get("helm_sudo", False)),
            kubectl_sudo=bool(helm_raw.get("kubectl_sudo", False)),
            helm_set=dict(helm_raw.get("set") or {}),
            helm_values=_deep_merge(default_values, user_values),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        dataset = self.dataset
        if not dataset.path.exists():
            errors.append(f"dataset path does not exist: {dataset.path}")
        if dataset.input_type != "pdf":
            errors.append(f"only PDF pressure is supported in this lab, got input_type={dataset.input_type!r}")
        if self.max_upload_attempts < 1:
            errors.append("max_upload_attempts must be >= 1")
        if self.sample_interval_s <= 0:
            errors.append("sample_interval_s must be > 0")
        if self.request_timeout_s <= 0:
            errors.append("request_timeout_s must be > 0")
        if self.job_timeout_s <= 0:
            errors.append("job_timeout_s must be > 0")
        return errors

    def write_generated_helm_values(self, artifact_dir: Path) -> Path:
        values_file = artifact_dir / "generated-helm-values.yaml"
        values_file.write_text(yaml.safe_dump(self.helm_values, sort_keys=False), encoding="utf-8")
        return values_file

    def to_harness_config(self, helm_values_file: Path) -> HarnessConfig:
        dataset = self.dataset
        return HarnessConfig(
            dataset_dir=str(dataset.path),
            dataset_label=dataset.label,
            preset="autoscaling_lab",
            run_mode="service",
            input_type=dataset.input_type,
            recall_required=False,
            evaluation_mode="none",
            service_max_concurrency=max(self.concurrency_steps),
            manage_service=True,
            keep_up=self.keep_up,
            helm_chart=self.helm_chart,
            helm_release=self.helm_release,
            helm_namespace=self.helm_namespace,
            helm_values_file=str(helm_values_file),
            helm_set=self.helm_set,
            helm_timeout=self.helm_timeout,
            readiness_timeout=self.readiness_timeout,
            helm_service_local_port=self.helm_service_local_port,
            helm_bin=self.helm_bin,
            kubectl_bin=self.kubectl_bin,
            helm_sudo=self.helm_sudo,
            kubectl_sudo=self.kubectl_sudo,
        )


@dataclass
class AttemptRecord:
    run_id: str
    step: str
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
            "step": self.step,
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


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw upload attempts without hiding retried 429s."""

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


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class Sampler:
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
        self.worker_urls = {name: url.rstrip("/") for name, url in worker_urls.items()}
        self.interval_s = interval_s
        self.samples_csv = artifact_dir / "samples.csv"
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
        self._prepare_outputs()
        self._thread = threading.Thread(target=self._loop, name="autoscaling-lab-sampler", daemon=True)
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
            "hpa_wide": ["get", "hpa", "-o", "wide"],
            "pods_json": ["get", "pods", "-o", "json"],
            "hpa_json": ["get", "hpa", "-o", "json"],
            "events": ["get", "events", "--sort-by=.lastTimestamp"],
        }
        for name, args in commands.items():
            path = self.kubectl_dir / f"{safe_label}_{name}.txt"
            rc = _run_kubectl_to_file(self.manager, args, path)
            self._record(timestamp=timestamp, kind="kubectl", name=name, return_code=rc, path=path)

        http_targets = {"gateway_metrics": f"{self.gateway_url}/metrics", "gateway_health": f"{self.gateway_url}/v1/health"}
        for role, url in self.worker_urls.items():
            http_targets[f"{role}_pool_stats"] = f"{url}/v1/admin/pool_stats"
            http_targets[f"{role}_metrics"] = f"{url}/metrics"
        for name, url in http_targets.items():
            path = self.http_dir / f"{safe_label}_{name}.txt"
            rc = _http_get_to_file(url, path)
            self._record(timestamp=timestamp, kind="http", name=name, return_code=rc, path=path)


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


def _kubectl_json(manager: HelmServiceManager, args: list[str]) -> dict[str, Any] | None:
    try:
        result = _run_kubectl(manager, args + ["-o", "json"], timeout_s=30)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def inspect_hpa_state(manager: HelmServiceManager) -> dict[str, Any]:
    hpa_json = _kubectl_json(manager, ["get", "hpa"]) or {}
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
    external_metrics = False
    try:
        result = subprocess.run(
            manager.kubectl_cmd + ["get", "--raw", "/apis/external.metrics.k8s.io/v1beta1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        external_metrics = result.returncode == 0
    except Exception:
        external_metrics = False
    return {"hpas": hpas, "external_metrics_api_available": external_metrics}


def _dump_pod_logs(manager: HelmServiceManager, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    pods = _run_kubectl(
        manager,
        [
            "get",
            "pods",
            "-l",
            f"app.kubernetes.io/instance={manager.release_name}",
            "-o",
            "name",
        ],
        timeout_s=30,
    )
    (logs_dir / "kubectl_get_pods.txt").write_text(pods.stdout, encoding="utf-8")
    if pods.stderr:
        (logs_dir / "kubectl_get_pods.err").write_text(pods.stderr, encoding="utf-8")
    if pods.returncode != 0:
        return
    for pod_ref in [line.strip() for line in pods.stdout.splitlines() if line.strip()]:
        pod = pod_ref.split("/", 1)[-1]
        try:
            result = _run_kubectl(manager, ["logs", pod, "--all-containers", "--tail=-1"], timeout_s=180)
        except Exception as exc:
            (logs_dir / f"{pod}.err").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            continue
        (logs_dir / f"{pod}.log").write_text(result.stdout, encoding="utf-8")
        if result.stderr:
            (logs_dir / f"{pod}.err").write_text(result.stderr, encoding="utf-8")


async def _create_job(client: httpx.AsyncClient, base_url: str, *, expected_documents: int, label: str) -> str:
    resp = await client.post(
        f"{base_url}/v1/ingest/job",
        json={
            "expected_documents": expected_documents,
            "label": label,
            "metadata": {"source": "autoscaling_lab", "created_at": _utc_now()},
            "retain_results": False,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise RuntimeError(f"job creation returned no job_id: {body!r}")
    return str(job_id)


async def _upload_file_with_attempts(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    file_path: Path,
    run_id: str,
    step: str,
    concurrency: int,
    attempts_writer: JsonlWriter,
    max_upload_attempts: int,
    default_retry_after_s: float,
) -> list[AttemptRecord]:
    file_bytes = file_path.read_bytes()
    records: list[AttemptRecord] = []
    url = f"{base_url}/v1/ingest/job/{job_id}/document"
    meta_json = json.dumps({"filename": file_path.name})

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
                data={"metadata": meta_json},
            )
            status_code = resp.status_code
            retry_after = float(resp.headers.get("retry-after", default_retry_after_s))
            body_preview = resp.text[:500] if resp.text else None
            if 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                    document_id = body.get("document_id")
                except Exception:
                    document_id = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        should_retry = status_code == 429 and attempt < max_upload_attempts
        terminal = not should_retry
        record = AttemptRecord(
            run_id=run_id,
            step=step,
            concurrency=concurrency,
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
            terminal=terminal,
        )
        records.append(record)
        attempts_writer.write(record.as_dict())

        if should_retry:
            await asyncio.sleep(retry_after or default_retry_after_s)
            continue
        return records

    return records


async def _poll_job_until_quiet(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
    *,
    timeout_s: float,
    artifact_dir: Path,
    step: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_body: dict[str, Any] = {}
    terminal_statuses = {"completed", "failed", "partial_success"}
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
            if body.get("status") in terminal_statuses:
                break
        except Exception as exc:
            consecutive_errors += 1
            last_body = {"error": f"{type(exc).__name__}: {exc}", "consecutive_errors": consecutive_errors}
            if consecutive_errors >= 5:
                break
        await asyncio.sleep(2.0)

    step_dir = artifact_dir / "jobs"
    step_dir.mkdir(parents=True, exist_ok=True)
    write_json(step_dir / f"{step}_{job_id}.json", last_body)
    return last_body


async def run_traffic_step(
    *,
    base_url: str,
    files: list[Path],
    concurrency: int,
    config: LabConfig,
    artifact_dir: Path,
    attempts_writer: JsonlWriter,
    run_id: str,
) -> dict[str, Any]:
    step = f"c{concurrency}"
    timeout = httpx.Timeout(config.request_timeout_s)
    limits = httpx.Limits(max_connections=max(concurrency * 2, 20), max_keepalive_connections=max(concurrency, 10))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        job_id = await _create_job(
            client,
            base_url,
            expected_documents=len(files),
            label=f"autoscaling-lab-{run_id}-{step}",
        )
        sem = asyncio.Semaphore(concurrency)

        async def _one(path: Path) -> list[AttemptRecord]:
            async with sem:
                return await _upload_file_with_attempts(
                    client=client,
                    base_url=base_url,
                    job_id=job_id,
                    file_path=path,
                    run_id=run_id,
                    step=step,
                    concurrency=concurrency,
                    attempts_writer=attempts_writer,
                    max_upload_attempts=config.max_upload_attempts,
                    default_retry_after_s=config.retry_after_s,
                )

        started = time.perf_counter()
        nested = await asyncio.gather(*[_one(path) for path in files])
        attempts = [record.as_dict() for records in nested for record in records]
        job_snapshot = await _poll_job_until_quiet(
            client,
            base_url,
            job_id,
            timeout_s=config.job_timeout_s,
            artifact_dir=artifact_dir,
            step=step,
        )
        elapsed_s = time.perf_counter() - started

    summary = summarize_attempts(attempts)
    summary.update(
        {
            "step": step,
            "concurrency": concurrency,
            "job_id": job_id,
            "elapsed_s": round(elapsed_s, 3),
            "job_status": job_snapshot.get("status"),
            "job_counts": job_snapshot.get("counts"),
        }
    )
    return summary


def select_files(config: LabConfig) -> list[Path]:
    files = resolve_input_files(config.dataset.path, config.dataset.input_type)
    if config.file_limit is not None:
        files = files[: config.file_limit]
    return files


def dry_run_payload(config: LabConfig, artifact_dir: Path, files: list[Path]) -> dict[str, Any]:
    return {
        "config": str(config.config_path),
        "dataset": config.dataset_label,
        "dataset_path": str(config.dataset.path),
        "selected_files": [str(path) for path in files],
        "selected_file_count": len(files),
        "concurrency_steps": list(config.concurrency_steps),
        "artifact_dir": str(artifact_dir),
        "helm_chart": config.helm_chart,
        "helm_release": config.helm_release,
        "helm_namespace": config.helm_namespace,
        "helm_values": config.helm_values,
        "helm_set": config.helm_set,
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


def run_lab(config: LabConfig, *, dry_run: bool = False) -> dict[str, Any]:
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))

    artifact_dir = create_run_artifact_dir(
        config.dataset_label,
        run_name=config.run_name,
        base_dir=config.artifacts_dir,
    )
    files = select_files(config)
    if not files:
        raise ValueError(f"no {config.dataset.input_type} files found in {config.dataset.path}")

    if dry_run:
        payload = dry_run_payload(config, artifact_dir, files)
        write_json(artifact_dir / "dry-run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=False))
        return payload

    helm_values_file = config.write_generated_helm_values(artifact_dir)
    harness_cfg = config.to_harness_config(helm_values_file)
    manager = HelmServiceManager(harness_cfg)
    attempts_writer = JsonlWriter(artifact_dir / "attempts.jsonl")
    run_id = now_timestr()
    step_summaries: list[dict[str, Any]] = []
    preflight: dict[str, Any] = {}
    sampler: Sampler | None = None
    start_rc = 1
    start_error: str | None = None

    try:
        try:
            start_rc = manager.start()
        except Exception as exc:
            start_error = f"{type(exc).__name__}: {exc}"
            start_rc = 1
        if start_rc != 0:
            raise RuntimeError(start_error or f"Helm service failed to start (exit {start_rc})")

        worker_urls = _start_worker_service_forwards(manager)
        preflight = {
            "services": {
                role: manager.find_services_by_component(role)
                for role in ("gateway", "realtime", "batch", "service")
            },
            "hpa_state": inspect_hpa_state(manager),
        }
        write_json(artifact_dir / "preflight.json", preflight)

        sampler = Sampler(
            manager=manager,
            artifact_dir=artifact_dir,
            gateway_url=manager.get_service_url(),
            worker_urls=worker_urls,
            interval_s=config.sample_interval_s,
        )
        sampler.collect_once("preflight")
        sampler.start()
        for concurrency in config.concurrency_steps:
            summary = asyncio.run(
                run_traffic_step(
                    base_url=manager.get_service_url(),
                    files=files,
                    concurrency=concurrency,
                    config=config,
                    artifact_dir=artifact_dir,
                    attempts_writer=attempts_writer,
                    run_id=run_id,
                )
            )
            step_summaries.append(summary)
            write_json(artifact_dir / f"summary_{summary['step']}.json", summary)
            if sampler is not None:
                sampler.collect_once(f"post_{summary['step']}")

        attempts = [
            json.loads(line)
            for line in (artifact_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        overall_attempts = summarize_attempts(attempts)
        result = {
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "success": True,
            "return_code": 0,
            "config": str(config.config_path),
            "dataset": config.dataset_label,
            "dataset_path": str(config.dataset.path),
            "files": len(files),
            "concurrency_steps": list(config.concurrency_steps),
            "preflight": preflight,
            "overall_attempts": overall_attempts,
            "steps": step_summaries,
            "artifacts": {
                "artifact_dir": str(artifact_dir),
                "attempts": str(artifact_dir / "attempts.jsonl"),
                "samples": str(artifact_dir / "samples.csv"),
                "kubectl": str(artifact_dir / "kubectl"),
                "logs": str(artifact_dir / "logs"),
            },
        }
        write_json(artifact_dir / "autoscaling_lab_summary.json", result)
        write_json(artifact_dir / "results.json", result)
        return result
    except Exception as exc:
        result = {
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "success": False,
            "return_code": start_rc if start_rc != 0 else 1,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "config": str(config.config_path),
            "dataset": config.dataset_label,
            "dataset_path": str(config.dataset.path),
            "preflight": preflight,
            "steps": step_summaries,
            "artifacts": {
                "artifact_dir": str(artifact_dir),
                "attempts": str(artifact_dir / "attempts.jsonl"),
                "samples": str(artifact_dir / "samples.csv"),
                "kubectl": str(artifact_dir / "kubectl"),
                "logs": str(artifact_dir / "logs"),
            },
        }
        write_json(artifact_dir / "autoscaling_lab_summary.json", result)
        write_json(artifact_dir / "results.json", result)
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        try:
            _dump_pod_logs(manager, artifact_dir / "logs")
        except Exception as exc:
            logger.warning("Could not dump pod logs: %s", exc)
        if not config.keep_up:
            manager.stop(uninstall=True)
        else:
            manager.stop(uninstall=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disposable Helm autoscaling pressure lab")
    parser.add_argument("--config", required=True, help="Path to autoscaling lab YAML config")
    parser.add_argument("--dataset", help="Dataset label from the config to run")
    parser.add_argument("--file-limit", type=int, help="Override the config file_limit")
    parser.add_argument("--artifacts-dir", help="Override artifact root")
    parser.add_argument("--keep-up", action="store_true", help="Leave the Helm release running after the lab")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved Helm values, files, and run matrix")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = LabConfig.from_file(
        Path(args.config),
        dataset_override=args.dataset,
        file_limit_override=args.file_limit,
        artifacts_dir_override=args.artifacts_dir,
        keep_up_override=True if args.keep_up else None,
    )
    result = run_lab(config, dry_run=bool(args.dry_run))
    return int(result.get("return_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
