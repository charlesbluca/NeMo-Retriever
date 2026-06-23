# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-Helm bo20 concurrency qualification runner.

This module intentionally sits beside the benchmark harness rather than in
the service client itself: it is an operator-facing lab tool for finding the
first saturation and hard-failure point when several independent bo20 service
jobs run at the same time.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import re
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from nemo_retriever.harness.artifacts import create_session_dir, last_commit, now_timestr, write_json
from nemo_retriever.harness.config import HarnessConfig, NEMO_RETRIEVER_ROOT
from nemo_retriever.harness.helm_manager import HelmServiceManager


logger = logging.getLogger(__name__)

DEFAULT_BO20_DATASET = "/localhome/charlesb/datasets/bo20"
EXPECTED_BO20_PDFS = 20
DEFAULT_MAX_N = 16
DEFAULT_JOB_MAX_CONCURRENCY = 8
DEFAULT_RUN_TIMEOUT_S = 7200
DEFAULT_IDLE_TIMEOUT_S = 900
DEFAULT_SAMPLE_INTERVAL_S = 15.0
DEFAULT_UX_PROBE_INTERVAL_S = 10.0
DEFAULT_UX_FALSE_N = 6
DEFAULT_UX_TRUE_N = 8
SATURATION_QUEUE_RATIO = 0.9
LATENCY_BLOWUP_FACTOR = 3.0
CLEAN_RECOVERY_MAX_N = 3
CLEAN_RECOVERY_HELM_SET_DEFAULTS: dict[str, Any] = {
    "serviceMonitor.autoEnableInSplitMode": False,
    "autoscaling.queueDepth.backend": "cpu",
    "topology.batch.gpu.enabled": False,
    "persistence.enabled": False,
    "retrieverResults.enabled": False,
}
NIM_BACKENDS = ("local", "nvcf", "proxy-dry-run")
NVCF_ENV_FILE = "~/.env"
NVCF_ENV_KEY = "NGC_NV_DEVELOPER_NVCF"
NVCF_SECRET_NAME = "ngc-api"
CORE_NIM_OPERATOR_SERVICES = {
    "page_elements": "nemotron-page-elements-v3",
    "table_structure": "nemotron-table-structure-v1",
    "ocr": "nemotron-ocr-v1",
    "vlm_embed": "llama-nemotron-embed-vl-1b-v2",
}
NVCF_HOSTED_ENDPOINTS: dict[str, str] = {
    "serviceConfig.nimEndpoints.pageElementsInvokeUrl": (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
    ),
    "serviceConfig.nimEndpoints.tableStructureInvokeUrl": (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
    ),
    "serviceConfig.nimEndpoints.ocrInvokeUrl": "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1",
}
NVCF_HELM_SET_DEFAULTS: dict[str, Any] = {
    "nims.enabled": False,
    "nimOperator.page_elements.enabled": False,
    "nimOperator.table_structure.enabled": False,
    "nimOperator.ocr.enabled": False,
    **NVCF_HOSTED_ENDPOINTS,
}
PROXY_DRY_RUN_HELM_SET_DEFAULTS: dict[str, Any] = {
    "nims.enabled": False,
    "nimOperator.page_elements.enabled": False,
    "nimOperator.table_structure.enabled": False,
    "nimOperator.ocr.enabled": False,
}
DRY_RUN_HEADER = "X-Nemo-Dry-Run"
PAGE_ELEMENTS_PATTERNS = (
    "nemotron-page-elements-v3",
    "page_elements_v3",
    "Page Elements NIM",
    "HTTPError: 500",
    "returned 500",
)
ENDPOINT_EVIDENCE_PATTERNS = (
    "ai.api.nvidia.com",
    "https://ai.api.nvidia.com",
    "http://nemotron-",
    "NIM endpoint http://nemotron-",
    "returned 429",
    "HTTP 429",
    "returned 500",
    "HTTPError: 500",
    "return_results",
    "failed to fetch/persist",
    "Gateway failed",
    "Gateway timed out",
    "Gateway callback",
    "unknown_document",
)


@dataclass
class Bo20Inventory:
    dataset_dir: str
    pdf_count: int
    total_pages: int
    page_counts: dict[str, int]


@dataclass
class JobRunResult:
    job_index: int
    job_id: str | None = None
    job_status: str | None = None
    success: bool = False
    hard_failure: bool = False
    hard_failure_reasons: list[str] = field(default_factory=list)
    exception: str | None = None
    traceback: str | None = None
    elapsed_s: float | None = None
    uploaded: int = 0
    completed: int = 0
    failed: int = 0
    upload_failed: int = 0
    failures: list[list[str]] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    result_rows: int | None = None
    dataframe_rows: int | None = None
    result_fetches: list[dict[str, Any]] = field(default_factory=list)
    retry_429_count: int = 0
    transient_retry_count: int = 0
    timed_out: bool = False
    exit_code: int | None = None


@dataclass
class SweepRoundResult:
    return_results: bool
    n: int
    attempt: str
    started_at: str
    finished_at: str
    wall_s: float
    success: bool
    hard_failure: bool
    hard_failure_reasons: list[str]
    saturation: bool
    saturation_reasons: list[str]
    job_results: list[dict[str, Any]]
    metrics: dict[str, Any]
    cluster_before: dict[str, Any]
    cluster_after: dict[str, Any]
    cluster_delta: dict[str, Any]
    samples: list[dict[str, Any]]
    idle_after_run: bool
    idle_wait_s: float
    idle_error: str | None = None
    phase: str | None = None
    failure_attribution: dict[str, Any] = field(default_factory=dict)


class _RetryCountingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.retry_429_count = 0
        self.transient_retry_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "429 for " in message:
            self.retry_429_count += 1
        if "Transient " in message and " retry in " in message:
            self.transient_retry_count += 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[idx]), 3)


def _normalize_nim_backend(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in NIM_BACKENDS:
        raise ValueError(f"nim backend must be one of {', '.join(NIM_BACKENDS)}, got {value!r}")
    return normalized


def _helm_set_for_nim_backend(helm_set: dict[str, Any] | None, nim_backend: str) -> dict[str, Any]:
    base = dict(helm_set or {})
    backend = _normalize_nim_backend(nim_backend)
    if backend == "nvcf":
        return {**base, **NVCF_HELM_SET_DEFAULTS}
    if backend == "proxy-dry-run":
        return {**base, **PROXY_DRY_RUN_HELM_SET_DEFAULTS}
    return base


def _effective_return_results_modes(nim_backend: str, modes: tuple[bool, ...] | None) -> tuple[bool, ...]:
    if _normalize_nim_backend(nim_backend) == "proxy-dry-run":
        return (False,)
    return modes or (False, True)


def _read_env_value(path: str | Path, key: str) -> str | None:
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return None
    prefix = f"{key}="
    export_prefix = f"export {key}="
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(export_prefix):
            value = line[len(export_prefix) :]
        elif line.startswith(prefix):
            value = line[len(prefix) :]
        else:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value.strip() or None
    return None


def _kubectl_apply_manifest(manager: HelmServiceManager, manifest: str, *, timeout_s: int = 60) -> None:
    result = subprocess.run(
        manager.kubectl_cmd + ["apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"kubectl apply failed: {detail}")


def _ensure_namespace(manager: HelmServiceManager) -> None:
    manifest = (
        "apiVersion: v1\n"
        "kind: Namespace\n"
        "metadata:\n"
        f"  name: {json.dumps(manager.namespace)}\n"
    )
    _kubectl_apply_manifest(manager, manifest)


def ensure_nvcf_secret_from_env(
    manager: HelmServiceManager,
    *,
    env_file: str | Path = NVCF_ENV_FILE,
    env_key: str = NVCF_ENV_KEY,
    secret_name: str = NVCF_SECRET_NAME,
) -> dict[str, Any]:
    token = _read_env_value(env_file, env_key)
    if not token:
        raise RuntimeError(f"{env_key} was not found in {Path(env_file).expanduser()}")
    _ensure_namespace(manager)
    manifest = (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        f"  name: {json.dumps(secret_name)}\n"
        f"  namespace: {json.dumps(manager.namespace)}\n"
        "type: Opaque\n"
        "stringData:\n"
        f"  NGC_API_KEY: {json.dumps(token)}\n"
        f"  NGC_CLI_API_KEY: {json.dumps(token)}\n"
    )
    _kubectl_apply_manifest(manager, manifest)
    return {"secret_name": secret_name, "env_file": str(Path(env_file).expanduser()), "env_key": env_key}


def _http_get_json(url: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, *, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _run_cmd(cmd: list[str], *, timeout_s: int = 60, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_s)


def _current_branch(repo_root: Path) -> str:
    result = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30, cwd=repo_root)
    if result.returncode != 0:
        return f"unknown ({result.stderr.strip() or result.stdout.strip()})"
    return result.stdout.strip()


def resolve_bo20_files(dataset_dir: str | Path, *, expected_pdfs: int = EXPECTED_BO20_PDFS) -> list[Path]:
    root = Path(dataset_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"bo20 dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"bo20 dataset path is not a directory: {root}")
    files = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    if len(files) != expected_pdfs:
        raise ValueError(f"Expected exactly {expected_pdfs} bo20 PDFs in {root}, found {len(files)}")
    return files


def _safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        count = len(doc)
        doc.close()
        return int(count)
    except Exception as exc:
        logger.warning("Could not count pages in %s: %s", path, exc)
        return None


def inventory_bo20_dataset(dataset_dir: str | Path, *, expected_pdfs: int = EXPECTED_BO20_PDFS) -> Bo20Inventory:
    files = resolve_bo20_files(dataset_dir, expected_pdfs=expected_pdfs)
    page_counts: dict[str, int] = {}
    for path in files:
        count = _safe_pdf_page_count(path)
        if count is None:
            raise ValueError(f"Could not read PDF page count for {path}")
        page_counts[path.name] = count
    return Bo20Inventory(
        dataset_dir=str(Path(dataset_dir).expanduser().resolve()),
        pdf_count=len(files),
        total_pages=sum(page_counts.values()),
        page_counts=page_counts,
    )


def _kubectl_json(manager: HelmServiceManager, args: list[str], *, timeout_s: int = 60) -> dict[str, Any]:
    result = _run_cmd(manager.kubectl_cmd + args, timeout_s=timeout_s)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or result.stdout.strip(), "return_code": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"invalid json: {exc}", "raw": result.stdout}


def _kubectl_text(manager: HelmServiceManager, args: list[str], *, timeout_s: int = 60) -> tuple[int, str, str]:
    result = _run_cmd(manager.kubectl_cmd + args, timeout_s=timeout_s)
    return result.returncode, result.stdout, result.stderr


def _pod_component(pod: dict[str, Any]) -> str:
    labels = ((pod.get("metadata") or {}).get("labels") or {})
    return str(labels.get("app.kubernetes.io/component") or "unknown")


def _pod_restart_totals(pods_snapshot: dict[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for pod in pods_snapshot.get("items", []) or []:
        component = _pod_component(pod)
        total = 0
        for status in ((pod.get("status") or {}).get("containerStatuses") or []):
            total += int(status.get("restartCount") or 0)
        totals[component] = totals.get(component, 0) + total
    return totals


def _pod_oom_events(pods_snapshot: dict[str, Any]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for pod in pods_snapshot.get("items", []) or []:
        pod_name = str((pod.get("metadata") or {}).get("name") or "")
        component = _pod_component(pod)
        for status in ((pod.get("status") or {}).get("containerStatuses") or []):
            container = str(status.get("name") or "")
            states = [status.get("state") or {}, status.get("lastState") or {}]
            for state in states:
                terminated = state.get("terminated") or {}
                waiting = state.get("waiting") or {}
                reason = str(terminated.get("reason") or waiting.get("reason") or "")
                if reason == "OOMKilled":
                    events.append({"pod": pod_name, "component": component, "container": container, "reason": reason})
    return events


def _parse_top_pods(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append({"pod": parts[0], "cpu": parts[1], "memory": parts[2]})
    return rows


def _parse_cpu_mcores(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("m"):
            return float(raw[:-1])
        if raw.endswith("n"):
            return float(raw[:-1]) / 1_000_000.0
        return float(raw) * 1000.0
    except ValueError:
        return None


def _parse_memory_mib(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024, "K": 1 / 1000, "M": 1, "G": 1000}
    for suffix, factor in units.items():
        if raw.endswith(suffix):
            try:
                return float(raw[: -len(suffix)]) * factor
            except ValueError:
                return None
    try:
        return float(raw) / (1024 * 1024)
    except ValueError:
        return None


def _component_by_pod(snapshot: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in snapshot.get("pods") or []:
        if isinstance(item, dict) and item.get("name"):
            mapping[str(item["name"])] = str(item.get("component") or "unknown")
    return mapping


def _resource_pressure_metrics(
    before: dict[str, Any],
    after: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshots = [before, after]
    snapshots.extend(sample.get("cluster") for sample in samples if isinstance(sample.get("cluster"), dict))
    max_cpu: dict[str, float] = {}
    max_mem: dict[str, float] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        components = _component_by_pod(snapshot)
        top_pods = snapshot.get("top_pods")
        if isinstance(top_pods, dict):
            continue
        for row in top_pods or []:
            pod_name = str(row.get("pod") or "")
            component = components.get(pod_name, "unknown")
            cpu = _parse_cpu_mcores(str(row.get("cpu") or ""))
            mem = _parse_memory_mib(str(row.get("memory") or ""))
            if cpu is not None:
                max_cpu[component] = max(max_cpu.get(component, 0.0), cpu)
            if mem is not None:
                max_mem[component] = max(max_mem.get(component, 0.0), mem)
    rounded_cpu = {key: round(value, 3) for key, value in sorted(max_cpu.items())}
    rounded_mem = {key: round(value, 3) for key, value in sorted(max_mem.items())}
    return {
        "max_cpu_mcores_by_component": rounded_cpu,
        "max_memory_mib_by_component": rounded_mem,
        "gateway_max_cpu_mcores": rounded_cpu.get("gateway"),
        "gateway_max_memory_mib": rounded_mem.get("gateway"),
        "batch_max_cpu_mcores": rounded_cpu.get("batch"),
        "batch_max_memory_mib": rounded_mem.get("batch"),
        "realtime_max_cpu_mcores": rounded_cpu.get("realtime"),
        "realtime_max_memory_mib": rounded_mem.get("realtime"),
    }


_INGEST_REQUEST_METRIC_RE = re.compile(r'^nemo_retriever_ingest_requests_total\{([^}]*)\}\s+([0-9.eE+-]+)')
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
_HELM_ENV_SET_RE = re.compile(r"^nimOperator\.([^.]+)\.env\[(\d+)\]\.(name|value)$")


def _parse_ingest_request_status_counts(metrics_text: str) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {}
    for line in metrics_text.splitlines():
        match = _INGEST_REQUEST_METRIC_RE.match(line.strip())
        if not match:
            continue
        labels = dict(_LABEL_RE.findall(match.group(1)))
        role = labels.get("role") or "unknown"
        status = labels.get("status") or "unknown"
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        counts.setdefault(role, {})[status] = counts.setdefault(role, {}).get(status, 0.0) + value
    return counts


def _capture_http_status_counts(metric_urls: dict[str, str] | None) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {}
    for component, base_url in sorted((metric_urls or {}).items()):
        try:
            text = _http_get_text(f"{base_url.rstrip('/')}/metrics", timeout_s=10.0)
        except Exception:
            continue
        parsed = _parse_ingest_request_status_counts(text)
        component_counts: dict[str, float] = {}
        for role_counts in parsed.values():
            for status, value in role_counts.items():
                component_counts[status] = component_counts.get(status, 0.0) + value
        counts[component] = component_counts
    return counts


def _http_status_delta(before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for component in sorted(set(before) | set(after)):
        statuses = set(before.get(component, {})) | set(after.get(component, {}))
        delta[component] = {}
        for status in sorted(statuses):
            value = after.get(component, {}).get(status, 0.0) - before.get(component, {}).get(status, 0.0)
            delta[component][status] = int(round(max(0.0, value)))
    return delta


def _http_status_delta_summary(delta: dict[str, dict[str, int]]) -> dict[str, int]:
    gateway = delta.get("gateway", {})
    worker_counts: dict[str, int] = {"4xx": 0, "5xx": 0}
    for component in ("realtime", "batch"):
        for status in ("4xx", "5xx"):
            worker_counts[status] += int(delta.get(component, {}).get(status, 0))
    return {
        "gateway_4xx_delta": int(gateway.get("4xx", 0)),
        "gateway_5xx_delta": int(gateway.get("5xx", 0)),
        "worker_4xx_delta": worker_counts["4xx"],
        "worker_5xx_delta": worker_counts["5xx"],
    }


def capture_cluster_snapshot(manager: HelmServiceManager) -> dict[str, Any]:
    selector = f"app.kubernetes.io/instance={manager.release_name}"
    pods = _kubectl_json(
        manager,
        ["get", "pods", "-n", manager.namespace, "-l", selector, "-o", "json"],
        timeout_s=60,
    )
    hpas = _kubectl_json(
        manager,
        ["get", "hpa", "-n", manager.namespace, "-l", selector, "-o", "json"],
        timeout_s=60,
    )
    top_rc, top_out, top_err = _kubectl_text(
        manager,
        ["top", "pods", "-n", manager.namespace, "-l", selector, "--no-headers"],
        timeout_s=60,
    )
    return {
        "timestamp": _utc_now_iso(),
        "pods": _summarize_pods(pods),
        "hpas": _summarize_hpas(hpas),
        "top_pods": _parse_top_pods(top_out) if top_rc == 0 else {"error": top_err.strip() or top_out.strip()},
        "raw_restart_totals": _pod_restart_totals(pods),
        "oom_events": _pod_oom_events(pods),
        "local_nim_runtime": _collect_local_nim_runtime_snapshot(manager),
    }


def _summarize_pods(pods: dict[str, Any]) -> Any:
    if "error" in pods:
        return pods
    summary: list[dict[str, Any]] = []
    for pod in pods.get("items", []) or []:
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        containers = []
        for item in status.get("containerStatuses") or []:
            containers.append(
                {
                    "name": item.get("name"),
                    "ready": item.get("ready"),
                    "restart_count": item.get("restartCount"),
                    "state": item.get("state"),
                    "last_state": item.get("lastState"),
                }
            )
        summary.append(
            {
                "name": metadata.get("name"),
                "component": _pod_component(pod),
                "phase": status.get("phase"),
                "pod_ip": status.get("podIP"),
                "containers": containers,
            }
        )
    return summary


def _summarize_hpas(hpas: dict[str, Any]) -> Any:
    if "error" in hpas:
        return hpas
    items = []
    for hpa in hpas.get("items", []) or []:
        status = hpa.get("status") or {}
        spec = hpa.get("spec") or {}
        items.append(
            {
                "name": (hpa.get("metadata") or {}).get("name"),
                "min_replicas": spec.get("minReplicas"),
                "max_replicas": spec.get("maxReplicas"),
                "current_replicas": status.get("currentReplicas"),
                "desired_replicas": status.get("desiredReplicas"),
                "current_metrics": status.get("currentMetrics"),
                "conditions": status.get("conditions") or [],
            }
        )
    return items


def cluster_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_restarts = before.get("raw_restart_totals") or {}
    after_restarts = after.get("raw_restart_totals") or {}
    components = sorted(set(before_restarts) | set(after_restarts))
    restart_delta = {
        component: int(after_restarts.get(component, 0)) - int(before_restarts.get(component, 0))
        for component in components
    }
    before_nim_restarts = ((before.get("local_nim_runtime") or {}).get("restart_totals_by_service") or {})
    after_nim_restarts = ((after.get("local_nim_runtime") or {}).get("restart_totals_by_service") or {})
    nim_services = sorted(set(before_nim_restarts) | set(after_nim_restarts))
    nim_restart_delta = {
        service: int(after_nim_restarts.get(service, 0)) - int(before_nim_restarts.get(service, 0))
        for service in nim_services
    }
    return {
        "restart_delta_by_component": restart_delta,
        "oom_events_after": after.get("oom_events") or [],
        "nim_restart_delta_by_service": nim_restart_delta,
        "nim_oom_events_after": (after.get("local_nim_runtime") or {}).get("oom_events") or [],
    }


def _cluster_delta_hard_reasons(delta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if any(value > 0 for value in (delta.get("restart_delta_by_component") or {}).values()):
        reasons.append("pod restarted during run")
    if delta.get("oom_events_after"):
        reasons.append("pod reported OOMKilled")
    if any(value > 0 for value in (delta.get("nim_restart_delta_by_service") or {}).values()):
        reasons.append("local NIM pod restarted during run")
    if delta.get("nim_oom_events_after"):
        reasons.append("local NIM pod reported OOMKilled")
    return reasons


def _add_cluster_delta_metrics(metrics: dict[str, Any], delta: dict[str, Any]) -> None:
    restart_delta = delta.get("restart_delta_by_component") or {}
    nim_restart_delta = delta.get("nim_restart_delta_by_service") or {}
    metrics["restart_delta_by_component"] = restart_delta
    metrics["restart_delta_total"] = sum(int(value or 0) for value in restart_delta.values())
    metrics["nim_restart_delta_by_service"] = nim_restart_delta
    metrics["nim_restart_delta_total"] = sum(int(value or 0) for value in nim_restart_delta.values())
    metrics["oom_events_count"] = len(delta.get("oom_events_after") or [])
    metrics["nim_oom_events_count"] = len(delta.get("nim_oom_events_after") or [])


def _hpas_scaling_active(hpas: Any) -> tuple[bool, list[str]]:
    if isinstance(hpas, dict) and hpas.get("error"):
        return False, [f"HPA query failed: {hpas['error']}"]
    if not hpas:
        return False, ["No HPA resources found for the Helm release"]
    inactive: list[str] = []
    for hpa in hpas:
        name = str(hpa.get("name") or "unknown")
        conditions = hpa.get("conditions") or []
        active = any(c.get("type") == "ScalingActive" and c.get("status") == "True" for c in conditions)
        if not active:
            inactive.append(f"{name} ScalingActive is not True")
    return not inactive, inactive


def _wait_for_hpas_scaling_active(
    manager: HelmServiceManager,
    *,
    timeout_s: int = 300,
    interval_s: float = 10.0,
) -> tuple[Any, bool, list[str]]:
    deadline = time.monotonic() + max(1, timeout_s)
    last_hpas: Any = []
    last_errors: list[str] = []
    while True:
        last_hpas = capture_cluster_snapshot(manager).get("hpas")
        active, last_errors = _hpas_scaling_active(last_hpas)
        if active or time.monotonic() >= deadline:
            return last_hpas, active, last_errors
        time.sleep(max(1.0, interval_s))


def _sample_overview(service_url: str) -> dict[str, Any]:
    return _http_get_json(f"{service_url.rstrip('/')}/v1/dashboard/api/overview", timeout_s=10.0)


class _ClusterSampler:
    def __init__(self, manager: HelmServiceManager, service_url: str, *, interval_s: float) -> None:
        self.manager = manager
        self.service_url = service_url.rstrip("/")
        self.interval_s = max(1.0, float(interval_s))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bo20-cluster-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s + 1.0))
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {"timestamp": _utc_now_iso()}
            try:
                sample["overview"] = _sample_overview(self.service_url)
            except Exception as exc:
                sample["overview_error"] = f"{type(exc).__name__}: {exc}"
            try:
                cluster = capture_cluster_snapshot(self.manager)
                sample["cluster"] = cluster
                sample["hpas"] = cluster.get("hpas")
                sample["top_pods"] = cluster.get("top_pods")
            except Exception as exc:
                sample["hpa_error"] = f"{type(exc).__name__}: {exc}"
            self.samples.append(sample)
            self._stop.wait(self.interval_s)


def _start_worker_metrics_forwards(manager: HelmServiceManager) -> dict[str, str]:
    urls: dict[str, str] = {}
    for offset, component in enumerate(("realtime", "batch"), start=10):
        services = manager.find_services_by_component(component)
        if not services:
            raise RuntimeError(f"No {component} service found for split Helm release")
        local_port = manager.local_port + offset
        manager.start_port_forward(services[0], local_port=local_port, remote_port=manager.remote_port)
        urls[component] = f"http://localhost:{local_port}"
    return urls


def run_split_preflight(
    *,
    manager: HelmServiceManager,
    service_url: str,
    repo_root: Path,
    require_main: bool,
    inventory: Bo20Inventory,
    require_service_monitor: bool = True,
    require_external_metrics: bool = True,
    require_hpa_active: bool = True,
    hpa_active_timeout_s: int = 300,
    nim_backend: str = "local",
) -> dict[str, Any]:
    nim_backend = _normalize_nim_backend(nim_backend)
    errors: list[str] = []
    skipped_checks: list[str] = []
    branch = _current_branch(repo_root)
    if require_main and branch != "main":
        errors.append(f"Repo branch must be main, got {branch!r}")

    overview: dict[str, Any] = {}
    try:
        overview = _sample_overview(service_url)
        if overview.get("mode") != "gateway":
            errors.append(f"Expected split gateway mode, got {overview.get('mode')!r}")
        backends = overview.get("backends") or {}
        for backend in ("realtime", "batch"):
            backend_health = backends.get(backend)
            if not _backend_health_ok(backend_health):
                errors.append(f"Gateway backend {backend!r} is not healthy: {backend_health!r}")
    except Exception as exc:
        errors.append(f"Gateway overview check failed: {type(exc).__name__}: {exc}")

    selector = f"app.kubernetes.io/instance={manager.release_name}"
    sm: dict[str, Any] = {"items": []}
    if require_service_monitor:
        sm = _kubectl_json(
            manager,
            ["get", "servicemonitor", "-n", manager.namespace, "-l", selector, "-o", "json"],
            timeout_s=60,
        )
        if sm.get("error"):
            errors.append(f"ServiceMonitor query failed: {sm['error']}")
        elif not (sm.get("items") or []):
            errors.append("No ServiceMonitor resources found for split Helm release")
    else:
        skipped_checks.append("ServiceMonitor disabled by Helm override")

    if require_external_metrics:
        ext_rc, _ext_out, ext_err = _kubectl_text(
            manager,
            ["get", "--raw", "/apis/external.metrics.k8s.io/v1beta1"],
            timeout_s=60,
        )
        if ext_rc != 0:
            errors.append(f"External Metrics API is not registered: {ext_err.strip()}")
    else:
        skipped_checks.append("External Metrics API disabled by Helm override")

    if require_hpa_active:
        hpas, hpa_active, hpa_errors = _wait_for_hpas_scaling_active(
            manager,
            timeout_s=hpa_active_timeout_s,
        )
    else:
        hpas = capture_cluster_snapshot(manager).get("hpas")
        hpa_active, hpa_errors = _hpas_scaling_active(hpas)
    if not hpa_active and require_hpa_active:
        errors.extend(hpa_errors)
    elif not hpa_active:
        skipped_checks.extend(hpa_errors)

    worker_metrics_urls: dict[str, str] = {}
    try:
        worker_metrics_urls = _start_worker_metrics_forwards(manager)
        for component, base_url in worker_metrics_urls.items():
            metrics_text = _http_get_text(f"{base_url}/metrics", timeout_s=10.0)
            if "nemo_retriever_pool_queue_depth" not in metrics_text:
                errors.append(f"{component} /metrics did not expose pool queue metrics")
            stats = _http_get_json(f"{base_url}/v1/admin/pool_stats", timeout_s=10.0)
            if component not in (stats.get("pools") or {}):
                errors.append(f"{component} /v1/admin/pool_stats did not include its pool")
    except Exception as exc:
        errors.append(f"Worker metrics scrape check failed: {type(exc).__name__}: {exc}")

    nim_status = _collect_nim_status(manager)
    nimcache_status = _collect_nimcache_status(manager)
    nvcf_preflight: dict[str, Any] | None = None
    local_nim_validation: dict[str, Any] | None = None
    local_nim_runtime: dict[str, Any] | None = None
    if nim_backend == "local":
        expected_services = set(_expected_local_nim_services(manager).values())
        for item in nim_status:
            name = str(item.get("name") or "")
            if name in expected_services and not item.get("exists"):
                errors.append(f"Expected NIMService {name} does not exist")
            if item.get("exists") and not item.get("ready"):
                errors.append(f"NIMService {item.get('name')} is not Ready")
        local_nim_validation = _validate_local_nim_specs(manager, nim_status)
        errors.extend(local_nim_validation.get("errors") or [])
        local_nim_runtime = _collect_local_nim_runtime_snapshot(manager)
        errors.extend(_local_nim_runtime_preflight_errors(local_nim_runtime))
    else:
        skipped_checks.append(f"Local NIM readiness skipped for nim_backend={nim_backend}")

    if nim_backend == "nvcf":
        nvcf_preflight = _nvcf_preflight(manager)
        errors.extend(nvcf_preflight.get("errors") or [])

    return {
        "success": not errors,
        "errors": errors,
        "skipped_checks": skipped_checks,
        "branch": branch,
        "latest_commit": last_commit(),
        "inventory": asdict(inventory),
        "overview": overview,
        "hpas": hpas,
        "service_monitor": _summarize_k8s_names(sm),
        "worker_metrics_urls": worker_metrics_urls,
        "nim_status": nim_status,
        "nimcache_status": nimcache_status,
        "local_nim_validation": local_nim_validation,
        "local_nim_runtime": local_nim_runtime,
        "nim_backend": nim_backend,
        "nvcf_preflight": nvcf_preflight,
    }


def _summarize_k8s_names(payload: dict[str, Any]) -> Any:
    if payload.get("error"):
        return payload
    return [((item.get("metadata") or {}).get("name")) for item in payload.get("items", []) or []]


def _helm_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _expect_service_monitor(helm_set: dict[str, Any]) -> bool:
    explicit_enabled = _helm_bool(helm_set.get("serviceMonitor.enabled"))
    auto_enabled = _helm_bool(helm_set.get("serviceMonitor.autoEnableInSplitMode"))
    if explicit_enabled is True:
        return True
    if auto_enabled is False:
        return False
    return True


def _expect_external_metrics(helm_set: dict[str, Any]) -> bool:
    backend = str(helm_set.get("autoscaling.queueDepth.backend", "prometheus-adapter")).strip().lower()
    return backend == "prometheus-adapter"


def _backend_health_ok(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, dict):
        return False
    status = str(value.get("status", "")).strip().lower()
    code = value.get("code")
    return status == "ok" and (code is None or int(code) < 400)


def _collect_nim_status(manager: HelmServiceManager) -> list[dict[str, Any]]:
    if not manager._crd_exists("nimservices.apps.nvidia.com"):
        return [{"crd": "nimservices.apps.nvidia.com", "exists": False, "ready": None}]
    results: list[dict[str, Any]] = []
    for name in manager.NIM_OPERATOR_RESOURCES:
        payload = _kubectl_json(manager, ["get", "nimservice", name, "-n", manager.namespace, "-o", "json"])
        if payload.get("error"):
            results.append({"name": name, "exists": False, "ready": None})
            continue
        conditions = ((payload.get("status") or {}).get("conditions") or [])
        ready = any(cond.get("type") == "Ready" and cond.get("status") == "True" for cond in conditions)
        spec = payload.get("spec") or {}
        image = spec.get("image") or {}
        results.append(
            {
                "name": name,
                "exists": True,
                "ready": ready,
                "conditions": conditions,
                "image": {
                    "repository": image.get("repository"),
                    "tag": str(image.get("tag")) if image.get("tag") is not None else None,
                    "pull_policy": image.get("pullPolicy"),
                },
                "env": spec.get("env") or [],
            }
        )
    return results


def _collect_nimcache_status(manager: HelmServiceManager) -> list[dict[str, Any]]:
    if not manager._crd_exists("nimcaches.apps.nvidia.com"):
        return [{"crd": "nimcaches.apps.nvidia.com", "exists": False, "ready": None}]
    names = sorted(set(manager.NIM_OPERATOR_RESOURCES) | set(_expected_local_nim_services(manager).values()))
    results: list[dict[str, Any]] = []
    for name in names:
        payload = _kubectl_json(manager, ["get", "nimcache", name, "-n", manager.namespace, "-o", "json"])
        if payload.get("error"):
            results.append({"name": name, "exists": False, "ready": None})
            continue
        status = payload.get("status") or {}
        conditions = status.get("conditions") or []
        ready = any(
            cond.get("type") in {"NIM_CACHE_JOB_COMPLETED", "Ready"} and cond.get("status") == "True"
            for cond in conditions
        )
        source = (((payload.get("spec") or {}).get("source") or {}).get("ngc") or {})
        results.append(
            {
                "name": name,
                "exists": True,
                "ready": ready,
                "conditions": conditions,
                "model_puller": source.get("modelPuller"),
                "status": status,
            }
        )
    return results


def _configured_nim_service_name(manager: HelmServiceManager, key: str) -> str:
    default = CORE_NIM_OPERATOR_SERVICES[key]
    return str(manager.config.helm_set.get(f"nimOperator.{key}.nimServiceName") or default)


def _expected_local_nim_services(manager: HelmServiceManager) -> dict[str, str]:
    if _helm_bool(manager.config.helm_set.get("nims.enabled")) is False:
        return {}
    expected: dict[str, str] = {}
    for key in CORE_NIM_OPERATOR_SERVICES:
        enabled = _helm_bool(manager.config.helm_set.get(f"nimOperator.{key}.enabled"))
        if enabled is not False:
            expected[key] = _configured_nim_service_name(manager, key)
    return expected


def _expected_nim_env_from_helm_set(helm_set: dict[str, Any], key: str) -> dict[str, str]:
    indexed: dict[int, dict[str, str]] = {}
    for raw_key, value in helm_set.items():
        match = _HELM_ENV_SET_RE.match(raw_key)
        if not match or match.group(1) != key:
            continue
        index = int(match.group(2))
        indexed.setdefault(index, {})[match.group(3)] = str(value)
    expected: dict[str, str] = {}
    for entry in indexed.values():
        name = entry.get("name")
        if name and "value" in entry:
            expected[name] = entry["value"]
    return expected


def _validate_local_nim_specs(manager: HelmServiceManager, nim_status: list[dict[str, Any]]) -> dict[str, Any]:
    status_by_name = {str(item.get("name")): item for item in nim_status if item.get("name")}
    errors: list[str] = []
    checked: list[dict[str, Any]] = []
    for key, service_name in _expected_local_nim_services(manager).items():
        item = status_by_name.get(service_name)
        if not item or not item.get("exists"):
            errors.append(f"Expected local NIMService {service_name} for {key} was not found")
            continue
        image = item.get("image") or {}
        expected_repo = manager.config.helm_set.get(f"nimOperator.{key}.image.repository")
        expected_tag = manager.config.helm_set.get(f"nimOperator.{key}.image.tag")
        if expected_repo and image.get("repository") != str(expected_repo):
            errors.append(
                f"NIMService {service_name} repository mismatch: {image.get('repository')!r} != {expected_repo!r}"
            )
        if expected_tag and str(image.get("tag")) != str(expected_tag):
            errors.append(f"NIMService {service_name} tag mismatch: {image.get('tag')!r} != {expected_tag!r}")

        actual_env = {str(env.get("name")): str(env.get("value")) for env in item.get("env") or [] if env.get("name")}
        expected_env = _expected_nim_env_from_helm_set(manager.config.helm_set, key)
        for env_name, env_value in expected_env.items():
            if actual_env.get(env_name) != env_value:
                errors.append(
                    f"NIMService {service_name} env {env_name} mismatch: {actual_env.get(env_name)!r} != {env_value!r}"
                )
        checked.append(
            {
                "key": key,
                "service_name": service_name,
                "ready": item.get("ready"),
                "image": image,
                "validated_env": expected_env,
            }
        )
    return {"success": not errors, "errors": errors, "checked": checked}


def _namespace_pods_payload(manager: HelmServiceManager) -> dict[str, Any]:
    return _kubectl_json(manager, ["get", "pods", "-n", manager.namespace, "-o", "json"])


def _nim_service_for_pod(pod: dict[str, Any], expected_services: set[str]) -> str | None:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    haystack = " ".join([str(metadata.get("name") or ""), *map(str, labels.keys()), *map(str, labels.values())])
    for service_name in sorted(expected_services, key=len, reverse=True):
        if service_name in haystack:
            return service_name
    return None


def _collect_local_nim_runtime_snapshot(manager: HelmServiceManager) -> dict[str, Any]:
    expected = _expected_local_nim_services(manager)
    expected_services = set(expected.values())
    if not expected_services:
        return {"expected_services": [], "pods": [], "restart_totals_by_service": {}, "oom_events": []}
    payload = _namespace_pods_payload(manager)
    if payload.get("error"):
        return {"expected_services": sorted(expected_services), "error": payload.get("error")}

    pods: list[dict[str, Any]] = []
    restart_totals: dict[str, int] = {name: 0 for name in expected_services}
    seen_services: set[str] = set()
    oom_events: list[dict[str, str]] = []
    for pod in payload.get("items", []) or []:
        service_name = _nim_service_for_pod(pod, expected_services)
        if not service_name:
            continue
        seen_services.add(service_name)
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        pod_name = str(metadata.get("name") or "")
        containers: list[dict[str, Any]] = []
        for item in status.get("containerStatuses") or []:
            restart_count = int(item.get("restartCount") or 0)
            restart_totals[service_name] = restart_totals.get(service_name, 0) + restart_count
            container = str(item.get("name") or "")
            containers.append(
                {
                    "name": container,
                    "ready": item.get("ready"),
                    "restart_count": restart_count,
                    "image": item.get("image"),
                    "image_id": item.get("imageID"),
                    "state": item.get("state"),
                    "last_state": item.get("lastState"),
                }
            )
            for state in (item.get("state") or {}, item.get("lastState") or {}):
                terminated = state.get("terminated") or {}
                waiting = state.get("waiting") or {}
                reason = str(terminated.get("reason") or waiting.get("reason") or "")
                if reason == "OOMKilled":
                    oom_events.append(
                        {
                            "pod": pod_name,
                            "service_name": service_name,
                            "container": container,
                            "reason": reason,
                        }
                    )
        pods.append(
            {
                "name": pod_name,
                "service_name": service_name,
                "phase": status.get("phase"),
                "pod_ip": status.get("podIP"),
                "containers": containers,
            }
        )
    return {
        "expected_services": sorted(expected_services),
        "pods": sorted(pods, key=lambda item: str(item.get("name") or "")),
        "restart_totals_by_service": dict(sorted(restart_totals.items())),
        "oom_events": oom_events,
        "missing_services": sorted(expected_services - seen_services),
    }


def _local_nim_runtime_preflight_errors(snapshot: dict[str, Any]) -> list[str]:
    if snapshot.get("error"):
        return [f"Local NIM pod query failed: {snapshot['error']}"]
    errors: list[str] = []
    for service_name in snapshot.get("missing_services") or []:
        errors.append(f"No running pod was found for local NIMService {service_name}")
    for service_name, count in sorted((snapshot.get("restart_totals_by_service") or {}).items()):
        if int(count or 0) > 0:
            errors.append(f"Local NIMService {service_name} had {count} restart(s) before measurement")
    for pod in snapshot.get("pods") or []:
        if pod.get("phase") != "Running":
            errors.append(f"Local NIM pod {pod.get('name')} was phase {pod.get('phase')}")
        for container in pod.get("containers") or []:
            if container.get("ready") is not True:
                errors.append(f"Local NIM pod {pod.get('name')} container {container.get('name')} was not Ready")
    if snapshot.get("oom_events"):
        errors.append("Local NIM pod reported OOMKilled before measurement")
    return errors


def _release_pods_payload(manager: HelmServiceManager) -> dict[str, Any]:
    selector = f"app.kubernetes.io/instance={manager.release_name}"
    return _kubectl_json(manager, ["get", "pods", "-n", manager.namespace, "-l", selector, "-o", "json"])


def _release_local_nim_pods(manager: HelmServiceManager) -> list[str]:
    payload = _release_pods_payload(manager)
    if payload.get("error"):
        return []
    names: list[str] = []
    markers = ("nemotron-", "llama-nemotron", "parakeet", "-nim")
    for pod in payload.get("items", []) or []:
        metadata = pod.get("metadata") or {}
        name = str(metadata.get("name") or "")
        component = _pod_component(pod)
        if component == "nim" or any(marker in name for marker in markers):
            names.append(name)
    return sorted(names)


def _release_pods_by_component(manager: HelmServiceManager, components: set[str]) -> list[str]:
    payload = _release_pods_payload(manager)
    if payload.get("error"):
        return []
    names: list[str] = []
    for pod in payload.get("items", []) or []:
        if _pod_component(pod) in components:
            name = ((pod.get("metadata") or {}).get("name") or "")
            if name:
                names.append(str(name))
    return sorted(names)


def _rendered_service_config_texts(manager: HelmServiceManager) -> dict[str, str]:
    selector = f"app.kubernetes.io/instance={manager.release_name}"
    payload = _kubectl_json(manager, ["get", "configmap", "-n", manager.namespace, "-l", selector, "-o", "json"])
    if payload.get("error"):
        return {"<error>": str(payload.get("error"))}
    configs: dict[str, str] = {}
    for item in payload.get("items", []) or []:
        name = str((item.get("metadata") or {}).get("name") or "")
        data = item.get("data") or {}
        body = data.get("retriever-service.yaml")
        if name and body:
            configs[name] = str(body)
    return configs


def _nvcf_api_key_env_checks(manager: HelmServiceManager) -> dict[str, Any]:
    checked: list[str] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for pod in _release_pods_by_component(manager, {"realtime", "batch"}):
        checked.append(pod)
        result = subprocess.run(
            manager.kubectl_cmd
            + ["exec", "-n", manager.namespace, pod, "--", "sh", "-c", 'test -n "$NVIDIA_API_KEY"'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            missing.append(pod)
            detail = (result.stderr or result.stdout).strip()
            if detail:
                errors.append({"pod": pod, "error": detail[:300]})
    return {"checked_pods": checked, "missing_pods": missing, "errors": errors, "success": bool(checked) and not missing}


def _nvcf_preflight(manager: HelmServiceManager) -> dict[str, Any]:
    configs = _rendered_service_config_texts(manager)
    combined = "\n".join(configs.values())
    missing_endpoints = [url for url in NVCF_HOSTED_ENDPOINTS.values() if url not in combined]
    local_nim_pods = _release_local_nim_pods(manager)
    api_key_env = _nvcf_api_key_env_checks(manager)
    errors: list[str] = []
    if local_nim_pods:
        errors.append(f"NVCF backend expected no release-managed local NIM pods, found {local_nim_pods}")
    if missing_endpoints:
        errors.append(f"NVCF backend config is missing hosted endpoint(s): {missing_endpoints}")
    if not api_key_env.get("success"):
        errors.append("NVCF backend worker pods do not all have non-empty NVIDIA_API_KEY")
    return {
        "success": not errors,
        "errors": errors,
        "configmaps_checked": sorted(configs),
        "missing_endpoints": missing_endpoints,
        "local_nim_pods": local_nim_pods,
        "api_key_env": api_key_env,
    }


def collect_endpoint_log_evidence(manager: HelmServiceManager, *, since: str = "30m") -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "since": since,
        "checked_pods": [],
        "pattern_counts": {pattern: 0 for pattern in ENDPOINT_EVIDENCE_PATTERNS},
        "sample_lines": [],
    }
    for pod in _release_pod_names(manager):
        if not any(marker in pod for marker in ("gateway", "realtime", "batch")):
            continue
        evidence["checked_pods"].append(pod)
        rc, stdout, stderr = _kubectl_text(
            manager,
            ["logs", pod, "-n", manager.namespace, "--all-containers", f"--since={since}"],
            timeout_s=180,
        )
        if rc != 0:
            evidence.setdefault("errors", []).append({"pod": pod, "error": stderr.strip() or stdout.strip()})
            continue
        for line in stdout.splitlines():
            matched = [pattern for pattern in ENDPOINT_EVIDENCE_PATTERNS if pattern in line]
            if not matched:
                continue
            for pattern in matched:
                evidence["pattern_counts"][pattern] += 1
            if len(evidence["sample_lines"]) < 40:
                evidence["sample_lines"].append({"pod": pod, "line": line[:500]})
    evidence["hosted_nvcf_observed"] = int(evidence["pattern_counts"].get("ai.api.nvidia.com", 0)) > 0
    evidence["local_nim_observed"] = int(evidence["pattern_counts"].get("http://nemotron-", 0)) > 0 or int(
        evidence["pattern_counts"].get("NIM endpoint http://nemotron-", 0)
    ) > 0
    return evidence



def _queue_put_best_effort(target_queue: Any, payload: dict[str, Any]) -> None:
    try:
        target_queue.put(payload)
    except Exception:
        logger.debug("Could not publish UX event", exc_info=True)


def _ux_service_job_worker(
    result_queue: mp.Queue,
    event_queue: mp.Queue,
    start_event: Any,
    *,
    job_index: int,
    base_url: str,
    documents: list[str],
    max_concurrency: int,
    return_results: bool,
    api_token: str | None,
) -> None:
    """Run one bo20 job while publishing live job/doc/result-fetch events."""
    handler = _RetryCountingHandler()
    client_logger = logging.getLogger("nemo_retriever.service.client")
    previous_level = client_logger.level
    client_logger.addHandler(handler)
    client_logger.setLevel(logging.DEBUG)
    payload = JobRunResult(job_index=job_index)
    try:
        start_event.wait()
        from nemo_retriever.service.service_ingestor import ServiceIngestor

        ingestor = ServiceIngestor(
            base_url=base_url,
            documents=documents,
            max_concurrency=max_concurrency,
            api_token=api_token,
        )
        started = time.monotonic()
        result_rows_from_events = 0
        dataframe_rows = 0
        for evt in ingestor.ingest_stream(retain_results=return_results):
            event_type = str(evt.get("event") or "")
            job_id = evt.get("job_id") or payload.job_id
            doc_id = evt.get("document_id") or evt.get("id")
            if job_id:
                payload.job_id = str(job_id)
            _queue_put_best_effort(
                event_queue,
                {
                    "timestamp": _utc_now_iso(),
                    "job_index": job_index,
                    "event": event_type,
                    "job_id": payload.job_id,
                    "document_id": str(doc_id) if doc_id else None,
                    "status": evt.get("status"),
                    "result_rows": evt.get("result_rows"),
                    "filename": evt.get("filename"),
                    "error": evt.get("error"),
                },
            )

            if event_type == "job_created":
                payload.job_id = str(evt.get("job_id") or payload.job_id or "") or None
                continue
            if event_type == "job_finalized":
                payload.job_status = "completed"
                continue
            if event_type == "job_partial":
                payload.job_status = "partial_success"
                continue
            if event_type == "job_failed":
                payload.job_status = "failed"
                continue
            if event_type in {"job_progress", "job_started"}:
                continue
            if event_type == "upload_complete":
                payload.uploaded += 1
                continue
            if event_type == "upload_failed":
                payload.upload_failed += 1
                fname = str(evt.get("filename") or "?")
                payload.failures.append([fname, f"upload failed: {evt.get('error', 'unknown')}"])
                continue
            if event_type != "document_complete":
                continue

            status = str(evt.get("status") or "completed")
            doc_id = str(evt.get("document_id") or evt.get("id") or "")
            if status == "failed":
                payload.failed += 1
                payload.failures.append([doc_id or "?", str(evt.get("error") or "unknown error")])
                continue

            payload.completed += 1
            result_rows_from_events += int(evt.get("result_rows") or 0)
            if return_results and doc_id:
                fetch_started = time.monotonic()
                fetch_record: dict[str, Any] = {
                    "timestamp": _utc_now_iso(),
                    "job_index": job_index,
                    "job_id": payload.job_id,
                    "document_id": doc_id,
                    "success": False,
                }
                try:
                    rows = ingestor._materialize_completed_document(doc_id, return_results=True)
                    row_count = len(rows or [])
                    dataframe_rows += row_count
                    fetch_record.update(
                        {
                            "success": True,
                            "rows": row_count,
                            "elapsed_s": round(time.monotonic() - fetch_started, 3),
                        }
                    )
                except Exception as exc:
                    fetch_record.update(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_s": round(time.monotonic() - fetch_started, 3),
                        }
                    )
                    payload.failures.append([doc_id, f"return_results: {exc}"])
                payload.result_fetches.append(fetch_record)
                _queue_put_best_effort(event_queue, {"event": "result_fetch", **fetch_record})

        payload.elapsed_s = time.monotonic() - started
        payload.document_ids = [str(item) for item in getattr(ingestor, "_document_ids", [])]
        payload.result_rows = result_rows_from_events
        payload.dataframe_rows = dataframe_rows if return_results else None
        payload.retry_429_count = handler.retry_429_count
        payload.transient_retry_count = handler.transient_retry_count
        _infer_terminal_status(payload, expected_documents=len(documents))
        payload.hard_failure_reasons = _job_hard_failure_reasons(payload, expected_documents=len(documents))
        payload.hard_failure = bool(payload.hard_failure_reasons)
        payload.success = not payload.hard_failure
    except BaseException as exc:  # pragma: no cover - exercised by live cluster runs
        payload.exception = f"{type(exc).__name__}: {exc}"
        payload.traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload.hard_failure = True
        payload.success = False
        payload.hard_failure_reasons = [payload.exception]
        payload.retry_429_count = handler.retry_429_count
        payload.transient_retry_count = handler.transient_retry_count
    finally:
        client_logger.removeHandler(handler)
        client_logger.setLevel(previous_level)
        _queue_put_best_effort(
            event_queue,
            {
                "timestamp": _utc_now_iso(),
                "job_index": job_index,
                "event": "job_done",
                "job_id": payload.job_id,
                "success": payload.success,
                "hard_failure": payload.hard_failure,
            },
        )
        result_queue.put(asdict(payload))


def _new_ux_state() -> dict[str, Any]:
    return {
        "job_ids": set(),
        "document_ids": set(),
        "completed_document_ids": set(),
        "failed_document_ids": set(),
        "result_fetch_success": 0,
        "result_fetch_failed": 0,
        "events_by_type": {},
    }


def _record_ux_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = str(event.get("event") or "unknown")
    state["events_by_type"][event_type] = int(state["events_by_type"].get(event_type, 0)) + 1
    job_id = event.get("job_id")
    if job_id:
        state["job_ids"].add(str(job_id))
    doc_id = event.get("document_id")
    if doc_id:
        doc_id = str(doc_id)
        state["document_ids"].add(doc_id)
        if event_type == "document_complete" and event.get("status") == "failed":
            state["failed_document_ids"].add(doc_id)
        elif event_type == "document_complete":
            state["completed_document_ids"].add(doc_id)
    if event_type == "result_fetch":
        if event.get("success"):
            state["result_fetch_success"] += 1
        else:
            state["result_fetch_failed"] += 1


def _json_probe_summary(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"body_type": type(body).__name__}
    if "job_id" in body and "counts" in body:
        return {
            "job_id": body.get("job_id"),
            "status": body.get("status"),
            "expected_documents": body.get("expected_documents"),
            "counts": body.get("counts"),
            "document_count": len(body.get("documents") or []),
        }
    if "items" in body and "terminal" in body:
        status_counts: dict[str, int] = {}
        for info in (body.get("items") or {}).values():
            status = str((info or {}).get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "total": body.get("total"),
            "terminal": body.get("terminal"),
            "pending": body.get("pending"),
            "status_counts": status_counts,
        }
    if "status" in body and "mode" in body:
        return {"status": body.get("status"), "mode": body.get("mode")}
    if "job_summary" in body or "pool_stats" in body:
        return {
            "mode": body.get("mode"),
            "job_summary": body.get("job_summary"),
            "pool_stats": body.get("pool_stats"),
            "backends": body.get("backends"),
        }
    if "status" in body and "result_data" in body:
        return {
            "id": body.get("id"),
            "status": body.get("status"),
            "result_rows": body.get("result_rows"),
            "has_result_data": bool(body.get("result_data")),
            "result_data_len": len(body.get("result_data") or []),
        }
    return {key: body.get(key) for key in list(body)[:10]}


def _probe_json_endpoint(
    service_url: str,
    *,
    method: str,
    path: str,
    api_token: str | None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    import httpx

    url = f"{service_url.rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s, headers=headers) as client:
            resp = client.request(method, url, json=json_body)
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
            try:
                body: Any = resp.json()
            except Exception:
                body = resp.text[:500]
            return {
                "ok": 200 <= resp.status_code < 300,
                "status_code": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "summary": _json_probe_summary(body),
            }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _probe_sse_endpoint(
    service_url: str,
    *,
    job_id: str,
    api_token: str | None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    import httpx

    url = f"{service_url.rstrip('/')}/v1/ingest/job/{job_id}/events"
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    started = time.monotonic()
    try:
        timeout = httpx.Timeout(timeout_s, connect=timeout_s, read=timeout_s, write=timeout_s)
        with httpx.Client(timeout=timeout, headers=headers) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    return {
                        "ok": False,
                        "status_code": resp.status_code,
                        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
                        "body": resp.read(500).decode(errors="replace") if hasattr(resp, "read") else "",
                    }
                first_line = None
                for line in resp.iter_lines():
                    if line:
                        first_line = line[:300]
                        break
                return {
                    "ok": first_line is not None,
                    "status_code": resp.status_code,
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
                    "first_line": first_line,
                }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_ux_probe_sample(
    *,
    service_url: str,
    api_token: str | None,
    state: dict[str, Any],
    return_results: bool,
) -> dict[str, Any]:
    job_ids = sorted(state["job_ids"])
    document_ids = sorted(state["document_ids"])
    completed_ids = sorted(state["completed_document_ids"])
    sample: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "known_jobs": len(job_ids),
        "known_documents": len(document_ids),
        "known_completed_documents": len(completed_ids),
        "known_failed_documents": len(state["failed_document_ids"]),
        "events_by_type": dict(sorted(state["events_by_type"].items())),
        "result_fetch_success": state["result_fetch_success"],
        "result_fetch_failed": state["result_fetch_failed"],
    }
    sample["health"] = _probe_json_endpoint(
        service_url,
        method="GET",
        path="/v1/health",
        api_token=api_token,
        timeout_s=5.0,
    )
    sample["overview"] = _probe_json_endpoint(
        service_url,
        method="GET",
        path="/v1/dashboard/api/overview",
        api_token=api_token,
        timeout_s=5.0,
    )
    if job_ids:
        sample["job_aggregate"] = _probe_json_endpoint(
            service_url,
            method="GET",
            path=f"/v1/ingest/job/{job_ids[0]}?include_documents=true",
            api_token=api_token,
            timeout_s=5.0,
        )
        sample["sse"] = _probe_sse_endpoint(
            service_url,
            job_id=job_ids[0],
            api_token=api_token,
            timeout_s=5.0,
        )
    else:
        sample["job_aggregate"] = {"skipped": "no job id observed yet"}
        sample["sse"] = {"skipped": "no job id observed yet"}
    if document_ids:
        sample["batch_status"] = _probe_json_endpoint(
            service_url,
            method="POST",
            path="/v1/ingest/status/batch",
            api_token=api_token,
            json_body={"ids": document_ids[:1000]},
            timeout_s=5.0,
        )
    else:
        sample["batch_status"] = {"skipped": "no document ids observed yet"}
    if return_results:
        sample["single_status"] = {
            "skipped": "GET /v1/ingest/status/{id} consumes result_data; using client result_fetch events instead"
        }
    elif completed_ids:
        sample["single_status"] = _probe_json_endpoint(
            service_url,
            method="GET",
            path=f"/v1/ingest/status/{completed_ids[0]}",
            api_token=api_token,
            timeout_s=5.0,
        )
    else:
        sample["single_status"] = {"skipped": "no completed document id observed yet"}
    return sample


def _probe_latency_stats(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    ok = 0
    failed = 0
    first_error: str | None = None
    for sample in samples:
        probe = sample.get(key)
        if not isinstance(probe, dict) or probe.get("skipped"):
            continue
        if probe.get("elapsed_ms") is not None:
            values.append(float(probe["elapsed_ms"]))
        if probe.get("ok"):
            ok += 1
        else:
            failed += 1
            first_error = first_error or str(probe.get("error") or probe.get("status_code") or "failed")
    return {
        "ok": ok,
        "failed": failed,
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "max_ms": round(max(values), 1) if values else None,
        "first_error": first_error,
    }


def summarize_ux_probe_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {key: _probe_latency_stats(samples, key) for key in ("health", "overview", "job_aggregate", "batch_status", "sse", "single_status")}
    max_known_documents = max((int(sample.get("known_documents") or 0) for sample in samples), default=0)
    max_completed = max((int(sample.get("known_completed_documents") or 0) for sample in samples), default=0)
    max_failed = max((int(sample.get("known_failed_documents") or 0) for sample in samples), default=0)
    summary.update(
        {
            "samples": len(samples),
            "max_known_documents": max_known_documents,
            "max_known_completed_documents": max_completed,
            "max_known_failed_documents": max_failed,
            "result_fetch_success_max": max((int(s.get("result_fetch_success") or 0) for s in samples), default=0),
            "result_fetch_failed_max": max((int(s.get("result_fetch_failed") or 0) for s in samples), default=0),
        }
    )
    return summary


def run_ux_probe_round(
    *,
    service_url: str,
    documents: list[str],
    return_results: bool,
    n: int,
    job_max_concurrency: int,
    api_token: str | None,
    run_timeout_s: int,
    total_pages_per_job: int,
    manager: HelmServiceManager,
    sample_interval_s: float,
    ux_probe_interval_s: float,
    idle_timeout_s: int,
    metric_urls: dict[str, str] | None = None,
) -> dict[str, Any]:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    result_queue: mp.Queue = ctx.Queue()
    event_queue: mp.Queue = ctx.Queue()
    start_event = ctx.Event()
    processes: list[mp.Process] = []
    for job_index in range(n):
        proc = ctx.Process(
            target=_ux_service_job_worker,
            kwargs={
                "result_queue": result_queue,
                "event_queue": event_queue,
                "start_event": start_event,
                "job_index": job_index,
                "base_url": service_url,
                "documents": documents,
                "max_concurrency": job_max_concurrency,
                "return_results": return_results,
                "api_token": api_token,
            },
        )
        proc.start()
        processes.append(proc)

    before = capture_cluster_snapshot(manager)
    http_before = _capture_http_status_counts(metric_urls)
    cluster_sampler = _ClusterSampler(manager, service_url, interval_s=sample_interval_s)
    state = _new_ux_state()
    ux_events: list[dict[str, Any]] = []
    ux_probe_samples: list[dict[str, Any]] = []
    job_payloads: dict[int, dict[str, Any]] = {}
    started_at = _utc_now_iso()
    wall_start = time.monotonic()
    next_probe = wall_start
    deadline = wall_start + run_timeout_s
    cluster_sampler.start()
    start_event.set()

    def _drain_queues() -> None:
        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                break
            ux_events.append(event)
            _record_ux_event(state, event)
        while True:
            try:
                item = result_queue.get_nowait()
            except queue.Empty:
                break
            job_payloads[int(item.get("job_index", len(job_payloads)))] = item

    while time.monotonic() < deadline:
        _drain_queues()
        now = time.monotonic()
        if now >= next_probe:
            ux_probe_samples.append(
                _run_ux_probe_sample(
                    service_url=service_url,
                    api_token=api_token,
                    state=state,
                    return_results=return_results,
                )
            )
            next_probe = now + max(1.0, ux_probe_interval_s)
        if all(not proc.is_alive() for proc in processes) and len(job_payloads) >= n:
            break
        time.sleep(0.5)

    timed_out_indexes: set[int] = set()
    for proc in processes:
        if proc.is_alive():
            timed_out_indexes.add(processes.index(proc))
            proc.terminate()
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
    _drain_queues()
    if not ux_probe_samples or (time.monotonic() - next_probe + max(1.0, ux_probe_interval_s)) > 2.0:
        ux_probe_samples.append(
            _run_ux_probe_sample(
                service_url=service_url,
                api_token=api_token,
                state=state,
                return_results=return_results,
            )
        )

    wall_s = time.monotonic() - wall_start
    cluster_samples = cluster_sampler.stop()
    after = capture_cluster_snapshot(manager)
    http_after = _capture_http_status_counts(metric_urls)
    http_delta = _http_status_delta(http_before, http_after)

    for job_index, proc in enumerate(processes):
        if job_index not in job_payloads:
            result = JobRunResult(
                job_index=job_index,
                timed_out=job_index in timed_out_indexes,
                exit_code=proc.exitcode,
                hard_failure=True,
                success=False,
            )
            _infer_terminal_status(result, expected_documents=len(documents))
            result.hard_failure_reasons = _job_hard_failure_reasons(result, expected_documents=len(documents))
            job_payloads[job_index] = asdict(result)
        else:
            job_payloads[job_index]["exit_code"] = proc.exitcode
            job_fields = JobRunResult.__dataclass_fields__
            job = JobRunResult(**{k: v for k, v in job_payloads[job_index].items() if k in job_fields})
            _infer_terminal_status(job, expected_documents=len(documents))
            job.hard_failure_reasons = _job_hard_failure_reasons(job, expected_documents=len(documents))
            job.hard_failure = bool(job.hard_failure_reasons)
            job.success = not job.hard_failure
            job_payloads[job_index] = asdict(job)

    idle_start = time.monotonic()
    idle_after_run, idle_error = wait_until_idle(service_url, timeout_s=idle_timeout_s)
    idle_wait_s = time.monotonic() - idle_start

    ordered_jobs = [job_payloads[idx] for idx in sorted(job_payloads)]
    hard_reasons = sorted({reason for job in ordered_jobs for reason in job.get("hard_failure_reasons", [])})
    delta = cluster_delta(before, after)
    hard_reasons.extend(_cluster_delta_hard_reasons(delta))
    if not idle_after_run:
        hard_reasons.append(f"service did not become idle after run: {idle_error or 'timeout'}")

    metrics = _round_metrics(ordered_jobs, total_pages_per_job=total_pages_per_job, wall_s=wall_s)
    _add_cluster_delta_metrics(metrics, delta)
    metrics.update(_resource_pressure_metrics(before, after, cluster_samples))
    metrics["http_status_delta"] = http_delta
    metrics.update(_http_status_delta_summary(http_delta))
    fetches = [fetch for job in ordered_jobs for fetch in (job.get("result_fetches") or [])]
    fetch_latencies = [float(fetch.get("elapsed_s")) for fetch in fetches if fetch.get("elapsed_s") is not None]
    metrics.update(
        {
            "result_fetch_attempts": len(fetches),
            "result_fetch_success": sum(1 for fetch in fetches if fetch.get("success")),
            "result_fetch_failed": sum(1 for fetch in fetches if not fetch.get("success")),
            "result_fetch_latency_p50_s": _percentile(fetch_latencies, 50),
            "result_fetch_latency_p95_s": _percentile(fetch_latencies, 95),
            "result_fetch_latency_max_s": round(max(fetch_latencies), 3) if fetch_latencies else None,
            "ux_probe_summary": summarize_ux_probe_samples(ux_probe_samples),
        }
    )
    attribution = classify_failure_attribution(ordered_jobs, hard_reasons, metrics, http_delta)
    metrics["failure_attribution"] = attribution.get("primary")
    hard_failure = bool(hard_reasons)
    return {
        "return_results": return_results,
        "n": n,
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "wall_s": round(wall_s, 3),
        "success": not hard_failure,
        "hard_failure": hard_failure,
        "hard_failure_reasons": sorted(set(hard_reasons)),
        "failure_attribution": attribution,
        "job_results": ordered_jobs,
        "metrics": metrics,
        "cluster_before": before,
        "cluster_after": after,
        "cluster_delta": delta,
        "cluster_samples": cluster_samples,
        "ux_events": ux_events,
        "ux_probe_samples": ux_probe_samples,
        "idle_after_run": idle_after_run,
        "idle_wait_s": round(idle_wait_s, 3),
        "idle_error": idle_error,
    }

def _service_job_worker(
    result_queue: mp.Queue,
    start_event: Any,
    *,
    job_index: int,
    base_url: str,
    documents: list[str],
    max_concurrency: int,
    return_results: bool,
    api_token: str | None,
) -> None:
    handler = _RetryCountingHandler()
    client_logger = logging.getLogger("nemo_retriever.service.client")
    previous_level = client_logger.level
    client_logger.addHandler(handler)
    client_logger.setLevel(logging.DEBUG)
    payload = JobRunResult(job_index=job_index)
    try:
        start_event.wait()
        from nemo_retriever.service.service_ingestor import ServiceIngestor

        ingestor = ServiceIngestor(
            base_url=base_url,
            documents=documents,
            max_concurrency=max_concurrency,
            api_token=api_token,
        )
        result, failures, traces = ingestor.ingest(
            return_failures=True,
            return_traces=True,
            return_results=return_results,
        )
        trace_events = [str(evt.get("event") or "") for evt in traces]
        doc_complete_events = [
            evt for evt in traces if evt.get("event") == "document_complete" and evt.get("status") == "completed"
        ]
        doc_failed_events = [
            evt for evt in traces if evt.get("event") == "document_complete" and evt.get("status") == "failed"
        ]
        dataframe = getattr(result, "dataframe", None)
        dataframe_rows = int(len(dataframe)) if dataframe is not None else None
        payload.job_id = getattr(result, "job_id", None)
        payload.job_status = getattr(result, "job_status", None)
        payload.elapsed_s = float(getattr(result, "elapsed_s", 0.0) or 0.0)
        payload.uploaded = trace_events.count("upload_complete")
        payload.completed = len(doc_complete_events)
        payload.failed = len(doc_failed_events)
        payload.upload_failed = trace_events.count("upload_failed")
        payload.failures = [[str(a), str(b)] for a, b in failures]
        payload.document_ids = [str(item) for item in getattr(result, "document_ids", [])]
        payload.result_rows = sum(int(evt.get("result_rows") or 0) for evt in doc_complete_events)
        payload.dataframe_rows = dataframe_rows
        payload.retry_429_count = handler.retry_429_count
        payload.transient_retry_count = handler.transient_retry_count
        payload.hard_failure_reasons = _job_hard_failure_reasons(payload, expected_documents=len(documents))
        payload.hard_failure = bool(payload.hard_failure_reasons)
        payload.success = not payload.hard_failure
    except BaseException as exc:  # pragma: no cover - exercised via process integration on real runs
        payload.exception = f"{type(exc).__name__}: {exc}"
        payload.traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload.hard_failure = True
        payload.success = False
        payload.hard_failure_reasons = [payload.exception]
        payload.retry_429_count = handler.retry_429_count
        payload.transient_retry_count = handler.transient_retry_count
    finally:
        client_logger.removeHandler(handler)
        client_logger.setLevel(previous_level)
        result_queue.put(asdict(payload))


async def _proxy_dry_run_job_async(
    *,
    base_url: str,
    documents: list[str],
    max_concurrency: int,
    api_token: str | None,
) -> JobRunResult:
    import asyncio

    import httpx

    payload = JobRunResult(job_index=-1)
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    timeout = httpx.Timeout(timeout=None, connect=30.0)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    started = time.monotonic()
    upload_failures: list[list[str]] = []
    document_ids: list[str] = []

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, limits=limits, headers=headers) as client:
        resp = await client.post(
            "/v1/ingest/job",
            json={"expected_documents": len(documents), "retain_results": False, "label": "bo20-proxy-dry-run"},
        )
        if resp.status_code >= 400:
            detail = resp.text[:500] if resp.text else "(empty)"
            raise RuntimeError(f"proxy dry-run job creation returned HTTP {resp.status_code}: {detail}")
        payload.job_id = resp.json().get("job_id")
        if not payload.job_id:
            raise RuntimeError(f"proxy dry-run job creation response missing job_id: {resp.text[:500]}")

        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _upload_one(filename: str) -> None:
            path = Path(filename)
            file_bytes = path.read_bytes()
            metadata = json.dumps({"filename": path.name})
            attempts = 0
            transient_attempts = 0
            while True:
                attempts += 1
                try:
                    response = await client.post(
                        f"/v1/ingest/job/{payload.job_id}/whole",
                        headers={DRY_RUN_HEADER: "true"},
                        files={"file": (path.name, file_bytes, "application/pdf")},
                        data={"metadata": metadata},
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    transient_attempts += 1
                    payload.transient_retry_count += 1
                    if transient_attempts > 5:
                        upload_failures.append([path.name, f"{type(exc).__name__}: {exc}"])
                        return
                    await asyncio.sleep(min(2 ** (transient_attempts - 1), 30.0))
                    continue

                if response.status_code == 429 and attempts <= 6:
                    payload.retry_429_count += 1
                    delay = float(response.headers.get("retry-after", "2") or 2)
                    await asyncio.sleep(delay)
                    continue
                if response.status_code >= 400:
                    detail = response.text[:500] if response.text else "(empty)"
                    upload_failures.append([path.name, f"HTTP {response.status_code}: {detail}"])
                    return
                body = response.json()
                doc_id = body.get("document_id") or body.get("page_id")
                if doc_id:
                    document_ids.append(str(doc_id))
                payload.uploaded += 1
                return

        async def _bounded_upload(filename: str) -> None:
            async with sem:
                await _upload_one(filename)

        await asyncio.gather(*(_bounded_upload(item) for item in documents))

    payload.elapsed_s = time.monotonic() - started
    payload.completed = payload.uploaded
    payload.failed = 0
    payload.upload_failed = len(upload_failures)
    payload.failures = upload_failures
    payload.document_ids = document_ids
    if payload.uploaded == len(documents) and not upload_failures:
        payload.job_status = "completed"
    payload.hard_failure_reasons = _job_hard_failure_reasons(payload, expected_documents=len(documents))
    payload.hard_failure = bool(payload.hard_failure_reasons)
    payload.success = not payload.hard_failure
    return payload


def _proxy_dry_run_job_worker(
    result_queue: mp.Queue,
    start_event: Any,
    *,
    job_index: int,
    base_url: str,
    documents: list[str],
    max_concurrency: int,
    api_token: str | None,
) -> None:
    try:
        import asyncio

        start_event.wait()
        payload = asyncio.run(
            _proxy_dry_run_job_async(
                base_url=base_url,
                documents=documents,
                max_concurrency=max_concurrency,
                api_token=api_token,
            )
        )
        payload.job_index = job_index
    except BaseException as exc:  # pragma: no cover - exercised via process integration on real runs
        payload = JobRunResult(
            job_index=job_index,
            exception=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            hard_failure=True,
            success=False,
        )
        payload.hard_failure_reasons = [payload.exception or type(exc).__name__]
    result_queue.put(asdict(payload))


def _infer_terminal_status(job: JobRunResult, *, expected_documents: int) -> None:
    if job.job_status:
        return
    if job.exception or job.timed_out or job.exit_code not in (None, 0):
        return
    if job.completed == expected_documents and job.failed == 0 and job.upload_failed == 0:
        job.job_status = "completed"


def _job_hard_failure_reasons(job: JobRunResult, *, expected_documents: int) -> list[str]:
    reasons: list[str] = []
    if job.exception:
        reasons.append(job.exception)
    if job.timed_out:
        reasons.append("job process timed out before terminal status")
    if job.upload_failed:
        reasons.append(f"{job.upload_failed} upload(s) failed")
    if job.failures:
        reasons.append(f"{len(job.failures)} service failure(s)")
    if job.job_status != "completed":
        reasons.append(f"job terminal status was {job.job_status or 'missing'}")
    if job.completed != expected_documents:
        reasons.append(f"completed {job.completed}/{expected_documents} documents")
    if job.exit_code not in (None, 0):
        reasons.append(f"job process exited with code {job.exit_code}")
    return reasons


def _round_failure_text(job_results: list[dict[str, Any]], hard_reasons: list[str]) -> str:
    parts: list[str] = list(hard_reasons)
    for job in job_results:
        for key in ("exception", "traceback"):
            if job.get(key):
                parts.append(str(job[key]))
        for failure in job.get("failures") or []:
            if isinstance(failure, (list, tuple)):
                parts.extend(str(item) for item in failure)
            else:
                parts.append(str(failure))
        parts.extend(str(item) for item in job.get("hard_failure_reasons") or [])
    return "\n".join(parts)


def _count_pattern(text: str, patterns: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(pattern.lower()) for pattern in patterns)


def classify_failure_attribution(
    job_results: list[dict[str, Any]],
    hard_reasons: list[str],
    metrics: dict[str, Any],
    http_delta: dict[str, dict[str, int]],
) -> dict[str, Any]:
    text = _round_failure_text(job_results, hard_reasons)
    retry_429_count = int(metrics.get("retry_429_count") or 0)
    gateway_http_errors = int(http_delta.get("gateway", {}).get("4xx", 0)) + int(
        http_delta.get("gateway", {}).get("5xx", 0)
    )
    worker_http_errors = sum(
        int(http_delta.get(component, {}).get(status, 0))
        for component in ("realtime", "batch")
        for status in ("4xx", "5xx")
    )
    signals = {
        "gateway_proxy": _count_pattern(
            text,
            (
                "Gateway failed",
                "Gateway timed out",
                "Gateway transport error",
                "forwarding to",
                "HTTP 502",
                "HTTP 504",
            ),
        )
        + gateway_http_errors,
        "worker_queue": _count_pattern(text, ("pipeline is at capacity", "Retry-After", "HTTP 429", "429 for"))
        + retry_429_count
        + int(http_delta.get("batch", {}).get("4xx", 0))
        + int(http_delta.get("realtime", {}).get("4xx", 0)),
        "result_fetch": _count_pattern(
            text,
            ("return_results", "failed to fetch/persist", "/v1/ingest/status", "document-result", "result fetch"),
        ),
        "hosted_nvcf": _count_pattern(text, ("ai.api.nvidia.com", "NVCF", "hosted NIM")),
        "local_nim": _count_pattern(
            text,
            ("http://nemotron-", "NIM endpoint http://nemotron-", "nemotron-page", "nemotron-table", "nemotron-ocr"),
        ),
        "downstream_nim_unknown": _count_pattern(text, ("GraphIngestionError", "remote NIM endpoint", "NIM endpoint")),
        "callback_sse": _count_pattern(
            text,
            ("job terminal status was missing", "unknown_document", "SSE", "Gateway callback", "callback"),
        ),
        "worker_http_errors": worker_http_errors,
    }
    if signals["gateway_proxy"]:
        primary = "gateway/proxy"
    elif signals["worker_queue"]:
        primary = "worker queue"
    elif signals["result_fetch"]:
        primary = "result-fetch/row materialization"
    elif signals["hosted_nvcf"]:
        primary = "hosted NVCF"
    elif signals["local_nim"]:
        primary = "local NIM"
    elif signals["downstream_nim_unknown"]:
        primary = "NIM/dependency"
    elif signals["callback_sse"]:
        primary = "job-tracker/SSE/callback"
    elif hard_reasons:
        primary = "unknown"
    else:
        primary = "none"
    return {"primary": primary, "signals": signals}


def run_concurrency_round(
    *,
    service_url: str,
    documents: list[str],
    return_results: bool,
    n: int,
    attempt: str,
    job_max_concurrency: int,
    api_token: str | None,
    run_timeout_s: int,
    total_pages_per_job: int,
    manager: HelmServiceManager,
    sample_interval_s: float,
    idle_timeout_s: int,
    phase: str | None = None,
    proxy_dry_run: bool = False,
    metric_urls: dict[str, str] | None = None,
) -> SweepRoundResult:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    result_queue: mp.Queue = ctx.Queue()
    start_event = ctx.Event()
    processes: list[mp.Process] = []
    for job_index in range(n):
        if proxy_dry_run:
            target = _proxy_dry_run_job_worker
            kwargs = {
                "result_queue": result_queue,
                "start_event": start_event,
                "job_index": job_index,
                "base_url": service_url,
                "documents": documents,
                "max_concurrency": job_max_concurrency,
                "api_token": api_token,
            }
        else:
            target = _service_job_worker
            kwargs = {
                "result_queue": result_queue,
                "start_event": start_event,
                "job_index": job_index,
                "base_url": service_url,
                "documents": documents,
                "max_concurrency": job_max_concurrency,
                "return_results": return_results,
                "api_token": api_token,
            }
        proc = ctx.Process(target=target, kwargs=kwargs)
        proc.start()
        processes.append(proc)

    before = capture_cluster_snapshot(manager)
    http_before = _capture_http_status_counts(metric_urls)
    sampler = _ClusterSampler(manager, service_url, interval_s=sample_interval_s)
    started_at = _utc_now_iso()
    wall_start = time.monotonic()
    sampler.start()
    start_event.set()

    deadline = wall_start + run_timeout_s
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.join(timeout=remaining)

    timed_out_indexes: set[int] = set()
    for proc in processes:
        if proc.is_alive():
            timed_out_indexes.add(processes.index(proc))
            proc.terminate()
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)

    wall_s = time.monotonic() - wall_start
    samples = sampler.stop()
    after = capture_cluster_snapshot(manager)
    http_after = _capture_http_status_counts(metric_urls)
    http_delta = _http_status_delta(http_before, http_after)

    job_payloads: dict[int, dict[str, Any]] = {}
    while True:
        try:
            item = result_queue.get_nowait()
        except queue.Empty:
            break
        job_payloads[int(item.get("job_index", len(job_payloads)))] = item

    for job_index, proc in enumerate(processes):
        if job_index not in job_payloads:
            result = JobRunResult(
                job_index=job_index,
                timed_out=job_index in timed_out_indexes,
                exit_code=proc.exitcode,
                hard_failure=True,
                success=False,
            )
            _infer_terminal_status(result, expected_documents=len(documents))
            result.hard_failure_reasons = _job_hard_failure_reasons(result, expected_documents=len(documents))
            job_payloads[job_index] = asdict(result)
        else:
            job_payloads[job_index]["exit_code"] = proc.exitcode
            job_fields = JobRunResult.__dataclass_fields__
            job = JobRunResult(**{k: v for k, v in job_payloads[job_index].items() if k in job_fields})
            _infer_terminal_status(job, expected_documents=len(documents))
            job.hard_failure_reasons = _job_hard_failure_reasons(job, expected_documents=len(documents))
            job.hard_failure = bool(job.hard_failure_reasons)
            job.success = not job.hard_failure
            job_payloads[job_index] = asdict(job)

    idle_start = time.monotonic()
    idle_after_run, idle_error = wait_until_idle(
        service_url,
        timeout_s=idle_timeout_s,
        require_jobs_idle=not proxy_dry_run,
    )
    idle_wait_s = time.monotonic() - idle_start

    ordered_jobs = [job_payloads[idx] for idx in sorted(job_payloads)]
    hard_reasons = sorted({reason for job in ordered_jobs for reason in job.get("hard_failure_reasons", [])})
    delta = cluster_delta(before, after)
    hard_reasons.extend(_cluster_delta_hard_reasons(delta))
    if not idle_after_run:
        hard_reasons.append(f"service did not become idle after run: {idle_error or 'timeout'}")

    metrics = _round_metrics(ordered_jobs, total_pages_per_job=total_pages_per_job, wall_s=wall_s)
    _add_cluster_delta_metrics(metrics, delta)
    metrics.update(_resource_pressure_metrics(before, after, samples))
    metrics["http_status_delta"] = http_delta
    metrics.update(_http_status_delta_summary(http_delta))
    saturation_reasons = _saturation_reasons(ordered_jobs, samples)
    attribution = classify_failure_attribution(ordered_jobs, hard_reasons, metrics, http_delta)
    for name, count in attribution.get("signals", {}).items():
        metrics[f"{name}_count"] = count
    metrics["failure_attribution"] = attribution.get("primary")
    hard_failure = bool(hard_reasons)
    if hard_failure:
        saturation_reasons.append("hard failure observed")

    finished_at = _utc_now_iso()
    return SweepRoundResult(
        return_results=return_results,
        n=n,
        attempt=attempt,
        started_at=started_at,
        finished_at=finished_at,
        wall_s=round(wall_s, 3),
        success=not hard_failure,
        hard_failure=hard_failure,
        hard_failure_reasons=sorted(set(hard_reasons)),
        saturation=bool(saturation_reasons),
        saturation_reasons=sorted(set(saturation_reasons)),
        job_results=ordered_jobs,
        metrics=metrics,
        cluster_before=before,
        cluster_after=after,
        cluster_delta=delta,
        samples=samples,
        idle_after_run=idle_after_run,
        idle_wait_s=round(idle_wait_s, 3),
        idle_error=idle_error,
        phase=phase,
        failure_attribution=attribution,
    )


def _round_metrics(job_results: list[dict[str, Any]], *, total_pages_per_job: int, wall_s: float) -> dict[str, Any]:
    elapsed_values = [float(job["elapsed_s"]) for job in job_results if job.get("elapsed_s") is not None]
    total_pages = total_pages_per_job * len(job_results)
    total_completed_docs = sum(int(job.get("completed") or 0) for job in job_results)
    total_failed_docs = sum(int(job.get("failed") or 0) for job in job_results)
    total_upload_failed = sum(int(job.get("upload_failed") or 0) for job in job_results)
    dataframe_rows = [job.get("dataframe_rows") for job in job_results if job.get("dataframe_rows") is not None]
    return {
        "jobs": len(job_results),
        "job_ids": [job.get("job_id") for job in job_results if job.get("job_id")],
        "completed_jobs": sum(1 for job in job_results if job.get("job_status") == "completed"),
        "failed_or_partial_jobs": sum(
            1 for job in job_results if job.get("job_status") in {"failed", "partial_success"}
        ),
        "documents_uploaded": sum(int(job.get("uploaded") or 0) for job in job_results),
        "documents_completed": total_completed_docs,
        "documents_failed": total_failed_docs,
        "upload_failed": total_upload_failed,
        "retry_429_count": sum(int(job.get("retry_429_count") or 0) for job in job_results),
        "transient_retry_count": sum(int(job.get("transient_retry_count") or 0) for job in job_results),
        "wall_s": round(wall_s, 3),
        "job_latency_p50_s": _percentile(elapsed_values, 50),
        "job_latency_p95_s": _percentile(elapsed_values, 95),
        "job_latency_max_s": round(max(elapsed_values), 3) if elapsed_values else None,
        "pages": total_pages,
        "pages_per_sec_wall": round(total_pages / wall_s, 3) if wall_s > 0 else None,
        "dataframe_rows_total": sum(int(v) for v in dataframe_rows) if dataframe_rows else None,
    }


def _saturation_reasons(job_results: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    retry_429_count = sum(int(job.get("retry_429_count") or 0) for job in job_results)
    if retry_429_count:
        reasons.append(f"{retry_429_count} client-observed HTTP 429 retry event(s)")

    max_ratios: dict[str, float] = {}
    for sample in samples:
        pool_stats = (((sample.get("overview") or {}).get("pool_stats")) or {})
        for pool_name, stats in pool_stats.items():
            ratio = stats.get("queue_depth_ratio")
            if ratio is None:
                max_queue = stats.get("max_queue_size") or 0
                depth = stats.get("queue_depth") or 0
                ratio = float(depth) / float(max_queue) if max_queue else 0.0
            max_ratios[pool_name] = max(float(ratio), max_ratios.get(pool_name, 0.0))
    for pool_name, ratio in sorted(max_ratios.items()):
        if ratio >= SATURATION_QUEUE_RATIO:
            reasons.append(f"{pool_name} queue ratio reached {ratio:.2f}")

    for sample in samples:
        hpas = sample.get("hpas")
        active, errors = _hpas_scaling_active(hpas)
        if hpas and not active:
            reasons.extend(errors)
    return reasons


def wait_until_idle(
    service_url: str,
    *,
    timeout_s: int,
    interval_s: float = 10.0,
    require_jobs_idle: bool = True,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            overview = _sample_overview(service_url)
            pool_stats = overview.get("pool_stats") or {}
            queues_idle = all(int(stats.get("queue_depth") or 0) == 0 for stats in pool_stats.values())
            summary = overview.get("job_summary") or {}
            jobs_idle = int(summary.get("pending") or 0) == 0 and int(summary.get("processing") or 0) == 0
            if queues_idle and (jobs_idle or not require_jobs_idle):
                return True, None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(interval_s)
    return False, last_error or "timeout waiting for queues/jobs to become idle"


def apply_latency_saturation(rounds: list[SweepRoundResult]) -> None:
    by_mode: dict[bool, list[SweepRoundResult]] = {}
    for item in rounds:
        if item.attempt == "primary":
            by_mode.setdefault(item.return_results, []).append(item)
    for mode, items in by_mode.items():
        baseline = next((item for item in items if item.n == 1 and not item.hard_failure), None)
        baseline_p95 = (baseline.metrics or {}).get("job_latency_p95_s") if baseline else None
        if not baseline_p95:
            continue
        for item in items:
            if item.n <= 1:
                continue
            p95 = item.metrics.get("job_latency_p95_s")
            if p95 is not None and float(p95) >= float(baseline_p95) * LATENCY_BLOWUP_FACTOR:
                item.saturation = True
                reason = (
                    f"job p95 latency {p95}s >= {LATENCY_BLOWUP_FACTOR:.1f}x "
                    f"single-job baseline {baseline_p95}s"
                )
                if reason not in item.saturation_reasons:
                    item.saturation_reasons.append(reason)
                    item.saturation_reasons.sort()


def summarize_thresholds(rounds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for mode in (False, True):
        label = str(mode)
        primary = [item for item in rounds if item.get("return_results") is mode and item.get("attempt") == "primary"]
        observed = [item for item in rounds if item.get("return_results") is mode]
        primary.sort(key=lambda item: int(item.get("n") or 0))
        observed.sort(key=lambda item: (int(item.get("n") or 0), 0 if item.get("attempt") == "primary" else 1))
        first_saturation = next((item for item in observed if item.get("saturation")), None)
        first_hard = next((item for item in observed if item.get("hard_failure")), None)
        primary_first_saturation = next((item for item in primary if item.get("saturation")), None)
        primary_first_hard = next((item for item in primary if item.get("hard_failure")), None)
        summary[label] = {
            "first_saturation_n": first_saturation.get("n") if first_saturation else None,
            "first_saturation_reasons": first_saturation.get("saturation_reasons", []) if first_saturation else [],
            "first_hard_failure_n": first_hard.get("n") if first_hard else None,
            "first_hard_failure_reasons": first_hard.get("hard_failure_reasons", []) if first_hard else [],
            "first_hard_failure_attempt": first_hard.get("attempt") if first_hard else None,
            "primary_first_saturation_n": primary_first_saturation.get("n") if primary_first_saturation else None,
            "primary_first_hard_failure_n": primary_first_hard.get("n") if primary_first_hard else None,
            "max_primary_n": max([int(item.get("n") or 0) for item in primary], default=0),
        }
    return summary


def build_split_harness_config(
    *,
    dataset_dir: str,
    artifacts_dir: str | None,
    helm_release: str,
    helm_namespace: str | None,
    helm_chart: str | None,
    helm_chart_version: str | None,
    helm_set: dict[str, Any],
    helm_timeout: int,
    readiness_timeout: int,
    helm_service_local_port: int,
    keep_up: bool,
    helm_bin: str,
    kubectl_bin: str,
    helm_sudo: bool,
    kubectl_sudo: bool,
    api_token: str | None,
) -> HarnessConfig:
    merged_helm_set = {**helm_set, "topology.mode": "split"}
    return HarnessConfig(
        dataset_dir=dataset_dir,
        dataset_label="bo20",
        preset="split-helm-concurrency",
        run_mode="service",
        input_type="pdf",
        evaluation_mode="none",
        recall_required=False,
        artifacts_dir=artifacts_dir,
        api_key=api_token,
        manage_service=True,
        keep_up=keep_up,
        helm_chart=helm_chart,
        helm_chart_version=helm_chart_version,
        helm_release=helm_release,
        helm_namespace=helm_namespace,
        helm_set=merged_helm_set,
        helm_timeout=helm_timeout,
        readiness_timeout=readiness_timeout,
        helm_service_local_port=helm_service_local_port,
        helm_bin=helm_bin,
        kubectl_bin=kubectl_bin,
        helm_sudo=helm_sudo,
        kubectl_sudo=kubectl_sudo,
    )


def _parse_helm_set(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--helm-set must use KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        stripped = raw.strip()
        if raw.lower() == "true":
            value: Any = True
        elif raw.lower() == "false":
            value = False
        elif raw.lower() == "null":
            value = None
        elif stripped.startswith(("[", "{")):
            value = json.loads(stripped)
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        parsed[key] = value
    return parsed


def _sweep_n_values(*, nim_backend: str, return_results: bool, max_n: int) -> list[int]:
    max_n = max(1, int(max_n))
    backend = _normalize_nim_backend(nim_backend)
    if backend != "nvcf":
        return list(range(1, max_n + 1))
    first_jump = 10 if return_results else 7
    if max_n < first_jump:
        return list(range(1, max_n + 1))
    seeds = [1, 10, 11] if return_results else [1, 7, 8]
    values = [item for item in seeds if item <= max_n]
    start = 12 if return_results else 9
    values.extend(range(start, max_n + 1))
    return sorted(set(values))


def run_sweep(
    *,
    manager: HelmServiceManager,
    service_url: str,
    documents: list[str],
    inventory: Bo20Inventory,
    max_n: int,
    job_max_concurrency: int,
    api_token: str | None,
    run_timeout_s: int,
    idle_timeout_s: int,
    sample_interval_s: float,
    return_results_modes: tuple[bool, ...] = (False, True),
    confirm_failures: bool = True,
    phase: str | None = None,
    nim_backend: str = "local",
    metric_urls: dict[str, str] | None = None,
) -> list[SweepRoundResult]:
    nim_backend = _normalize_nim_backend(nim_backend)
    rounds: list[SweepRoundResult] = []
    for return_results in return_results_modes:
        hard_failure_seen = False
        for n in _sweep_n_values(nim_backend=nim_backend, return_results=return_results, max_n=max_n):
            result = run_concurrency_round(
                service_url=service_url,
                documents=documents,
                return_results=return_results,
                n=n,
                attempt="primary",
                job_max_concurrency=job_max_concurrency,
                api_token=api_token,
                run_timeout_s=run_timeout_s,
                total_pages_per_job=inventory.total_pages,
                manager=manager,
                sample_interval_s=sample_interval_s,
                idle_timeout_s=idle_timeout_s,
                phase=phase,
                proxy_dry_run=nim_backend == "proxy-dry-run",
                metric_urls=metric_urls,
            )
            rounds.append(result)
            if result.hard_failure:
                hard_failure_seen = True
                confirmed_at_n = True
                if confirm_failures:
                    confirmed_at_n = False
                    for confirm_n in sorted({max(1, n - 1), n}):
                        confirm = run_concurrency_round(
                            service_url=service_url,
                            documents=documents,
                            return_results=return_results,
                            n=confirm_n,
                            attempt="confirm",
                            job_max_concurrency=job_max_concurrency,
                            api_token=api_token,
                            run_timeout_s=run_timeout_s,
                            total_pages_per_job=inventory.total_pages,
                            manager=manager,
                            sample_interval_s=sample_interval_s,
                            idle_timeout_s=idle_timeout_s,
                            phase=phase,
                            proxy_dry_run=nim_backend == "proxy-dry-run",
                            metric_urls=metric_urls,
                        )
                        rounds.append(confirm)
                        if confirm_n == n and confirm.hard_failure:
                            confirmed_at_n = True
                    if not confirmed_at_n:
                        hard_failure_seen = False
                        continue
                break
        if not hard_failure_seen:
            continue
    apply_latency_saturation(rounds)
    return rounds


def _mode_label(value: bool) -> str:
    return "True" if value else "False"


def render_markdown_report(payload: dict[str, Any]) -> str:
    rounds = payload.get("rounds") or []
    thresholds = payload.get("thresholds") or {}
    max_n = int(payload.get("config", {}).get("max_n") or DEFAULT_MAX_N)

    lines = [
        "# Split Helm bo20 Concurrency Qualification",
        "",
        f"Prepared: {payload.get('timestamp')}",
        f"Commit: {payload.get('latest_commit')}",
        f"Dataset: {payload.get('inventory', {}).get('dataset_dir')}",
        f"NIM backend: {payload.get('config', {}).get('nim_backend', 'local')}",
        "",
        "## Thresholds",
        "",
        "| return_results | first_saturation_n | first_hard_failure_n | notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for mode in (False, True):
        entry = thresholds.get(str(mode), {})
        sat = entry.get("first_saturation_n")
        hard = entry.get("first_hard_failure_n")
        notes = "; ".join((entry.get("first_hard_failure_reasons") or entry.get("first_saturation_reasons") or [])[:3])
        primary_hard = entry.get("primary_first_hard_failure_n")
        if hard and primary_hard and hard != primary_hard:
            notes = f"primary first hard failure at N={primary_hard}; {notes}" if notes else (
                f"primary first hard failure at N={primary_hard}"
            )
        lines.append(f"| {_mode_label(mode)} | {sat or 'none'} | {hard or 'none'} | {notes or ''} |")

    lines += [
        "",
        "## Primary Sweep",
        "",
        "| return_results | N | hard_failure | attribution | completed_jobs | docs_failed | 429 | gw_5xx | worker_5xx | nim_restarts | nim_ooms | nvcf | result_fetch | gw_mem_mib | p95_s | pages_per_sec |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rounds:
        if item.get("attempt") != "primary":
            continue
        metrics = item.get("metrics") or {}
        lines.append(
            "| {mode} | {n} | {hard} | {attr} | {jobs} | {docs_failed} | {r429} | {gw5} | {worker5} | {nim_restarts} | {nim_ooms} | {nvcf} | {result_fetch} | {gw_mem} | {p95} | {pps} |".format(
                mode=_mode_label(bool(item.get("return_results"))),
                n=item.get("n"),
                hard="yes" if item.get("hard_failure") else "no",
                attr=metrics.get("failure_attribution", ""),
                jobs=metrics.get("completed_jobs", ""),
                docs_failed=metrics.get("documents_failed", ""),
                r429=metrics.get("retry_429_count", ""),
                gw5=metrics.get("gateway_5xx_delta", ""),
                worker5=metrics.get("worker_5xx_delta", ""),
                nim_restarts=metrics.get("nim_restart_delta_total", ""),
                nim_ooms=metrics.get("nim_oom_events_count", ""),
                nvcf=metrics.get("hosted_nvcf_count", ""),
                result_fetch=metrics.get("result_fetch_count", ""),
                gw_mem=metrics.get("gateway_max_memory_mib", ""),
                p95=metrics.get("job_latency_p95_s", ""),
                pps=metrics.get("pages_per_sec_wall", ""),
            )
        )

    confirmation = [item for item in rounds if item.get("attempt") == "confirm"]
    if confirmation:
        lines += [
            "",
            "## Confirmation Runs",
            "",
            "| return_results | N | hard_failure | saturation | reasons |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for item in confirmation:
            reasons = "; ".join((item.get("hard_failure_reasons") or item.get("saturation_reasons") or [])[:4])
            lines.append(
                f"| {_mode_label(bool(item.get('return_results')))} | {item.get('n')} | "
                f"{'yes' if item.get('hard_failure') else 'no'} | "
                f"{'yes' if item.get('saturation') else 'no'} | {reasons} |"
            )

    hard_ns = [
        entry.get("first_hard_failure_n")
        for entry in thresholds.values()
        if entry.get("first_hard_failure_n") is not None
    ]
    if not hard_ns and max_n == 16:
        lines += ["", "No hard failure observed up to 16 simultaneous bo20 jobs"]

    lines += [
        "",
        "## Interpretation",
        "",
        _compare_return_results(thresholds),
        _report_conclusions(payload),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload.get('artifact_paths', {}).get('json', '')}`",
        f"- Markdown: `{payload.get('artifact_paths', {}).get('markdown', '')}`",
    ]
    return "\n".join(lines) + "\n"


def _compare_return_results(thresholds: dict[str, dict[str, Any]]) -> str:
    false_hard = thresholds.get("False", {}).get("first_hard_failure_n")
    true_hard = thresholds.get("True", {}).get("first_hard_failure_n")
    false_sat = thresholds.get("False", {}).get("first_saturation_n")
    true_sat = thresholds.get("True", {}).get("first_saturation_n")
    if true_hard and (not false_hard or true_hard < false_hard):
        return "`return_results=True` lowered the hard-failure threshold relative to `return_results=False`."
    if true_sat and (not false_sat or true_sat < false_sat):
        return "`return_results=True` saturated earlier, even if the hard-failure threshold did not move."
    return (
        "`return_results=True` did not lower the observed threshold in this sweep; "
        "compare latency and memory columns for overhead."
    )


def _report_conclusions(payload: dict[str, Any]) -> str:
    backend = str((payload.get("config") or {}).get("nim_backend") or "local")
    thresholds = payload.get("thresholds") or {}
    rounds = [item for item in payload.get("rounds") or [] if item.get("attempt") == "primary"]
    lines: list[str] = []
    if backend == "nvcf":
        false_hard = thresholds.get("False", {}).get("first_hard_failure_n")
        if false_hard is None or int(false_hard) > 8:
            lines.append("NVCF `return_results=False` passed beyond the prior local-NIM N=8 cliff, which points at local NIM capacity/configuration rather than gateway proxy admission.")
        else:
            lines.append("NVCF `return_results=False` did not pass beyond the prior local-NIM N=8 cliff; inspect attribution and HTTP columns before assigning blame to local NIMs.")
    if backend == "proxy-dry-run":
        first_hard = thresholds.get("False", {}).get("first_hard_failure_n")
        if first_hard is None:
            lines.append("Proxy dry-run did not hard-fail, so gateway body buffering/admission alone was not the observed cliff in this lane.")
        else:
            lines.append(f"Proxy dry-run hard-failed at N={first_hard}, which isolates the failure to gateway/proxy admission before NIM work.")
    primary_attrs = {((item.get("metrics") or {}).get("failure_attribution")) for item in rounds}
    if "result-fetch/row materialization" in primary_attrs:
        lines.append("A primary failure was attributed to result-fetch/row materialization: service-side completion and client result retrieval should be treated separately.")
    if "hosted NVCF" in primary_attrs:
        lines.append("A primary failure was attributed to hosted NVCF; check hosted 429/5xx/timeout signals before changing gateway sizing.")
    if "gateway/proxy" in primary_attrs:
        lines.append("A primary failure was attributed to gateway/proxy; inspect gateway HTTP deltas, RSS, and forward latency first.")
    return "\n".join(lines) if lines else "No additional backend-specific conclusion."


def _clean_recovery_helm_set(helm_set: dict[str, Any] | None) -> dict[str, Any]:
    return {**CLEAN_RECOVERY_HELM_SET_DEFAULTS, **(helm_set or {})}


def _resolve_cli_max_n(max_n: int | None, *, clean_page_elements_rerun: bool) -> int:
    if max_n is not None:
        return int(max_n)
    if clean_page_elements_rerun:
        return CLEAN_RECOVERY_MAX_N
    return DEFAULT_MAX_N


def _return_results_modes_from_option(value: str) -> tuple[bool, ...]:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"both", "all"}:
        return (False, True)
    if normalized in {"false", "off", "no", "recommended", "no-results"}:
        return (False,)
    if normalized in {"true", "on", "yes", "results"}:
        return (True,)
    raise ValueError("return results mode must be one of: false, true, both")


def _page_elements_ready_error(preflight: dict[str, Any]) -> str | None:
    for item in preflight.get("nim_status") or []:
        if item.get("name") == "nemotron-page-elements-v3":
            if item.get("exists") and item.get("ready"):
                return None
            return f"NIMService nemotron-page-elements-v3 was not Ready: {item!r}"
    return "NIMService nemotron-page-elements-v3 was not found in preflight status"


def _nonzero_restart_reasons(snapshot: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for component, count in sorted((snapshot.get("raw_restart_totals") or {}).items()):
        if int(count or 0) > 0:
            reasons.append(f"{component} had {count} restart(s) before measurement")
    return reasons


def _pod_names_from_payload(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in payload.get("items", []) or []:
        name = ((item.get("metadata") or {}).get("name") or "")
        if name:
            names.append(str(name))
    return names


def _release_pod_names(manager: HelmServiceManager) -> list[str]:
    selector = f"app.kubernetes.io/instance={manager.release_name}"
    payload = _kubectl_json(manager, ["get", "pods", "-n", manager.namespace, "-l", selector, "-o", "json"])
    return _pod_names_from_payload(payload)


def _namespace_pod_names(manager: HelmServiceManager) -> list[str]:
    payload = _kubectl_json(manager, ["get", "pods", "-n", manager.namespace, "-o", "json"])
    return _pod_names_from_payload(payload)


def _clean_recovery_log_pods(manager: HelmServiceManager) -> list[str]:
    names = set(_release_pod_names(manager))
    for pod in _namespace_pod_names(manager):
        if "nemotron-page-elements-v3" in pod:
            names.add(pod)
    return sorted(names)


def collect_page_elements_log_evidence(manager: HelmServiceManager, *, since: str = "90m") -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "since": since,
        "checked_pods": [],
        "pattern_counts": {pattern: 0 for pattern in PAGE_ELEMENTS_PATTERNS},
        "sample_lines": [],
    }
    for pod in _clean_recovery_log_pods(manager):
        if "batch" not in pod and "page-elements" not in pod:
            continue
        evidence["checked_pods"].append(pod)
        rc, stdout, stderr = _kubectl_text(
            manager,
            ["logs", pod, "-n", manager.namespace, "--all-containers", f"--since={since}"],
            timeout_s=180,
        )
        if rc != 0:
            evidence.setdefault("errors", []).append({"pod": pod, "error": stderr.strip() or stdout.strip()})
            continue
        for line in stdout.splitlines():
            matched = [pattern for pattern in PAGE_ELEMENTS_PATTERNS if pattern in line]
            if not matched:
                continue
            for pattern in matched:
                evidence["pattern_counts"][pattern] += 1
            if len(evidence["sample_lines"]) < 30:
                evidence["sample_lines"].append({"pod": pod, "line": line[:500]})
    evidence["page_elements_failure_observed"] = any(
        int(evidence["pattern_counts"].get(pattern, 0)) > 0
        for pattern in ("HTTPError: 500", "returned 500", "page_elements_v3", "Page Elements NIM")
    )
    return evidence


def _dump_phase_logs(manager: HelmServiceManager, phase_dir: Path) -> dict[str, Any]:
    logs_dir = phase_dir / "service_logs"
    try:
        rc = manager.dump_logs(phase_dir)
        logged_pods = {path.stem for path in logs_dir.glob("*.log")}
        for pod in _clean_recovery_log_pods(manager):
            if pod in logged_pods:
                continue
            log_cmd = manager.kubectl_cmd + ["logs", pod, "-n", manager.namespace, "--all-containers", "--tail=-1"]
            result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=120)
            (logs_dir / f"{pod}.log").write_text(result.stdout, encoding="utf-8")
            if result.stderr:
                (logs_dir / f"{pod}.err").write_text(result.stderr, encoding="utf-8")
        return {"path": str(logs_dir), "return_code": rc}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _wait_for_release_resources_deleted(
    manager: HelmServiceManager, *, timeout_s: int = 300, interval_s: int = 5
) -> bool:
    selector = f"app.kubernetes.io/instance={manager.release_name}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc, stdout, stderr = _kubectl_text(
            manager,
            ["get", "pods,pvc", "-n", manager.namespace, "-l", selector, "-o", "name"],
            timeout_s=30,
        )
        if rc != 0:
            detail = (stderr or stdout).lower()
            if "not found" in detail:
                return True
            logger.warning("Could not inspect release resources during cleanup wait: %s", stderr.strip())
            return False
        if not [line for line in stdout.splitlines() if line.strip()]:
            return True
        time.sleep(interval_s)
    return False


def _phase_had_hard_failure(phase_payload: dict[str, Any]) -> bool:
    smoke = phase_payload.get("health_smoke") or {}
    if smoke.get("hard_failure"):
        return True
    return any(item.get("hard_failure") for item in phase_payload.get("rounds") or [])


def _clean_phase_label(name: str) -> str:
    if name == "phase_a_return_results_false":
        return "Phase A"
    if name == "phase_b_return_results_true":
        return "Phase B"
    return name


def render_clean_recovery_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Clean Page-Elements Recovery Rerun",
        "",
        f"Prepared: {payload.get('timestamp')}",
        f"Commit: {payload.get('latest_commit')}",
        f"Dataset: {payload.get('inventory', {}).get('dataset_dir')}",
        "",
        "## Phase Summary",
        "",
        "| phase | return_results | health_smoke | first_hard_failure_n | page-elements evidence | notes |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for phase in payload.get("phases") or []:
        thresholds = phase.get("thresholds") or {}
        mode_label = str(bool(phase.get("return_results")))
        entry = thresholds.get(mode_label, {})
        smoke = phase.get("health_smoke") or {}
        evidence = phase.get("page_elements_evidence") or {}
        notes = phase.get("failure_reason") or phase.get("stop_reason") or ""
        lines.append(
            "| {phase} | {mode} | {smoke} | {hard} | {evidence} | {notes} |".format(
                phase=_clean_phase_label(str(phase.get("name") or "")),
                mode=_mode_label(bool(phase.get("return_results"))),
                smoke="fail" if smoke.get("hard_failure") else "pass" if smoke else "n/a",
                hard=entry.get("first_hard_failure_n") or "none",
                evidence="yes" if evidence.get("page_elements_failure_observed") else "no",
                notes=notes,
            )
        )

    lines += [
        "",
        "## Rounds",
        "",
        "| phase | attempt | return_results | N | hard_failure | completed_jobs | "
        "docs_completed | docs_failed | 429 | nim_restarts | nim_ooms | p95_s | pages_per_sec |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in payload.get("phases") or []:
        phase_rounds = ([phase.get("health_smoke")] if phase.get("health_smoke") else [])
        phase_rounds += list(phase.get("rounds") or [])
        for item in phase_rounds:
            metrics = item.get("metrics") or {}
            row_template = (
                "| {phase} | {attempt} | {mode} | {n} | {hard} | {jobs} | {docs} | "
                "{failed} | {r429} | {p95} | {pps} |"
            )
            lines.append(
                row_template.format(
                    phase=_clean_phase_label(str(phase.get("name") or "")),
                    attempt=item.get("attempt"),
                    mode=_mode_label(bool(item.get("return_results"))),
                    n=item.get("n"),
                    hard="yes" if item.get("hard_failure") else "no",
                    jobs=metrics.get("completed_jobs", ""),
                    docs=metrics.get("documents_completed", ""),
                    failed=metrics.get("documents_failed", ""),
                    r429=metrics.get("retry_429_count", ""),
                    nim_restarts=metrics.get("nim_restart_delta_total", ""),
                    nim_ooms=metrics.get("nim_oom_events_count", ""),
                    p95=metrics.get("job_latency_p95_s", ""),
                    pps=metrics.get("pages_per_sec_wall", ""),
                )
            )

    lines += ["", "## Interpretation", "", str(payload.get("interpretation") or "n/a")]
    lines += ["", "## Artifacts", ""]
    lines.append(f"- JSON: `{payload.get('artifact_paths', {}).get('json', '')}`")
    lines.append(f"- Markdown: `{payload.get('artifact_paths', {}).get('markdown', '')}`")
    for phase in payload.get("phases") or []:
        logs = phase.get("service_logs") or {}
        if logs.get("path"):
            lines.append(f"- {_clean_phase_label(str(phase.get('name') or ''))} logs: `{logs['path']}`")
    return "\n".join(lines) + "\n"


def _clean_recovery_interpretation(phases: list[dict[str, Any]], phase_max_n: int) -> str:
    if not phases:
        return "No phase ran."
    phase_a = phases[0]
    if _phase_had_hard_failure(phase_a):
        evidence = phase_a.get("page_elements_evidence") or {}
        if evidence.get("page_elements_failure_observed"):
            return (
                "A clean redeploy reproduced page-elements failure during Phase A. "
                "Treat the Phase A failure threshold as uncontaminated by the prior degraded deployment."
            )
        return "Phase A failed after a clean redeploy, but saved logs did not show page-elements 500 evidence."
    if len(phases) == 1:
        return f"Phase A passed through N={phase_max_n}; Phase B did not run."
    phase_b = phases[1]
    if _phase_had_hard_failure(phase_b):
        evidence = phase_b.get("page_elements_evidence") or {}
        if evidence.get("page_elements_failure_observed"):
            return (
                "Phase A passed cleanly, then Phase B reproduced page-elements failures from a separate redeploy. "
                "This isolates return_results=True from Phase A service-state contamination."
            )
        return "Phase A passed cleanly, while Phase B failed without page-elements 500 evidence in saved logs."
    return f"Both clean phases passed through N={phase_max_n}; no hard failure observed in the recovery rerun."


def run_clean_page_elements_recovery_rerun(
    *,
    dataset_dir: str,
    expected_pdfs: int,
    max_n: int,
    job_max_concurrency: int,
    run_timeout_s: int,
    idle_timeout_s: int,
    sample_interval_s: float,
    artifacts_dir: str | None,
    helm_release: str,
    helm_namespace: str | None,
    helm_chart: str | None,
    helm_chart_version: str | None,
    helm_set: dict[str, Any] | None,
    helm_timeout: int,
    readiness_timeout: int,
    helm_service_local_port: int,
    keep_up: bool,
    helm_bin: str,
    kubectl_bin: str,
    helm_sudo: bool,
    kubectl_sudo: bool,
    api_token: str | None,
    require_main: bool,
    dry_run: bool,
    return_results_modes: tuple[bool, ...] | None = None,
    nim_backend: str = "local",
) -> dict[str, Any]:
    repo_root = NEMO_RETRIEVER_ROOT.parent
    inventory = inventory_bo20_dataset(dataset_dir, expected_pdfs=expected_pdfs)
    documents = [str(path) for path in resolve_bo20_files(dataset_dir, expected_pdfs=expected_pdfs)]
    phase_max_n = max(1, int(max_n))
    nim_backend = _normalize_nim_backend(nim_backend)
    effective_return_results_modes = _effective_return_results_modes(nim_backend, return_results_modes)
    merged_helm_set = _helm_set_for_nim_backend(_clean_recovery_helm_set(helm_set), nim_backend)
    cfg = build_split_harness_config(
        dataset_dir=inventory.dataset_dir,
        artifacts_dir=artifacts_dir,
        helm_release=helm_release,
        helm_namespace=helm_namespace,
        helm_chart=helm_chart,
        helm_chart_version=helm_chart_version,
        helm_set=merged_helm_set,
        helm_timeout=helm_timeout,
        readiness_timeout=readiness_timeout,
        helm_service_local_port=helm_service_local_port,
        keep_up=keep_up,
        helm_bin=helm_bin,
        kubectl_bin=kubectl_bin,
        helm_sudo=helm_sudo,
        kubectl_sudo=kubectl_sudo,
        api_token=api_token,
    )
    session_dir = create_session_dir("bo20_page_elements_recovery", base_dir=artifacts_dir)
    first_manager = HelmServiceManager(cfg, repo_root=repo_root)
    payload: dict[str, Any] = {
        "success": False,
        "clean_page_elements_recovery": True,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "inventory": asdict(inventory),
        "config": {
            "max_n": phase_max_n,
            "requested_max_n": max_n,
            "job_max_concurrency": job_max_concurrency,
            "run_timeout_s": run_timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "sample_interval_s": sample_interval_s,
            "require_main": require_main,
            "nim_backend": nim_backend,
            "return_results_modes": list(effective_return_results_modes),
            "helm_release": helm_release,
            "helm_namespace": helm_namespace or helm_release,
            "helm_chart": helm_chart or str(NEMO_RETRIEVER_ROOT / "helm"),
            "helm_set": cfg.helm_set,
        },
        "helm_command": first_manager.format_command(first_manager.build_upgrade_command()),
        "phases": [],
    }
    if dry_run:
        payload["success"] = True
        payload["dry_run"] = True
        out_path = session_dir / "dry_run.json"
        write_json(out_path, payload)
        payload["artifact_paths"] = {"json": str(out_path)}
        return payload

    phase_defs = [
        ("phase_a_return_results_false" if not mode else "phase_b_return_results_true", mode)
        for mode in effective_return_results_modes
    ]
    for index, (phase_name, return_results) in enumerate(phase_defs):
        phase_dir = session_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        manager = HelmServiceManager(cfg, repo_root=repo_root)
        phase_payload: dict[str, Any] = {
            "name": phase_name,
            "return_results": return_results,
            "max_n": phase_max_n,
            "rounds": [],
        }
        payload["phases"].append(phase_payload)
        phase_keep_up = bool(keep_up and index == len(phase_defs) - 1)
        try:
            if nim_backend == "nvcf":
                phase_payload["nvcf_secret"] = ensure_nvcf_secret_from_env(manager)
            start_rc = manager.start()
            if start_rc != 0:
                phase_payload["failure_reason"] = f"managed split Helm service failed to become ready (exit {start_rc})"
                phase_payload["nim_status"] = _collect_nim_status(manager)
                phase_payload["nimcache_status"] = _collect_nimcache_status(manager)
                phase_payload["local_nim_runtime"] = _collect_local_nim_runtime_snapshot(manager)
                phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                payload["failure_reason"] = phase_payload["failure_reason"]
                return _write_success_payload(session_dir, payload)

            service_url = manager.get_service_url()
            phase_payload["service_url"] = service_url
            preflight = run_split_preflight(
                manager=manager,
                service_url=service_url,
                repo_root=repo_root,
                require_main=require_main,
                inventory=inventory,
                require_service_monitor=_expect_service_monitor(cfg.helm_set),
                require_external_metrics=_expect_external_metrics(cfg.helm_set),
                require_hpa_active=False,
                nim_backend=nim_backend,
            )
            phase_payload["preflight"] = preflight
            preflight_errors = list(preflight.get("errors") or [])
            if nim_backend == "local" and (page_error := _page_elements_ready_error(preflight)):
                preflight_errors.append(page_error)
            initial_snapshot = capture_cluster_snapshot(manager)
            phase_payload["initial_cluster_snapshot"] = initial_snapshot
            preflight_errors.extend(_nonzero_restart_reasons(initial_snapshot))
            if preflight_errors:
                phase_payload["failure_reason"] = "preflight failed"
                phase_payload["preflight_errors"] = preflight_errors
                phase_payload["page_elements_evidence"] = collect_page_elements_log_evidence(manager)
                phase_payload["endpoint_evidence"] = collect_endpoint_log_evidence(manager)
                phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                payload["failure_reason"] = "preflight failed"
                return _write_success_payload(session_dir, payload)

            metric_urls = {"gateway": service_url, **(preflight.get("worker_metrics_urls") or {})}
            smoke = run_concurrency_round(
                service_url=service_url,
                documents=documents,
                return_results=False,
                n=1,
                attempt="health_smoke",
                job_max_concurrency=job_max_concurrency,
                api_token=api_token,
                run_timeout_s=run_timeout_s,
                total_pages_per_job=inventory.total_pages,
                manager=manager,
                sample_interval_s=sample_interval_s,
                idle_timeout_s=idle_timeout_s,
                phase=phase_name,
                proxy_dry_run=nim_backend == "proxy-dry-run",
                metric_urls=metric_urls,
            )
            phase_payload["health_smoke"] = asdict(smoke)
            phase_payload["endpoint_evidence_after_smoke"] = collect_endpoint_log_evidence(manager)
            if nim_backend == "nvcf":
                evidence = phase_payload["endpoint_evidence_after_smoke"]
                if not evidence.get("hosted_nvcf_observed") or evidence.get("local_nim_observed"):
                    phase_payload["stop_reason"] = "NVCF smoke did not prove hosted endpoint usage"
                    phase_payload["page_elements_evidence"] = collect_page_elements_log_evidence(manager)
                    phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                    payload["stopped_after_phase"] = phase_name
                    break
            if smoke.hard_failure:
                phase_payload["stop_reason"] = "health smoke failed"
                phase_payload["page_elements_evidence"] = collect_page_elements_log_evidence(manager)
                phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                payload["stopped_after_phase"] = phase_name
                break

            rounds = run_sweep(
                manager=manager,
                service_url=service_url,
                documents=documents,
                inventory=inventory,
                max_n=phase_max_n,
                job_max_concurrency=job_max_concurrency,
                api_token=api_token,
                run_timeout_s=run_timeout_s,
                idle_timeout_s=idle_timeout_s,
                sample_interval_s=sample_interval_s,
                return_results_modes=(return_results,),
                confirm_failures=True,
                phase=phase_name,
                nim_backend=nim_backend,
                metric_urls=metric_urls,
            )
            round_dicts = [asdict(item) for item in rounds]
            phase_payload["rounds"] = round_dicts
            phase_payload["thresholds"] = summarize_thresholds(round_dicts)
            phase_payload["page_elements_evidence"] = collect_page_elements_log_evidence(manager)
            phase_payload["endpoint_evidence"] = collect_endpoint_log_evidence(manager)
            phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
            if any(item.hard_failure for item in rounds):
                phase_payload["stop_reason"] = "hard failure observed"
        finally:
            try:
                stop_rc = manager.stop(uninstall=not phase_keep_up)
                if stop_rc == 0 and not phase_keep_up:
                    deleted = _wait_for_release_resources_deleted(
                        manager,
                        timeout_s=min(300, max(60, int(readiness_timeout))),
                    )
                    if not deleted:
                        phase_payload["cleanup_wait_warning"] = (
                            "timed out waiting for release pods/PVCs to disappear after uninstall"
                        )
            except Exception as exc:
                phase_payload["cleanup_error"] = f"{type(exc).__name__}: {exc}"

    payload["success"] = True
    payload["interpretation"] = _clean_recovery_interpretation(payload["phases"], phase_max_n)
    return _write_success_payload(session_dir, payload)


def run_bo20_concurrency_qualification(
    *,
    dataset_dir: str = DEFAULT_BO20_DATASET,
    expected_pdfs: int = EXPECTED_BO20_PDFS,
    max_n: int = DEFAULT_MAX_N,
    job_max_concurrency: int = DEFAULT_JOB_MAX_CONCURRENCY,
    run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    artifacts_dir: str | None = None,
    helm_release: str = "nemo-retriever-bo20-concurrency",
    helm_namespace: str | None = None,
    helm_chart: str | None = None,
    helm_chart_version: str | None = None,
    helm_set: dict[str, Any] | None = None,
    helm_timeout: int = 900,
    readiness_timeout: int = 900,
    helm_service_local_port: int = 7670,
    keep_up: bool = False,
    helm_bin: str = "helm",
    kubectl_bin: str = "kubectl",
    helm_sudo: bool = False,
    kubectl_sudo: bool = False,
    api_token: str | None = None,
    require_main: bool = True,
    dry_run: bool = False,
    clean_page_elements_rerun: bool = False,
    return_results_modes: tuple[bool, ...] | None = None,
    nim_backend: str = "local",
) -> dict[str, Any]:
    nim_backend = _normalize_nim_backend(nim_backend)
    effective_return_results_modes = _effective_return_results_modes(nim_backend, return_results_modes)
    if clean_page_elements_rerun:
        return run_clean_page_elements_recovery_rerun(
            dataset_dir=dataset_dir,
            expected_pdfs=expected_pdfs,
            max_n=max_n,
            job_max_concurrency=job_max_concurrency,
            run_timeout_s=run_timeout_s,
            idle_timeout_s=idle_timeout_s,
            sample_interval_s=sample_interval_s,
            artifacts_dir=artifacts_dir,
            helm_release=helm_release,
            helm_namespace=helm_namespace,
            helm_chart=helm_chart,
            helm_chart_version=helm_chart_version,
            helm_set=helm_set,
            helm_timeout=helm_timeout,
            readiness_timeout=readiness_timeout,
            helm_service_local_port=helm_service_local_port,
            keep_up=keep_up,
            helm_bin=helm_bin,
            kubectl_bin=kubectl_bin,
            helm_sudo=helm_sudo,
            kubectl_sudo=kubectl_sudo,
            api_token=api_token,
            require_main=require_main,
            dry_run=dry_run,
            return_results_modes=effective_return_results_modes,
            nim_backend=nim_backend,
        )

    repo_root = NEMO_RETRIEVER_ROOT.parent
    inventory = inventory_bo20_dataset(dataset_dir, expected_pdfs=expected_pdfs)
    documents = [str(path) for path in resolve_bo20_files(dataset_dir, expected_pdfs=expected_pdfs)]
    cfg = build_split_harness_config(
        dataset_dir=inventory.dataset_dir,
        artifacts_dir=artifacts_dir,
        helm_release=helm_release,
        helm_namespace=helm_namespace,
        helm_chart=helm_chart,
        helm_chart_version=helm_chart_version,
        helm_set=_helm_set_for_nim_backend(helm_set or {}, nim_backend),
        helm_timeout=helm_timeout,
        readiness_timeout=readiness_timeout,
        helm_service_local_port=helm_service_local_port,
        keep_up=keep_up,
        helm_bin=helm_bin,
        kubectl_bin=kubectl_bin,
        helm_sudo=helm_sudo,
        kubectl_sudo=kubectl_sudo,
        api_token=api_token,
    )
    session_dir = create_session_dir("bo20_split_concurrency", base_dir=artifacts_dir)
    manager = HelmServiceManager(cfg, repo_root=repo_root)

    if dry_run:
        payload = {
            "success": True,
            "dry_run": True,
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "inventory": asdict(inventory),
            "helm_command": manager.format_command(manager.build_upgrade_command()),
            "config": {
                "max_n": max_n,
                "job_max_concurrency": job_max_concurrency,
                "run_timeout_s": run_timeout_s,
                "idle_timeout_s": idle_timeout_s,
                "sample_interval_s": sample_interval_s,
                "require_main": require_main,
                "nim_backend": nim_backend,
                "return_results_modes": list(effective_return_results_modes),
            },
        }
        out_path = session_dir / "dry_run.json"
        write_json(out_path, payload)
        payload["artifact_paths"] = {"json": str(out_path)}
        return payload

    payload: dict[str, Any] = {
        "success": False,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "inventory": asdict(inventory),
        "config": {
            "max_n": max_n,
            "job_max_concurrency": job_max_concurrency,
            "run_timeout_s": run_timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "sample_interval_s": sample_interval_s,
            "require_main": require_main,
            "nim_backend": nim_backend,
            "return_results_modes": list(effective_return_results_modes),
            "helm_release": helm_release,
            "helm_namespace": helm_namespace or helm_release,
            "helm_chart": helm_chart or str(NEMO_RETRIEVER_ROOT / "helm"),
            "helm_set": cfg.helm_set,
        },
        "rounds": [],
    }

    try:
        if nim_backend == "nvcf":
            payload["nvcf_secret"] = ensure_nvcf_secret_from_env(manager)
        start_rc = manager.start()
        if start_rc != 0:
            payload["failure_reason"] = f"managed split Helm service failed to become ready (exit {start_rc})"
            payload["nim_status"] = _collect_nim_status(manager)
            payload["nimcache_status"] = _collect_nimcache_status(manager)
            payload["local_nim_runtime"] = _collect_local_nim_runtime_snapshot(manager)
            return _write_failure_payload(session_dir, payload, manager)

        service_url = manager.get_service_url()
        payload["service_url"] = service_url
        preflight = run_split_preflight(
            manager=manager,
            service_url=service_url,
            repo_root=repo_root,
            require_main=require_main,
            inventory=inventory,
            require_service_monitor=_expect_service_monitor(cfg.helm_set),
            require_external_metrics=_expect_external_metrics(cfg.helm_set),
            hpa_active_timeout_s=min(300, max(60, int(readiness_timeout))),
            nim_backend=nim_backend,
        )
        payload["preflight"] = preflight
        if not preflight["success"]:
            payload["failure_reason"] = "preflight failed"
            payload["endpoint_evidence"] = collect_endpoint_log_evidence(manager)
            return _write_failure_payload(session_dir, payload, manager)

        metric_urls = {"gateway": service_url, **(preflight.get("worker_metrics_urls") or {})}
        if nim_backend == "nvcf":
            smoke = run_concurrency_round(
                service_url=service_url,
                documents=documents,
                return_results=False,
                n=1,
                attempt="nvcf_smoke",
                job_max_concurrency=job_max_concurrency,
                api_token=api_token,
                run_timeout_s=run_timeout_s,
                total_pages_per_job=inventory.total_pages,
                manager=manager,
                sample_interval_s=sample_interval_s,
                idle_timeout_s=idle_timeout_s,
                metric_urls=metric_urls,
            )
            payload["nvcf_smoke"] = asdict(smoke)
            endpoint_evidence = collect_endpoint_log_evidence(manager)
            payload["endpoint_evidence_after_smoke"] = endpoint_evidence
            if smoke.hard_failure or not endpoint_evidence.get("hosted_nvcf_observed") or endpoint_evidence.get("local_nim_observed"):
                payload["failure_reason"] = "NVCF smoke failed or did not prove hosted endpoint usage"
                return _write_failure_payload(session_dir, payload, manager)

        rounds = run_sweep(
            manager=manager,
            service_url=service_url,
            documents=documents,
            inventory=inventory,
            max_n=max_n,
            job_max_concurrency=job_max_concurrency,
            api_token=api_token,
            run_timeout_s=run_timeout_s,
            idle_timeout_s=idle_timeout_s,
            sample_interval_s=sample_interval_s,
            return_results_modes=effective_return_results_modes,
            nim_backend=nim_backend,
            metric_urls=metric_urls,
        )
        round_dicts = [asdict(item) for item in rounds]
        thresholds = summarize_thresholds(round_dicts)
        payload.update(
            {
                "success": True,
                "rounds": round_dicts,
                "thresholds": thresholds,
                "endpoint_evidence": collect_endpoint_log_evidence(manager),
            }
        )
        return _write_success_payload(session_dir, payload, manager)
    finally:
        try:
            manager.stop(uninstall=not keep_up)
        except Exception as exc:
            logger.warning("Managed Helm cleanup failed: %s", exc)


def _write_failure_payload(session_dir: Path, payload: dict[str, Any], manager: HelmServiceManager) -> dict[str, Any]:
    try:
        manager.dump_logs(session_dir)
    except Exception as exc:
        payload["service_log_collection_error"] = f"{type(exc).__name__}: {exc}"
    out_path = session_dir / "results.json"
    write_json(out_path, payload)
    payload["artifact_paths"] = {"json": str(out_path)}
    return payload


def _write_success_payload(
    session_dir: Path,
    payload: dict[str, Any],
    manager: HelmServiceManager | None = None,
) -> dict[str, Any]:
    out_path = session_dir / "results.json"
    md_path = session_dir / "report.md"
    logs_dir = session_dir / "service_logs"
    if manager is not None:
        try:
            log_rc = manager.dump_logs(session_dir)
            payload["service_logs"] = {"path": str(logs_dir), "return_code": log_rc}
        except Exception as exc:
            payload["service_log_collection_error"] = f"{type(exc).__name__}: {exc}"
    payload["artifact_paths"] = {"json": str(out_path), "markdown": str(md_path)}
    renderer = render_clean_recovery_report if payload.get("clean_page_elements_recovery") else render_markdown_report
    write_json(out_path, payload)
    md_path.write_text(renderer(payload), encoding="utf-8")
    return payload



def _ux_target_n(return_results: bool, *, false_n: int, true_n: int) -> int:
    return int(true_n if return_results else false_n)


def _ux_probe_failed(summary: dict[str, Any], probe_name: str) -> bool:
    probe = summary.get(probe_name) or {}
    return int(probe.get("failed") or 0) > 0 and int(probe.get("ok") or 0) == 0


def _ux_interpretation(phases: list[dict[str, Any]], *, nim_backend: str = "local") -> str:
    backend = _normalize_nim_backend(nim_backend)
    dependency_label = "hosted NVCF" if backend == "nvcf" else "local NIM" if backend == "local" else backend
    lines: list[str] = []
    for phase in phases:
        mode = _mode_label(bool(phase.get("return_results")))
        round_payload = phase.get("round") or {}
        metrics = round_payload.get("metrics") or {}
        summary = metrics.get("ux_probe_summary") or {}
        delta = round_payload.get("cluster_delta") or {}
        restarts = delta.get("restart_delta_by_component") or {}
        oom = delta.get("oom_events_after") or []
        if any(int(v or 0) > 0 for v in restarts.values()) or oom:
            lines.append(f"return_results={mode}: pod restart/OOM evidence was observed; treat this as service instability.")
        elif round_payload.get("hard_failure"):
            attr = metrics.get("failure_attribution") or "unknown"
            if attr == "result-fetch/row materialization":
                lines.append(
                    f"return_results={mode}: jobs reached terminal events, but result row retrieval failed or timed out; "
                    "health/status probes indicate whether the rest of the service stayed responsive."
                )
            elif attr in {"hosted NVCF", "local NIM", "NIM/dependency"}:
                lines.append(
                    f"return_results={mode}: hard failure was downstream extraction/{dependency_label}-side; "
                    "client status probes show the gateway behavior while documents fail."
                )
            else:
                lines.append(f"return_results={mode}: hard failure attribution was {attr}.")
        else:
            lines.append(f"return_results={mode}: no hard failure observed at the requested N.")
        if _ux_probe_failed(summary, "health"):
            lines.append(f"return_results={mode}: health checks failed throughout the probe window.")
        if _ux_probe_failed(summary, "batch_status"):
            lines.append(f"return_results={mode}: lightweight batch status polling failed throughout the probe window.")
        if _ux_probe_failed(summary, "sse"):
            lines.append(f"return_results={mode}: independent SSE attachment failed throughout the probe window.")
    return "\n".join(lines) if lines else "No UX probe phases ran."


def render_ux_probe_report(payload: dict[str, Any]) -> str:
    nim_backend = str((payload.get("config") or {}).get("nim_backend") or "local")
    lines = [
        "# bo20 User-Experience Failure Probe",
        "",
        f"Prepared: {payload.get('timestamp')}",
        f"Commit: {payload.get('latest_commit')}",
        f"Dataset: {payload.get('inventory', {}).get('dataset_dir')}",
        f"NIM backend: {nim_backend}",
        "",
        "## Summary",
        "",
        "| return_results | N | hard_failure | attribution | completed_jobs | docs_completed | docs_failed | result_fetch_failed | restarts/OOM | health | batch_status | job_status | SSE |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for phase in payload.get("phases") or []:
        round_payload = phase.get("round") or {}
        metrics = round_payload.get("metrics") or {}
        summary = metrics.get("ux_probe_summary") or {}
        delta = round_payload.get("cluster_delta") or {}
        restart_total = sum(int(v or 0) for v in (delta.get("restart_delta_by_component") or {}).values())
        oom_count = len(delta.get("oom_events_after") or [])

        def probe_cell(name: str) -> str:
            probe = summary.get(name) or {}
            ok = int(probe.get("ok") or 0)
            failed = int(probe.get("failed") or 0)
            p95 = probe.get("p95_ms")
            if ok or failed:
                return f"{ok}/{ok + failed} ok" + (f" p95={p95}ms" if p95 is not None else "")
            return "n/a"

        lines.append(
            "| {mode} | {n} | {hard} | {attr} | {jobs} | {docs_completed} | {docs_failed} | {fetch_failed} | {restart} | {health} | {batch} | {job} | {sse} |".format(
                mode=_mode_label(bool(phase.get("return_results"))),
                n=round_payload.get("n"),
                hard="yes" if round_payload.get("hard_failure") else "no",
                attr=metrics.get("failure_attribution", ""),
                jobs=metrics.get("completed_jobs", ""),
                docs_completed=metrics.get("documents_completed", ""),
                docs_failed=metrics.get("documents_failed", ""),
                fetch_failed=metrics.get("result_fetch_failed", ""),
                restart=f"{restart_total}/{oom_count}",
                health=probe_cell("health"),
                batch=probe_cell("batch_status"),
                job=probe_cell("job_aggregate"),
                sse=probe_cell("sse"),
            )
        )
    lines += ["", "## Interpretation", "", str(payload.get("interpretation") or "n/a")]
    lines += ["", "## Details", ""]
    for phase in payload.get("phases") or []:
        mode = _mode_label(bool(phase.get("return_results")))
        round_payload = phase.get("round") or {}
        metrics = round_payload.get("metrics") or {}
        summary = metrics.get("ux_probe_summary") or {}
        lines.append(f"### return_results={mode}, N={round_payload.get('n')}")
        lines.append(f"- hard failure reasons: {', '.join(round_payload.get('hard_failure_reasons') or []) or 'none'}")
        lines.append(f"- result fetch: attempts={metrics.get('result_fetch_attempts')} success={metrics.get('result_fetch_success')} failed={metrics.get('result_fetch_failed')} p95_s={metrics.get('result_fetch_latency_p95_s')}")
        lines.append(f"- known docs during probes: max={summary.get('max_known_documents')} completed={summary.get('max_known_completed_documents')} failed={summary.get('max_known_failed_documents')}")
        lines.append(f"- idle after run: {round_payload.get('idle_after_run')} ({round_payload.get('idle_error') or 'ok'})")
        evidence = phase.get("endpoint_evidence") or {}
        counts = evidence.get("pattern_counts") or {}
        if counts:
            lines.append(f"- endpoint evidence: ai.api.nvidia.com={counts.get('ai.api.nvidia.com', 0)}, returned 429={counts.get('returned 429', 0)}, returned 500={counts.get('returned 500', 0)}")
    lines += ["", "## Artifacts", "", f"- JSON: `{payload.get('artifact_paths', {}).get('json', '')}`", f"- Markdown: `{payload.get('artifact_paths', {}).get('markdown', '')}`"]
    for phase in payload.get("phases") or []:
        logs = phase.get("service_logs") or {}
        if logs.get("path"):
            lines.append(f"- return_results={_mode_label(bool(phase.get('return_results')))} logs: `{logs['path']}`")
    return "\n".join(lines) + "\n"


def _write_ux_probe_payload(session_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    out_path = session_dir / "results.json"
    md_path = session_dir / "report.md"
    payload["artifact_paths"] = {"json": str(out_path), "markdown": str(md_path)}
    write_json(out_path, payload)
    md_path.write_text(render_ux_probe_report(payload), encoding="utf-8")
    return payload


def run_bo20_ux_probe_qualification(
    *,
    dataset_dir: str = DEFAULT_BO20_DATASET,
    expected_pdfs: int = EXPECTED_BO20_PDFS,
    false_n: int = DEFAULT_UX_FALSE_N,
    true_n: int = DEFAULT_UX_TRUE_N,
    job_max_concurrency: int = DEFAULT_JOB_MAX_CONCURRENCY,
    run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ux_probe_interval_s: float = DEFAULT_UX_PROBE_INTERVAL_S,
    artifacts_dir: str | None = None,
    helm_release: str = "nemo-retriever-bo20-concurrency",
    helm_namespace: str | None = None,
    helm_chart: str | None = None,
    helm_chart_version: str | None = None,
    helm_set: dict[str, Any] | None = None,
    helm_timeout: int = 900,
    readiness_timeout: int = 900,
    helm_service_local_port: int = 7670,
    keep_up: bool = False,
    helm_bin: str = "helm",
    kubectl_bin: str = "kubectl",
    helm_sudo: bool = False,
    kubectl_sudo: bool = False,
    api_token: str | None = None,
    require_main: bool = True,
    dry_run: bool = False,
    return_results_modes: tuple[bool, ...] = (False, True),
    between_phase_wait_s: int = 180,
    nim_backend: str = "nvcf",
) -> dict[str, Any]:
    repo_root = NEMO_RETRIEVER_ROOT.parent
    inventory = inventory_bo20_dataset(dataset_dir, expected_pdfs=expected_pdfs)
    documents = [str(path) for path in resolve_bo20_files(dataset_dir, expected_pdfs=expected_pdfs)]
    nim_backend = _normalize_nim_backend(nim_backend)
    effective_return_results_modes = _effective_return_results_modes(nim_backend, return_results_modes)
    merged_helm_set = _helm_set_for_nim_backend(_clean_recovery_helm_set(helm_set), nim_backend)
    cfg = build_split_harness_config(
        dataset_dir=inventory.dataset_dir,
        artifacts_dir=artifacts_dir,
        helm_release=helm_release,
        helm_namespace=helm_namespace,
        helm_chart=helm_chart,
        helm_chart_version=helm_chart_version,
        helm_set=merged_helm_set,
        helm_timeout=helm_timeout,
        readiness_timeout=readiness_timeout,
        helm_service_local_port=helm_service_local_port,
        keep_up=keep_up,
        helm_bin=helm_bin,
        kubectl_bin=kubectl_bin,
        helm_sudo=helm_sudo,
        kubectl_sudo=kubectl_sudo,
        api_token=api_token,
    )
    session_dir = create_session_dir("bo20_ux_probe", base_dir=artifacts_dir)
    first_manager = HelmServiceManager(cfg, repo_root=repo_root)
    payload: dict[str, Any] = {
        "success": False,
        "ux_probe": True,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "inventory": asdict(inventory),
        "config": {
            "false_n": false_n,
            "true_n": true_n,
            "job_max_concurrency": job_max_concurrency,
            "run_timeout_s": run_timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "sample_interval_s": sample_interval_s,
            "ux_probe_interval_s": ux_probe_interval_s,
            "between_phase_wait_s": between_phase_wait_s,
            "require_main": require_main,
            "nim_backend": nim_backend,
            "return_results_modes": list(effective_return_results_modes),
            "helm_release": helm_release,
            "helm_namespace": helm_namespace or helm_release,
            "helm_chart": helm_chart or str(NEMO_RETRIEVER_ROOT / "helm"),
            "helm_set": cfg.helm_set,
        },
        "helm_command": first_manager.format_command(first_manager.build_upgrade_command()),
        "phases": [],
    }
    if dry_run:
        payload["success"] = True
        payload["dry_run"] = True
        return _write_ux_probe_payload(session_dir, payload)

    for index, return_results in enumerate(effective_return_results_modes):
        phase_dir = session_dir / ("return_results_true" if return_results else "return_results_false")
        phase_dir.mkdir(parents=True, exist_ok=True)
        manager = HelmServiceManager(cfg, repo_root=repo_root)
        phase_payload: dict[str, Any] = {
            "return_results": return_results,
            "n": _ux_target_n(return_results, false_n=false_n, true_n=true_n),
        }
        payload["phases"].append(phase_payload)
        phase_keep_up = bool(keep_up and index == len(effective_return_results_modes) - 1)
        try:
            if nim_backend == "nvcf":
                phase_payload["nvcf_secret"] = ensure_nvcf_secret_from_env(manager)
            start_rc = manager.start()
            if start_rc != 0:
                phase_payload["failure_reason"] = f"managed split Helm service failed to become ready (exit {start_rc})"
                phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                payload["failure_reason"] = phase_payload["failure_reason"]
                break
            service_url = manager.get_service_url()
            phase_payload["service_url"] = service_url
            preflight = run_split_preflight(
                manager=manager,
                service_url=service_url,
                repo_root=repo_root,
                require_main=require_main,
                inventory=inventory,
                require_service_monitor=_expect_service_monitor(cfg.helm_set),
                require_external_metrics=_expect_external_metrics(cfg.helm_set),
                require_hpa_active=False,
                nim_backend=nim_backend,
            )
            phase_payload["preflight"] = preflight
            if not preflight.get("success"):
                phase_payload["failure_reason"] = "preflight failed"
                phase_payload["endpoint_evidence"] = collect_endpoint_log_evidence(manager)
                phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
                payload["failure_reason"] = "preflight failed"
                break
            metric_urls = {"gateway": service_url, **(preflight.get("worker_metrics_urls") or {})}
            phase_payload["round"] = run_ux_probe_round(
                service_url=service_url,
                documents=documents,
                return_results=return_results,
                n=phase_payload["n"],
                job_max_concurrency=job_max_concurrency,
                api_token=api_token,
                run_timeout_s=run_timeout_s,
                total_pages_per_job=inventory.total_pages,
                manager=manager,
                sample_interval_s=sample_interval_s,
                ux_probe_interval_s=ux_probe_interval_s,
                idle_timeout_s=idle_timeout_s,
                metric_urls=metric_urls,
            )
            phase_payload["endpoint_evidence"] = collect_endpoint_log_evidence(manager)
            phase_payload["service_logs"] = _dump_phase_logs(manager, phase_dir)
        finally:
            try:
                stop_rc = manager.stop(uninstall=not phase_keep_up)
                phase_payload["cleanup_return_code"] = stop_rc
                if stop_rc == 0 and not phase_keep_up:
                    deleted = _wait_for_release_resources_deleted(
                        manager,
                        timeout_s=min(300, max(60, int(readiness_timeout))),
                    )
                    if not deleted:
                        phase_payload["cleanup_wait_warning"] = "timed out waiting for release pods/PVCs to disappear after uninstall"
            except Exception as exc:
                phase_payload["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if index < len(effective_return_results_modes) - 1 and between_phase_wait_s > 0:
            time.sleep(between_phase_wait_s)

    payload["success"] = not payload.get("failure_reason")
    payload["interpretation"] = _ux_interpretation(payload.get("phases") or [], nim_backend=nim_backend)
    return _write_ux_probe_payload(session_dir, payload)


def bo20_ux_probe_command(
    dataset_dir: str = typer.Option(DEFAULT_BO20_DATASET, "--dataset-dir", help="Canonical bo20 dataset directory."),
    expected_pdfs: int = typer.Option(EXPECTED_BO20_PDFS, "--expected-pdfs", help="Expected number of bo20 PDFs."),
    false_n: int = typer.Option(DEFAULT_UX_FALSE_N, "--false-n", min=1, help="Simultaneous bo20 jobs for return_results=False."),
    true_n: int = typer.Option(DEFAULT_UX_TRUE_N, "--true-n", min=1, help="Simultaneous bo20 jobs for return_results=True."),
    return_results_mode: str = typer.Option("both", "--return-results-mode", help="false, true, or both."),
    job_max_concurrency: int = typer.Option(DEFAULT_JOB_MAX_CONCURRENCY, "--job-max-concurrency", min=1, help="Per-job ServiceIngestor upload concurrency."),
    run_timeout_s: int = typer.Option(DEFAULT_RUN_TIMEOUT_S, "--run-timeout-s", min=1, help="Timeout per UX probe phase."),
    idle_timeout_s: int = typer.Option(DEFAULT_IDLE_TIMEOUT_S, "--idle-timeout-s", min=1, help="Timeout waiting for idle after each phase."),
    sample_interval_s: float = typer.Option(DEFAULT_SAMPLE_INTERVAL_S, "--sample-interval-s", min=1.0, help="Cluster sampler interval."),
    ux_probe_interval_s: float = typer.Option(DEFAULT_UX_PROBE_INTERVAL_S, "--ux-probe-interval-s", min=1.0, help="User-facing endpoint probe interval."),
    between_phase_wait_s: int = typer.Option(180, "--between-phase-wait-s", min=0, help="Wait between False/True phases to reduce hosted-rate-limit carryover."),
    artifacts_dir: str | None = typer.Option(None, "--artifacts-dir", help="Artifacts root directory."),
    helm_release: str = typer.Option("nemo-retriever-bo20-concurrency", "--helm-release", help="Helm release name."),
    helm_namespace: str | None = typer.Option(None, "--helm-namespace", help="Helm namespace. Defaults to release name."),
    helm_chart: str | None = typer.Option(None, "--helm-chart", help="Helm chart path."),
    helm_chart_version: str | None = typer.Option(None, "--helm-chart-version", help="Helm chart version."),
    helm_set: list[str] = typer.Option([], "--helm-set", help="Additional Helm --set KEY=VALUE overrides."),
    helm_timeout: int = typer.Option(900, "--helm-timeout", min=1, help="Helm install/upgrade timeout."),
    readiness_timeout: int = typer.Option(900, "--readiness-timeout", min=1, help="Managed service readiness timeout."),
    helm_service_local_port: int = typer.Option(7670, "--helm-service-local-port", min=1, help="Local gateway port-forward port."),
    keep_up: bool = typer.Option(False, "--keep-up", help="Leave the last Helm release running."),
    helm_bin: str = typer.Option("helm", "--helm-bin", help="Helm executable."),
    kubectl_bin: str = typer.Option("kubectl", "--kubectl-bin", help="kubectl executable."),
    helm_sudo: bool = typer.Option(False, "--helm-sudo", help="Run Helm through sudo."),
    kubectl_sudo: bool = typer.Option(False, "--kubectl-sudo", help="Run kubectl through sudo."),
    api_token: str | None = typer.Option(None, "--api-token", help="Bearer token for the Retriever service."),
    require_main: bool = typer.Option(True, "--require-main/--no-require-main", help="Require git branch main."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render config without deploying."),
    nim_backend: str = typer.Option("nvcf", "--nim-backend", help="NIM backend lane: local, nvcf, or proxy-dry-run."),
) -> None:
    """Run live probes describing client UX at known bo20 failure scales."""
    try:
        modes = _return_results_modes_from_option(return_results_mode)
        parsed_helm_set = _parse_helm_set(helm_set)
        result = run_bo20_ux_probe_qualification(
            dataset_dir=dataset_dir,
            expected_pdfs=expected_pdfs,
            false_n=false_n,
            true_n=true_n,
            job_max_concurrency=job_max_concurrency,
            run_timeout_s=run_timeout_s,
            idle_timeout_s=idle_timeout_s,
            sample_interval_s=sample_interval_s,
            ux_probe_interval_s=ux_probe_interval_s,
            artifacts_dir=artifacts_dir,
            helm_release=helm_release,
            helm_namespace=helm_namespace,
            helm_chart=helm_chart,
            helm_chart_version=helm_chart_version,
            helm_set=parsed_helm_set,
            helm_timeout=helm_timeout,
            readiness_timeout=readiness_timeout,
            helm_service_local_port=helm_service_local_port,
            keep_up=keep_up,
            helm_bin=helm_bin,
            kubectl_bin=kubectl_bin,
            helm_sudo=helm_sudo,
            kubectl_sudo=kubectl_sudo,
            api_token=api_token,
            require_main=require_main,
            dry_run=dry_run,
            return_results_modes=modes,
            between_phase_wait_s=between_phase_wait_s,
            nim_backend=nim_backend,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    report_path = (result.get("artifact_paths") or {}).get("markdown") or (result.get("artifact_paths") or {}).get("json")
    typer.echo(f"bo20 UX probe complete: {report_path}")

def bo20_concurrency_command(
    dataset_dir: str = typer.Option(DEFAULT_BO20_DATASET, "--dataset-dir", help="Canonical bo20 dataset directory."),
    expected_pdfs: int = typer.Option(EXPECTED_BO20_PDFS, "--expected-pdfs", help="Expected number of bo20 PDFs."),
    max_n: int | None = typer.Option(
        None,
        "--max-n",
        min=1,
        help="Highest simultaneous bo20 job count to test. Defaults to 16, or 3 in clean recovery mode.",
    ),
    job_max_concurrency: int = typer.Option(
        DEFAULT_JOB_MAX_CONCURRENCY,
        "--job-max-concurrency",
        min=1,
        help="Per-job ServiceIngestor upload concurrency.",
    ),
    run_timeout_s: int = typer.Option(DEFAULT_RUN_TIMEOUT_S, "--run-timeout-s", min=1, help="Timeout per N run."),
    idle_timeout_s: int = typer.Option(
        DEFAULT_IDLE_TIMEOUT_S,
        "--idle-timeout-s",
        min=1,
        help="Timeout waiting for queues/jobs to return to idle between runs.",
    ),
    sample_interval_s: float = typer.Option(
        DEFAULT_SAMPLE_INTERVAL_S,
        "--sample-interval-s",
        min=1.0,
        help="Seconds between queue/HPA samples during each run.",
    ),
    artifacts_dir: str | None = typer.Option(None, "--artifacts-dir", help="Artifacts root override."),
    helm_release: str = typer.Option(
        "nemo-retriever-bo20-concurrency",
        "--helm-release",
        help="Managed Helm release name.",
    ),
    helm_namespace: str | None = typer.Option(None, "--helm-namespace", help="Managed Helm namespace."),
    helm_chart: str | None = typer.Option(None, "--helm-chart", help="Helm chart path or remote ref."),
    helm_chart_version: str | None = typer.Option(None, "--helm-chart-version", help="Remote Helm chart version."),
    helm_set: list[str] = typer.Option([], "--helm-set", help="Additional Helm KEY=VALUE override. Repeatable."),
    helm_timeout: int = typer.Option(900, "--helm-timeout", min=1, help="Helm upgrade/install timeout."),
    readiness_timeout: int = typer.Option(900, "--readiness-timeout", min=1, help="Service readiness timeout."),
    helm_service_local_port: int = typer.Option(7670, "--helm-service-local-port", min=1, help="Local gateway port."),
    keep_up: bool = typer.Option(False, "--keep-up/--no-keep-up", help="Keep Helm release running after the sweep."),
    helm_bin: str = typer.Option("helm", "--helm-bin", help="Helm binary command."),
    kubectl_bin: str = typer.Option("kubectl", "--kubectl-bin", help="kubectl binary command."),
    helm_sudo: bool = typer.Option(False, "--helm-sudo/--no-helm-sudo", help="Run Helm through sudo."),
    kubectl_sudo: bool = typer.Option(False, "--kubectl-sudo/--no-kubectl-sudo", help="Run kubectl through sudo."),
    api_token: str | None = typer.Option(None, "--api-token", help="Optional service bearer token."),
    require_main: bool = typer.Option(True, "--require-main/--allow-non-main", help="Fail preflight unless on main."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs and print Helm command without deploying."),
    clean_page_elements_rerun: bool = typer.Option(
        False,
        "--clean-page-elements-rerun",
        help="Run clean Phase A/Phase B reruns with a redeploy between return_results modes.",
    ),
    return_results_mode: str = typer.Option(
        "both",
        "--return-results-mode",
        help="Which return_results modes to run: false, true, or both.",
    ),
    nim_backend: str = typer.Option(
        "local",
        "--nim-backend",
        help="NIM backend lane: local, nvcf, or proxy-dry-run.",
    ),
) -> None:
    """Run split-Helm bo20 concurrent service-ingest qualification."""
    effective_max_n = _resolve_cli_max_n(max_n, clean_page_elements_rerun=clean_page_elements_rerun)
    try:
        return_results_modes = _return_results_modes_from_option(return_results_mode)
        result = run_bo20_concurrency_qualification(
            dataset_dir=dataset_dir,
            expected_pdfs=expected_pdfs,
            max_n=effective_max_n,
            job_max_concurrency=job_max_concurrency,
            run_timeout_s=run_timeout_s,
            idle_timeout_s=idle_timeout_s,
            sample_interval_s=sample_interval_s,
            artifacts_dir=artifacts_dir,
            helm_release=helm_release,
            helm_namespace=helm_namespace,
            helm_chart=helm_chart,
            helm_chart_version=helm_chart_version,
            helm_set=_parse_helm_set(helm_set),
            helm_timeout=helm_timeout,
            readiness_timeout=readiness_timeout,
            helm_service_local_port=helm_service_local_port,
            keep_up=keep_up,
            helm_bin=helm_bin,
            kubectl_bin=kubectl_bin,
            helm_sudo=helm_sudo,
            kubectl_sudo=kubectl_sudo,
            api_token=api_token,
            require_main=require_main,
            dry_run=dry_run,
            clean_page_elements_rerun=clean_page_elements_rerun,
            return_results_modes=return_results_modes,
            nim_backend=nim_backend,
        )
    except Exception as exc:
        typer.echo(f"bo20 split concurrency qualification failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    artifacts = result.get("artifact_paths") or {}
    if result.get("success"):
        report_path = artifacts.get("markdown") or artifacts.get("json")
        typer.echo(f"bo20 split concurrency qualification complete: {report_path}")
        raise typer.Exit(0)
    typer.echo(f"bo20 split concurrency qualification failed: {result.get('failure_reason')}")
    if (preflight := result.get("preflight")) and preflight.get("errors"):
        for error in preflight["errors"]:
            typer.echo(f"  - {error}")
    typer.echo(f"Artifacts: {artifacts.get('json', 'n/a')}")
    raise typer.Exit(1)
