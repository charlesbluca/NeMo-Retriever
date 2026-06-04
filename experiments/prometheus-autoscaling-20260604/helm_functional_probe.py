#!/usr/bin/env python3
"""Probe functional autoscaling failure modes against a live Helm release."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

METRIC_NAMES = {
    "queue_depth": "nemo_retriever_pool_queue_depth",
    "queue_ratio": "nemo_retriever_pool_queue_depth_ratio",
    "max_queue_size": "nemo_retriever_pool_max_queue_size",
    "workers": "nemo_retriever_pool_workers",
    "processed_total": "nemo_retriever_pool_processed_total",
    "processing_count": "nemo_retriever_pool_processing_duration_seconds_count",
    "processing_sum": "nemo_retriever_pool_processing_duration_seconds_sum",
}


def _parse_metric(text: str, metric: str, *, pool: str = "realtime", outcome: str | None = None) -> float | None:
    for line in text.splitlines():
        if not line.startswith(metric):
            continue
        if f'pool="{pool}"' not in line:
            continue
        if outcome is not None and f'outcome="{outcome}"' not in line:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _scrape(url: str) -> dict[str, float | None]:
    try:
        resp = requests.get(url.rstrip("/") + "/metrics", timeout=5)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return {
            "queue_depth": None,
            "queue_ratio": None,
            "max_queue_size": None,
            "workers": None,
            "processed_completed": None,
            "processed_failed": None,
            "processing_count": None,
            "processing_sum": None,
        }
    return {
        "queue_depth": _parse_metric(text, METRIC_NAMES["queue_depth"]),
        "queue_ratio": _parse_metric(text, METRIC_NAMES["queue_ratio"]),
        "max_queue_size": _parse_metric(text, METRIC_NAMES["max_queue_size"]),
        "workers": _parse_metric(text, METRIC_NAMES["workers"]),
        "processed_completed": _parse_metric(text, METRIC_NAMES["processed_total"], outcome="completed"),
        "processed_failed": _parse_metric(text, METRIC_NAMES["processed_total"], outcome="failed"),
        "processing_count": _parse_metric(text, METRIC_NAMES["processing_count"]),
        "processing_sum": _parse_metric(text, METRIC_NAMES["processing_sum"]),
    }


def _sample_loop(metrics: dict[str, str], out_path: Path, stop: threading.Event, interval: float) -> None:
    start = time.time()
    fields = [
        "ts",
        "elapsed_s",
        "target",
        "queue_depth",
        "queue_ratio",
        "max_queue_size",
        "workers",
        "processed_completed",
        "processed_failed",
        "processing_count",
        "processing_sum",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while True:
            now = time.time()
            for label, url in metrics.items():
                row: dict[str, Any] = {"ts": f"{now:.3f}", "elapsed_s": f"{now - start:.3f}", "target": label}
                row.update(_scrape(url))
                writer.writerow(row)
            f.flush()
            if stop.wait(interval):
                return


def _create_job(base_url: str, count: int, label: str) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/ingest/job",
        json={"expected_documents": count, "label": label, "retain_results": False},
        timeout=15,
    )
    resp.raise_for_status()
    return str(resp.json()["job_id"])


def _post_page(base_url: str, payload: bytes, filename: str, job_id: str, idx: int) -> dict[str, Any]:
    started = time.time()
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/ingest/job/{job_id}/page",
            data={"document_id": job_id, "page_number": str(idx + 1), "filename": filename},
            files={"file": (filename, payload, "application/pdf")},
            timeout=60,
        )
        return {
            "idx": idx,
            "status_code": resp.status_code,
            "ok": 200 <= resp.status_code < 300,
            "elapsed_s": round(time.time() - started, 3),
            "body": resp.text[:500],
        }
    except Exception as exc:
        return {
            "idx": idx,
            "status_code": None,
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "body": repr(exc),
        }


def _metric_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        parsed = urlparse(value)
        label = parsed.netloc.replace(":", "_") or f"metric_{abs(hash(value))}"
        return label, value
    label, url = value.split("=", 1)
    return label, url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metric", action="append", default=[], help="LABEL=URL for pod-local metrics endpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--post-sample-seconds", type=float, default=20.0)
    parser.add_argument("--label", default="functional-probe")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = Path(args.input).read_bytes()
    filename = Path(args.input).name
    metrics = dict(_metric_arg(item) for item in args.metric)

    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_loop,
        args=(metrics, run_dir / "samples.csv", stop, args.sample_interval),
        daemon=True,
    )
    sampler.start()

    started = time.time()
    job_id = _create_job(args.base_url, args.count, args.label)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_post_page, args.base_url, payload, filename, job_id, idx)
            for idx in range(args.count)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    time.sleep(args.post_sample_seconds)
    stop.set()
    sampler.join(timeout=10)

    summary: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "job_id": job_id,
        "count": args.count,
        "concurrency": args.concurrency,
        "duration_s": round(time.time() - started, 3),
        "metrics": metrics,
        "status_counts": {},
    }
    for result in results:
        key = str(result["status_code"])
        summary["status_counts"][key] = summary["status_counts"].get(key, 0) + 1

    (run_dir / "responses.json").write_text(json.dumps(results, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
