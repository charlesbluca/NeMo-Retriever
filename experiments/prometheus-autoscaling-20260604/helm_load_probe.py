#!/usr/bin/env python3
"""Drive Helm-release ingest load while sampling Prometheus autoscaling signals."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


QUEUE_RATIO_AVG = (
    'avg by (pool) (nemo_retriever_pool_queue_depth{pool="realtime"} '
    '/ on(pool, instance) group_left() '
    'nemo_retriever_pool_max_queue_size{pool="realtime"})'
)
QUEUE_RATIO_MAX = (
    'max by (pool) (nemo_retriever_pool_queue_depth{pool="realtime"} '
    '/ on(pool, instance) group_left() '
    'nemo_retriever_pool_max_queue_size{pool="realtime"})'
)
QUEUE_DEPTH_SUM = 'sum(nemo_retriever_pool_queue_depth{pool="realtime"})'
QUEUE_DEPTH_MAX = 'max(nemo_retriever_pool_queue_depth{pool="realtime"})'
P95 = (
    'histogram_quantile(0.95, sum by (le, pool) '
    '(rate(nemo_retriever_pool_processing_duration_seconds_bucket{pool="realtime"}[2m])))'
)
COMPLETED = 'sum(nemo_retriever_pool_processed_total{pool="realtime",outcome="completed"})'
FAILED = 'sum(nemo_retriever_pool_processed_total{pool="realtime",outcome="failed"})'


def _query(prom_url: str, query: str) -> float | None:
    try:
        resp = requests.get(
            f"{prom_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()["data"]["result"]
        if not data:
            return None
        return float(data[0]["value"][1])
    except Exception:
        return None


def _sample_loop(prom_url: str, out_path: Path, stop: threading.Event, interval: float) -> None:
    start = time.time()
    fields = [
        "ts",
        "elapsed_s",
        "queue_ratio_avg",
        "queue_ratio_max",
        "queue_depth_sum",
        "queue_depth_max",
        "processing_p95_s",
        "processed_completed",
        "processed_failed",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while True:
            now = time.time()
            writer.writerow(
                {
                    "ts": f"{now:.3f}",
                    "elapsed_s": f"{now - start:.3f}",
                    "queue_ratio_avg": _query(prom_url, QUEUE_RATIO_AVG),
                    "queue_ratio_max": _query(prom_url, QUEUE_RATIO_MAX),
                    "queue_depth_sum": _query(prom_url, QUEUE_DEPTH_SUM),
                    "queue_depth_max": _query(prom_url, QUEUE_DEPTH_MAX),
                    "processing_p95_s": _query(prom_url, P95),
                    "processed_completed": _query(prom_url, COMPLETED),
                    "processed_failed": _query(prom_url, FAILED),
                }
            )
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


def _post_page(
    *,
    base_url: str,
    payload: bytes,
    filename: str,
    job_id: str,
    idx: int,
    mode: str,
    callback_url: str,
) -> dict[str, Any]:
    page_id = f"{job_id}-page-{idx:04d}"
    headers = {}
    if mode == "worker":
        headers = {
            "X-Gateway-Document-Id": page_id,
            "X-Gateway-Job-Id": job_id,
            "X-Gateway-Callback-Url": callback_url,
        }
    started = time.time()
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/ingest/job/{job_id}/page",
            headers=headers,
            data={
                "document_id": job_id,
                "page_number": str(idx + 1),
                "filename": filename,
            },
            files={"file": (filename, payload, "application/pdf")},
            timeout=45,
        )
        body = resp.text[:500]
        return {
            "idx": idx,
            "status_code": resp.status_code,
            "ok": 200 <= resp.status_code < 300,
            "elapsed_s": round(time.time() - started, 3),
            "body": body,
        }
    except Exception as exc:
        return {
            "idx": idx,
            "status_code": None,
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "body": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gateway", "worker"], default="gateway")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prom-url", default="http://127.0.0.1:9090")
    parser.add_argument("--input", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--post-sample-seconds", type=float, default=90.0)
    parser.add_argument("--label", default="helm-hpa-probe")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--callback-url", default="http://127.0.0.1:1/callback")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = Path(args.input).read_bytes()
    filename = Path(args.input).name

    job_id = args.job_id or args.label
    if args.mode == "gateway":
        job_id = _create_job(args.base_url, args.count, args.label)

    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_loop,
        args=(args.prom_url, run_dir / "samples.csv", stop, args.sample_interval),
        daemon=True,
    )
    sampler.start()

    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _post_page,
                base_url=args.base_url,
                payload=payload,
                filename=filename,
                job_id=job_id,
                idx=i,
                mode=args.mode,
                callback_url=args.callback_url,
            )
            for i in range(args.count)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    time.sleep(args.post_sample_seconds)
    stop.set()
    sampler.join(timeout=10)

    summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "job_id": job_id,
        "count": args.count,
        "concurrency": args.concurrency,
        "duration_s": round(time.time() - started, 3),
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
