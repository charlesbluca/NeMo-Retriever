# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from nemo_retriever.service.routers import admin
from nemo_retriever.service.services import pipeline_pool as pp
from nemo_retriever.service.services.pipeline_pool import PoolType, WorkItem, _Pool
from nemo_retriever.service.services.proxy import _backend_attempt_reason, _record_backend_attempt


def _sample_value(name: str, labels: dict[str, str]) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_pool_autoscaling_metrics_track_active_inflight_wait_and_rejections(monkeypatch):
    monkeypatch.setattr(pp, "_QUEUE_DEPTH_REPORT_INTERVAL_S", 0.02)

    async def body():
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_work(_item):
            started.set()
            await release.wait()
            return 0

        pool = _Pool(name="autoscale-obs", num_workers=1, max_queue_size=1, work_fn=slow_work)
        pool.start()
        try:
            assert await pool.submit(WorkItem(id="first")) is True
            await asyncio.wait_for(started.wait(), timeout=2.0)
            assert await pool.submit(WorkItem(id="queued")) is True
            assert await pool.submit(WorkItem(id="rejected")) is False
            await asyncio.sleep(0.1)

            assert _sample_value("nemo_retriever_pool_active_workers", {"pool": "autoscale-obs"}) == 1.0
            assert _sample_value("nemo_retriever_pool_inflight_work_items", {"pool": "autoscale-obs"}) == 2.0
            assert _sample_value(
                "nemo_retriever_pool_queue_wait_duration_seconds_count",
                {"pool": "autoscale-obs"},
            ) == 1.0
            assert _sample_value(
                "nemo_retriever_pool_enqueue_rejected_total",
                {"pool": "autoscale-obs", "reason": "full"},
            ) == 1.0
        finally:
            release.set()
            await pool.shutdown()

        assert _sample_value("nemo_retriever_pool_active_workers", {"pool": "autoscale-obs"}) == 0.0
        assert _sample_value("nemo_retriever_pool_inflight_work_items", {"pool": "autoscale-obs"}) == 0.0

    _run(body())


def test_admin_pool_stats_include_active_and_inflight(monkeypatch):
    class FakePool:
        queue_depth = 3
        max_queue_size = 4
        active_workers = 2
        inflight_work_items = 5
        num_workers = 8
        processed = 13
        is_running = True

    class FakePipelinePool:
        def pool_for(self, pool_type):
            if pool_type is PoolType.REALTIME:
                return FakePool()
            return None

    monkeypatch.setattr(pp, "get_pipeline_pool", lambda: FakePipelinePool())

    app = FastAPI()
    app.state.config = SimpleNamespace(mode="standalone")
    app.include_router(admin.router, prefix="/v1")

    with TestClient(app) as client:
        resp = client.get("/v1/admin/pool_stats")

    assert resp.status_code == 200, resp.text
    stats = resp.json()["pools"]["realtime"]
    assert stats["queue_depth"] == 3
    assert stats["queue_depth_ratio"] == 0.75
    assert stats["active_workers"] == 2
    assert stats["inflight_work_items"] == 5


def test_gateway_backend_attempt_metric_records_429_reason():
    before = _sample_value(
        "nemo_retriever_gateway_backend_attempts_total",
        {"pool": "realtime", "status": "429", "reason": "backend_429"},
    ) or 0.0

    assert _backend_attempt_reason(429) == "backend_429"
    _record_backend_attempt(PoolType.REALTIME, 429, "backend_429")

    after = _sample_value(
        "nemo_retriever_gateway_backend_attempts_total",
        {"pool": "realtime", "status": "429", "reason": "backend_429"},
    )
    assert after == before + 1.0
