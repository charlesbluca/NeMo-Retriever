# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Split-topology worker callback must not POST full result_data payloads."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service.services.pipeline_pool import _fire_gateway_callback
from nemo_retriever.service.services.worker_result_store import (
    clear_for_tests,
    consume_result_data,
    store_result_data,
)


@pytest.fixture(autouse=True)
def _clear_worker_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_RETRIEVER_RESULTS_DIR", raising=False)
    clear_for_tests()
    yield
    clear_for_tests()


def test_fire_gateway_callback_omits_result_data() -> None:
    posted: dict[str, Any] = {}

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            posted["url"] = url
            posted["json"] = json
            return _Resp()

    rows = [{"page": 1, "text": "x" * 10_000}]

    async def _run() -> None:
        with patch("httpx.AsyncClient", _Client):
            store_result_data("doc-1", rows)
            await _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "doc-1",
                "completed",
                result_rows=42,
            )

    asyncio.run(_run())

    assert posted["json"] == {"id": "doc-1", "status": "completed", "result_rows": 42}
    assert "result_data" not in posted["json"]
    assert consume_result_data("doc-1") == rows


def test_worker_document_result_endpoint() -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import PipelineOverridesConfig, PipelinePoolConfig, ServiceConfig

    cfg = ServiceConfig(
        mode="batch",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    store_result_data("doc-x", [{"text": "hello"}])
    with TestClient(create_app(cfg)) as client:
        resp = client.get("/v1/internal/document-result/doc-x")
        assert resp.status_code == 200
        assert resp.json()["result_data"] == [{"text": "hello"}]
        assert client.get("/v1/internal/document-result/doc-x").status_code == 404


def test_shared_result_store_is_visible_across_memory_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"page": 1, "text": "shared"}]

    store_result_data("../unsafe/document-id", rows)
    clear_for_tests()  # Simulate reading from another pod/process.

    assert consume_result_data("../unsafe/document-id") == rows
    assert consume_result_data("../unsafe/document-id") is None
    assert not list(tmp_path.iterdir())


def test_shared_result_store_has_single_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "consume once"}]
    store_result_data("doc-concurrent", rows)

    with ThreadPoolExecutor(max_workers=8) as executor:
        consumed = list(executor.map(lambda _: consume_result_data("doc-concurrent"), range(8)))

    assert consumed.count(rows) == 1
    assert consumed.count(None) == 7


def test_gateway_fetches_shared_result_before_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemo_retriever.service.routers.ingest import _fetch_result_data_from_workers

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "gateway"}]
    store_result_data("doc-gateway", rows)

    assert asyncio.run(_fetch_result_data_from_workers("doc-gateway")) == rows


def test_gateway_status_routes_consume_shared_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "shared route"}]

    with TestClient(create_app(ServiceConfig(mode="gateway"))) as client:
        tracker = get_job_tracker()
        assert tracker is not None
        tracker.register_job("job-shared", expected_documents=2, retain_results=True)

        tracker.register_document("doc-job-route", job_id="job-shared")
        tracker.mark_completed("doc-job-route", result_rows=1)
        store_result_data("doc-job-route", rows)
        response = client.get("/v1/ingest/job/job-shared/document/doc-job-route")
        assert response.status_code == 200
        assert response.json()["result_data"] == rows

        tracker.register_document("doc-status-route", job_id="job-shared")
        tracker.mark_completed("doc-status-route", result_rows=1)
        store_result_data("doc-status-route", rows)
        response = client.get("/v1/ingest/status/doc-status-route")
        assert response.status_code == 200
        assert response.json()["result_data"] == rows
