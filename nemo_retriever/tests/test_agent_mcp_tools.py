# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from nemo_retriever.agent_mcp.models import (
    AgentMcpError,
    AgentMcpErrorCode,
    CollectionRecord,
    EvidenceHit,
    IngestJobRecord,
    JobStatus,
)
from nemo_retriever.agent_mcp.tools import AgentMcpToolService


class FakeBackend:
    def __init__(self) -> None:
        self.rerank_seen_hits: list[EvidenceHit] = []

    def list_collections(self) -> list[CollectionRecord]:
        return [
            CollectionRecord(name="docs", root_path="/tmp/docs", queryable=True),
            CollectionRecord(name="scratch", root_path="/tmp/scratch"),
        ]

    def create_collection(
        self,
        name: str = "default",
        temporary: bool = False,
        hybrid: bool = False,
    ) -> CollectionRecord:
        return CollectionRecord(
            name=name,
            root_path=f"/tmp/{name}",
            temporary=temporary,
            hybrid=hybrid,
        )

    def describe_collection(self, name: str = "default") -> dict[str, Any]:
        return {
            "name": name,
            "queryable": True,
            "recent_jobs": [IngestJobRecord(job_id="job-1", collection=name, status=JobStatus.COMPLETE)],
        }

    def delete_collection(self, name: str, delete_data: bool = False) -> CollectionRecord:
        return CollectionRecord(
            name=name,
            root_path=f"/tmp/{name}",
            metadata={"delete_data": delete_data},
        )

    def start_ingestion(
        self,
        collection: str,
        paths: list[str],
        *,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        return IngestJobRecord(
            job_id="job-1",
            collection=collection,
            status=JobStatus.RUNNING,
            source_count=len(paths),
        )

    def ingest_local_paths(
        self,
        collection: str,
        paths: list[str],
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        return IngestJobRecord(
            job_id="job-1",
            collection=collection,
            status=JobStatus.COMPLETE,
            source_count=len(paths),
            accepted_count=len(paths),
        )

    def get_ingestion_status(self, job_id: str) -> IngestJobRecord:
        return IngestJobRecord(job_id=job_id, collection="docs", status=JobStatus.COMPLETE)

    def query_collection(
        self,
        collection: str = "default",
        query: str = "",
        top_k: int = 10,
        hybrid: bool = False,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[EvidenceHit]:
        return [
            EvidenceHit(
                text=f"{query} answer {index}",
                score=1.0 / index,
                source_path=f"/tmp/{collection}-{index}.txt",
                media_type="text",
                content_type="text",
                metadata={"hybrid": hybrid, "rerank": rerank, "filters": filters},
            )
            for index in range(1, top_k + 1)
        ]

    def rerank_results(
        self,
        query: str,
        hits: list[EvidenceHit],
        top_n: int | None = None,
    ) -> list[EvidenceHit]:
        self.rerank_seen_hits = hits
        return hits[:top_n]


class FailingBackend(FakeBackend):
    def list_collections(self) -> list[CollectionRecord]:
        raise AgentMcpError(
            AgentMcpErrorCode.BACKEND_ERROR,
            "backend unavailable",
            retryable=True,
            details={"backend": "fake"},
        )


def assert_json_serializable(value: Any) -> None:
    json.dumps(value)


def test_tool_service_returns_json_serializable_collection_payloads() -> None:
    service = AgentMcpToolService(FakeBackend())

    payload = service.list_collections()

    assert payload[0]["name"] == "docs"
    assert payload[0]["queryable"] is True
    assert payload[1]["name"] == "scratch"
    assert payload[1]["queryable"] is False
    assert_json_serializable(payload)


def test_tool_service_ingest_and_query() -> None:
    backend = FakeBackend()
    service = AgentMcpToolService(backend)

    job = service.ingest_local_paths(collection="docs", paths=["/tmp/a.pdf"])
    hits = service.query_collection(collection="docs", query="hello", top_k=2, hybrid=True, rerank=True)
    reranked = service.rerank_results("hello", hits, top_n=1)

    assert job["status"] == "complete"
    assert hits[0]["text"] == "hello answer 1"
    assert hits[0]["metadata"] == {"hybrid": True, "rerank": True, "filters": None}
    assert reranked == [hits[0]]
    assert isinstance(backend.rerank_seen_hits[0], EvidenceHit)
    assert backend.rerank_seen_hits[0].text == hits[0]["text"]
    assert_json_serializable(job)
    assert_json_serializable(hits)
    assert_json_serializable(reranked)


def test_tool_service_serializes_agent_mcp_errors() -> None:
    service = AgentMcpToolService(FailingBackend())

    payload = service.list_collections()

    assert payload == {
        "error": {
            "code": "BACKEND_ERROR",
            "message": "backend unavailable",
            "retryable": True,
            "details": {"backend": "fake"},
        }
    }
    assert_json_serializable(payload)


def test_tool_service_serializes_invalid_rerank_hit_payloads() -> None:
    service = AgentMcpToolService(FakeBackend())

    payload = service.rerank_results("hello", [{"score": 0.2}], top_n=1)

    assert payload["error"]["code"] == "BACKEND_ERROR"
    assert payload["error"]["message"] == "Invalid rerank hit payload."
    assert payload["error"]["retryable"] is False
    assert "errors" in payload["error"]["details"]
    assert_json_serializable(payload)
