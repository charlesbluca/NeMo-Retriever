# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nemo_retriever.agent_mcp.backend import InProcessRetrieverBackend
from nemo_retriever.agent_mcp.models import AgentMcpError, AgentMcpErrorCode, EvidenceArtifacts, EvidenceHit, Locator
from nemo_retriever.agent_mcp.paths import PathPolicy
from nemo_retriever.agent_mcp.registry import CollectionRegistry


class FakeRetriever:
    constructor_kwargs: list[dict[str, Any]] = []
    query_calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_kwargs.append(kwargs)

    def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append((query, kwargs))
        return [
            {
                "text": "A normalized answer.",
                "_distance": "0.42",
                "source": json.dumps({"source_id": "/tmp/source.pdf"}),
                "metadata": json.dumps({"page_number": "3", "content_type": "text"}),
            }
        ]


class HybridUnsupportedRetriever(FakeRetriever):
    def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("hybrid is not supported")


def _backend(tmp_path: Path, **kwargs: Any) -> InProcessRetrieverBackend:
    FakeRetriever.constructor_kwargs = []
    FakeRetriever.query_calls = []
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    retriever_factory = kwargs.pop("retriever_factory", FakeRetriever)
    return InProcessRetrieverBackend(
        registry,
        path_policy=PathPolicy(allowed_roots=[tmp_path]),
        retriever_factory=retriever_factory,
        **kwargs,
    )


def test_query_collection_uses_collection_embedding_and_vdb_config(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    record = backend.create_collection("docs")
    record = backend.registry.mark_collection_queryable(record.name, row_count=1)

    hits = backend.query_collection(collection="docs", query="what is here?", top_k=3)

    assert hits[0].text == "A normalized answer."
    assert hits[0].score == 0.42
    assert hits[0].locator.page_number == 3

    kwargs = FakeRetriever.constructor_kwargs[0]
    assert kwargs["vdb"] == record.vdb_backend
    assert kwargs["embedder"] == record.embedding_model
    assert kwargs["vdb_kwargs"]["uri"] == record.vdb_uri
    assert kwargs["vdb_kwargs"]["table_name"] == record.vdb_table
    assert FakeRetriever.query_calls == [("what is here?", {"top_k": 3, "vdb_kwargs": None})]


def test_query_rejects_unqueryable_collection(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="not yet")

    assert exc.value.code is AgentMcpErrorCode.COLLECTION_NOT_QUERYABLE


def test_query_rejects_hybrid_when_collection_is_not_hybrid(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs", hybrid=False)
    backend.registry.mark_collection_queryable("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert exc.value.code is AgentMcpErrorCode.HYBRID_NOT_AVAILABLE


def test_query_sets_reranker_flag_when_requested(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="rerank it", rerank=True)

    assert FakeRetriever.constructor_kwargs[0]["reranker"] is True


def test_query_forwards_hybrid_flag_for_hybrid_collection(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs", hybrid=True)
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert FakeRetriever.query_calls == [("hybrid please", {"top_k": 10, "vdb_kwargs": {"hybrid": True}})]


def test_query_translates_hybrid_not_implemented(tmp_path: Path) -> None:
    backend = _backend(tmp_path, retriever_factory=HybridUnsupportedRetriever)
    backend.create_collection("docs", hybrid=True)
    backend.registry.mark_collection_queryable("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert exc.value.code is AgentMcpErrorCode.HYBRID_NOT_AVAILABLE


def test_query_forwards_filters(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="filtered", filters={"tenant": "a"})

    assert FakeRetriever.query_calls == [("filtered", {"top_k": 10, "vdb_kwargs": {"tenant": "a"}})]


def test_standalone_rerank_results_uses_injected_reranker(tmp_path: Path) -> None:
    rerank_calls = []

    def fake_rerank(query: str, hits: list[dict[str, Any]], *, top_n: int | None = None) -> list[dict[str, Any]]:
        rerank_calls.append((query, hits, top_n))
        return [{**hits[0], "_rerank_score": 0.99}]

    backend = _backend(tmp_path, rerank_fn=fake_rerank)
    hit = EvidenceHit(text="candidate", score=0.1, source_path="/tmp/source.pdf")

    reranked = backend.rerank_results("best one?", [hit], top_n=1)

    assert rerank_calls[0][0] == "best one?"
    assert rerank_calls[0][1][0]["text"] == "candidate"
    assert rerank_calls[0][1][0]["source_path"] == "/tmp/source.pdf"
    assert rerank_calls[0][1][0]["path"] == "/tmp/source.pdf"
    assert rerank_calls[0][2] == 1
    assert reranked[0].text == "candidate"
    assert reranked[0].score == 0.99


def test_default_standalone_rerank_uses_supplied_reranker_model(tmp_path: Path) -> None:
    class FakeRerankerModel:
        model_name = "nvidia/llama-nemotron-rerank-1b-v2"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def score(
            self,
            query: str,
            documents: list[str],
            *,
            max_length: int,
            batch_size: int,
        ) -> list[float]:
            self.calls.append(
                {
                    "query": query,
                    "documents": documents,
                    "max_length": max_length,
                    "batch_size": batch_size,
                }
            )
            return [0.77]

    model = FakeRerankerModel()
    backend = _backend(tmp_path, reranker_model=model, reranker_max_length=128, reranker_batch_size=2)

    reranked = backend.rerank_results("best one?", [EvidenceHit(text="candidate")], top_n=1)

    assert model.calls == [
        {
            "query": "best one?",
            "documents": ["candidate"],
            "max_length": 128,
            "batch_size": 2,
        }
    ]
    assert reranked[0].score == 0.77


def test_standalone_rerank_preserves_evidence_fields(tmp_path: Path) -> None:
    def fake_rerank(query: str, hits: list[dict[str, Any]], *, top_n: int | None = None) -> list[dict[str, Any]]:
        return [{**hits[0], "_rerank_score": 0.99}]

    backend = _backend(tmp_path, rerank_fn=fake_rerank)
    hit = EvidenceHit(
        text="candidate",
        score=0.1,
        source_path="/tmp/source.pdf",
        content_type="text",
        locator=Locator(
            page_number=7,
            timestamp_start_s=1.25,
            timestamp_end_s=2.5,
            frame_index=3,
            bbox_xyxy_norm=[0.1, 0.2, 0.8, 0.9],
        ),
        artifacts=EvidenceArtifacts(
            stored_image_uri="file:///tmp/page.png",
            thumbnail_uri="file:///tmp/thumb.png",
        ),
        metadata={"section": "intro"},
    )

    reranked = backend.rerank_results("best one?", [hit], top_n=1)

    assert reranked[0].score == 0.99
    assert reranked[0].source_path == "/tmp/source.pdf"
    assert reranked[0].content_type == "text"
    assert reranked[0].locator.page_number == 7
    assert reranked[0].locator.timestamp_start_s == 1.25
    assert reranked[0].locator.timestamp_end_s == 2.5
    assert reranked[0].locator.frame_index == 3
    assert reranked[0].locator.bbox_xyxy_norm == [0.1, 0.2, 0.8, 0.9]
    assert reranked[0].artifacts.stored_image_uri == "file:///tmp/page.png"
    assert reranked[0].artifacts.thumbnail_uri == "file:///tmp/thumb.png"
    assert reranked[0].metadata["section"] == "intro"
