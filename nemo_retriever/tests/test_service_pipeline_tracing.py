# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing coverage for process-isolated pipeline execution."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service import tracing
from nemo_retriever.service.services import pipeline_executor


class _CollectingExporter:
    def __init__(self, exported: list[Any]) -> None:
        self._exported = exported

    def export(self, spans: Any) -> SpanExportResult:
        self._exported.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class _FakeIngestor:
    def ingest(self) -> pd.DataFrame:
        return pd.DataFrame([{"document_id": "doc-1", "text": "chunk"}])


@pytest.fixture
def exported_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    tracing._reset_tracing_for_tests()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)
    try:
        yield exported
    finally:
        tracing._reset_tracing_for_tests()


def test_run_pipeline_in_process_links_child_span_to_parent_trace(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    def _fake_build_graph_ingestor_from_spec(*args: Any, **kwargs: Any) -> tuple[_FakeIngestor, str, bool]:
        return _FakeIngestor(), "pdf", False

    monkeypatch.setattr(
        pipeline_executor,
        "_build_graph_ingestor_from_spec",
        _fake_build_graph_ingestor_from_spec,
    )

    tracing.configure_tracing(service_role="parent-test")
    with tracing.start_span("parent.request"):
        parent_trace_id = tracing.current_trace_id_hex()
        carrier = dict(tracing.inject_trace_context())

        row_count, result_data, _elapsed = pipeline_executor._run_pipeline_in_process(
            "contract.pdf",
            b"%PDF-1.4\n",
            {},
            None,
            trace_context=carrier,
            pool_label="Realtime",
            service_role="standalone",
        )

    assert parent_trace_id is not None
    assert row_count == 1
    assert result_data

    pipeline_span = next(span for span in exported_spans if span.name == "pipeline.ingest")
    assert f"{pipeline_span.context.trace_id:032x}" == parent_trace_id
    assert pipeline_span.attributes["pool"] == "realtime"
    assert pipeline_span.attributes["document.filename"] == "contract.pdf"
