# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenTelemetry propagation in NIM HTTP clients."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.nim.nim import _post_with_retries
from nemo_retriever.service import tracing


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


@pytest.fixture(autouse=True)
def reset_tracing() -> None:
    tracing._reset_tracing_for_tests()
    yield
    tracing._reset_tracing_for_tests()


@pytest.fixture
def exported_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)
    assert tracing.configure_tracing(service_role="standalone") is True
    return exported


def test_post_with_retries_injects_trace_context_and_emits_safe_span(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        text = "{\"ok\": true}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.nim.nim.requests.post", _post)
    original_headers = {
        "Accept": "application/json",
        "Authorization": "Bearer secret-token",
    }
    payload = {"input": [{"secret": "request-body"}]}

    with tracing.start_span("test.parent"):
        parent_trace_id = tracing.current_trace_id_hex()
        result = _post_with_retries(
            invoke_url="http://nim.example/v1/infer",
            payload=payload,
            headers=original_headers,
            timeout_s=10,
            max_retries=1,
            max_429_retries=1,
        )

    assert result == {"ok": True}
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]
    assert captured_headers[0]["Authorization"] == "Bearer secret-token"
    assert "traceparent" not in original_headers

    spans_by_name = {span.name: span for span in exported_spans}
    assert "nim.http.post" in spans_by_name
    span = spans_by_name["nim.http.post"]
    assert f"{span.context.trace_id:032x}" == parent_trace_id

    attrs = dict(span.attributes)
    assert attrs["http.method"] == "POST"
    assert attrs["nim.endpoint"] == "http://nim.example/v1/infer"
    assert attrs["retry.attempt"] == 0
    assert attrs["http.status_code"] == 200
    assert "Authorization" not in attrs
    assert "request.body" not in attrs
    assert "payload" not in attrs
