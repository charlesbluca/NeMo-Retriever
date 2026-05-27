#!/usr/bin/env python3
"""Local PDF RAG helper: hosted extraction + local Super-49B answers.

Typical use from the repository root:

    source scripts/support/super49b_remote_pdf_env.sh
    uv run --project nemo_retriever --extra llm python examples/local_super49b_pdf_rag.py ingest data/multimodal_test.pdf
    uv run --project nemo_retriever --extra llm python examples/local_super49b_pdf_rag.py answer "What is in this document?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


DEFAULT_PAGE_ELEMENTS_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
DEFAULT_OCR_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"
DEFAULT_GRAPHIC_ELEMENTS_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1"
DEFAULT_TABLE_STRUCTURE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
DEFAULT_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_LLM_MODEL = "openai/nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_LLM_API_BASE = "http://localhost:8000/v1"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _int_env(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    return float(value)


def _common_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lancedb_uri": args.lancedb_uri,
        "table_name": args.table_name,
        "embed_invoke_url": _env("EMBED_INVOKE_URL", DEFAULT_EMBED_URL),
        "embed_model_name": _env("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL),
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def ingest(args: argparse.Namespace) -> int:
    from nemo_retriever.adapters.cli.sdk_workflow import ingest_documents

    common = _common_payload(args)
    summary = ingest_documents(
        args.documents,
        input_type=args.input_type,
        run_mode=args.run_mode,
        lancedb_uri=common["lancedb_uri"],
        table_name=common["table_name"],
        overwrite=not args.append,
        page_elements_invoke_url=_env("PAGE_ELEMENTS_INVOKE_URL", DEFAULT_PAGE_ELEMENTS_URL),
        ocr_invoke_url=_env("OCR_INVOKE_URL", DEFAULT_OCR_URL),
        ocr_version=_env("OCR_VERSION", "v1"),
        graphic_elements_invoke_url=_env("GRAPHIC_ELEMENTS_INVOKE_URL", DEFAULT_GRAPHIC_ELEMENTS_URL),
        table_structure_invoke_url=_env("TABLE_STRUCTURE_INVOKE_URL", DEFAULT_TABLE_STRUCTURE_URL),
        table_output_format=args.table_output_format,
        embed_invoke_url=common["embed_invoke_url"],
        embed_model_name=common["embed_model_name"],
    )
    _print_json(
        {
            "documents": summary["documents"],
            "lancedb_uri": summary["lancedb_uri"],
            "table_name": summary["table_name"],
            "status": "ingested",
        }
    )
    return 0


def _build_retriever(args: argparse.Namespace):
    from nemo_retriever.retriever import Retriever

    common = _common_payload(args)
    return Retriever(
        run_mode="service",
        top_k=args.top_k,
        vdb_kwargs={"uri": common["lancedb_uri"], "table_name": common["table_name"]},
        embed_kwargs={
            "embed_invoke_url": common["embed_invoke_url"],
            "model_name": common["embed_model_name"],
            "embed_model_name": common["embed_model_name"],
        },
    )


def _build_llm():
    from nemo_retriever.llm import LiteLLMClient

    return LiteLLMClient.from_kwargs(
        model=_env("RETRIEVER_LLM_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL,
        api_base=_env("RETRIEVER_LLM_API_BASE", DEFAULT_LLM_API_BASE),
        api_key=_env("RETRIEVER_LLM_API_KEY", "not-needed"),
        temperature=0.0,
        max_tokens=_int_env("RETRIEVER_LLM_MAX_TOKENS", 512),
        timeout=_float_env("RETRIEVER_LLM_TIMEOUT", 180.0),
        rag_system_prompt_prefix=_env("RETRIEVER_LLM_RAG_SYSTEM_PROMPT_PREFIX", "/no_think"),
    )


def answer(args: argparse.Namespace) -> int:
    retriever = _build_retriever(args)
    llm = _build_llm()
    result = retriever.answer(args.query, llm=llm, top_k=args.top_k)
    payload: dict[str, Any] = {
        "query": result.query,
        "answer": result.answer,
        "model": result.model,
        "latency_s": result.latency_s,
        "error": result.error,
        "chunk_count": len(result.chunks),
    }
    if args.show_chunks:
        payload["chunks"] = result.chunks
        payload["metadata"] = result.metadata
    _print_json(payload)
    return 1 if result.error else 0


def smoke_llm(args: argparse.Namespace) -> int:
    llm = _build_llm()
    text, latency_s = llm.complete(
        [
            {
                "role": "system",
                "content": f"{_env('RETRIEVER_LLM_RAG_SYSTEM_PROMPT_PREFIX', '/no_think')}\nAnswer directly.",
            },
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
    )
    _print_json({"model": llm.model, "latency_s": latency_s, "text": text})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    parser.add_argument("--lancedb-uri", default=_env("RETRIEVER_LANCEDB_URI", "lancedb"))
    parser.add_argument("--table-name", default=_env("RETRIEVER_LANCEDB_TABLE", "nv-ingest"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Extract PDFs remotely, embed remotely, and write LanceDB.")
    ingest_parser.add_argument("documents", nargs="+")
    ingest_parser.add_argument("--input-type", default="pdf", choices=["auto", "pdf", "doc", "txt", "html", "image"])
    ingest_parser.add_argument("--run-mode", default="inprocess", choices=["inprocess", "batch"])
    ingest_parser.add_argument("--append", action="store_true", help="Append instead of overwriting the LanceDB table.")
    ingest_parser.add_argument("--table-output-format", default="markdown", choices=["pseudo_markdown", "markdown"])
    ingest_parser.set_defaults(func=ingest)

    answer_parser = subparsers.add_parser("answer", help="Retrieve from LanceDB and answer with local Super-49B.")
    answer_parser.add_argument("query")
    answer_parser.add_argument("--top-k", type=int, default=5)
    answer_parser.add_argument("--show-chunks", action="store_true")
    answer_parser.set_defaults(func=answer)

    smoke_parser = subparsers.add_parser("smoke-llm", help="Call the local Super-49B chat endpoint without retrieval.")
    smoke_parser.add_argument("prompt", nargs="?", default="Say hello in one short sentence.")
    smoke_parser.add_argument("--max-tokens", type=int, default=64)
    smoke_parser.set_defaults(func=smoke_llm)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
