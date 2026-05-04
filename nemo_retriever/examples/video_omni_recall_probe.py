# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental video -> Nano Omni extraction -> retrieval probe.

This script is intentionally self-contained for quick recall experiments and is
not the production GraphIngestor path.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from nemo_retriever.text_embed.main_text_embed import TextEmbeddingConfig, create_text_embeddings_for_df


API_KEY_ENV = "NGC_NV_DEVELOPER_NVCF"
ARTIFACT_DIR = Path(".artifacts/video_omni_probe")
CHAT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_ENDPOINT = "https://integrate.api.nvidia.com/v1"
NVIDIA_ENDPOINT_HOSTS = {"integrate.api.nvidia.com"}
OMNI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
TEXT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
PROMPT_VERSION = "video-omni-recall-probe-v1"
FIXTURE_VERSION = "synthetic-video-fixture-v2"
FIXTURE_FRAME_SIZE = (960, 540)
FIXTURE_METADATA_FILENAME = "synthetic_fixture_metadata.json"
DEFAULT_EVAL_MAX_VIDEOS = 5
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 5.0


@dataclass(frozen=True)
class SceneSpec:
    segment_id: str
    start_seconds: float
    end_seconds: float
    title: str
    subtitle: str
    object_label: str
    background_rgb: tuple[int, int, int]
    accent_rgb: tuple[int, int, int]
    expected_terms: tuple[str, ...]


@dataclass(frozen=True)
class ProbeConfig:
    artifact_dir: Path
    chat_endpoint: str
    embed_endpoint: str
    omni_model: str
    embed_model: str
    text_embed_model: str
    top_k: int
    force_fixture: bool
    use_cache: bool
    dry_run: bool
    queries: tuple[str, ...]
    video_path: Path | None = None
    dataset_dir: Path | None = None
    video_name: str | None = None
    queries_file: Path | None = None
    query_expectations: tuple[dict[str, Any], ...] = ()
    allow_custom_endpoint: bool = False
    dataset_eval: bool = False
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS
    max_videos: int | None = DEFAULT_EVAL_MAX_VIDEOS
    video_bin: str | None = None
    query_limit: int | None = None


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class EvalVideoJob:
    video_name: str
    video_path: Path
    duration_seconds: float
    queries: tuple[str, ...]
    query_expectations: tuple[dict[str, Any], ...]


DEFAULT_SCENES = (
    SceneSpec(
        segment_id="scene-alpha",
        start_seconds=0.0,
        end_seconds=2.5,
        title="ALPHA-17",
        subtitle="Red warning triangle beside a blue crate",
        object_label="red triangle",
        background_rgb=(34, 45, 64),
        accent_rgb=(230, 70, 70),
        expected_terms=("alpha-17", "warning", "red triangle", "blue crate"),
    ),
    SceneSpec(
        segment_id="scene-beta",
        start_seconds=2.5,
        end_seconds=5.0,
        title="BETA PANEL",
        subtitle="Green square status panel next to yellow gauge",
        object_label="green square",
        background_rgb=(31, 61, 44),
        accent_rgb=(76, 175, 80),
        expected_terms=("beta panel", "green square", "yellow gauge"),
    ),
    SceneSpec(
        segment_id="scene-calibration",
        start_seconds=5.0,
        end_seconds=7.5,
        title="CALIBRATION",
        subtitle="Purple circle confirms camera alignment",
        object_label="purple circle",
        background_rgb=(55, 42, 74),
        accent_rgb=(155, 93, 229),
        expected_terms=("calibration", "purple circle", "camera alignment"),
    ),
    SceneSpec(
        segment_id="scene-audio",
        start_seconds=7.5,
        end_seconds=10.0,
        title="AUDIO CHECK",
        subtitle="White waveform marker and spoken content if audio is present",
        object_label="white waveform",
        background_rgb=(45, 45, 45),
        accent_rgb=(245, 245, 245),
        expected_terms=("audio check", "waveform", "spoken", "speech"),
    ),
)

DEFAULT_QUERIES = (
    "Which segment shows ALPHA-17?",
    "What scene contains the calibration panel?",
    "Which segment includes spoken or audio content?",
    "What colored object appears with the warning text?",
)


def parse_args(argv: list[str] | None = None) -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--chat-endpoint", default=CHAT_ENDPOINT)
    parser.add_argument("--embed-endpoint", default=EMBED_ENDPOINT)
    parser.add_argument("--omni-model", default=OMNI_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--text-embed-model", default=TEXT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--force-fixture", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--video-name")
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--allow-custom-endpoint", action="store_true")
    parser.add_argument("--dataset-eval", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=DEFAULT_CHUNK_OVERLAP_SECONDS)
    parser.add_argument("--max-videos", type=int, default=DEFAULT_EVAL_MAX_VIDEOS)
    parser.add_argument("--video-bin")
    args = parser.parse_args(argv)

    video_path = args.video_path
    video_name = args.video_name
    queries_file = args.queries_file

    if args.dataset_eval:
        queries = tuple(args.queries) if args.queries else DEFAULT_QUERIES
        query_expectations = tuple({} for _ in queries)
    else:
        if args.dataset_dir:
            if queries_file is None:
                queries_file = args.dataset_dir / "query.csv"
            if video_path is None:
                video_name = video_name or first_query_video_name(queries_file)
                video_path = resolve_dataset_video_path(args.dataset_dir, video_name)
            if video_name is None and video_path is not None:
                video_name = video_path.stem
        elif video_path is not None and video_name is None:
            video_name = video_path.stem

        query_expectations = ()
        if args.queries:
            queries = tuple(args.queries)
            query_expectations = tuple({} for _ in queries)
        elif queries_file is not None:
            loaded_queries, query_expectations = load_queries_csv(queries_file, limit=args.query_limit, video_name=video_name)
            queries = loaded_queries
        else:
            queries = DEFAULT_QUERIES

    return ProbeConfig(
        artifact_dir=args.artifact_dir,
        chat_endpoint=args.chat_endpoint,
        embed_endpoint=args.embed_endpoint,
        omni_model=args.omni_model,
        embed_model=args.embed_model,
        text_embed_model=args.text_embed_model,
        top_k=max(1, args.top_k),
        force_fixture=args.force_fixture,
        use_cache=not args.no_cache,
        dry_run=args.dry_run,
        queries=queries,
        video_path=video_path,
        dataset_dir=args.dataset_dir,
        video_name=video_name,
        queries_file=queries_file,
        query_expectations=query_expectations,
        allow_custom_endpoint=args.allow_custom_endpoint,
        dataset_eval=args.dataset_eval,
        chunk_seconds=float(args.chunk_seconds),
        chunk_overlap_seconds=float(args.chunk_overlap_seconds),
        max_videos=max(1, args.max_videos) if args.max_videos is not None else None,
        video_bin=args.video_bin,
        query_limit=args.query_limit,
    )


def parse_time_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        if ":" not in text:
            seconds = float(text)
            return seconds if math.isfinite(seconds) else None
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3 or any(not math.isfinite(part) for part in parts):
        return None
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total


def first_query_video_name(queries_file: Path) -> str | None:
    if not queries_file.exists():
        return None
    query_df = pd.read_csv(queries_file)
    if "name" not in query_df.columns or query_df.empty:
        return None
    value = query_df.loc[0, "name"]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def resolve_dataset_video_path(dataset_dir: Path, video_name: str | None = None) -> Path | None:
    corpus_dir = dataset_dir / "corpus"
    search_dir = corpus_dir if corpus_dir.exists() else dataset_dir
    video_extensions = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
    candidates = sorted(path for path in search_dir.iterdir() if path.is_file() and path.suffix.lower() in video_extensions)
    if not candidates:
        if video_name:
            raise FileNotFoundError(f"No video named {video_name!r} found under {search_dir}.")
        return None
    if video_name:
        requested = Path(video_name)
        requested_stem = requested.stem
        requested_name = requested.name
        for candidate in candidates:
            if candidate.name == requested_name or candidate.stem == requested_stem:
                return candidate
        raise FileNotFoundError(f"No video named {video_name!r} found under {search_dir}.")
    return candidates[0]


def load_queries_csv(
    path: Path,
    *,
    limit: int | None = None,
    video_name: str | None = None,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    query_df = pd.read_csv(path)
    if "question" not in query_df.columns:
        raise ValueError(f"Query CSV {path} must contain a 'question' column.")
    if video_name and "name" in query_df.columns:
        video_stem = Path(video_name).stem
        name_series = query_df["name"].fillna("").astype(str).str.strip()
        query_df = query_df[name_series.map(lambda value: Path(value).stem == video_stem)]
    if limit is not None:
        query_df = query_df.head(max(0, limit))
    queries, expectations = queries_from_dataframe(query_df)
    if not queries:
        source = f" for video {video_name!r}" if video_name else ""
        raise ValueError(f"No usable queries found in {path}{source}.")
    return queries, expectations


def queries_from_dataframe(query_df: pd.DataFrame) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    queries: list[str] = []
    expectations: list[dict[str, Any]] = []
    for _, row in query_df.iterrows():
        question = row.get("question")
        if pd.isna(question) or not str(question).strip():
            continue
        queries.append(str(question))
        expectation: dict[str, Any] = {}
        name = row.get("name")
        if name is not None and not pd.isna(name) and str(name).strip():
            expectation["source_video_name"] = str(name).strip()
        answer = row.get("answer")
        if answer is not None and not pd.isna(answer):
            expectation["expected_answer"] = str(answer)
        modality = row.get("answer_modality")
        if modality is not None and not pd.isna(modality):
            expectation["answer_modality"] = str(modality)
        start = parse_time_seconds(row.get("start_time"))
        end = parse_time_seconds(row.get("end_time"))
        if start is not None:
            expectation["expected_start_seconds"] = start
        if end is not None:
            expectation["expected_end_seconds"] = end
        expectations.append(expectation)
    return tuple(queries), tuple(expectations)


def plan_video_chunks(duration_seconds: float, *, chunk_seconds: float, overlap_seconds: float) -> list[ChunkSpec]:
    duration = finite_float(duration_seconds, 0.0)
    chunk = finite_float(chunk_seconds, 0.0)
    overlap = finite_float(overlap_seconds, 0.0)
    if duration <= 0.0:
        return []
    if chunk <= 0.0:
        raise ValueError("chunk_seconds must be positive.")
    if overlap < 0.0 or overlap >= chunk:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk_seconds.")

    chunks: list[ChunkSpec] = []
    step = chunk - overlap
    start = 0.0
    index = 0
    while start < duration:
        end = min(duration, start + chunk)
        chunks.append(ChunkSpec(f"chunk-{index:04d}", round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
        index += 1
    return chunks


def load_eval_video_jobs(
    dataset_dir: Path,
    *,
    video_name: str | None = None,
    video_bin: str | None = None,
    max_videos: int | None = DEFAULT_EVAL_MAX_VIDEOS,
    query_limit: int | None = None,
) -> list[EvalVideoJob]:
    query_path = dataset_dir / "query.csv"
    query_df = pd.read_csv(query_path)
    for column in ("name", "question"):
        if column not in query_df.columns:
            raise ValueError(f"Query CSV {query_path} must contain a '{column}' column.")
    if video_bin and "video_bin" in query_df.columns:
        query_df = query_df[query_df["video_bin"].fillna("").astype(str) == video_bin]
    if video_name:
        video_stem = Path(video_name).stem
        names = query_df["name"].fillna("").astype(str).str.strip()
        query_df = query_df[names.map(lambda value: Path(value).stem == video_stem)]

    jobs: list[EvalVideoJob] = []
    seen: set[str] = set()
    ordered_names = [str(value).strip() for value in query_df["name"].tolist() if str(value).strip()]
    for name in ordered_names:
        stem = Path(name).stem
        if stem in seen:
            continue
        seen.add(stem)
        group = query_df[query_df["name"].fillna("").astype(str).str.strip().map(lambda value: Path(value).stem == stem)]
        if query_limit is not None:
            group = group.head(max(0, query_limit))
        queries, expectations = queries_from_dataframe(group)
        if not queries:
            continue
        video_path = resolve_dataset_video_path(dataset_dir, stem)
        duration = None
        if "video_duration" in group.columns:
            for value in group["video_duration"].tolist():
                duration = parse_time_seconds(value)
                if duration is not None:
                    break
        if duration is None:
            duration = max(
                (expectation.get("expected_end_seconds", 0.0) for expectation in expectations),
                default=0.0,
            )
        jobs.append(EvalVideoJob(stem, video_path, float(duration), queries, expectations))
        if max_videos is not None and len(jobs) >= max_videos:
            break
    if not jobs:
        detail = f" for video {video_name!r}" if video_name else ""
        raise ValueError(f"No usable eval jobs found in {query_path}{detail}.")
    return jobs


def safe_excerpt(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text

    head = max(1, limit // 2)
    tail = max(1, limit - head - 20)
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def require_api_key(env: dict[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    api_key = source.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing required NVIDIA API key in environment variable {API_KEY_ENV}.")
    return api_key


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Missing system ffmpeg: ffmpeg is required for the video recall probe but was not found on PATH.")


def run_command(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        command = shlex.join(args)
        stdout = safe_excerpt(result.stdout)
        stderr = safe_excerpt(result.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


def artifact_path(config: ProbeConfig, name: str) -> Path:
    return config.artifact_dir / name


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def embedding_from_payload(payload: Any) -> list[float]:
    if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
        return [float(v) for v in payload["embedding"]]
    return []


def embed_dataframe(
    rows_df: pd.DataFrame,
    *,
    api_key: str,
    endpoint: str,
    model_name: str,
    modality: str,
    output_column: str,
    batch_size: int = 8,
    input_type: str = "passage",
    allow_custom_endpoint: bool = False,
) -> pd.DataFrame:
    require_allowed_endpoint(endpoint, allow_custom_endpoint=allow_custom_endpoint)
    cfg = TextEmbeddingConfig(
        api_key=api_key,
        embedding_nim_endpoint=endpoint,
        embedding_model=model_name,
        input_type=input_type,
        batch_size=batch_size,
        text_column="text",
        write_embedding_to_metadata=False,
        output_payload_column=output_column,
        embed_modality=modality,
        nim_http_max_concurrent=4,
    )
    out_df, _ = create_text_embeddings_for_df(
        rows_df.copy(),
        task_config={
            "api_key": api_key,
            "endpoint_url": endpoint,
            "model_name": model_name,
            "nim_http_max_concurrent": 4,
        },
        transform_config=cfg,
    )

    if output_column not in out_df.columns:
        raise RuntimeError(f"Embedding response missing output column {output_column}.")
    if len(out_df) != len(rows_df):
        raise RuntimeError(f"Embedding response returned {len(out_df)} rows for {len(rows_df)} input rows.")

    embeddings = [embedding_from_payload(payload) for payload in out_df[output_column].tolist()]
    if not embeddings or not any(embeddings):
        raise RuntimeError(f"No embeddings produced for {output_column}.")

    expected_dimension: int | None = None
    for row_number, embedding in enumerate(embeddings):
        if not embedding:
            raise RuntimeError(f"Missing or empty embedding at row {row_number} in {output_column}.")
        dimension = len(embedding)
        if expected_dimension is None:
            expected_dimension = dimension
        elif dimension != expected_dimension:
            raise RuntimeError(
                f"Inconsistent embedding dimensions in {output_column}: "
                f"row 0 has {expected_dimension}, row {row_number} has {dimension}."
            )
    return out_df


def embed_query(
    query: str,
    *,
    api_key: str,
    endpoint: str,
    model_name: str,
    allow_custom_endpoint: bool = False,
) -> list[float]:
    query_df = pd.DataFrame([{"text": query, "metadata": {"query": query}}])
    out_df = embed_dataframe(
        query_df,
        api_key=api_key,
        endpoint=endpoint,
        model_name=model_name,
        modality="text",
        output_column="query_embedding",
        batch_size=1,
        input_type="query",
        allow_custom_endpoint=allow_custom_endpoint,
    )
    return embedding_from_payload(out_df.loc[0, "query_embedding"])


def rank_rows(rows_df: pd.DataFrame, query_embedding: list[float], embedding_column: str, top_k: int) -> list[dict[str, Any]]:
    if not query_embedding:
        raise RuntimeError("Query embedding is empty; cannot rank rows without a query vector.")

    ranked = []
    for _, row in rows_df.iterrows():
        segment_id = str(row.get("segment_id"))
        embedding = embedding_from_payload(row.get(embedding_column))
        if not embedding:
            raise RuntimeError(f"Missing or empty embedding in {embedding_column} for segment {segment_id}.")
        if len(query_embedding) != len(embedding):
            raise RuntimeError(
                f"Embedding dimension mismatch in {embedding_column}: "
                f"query dimension {len(query_embedding)}, row dimension {len(embedding)} "
                f"for segment {segment_id}."
            )
        score = cosine_similarity(query_embedding, embedding)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        ranked.append(
            {
                "segment_id": segment_id,
                "score": score,
                "text": str(row.get("text") or ""),
                "source_video_name": metadata.get("source_video_name"),
                "start_seconds": metadata.get("start_seconds"),
                "end_seconds": metadata.get("end_seconds"),
                "raw_omni_segment": metadata.get("raw_omni_segment", {}),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, int(top_k))]


def expected_query_terms(query: str) -> tuple[str, ...]:
    q = query.lower()
    if "colored object" in q:
        return ("red triangle", "warning", "alpha-17")
    if "alpha" in q or "warning" in q:
        return ("alpha-17", "warning", "scene-alpha")
    if "calibration" in q:
        return ("calibration",)
    if "audio" in q or "spoken" in q or "speech" in q:
        return ("audio", "speech", "spoken", "waveform")
    return ()


def hit_evidence_text(hit: dict[str, Any]) -> str:
    evidence = [str(hit.get("text") or ""), str(hit.get("segment_id") or "")]
    raw_segment = hit.get("raw_omni_segment")
    if isinstance(raw_segment, dict):
        for field in (
            "segment_id",
            "summary",
            "visual_text",
            "objects",
            "actions",
            "audio_or_speech",
            "retrieval_keywords",
            "uncertainties",
        ):
            evidence.extend(flatten_str_list(raw_segment.get(field)))
    return "\n".join(part for part in evidence if part).lower()


def hit_audio_evidence_text(hit: dict[str, Any]) -> str:
    evidence = [str(hit.get("text") or "")]
    segment_ids = flatten_str_list(hit.get("segment_id"))
    raw_segment = hit.get("raw_omni_segment")
    if isinstance(raw_segment, dict):
        segment_ids.extend(flatten_str_list(raw_segment.get("segment_id")))
        for field in (
            "summary",
            "visual_text",
            "objects",
            "actions",
            "audio_or_speech",
            "retrieval_keywords",
            "uncertainties",
        ):
            evidence.extend(flatten_str_list(raw_segment.get(field)))
    haystack = "\n".join(part for part in evidence if part)
    for segment_id in segment_ids:
        haystack = haystack.replace(segment_id, " ")
    haystack = re.sub(r"\baudio\s+or\s+speech\s*:", " ", haystack, flags=re.IGNORECASE)
    return haystack.lower()


def hit_contains_terms(hit: dict[str, Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    haystack = hit_evidence_text(hit)
    lower_terms = tuple(term.lower() for term in terms)

    if lower_terms == ("alpha-17", "warning", "scene-alpha"):
        has_cue = "alpha-17" in haystack or "warning" in haystack
        return ("alpha-17" in haystack and "warning" in haystack) or ("scene-alpha" in haystack and has_cue)
    if lower_terms == ("audio", "speech", "spoken", "waveform"):
        audio_haystack = hit_audio_evidence_text(hit)
        return any(term in audio_haystack for term in lower_terms)
    if lower_terms == ("red triangle", "warning", "alpha-17"):
        return "red triangle" in haystack and (
            "warning" in haystack or "alpha-17" in haystack or "scene-alpha" in haystack
        )
    return any(term in haystack for term in lower_terms)


def normalized_lookup_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def hit_contains_expected_answer(hit: dict[str, Any], expected_answer: Any) -> bool:
    answer = normalized_lookup_text(expected_answer)
    if not answer:
        return False
    haystack = normalized_lookup_text(hit_evidence_text(hit))
    return answer in haystack


def hit_contains_expected(hit: dict[str, Any], terms: tuple[str, ...], expected_answer: Any = None) -> bool:
    return hit_contains_terms(hit, terms) or hit_contains_expected_answer(hit, expected_answer)


def format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(score):
        return "n/a"
    return f"{score:.4f}"


def ranges_overlap(start_a: Any, end_a: Any, start_b: Any, end_b: Any) -> bool:
    parsed = [finite_float(value, math.nan) for value in (start_a, end_a, start_b, end_b)]
    if any(not math.isfinite(value) for value in parsed):
        return False
    a_start, a_end, b_start, b_end = parsed
    return max(a_start, b_start) <= min(a_end, b_end)


def hit_overlaps_expected_time(hit: dict[str, Any], expectation: dict[str, Any]) -> bool:
    if "expected_start_seconds" not in expectation or "expected_end_seconds" not in expectation:
        return False
    expected_source = expectation.get("source_video_name")
    hit_source = hit.get("source_video_name")
    if expected_source and hit_source and Path(str(expected_source)).stem != Path(str(hit_source)).stem:
        return False
    return ranges_overlap(
        hit.get("start_seconds"),
        hit.get("end_seconds"),
        expectation.get("expected_start_seconds"),
        expectation.get("expected_end_seconds"),
    )


def compare_query_results(
    queries: tuple[str, ...],
    rows_df: pd.DataFrame,
    *,
    api_key: str,
    endpoint: str,
    model_name: str,
    top_k: int,
    query_expectations: tuple[dict[str, Any], ...] = (),
    allow_custom_endpoint: bool = False,
    text_model_name: str | None = None,
) -> list[dict[str, Any]]:
    if not queries:
        return []

    text_embedding_model = text_model_name or model_name
    text_df = embed_dataframe(
        rows_df,
        api_key=api_key,
        endpoint=endpoint,
        model_name=text_embedding_model,
        modality="text",
        output_column="text_embedding",
        allow_custom_endpoint=allow_custom_endpoint,
    )
    vl_df = embed_dataframe(
        rows_df,
        api_key=api_key,
        endpoint=endpoint,
        model_name=model_name,
        modality="text_image",
        output_column="vl_text_image_embedding",
        allow_custom_endpoint=allow_custom_endpoint,
    )
    results = []
    for index, query in enumerate(queries):
        text_query_embedding = embed_query(
            query,
            api_key=api_key,
            endpoint=endpoint,
            model_name=text_embedding_model,
            allow_custom_endpoint=allow_custom_endpoint,
        )
        vl_query_embedding = (
            text_query_embedding
            if text_embedding_model == model_name
            else embed_query(
                query,
                api_key=api_key,
                endpoint=endpoint,
                model_name=model_name,
                allow_custom_endpoint=allow_custom_endpoint,
            )
        )
        text_hits = rank_rows(text_df, text_query_embedding, "text_embedding", top_k)
        vl_hits = rank_rows(vl_df, vl_query_embedding, "vl_text_image_embedding", top_k)
        terms = expected_query_terms(query)
        expectation = query_expectations[index] if index < len(query_expectations) else {}
        expected_answer = expectation.get("expected_answer") if isinstance(expectation, dict) else None
        result = {
            "query": query,
            "expected_terms": list(terms),
            "text_embedding_model": text_embedding_model,
            "vl_embedding_model": model_name,
            "text_only": text_hits,
            "vl_text_image": vl_hits,
            "text_only_top_contains_expected": bool(
                text_hits and hit_contains_expected(text_hits[0], terms, expected_answer)
            ),
            "vl_top_contains_expected": bool(vl_hits and hit_contains_expected(vl_hits[0], terms, expected_answer)),
            "top_result_changed": bool(text_hits and vl_hits and text_hits[0]["segment_id"] != vl_hits[0]["segment_id"]),
        }
        if expectation:
            if "expected_answer" in expectation:
                result["expected_answer"] = expectation["expected_answer"]
            if "answer_modality" in expectation:
                result["answer_modality"] = expectation["answer_modality"]
            if "source_video_name" in expectation:
                result["source_video_name"] = expectation["source_video_name"]
            if "expected_start_seconds" in expectation and "expected_end_seconds" in expectation:
                result["expected_time_range"] = {
                    "start_seconds": expectation["expected_start_seconds"],
                    "end_seconds": expectation["expected_end_seconds"],
                }
                result["text_only_top_overlaps_expected_time"] = bool(
                    text_hits and hit_overlaps_expected_time(text_hits[0], expectation)
                )
                result["vl_top_overlaps_expected_time"] = bool(vl_hits and hit_overlaps_expected_time(vl_hits[0], expectation))
                result["text_only_any_overlaps_expected_time"] = any(
                    hit_overlaps_expected_time(hit, expectation) for hit in text_hits
                )
                result["vl_any_overlaps_expected_time"] = any(hit_overlaps_expected_time(hit, expectation) for hit in vl_hits)
        results.append(result)
    return results


def scene_specs() -> tuple[SceneSpec, ...]:
    return DEFAULT_SCENES


def total_duration_seconds(scenes: tuple[SceneSpec, ...] = DEFAULT_SCENES) -> float:
    return max(scene.end_seconds for scene in scenes)


def uniform_scene_duration_seconds(scenes: tuple[SceneSpec, ...]) -> float:
    durations = [scene.end_seconds - scene.start_seconds for scene in scenes]
    if not durations:
        raise ValueError("At least one scene is required to build the synthetic fixture.")
    first_duration = durations[0]
    if any(duration != first_duration for duration in durations):
        raise ValueError("Synthetic fixture generation expects uniform scene durations.")
    return first_duration


def load_default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_scene_frame(scene: SceneSpec, path: Path, *, size: tuple[int, int] = (960, 540)) -> None:
    image = Image.new("RGB", size, scene.background_rgb)
    draw = ImageDraw.Draw(image)
    font_large = load_default_font(64)
    font_small = load_default_font(28)
    w, h = size

    draw.rectangle((40, 40, w - 40, h - 40), outline=scene.accent_rgb, width=8)
    draw.text((72, 76), scene.title, fill=(255, 255, 255), font=font_large)
    draw.text((76, 168), scene.subtitle, fill=(235, 235, 235), font=font_small)
    draw.text((76, h - 92), f"SEGMENT: {scene.segment_id}", fill=(220, 220, 220), font=font_small)

    cx, cy = int(w * 0.74), int(h * 0.55)
    if "triangle" in scene.object_label:
        draw.polygon([(cx, cy - 90), (cx - 95, cy + 80), (cx + 95, cy + 80)], fill=scene.accent_rgb)
        draw.rectangle((cx - 55, cy + 100, cx + 55, cy + 160), fill=(40, 95, 190))
    elif "square" in scene.object_label:
        draw.rectangle((cx - 95, cy - 95, cx + 95, cy + 95), fill=scene.accent_rgb)
        draw.ellipse((cx + 125, cy - 60, cx + 245, cy + 60), fill=(235, 200, 55))
    elif "circle" in scene.object_label:
        draw.ellipse((cx - 105, cy - 105, cx + 105, cy + 105), fill=scene.accent_rgb)
        draw.line((cx - 160, cy, cx + 160, cy), fill=(255, 255, 255), width=5)
        draw.line((cx, cy - 160, cx, cy + 160), fill=(255, 255, 255), width=5)
    else:
        for offset in range(0, 220, 40):
            draw.arc((cx - 120 + offset, cy - 80, cx - 40 + offset, cy + 80), 270, 90, fill=scene.accent_rgb, width=6)

    image.save(path)


def image_file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_silent_wav(path: Path, duration_seconds: float, sample_rate: int = 16000) -> None:
    frames = int(duration_seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def scene_spec_metadata(scene: SceneSpec) -> dict[str, Any]:
    return {
        "segment_id": scene.segment_id,
        "start_seconds": scene.start_seconds,
        "end_seconds": scene.end_seconds,
        "title": scene.title,
        "subtitle": scene.subtitle,
        "object_label": scene.object_label,
        "background_rgb": list(scene.background_rgb),
        "accent_rgb": list(scene.accent_rgb),
        "expected_terms": list(scene.expected_terms),
    }


def fixture_audio_metadata() -> dict[str, Any]:
    source = Path("data/multimodal_test.wav")
    source_exists = source.exists()
    return {
        "audio_source": "source" if source_exists else "silent_fallback",
        "audio_source_exists": source_exists,
        "audio_source_sha256": file_sha256(source) if source_exists else None,
    }


def expected_fixture_metadata() -> dict[str, Any]:
    scenes = scene_specs()
    metadata = {
        "fixture_version": FIXTURE_VERSION,
        "scene_specs": [scene_spec_metadata(scene) for scene in scenes],
        "frame_size": list(FIXTURE_FRAME_SIZE),
        "expected_duration_seconds": total_duration_seconds(scenes),
        "frame_manifest_keys": sorted(scene.segment_id for scene in scenes),
    }
    metadata.update(fixture_audio_metadata())
    return metadata


def fixture_cache_matches(frame_manifest_path: Path, metadata_path: Path) -> bool:
    try:
        frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(frame_manifest, dict) or not isinstance(metadata, dict):
        return False
    expected = expected_fixture_metadata()
    return metadata == expected and sorted(frame_manifest.keys()) == expected["frame_manifest_keys"]


def build_video_from_frames(frame_paths: list[Path], output_path: Path, seconds_per_scene: float) -> None:
    list_file = output_path.with_suffix(".frames.txt")
    lines: list[str] = []
    for frame in frame_paths:
        lines.append(f"file '{frame.resolve()}'")
        lines.append(f"duration {seconds_per_scene:.3f}")
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vf",
            "fps=24,format=yuv420p",
            str(output_path),
        ]
    )


def prepare_audio_track(artifact_dir: Path, duration_seconds: float) -> Path:
    source = Path("data/multimodal_test.wav")
    target = artifact_dir / "fixture_audio.wav"
    if source.exists():
        run_command(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source), "-t", f"{duration_seconds:.3f}", str(target)])
    else:
        write_silent_wav(target, duration_seconds)
    return target


def generate_synthetic_fixture(config: ProbeConfig) -> tuple[Path, dict[str, str]]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = artifact_path(config, "synthetic_fixture.mp4")
    frame_manifest_path = artifact_path(config, "scene_frames.json")
    metadata_path = artifact_path(config, FIXTURE_METADATA_FILENAME)
    if (
        config.use_cache
        and not config.force_fixture
        and fixture_path.exists()
        and frame_manifest_path.exists()
        and metadata_path.exists()
        and fixture_cache_matches(frame_manifest_path, metadata_path)
    ):
        return fixture_path, json.loads(frame_manifest_path.read_text(encoding="utf-8"))

    check_ffmpeg()
    scenes = scene_specs()
    with tempfile.TemporaryDirectory(prefix="video_omni_fixture_") as tmp:
        tmp_path = Path(tmp)
        frames: list[Path] = []
        frame_b64: dict[str, str] = {}
        for scene in scenes:
            frame_path = tmp_path / f"{scene.segment_id}.png"
            draw_scene_frame(scene, frame_path)
            frames.append(frame_path)
            frame_b64[scene.segment_id] = image_file_to_b64(frame_path)

        video_only = tmp_path / "fixture_video_only.mp4"
        build_video_from_frames(frames, video_only, seconds_per_scene=uniform_scene_duration_seconds(scenes))
        audio_path = prepare_audio_track(tmp_path, total_duration_seconds(scenes))
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(fixture_path),
            ]
        )

    write_json(frame_manifest_path, frame_b64)
    write_json(metadata_path, expected_fixture_metadata())
    return fixture_path, frame_b64


def video_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def build_omni_prompt(*, include_fixture_cues: bool = True, video_name: str | None = None) -> str:
    source_description = "a short synthetic video"
    if video_name:
        source_description = f"video named {video_name}"
    base_prompt = (
        f"You are extracting retrieval records from {source_description}. "
        "Return only valid JSON. Do not wrap it in markdown. "
        "Use exact visible text when it is readable. Do not invent details. "
        "Use this schema: {\"segments\": [{\"segment_id\": string, "
        "\"start_seconds\": number, \"end_seconds\": number, \"summary\": string, "
        "\"visual_text\": [string], \"objects\": [string], \"actions\": [string], "
        "\"audio_or_speech\": string, \"retrieval_keywords\": [string], "
        "\"uncertainties\": [string], \"confidence\": number}]}. "
    )
    if not include_fixture_cues:
        return (
            base_prompt
            + "Create segment windows that cover the relevant visual and audio evidence in the source video."
        )

    expected = [
        {
            "segment_id": scene.segment_id,
            "time_range": [scene.start_seconds, scene.end_seconds],
            "known_fixture_cues": list(dict.fromkeys((scene.title, *scene.expected_terms))),
        }
        for scene in scene_specs()
    ]
    return (
        base_prompt
        + "The fixture has these expected segment windows and cues for calibration only: "
        + json.dumps(expected, sort_keys=True)
    )


def build_omni_payload(
    video_url: str,
    model: str,
    *,
    include_fixture_cues: bool = True,
    video_name: str | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_omni_prompt(include_fixture_cues=include_fixture_cues, video_name=video_name),
                    },
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": True},
        "media_io_kwargs": {"video": {"num_frames": 16, "fps": -1}},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def omni_request_metadata(video_path: Path, config: ProbeConfig) -> dict[str, Any]:
    include_fixture_cues = config.video_path is None
    payload = build_omni_payload(
        "data:video/mp4;base64,<omitted>",
        config.omni_model,
        include_fixture_cues=include_fixture_cues,
        video_name=config.video_name,
    )
    return {
        "model": config.omni_model,
        "endpoint": config.chat_endpoint,
        "prompt_version": PROMPT_VERSION,
        "fixture_sha256": file_sha256(video_path),
        "source_video_name": config.video_name,
        "include_fixture_cues": include_fixture_cues,
        "allow_custom_endpoint": config.allow_custom_endpoint,
        "media_io_kwargs": payload["media_io_kwargs"],
        "mm_processor_kwargs": payload["mm_processor_kwargs"],
        "chat_template_kwargs": payload["chat_template_kwargs"],
        "temperature": payload["temperature"],
        "max_tokens": payload["max_tokens"],
    }


def omni_cache_matches(cached: dict[str, Any], fixture_path: Path, config: ProbeConfig) -> bool:
    request = cached.get("request")
    if not isinstance(request, dict):
        return False
    return request == omni_request_metadata(fixture_path, config)


def require_https_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Refusing to send Authorization header to non-HTTPS endpoint: {url}. Use HTTPS.")


def require_allowed_endpoint(url: str, *, allow_custom_endpoint: bool = False) -> None:
    require_https_endpoint(url)
    if allow_custom_endpoint:
        return
    host = (urlparse(url).hostname or "").lower()
    if host not in NVIDIA_ENDPOINT_HOSTS:
        raise ValueError(
            f"Refusing to send credentials or media to non-NVIDIA endpoint {url}. "
            "Use --allow-custom-endpoint only for endpoints you trust."
        )


def post_json(
    url: str,
    token: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 120.0,
    allow_custom_endpoint: bool = False,
) -> dict[str, Any]:
    require_allowed_endpoint(url, allow_custom_endpoint=allow_custom_endpoint)
    try:
        response = requests.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json=payload,
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {safe_excerpt(response.text)}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Expected JSON object from {url}: {safe_excerpt(response.text)}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}.")
    return data


def extract_message_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Hosted response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Hosted response choice must be an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Hosted response choice missing message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Hosted response message content must be a string.")
    return content


def call_nano_omni(video_path: Path, config: ProbeConfig, token: str) -> dict[str, Any]:
    payload = build_omni_payload(
        video_data_url(video_path),
        config.omni_model,
        include_fixture_cues=config.video_path is None,
        video_name=config.video_name,
    )
    started = time.perf_counter()
    raw_response = post_json(
        config.chat_endpoint,
        token,
        payload,
        allow_custom_endpoint=config.allow_custom_endpoint,
    )
    elapsed = time.perf_counter() - started
    content = extract_message_content(raw_response)
    parsed = extract_json_object(content)
    return {
        "elapsed_seconds": elapsed,
        "request": omni_request_metadata(video_path, config),
        "raw_response": raw_response,
        "parsed": parsed,
    }


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def finite_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(coerced):
        return default
    return coerced


def validate_omni_segments(parsed: Any) -> list[dict[str, Any]]:
    segment_fields = {
        "segment_id",
        "start_seconds",
        "end_seconds",
        "summary",
        "visual_text",
        "objects",
        "actions",
        "audio_or_speech",
        "retrieval_keywords",
        "uncertainties",
        "confidence",
    }
    if isinstance(parsed, list):
        segments = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("segments"), list):
        segments = parsed["segments"]
    elif isinstance(parsed, dict) and any(field in parsed for field in segment_fields):
        segments = [parsed]
    else:
        segments = None
    if not isinstance(segments, list) or not segments:
        raise ValueError("Nano Omni parsed JSON must contain a non-empty 'segments' list.")
    normalized = []
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {i} must be an object.")
        segment_id = str(segment.get("segment_id") or f"segment-{i}")
        start = finite_float(segment.get("start_seconds", 0.0), 0.0)
        end = finite_float(segment.get("end_seconds", start), start)
        if end < start:
            end = start
        confidence = min(1.0, max(0.0, finite_float(segment.get("confidence", 0.0), 0.0)))
        normalized.append(
            {
                "segment_id": segment_id,
                "start_seconds": start,
                "end_seconds": end,
                "summary": str(segment.get("summary") or ""),
                "visual_text": coerce_string_list(segment.get("visual_text")),
                "objects": coerce_string_list(segment.get("objects")),
                "actions": coerce_string_list(segment.get("actions")),
                "audio_or_speech": str(segment.get("audio_or_speech") or ""),
                "retrieval_keywords": coerce_string_list(segment.get("retrieval_keywords")),
                "uncertainties": coerce_string_list(segment.get("uncertainties")),
                "confidence": confidence,
            }
        )
    return normalized


def flatten_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if value is not None and str(value).strip()]


def build_retrieval_text(segment: dict[str, Any]) -> str:
    parts = [
        f"Segment: {segment['segment_id']}",
        f"Summary: {segment.get('summary', '')}",
        "Visual text: " + ", ".join(flatten_str_list(segment.get("visual_text", []))),
        "Objects: " + ", ".join(flatten_str_list(segment.get("objects", []))),
        "Actions: " + ", ".join(flatten_str_list(segment.get("actions", []))),
        f"Audio or speech: {segment.get('audio_or_speech', '')}",
        "Keywords: " + ", ".join(flatten_str_list(segment.get("retrieval_keywords", []))),
    ]
    return "\n".join(part for part in parts if part.strip() and not part.endswith(": "))


def nearest_scene_frame(segment_id: str, frame_b64: dict[str, str], segment: dict[str, Any] | None = None) -> str:
    if segment_id in frame_b64:
        return frame_b64[segment_id]
    if segment is None:
        return ""

    start = finite_float(segment.get("start_seconds"), math.nan)
    end = finite_float(segment.get("end_seconds"), math.nan)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return ""

    midpoint = start + ((end - start) / 2.0)
    scenes = scene_specs()
    for index, scene in enumerate(scenes):
        is_last = index == len(scenes) - 1
        if scene.start_seconds <= midpoint < scene.end_seconds or (is_last and midpoint == scene.end_seconds):
            return frame_b64.get(scene.segment_id, "")
    return ""


def representative_frame_filename(segment_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", segment_id).strip("._")
    return f"{sanitized or 'segment'}.png"


def extract_video_chunk(source_video: Path, chunk: ChunkSpec, output_path: Path) -> None:
    check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{chunk.start_seconds:.3f}",
            "-i",
            str(source_video),
            "-t",
            f"{chunk.end_seconds - chunk.start_seconds:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    )


def extract_representative_frames_from_segments(
    video_path: Path,
    segments: list[dict[str, Any]],
    artifact_dir: Path,
) -> dict[str, str]:
    try:
        check_ffmpeg()
    except RuntimeError as exc:
        print(f"Skipping representative frame extraction: {safe_excerpt(str(exc), limit=240)}")
        return {}
    frames_dir = artifact_dir / "representative_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_b64: dict[str, str] = {}
    used_filenames: set[str] = set()
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "segment")
        start = finite_float(segment.get("start_seconds"), math.nan)
        end = finite_float(segment.get("end_seconds"), math.nan)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            continue
        filename = representative_frame_filename(segment_id)
        if filename in used_filenames:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            index = 2
            while f"{stem}_{index}{suffix}" in used_filenames:
                index += 1
            filename = f"{stem}_{index}{suffix}"
        used_filenames.add(filename)
        frame_path = frames_dir / filename
        midpoint = start + ((end - start) / 2.0)
        try:
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{midpoint:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=960:-1",
                    str(frame_path),
                ]
            )
        except RuntimeError as exc:
            print(f"Skipping representative frame for segment {segment_id}: {safe_excerpt(str(exc), limit=240)}")
            continue
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            print(f"Skipping representative frame for segment {segment_id}: ffmpeg did not produce a frame.")
            continue
        frame_b64[segment_id] = image_file_to_b64(frame_path)
    return frame_b64


def chunk_omni_cache_matches(cached: Any, chunk_path: Path, config: ProbeConfig) -> bool:
    return isinstance(cached, dict) and omni_cache_matches(cached, chunk_path, config)


def load_cached_omni(path: Path, video_path: Path, config: ProbeConfig) -> dict[str, Any] | None:
    if not config.use_cache or config.force_fixture or not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cached if chunk_omni_cache_matches(cached, video_path, config) else None


def write_eval_chunk_error(path: Path, job: EvalVideoJob, chunk: ChunkSpec, exc: Exception) -> None:
    write_json(
        path,
        {
            "video_name": job.video_name,
            "video_path": str(job.video_path),
            "chunk_id": chunk.chunk_id,
            "start_seconds": chunk.start_seconds,
            "end_seconds": chunk.end_seconds,
            "error_type": type(exc).__name__,
            "error": safe_excerpt(str(exc), limit=2000),
        },
    )


def extract_rows_for_eval_job(job: EvalVideoJob, config: ProbeConfig, token: str) -> pd.DataFrame:
    job_dir = config.artifact_dir / "videos" / representative_frame_filename(job.video_name).removesuffix(".png")
    chunks_dir = job_dir / "chunks"
    segments: list[dict[str, Any]] = []
    for chunk in plan_video_chunks(job.duration_seconds, chunk_seconds=config.chunk_seconds, overlap_seconds=config.chunk_overlap_seconds):
        chunk_path = chunks_dir / f"{chunk.chunk_id}.mp4"
        omni_path = chunks_dir / f"{chunk.chunk_id}.omni_response.json"
        error_path = chunks_dir / f"{chunk.chunk_id}.error.json"
        try:
            if not (config.use_cache and chunk_path.exists()):
                extract_video_chunk(job.video_path, chunk, chunk_path)
            chunk_config = replace(config, video_path=chunk_path, video_name=f"{job.video_name}:{chunk.chunk_id}")
            omni_result = load_cached_omni(omni_path, chunk_path, chunk_config)
            if omni_result is None:
                omni_result = call_nano_omni(chunk_path, chunk_config, token)
                write_json(omni_path, omni_result)
            chunk_segments = validate_omni_segments(omni_result["parsed"])
            error_path.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            write_eval_chunk_error(error_path, job, chunk, exc)
            print(f"Skipping {job.video_name} {chunk.chunk_id}: {safe_excerpt(str(exc), limit=240)}")
            continue
        for segment in chunk_segments:
            local_start = max(0.0, float(segment["start_seconds"]))
            local_end = max(local_start, float(segment["end_seconds"]))
            segment = dict(segment)
            segment["segment_id"] = f"{job.video_name}:{chunk.chunk_id}:{segment['segment_id']}"
            segment["start_seconds"] = chunk.start_seconds + local_start
            segment["end_seconds"] = min(job.duration_seconds, chunk.start_seconds + local_end)
            segment["source_video_name"] = job.video_name
            segment["chunk_id"] = chunk.chunk_id
            segments.append(segment)
    if not segments:
        raise RuntimeError(f"Nano Omni produced no usable segments for {job.video_name}.")
    frame_b64 = extract_representative_frames_from_segments(job.video_path, segments, job_dir)
    rows_config = replace(config, video_path=job.video_path, video_name=job.video_name)
    return rows_from_segments(segments, frame_b64, job.video_path, rows_config)


def rows_from_segments(
    segments: list[dict[str, Any]],
    frame_b64: dict[str, str],
    fixture_path: Path,
    config: ProbeConfig,
) -> pd.DataFrame:
    rows = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        source_video_name = str(segment.get("source_video_name") or config.video_name or "")
        image_b64 = nearest_scene_frame(segment_id, frame_b64, segment)
        rows.append(
            {
                "segment_id": segment_id,
                "text": build_retrieval_text(segment),
                "_image_b64": image_b64,
                "_embed_modality": "text_image" if image_b64 else "text",
                "metadata": {
                    "source_path": str(fixture_path),
                    "source_video_name": source_video_name or None,
                    "segment_id": segment_id,
                    "start_seconds": float(segment["start_seconds"]),
                    "end_seconds": float(segment["end_seconds"]),
                    "raw_omni_segment": segment,
                    "extraction_model": config.omni_model,
                    "embedding_model": config.embed_model,
                    "prompt_version": PROMPT_VERSION,
                    "scaffold": {
                        "chat_endpoint": config.chat_endpoint,
                        "embed_endpoint": config.embed_endpoint,
                    },
                },
            }
        )
    return pd.DataFrame(rows)


def extract_json_object(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced_match.group(1)] if fenced_match else [text]
    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"segments": parsed}
        if parsed is not None:
            raise ValueError("Expected a JSON object.")

        segments_match = re.search(r'"segments"\s*:\s*(\[)', candidate)
        if segments_match:
            try:
                segments, _ = decoder.raw_decode(candidate[segments_match.start(1) :])
            except json.JSONDecodeError:
                segments = None
            if isinstance(segments, list):
                return {"segments": segments}

        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"segments": parsed}
            raise ValueError("Expected a JSON object.")

    raise ValueError("No JSON object found.")


def cosine_similarity(a: Any, b: Any) -> float:
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)
    if vector_a.size == 0 or vector_b.size == 0 or vector_a.shape != vector_b.shape:
        return 0.0

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def load_fixture_metadata(fixture_path: Path) -> dict[str, Any]:
    metadata_path = fixture_path.with_name(FIXTURE_METADATA_FILENAME)
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def summarize_observations(
    omni_elapsed_seconds: float,
    fixture_path: Path,
    query_results: list[dict[str, Any]],
    fixture_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = load_fixture_metadata(fixture_path) if fixture_metadata is None else fixture_metadata
    fixture_bytes = fixture_path.stat().st_size if fixture_path.exists() else 0
    text_successes = sum(1 for result in query_results if result.get("text_only_top_contains_expected"))
    vl_successes = sum(1 for result in query_results if result.get("vl_top_contains_expected"))
    text_top_time_overlaps = sum(1 for result in query_results if result.get("text_only_top_overlaps_expected_time"))
    vl_top_time_overlaps = sum(1 for result in query_results if result.get("vl_top_overlaps_expected_time"))
    text_any_time_overlaps = sum(1 for result in query_results if result.get("text_only_any_overlaps_expected_time"))
    vl_any_time_overlaps = sum(1 for result in query_results if result.get("vl_any_overlaps_expected_time"))
    has_time_expectations = any(
        "expected_time_range" in result
        or "text_only_top_overlaps_expected_time" in result
        or "vl_top_overlaps_expected_time" in result
        for result in query_results
    )
    changed = sum(1 for result in query_results if result.get("top_result_changed"))
    total = len(query_results)
    source_kind = metadata.get("source_kind") if isinstance(metadata.get("source_kind"), str) else None
    source_label = "external video" if source_kind == "external_video" else f"{total_duration_seconds():.1f}s of synthetic video"
    pros = [f"Nano Omni returned structured segments for {source_label}."]
    cons = [
        f"Hosted Omni call latency was {omni_elapsed_seconds:.2f}s for a {fixture_bytes} byte MP4.",
        "Segment timing and visual facts are model-generated and should be treated as inspectable evidence, not ground truth.",
    ]
    if total == 0:
        cons.insert(1, "No diagnostic queries were run; retrieval hit metrics are unevaluated.")
    elif has_time_expectations:
        pros.extend(
            [
                f"Text-only top result overlapped expected timestamp windows for {text_top_time_overlaps}/{total} diagnostic queries.",
                f"VL top result overlapped expected timestamp windows for {vl_top_time_overlaps}/{total} diagnostic queries.",
            ]
        )
        cons.insert(1, f"Top result changed between baseline and VL retrieval for {changed}/{total} diagnostic queries.")
    else:
        pros.extend(
            [
                f"Text-only baseline hit expected terms for {text_successes}/{total} diagnostic queries.",
                f"VL text+image hit expected terms for {vl_successes}/{total} diagnostic queries.",
            ]
        )
        cons.insert(1, f"Top result changed between baseline and VL retrieval for {changed}/{total} diagnostic queries.")
    audio_source = metadata.get("audio_source") if isinstance(metadata.get("audio_source"), str) else None
    if audio_source == "silent_fallback":
        cons.append(
            "Fixture audio used a silent fallback; audio-query hits are visual/text diagnostics only and not proof of spoken-audio extraction."
        )
    metrics = {
        "omni_elapsed_seconds": omni_elapsed_seconds,
        "fixture_bytes": fixture_bytes,
        "text_only_expected_hits": text_successes,
        "vl_expected_hits": vl_successes,
        "top_result_changes": changed,
        "query_count": total,
    }
    if has_time_expectations:
        metrics.update(
            {
                "text_only_top_time_overlaps": text_top_time_overlaps,
                "vl_top_time_overlaps": vl_top_time_overlaps,
                "text_only_any_time_overlaps": text_any_time_overlaps,
                "vl_any_time_overlaps": vl_any_time_overlaps,
            }
        )
    if audio_source:
        metrics["audio_source"] = audio_source
    return {
        "pros": pros,
        "cons": cons,
        "metrics": metrics,
    }


def expectation_from_query_result(result: dict[str, Any]) -> dict[str, Any]:
    time_range = result.get("expected_time_range") if isinstance(result.get("expected_time_range"), dict) else {}
    expectation: dict[str, Any] = {}
    if result.get("source_video_name"):
        expectation["source_video_name"] = result["source_video_name"]
    if "start_seconds" in time_range:
        expectation["expected_start_seconds"] = time_range["start_seconds"]
    if "end_seconds" in time_range:
        expectation["expected_end_seconds"] = time_range["end_seconds"]
    return expectation


def reciprocal_rank_for_expected_time(hits: list[dict[str, Any]], expectation: dict[str, Any]) -> float:
    if not expectation:
        return 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit_overlaps_expected_time(hit, expectation):
            return 1.0 / rank
    return 0.0


def rate(numerator: int | float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def dataset_eval_metrics(query_results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    total = len(query_results)
    text_top1 = sum(1 for result in query_results if result.get("text_only_top_overlaps_expected_time"))
    vl_top1 = sum(1 for result in query_results if result.get("vl_top_overlaps_expected_time"))
    text_topk = sum(1 for result in query_results if result.get("text_only_any_overlaps_expected_time"))
    vl_topk = sum(1 for result in query_results if result.get("vl_any_overlaps_expected_time"))
    text_answer = sum(1 for result in query_results if result.get("text_only_top_contains_expected"))
    vl_answer = sum(1 for result in query_results if result.get("vl_top_contains_expected"))
    text_mrr = 0.0
    vl_mrr = 0.0
    for result in query_results:
        expectation = expectation_from_query_result(result)
        text_mrr += reciprocal_rank_for_expected_time(result.get("text_only", []), expectation)
        vl_mrr += reciprocal_rank_for_expected_time(result.get("vl_text_image", []), expectation)
    return {
        "query_count": total,
        "top_k": top_k,
        "text_only_top1_time_overlap_rate": rate(text_top1, total),
        "vl_top1_time_overlap_rate": rate(vl_top1, total),
        f"text_only_top{top_k}_time_overlap_rate": rate(text_topk, total),
        f"vl_top{top_k}_time_overlap_rate": rate(vl_topk, total),
        "text_only_time_mrr": rate(text_mrr, total),
        "vl_time_mrr": rate(vl_mrr, total),
        "text_only_answer_hit_rate": rate(text_answer, total),
        "vl_answer_hit_rate": rate(vl_answer, total),
    }


def summarize_dataset_eval_results(query_results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    by_modality: dict[str, dict[str, Any]] = {}
    modalities = sorted({str(result.get("answer_modality") or "unknown") for result in query_results})
    for modality in modalities:
        modality_results = [result for result in query_results if str(result.get("answer_modality") or "unknown") == modality]
        by_modality[modality] = dataset_eval_metrics(modality_results, top_k=top_k)
    return {
        "metrics": dataset_eval_metrics(query_results, top_k=top_k),
        "by_answer_modality": by_modality,
    }


def compact_text_snippet(text: Any, limit: int = 180) -> str:
    return " ".join(str(text or "").split())[:limit]


def print_query_results(query_results: list[dict[str, Any]]) -> None:
    for result in query_results:
        print(f"Query: {result.get('query', '')}")
        expected_terms = result.get("expected_terms", [])
        print(f"  expected_terms: {', '.join(str(term) for term in expected_terms)}")
        for label in ("text_only", "vl_text_image"):
            print(f"  {label}:")
            for index, hit in enumerate(result.get(label, []), start=1):
                start = hit.get("start_seconds")
                end = hit.get("end_seconds")
                print(
                    f"    rank={index} score={format_score(hit.get('score'))} "
                    f"segment={hit.get('segment_id')} time={start}-{end} "
                    f"snippet={compact_text_snippet(hit.get('text', ''))}"
                )
        print(f"  text_only_top_contains_expected: {bool(result.get('text_only_top_contains_expected'))}")
        print(f"  vl_top_contains_expected: {bool(result.get('vl_top_contains_expected'))}")
        print(f"  top_result_changed: {bool(result.get('top_result_changed'))}")


def print_observations(observations: dict[str, Any]) -> None:
    print("\nObservations")
    print("  Pros:")
    for item in observations["pros"]:
        print(f"    - {item}")
    print("  Cons:")
    for item in observations["cons"]:
        print(f"    - {item}")


def load_jobs_for_config(config: ProbeConfig) -> list[EvalVideoJob]:
    if config.dataset_dir is None:
        raise ValueError("--dataset-dir is required with --dataset-eval.")
    return load_eval_video_jobs(
        config.dataset_dir,
        video_name=config.video_name,
        video_bin=config.video_bin,
        max_videos=config.max_videos,
        query_limit=config.query_limit,
    )


def run_dataset_eval(config: ProbeConfig, token: str) -> dict[str, Any]:
    jobs = load_jobs_for_config(config)
    all_rows: list[pd.DataFrame] = []
    queries: list[str] = []
    expectations: list[dict[str, Any]] = []
    for job in jobs:
        all_rows.append(extract_rows_for_eval_job(job, config, token))
        queries.extend(job.queries)
        expectations.extend(job.query_expectations)
    if not all_rows:
        raise RuntimeError("Dataset eval produced no extracted rows.")
    rows_df = pd.concat(all_rows, ignore_index=True)
    rows_path = artifact_path(config, "eval_rows.json")
    write_json(rows_path, rows_df.to_dict(orient="records"))
    query_results = compare_query_results(
        tuple(queries),
        rows_df,
        api_key=token,
        endpoint=config.embed_endpoint,
        model_name=config.embed_model,
        text_model_name=config.text_embed_model,
        top_k=config.top_k,
        query_expectations=tuple(expectations),
        allow_custom_endpoint=config.allow_custom_endpoint,
    )
    results_path = artifact_path(config, "eval_results.json")
    write_json(results_path, query_results)
    summary = summarize_dataset_eval_results(query_results, top_k=config.top_k)
    summary["config"] = {
        "dataset_dir": str(config.dataset_dir),
        "max_videos": config.max_videos,
        "video_bin": config.video_bin,
        "query_limit": config.query_limit,
        "chunk_seconds": config.chunk_seconds,
        "chunk_overlap_seconds": config.chunk_overlap_seconds,
        "text_embed_model": config.text_embed_model,
        "vl_embed_model": config.embed_model,
    }
    summary_path = artifact_path(config, "eval_summary.json")
    write_json(summary_path, summary)
    return {
        "jobs": jobs,
        "rows_path": rows_path,
        "results": query_results,
        "results_path": results_path,
        "summary": summary,
        "summary_path": summary_path,
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    if config.dataset_eval:
        jobs = load_jobs_for_config(config)
        query_count = sum(len(job.queries) for job in jobs)
        print(f"Dataset eval jobs: {len(jobs)} videos, {query_count} queries")
        if config.dry_run:
            print("Dry run complete; hosted model and embedding calls were skipped.")
            return 0
        token = require_api_key()
        print(f"Loaded API key from {API_KEY_ENV}: True")
        result = run_dataset_eval(config, token)
        print(f"Wrote eval results: {result['results_path']}")
        print(f"Wrote eval summary: {result['summary_path']}")
        print(json.dumps(result["summary"]["metrics"], indent=2, sort_keys=True))
        return 0

    if config.video_path is not None:
        fixture_path = config.video_path
        frame_b64: dict[str, str] = {}
    else:
        fixture_path, frame_b64 = generate_synthetic_fixture(config)
    print(f"Video Omni recall probe artifacts: {config.artifact_dir}")
    if config.video_path is not None:
        print(f"Source video: {fixture_path}")
    else:
        print(f"Synthetic fixture: {fixture_path}")
    print(f"Representative frame count: {len(frame_b64)}")
    if config.dry_run:
        print("Dry run complete; hosted model and embedding calls were skipped.")
        return 0

    omni_path = artifact_path(config, "omni_response.json")
    omni_result: dict[str, Any] | None = None
    token: str | None = None
    if config.use_cache and not config.force_fixture and omni_path.exists():
        try:
            cached_omni_result = json.loads(omni_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"Cached Omni response is unreadable; refreshing: {omni_path}")
            cached_omni_result = None
        if isinstance(cached_omni_result, dict) and omni_cache_matches(cached_omni_result, fixture_path, config):
            omni_result = cached_omni_result
            print(f"Loaded cached Omni response: {omni_path}")
        elif cached_omni_result is not None:
            print(f"Cached Omni response is stale; refreshing: {omni_path}")

    if omni_result is None:
        token = require_api_key()
        print(f"Loaded API key from {API_KEY_ENV}: True")
        omni_result = call_nano_omni(fixture_path, config, token)
        write_json(omni_path, omni_result)
        print(f"Wrote Omni response: {omni_path}")
    segments = validate_omni_segments(omni_result["parsed"])
    print(f"Validated Omni segments: {len(segments)}")
    if config.video_path is not None:
        frame_b64 = extract_representative_frames_from_segments(fixture_path, segments, config.artifact_dir)
        print(f"Representative frame count: {len(frame_b64)}")
    rows_df = rows_from_segments(segments, frame_b64, fixture_path, config)
    rows_path = artifact_path(config, "extracted_rows.json")
    write_json(rows_path, rows_df.to_dict(orient="records"))
    print(f"Wrote extracted rows: {rows_path}")
    if token is None:
        token = require_api_key()
        print(f"Loaded API key from {API_KEY_ENV}: True")
    compare_kwargs: dict[str, Any] = {
        "api_key": token,
        "endpoint": config.embed_endpoint,
        "model_name": config.embed_model,
        "text_model_name": config.text_embed_model,
        "top_k": config.top_k,
        "allow_custom_endpoint": config.allow_custom_endpoint,
    }
    if any(config.query_expectations):
        compare_kwargs["query_expectations"] = config.query_expectations
    query_results = compare_query_results(config.queries, rows_df, **compare_kwargs)
    query_path = artifact_path(config, "query_results.json")
    write_json(query_path, query_results)
    print_query_results(query_results)
    print(f"Wrote query results: {query_path}")
    observations = summarize_observations(
        omni_elapsed_seconds=float(omni_result.get("elapsed_seconds", 0.0)),
        fixture_path=fixture_path,
        query_results=query_results,
        fixture_metadata={"source_kind": "external_video"} if config.video_path is not None else None,
    )
    observations_path = artifact_path(config, "observations.json")
    write_json(observations_path, observations)
    print_observations(observations)
    print(f"Wrote observations: {observations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
