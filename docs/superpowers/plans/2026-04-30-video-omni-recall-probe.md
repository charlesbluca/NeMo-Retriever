# Video Omni Recall Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-file experimental scaffold that generates a synthetic MP4, extracts structured video observations with Nano Omni, embeds the rows with both text-only and VL text+image paths, and compares query recall in memory.

**Architecture:** Keep the runnable prototype in `nemo_retriever/examples/video_omni_recall_probe.py` with local helper functions and dataclasses. Add a focused test module that imports the script by path and tests pure helpers without hosted network calls. Hosted calls are exercised only by the final manual verification command.

**Tech Stack:** Python standard library, `argparse`, `subprocess`/system `ffmpeg`, `requests`, `pandas`, `numpy`, `Pillow`, and existing `nemo_retriever.text_embed.main_text_embed` helpers.

---

## File Structure

- Create: `nemo_retriever/examples/video_omni_recall_probe.py`
  - Owns fixture generation, Nano Omni invocation, response parsing, row construction, dual embedding, ranking, artifact writing, and CLI orchestration.
- Create: `nemo_retriever/tests/test_video_omni_recall_probe.py`
  - Imports the example script by path and validates JSON parsing, row construction, cosine ranking, side-by-side comparison, observation summarization, and hosted payload shape with mocked HTTP.
- No production graph files are modified.
- No package exports are added. This remains an experimental script.

## Task 1: Script Skeleton and Pure Helper Tests

**Files:**
- Create: `nemo_retriever/examples/video_omni_recall_probe.py`
- Create: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Create the example script skeleton**

Add `nemo_retriever/examples/video_omni_recall_probe.py` with the constants, dataclasses, CLI parser, and pure helpers below.

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental video -> Nano Omni extraction -> retrieval probe.

This file is intentionally self-contained while we learn the shape of a useful
video extraction pipeline. It does not modify the production GraphIngestor path.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from nemo_retriever.text_embed.main_text_embed import TextEmbeddingConfig
from nemo_retriever.text_embed.main_text_embed import create_text_embeddings_for_df


API_KEY_ENV = "NGC_NV_DEVELOPER_NVCF"
ARTIFACT_DIR = Path(".artifacts/video_omni_probe")
CHAT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_ENDPOINT = "https://integrate.api.nvidia.com/v1"
OMNI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
PROMPT_VERSION = "video-omni-recall-probe-v1"


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
    top_k: int
    force_fixture: bool
    use_cache: bool
    dry_run: bool
    queries: tuple[str, ...]


DEFAULT_SCENES: tuple[SceneSpec, ...] = (
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


DEFAULT_QUERIES: tuple[str, ...] = (
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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--force-fixture", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args(argv)

    queries = tuple(args.query) if args.query else DEFAULT_QUERIES
    return ProbeConfig(
        artifact_dir=args.artifact_dir,
        chat_endpoint=str(args.chat_endpoint),
        embed_endpoint=str(args.embed_endpoint),
        omni_model=str(args.omni_model),
        embed_model=str(args.embed_model),
        top_k=max(1, int(args.top_k)),
        force_fixture=bool(args.force_fixture),
        use_cache=not bool(args.no_cache),
        dry_run=bool(args.dry_run),
        queries=queries,
    )


def safe_excerpt(text: str, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = max(1, limit // 2)
    tail = max(1, limit - head - 20)
    return value[:head] + "\n...[truncated]...\n" + value[-tail:]


def require_api_key(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    token = (env.get(API_KEY_ENV) or "").strip()
    if not token:
        raise RuntimeError(f"Missing {API_KEY_ENV}. Export it before running hosted model calls.")
    return token


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Missing system ffmpeg. Install ffmpeg and ensure it is on PATH.")


def run_command(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(args)
            + "\nstdout:\n"
            + safe_excerpt(result.stdout)
            + "\nstderr:\n"
            + safe_excerpt(result.stderr)
        )


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Nano Omni response did not contain a JSON object.") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Nano Omni response JSON must be an object.")
    return parsed


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    print(f"Video Omni recall probe artifacts: {config.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create tests for pure helpers and CLI defaults**

Add `nemo_retriever/tests/test_video_omni_recall_probe.py` with these tests.

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "video_omni_recall_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("video_omni_recall_probe", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_builtin_queries(tmp_path: Path):
    probe = load_probe_module()

    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--dry-run"])

    assert config.artifact_dir == tmp_path
    assert config.dry_run is True
    assert config.use_cache is True
    assert "ALPHA-17" in config.queries[0]


def test_safe_excerpt_keeps_short_text():
    probe = load_probe_module()

    assert probe.safe_excerpt("small", limit=20) == "small"


def test_safe_excerpt_truncates_long_text():
    probe = load_probe_module()

    excerpt = probe.safe_excerpt("a" * 80 + "b" * 80, limit=60)

    assert "[truncated]" in excerpt
    assert excerpt.startswith("a")
    assert excerpt.endswith("b" * 10)


def test_require_api_key_reports_missing_key():
    probe = load_probe_module()

    with pytest.raises(RuntimeError, match="NGC_NV_DEVELOPER_NVCF"):
        probe.require_api_key({})


def test_extract_json_object_accepts_fenced_json():
    probe = load_probe_module()

    parsed = probe.extract_json_object('```json\n{"segments": []}\n```')

    assert parsed == {"segments": []}


def test_extract_json_object_accepts_surrounded_json():
    probe = load_probe_module()

    parsed = probe.extract_json_object('Here is the answer:\n{"segments": [{"segment_id": "a"}]}')

    assert parsed["segments"][0]["segment_id"] == "a"


def test_cosine_similarity_handles_identical_and_zero_vectors():
    probe = load_probe_module()

    assert np.isclose(probe.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert probe.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert probe.cosine_similarity([1.0], [1.0, 0.0]) == 0.0
```

- [ ] **Step 3: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit skeleton and helper tests**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add video omni probe skeleton"
```

## Task 2: Synthetic Fixture Generation and Dry Run

**Files:**
- Modify: `nemo_retriever/examples/video_omni_recall_probe.py`
- Modify: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Add image/audio/video fixture helpers**

Append these helpers after `run_command()`.

```python
def artifact_path(config: ProbeConfig, name: str) -> Path:
    return config.artifact_dir / name


def scene_specs() -> tuple[SceneSpec, ...]:
    return DEFAULT_SCENES


def total_duration_seconds(scenes: tuple[SceneSpec, ...] = DEFAULT_SCENES) -> float:
    return max(scene.end_seconds for scene in scenes)


def draw_scene_frame(scene: SceneSpec, path: Path, *, size: tuple[int, int] = (960, 540)) -> None:
    image = Image.new("RGB", size, scene.background_rgb)
    draw = ImageDraw.Draw(image)
    font_large = ImageFont.load_default(size=64)
    font_small = ImageFont.load_default(size=28)
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
    check_ffmpeg()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = artifact_path(config, "synthetic_fixture.mp4")
    frame_manifest_path = artifact_path(config, "scene_frames.json")
    if fixture_path.exists() and frame_manifest_path.exists() and not config.force_fixture:
        return fixture_path, json.loads(frame_manifest_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="video_omni_fixture_") as tmp:
        tmp_path = Path(tmp)
        frames: list[Path] = []
        frame_b64: dict[str, str] = {}
        for scene in scene_specs():
            frame_path = tmp_path / f"{scene.segment_id}.png"
            draw_scene_frame(scene, frame_path)
            frames.append(frame_path)
            frame_b64[scene.segment_id] = image_file_to_b64(frame_path)

        video_only = tmp_path / "fixture_video_only.mp4"
        build_video_from_frames(frames, video_only, seconds_per_scene=2.5)
        audio_path = prepare_audio_track(tmp_path, total_duration_seconds())
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

    frame_manifest_path.write_text(json.dumps(frame_b64, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixture_path, frame_b64
```

- [ ] **Step 2: Update `main()` for dry-run fixture generation**

Replace `main()` with this implementation.

```python
def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    fixture_path, frame_b64 = generate_synthetic_fixture(config)
    print(f"Video Omni recall probe artifacts: {config.artifact_dir}")
    print(f"Synthetic fixture: {fixture_path}")
    print(f"Representative frames: {len(frame_b64)}")
    if config.dry_run:
        print("Dry run complete; hosted model and embedding calls were skipped.")
        return 0
    token = require_api_key()
    print(f"Loaded API key from {API_KEY_ENV}: {bool(token)}")
    return 0
```

- [ ] **Step 3: Add fixture helper tests**

Append these tests to `nemo_retriever/tests/test_video_omni_recall_probe.py`.

```python
def test_scene_specs_are_deterministic():
    probe = load_probe_module()

    scenes = probe.scene_specs()

    assert [scene.segment_id for scene in scenes] == [
        "scene-alpha",
        "scene-beta",
        "scene-calibration",
        "scene-audio",
    ]
    assert scenes[0].expected_terms[0] == "alpha-17"
    assert probe.total_duration_seconds(scenes) == 10.0


def test_draw_scene_frame_writes_png(tmp_path: Path):
    probe = load_probe_module()
    scene = probe.scene_specs()[0]
    output = tmp_path / "scene.png"

    probe.draw_scene_frame(scene, output, size=(320, 180))

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")
    assert len(probe.image_file_to_b64(output)) > 100


def test_write_silent_wav_writes_expected_header(tmp_path: Path):
    probe = load_probe_module()
    output = tmp_path / "silence.wav"

    probe.write_silent_wav(output, duration_seconds=0.25, sample_rate=8000)

    assert output.exists()
    assert output.read_bytes().startswith(b"RIFF")
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run the script in dry-run mode**

Run:

```bash
PYTHONPATH=nemo_retriever/src:api/src python nemo_retriever/examples/video_omni_recall_probe.py --dry-run --force-fixture
```

Expected:

- `.artifacts/video_omni_probe/synthetic_fixture.mp4` exists.
- `.artifacts/video_omni_probe/scene_frames.json` exists.
- Console includes `Dry run complete; hosted model and embedding calls were skipped.`

- [ ] **Step 6: Commit fixture generation**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add synthetic video fixture probe"
```

## Task 3: Nano Omni Hosted Extraction Client

**Files:**
- Modify: `nemo_retriever/examples/video_omni_recall_probe.py`
- Modify: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Add prompt and hosted request helpers**

Append these helpers after `generate_synthetic_fixture()`.

```python
def video_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def build_omni_prompt() -> str:
    expected = [
        {
            "segment_id": scene.segment_id,
            "time_range": [scene.start_seconds, scene.end_seconds],
            "known_fixture_cues": list(scene.expected_terms),
        }
        for scene in scene_specs()
    ]
    return (
        "You are extracting retrieval records from a short synthetic video. "
        "Return only valid JSON. Do not wrap it in markdown. "
        "Use exact visible text when it is readable. Do not invent details. "
        "Use this schema: {\"segments\": [{\"segment_id\": string, "
        "\"start_seconds\": number, \"end_seconds\": number, \"summary\": string, "
        "\"visual_text\": [string], \"objects\": [string], \"actions\": [string], "
        "\"audio_or_speech\": string, \"retrieval_keywords\": [string], "
        "\"uncertainties\": [string], \"confidence\": number}]}. "
        "The fixture has these expected segment windows and cues for calibration only: "
        + json.dumps(expected, sort_keys=True)
    )


def build_omni_payload(video_url: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_omni_prompt()},
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


def post_json(url: str, token: str, payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]:
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
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {safe_excerpt(response.text)}")
    data = response.json()
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


def call_nano_omni(fixture_path: Path, config: ProbeConfig, token: str) -> dict[str, Any]:
    payload = build_omni_payload(video_data_url(fixture_path), config.omni_model)
    started = time.perf_counter()
    raw_response = post_json(config.chat_endpoint, token, payload)
    elapsed = time.perf_counter() - started
    content = extract_message_content(raw_response)
    parsed = extract_json_object(content)
    return {
        "elapsed_seconds": elapsed,
        "request": {
            "model": config.omni_model,
            "endpoint": config.chat_endpoint,
            "prompt_version": PROMPT_VERSION,
            "media_io_kwargs": payload["media_io_kwargs"],
            "mm_processor_kwargs": payload["mm_processor_kwargs"],
        },
        "raw_response": raw_response,
        "parsed": parsed,
    }
```

- [ ] **Step 2: Add response schema validation**

Append this helper after `call_nano_omni()`.

```python
def validate_omni_segments(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    segments = parsed.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Nano Omni parsed JSON must contain a non-empty 'segments' list.")
    normalized: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {i} must be an object.")
        segment_id = str(segment.get("segment_id") or f"segment-{i}")
        start = float(segment.get("start_seconds", 0.0))
        end = float(segment.get("end_seconds", start))
        if end < start:
            end = start
        normalized.append(
            {
                "segment_id": segment_id,
                "start_seconds": start,
                "end_seconds": end,
                "summary": str(segment.get("summary") or ""),
                "visual_text": list(segment.get("visual_text") or []),
                "objects": list(segment.get("objects") or []),
                "actions": list(segment.get("actions") or []),
                "audio_or_speech": str(segment.get("audio_or_speech") or ""),
                "retrieval_keywords": list(segment.get("retrieval_keywords") or []),
                "uncertainties": list(segment.get("uncertainties") or []),
                "confidence": float(segment.get("confidence", 0.0) or 0.0),
            }
        )
    return normalized
```

- [ ] **Step 3: Update `main()` to call Omni or use cached response**

Replace the non-dry-run section of `main()` after API key loading with this code.

```python
    token = require_api_key()
    omni_path = artifact_path(config, "omni_response.json")
    if config.use_cache and omni_path.exists():
        omni_result = json.loads(omni_path.read_text(encoding="utf-8"))
        print(f"Loaded cached Omni response: {omni_path}")
    else:
        omni_result = call_nano_omni(fixture_path, config, token)
        omni_path.write_text(json.dumps(omni_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote Omni response: {omni_path}")
    segments = validate_omni_segments(omni_result["parsed"])
    print(f"Validated Omni segments: {len(segments)}")
    return 0
```

- [ ] **Step 4: Add hosted client tests with mocked HTTP**

Append these tests.

```python
def test_build_omni_payload_enables_audio_and_sampling():
    probe = load_probe_module()

    payload = probe.build_omni_payload("data:video/mp4;base64,abc", "model-id")

    assert payload["model"] == "model-id"
    assert payload["mm_processor_kwargs"] == {"use_audio_in_video": True}
    assert payload["media_io_kwargs"]["video"]["num_frames"] == 16
    assert payload["messages"][0]["content"][1]["type"] == "video_url"


def test_extract_message_content_and_validate_segments():
    probe = load_probe_module()
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"segments":[{"segment_id":"scene-alpha","start_seconds":0,"end_seconds":2.5,"summary":"ALPHA-17 is visible","visual_text":["ALPHA-17"],"objects":["red triangle"],"actions":[],"audio_or_speech":"","retrieval_keywords":["warning"],"uncertainties":[],"confidence":0.9}]}'
                }
            }
        ]
    }

    content = probe.extract_message_content(response)
    parsed = probe.extract_json_object(content)
    segments = probe.validate_omni_segments(parsed)

    assert segments[0]["segment_id"] == "scene-alpha"
    assert segments[0]["visual_text"] == ["ALPHA-17"]
    assert segments[0]["confidence"] == 0.9


def test_post_json_reports_http_errors(monkeypatch):
    probe = load_probe_module()

    class FakeResponse:
        status_code = 401
        text = "bad token"

        def json(self):
            return {}

    monkeypatch.setattr(probe.requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="HTTP 401"):
        probe.post_json("https://example.test", "token", {"model": "m"})
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Omni client**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add hosted Nano Omni extraction client"
```

## Task 4: Row Construction and Artifact Writing

**Files:**
- Modify: `nemo_retriever/examples/video_omni_recall_probe.py`
- Modify: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Add retrieval text and row construction helpers**

Append these helpers after `validate_omni_segments()`.

```python
def flatten_str_list(values: list[Any]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


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


def nearest_scene_frame(segment_id: str, frame_b64: dict[str, str]) -> str:
    if segment_id in frame_b64:
        return frame_b64[segment_id]
    if frame_b64:
        return next(iter(frame_b64.values()))
    return ""


def rows_from_segments(
    segments: list[dict[str, Any]],
    frame_b64: dict[str, str],
    fixture_path: Path,
    config: ProbeConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        rows.append(
            {
                "segment_id": segment_id,
                "text": build_retrieval_text(segment),
                "_image_b64": nearest_scene_frame(segment_id, frame_b64),
                "metadata": {
                    "source_path": str(fixture_path),
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 2: Update `main()` to save extracted rows**

After segment validation in `main()`, add this code before `return 0`.

```python
    rows_df = rows_from_segments(segments, frame_b64, fixture_path, config)
    rows_path = artifact_path(config, "extracted_rows.json")
    write_json(rows_path, rows_df.to_dict(orient="records"))
    print(f"Wrote extracted rows: {rows_path}")
```

- [ ] **Step 3: Add row construction tests**

Append these tests.

```python
def sample_segment():
    return {
        "segment_id": "scene-alpha",
        "start_seconds": 0.0,
        "end_seconds": 2.5,
        "summary": "ALPHA-17 warning screen",
        "visual_text": ["ALPHA-17"],
        "objects": ["red triangle", "blue crate"],
        "actions": ["static warning card"],
        "audio_or_speech": "no speech",
        "retrieval_keywords": ["warning", "alpha"],
        "uncertainties": [],
        "confidence": 0.95,
    }


def test_build_retrieval_text_contains_visual_audio_and_keywords():
    probe = load_probe_module()

    text = probe.build_retrieval_text(sample_segment())

    assert "ALPHA-17" in text
    assert "red triangle" in text
    assert "no speech" in text
    assert "warning" in text


def test_rows_from_segments_preserves_metadata(tmp_path: Path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--dry-run"])

    df = probe.rows_from_segments([sample_segment()], {"scene-alpha": "abc123"}, tmp_path / "fixture.mp4", config)

    assert df.loc[0, "segment_id"] == "scene-alpha"
    assert df.loc[0, "_image_b64"] == "abc123"
    assert df.loc[0, "metadata"]["raw_omni_segment"]["visual_text"] == ["ALPHA-17"]
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit row construction**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add video omni retrieval row construction"
```

## Task 5: Dual Embedding and In-Memory Ranking

**Files:**
- Modify: `nemo_retriever/examples/video_omni_recall_probe.py`
- Modify: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Add embedding and ranking helpers**

Append these helpers after `write_json()`.

```python
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
) -> pd.DataFrame:
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
    embeddings = [embedding_from_payload(payload) for payload in out_df[output_column].tolist()]
    if not embeddings or not any(embeddings):
        raise RuntimeError(f"No embeddings produced for {output_column}.")
    return out_df


def embed_query(
    query: str,
    *,
    api_key: str,
    endpoint: str,
    model_name: str,
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
    )
    return embedding_from_payload(out_df.loc[0, "query_embedding"])


def rank_rows(rows_df: pd.DataFrame, query_embedding: list[float], embedding_column: str, top_k: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        embedding = embedding_from_payload(row.get(embedding_column))
        score = cosine_similarity(query_embedding, embedding)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        ranked.append(
            {
                "segment_id": str(row.get("segment_id")),
                "score": score,
                "text": str(row.get("text") or ""),
                "start_seconds": metadata.get("start_seconds"),
                "end_seconds": metadata.get("end_seconds"),
                "raw_omni_segment": metadata.get("raw_omni_segment", {}),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, int(top_k))]


def expected_query_terms(query: str) -> tuple[str, ...]:
    q = query.lower()
    if "alpha" in q or "warning" in q:
        return ("alpha-17", "warning")
    if "calibration" in q:
        return ("calibration",)
    if "audio" in q or "spoken" in q or "speech" in q:
        return ("audio", "speech", "spoken")
    if "colored object" in q:
        return ("red triangle", "green square", "purple circle")
    return ()


def hit_contains_terms(hit: dict[str, Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    haystack = json.dumps(hit, sort_keys=True).lower()
    return any(term.lower() in haystack for term in terms)
```

- [ ] **Step 2: Add query comparison helpers**

Append these helpers after `hit_contains_terms()`.

```python
def compare_query_results(
    queries: tuple[str, ...],
    rows_df: pd.DataFrame,
    *,
    api_key: str,
    endpoint: str,
    model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    text_df = embed_dataframe(
        rows_df,
        api_key=api_key,
        endpoint=endpoint,
        model_name=model_name,
        modality="text",
        output_column="text_embedding",
    )
    vl_df = embed_dataframe(
        rows_df,
        api_key=api_key,
        endpoint=endpoint,
        model_name=model_name,
        modality="text_image",
        output_column="vl_text_image_embedding",
    )

    results: list[dict[str, Any]] = []
    for query in queries:
        query_embedding = embed_query(query, api_key=api_key, endpoint=endpoint, model_name=model_name)
        text_hits = rank_rows(text_df, query_embedding, "text_embedding", top_k)
        vl_hits = rank_rows(vl_df, query_embedding, "vl_text_image_embedding", top_k)
        terms = expected_query_terms(query)
        results.append(
            {
                "query": query,
                "expected_terms": list(terms),
                "text_only": text_hits,
                "vl_text_image": vl_hits,
                "text_only_top_contains_expected": bool(text_hits and hit_contains_terms(text_hits[0], terms)),
                "vl_top_contains_expected": bool(vl_hits and hit_contains_terms(vl_hits[0], terms)),
                "top_result_changed": bool(text_hits and vl_hits and text_hits[0]["segment_id"] != vl_hits[0]["segment_id"]),
            }
        )
    return results
```

- [ ] **Step 3: Update `main()` to run embeddings and save query results**

After writing extracted rows in `main()`, add this code before `return 0`.

```python
    query_results = compare_query_results(
        config.queries,
        rows_df,
        api_key=token,
        endpoint=config.embed_endpoint,
        model_name=config.embed_model,
        top_k=config.top_k,
    )
    query_path = artifact_path(config, "query_results.json")
    write_json(query_path, query_results)
    print_query_results(query_results)
    print(f"Wrote query results: {query_path}")
```

Also add a temporary `print_query_results()` helper just above `main()`. Task 6 expands observations.

```python
def print_query_results(query_results: list[dict[str, Any]]) -> None:
    for result in query_results:
        print("\nQuery:", result["query"])
        for label in ("text_only", "vl_text_image"):
            print(f"  {label}:")
            for rank, hit in enumerate(result[label], start=1):
                start = hit.get("start_seconds")
                end = hit.get("end_seconds")
                print(f"    {rank}. score={hit['score']:.4f} segment={hit['segment_id']} time={start}-{end}")
        print(f"  top_result_changed={result['top_result_changed']}")
```

- [ ] **Step 4: Add ranking tests with monkeypatched embeddings**

Append these tests.

```python
def test_rank_rows_orders_by_cosine_similarity():
    probe = load_probe_module()
    df = probe.pd.DataFrame(
        [
            {"segment_id": "a", "text_embedding": {"embedding": [1.0, 0.0]}, "metadata": {"start_seconds": 0, "end_seconds": 1}},
            {"segment_id": "b", "text_embedding": {"embedding": [0.0, 1.0]}, "metadata": {"start_seconds": 1, "end_seconds": 2}},
        ]
    )

    ranked = probe.rank_rows(df, [1.0, 0.0], "text_embedding", top_k=2)

    assert [hit["segment_id"] for hit in ranked] == ["a", "b"]


def test_compare_query_results_reports_top_result_change(monkeypatch, tmp_path: Path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--dry-run"])
    rows = probe.rows_from_segments(
        [
            {**sample_segment(), "segment_id": "scene-alpha", "summary": "ALPHA-17 warning"},
            {**sample_segment(), "segment_id": "scene-calibration", "summary": "CALIBRATION purple circle", "visual_text": ["CALIBRATION"]},
        ],
        {"scene-alpha": "aaa", "scene-calibration": "bbb"},
        tmp_path / "fixture.mp4",
        config,
    )

    def fake_embed_dataframe(rows_df, *, api_key, endpoint, model_name, modality, output_column, batch_size=8, input_type="passage"):
        out = rows_df.copy()
        if output_column == "query_embedding":
            out[output_column] = [{"embedding": [1.0, 0.0]}]
        elif modality == "text":
            out[output_column] = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        else:
            out[output_column] = [{"embedding": [0.0, 1.0]}, {"embedding": [1.0, 0.0]}]
        return out

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)

    results = probe.compare_query_results(
        ("Which segment shows ALPHA-17?",),
        rows,
        api_key="token",
        endpoint="https://example.test/v1",
        model_name="embed",
        top_k=1,
    )

    assert results[0]["text_only"][0]["segment_id"] == "scene-alpha"
    assert results[0]["vl_text_image"][0]["segment_id"] == "scene-calibration"
    assert results[0]["top_result_changed"] is True
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit embedding and ranking**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add dual embedding comparison for video probe"
```

## Task 6: Observations, Artifact Completeness, and CLI Polish

**Files:**
- Modify: `nemo_retriever/examples/video_omni_recall_probe.py`
- Modify: `nemo_retriever/tests/test_video_omni_recall_probe.py`

- [ ] **Step 1: Add observation summary helper**

Append this helper above `print_query_results()`.

```python
def summarize_observations(
    *,
    omni_elapsed_seconds: float,
    fixture_path: Path,
    query_results: list[dict[str, Any]],
) -> dict[str, Any]:
    fixture_bytes = fixture_path.stat().st_size if fixture_path.exists() else 0
    text_successes = sum(1 for result in query_results if result.get("text_only_top_contains_expected"))
    vl_successes = sum(1 for result in query_results if result.get("vl_top_contains_expected"))
    changed = sum(1 for result in query_results if result.get("top_result_changed"))
    total = len(query_results)
    return {
        "pros": [
            f"Nano Omni returned structured segments for {total_duration_seconds():.1f}s of synthetic video.",
            f"Text-only baseline hit expected terms for {text_successes}/{total} diagnostic queries.",
            f"VL text+image hit expected terms for {vl_successes}/{total} diagnostic queries.",
        ],
        "cons": [
            f"Hosted Omni call latency was {omni_elapsed_seconds:.2f}s for a {fixture_bytes} byte MP4.",
            f"Top result changed between baseline and VL retrieval for {changed}/{total} diagnostic queries.",
            "Segment timing and visual facts are model-generated and should be treated as inspectable evidence, not ground truth.",
        ],
        "metrics": {
            "omni_elapsed_seconds": omni_elapsed_seconds,
            "fixture_bytes": fixture_bytes,
            "text_only_expected_hits": text_successes,
            "vl_expected_hits": vl_successes,
            "top_result_changes": changed,
            "query_count": total,
        },
    }
```

- [ ] **Step 2: Expand console output**

Replace `print_query_results()` with this version.

```python
def print_query_results(query_results: list[dict[str, Any]]) -> None:
    for result in query_results:
        print("\nQuery:", result["query"])
        print("  expected_terms:", ", ".join(result["expected_terms"]) or "none")
        for label in ("text_only", "vl_text_image"):
            print(f"  {label}:")
            for rank, hit in enumerate(result[label], start=1):
                start = hit.get("start_seconds")
                end = hit.get("end_seconds")
                snippet = " ".join(hit["text"].split())[:180]
                print(f"    {rank}. score={hit['score']:.4f} segment={hit['segment_id']} time={start}-{end}")
                print(f"       {snippet}")
        print(f"  text_only_top_contains_expected={result['text_only_top_contains_expected']}")
        print(f"  vl_top_contains_expected={result['vl_top_contains_expected']}")
        print(f"  top_result_changed={result['top_result_changed']}")


def print_observations(observations: dict[str, Any]) -> None:
    print("\nObservations")
    print("  Pros:")
    for item in observations["pros"]:
        print(f"    - {item}")
    print("  Cons:")
    for item in observations["cons"]:
        print(f"    - {item}")
```

- [ ] **Step 3: Save observations in `main()`**

After `print_query_results(query_results)` in `main()`, add:

```python
    observations = summarize_observations(
        omni_elapsed_seconds=float(omni_result.get("elapsed_seconds", 0.0)),
        fixture_path=fixture_path,
        query_results=query_results,
    )
    observations_path = artifact_path(config, "observations.json")
    write_json(observations_path, observations)
    print_observations(observations)
    print(f"Wrote observations: {observations_path}")
```

- [ ] **Step 4: Add observations tests**

Append this test.

```python
def test_summarize_observations_counts_baseline_and_vl_hits(tmp_path: Path):
    probe = load_probe_module()
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"video")

    observations = probe.summarize_observations(
        omni_elapsed_seconds=1.25,
        fixture_path=fixture,
        query_results=[
            {
                "text_only_top_contains_expected": True,
                "vl_top_contains_expected": False,
                "top_result_changed": True,
            },
            {
                "text_only_top_contains_expected": False,
                "vl_top_contains_expected": True,
                "top_result_changed": False,
            },
        ],
    )

    assert observations["metrics"]["text_only_expected_hits"] == 1
    assert observations["metrics"]["vl_expected_hits"] == 1
    assert observations["metrics"]["top_result_changes"] == 1
    assert observations["metrics"]["fixture_bytes"] == 5
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit observations and CLI polish**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add video probe observations summary"
```

## Task 7: End-to-End Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=nemo_retriever/src:api/src pytest nemo_retriever/tests/test_video_omni_recall_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run dry-run fixture generation**

```bash
PYTHONPATH=nemo_retriever/src:api/src python nemo_retriever/examples/video_omni_recall_probe.py --dry-run --force-fixture
```

Expected:

- Console prints the fixture path and representative frame count.
- `.artifacts/video_omni_probe/synthetic_fixture.mp4` exists.
- `.artifacts/video_omni_probe/scene_frames.json` exists.

- [ ] **Step 3: Run hosted end-to-end probe**

Run this only when network access is available and `NGC_NV_DEVELOPER_NVCF` is set.

```bash
PYTHONPATH=nemo_retriever/src:api/src python nemo_retriever/examples/video_omni_recall_probe.py --force-fixture --no-cache
```

Expected:

- `.artifacts/video_omni_probe/omni_response.json` exists.
- `.artifacts/video_omni_probe/extracted_rows.json` exists.
- `.artifacts/video_omni_probe/query_results.json` exists.
- `.artifacts/video_omni_probe/observations.json` exists.
- Console prints both `text_only` and `vl_text_image` rankings for each query.
- At least one built-in query has `text_only_top_contains_expected=true` in `query_results.json`.

- [ ] **Step 4: Run cached end-to-end probe**

```bash
PYTHONPATH=nemo_retriever/src:api/src python nemo_retriever/examples/video_omni_recall_probe.py --query "Which scene has the green square?"
```

Expected:

- Console prints `Loaded cached Omni response`.
- Console prints rankings for the custom query.
- `query_results.json` is updated with the custom query result.

- [ ] **Step 5: Inspect git diff**

```bash
git diff --stat
git diff -- nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
```

Expected:

- Only the example script and test file contain implementation changes.
- Generated `.artifacts/` files are untracked or ignored and are not staged.

- [ ] **Step 6: Final commit**

```bash
git add nemo_retriever/examples/video_omni_recall_probe.py nemo_retriever/tests/test_video_omni_recall_probe.py
git commit -m "Add video omni recall probe"
```

## Self-Review Checklist

- Spec coverage:
  - Synthetic MP4 fixture: Task 2.
  - Hosted Nano Omni extraction: Task 3.
  - Structured retrieval rows: Task 4.
  - Text-only baseline and VL text+image comparison: Task 5.
  - Query loop and saved query artifacts: Tasks 5 and 6.
  - Pros/cons scaffold output: Task 6.
  - Error handling for missing key, missing ffmpeg, HTTP errors, parse errors, and empty embeddings: Tasks 1, 2, 3, and 5.
- Placeholder scan:
  - Every helper referenced by later tasks is defined in an earlier snippet or the same task.
  - Each step includes concrete code or exact verification commands.
- Type consistency:
  - `ProbeConfig`, `SceneSpec`, segment dictionaries, embedding payloads, and query result dictionaries are introduced before use.
  - `text_embedding`, `vl_text_image_embedding`, and `query_embedding` column names are used consistently.
