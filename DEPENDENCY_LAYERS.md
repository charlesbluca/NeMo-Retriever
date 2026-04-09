# Dependency Layering Plan

This document describes the optional-extras model for `nemo_retriever/pyproject.toml`.

## Structure

The base install is the core install: everything needed to run the full pipeline via remote NIM
endpoints, including document parsing tools, the default LanceDB VDB backend, and NIM client libs.
The `[local]` extra adds GPU model inference on top of that.

### Extra hierarchy

```
nemo_retriever          ← core install: ray, fastapi, pydantic, HTTP clients, nv-ingest*,
                          pypdfium2, pillow, nltk, markitdown, langchain-nvidia-ai-endpoints,
                          lancedb
 └── [local]            ← adds: torch (CUDA on Linux), transformers, nemotron models,
                          nvidia-ml-py, vLLM  (Linux/CUDA recommended)

[multimedia]   ← soundfile + scipy (ASR), cairosvg (SVG)  (independent, add to any tier)
[stores]       ← duckdb, duckdb-engine, neo4j  (independent; lancedb is in core)
[benchmarks]   ← datasets, open-clip-torch  (BEIR evaluation only)
[dev]          ← build, pytest
[all]          ← local + multimedia + stores + benchmarks
```

### Install commands by use case

| Use case | Platform | Command |
|---|---|---|
| All remote (NIM) inference | Any | `uv pip install "nemo_retriever"` |
| Local PDF ingestion, GPU | Linux + CUDA | `uv pip install "nemo_retriever[local]"` |
| Full multimedia (GPU + audio + SVG) | Linux + CUDA | `uv pip install "nemo_retriever[local,multimedia]"` |
| Everything | Linux + CUDA | `uv pip install "nemo_retriever[all]"` |

## What Each Extra Contains

### Base (always installed)
Framework infrastructure, document parsing, NIM clients, and the default VDB backend.

- `ray[data,serve]` — pipeline orchestration
- `pandas`, `numpy`, `tqdm` — data handling
- `fastapi`, `uvicorn`, `python-multipart` — service API
- `httpx`, `requests`, `urllib3` — HTTP clients
- `pydantic`, `typer`, `pyyaml`, `rich` — config, CLI, output
- `universal-pathlib`, `debugpy` — utilities
- `nv-ingest-api`, `nv-ingest-client` — core ingest packages
- `pypdfium2` — PDF page splitting and rendering
- `pillow` — image I/O
- `nltk` — text splitting utilities
- `markitdown` — HTML/document-to-markdown conversion
- `langchain-nvidia-ai-endpoints` — LLM/SQL via NVIDIA NIM
- `lancedb` — default vector database for embedding storage and hybrid search

### `[local]`
Adds local HuggingFace model inference, GPU monitoring, and fast LLM inference. On Linux,
torch resolves to a CUDA wheel from the PyTorch index; on Mac it falls through to the PyPI
CPU wheel.

- `transformers`, `tokenizers`, `accelerate==1.12.0` — HuggingFace model loading
- `torch~=2.9.1`, `torchvision` — PyTorch (CUDA on Linux, CPU on Mac)
- `einops`, `easydict`, `addict`, `timm`, `albumentations`, `scikit-learn` — model utilities
- `nemotron-page-elements-v3`, `nemotron-graphic-elements-v1`, `nemotron-table-structure-v1` — layout/table/chart detection
- `nemotron-ocr` — end-to-end OCR (Linux only)
- `nvidia-ml-py` — GPU memory and utilization monitoring
- `vllm==0.16.0` — fast GPU-accelerated LLM inference (Linux only)

### `[multimedia]`
Specialized media format support. Can be combined with any inference tier.

- `soundfile`, `scipy` — audio file I/O and resampling for local Parakeet ASR
- `cairosvg` — SVG-to-image rendering (requires `libcairo` system library)

### `[stores]`
Additional SQL and graph storage backends. LanceDB is already in the base install.

- `duckdb`, `duckdb-engine` — SQL execution on structured/tabular data
- `neo4j` — graph database for knowledge graph ingestion

### `[benchmarks]`
BEIR evaluation tools. Not needed for production use.

- `datasets` — HuggingFace datasets (used in `recall/beir.py`)
- `open-clip-torch` — OpenAI CLIP implementation

## Torch Index Configuration

`[tool.uv.sources]` uses a platform marker so the right torch wheel is resolved automatically:

```toml
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
  # Mac: falls through to PyPI CPU wheel
]
```

No manual intervention needed — `uv` picks the right wheel per platform.
