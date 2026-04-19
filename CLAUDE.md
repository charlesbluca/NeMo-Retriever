# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Focus

The active library is **`nemo_retriever/`**. The top-level `src/`, `api/`, and `client/` directories are deprecated legacy packages that will be removed; avoid adding features or fixes there.

## Development Setup

Install in editable mode (from `nemo_retriever/`):

```bash
cd nemo_retriever
uv pip install -e ".[dev]"
```

Requires Python 3.12+ and CUDA 13.x on Linux. The `pyproject.toml` already pins torch to the CUDA 13 wheel index via `[tool.uv.sources]`; `uv` resolves this automatically.

## Common Commands

```bash
# Run unit tests
PYTHONPATH=nemo_retriever/src python -m pytest nemo_retriever/tests -q

# Run a single test file
PYTHONPATH=nemo_retriever/src python -m pytest nemo_retriever/tests/path/to/test_foo.py -q

# Lint & format checks (pre-commit must be installed first)
pre-commit install
pre-commit run --all-files

# CLI entry point
retriever --help
retriever-harness --help
```

## Architecture

`nemo_retriever/src/nemo_retriever/` is the package root. Key layers:

**Public API** (`api/`, `__init__.py`, `ingestor.py`, `retriever.py`):
- `create_ingestor(run_mode="batch"|"inprocess")` — factory that returns a chainable `Ingestor`
- `Ingestor` pipeline: `.files()` → `.extract()` → `.embed()` → `.vdb_upload()` → `.ingest()`
- `retriever` — singleton `Retriever` for LanceDB querying with embedders/rerankers
- `RunMode` and all `*Params` classes (Pydantic models) live in `params/`

**Run modes** (`application/modes/`):
- `batch` — primary mode, Ray-based large-scale execution
- `inprocess` — local Python process, no framework
- `fused` / `online` — experimental low-latency serving; subject to change

**Processing stages** (`local/stages/`): numbered `stage1`–`stage7` + `stage999`, each handling a discrete step of the GPU pipeline (OCR, table extraction, embedding, etc.)

**Format handlers**: `pdf/`, `html/`, `txt/`, `audio/` — format-specific extraction; `chart/`, `infographic/`, `table/`, `ocr/` — model-driven content extraction

**Inference backends** (`model/`, `nim/`): local GPU execution vs. NVIDIA NIM endpoint calls

**Storage** (`vector_store/`): LanceDB operations; `graph/` for Neo4j knowledge graphs; `tabular_data/` for DuckDB

**Benchmarking** (`harness/`, `recall/`): `retriever harness run --dataset jp20 --preset single_gpu`; configs in `nemo_retriever/harness/test_configs.yaml`

## claude/config branch (temporary — remove before merging to main)

`claude/config` is a persistent meta-branch that carries this file and `.claude/` on top of any dev branch. All Claude config work happens here.

**Keep it rebased on main as main advances:**
```bash
git rebase main claude/config && git push --force-with-lease origin claude/config
```

**Start new feature work with config included:**
```bash
git checkout -b feature/my-work claude/config
```

**Apply config onto an existing feature branch (always conflict-free):**
```bash
git merge claude/config
```

When the config is ready to land, open a standalone PR from `claude/config` into `main`, strip this section, and delete the branch.

## Skill Opportunities

When working in this repo, proactively watch for patterns that should become slash commands:
- Any workflow invoked more than once in a session (build steps, data prep, config generation)
- Multi-step sequences where order matters and is easy to get wrong
- Commands with non-obvious flags or env vars that always need to be set

When you spot one, say so and offer to run `/new-skill` to capture it. Don't wait to be asked.

When bootstrapping feels complete, run `/compact-config` to prune thin or redundant commands and tighten this file.

## Code Style

- Line length: 120 characters (black + flake8)
- Formatting enforced by black via pre-commit; imports by isort
- DCO sign-off required on all commits: `git commit --signoff`
