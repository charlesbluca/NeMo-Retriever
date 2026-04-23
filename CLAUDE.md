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
# Run unit tests (from nemo_retriever/ subdir; uv provides the project's Python env)
cd nemo_retriever && PYTHONPATH=src uv run python -m pytest tests -q

# Run a single test file
cd nemo_retriever && PYTHONPATH=src uv run python -m pytest tests/path/to/test_foo.py -q

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

**Before any config work, pull the latest:**
```bash
git fetch origin && git merge origin/claude/config
```

**Keep it current as main advances:**
```bash
git merge origin/main && git push origin claude/config
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

## Known Friction Points

Things that have caused repeated pain and should eventually be addressed:

- **Harness dataset paths are machine-specific** — `test_configs.yaml` embeds absolute paths (e.g. `/raid/cjarrett/...`) that differ per machine. Every new machine requires editing the committed file, which creates noise in git history and risks accidentally committing a local path. The desired fix is a local-override mechanism (e.g. a gitignored `test_configs.local.yaml` or env-var path substitution) so per-machine paths never touch the committed config.

- **Ray workers can't resolve editable path dependencies outside working_dir** — `ray.init()` is called without `runtime_env`, so Ray auto-packages `nemo_retriever/` as the working dir and ships it to workers. Workers then try to `uv sync`, which includes `nv-ingest @ editable+../src`, `nv-ingest-api @ editable+../api`, `nv-ingest-client @ editable+../client`. Those paths don't exist in Ray's temp dir → workers crash-loop silently. Symptom: `(raylet) error: Failed to generate package metadata for nv-ingest @ editable+../src`. Fix candidates: pass `runtime_env` to `ray.init()` to disable working-dir packaging, or remove the legacy editable deps from `nemo_retriever/pyproject.toml` since `src/`/`api/`/`client/` are deprecated.

- **Ray batch mode requires `dangerouslyDisableSandbox: true`** — the sandbox sets `TMPDIR=/tmp/claude-40703`; Ray's compiled Plasma Object Store binary (`plasma_store_server`) fails to create its Unix socket in that path (SIGABRT in `plasma::PlasmaStore`), even though Python can bind AF_UNIX sockets there fine. Running unsandboxed restores `TMPDIR=/tmp` where Plasma works. `RAY_INCLUDE_DASHBOARD=0` suppresses the dashboard crash noise but is not sufficient on its own. Both are needed for harness runs inside Claude Code.

## Code Style

- Line length: 120 characters (black + flake8)
- Formatting enforced by black via pre-commit; imports by isort
- DCO sign-off required on all commits: `git commit --signoff`
