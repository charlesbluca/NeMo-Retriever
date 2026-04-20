Run the nemo_retriever benchmarking harness. Before running, validate and (if needed) configure dataset paths.

TRIGGER when: user asks to run the harness, run a benchmark, run a sweep, test recall or BEIR metrics, or validate dataset paths. Also trigger when adding or modifying harness configs, sweep YAMLs, or dataset entries in test_configs.yaml — always run Step 1 path validation first, even for dry-runs.
SKIP: unit tests (use run-tests skill instead).

## Step 1 — validate dataset configuration

Read `nemo_retriever/harness/test_configs.yaml`. For each entry in `datasets:`, check whether:
- `path` resolves to an existing directory (expand `~`)
- `query_csv` resolves to an existing file, checking in this order:
  1. As an absolute path (expand `~`)
  2. Relative to `nemo_retriever/harness/` (the config's own directory)
  3. Relative to repo root

Run these checks with (works regardless of current working directory):
```bash
python3 - <<'EOF'
import yaml, subprocess
from pathlib import Path

repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
cfg_path = repo_root / "nemo_retriever" / "harness" / "test_configs.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

for name, ds in cfg.get("datasets", {}).items():
    p = Path(ds.get("path", "")).expanduser()
    exists = p.exists()
    q = ds.get("query_csv")
    q_ok = True
    if q:
        qp = Path(q).expanduser()
        q_ok = qp.exists() or (cfg_path.parent / q).exists() or (repo_root / q).exists()
    status = "OK" if exists else "MISSING"
    q_status = "" if not q else (" query:OK" if q_ok else " query:MISSING")
    print(f"  {name}: {status}{q_status}  ({p})")
EOF
```

Also scan `~/datasets/` to show what's available locally:
```bash
ls ~/datasets/
```

## Step 2 — configure missing paths (if any)

If any `path` entries are MISSING, propose edits to `nemo_retriever/harness/test_configs.yaml`.

**Dataset path mapping**: `~/datasets/<name>` is the correct `path` for datasets named after their directory. The harness scans recursively (`<path>/**/*.pdf`) in both `graph_pipeline._resolve_file_patterns` and `resolve_input_files`, so point to the dataset root, not an internal subdir like `corpus/`.

**Query CSV mapping**: If a dataset has a `query.csv` at `~/datasets/<name>/query.csv`, that should be the `query_csv` value. The repo's `data/` directory contains only `bo767_annotations.csv` and `digital_corpora_10k_annotations.csv`; all other query CSVs must come from `~/datasets/`.

For the vidore_v3 datasets, the path should be `~/datasets/vidore_v3/<subdataset_name>` (e.g. `~/datasets/vidore_v3/computer_science`). Check that these subdirs exist before proposing them.

Present the proposed changes as a diff and ask for confirmation before editing the file.

## Monitoring runs for known red flags

Harness runs (especially sweeps) are long-running and fail silently. When a run is in progress, periodically scan the output for these patterns — if any appear, the run is doomed and should be killed before wasting more time:

| Red flag | Meaning | Action |
|----------|---------|--------|
| `error: Failed to generate package metadata for nv-ingest @ editable+../src` | Ray workers can't resolve editable deps outside working_dir; workers will crash-loop | Kill, see Known Friction in CLAUDE.md |
| `Some workers of the worker process have not registered within the timeout` | Ray workers are crash-looping (often caused by above) | Kill |
| `No files found for input_type='pdf'` | Dataset path exists but has no PDFs at top level (check for `corpus/` subdir) | Kill, fix path |
| `Distribution not found at: file:///tmp/ray/` | Same as editable dep issue above | Kill |
| `VIRTUAL_ENV=... does not match the project environment path` | Ray raylet ignoring active venv, creating a new broken one | Kill |
| `Module MetricsHead failed to start` + `PermissionError.*AF_UNIX` | Sandbox blocks Unix sockets; Ray dashboard + Plasma store both crash | Kill, use `dangerouslyDisableSandbox: true` for the Bash run command |
| `CUDA out of memory` | Embedder OOM — three possible causes: (1) prior vLLM run left GPU memory pinned; (2) `embed_workers × gpu_memory_utilization` exceeds one GPU — total Ray GPU fraction > 1.0 spills actors to a second GPU; (3) HF ingest path with large `embed_batch_size` (256 materialises full batch×seq activations — use ≤ 32 for HF) | Kill, run `nvidia-smi`, check cause from table below, then retry |
| `EngineCore failed to start` / `InductorError: CalledProcessError` | vLLM engine init failed (torch inductor / GCC compilation) **at actor initialization, before any rows are processed**; `bo767_baseline` requires the vLLM PR branch (`retriever-vllm-for-embeddings-1`) | Kill; skip `bo767_baseline` or switch to vLLM branch |

**vLLM GPU memory poisoning:** When a vLLM actor crashes at initialization, CUDA does not immediately reclaim its GPU allocation (model weights + KV cache reservation). Ray marks the actor dead but the CUDA context persists until the process is fully reaped. Subsequent runs on the same node inherit a partially occupied GPU and hit OOM during embedding. Always schedule vLLM experiments **last** in a sweep and verify `nvidia-smi` shows 0 MiB used before starting the next run if a vLLM run failed.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
```

**Killing a hanging Ray cluster cleanly:**
```bash
# Kill the graph_pipeline subprocess and Ray
pkill -f graph_pipeline
ray stop --force 2>/dev/null || true
# Verify nothing left
ray status 2>/dev/null || echo "Ray stopped"
```

If `ray stop` hangs, find and kill the GCS server directly:
```bash
pkill -f "gcs_server" && pkill -f "raylet"
```

## Per-machine path friction

Dataset paths in `test_configs.yaml` are absolute paths set by whoever last edited the file (e.g. `/raid/cjarrett/...`). They will fail on a different machine. **Always run Step 1 before any harness run or sweep** — including dry-runs and new sweep configs — so missing paths surface before starting a long ingestion job.

When adding new dataset entries or sweep configs (e.g. `vllm_bo767_sweep.yaml`), invoke this skill rather than calling `retriever harness sweep --dry-run` directly; the dry-run only validates config structure, not dataset path existence.

## Step 3 — run the harness (with live red-flag monitoring)

Harness runs are long-running and fail silently. Always start the run in background, then immediately
use the Monitor tool to watch for red flags in real time so a doomed run can be killed early.

### 3a — start the run in background

Tee all output to `harness.log` in the repo root (CWD is always writable; results land there too):

The `retriever` binary lives in the project venv. Always invoke via its full path or activate first:
```bash
RETRIEVER=nemo_retriever/.venv/bin/retriever
```

**Claude Code sandbox**: the sandbox sets `TMPDIR=/tmp/claude-XXXXX`. Ray's compiled Plasma Object Store binary (`plasma_store_server`) SIGABRTs trying to create its Unix socket there — even though Python-level socket tests pass. Running unsandboxed restores `TMPDIR=/tmp` where Plasma works.

**Always run harness Bash commands with `dangerouslyDisableSandbox: true`.**

```bash
# Single dataset run
$RETRIEVER harness run --dataset <name> --preset <single_gpu|dgx_8gpu> 2>&1 | tee harness.log

# Sweep
$RETRIEVER harness sweep --runs-config nemo_retriever/harness/vllm_bo767_sweep.yaml 2>&1 | tee harness.log
```

Pass `run_in_background=true` so the conversation stays unblocked. With key=value overrides:
```bash
$RETRIEVER harness run --dataset <name> -- embed_workers=4 embed_batch_size=128 2>&1 | tee harness.log
```

If the user provides `$ARGUMENTS`, append them before the `2>&1 | tee` redirect.

### 3a.5 — sanity-check key params from the logged command

After the run starts, the harness logs the full `graph_pipeline` command. Read it and verify the params look sensible before committing to a long wait:

```bash
grep -m1 "graph_pipeline" harness.log
```

Key things to check:

| Param | Flag | Healthy value | Red flag |
|-------|------|---------------|----------|
| Embed workers | `--embed-actors N` | 1–3 | >1 when embed backend is vLLM — may spill across GPUs and OOM |
| Embed batch size | `--embed-batch-size N` | ≤32 for HF, 256 for vLLM | >32 with `--embed-local-ingest-backend hf` → huge activation OOM |
| Ingest backend | `--embed-local-ingest-backend` | hf or vllm | absent = vllm default |
| GPU per embed actor | `--embed-gpus-per-actor F` | 0.25 typical | if `N actors × F > 1.0` and also page_elements + OCR also have GPU fractions, total > 1 GPU → multi-GPU spill |

**Total GPU fraction check**: `(embed_workers × gpu_embed) + (page_elements_workers × gpu_page_elements) + (ocr_workers × gpu_ocr)` must be ≤ 1.0 to stay on a single GPU. For `single_gpu` preset defaults (3×0.25 + 3×0.1 + 3×0.1) = 1.35 — exceeds 1.0. Use `embed_workers: 1` in the sweep `overrides:` for vLLM runs.

If anything looks wrong, kill before the run gets far:
```bash
pkill -f graph_pipeline; ray stop --force 2>/dev/null || true
```

### 3b — immediately launch Monitor to watch for red flags

After starting the background run, launch Monitor on:
```bash
tail -F harness.log | grep --line-buffered -E \
  "Failed to generate package metadata|workers.*not registered within|No files found for input_type|Distribution not found at.*ray|VIRTUAL_ENV.*does not match|CUDA out of memory|EngineCore failed to start|InductorError"
```

**If Monitor fires** (red flag line appears): kill the run immediately — don't wait for it to finish:
```bash
pkill -f graph_pipeline; ray stop --force 2>/dev/null || true
```
Then diagnose using the red-flag table above and tell the user what went wrong.

**If the background run completes normally** (Bash notifies you it's done): read the tail of the log
and report the key metrics:
```bash
tail -50 harness.log
```

Results also land in `results.json` and `session_summary.json` in the working directory.

## Available datasets (from test_configs.yaml)

| key | evaluation_mode | notes |
|-----|----------------|-------|
| jp20 | recall | pdf |
| bo767 | beir | pdf |
| bo767_text_hf | beir | HF text embedder; pre-#1494 baseline |
| bo767_vl_text | beir | Transformers VL, text-only |
| bo767_vl_text_image | beir | Transformers VL, text+image |
| bo767_vl_text_image_reranked | beir | Transformers VL, text+image + VL reranker |
| bo767_baseline | beir | vLLM text embedder; requires vLLM branch |
| bo10k | beir | pdf |
| earnings | recall | pdf |
| financebench | recall | pdf |
| audio_retrieval | recall | audio |
| vidore_v3_* (7 variants) | beir | pdf (page-as-image) |

## Embed backend and reranker configuration

Harness fields that control embedding and reranking (set in dataset YAML, sweep overrides, or `--override`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_local_ingest_backend` | str | vllm | Ingest-time text embedder for non-VL models: `vllm` or `hf`. HF requires `embed_batch_size` ≤ 32 (default 256 OOMs at seq_len=8192). |
| `embed_local_query_backend` | str | auto | Query-time embed backend: `auto` (vLLM for text, HF for VL), `hf`, or `vllm` |
| `reranker` | bool | false | Enable post-retrieval reranking |
| `reranker_model_name` | str | nvidia/llama-nemotron-rerank-vl-1b-v2 | HF model ID |
| `rerank_modality` | str or null | null | text, text_image, or null (inherits embed_modality) |

```bash
# Enable reranking via CLI override
retriever harness run --dataset bo767_vl_text_image -- reranker=true
```

## vLLM/VL bo767 experiment sweep

Four sequential experiments comparing text vs. VL embedder with varying modalities and reranking:

```bash
retriever harness sweep --runs-config nemo_retriever/harness/vllm_bo767_sweep.yaml
```

Individual runs:
```bash
retriever harness run --dataset bo767_vl_text
retriever harness run --dataset bo767_vl_text_image
retriever harness run --dataset bo767_vl_text_image_reranked
```

Note: `bo767_baseline` requires the vLLM PR branch (`retriever-vllm-for-embeddings-1`).
