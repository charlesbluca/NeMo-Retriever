Run the nemo_retriever benchmarking harness. Before running, validate and (if needed) configure dataset paths.

TRIGGER when: user asks to run the harness, run a benchmark, run a sweep, test recall or BEIR metrics, or validate dataset paths. Also trigger when adding or modifying harness configs, sweep YAMLs, or dataset entries in test_configs.yaml — always run Step 1 path validation first, even for dry-runs.
SKIP: unit tests (use run-tests skill instead).

## Step 0 — ensure venv, retriever binary, and CUDA 13 compat

### 0a — venv and retriever binary

Check whether the project venv exists and contains the `retriever` entry point:

```bash
RETRIEVER_BIN=nemo_retriever/.venv/bin/retriever
if [ -f "$RETRIEVER_BIN" ]; then
  echo "venv OK: $RETRIEVER_BIN"
else
  echo "venv missing — creating venv and installing via uv (CLAUDE.md dev setup)"
  cd nemo_retriever && uv venv --python 3.12 && uv pip install -e ".[all,dev]" && cd ..
fi
RETRIEVER=nemo_retriever/.venv/bin/retriever
```

If the install fails (e.g. CUDA wheel mismatch for the installed driver), report the error and stop — do not proceed on a broken environment.

### 0a.5 — HF token (required for local inference)

Any run that loads a local model (vLLM embedder, VL reranker, HF embedder) will trigger a `model_info()` call to the HuggingFace API at actor init time — an internal `is_base_mistral` check inside `transformers`/vLLM. This call is **not** suppressed by `HF_HUB_OFFLINE=1`. Without a token, anonymous HF API requests hit rate limits (HTTP 429) and the Ray actor crashes before any inference runs.

Check that `HF_TOKEN` is set before any harness run:

```bash
echo "HF_TOKEN: ${HF_TOKEN:+set (${HF_TOKEN:0:4}...${HF_TOKEN: -4})}"
```

If not set, ask the user to set it. The token is automatically forwarded to all Ray workers by the existing `ray_env_vars` forwarding in `executor.py`, `run.py`, `runner.py`, and `graph_ingestor.py` — no code changes needed, just a shell export.

**Model hub cache pre-warming**: on a fresh machine, models must be downloaded before `HF_HUB_OFFLINE=1` can work. Use `snapshot_download` with `HF_HUB_OFFLINE=0` once per model:

```python
HF_HUB_OFFLINE=0 nemo_retriever/.venv/bin/python -c "
from huggingface_hub import snapshot_download
from nemo_retriever.utils.hf_model_registry import get_hf_revision
model = '<model-id>'
path = snapshot_download(model, revision=get_hf_revision(model, strict=False), local_files_only=False)
print('cached at:', path)
"
```

### 0b — CUDA 13 compat libs (cu130 wheels on pre-13 drivers)

The venv installs `torch+cu130` and `vllm+cu130`. On machines where the driver supports
CUDA < 13 (e.g. driver 570 = CUDA 12.8), these wheels need `libcudart.so.13` and a
forward-compat `libcuda.so.1` on `LD_LIBRARY_PATH`.

Check whether CUDA is already visible, then discover the compat libs if needed:

```bash
CUDA_OK=$(nemo_retriever/.venv/bin/python -c \
  "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo "0")

if [ "$CUDA_OK" = "1" ]; then
  echo "CUDA already available — no LD_LIBRARY_PATH override needed"
else
  # 1. Check canonical bootstrap path (~/.pixi-cuda13 from lab-foundations aselab bootstrap)
  PIXI_CUDA13="$HOME/.pixi-cuda13/.pixi/envs/default"
  if [ -f "$PIXI_CUDA13/lib/libcudart.so.13" ] && [ -f "$PIXI_CUDA13/cuda-compat/libcuda.so.1" ]; then
    echo "Found CUDA 13 compat at bootstrap path: $PIXI_CUDA13"
    CUDA13_LIB="$PIXI_CUDA13"
  else
    # 2. Search userspace for libcudart.so.13 (other pixi/conda installs)
    CUDART=$(find "$HOME" -maxdepth 6 -name "libcudart.so.13" 2>/dev/null | head -1)
    if [ -n "$CUDART" ]; then
      CUDA13_LIB=$(dirname "$CUDART")
      # Look for cuda-compat alongside (sibling cuda-compat/ dir or same dir)
      COMPAT_DIR=$(dirname "$CUDA13_LIB")/cuda-compat
      [ -d "$COMPAT_DIR" ] || COMPAT_DIR="$CUDA13_LIB"
      echo "Found libcudart.so.13 via search: $CUDA13_LIB (compat: $COMPAT_DIR)"
    else
      echo "WARNING: torch CUDA unavailable and no CUDA 13 compat libs found in \$HOME"
      echo "Run the lab-foundations bootstrap (aselab profile) to install them, or set LD_LIBRARY_PATH manually."
      CUDA13_LIB=""
    fi
  fi

  if [ -n "$CUDA13_LIB" ]; then
    COMPAT_DIR="${COMPAT_DIR:-$(dirname "$CUDA13_LIB")/cuda-compat}"
    export LD_LIBRARY_PATH="$COMPAT_DIR:$CUDA13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "LD_LIBRARY_PATH set; verifying..."
    nemo_retriever/.venv/bin/python -c \
      "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'still unavailable')"
  fi
fi
```

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
| `CUDA out of memory` | Embedder OOM — four possible causes: (1) prior vLLM run left GPU memory pinned; (2) `embed_workers × gpu_memory_utilization` exceeds one GPU — total Ray GPU fraction > 1.0 spills actors to a second GPU; (3) HF ingest path with large `embed_batch_size` (256 materialises full batch×seq activations — use ≤ 32 for HF); (4) VL image modality with `embed_workers > 1` — each image batch needs ~15 GiB activation; 3 concurrent actors exhaust an 80 GiB GPU — VL text_image dataset configs should set `embed_workers: 1` | Kill, run `nvidia-smi`, check cause from table below, then retry |
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

Each Bash call is a separate shell — `LD_LIBRARY_PATH` set in Step 0b does not carry over.
Resolve the CUDA 13 compat prefix inline at run time:

```bash
RETRIEVER=nemo_retriever/.venv/bin/retriever

# Resolve CUDA 13 compat libs (same logic as Step 0b, self-contained for this shell)
_cuda13_ldpath() {
  local p="$HOME/.pixi-cuda13/.pixi/envs/default"
  if [ -f "$p/lib/libcudart.so.13" ] && [ -f "$p/cuda-compat/libcuda.so.1" ]; then
    echo "$p/cuda-compat:$p/lib"
    return
  fi
  local cudart
  cudart=$(find "$HOME" -maxdepth 6 -name "libcudart.so.13" 2>/dev/null | head -1)
  if [ -n "$cudart" ]; then
    local lib; lib=$(dirname "$cudart")
    local compat; compat=$(dirname "$lib")/cuda-compat
    [ -d "$compat" ] && echo "$compat:$lib" || echo "$lib"
  fi
}
CUDA13_PREFIX=$(_cuda13_ldpath)
[ -n "$CUDA13_PREFIX" ] && export LD_LIBRARY_PATH="$CUDA13_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

**Claude Code sandbox**: the sandbox sets `TMPDIR=/tmp/claude-XXXXX`. Ray's compiled Plasma Object Store binary (`plasma_store_server`) SIGABRTs trying to create its Unix socket there — even though Python-level socket tests pass. Running unsandboxed restores `TMPDIR=/tmp` where Plasma works.

**Always run harness Bash commands with `dangerouslyDisableSandbox: true`.**

```bash
# Single dataset run
$RETRIEVER harness run --dataset <name> --preset <single_gpu|dgx_8gpu> 2>&1 | tee harness.log

# Sweep
$RETRIEVER harness sweep --runs-config nemo_retriever/harness/vllm_bo767_sweep.yaml 2>&1 | tee harness.log
```

Pass `run_in_background=true` so the conversation stays unblocked. With key=value overrides (use `--override`, not `-- key=value` which is sweep syntax and does not work for `harness run`):
```bash
$RETRIEVER harness run --dataset <name> --override embed_workers=1 --override embed_batch_size=16 2>&1 | tee harness.log
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
| Embed workers | `--embed-actors N` | 1–3 | >1 when embed backend is vLLM — may spill across GPUs and OOM; must be 1 for VL text_image — each 8-image batch needs ~15 GiB activation |
| Embed batch size | `--embed-batch-size N` | ≤32 for HF, 256 for vLLM | >32 with `--embed-local-ingest-backend hf` → huge activation OOM |
| Ingest backend | `--embed-local-ingest-backend` | hf or vllm | absent = vllm default |
| GPU per embed actor | `--embed-gpus-per-actor F` | 0.25 typical | if `N actors × F > 1.0` and also page_elements + OCR also have GPU fractions, total > 1 GPU → multi-GPU spill |

**Total GPU fraction check**: `(embed_workers × gpu_embed) + (page_elements_workers × gpu_page_elements) + (ocr_workers × gpu_ocr)` must be ≤ 1.0 to stay on a single GPU. For `single_gpu` preset defaults (3×0.25 + 3×0.1 + 3×0.1) = 1.35 — exceeds 1.0. Use `embed_workers: 1` in the sweep `overrides:` for vLLM runs.

If anything looks wrong, kill before the run gets far:
```bash
pkill -f graph_pipeline; ray stop --force 2>/dev/null || true
```

### 3b — monitor strategy (short vs. long runs)

**Short runs (≤ ~10 min, e.g. jp20, single-dataset recall):** skip Monitor entirely.
Just await the Bash completion notification, then read results:
```bash
tail -60 /raid/charlesb/dev/NeMo-Retriever/harness.log | sed 's/\x1b\[[0-9;]*m//g'
```

**1 shell : 1 monitor.** Stop the monitor as soon as the run ends — either by calling `TaskStop` with the monitor's task ID, or by noting the monitor has a finite timeout. Never leave stale monitors accumulating across retries; start a fresh one for each new run attempt.

**Before context compaction:** stop all active monitors with `TaskStop` before the conversation compacts. Monitor IDs are not preserved across compaction and orphaned monitors cannot be recovered by name — the user must kill them manually from the UI. When wrapping up a session or handing off work, explicitly stop every monitor you started.

**Long runs (sweeps, bo767, bo10k, vidore):** use a polling loop — `tail -F | grep` is
unreliable because: (1) grep exits 1 on no matches when the pipe closes, which Monitor
treats as failure; (2) Ray's progress bars are ANSI-encoded and swallow plain grep patterns.
Monitor also kills scripts that produce no stdout for an extended period.

Use this loop pattern instead:
```bash
LOG=/raid/charlesb/dev/NeMo-Retriever/harness.log
last=0
while true; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [ "$cur" -gt "$last" ]; then
    sed -n "$((last+1)),${cur}p" "$LOG" 2>/dev/null \
      | sed 's/\x1b\[[0-9;]*m//g' \
      | grep --line-buffered -iE \
        "Failed to generate package metadata|not registered within|No files found for input_type|Distribution not found at.*ray|VIRTUAL_ENV.*does not match|CUDA out of memory|EngineCore failed|InductorError|SIGABRT|recall@|ndcg@|session complete|Traceback" \
      || true
    last=$cur
  fi
  sleep 5
done
```

Keep Monitor `persistent: true` and `timeout_ms: 3600000`. Even this loop may exit early
on very quiet runs — if it does, check completion via the Bash background notification.

**If Monitor fires** (red flag line appears): kill the run immediately — don't wait for it to finish:
```bash
pkill -f graph_pipeline; ray stop --force 2>/dev/null || true
```
Stop the monitor with TaskStop, then diagnose using the red-flag table above and tell the user what went wrong.

**If the background run completes normally** (Bash notifies you it's done): stop the monitor with TaskStop, read the tail of the log and report the key metrics:
```bash
tail -60 /raid/charlesb/dev/NeMo-Retriever/harness.log | sed 's/\x1b\[[0-9;]*m//g'
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
| `embed_workers` | int | 3 (preset) | Number of concurrent embed actors. VL text_image configs must set `embed_workers: 1` — each image batch needs ~15 GiB activation; 3 actors OOM an 80 GiB GPU. Dataset config wins over preset (low→high: active < preset < dataset < sweep < CLI). |
| `reranker` | bool | false | Enable post-retrieval reranking |
| `reranker_model_name` | str | nvidia/llama-nemotron-rerank-vl-1b-v2 | HF model ID |
| `rerank_modality` | str or null | null | text, text_image, or null (inherits embed_modality) |

```bash
# Enable reranking via CLI override
retriever harness run --dataset bo767_vl_text_image --override reranker=true
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
