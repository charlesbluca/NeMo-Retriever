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

Run these checks with:
```bash
python3 - <<'EOF'
import yaml, os
from pathlib import Path

repo = Path("nemo_retriever")
cfg_path = repo / "harness" / "test_configs.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

for name, ds in cfg.get("datasets", {}).items():
    p = Path(ds.get("path", "")).expanduser()
    exists = p.exists()
    q = ds.get("query_csv")
    q_ok = True
    if q:
        qp = Path(q).expanduser()
        q_ok = qp.exists() or (cfg_path.parent / q).exists() or (Path(".") / q).exists()
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

**Dataset path mapping**: `~/datasets/<name>` is the correct `path` for datasets named after their directory. The harness scans recursively (`<path>/**/*.pdf`), so point to the parent dir, not `corpus/`.

**Query CSV mapping**: If a dataset has a `query.csv` at `~/datasets/<name>/query.csv`, that should be the `query_csv` value. The repo's `data/` directory contains only `bo767_annotations.csv` and `digital_corpora_10k_annotations.csv`; all other query CSVs must come from `~/datasets/`.

For the vidore_v3 datasets, the path should be `~/datasets/vidore_v3/<subdataset_name>` (e.g. `~/datasets/vidore_v3/computer_science`). Check that these subdirs exist before proposing them.

Present the proposed changes as a diff and ask for confirmation before editing the file.

## Per-machine path friction

Dataset paths in `test_configs.yaml` are absolute paths set by whoever last edited the file (e.g. `/raid/cjarrett/...`). They will fail on a different machine. **Always run Step 1 before any harness run or sweep** — including dry-runs and new sweep configs — so missing paths surface before starting a long ingestion job.

When adding new dataset entries or sweep configs (e.g. `vllm_bo767_sweep.yaml`), invoke this skill rather than calling `retriever harness sweep --dry-run` directly; the dry-run only validates config structure, not dataset path existence.

## Step 3 — run the harness

Default (active dataset from config, single_gpu preset):
```bash
retriever harness run
```

With explicit dataset and preset:
```bash
retriever harness run --dataset <name> --preset <single_gpu|dgx_8gpu>
```

With key=value overrides (bypasses YAML, no file edit needed):
```bash
retriever harness run --dataset <name> -- embed_workers=4 embed_batch_size=128
```

If the user provides `$ARGUMENTS`, append them. Results land in `results.json` and `session_summary.json` in the working directory.

## Available datasets (from test_configs.yaml)

| key | evaluation_mode | input_type |
|-----|----------------|------------|
| jp20 | recall | pdf |
| bo767 | beir | pdf |
| bo767_baseline | beir | pdf |
| bo767_vl_text | beir | pdf |
| bo767_vl_text_image | beir | pdf |
| bo767_vl_text_image_reranked | beir | pdf |
| bo10k | beir | pdf |
| earnings | recall | pdf |
| financebench | recall | pdf |
| audio_retrieval | recall | audio |
| vidore_v3_* (7 variants) | beir | pdf (page-as-image) |

## Reranker configuration

Three harness fields control reranking (can be set in dataset YAML, sweep overrides, or `--override`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
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
