# NeMo Retriever – Dev branch PR plan

This doc tracks the breakdown of this branch’s changes into coherent PRs for review and the order we want to merge them. Update it as PRs are opened/merged or the plan changes.

**Branch context:** The original goal was to allow audio extraction in `batch_pipeline.py` with configurable ASR actor count and resources. Additional changes came from issues found along the way (LanceDB document-level path parity, long-audio ASR, inprocess refactor, etc.).

**Diff baseline:** Changes below are relative to `upstream/main` (or `origin/main`).

---

## Merge order

1. **PR 1** – ASR refactor, long-audio handling, inprocess loader (foundation)
2. **PR 2** – GPU pool ASR support (depends on PR 1)
3. **PR 3** – LanceDB document-level path (no dependency on PR 2/4/5)
4. **PR 4** – Batch pipeline audio + ASR resource knobs (depends on PR 1 + PR 3)
5. **PR 5** – Inprocess pipeline audio support (depends on PR 1 only; can merge in parallel with 2/3 after 1)
6. **PR 6** (optional) – Compare script + doc (after 1–5)

Suggested sequence: **1 → 2, 3, 5 (any order) → 4**. Or: **1 → 3 → 4 → 2 → 5** to keep LanceDB and batch back-to-back.

---

## PR 1: ASR refactor, long-audio handling, inprocess loader

**Goal:** Reusable ASR batch function, fix long-audio, unify inprocess doc loading.


| File                                                                     | Changes                                                                                                                   |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/audio/asr_actor.py`                   | Extract `asr_chunks_to_text(...)`; ASRActor delegates; remote `segment_audio` fans out punctuation segments via `_build_output_rows`; `_infer_remote` returns `(segments, transcript)`. |
| `nemo_retriever/src/nemo_retriever/audio/__init__.py`                    | Export `asr_chunks_to_text`.                                                                                              |
| `nemo_retriever/src/nemo_retriever/model/local/parakeet_ctc_1_1b_asr.py` | Cap segment duration (`MAX_AUDIO_DURATION_SEC`), split long audio, concatenate segment transcripts.                       |
| `nemo_retriever/src/nemo_retriever/audio/media_interface.py`             | Robust duration/bitrate handling when ffprobe omits `duration` or `bit_rate`.                                             |
| `nemo_retriever/src/nemo_retriever/audio/stage.py`                       | Use `asr_chunks_to_text` for chunk → transcript.                                                                          |
| `nemo_retriever/src/nemo_retriever/ingest_modes/inprocess.py`            | Use `asr_chunks_to_text` with model injection; add `_load_doc_to_df` / `_iter_doc_chunks` and use for all pipeline types. |
| `nemo_retriever/tests/test_asr_actor.py`                                 | Update for `asr_chunks_to_text` / ASRActor.                                                                               |


**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## PR 2: GPU pool ASR support

**Goal:** GPU pool can run and describe ASR tasks.


| File                                                         | Changes                                                                                                           |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/ingest_modes/gpu_pool.py` | Add `ASRModelConfig`; in `_extract_model_config` handle `asr_chunks_to_text` (local vs remote, picklable config). |


**Depends on:** PR 1.

**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## PR 3: LanceDB document-level path (shared utils + batch driver path)

**Goal:** Batch and inprocess both write document-level `path` / `pdf_basename` to LanceDB.


| File                                                              | Changes                                                                                                                                                                                              |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/ingest_modes/lancedb_utils.py` | Add `_row_get`, `_metadata_to_dict`; extend `extract_source_path_and_page` (doc-level + chunk-path fallback); use in `extract_embedding_from_row`, `build_lancedb_row`, `_build_detection_metadata`. |
| `nemo_retriever/src/nemo_retriever/vector_store/lancedb_store.py` | Add `_row_to_dict`, `_build_lancedb_rows_from_batch_results()`; `handle_lancedb(rows: Iterable[Any], ...)` uses it, doc-level path, overwrite, return value.                                         |


**Depends on:** None from this branch.

**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## PR 4: Batch pipeline audio + ASR resource knobs

**Goal:** Original goal — audio in batch_pipeline with configurable ASR actor count and resources.


| File                                                           | Changes                                                                                                                                                                                                                                  |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/ingest_modes/batch.py`      | `_metadata_to_dict`, `_ensure_source_id_batch`; `extract_audio()` accepts `asr_actors`, `asr_cpus_per_actor`, `asr_gpus_per_actor` (and `ActorPoolStrategy` when `asr_actors > 1`); after embed, `map_batches(_ensure_source_id_batch)`. |
| `nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py` | Add `--input-type audio`, CLI for audio/ASR; audio branch in pipeline; skip text chunk for audio.                                                                                                                                        |


**Depends on:** PR 1, PR 3.

**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## PR 5: Inprocess pipeline audio support

**Goal:** Run audio (chunk + ASR → embed) from the inprocess example.


| File                                                               | Changes                                                                                                                                                                             |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/examples/inprocess_pipeline.py` | Add `--input-type audio`, `--audio-invoke-url`, `--audio-chunk-interval`; file_patterns from matching globs only; audio branch; skip text chunk for audio; recall `.mp3` for audio. |


**Depends on:** PR 1.

**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## PR 6 (optional): Compare script + doc

**Goal:** Script and doc to compare inprocess vs batch runs (e.g. LanceDB row counts, doc basenames).


| File                                                                   | Changes                                                                       |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `nemo_retriever/src/nemo_retriever/scripts/compare_inprocess_batch.py` | New script: cleanup LanceDB dirs, run inprocess + batch to logs, compare DBs. |
| `docs/audio_inprocess_vs_batch_comparison.md`                          | New doc describing comparison and usage.                                      |


**Depends on:** None; can be merged after 1–5.

**Status:** [ ] Not started / [ ] In progress / [ ] Open / [ ] Merged

---

## Changelog (updates to this plan)

- *(Initial version: PR breakdown and merge order from branch diff vs upstream/main.)*
- **Merge `upstream/main` into `image-extract-dev`:** `asr_actor.py` conflict with **5c5557aa** (punctuation-based remote segmenting, `ASRParams.segment_audio`, tests). Resolution keeps `asr_chunks_to_text` + model/client injection and ports `_build_output_rows` / segment fan-out from main.
- Dropped **`apply_asr_to_df`** (was only test-compat); tests call **`asr_chunks_to_text`** directly.

---

## Feature branches (`asr-refactor`, `lancedb-doc-path`)

After merging current `main` into `image-extract-dev`:

- **`asr-refactor`:** Rebase onto latest `main` (or merge `main`), then **re-resolve** `audio/asr_actor.py` the same way as on `image-extract-dev` (or cherry-pick the merge-resolution commit). The old PR1 snapshot will not include `segment_audio` / `_build_output_rows` / new tests until updated.
- **`lancedb-doc-path`:** **No code overlap** with 5c5557aa. Rebase onto latest `main` only for a clean base; no ASR-specific conflict expected.
