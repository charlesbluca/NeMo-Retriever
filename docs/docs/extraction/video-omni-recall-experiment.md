# Experimental Video Omni Recall Probe

This note summarizes the current experimental scaffold in
`nemo_retriever/examples/video_omni_recall_probe.py`. The probe is intentionally
self-contained: it does not change the production GraphIngestor path, and it is
designed to make the video extraction and retrieval tradeoffs visible while the
approach is still early.

## Goal

The experiment asks whether a Nano Omni video extraction step can produce useful
retrieval records for short videos, then compares two embedding strategies over
the extracted records:

- A text-only embedder over the extracted segment text.
- A VL embedder over the same extracted text plus a representative frame.

The current scaffold can run against either a synthetic fixture or a repo-local
video retrieval dataset with golden queries and expected answer time ranges.

## Models

The hosted model defaults are:

| Role | Model |
| --- | --- |
| Video extraction | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Text-only embedding | `nvidia/llama-nemotron-embed-1b-v2` |
| VL text+image embedding | `nvidia/llama-nemotron-embed-vl-1b-v2` |

Hosted calls use `https://integrate.api.nvidia.com/v1` and read the API key from
`NGC_NV_DEVELOPER_NVCF`. Non-NVIDIA endpoints are rejected unless the user opts
in with `--allow-custom-endpoint`.

## Architecture

The high-level flow is:

```text
video or dataset row
  -> optional ffmpeg chunking
  -> Nano Omni video prompt
  -> structured segment JSON
  -> normalized retrieval rows
  -> representative frame extraction
  -> text-only embedding path
  -> VL text+image embedding path
  -> cosine ranking
  -> golden query metrics
```

### Extraction

For dataset evaluation, each video is represented as an `EvalVideoJob` with:

- the source video path,
- the video duration,
- query strings, and
- optional golden expectations such as expected answer text and expected
  start/end timestamps.

Longer videos are split into overlapping `ChunkSpec` windows. The current
default is 30 second chunks with 5 seconds of overlap. Each chunk is written as
an MP4 with `ffmpeg`, then sent to Nano Omni using a prompt that asks for JSON
segments with the following shape:

```json
{
  "segments": [
    {
      "segment_id": "string",
      "start_seconds": 0,
      "end_seconds": 10,
      "summary": "string",
      "visual_text": ["string"],
      "objects": ["string"],
      "actions": ["string"],
      "audio_or_speech": "string",
      "retrieval_keywords": ["string"],
      "uncertainties": ["string"],
      "confidence": 0.9
    }
  ]
}
```

The parser accepts several model-output shapes that appeared during testing:

- strict `{"segments": [...]}` JSON,
- markdown-wrapped JSON,
- bare JSON arrays, and
- a single segment object when the model returns a recoverable fragment.

Chunk failures are isolated. If a chunk cannot be extracted, parsed, or
validated, the probe writes a `chunk-xxxx.error.json` artifact and continues with
the rest of the video. A later successful retry removes the stale error artifact.

### Row Construction

Each valid Omni segment becomes one retrieval row. The row text concatenates:

- segment id,
- summary,
- visual text,
- objects,
- actions,
- audio or speech,
- retrieval keywords.

The row metadata carries source video name, absolute segment time, local chunk
id, confidence, and the raw normalized Omni segment. Absolute timestamps are
computed by adding the chunk start time to each local segment timestamp.

For VL retrieval, the probe extracts one representative frame for each segment
at the segment midpoint. That frame is base64 encoded and attached to the row.
If frame extraction fails, the row remains usable for text-only retrieval.

### Embedding and Ranking

The probe materializes two embedding views from the same row set:

- text-only rows, using `text` as the embedding input,
- VL rows, using `text` plus the representative image payload where available.

Queries are embedded with the corresponding model. The probe ranks rows with
cosine similarity and reports the top `k` results for both paths. The dataset
evaluation path writes:

- `eval_rows.json` for extracted/indexable rows,
- `eval_results.json` for per-query rankings and hit flags,
- `eval_summary.json` for aggregate metrics.

The current metrics focus on early recall signals rather than production-quality
answering:

- top-1 timestamp overlap with the golden answer window,
- top-k timestamp overlap,
- mean reciprocal rank over timestamp hits,
- whether the top result contains the expected answer text.

## Current Very Short Dataset Run

The latest checkpoint ran against `~/datasets/video_retrieval` with:

- `--video-bin "Very Short"`
- `--max-videos 3`
- `--query-limit 4`
- `--chunk-seconds 30`
- `--chunk-overlap-seconds 5`
- `--top-k 3`

This produced 12 evaluated queries across 3 SAP Datasphere videos. After parser
hardening and cache reuse, the final extraction row counts were:

| Video | Extracted rows |
| --- | ---: |
| `2024_04_SAP_Datasphere_Top_Features_1_36zflqjo_` | 14 |
| `2024_06_SAP_Datasphere_Top_Features_1_6f33wyx8_` | 2 |
| `2024_07_SAP_Datasphere_Top_Features_1_6vjiue52_` | 13 |

The final aggregate retrieval metrics were:

| Metric | Text-only | VL text+image |
| --- | ---: | ---: |
| Top-1 time overlap | 50.0% | 41.7% |
| Top-3 time overlap | 66.7% | 58.3% |
| Time MRR | 56.9% | 50.0% |
| Answer hit rate | 16.7% | 8.3% |

Per-video top-k timestamp recall showed that most of the weakness came from the
June video, where extraction produced only two usable rows:

| Video | Queries | Text top-1 | Text top-3 | VL top-1 | VL top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| April | 4 | 3 | 4 | 3 | 4 |
| June | 4 | 0 | 0 | 0 | 0 |
| July | 4 | 3 | 4 | 2 | 3 |

## What We Learned

The strongest positive signal is that the architecture is usable as a fast
experiment loop. We can point it at local video data, cache intermediate Omni
responses, compare text-only and VL retrieval, and inspect per-query misses.

The current results also show that extraction reliability dominates retrieval
quality. Six chunk windows still failed with `No JSON object found` after retry.
Those failures removed coverage from the index, and the June video shows the
impact clearly: too few extracted rows means neither embedding path has enough
evidence to retrieve.

In this setup, text-only retrieval is ahead of VL retrieval. That does not mean
the VL embedder is unhelpful; it means the current image signal is not yet
improving ranking over the extracted text. Possible reasons include:

- the Omni text already captures the relevant evidence for many queries,
- one midpoint frame may be a weak visual representative for a segment,
- UI-heavy videos contain dense text where OCR/transcription dominates,
- failed chunks skew the comparison by removing evidence before embedding, and
- cosine-only ranking without reranking is a thin retrieval stack.

The answer hit rate is much lower than timestamp recall. That is expected for
this early metric because the extracted segment text may overlap the correct
time window without repeating the golden answer wording exactly.

## Reasonable Follow-Up Experiments

1. Capture raw malformed Omni responses.
   The remaining `No JSON object found` failures currently persist only as error
   artifacts. Persisting raw response text for parse failures would make prompt
   and parser iteration much faster.

2. Tighten the extraction contract.
   Try stronger JSON-only prompting, lower token budgets per chunk, explicit
   segment count limits, or an endpoint mode that supports structured outputs if
   available. Measure chunk success rate before looking at retrieval metrics.

3. Sweep chunk sizes and overlap.
   Compare 15s, 30s, and 60s chunks with 0s, 5s, and 10s overlap. Shorter chunks
   may improve model compliance and timestamp precision; longer chunks may
   preserve context but increase malformed outputs.

4. Improve visual evidence selection.
   Replace one midpoint frame with multiple frames per segment, shot-change
   frames, or frame thumbnails near visible text transitions. Then compare VL
   retrieval again.

5. Add extraction baselines.
   Compare Omni extraction against simpler pipelines such as transcript-only,
   OCR-only sampled frames, and transcript+OCR. This will clarify how much Nano
   Omni contributes beyond cheaper signals.

6. Add retrieval baselines.
   Compare text-only embedder, VL embedder, hybrid sparse+dense, and reranked
   top-k. A reranker may matter more than the first-stage embedder once chunk
   coverage improves.

7. Evaluate more bins.
   Run the same harness over additional `video_bin` values and report extraction
   success rate, rows per minute of video, and recall by answer modality.

8. Normalize segment granularity.
   Some model outputs produce very broad or uneven segments. Post-processing
   segments into fixed-size retrieval windows, or splitting long segments by
   sentence and frame, may produce more stable indexing behavior.

9. Track extraction coverage as a first-class metric.
   Add per-video chunk attempts, chunk success rate, row count, rows per minute,
   and skipped time coverage to `eval_summary.json`. Retrieval misses are hard to
   interpret without knowing how much evidence entered the index.

## Current Caveats

- This is a one-file experimental scaffold, not a production API.
- Hosted endpoint behavior may vary between runs.
- Failed `No JSON object found` chunks are skipped, so current recall numbers are
  partly extraction-reliability numbers.
- The comparison is over a very small sample: 3 videos and 12 queries.
- VL retrieval uses only one representative frame per segment.
- Ranking is cosine-only and does not include reranking or hybrid search.

## Practical Next Step

The next best experiment is to make extraction reliability measurable before
changing embedders: persist raw malformed responses, add chunk coverage metrics,
then rerun the same Very Short slice. Once the extraction success rate is stable,
the text-only versus VL comparison will be much easier to interpret.
