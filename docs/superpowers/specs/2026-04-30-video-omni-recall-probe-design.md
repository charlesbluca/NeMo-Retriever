# Video Omni Recall Probe Design

## Goal

Build a one-file experimental scaffold for probing video extraction with
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` through NVIDIA hosted
OpenAI-compatible endpoints. The scaffold should make it easy to inspect how
video/audio extraction design choices affect retrieval behavior before we wire
anything into the production graph pipeline.

The prototype is intentionally narrow:

- Generate a deterministic synthetic MP4 fixture.
- Send the video to Nano Omni using `NGC_NV_DEVELOPER_NVCF`.
- Convert the model response into structured retrieval rows.
- Embed rows twice: first as a text-only control, then with VL text+image
  inputs.
- Run a small query loop and print ranked hits.
- Save artifacts that expose what worked, what failed, and what tradeoffs were
  visible.

## Why This Exists

The current `nemo_retriever` video path is effectively an audio/media path:
video files are split by ffmpeg and transcribed by Parakeet ASR. That is useful
for speech, but it does not test scene understanding, visual text, frame-level
objects, or text+image retrieval. This probe explores whether a video-native
Omni model can produce richer retrieval rows and whether a VL embedder can make
those rows more discoverable.

## Non-Goals

- Do not modify `GraphIngestor`, the Ray graph, or production CLI behavior.
- Do not add a persistent vector database dependency for the first pass.
- Do not require real user-provided videos yet.
- Do not benchmark throughput or optimize batching.
- Do not claim model quality from one synthetic fixture.

## File Shape

Create one implementation file:

`nemo_retriever/examples/video_omni_recall_probe.py`

The file should be runnable as a script and keep helper functions local so the
experiment can move quickly. If the approach proves useful, later work can
extract stable pieces into package modules.

## Synthetic Fixture

The script creates a small deterministic MP4 under:

`.artifacts/video_omni_probe/synthetic_fixture.mp4`

Fixture requirements:

- Three or four scenes, each with a unique visual signal.
- Large on-screen text, for example `ALPHA-17`, `BETA PANEL`, and `CALIBRATION`.
- Distinct colors/shapes/objects per scene.
- An audio track. Prefer muxing `data/multimodal_test.wav` if present; otherwise
  generate silence so the video path still exercises `use_audio_in_video`.
- A representative frame per scene saved as base64 PNG for embedding rows.

The fixture should be regenerated only when missing unless the user passes a
force option.

## Hosted Model Call

Use raw `requests` rather than adding the OpenAI SDK as a core dependency.

Endpoint:

`https://integrate.api.nvidia.com/v1/chat/completions`

Model:

`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

Authentication:

`Authorization: Bearer $NGC_NV_DEVELOPER_NVCF`

Request content:

- A text prompt requesting strict JSON.
- A `video_url` data URL containing the small MP4 as base64.
- Extra body options that enable audio-in-video processing.
- Conservative media sampling options to keep the first experiment cheap and
  repeatable.

The prompt should request rows with:

- `segment_id`
- `start_seconds`
- `end_seconds`
- `summary`
- `visual_text`
- `objects`
- `actions`
- `audio_or_speech`
- `retrieval_keywords`
- `uncertainties`
- `confidence`

The prompt should also ask the model to avoid inventing details and to preserve
exact visible text when it can read it.

## Row Construction

The script converts model JSON into a pandas DataFrame with one row per
segment.

Each row contains:

- `text`: compact retrieval text composed from summary, visual text, objects,
  audio, and keywords.
- `_image_b64`: representative scene frame.
- `metadata`: source path, segment timing, raw model fields, extraction model,
  prompt version, and scaffold configuration.

Rows should be saved to:

`.artifacts/video_omni_probe/extracted_rows.json`

The raw model response should be saved to:

`.artifacts/video_omni_probe/omni_response.json`

## Embedding

Use the existing retriever-local embedding helper where practical:

`nemo_retriever.text_embed.main_text_embed.create_text_embeddings_for_df`

Embedding endpoint:

`https://integrate.api.nvidia.com/v1`

Embedding model:

`nvidia/llama-nemotron-embed-vl-1b-v2`

The first implementation must create two retrieval indexes in memory:

- **Text-only baseline:** embed each row using only its synthesized retrieval
  text. This is the anchor because a text-only embedder is a likely production
  endpoint for some deployments.
- **VL text+image variant:** embed the same rows with `embed_modality="text_image"`
  using the representative frame as `_image_b64`.

Query embedding mode:

Text-only with the same embedding model endpoint for both indexes.

The design intentionally uses in-memory cosine similarity rather than LanceDB
so the early experiment has fewer moving parts and makes row construction
effects easier to inspect. The console and saved artifacts should report both
rankings side by side so prompt/schema/frame changes can be evaluated against
the text-only control.

## Query Loop

The script includes built-in diagnostic queries and accepts one-off queries from
the command line.

Example built-in queries:

- `Which segment shows ALPHA-17?`
- `What scene contains the calibration panel?`
- `Which segment includes spoken or audio content?`
- `What colored object appears with the warning text?`

For each query, print:

- Rank
- Cosine similarity for the text-only baseline and the VL text+image variant
- Segment ID and time range
- Retrieval text
- Selected raw extracted fields
- Whether the expected synthetic cue was present
- Whether the top result changed between baseline and VL retrieval

Save query results to:

`.artifacts/video_omni_probe/query_results.json`

## Pros and Cons Surface

The scaffold should make tradeoffs visible in saved artifacts and console output.

Pros to expose:

- Whether the Omni model captures visual text without separate OCR.
- Whether it combines audio and visual context in one row.
- Whether text+image embeddings retrieve the intended segment better than text
  alone, worse than text alone, or simply tie the text-only baseline.
- Whether the row schema gives useful metadata for debugging misses.

Cons to expose:

- Hosted call latency and cost sensitivity.
- Base64 video payload size limits.
- Prompt/schema brittleness when strict JSON is not followed.
- Segment boundary uncertainty from model-generated extraction.
- Embedding sensitivity to representative frame choice.
- Cases where text-only retrieval is simpler and performs as well as, or better
  than, VL retrieval.
- Risk of hallucinated visual details.

The first implementation should print a short "observations" section after the
query results that summarizes these tradeoffs from the current run.

## Error Handling

Fail fast with actionable messages for:

- Missing `NGC_NV_DEVELOPER_NVCF`.
- Missing system `ffmpeg`.
- Hosted model HTTP errors, including status code and safe response excerpt.
- Non-JSON or schema-mismatched Omni responses.
- Empty embeddings.

The script should still save raw responses when parsing fails, so prompt issues
are debuggable.

## Testing

Add focused tests only if the implementation stops being trivial. For the first
pass, keep verification command-driven:

- Run `python nemo_retriever/examples/video_omni_recall_probe.py --dry-run` to
  generate and inspect the fixture without hosted calls.
- Run the full script with `NGC_NV_DEVELOPER_NVCF` set.
- Confirm artifacts are written.
- Confirm at least one built-in query retrieves its intended segment at rank 1
  in the text-only baseline.
- Confirm side-by-side text-only and VL query results are written.

If network access is unavailable in CI, tests should cover pure helpers only:
JSON parsing, row construction, cosine ranking, and fixture path handling.

## Open Decisions

- Whether to include sampled frames from every scene or only the representative
  frame selected by deterministic timestamps.
- Whether to add a real-video input flag immediately after the synthetic fixture
  works.

Initial recommendation: start implementation with the text-only control path,
then add the VL text+image path against the same rows and queries. That keeps a
simple retrieval anchor in place while making multimodal lift, regressions, and
debugging costs visible in the same run.
