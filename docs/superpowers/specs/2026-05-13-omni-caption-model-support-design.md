# Nemotron 3 Nano Omni Caption Model Support Design

Date: 2026-05-13

## Summary

Add opt-in support for the Nemotron 3 Nano Omni model family in the VLM captioner stage while keeping the current Nano 12B VL model as the default. The first implementation is image-captioning only, across both local in-process execution and remote OpenAI-compatible endpoints.

The design introduces a small caption model profile layer so model-family differences are represented as data and narrow request-shaping behavior, not scattered conditionals. The profile layer should also chart the path to future Omni capabilities in this priority order:

1. Audio/video input support.
2. OCR and document-intelligence tasks.
3. Reasoning mode control.

## Goals

- Preserve existing caption stage behavior and defaults.
- Support Omni BF16, FP8, and NVFP4 local Hugging Face model IDs.
- Support the hosted Omni remote model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.
- Keep local and remote captioning behavior consistent where practical.
- Disable Omni reasoning traces by default for image captioning.
- Leave clear extension points for audio/video, document intelligence, and reasoning controls without implementing those features in this slice.

## Non-Goals

- Do not change the default caption model from `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`.
- Do not add audio or video ingestion into the caption stage yet.
- Do not add raw PDF handling through Omni; document work should continue to use rendered page images when added later.
- Do not create a broad multimodal pipeline redesign.
- Do not document runtime internals for end users beyond the normal local install path.

## Current Context

The current captioning implementation is centered on:

- `nemo_retriever/src/nemo_retriever/caption/caption.py`
- `nemo_retriever/src/nemo_retriever/model/local/nemotron_vlm_captioner.py`
- `nemo_retriever/src/nemo_retriever/api/internal/primitives/nim/model_interface/vlm.py`
- `nemo_retriever/src/nemo_retriever/params/models.py`

Today, model support is mostly hard-coded in alias maps, supported model dictionaries, revision dictionaries, and quantization profiles. This is fine for one model family, but Omni adds family-level behavior differences: hosted model naming differs from local HF naming, reasoning is enabled by default unless disabled, and future audio/video/document tasks need modality-aware request formatting.

## Proposed Approach

Use a caption model profile registry.

Each profile describes a model family/variant and includes:

- Canonical family name.
- Variant or precision name.
- Local Hugging Face model ID when local execution is supported.
- Remote served model ID when remote execution is supported.
- Accepted aliases.
- vLLM engine or quantization hints for local execution.
- Default request extras for local and remote chat requests.
- Capability flags for supported and future modalities/tasks.

Two model families should be represented initially:

- Current Nano 12B VL: existing supported BF16, FP8, and NVFP4-QAD variants, existing aliases, current behavior.
- Nemotron 3 Nano Omni: BF16, FP8, and NVFP4 local HF variants, hosted remote ID, image-caption support enabled now, future capability flags for audio/video, OCR/doc intelligence, and reasoning.

## Model Names

The current default remains:

- `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`

The new Omni local HF IDs are:

- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

Convenience aliases should include:

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8`
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4`

For remote execution, all Omni variants resolve to:

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

If NVIDIA later exposes precision-specific hosted names, those can be added to the profile registry without changing caption-stage call sites.

## Component Design

### Caption Model Profile

Add a small profile type, likely near `nemotron_vlm_captioner.py` unless implementation discovers a cleaner split. The type should be simple and testable. A dataclass is sufficient.

The profile lookup API should provide:

- Resolve by alias or canonical model ID.
- Resolve for a target, either `local` or `remote`.
- Return a rich profile object for new code.
- Preserve a string-returning compatibility helper for existing callers and tests.

### Local Captioner

`NemotronVLMCaptioner` continues to own vLLM loading and `caption_batch()`.

Changes:

- Resolve `model_path` through the profile registry.
- Validate against local-supported caption profiles.
- Use the profile's local HF model ID.
- Use the profile's revision and engine kwargs.
- Apply profile request extras when building chat calls.

For image captioning with Omni, the profile should disable reasoning traces by default through chat-template request options where supported.

### Remote Captioning

`caption_images()` and `_caption_batch_remote()` keep the current remote flow:

1. Scale base64 images.
2. Build OpenAI-compatible messages.
3. Call `nim_client.infer(...)`.
4. Write returned captions back to the dataframe.

Changes:

- Resolve `model_name` to the profile's remote served ID for known models.
- Preserve pass-through behavior for unknown remote model names so custom endpoints still work.
- Pass optional profile request extras into the NIM interface.
- Merge profile extras with user-supplied extras predictably, with user values winning.

### VLMModelInterface

Keep the current image chat request shape. Extend it only enough to include optional request extras such as:

- `chat_template_kwargs`
- future `mm_processor_kwargs`
- future media-related request options

This avoids a broad multimodal interface rewrite while making the next feature slice easier.

### CaptionParams

Keep `CaptionParams` backward compatible.

Potential optional fields:

- `extra_body`: an advanced escape hatch for OpenAI-compatible request extras.
- `enable_reasoning`: only if implementation needs a user-facing override in this slice.

If the implementation can support the image-captioning behavior without a new public parameter, prefer avoiding new API surface for now.

## Data Flow

The stage boundary remains unchanged:

1. `caption_images()` finds pending images or infographic crops.
2. It resolves `model_name` into a caption profile for local or remote execution.
3. Local execution calls `NemotronVLMCaptioner.caption_batch()`.
4. Remote execution calls `_caption_batch_remote()` through `VLMModelInterface`.
5. Captions are written back into `images[i]["text"]` or `infographic[i]["caption"]`.

Context-text prompting, minimum image size filtering, batching, and writeback semantics remain unchanged.

## Future Capability Path

The profile registry should include capability flags even though only image captioning is active now.

Recommended capability names:

- `image_captioning`
- `audio_input`
- `video_input`
- `document_intelligence`
- `reasoning_control`

Future slices can use these flags as follows:

- Audio/video: add typed media request builders and route media content through the same profile-backed request formatting.
- OCR/document intelligence: add task-specific prompts over rendered page images, not raw PDF payloads.
- Reasoning mode: expose a user-facing option that maps to profile request extras, while preserving non-thinking defaults for captioning and likely audio/video transcription-style tasks.

## Error Handling

- Unsupported local caption model: raise `ValueError` listing supported local caption model IDs and aliases.
- Unsupported profile/target combination: raise a clear error naming the model and target.
- Unknown remote model: preserve pass-through behavior for custom OpenAI-compatible VLM endpoints.
- Request extras: deep-merge profile defaults and user extras with user values winning.
- Missing local dependencies: keep the existing local import error pattern and instruct users to install the local extra, for example `nemo_retriever[local]`.

## Testing

Focused unit tests should cover:

- Existing Nano model-name resolution remains unchanged.
- Omni full HF IDs map to the expected local HF IDs.
- Omni aliases map to the expected local HF IDs.
- Omni full HF IDs and aliases map to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` for remote execution.
- `CaptionParams()` default model remains current Nano.
- Remote Omni requests include non-thinking profile extras by default.
- User-supplied request extras override profile defaults.
- Local Omni construction selects expected local profile metadata for BF16, FP8, and NVFP4 without importing real vLLM.
- Caption image writeback behavior stays unchanged for mocked Nano and Omni flows.
- New local HF model IDs are pinned in the Hugging Face model registry once exact revisions are selected.

## Documentation

Update documentation only where users need to discover the opt-in model support:

- Support matrix or caption model references should mention that Omni is opt-in for image captioning.
- Remote examples should show the hosted model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` when relevant.
- Local examples should use the normal local install path, `nemo_retriever[local]`, without documenting dependency-stack internals.

Do not turn this into a full Omni feature guide until audio/video or document-intelligence support is implemented.

## References

- NVIDIA API reference for hosted Omni model: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-infer
- Hugging Face model card: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- Current captioner implementation: `nemo_retriever/src/nemo_retriever/model/local/nemotron_vlm_captioner.py`
- Current caption stage: `nemo_retriever/src/nemo_retriever/caption/caption.py`
