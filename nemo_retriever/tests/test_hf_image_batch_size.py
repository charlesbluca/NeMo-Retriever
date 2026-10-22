# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in HF image forward batching, validated without weights or network."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params.utils import build_embed_kwargs, build_embed_option_kwargs
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.ingest import plan as ingest_plan
from nemo_retriever.models import create_local_embedder, create_local_query_embedder
from nemo_retriever.models.embed_model_spec import EmbedModelSpec
from nemo_retriever.models.inference.shared import build_embed_kwargs as actor_embed_kwargs
from nemo_retriever.models.local_embedder_spec import LocalEmbedderSpec
from nemo_retriever.models.warmup_registry import (
    build_warmup_spec,
    clear_warmed_models,
    get_warmed_model,
    warm_local_models,
)

MODEL = "nvidia/Nemotron-3.5-Embed-1B-BF16"
REVISION = "a" * 40
INVALID = [True, False, 0, -1, 9, 4.0, 2.5, "4", None]


def checkpoint(model_type="mistral3"):
    return EmbedModelSpec(
        model_id=MODEL,
        revision=REVISION,
        family="vl" if model_type == "mistral3" else "text",
        model_type=model_type,
        output_dimension=16,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_tokens=128,
    )


def config(cap=2, **kwargs):
    return {"embed_model_name": MODEL, "local_ingest_embed_backend": "hf", "local_hf_image_batch_size": cap, **kwargs}


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)


@pytest.fixture
def metadata(monkeypatch):
    resolved = MagicMock(return_value=checkpoint())
    monkeypatch.setattr("nemo_retriever.models.resolve_embed_model_spec", resolved)
    monkeypatch.setattr("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", resolved)
    monkeypatch.setattr("nemo_retriever.ingest.plan.resolve_embed_model_revision", lambda *_: REVISION)
    return resolved


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_typed_roundtrip_and_both_flattening_helpers(cap):
    params = EmbedParams(**config(cap))
    restored = EmbedParams.model_validate(params.model_dump())
    assert restored.local_hf_image_batch_size == cap
    assert build_embed_kwargs(restored)["local_hf_image_batch_size"] == cap
    assert actor_embed_kwargs(restored)["local_hf_image_batch_size"] == cap
    payload = PipelineSpec(embed_params=params.model_dump())
    assert EmbedParams(**payload.model_dump()["embed_params"]).local_hf_image_batch_size == cap


@pytest.mark.parametrize("cap", INVALID)
def test_params_and_serialized_spec_reject_non_strict_integers_before_metadata(cap, monkeypatch):
    resolve = MagicMock(side_effect=AssertionError("invalid cap must not resolve metadata"))
    monkeypatch.setattr("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", resolve)
    with pytest.raises(ValueError, match="local_hf_image_batch_size"):
        EmbedParams(**config(cap))
    with pytest.raises(ValueError, match="local_hf_image_batch_size"):
        LocalEmbedderSpec.from_config(config(cap))
    if cap is not None:  # None is omission only at the optional request boundary.
        with pytest.raises(ValueError, match="local_hf_image_batch_size"):
            ingest_plan.IngestEmbedOptions(local_hf_image_batch_size=cap)
    resolve.assert_not_called()


def test_optional_request_defers_but_resolved_null_is_invalid():
    assert ingest_plan.IngestEmbedOptions().local_hf_image_batch_size is None
    assert EmbedParams().local_hf_image_batch_size == 2
    assert "local_hf_image_batch_size" not in build_embed_option_kwargs(None, None)
    with pytest.raises(ValueError, match="local_hf_image_batch_size"):
        EmbedParams(local_hf_image_batch_size=None)


@pytest.mark.parametrize(
    "options",
    [
        {"local_ingest_embed_backend": "vllm"},
        {"local_ingest_embed_backend": "hf", "embedding_endpoint": "http://nim:8000/v1"},
        {"local_ingest_embed_backend": "hf", "embed_invoke_url": " http://nim:8000/v1 "},
    ],
)
def test_nondefault_params_reject_unsupported_paths(options):
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        EmbedParams(local_hf_image_batch_size=4, **options)
    assert EmbedParams(local_hf_image_batch_size=2, **options).local_hf_image_batch_size == 2


@pytest.mark.parametrize("model_type", [None, "llama_bidirec", "llama_nemotron_vl", "ministral3"])
def test_nondefault_spec_and_factory_reject_other_architectures_before_construction(model_type, metadata, monkeypatch):
    metadata.return_value = checkpoint(model_type)
    constructor = MagicMock(side_effect=AssertionError("unsupported model must not be constructed"))
    monkeypatch.setattr("nemo_retriever.models.local.nemotron35_embedder.Nemotron35Embedder", constructor)
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        LocalEmbedderSpec.from_config(config(4))
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        create_local_embedder(MODEL, backend="hf", local_hf_image_batch_size=4)
    constructor.assert_not_called()
    assert LocalEmbedderSpec.from_config(config(2)).local_hf_image_batch_size == 2


@pytest.mark.parametrize("cap", INVALID)
def test_direct_factories_validate_before_checkpoint_work(cap, monkeypatch):
    resolve = MagicMock(side_effect=AssertionError("must reject before checkpoint resolution"))
    monkeypatch.setattr("nemo_retriever.models.resolve_embed_model_spec", resolve)
    for factory in (create_local_embedder, create_local_query_embedder):
        with pytest.raises(ValueError, match="local_hf_image_batch_size"):
            factory(MODEL, backend="hf", local_hf_image_batch_size=cap)
    resolve.assert_not_called()


@pytest.mark.parametrize("factory", [create_local_embedder, create_local_query_embedder])
def test_public_and_resolved_factories_forward_cap(factory, metadata):
    adapter = factory(MODEL, backend="hf", local_hf_image_batch_size=4)
    assert adapter.local_hf_image_batch_size == 4
    assert adapter._model is None
    assert adapter.revision == REVISION


def test_spec_identity_default_and_warmup_cannot_cross_caps(metadata, monkeypatch):
    default = LocalEmbedderSpec.from_config(
        {key: value for key, value in config().items() if key != "local_hf_image_batch_size"}
    )
    explicit = LocalEmbedderSpec.from_config(config(2))
    candidate = LocalEmbedderSpec.from_config(config(4))
    assert default == explicit
    assert candidate != default
    assert len({candidate, default}) == 2
    assert asdict(candidate)["local_hf_image_batch_size"] == 4
    with pytest.raises(ValueError, match="local_hf_image_batch_size"):
        replace(candidate, local_hf_image_batch_size=True)
    warmed = SimpleNamespace(_ensure_loaded=MagicMock())
    created = []

    def create(spec):
        created.append(spec)
        return warmed

    monkeypatch.setattr(LocalEmbedderSpec, "create", create)
    clear_warmed_models()
    try:
        params = EmbedParams(**config(4))
        warmup = build_warmup_spec({}, params.model_dump(), None)
        assert warmup["embed"]["local_hf_image_batch_size"] == 4
        warm_local_models({"embed": warmup["embed"]})
        assert get_warmed_model("embed", expected_identity=candidate) is warmed
        assert get_warmed_model("embed", expected_identity=default) is None
        assert created == [candidate]
        warmed._ensure_loaded.assert_called_once()
    finally:
        clear_warmed_models()


def test_spec_create_forwards_cap_to_resolved_factory(metadata, monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr("nemo_retriever.models._create_local_embedder_from_spec", factory)
    spec = LocalEmbedderSpec.from_config(config(4))
    spec.create()
    assert factory.call_args.args == (spec.checkpoint,)
    assert factory.call_args.kwargs["local_hf_image_batch_size"] == 4


def test_rejected_warmup_cannot_start_extraction_models(metadata, monkeypatch):
    metadata.return_value = checkpoint("llama_bidirec")
    preload = MagicMock(side_effect=AssertionError("no extraction preload on rejected embed config"))
    monkeypatch.setattr("nemo_retriever.models.local.NemotronPageElementsV3", preload)
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        warm_local_models({"stages": ["page_elements"], "embed": config(4)})
    preload.assert_not_called()


@pytest.mark.parametrize(
    "options,index_mode",
    [
        ({"local_ingest_embed_backend": "vllm"}, "hybrid"),
        ({"local_ingest_embed_backend": "hf", "embed_invoke_url": "http://nim/v1"}, "hybrid"),
        ({"local_ingest_embed_backend": "hf"}, "sparse"),
    ],
)
def test_plan_rejects_unsupported_paths_before_document_expansion(options, index_mode, monkeypatch):
    expand = MagicMock(side_effect=AssertionError("rejected request expanded documents"))
    monkeypatch.setattr(ingest_plan, "expand_ingest_documents", expand)
    request = ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[]),
        embed=ingest_plan.IngestEmbedOptions(local_hf_image_batch_size=4, **options),
        storage=ingest_plan.IngestStorageOptions(index_mode=index_mode),
    )
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        ingest_plan.resolve_ingest_plan(request)
    expand.assert_not_called()


def test_plan_rejects_other_architecture_before_documents(metadata, monkeypatch):
    metadata.return_value = checkpoint("llama_bidirec")
    expand = MagicMock(side_effect=AssertionError("rejected request expanded documents"))
    monkeypatch.setattr(ingest_plan, "expand_ingest_documents", expand)
    request = ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[]),
        embed=ingest_plan.IngestEmbedOptions(
            embed_model_name=MODEL, local_ingest_embed_backend="hf", local_hf_image_batch_size=4
        ),
    )
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        ingest_plan.resolve_ingest_plan(request)
    expand.assert_not_called()


@pytest.mark.parametrize("cap,expected", [(None, 2), (2, 2), (4, 4)])
def test_resolved_plan_preserves_optional_and_explicit_setting(cap, expected, metadata, tmp_path):
    document = tmp_path / "input.txt"
    document.write_text("small source fixture")
    options = ingest_plan.IngestEmbedOptions(
        embed_model_name=MODEL, local_ingest_embed_backend="hf", local_hf_image_batch_size=cap
    )
    request = ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[str(document)]), embed=options
    )
    resolved = ingest_plan.resolve_ingest_plan(request)
    assert resolved.embed_params.local_hf_image_batch_size == expected
    assert build_embed_kwargs(resolved.embed_params)["local_hf_image_batch_size"] == expected


def test_service_does_not_silently_filter_nondefault_setting():
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    for value in (config(4), EmbedParams(**config(4))):
        with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
            ServiceIngestor(base_url="http://example:7670").embed(value)
    # Default 2 must not widen the service allowlist or alter existing behavior.
    service = ServiceIngestor(base_url="http://example:7670").embed(local_hf_image_batch_size=2)
    assert "local_hf_image_batch_size" not in (service._pipeline_spec.get("embed_params") or {})


@pytest.mark.parametrize("cap", INVALID)
def test_direct_adapter_validates_without_loading(cap):
    from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

    with pytest.raises(ValueError, match="local_hf_image_batch_size"):
        Nemotron35Embedder(MODEL, REVISION, "query: ", "passage: ", backend="hf", local_hf_image_batch_size=cap)


def test_direct_adapter_rejects_nondefault_vllm_without_loading():
    from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        Nemotron35Embedder(MODEL, REVISION, "query: ", "passage: ", backend="vllm", local_hf_image_batch_size=4)


def _recording_adapter(cap=2):
    import torch
    from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

    calls, processing = [], []

    class Batch(dict):
        def to(self, device):
            assert str(device) == "cpu"
            return self

    class Processor:
        def tokenizer(self, prompts, **kwargs):
            lengths = [len(prompt.split()) + 1 for prompt in prompts]
            width = max(lengths)
            return Batch(
                input_ids=torch.tensor([[i + 1] * n + [0] * (width - n) for i, n in enumerate(lengths)]),
                attention_mask=torch.tensor([[1] * n + [0] * (width - n) for n in lengths]),
            )

        def __call__(self, *, text, images, **kwargs):
            assert kwargs["truncation"] is False
            assert kwargs["add_special_tokens"] is False
            processing.append((list(text), list(images)))
            return self.tokenizer(text)

    class Model:
        device = "cpu"

        def __call__(self, **batch):
            calls.append({key: value.clone() for key, value in batch.items()})
            return SimpleNamespace(last_hidden_state=batch["input_ids"].float().unsqueeze(-1).expand(-1, -1, 4))

    adapter = Nemotron35Embedder(
        MODEL, REVISION, "query: ", "passage: ", backend="hf", output_dimension=4, local_hf_image_batch_size=cap
    )
    adapter._template = (
        "{{ messages[0].content }}{% if messages[1].content is string %}"
        "{{ messages[1].content }}{% else %}{{ messages[1].content[1].text }}{% endif %}"
    )
    adapter._model, adapter._processor = Model(), Processor()
    return adapter, calls, processing


@pytest.mark.parametrize(
    "cap,requested_batch,expected", [(2, 8, [2, 2, 2, 2, 1]), (4, 8, [4, 4, 1]), (4, 1, [1] * 9), (8, 4, [4, 4, 1])]
)
def test_actual_forward_grouping_pairing_and_positions(cap, requested_batch, expected):
    import torch

    adapter, calls, processing = _recording_adapter(cap)
    texts = ["word " * (i + 1) for i in range(9)]
    images = list(range(9))
    result = adapter._encode(texts, images=images, batch_size=requested_batch)
    assert [len(call["input_ids"]) for call in calls] == expected
    assert [image for _, group in processing for image in group] == images
    assert [text for group, _ in processing for text in group] == ["passage: " + text for text in texts]
    for call in calls:
        mask = call["attention_mask"]
        assert torch.equal(call["position_ids"], (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0))
    assert result.shape == (9, 4) and torch.isfinite(result).all()


@pytest.mark.parametrize("query", [False, True])
def test_image_cap_never_changes_text_only_or_query_cap(query):
    adapter, calls, _ = _recording_adapter(4)
    adapter._encode(["text"] * 17, query=query, batch_size=16)
    assert [len(call["input_ids"]) for call in calls] == [8, 8, 1]


def test_outer_local_runtime_remains_eight_when_ray_batch_is_sixteen():
    from nemo_retriever.models.inference.runtime import _embed_group

    observed = []

    class Model:
        def embed_images(self, images, *, batch_size):
            observed.append((len(images), batch_size))
            return [[1.0, 0.0]] * len(images)

    data = pd.DataFrame({"text": [""] * 16, "_image_b64": ["opaque-image"] * 16})
    out = _embed_group(
        data,
        group_modality="image",
        model=Model(),
        endpoint=None,
        api_key=None,
        text_column="text",
        inference_batch_size=32,
        remote_image_batch_size=1,
        output_column="vectors",
        resolved_model_name=MODEL,
    )
    assert observed == [(8, 8), (8, 8)]
    assert len(out) == 16


def test_overflow_never_truncates_or_calls_model_at_either_cap():
    for cap in (2, 4):
        adapter, calls, processing = _recording_adapter(cap)
        adapter.max_length = 2
        with pytest.raises(ValueError, match="after image expansion"):
            adapter._encode(["over budget text"], images=[0], batch_size=8)
        assert len(processing) == 1 and calls == []


@pytest.mark.parametrize("failure", ["decode", "processor", "model"])
def test_owned_images_close_on_partial_decode_or_inference_failure(failure, monkeypatch):
    adapter, _, _ = _recording_adapter(4)
    images = [SimpleNamespace(close=MagicMock()) for _ in range(2)]
    decoded = iter(images)

    def decode(value):
        if failure == "decode" and value == "second":
            raise RuntimeError("decode fixture")
        return next(decoded)

    monkeypatch.setattr("nemo_retriever.models.inference.image.decode_base64_image", decode)
    if failure == "processor":
        adapter._processor = MagicMock(side_effect=RuntimeError("processor fixture"))
    elif failure == "model":
        adapter._model = MagicMock(device="cpu", side_effect=RuntimeError("model fixture"))
    with pytest.raises(RuntimeError, match=failure + " fixture"):
        adapter.embed_text_image(["first", "second"], ["first", "second"])
    images[0].close.assert_called_once()
    assert images[1].close.call_count == (0 if failure == "decode" else 1)


def test_empty_and_count_mismatch_do_not_load(monkeypatch):
    from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

    adapter = Nemotron35Embedder(MODEL, REVISION, "q", "d", backend="hf", local_hf_image_batch_size=4)
    load = MagicMock(side_effect=AssertionError("no load"))
    monkeypatch.setattr(adapter, "_ensure_loaded", load)
    assert adapter._encode([], images=[]).shape == (0, 2048)
    with pytest.raises(ValueError, match="counts must match"):
        adapter._encode(["text"], images=[])
    load.assert_not_called()


def test_omitted_adapter_option_preserves_original_two_pair_grouping():
    from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

    prepared, calls, _ = _recording_adapter(2)
    default = Nemotron35Embedder(MODEL, REVISION, "query: ", "passage: ", backend="hf", output_dimension=4)
    default._model, default._processor, default._template = prepared._model, prepared._processor, prepared._template
    default._encode(["text"] * 9, images=list(range(9)), batch_size=8)
    assert [len(call["input_ids"]) for call in calls] == [2, 2, 2, 2, 1]


@pytest.mark.parametrize("model_id", [MODEL, "nvidia/Nemotron-3.5-Embed-8B-BF16"])
def test_same_metadata_contract_supports_both_checkpoint_sizes(model_id, metadata):
    metadata.return_value = replace(checkpoint(), model_id=model_id)
    result = create_local_embedder(model_id, backend="hf", local_hf_image_batch_size=4)
    assert result.model_id == model_id and result.local_hf_image_batch_size == 4
    assert result._model is None


@pytest.mark.parametrize("model_type", ["mistral3", "llama_bidirec"])
def test_graph_validates_architecture_and_retains_cap_before_actor_creation(
    model_type, metadata, monkeypatch, tmp_path
):
    from tests.test_graph_embedding_identity import resolved_stages

    metadata.return_value = checkpoint(model_type)
    monkeypatch.setattr("nemo_retriever.models.embed_model_spec.resolve_embed_model_revision", lambda *_: REVISION)
    params = EmbedParams(**config(4))
    if model_type != "mistral3":
        with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
            resolved_stages(tmp_path, params)
    else:
        actor_params, sink = resolved_stages(tmp_path, params)
        assert actor_params.local_hf_image_batch_size == 4
        assert actor_params.embed_model_revision == sink["vdb_kwargs"]["embedding_model_revision"] == REVISION


@pytest.mark.parametrize("failure", ["backend", "architecture"])
def test_unsupported_append_request_rejects_before_storage_inspection(failure, metadata, monkeypatch):
    inspect = MagicMock(side_effect=AssertionError("invalid tuning must not open or create storage"))
    monkeypatch.setattr(ingest_plan, "inspect_existing_lancedb_mode", inspect)
    metadata.return_value = checkpoint("llama_bidirec" if failure == "architecture" else "mistral3")
    request = ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[]),
        embed=ingest_plan.IngestEmbedOptions(
            embed_model_name=MODEL,
            local_ingest_embed_backend="hf" if failure == "architecture" else "vllm",
            local_hf_image_batch_size=4,
        ),
        storage=ingest_plan.IngestStorageOptions(overwrite=False),
    )
    with pytest.raises(ValueError, match="Nondefault local_hf_image_batch_size"):
        ingest_plan.resolve_ingest_plan(request)
    inspect.assert_not_called()
