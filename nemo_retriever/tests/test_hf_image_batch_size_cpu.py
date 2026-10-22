# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real tiny CPU attention and vision, with no pretrained assets or network.

The lossless fixture processor supplies explicit image tokens and pixel tensors.
The forward is an actual randomly initialized Mistral3/Pixtral/Ministral3 model,
not a stub; this isolates batching/padding from pretrained processor behavior.
"""

import base64
from io import BytesIO

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

from nemo_retriever.models.inference.runtime import _embed_group
from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

TEMPLATE = (
    "{{ messages[0].content }}{% if messages[1].content is string %}"
    "{{ messages[1].content | replace('[', '[‌') }}{% else %}[IMG] "
    "{{ messages[1].content[1].text | replace('[', '[‌') }}{% endif %}"
)


class Batch(dict):
    def to(self, device):
        assert str(device) == "cpu"
        return Batch({key: value.to(device) for key, value in self.items()})


class LosslessProcessor:
    """Deterministic text IDs and actual patch-aligned variable-size images."""

    def __init__(self, padding_side):
        self.padding_side = padding_side
        self.calls = []

    def _batch(self, prompts, images=None):
        sequences = []
        for i, prompt in enumerate(prompts):
            ids = [3 + ord(char) % 100 for char in prompt]
            if images is not None:
                image = images[i]
                ids += [127] * ((image.height // 4) * (image.width // 4))
            sequences.append(ids)
        width = max(map(len, sequences))
        ids, masks = [], []
        for sequence in sequences:
            pad = [0] * (width - len(sequence))
            mask = [1] * len(sequence)
            if self.padding_side == "left":
                ids.append(pad + sequence)
                masks.append(pad + mask)
            else:
                ids.append(sequence + pad)
                masks.append(mask + pad)
        batch = Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))
        if images is not None:
            height, width = max(i.height for i in images), max(i.width for i in images)
            pixels = torch.zeros(len(images), 3, height, width)
            for i, image in enumerate(images):
                raw = torch.tensor(bytearray(image.tobytes()), dtype=torch.float32).reshape(
                    image.height, image.width, 3
                )
                pixels[i, :, : image.height, : image.width] = raw.permute(2, 0, 1) / 255
            batch.update(pixel_values=pixels, image_sizes=torch.tensor([[i.height, i.width] for i in images]))
        return batch

    def tokenizer(self, prompts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return self._batch(prompts)

    def __call__(self, *, text, images, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is False
        self.calls.append((list(text), [image.size for image in images]))
        return self._batch(text, images)


@pytest.fixture
def real_model():
    # CPU generator only, with bounded fixture dimensions and no checkpoint load.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(127)
        text = Ministral3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=256,
            pad_token_id=0,
            attention_dropout=0.0,
            use_cache=False,
            is_causal=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 256,
            },
        )
        vision = PixtralVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=2,
            attention_dropout=0.0,
        )
        config = Mistral3Config(
            text_config=text,
            vision_config=vision,
            image_token_index=127,
            spatial_merge_size=2,
        )
        config._attn_implementation = "sdpa"
        model = Mistral3Model(config).float().eval()
    # Match the real adapter's shipped bidirectional module initialization.
    for module in model.language_model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
    assert next(model.parameters()).device.type == "cpu"
    return model


def adapter(model, cap, padding_side):
    result = Nemotron35Embedder(
        "tiny-private-fixture",
        None,
        "query: ",
        "passage: ",
        backend="hf",
        device="cpu",
        output_dimension=16,
        max_length=256,
        query_max_length=256,
        local_hf_image_batch_size=cap,
    )
    result._model = model
    result._processor = LosslessProcessor(padding_side)
    result._template = TEMPLATE
    return result


def image_payload(index, size):
    with Image.new("RGB", size) as image:
        image.putdata(
            [
                ((index * 41 + x * 17) % 256, (y * 33 + index * 7) % 256, (x + y * 19 + index * 23) % 256)
                for y in range(size[1])
                for x in range(size[0])
            ]
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def inputs():
    texts = ["short", "longer content", "Literal [IMG]", "passage: already prefixed", "last"]
    images = [image_payload(i, size) for i, size in enumerate([(8, 4), (4, 8), (8, 8), (12, 8), (8, 4)])]
    return texts, images


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_real_attention_and_vision_are_batch_invariant(real_model, padding_side, modality):
    texts, images = inputs()
    snapshots, vision_calls = [], []

    def record(_module, _args, kwargs):
        snapshots.append(
            {key: value.detach().clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
        )

    hook = real_model.register_forward_pre_hook(record, with_kwargs=True)
    vision_hook = real_model.vision_tower.register_forward_hook(lambda *_: vision_calls.append(True))
    try:
        outputs = []
        groupings = []
        for cap in (1, 2, 4):
            snapshots.clear()
            embedder = adapter(real_model, cap, padding_side)
            result = (
                embedder.embed_images(images, batch_size=8)
                if modality == "image"
                else embedder.embed_text_image(texts, images, batch_size=8)
            )
            outputs.append(result)
            groupings.append([len(call["input_ids"]) for call in snapshots])
            assert result.shape == (5, 16) and torch.isfinite(result).all()
            torch.testing.assert_close(result.norm(dim=1), torch.ones(5), atol=2e-6, rtol=2e-5)
            for call in snapshots:
                mask = call["attention_mask"]
                expected = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
                assert torch.equal(call["position_ids"], expected)
                assert call["pixel_values"].device.type == "cpu"
            prompts = [prompt for prompts, _ in embedder._processor.calls for prompt in prompts]
            assert len(prompts) == 5
            assert all(prompt.startswith("passage: ") and prompt.count("passage: ") == 1 for prompt in prompts)
            assert all("[\u200cIMG]" in prompt for prompt in prompts[2:3]) if modality == "text_image" else True
        assert groupings == [[1, 1, 1, 1, 1], [2, 2, 1], [4, 1]]
        torch.testing.assert_close(outputs[1], outputs[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(outputs[2], outputs[0], atol=2e-6, rtol=2e-5)
        assert len(vision_calls) == 10
        # This fixture must detect a changed image rather than constant outputs.
        changed = adapter(real_model, 4, padding_side).embed_text_image(texts, list(reversed(images)), batch_size=8)
        original = adapter(real_model, 4, padding_side).embed_text_image(texts, images, batch_size=8)
        assert (changed - original).abs().max().item() > 1e-6
    finally:
        hook.remove()
        vision_hook.remove()


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_real_mixed_modal_runtime_retains_row_positions_and_vectors(real_model, padding_side):
    texts, images = inputs()
    selected = [images[0], None, images[2], None, images[4]]
    data = pd.DataFrame({"text": texts, "_image_b64": selected}, index=[8, 1, 2, 10, 7])
    outputs = []
    for cap in (2, 4):
        embedder = adapter(real_model, cap, padding_side)
        out = _embed_group(
            data,
            group_modality="text_image",
            model=embedder,
            endpoint=None,
            api_key=None,
            text_column="text",
            inference_batch_size=32,
            output_column="vectors",
            resolved_model_name="tiny-private-fixture",
        )
        assert len(out) == 5 and list(out.index) == list(data.index)
        vectors = torch.tensor([payload["embedding"] for payload in out["vectors"]])
        reference = []
        for text, image in zip(texts, selected):
            vector = embedder.embed([text]) if image is None else embedder.embed_text_image([text], [image])
            reference.append(vector[0])
        torch.testing.assert_close(vectors, torch.stack(reference), atol=2e-6, rtol=2e-5)
        outputs.append(vectors)
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_real_text_queries_preserve_prompt_and_cap_eight(real_model, padding_side):
    texts = ["query " + str(i) for i in range(17)]
    embedder = adapter(real_model, 4, padding_side)
    snapshots = []
    hook = real_model.register_forward_pre_hook(
        lambda _m, _a, kw: snapshots.append(kw["input_ids"].clone()), with_kwargs=True
    )
    try:
        result = embedder.embed_queries(texts, batch_size=16)
        assert result.shape == (17, 16)
        assert [len(batch) for batch in snapshots] == [8, 8, 1]
        expected = torch.cat([embedder.embed_queries([text]) for text in texts])
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    finally:
        hook.remove()
