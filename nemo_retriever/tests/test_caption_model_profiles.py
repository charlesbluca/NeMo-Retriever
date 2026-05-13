# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

import pytest


NANO_BF16 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
NANO_FP8 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
NANO_NVFP4 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"
NANO_REMOTE = "nvidia/nemotron-nano-12b-v2-vl"
OMNI_BF16 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
OMNI_FP8 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
OMNI_NVFP4 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
OMNI_REMOTE = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def test_nano_resolution_remains_unchanged():
    from nemo_retriever.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl", target="local") == NANO_BF16
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-bf16", target="local") == NANO_BF16
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-fp8", target="local") == NANO_FP8
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad", target="local") == NANO_NVFP4
    assert resolve_caption_model_name(NANO_BF16, target="remote") == NANO_REMOTE
    assert resolve_caption_model_name(NANO_FP8, target="remote") == "nvidia/nemotron-nano-12b-v2-vl-fp8"
    assert resolve_caption_model_name(NANO_NVFP4, target="remote") == "nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (OMNI_BF16, OMNI_BF16),
        (OMNI_FP8, OMNI_FP8),
        (OMNI_NVFP4, OMNI_NVFP4),
        (OMNI_REMOTE, OMNI_BF16),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16", OMNI_BF16),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8", OMNI_FP8),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4", OMNI_NVFP4),
    ],
)
def test_omni_local_names_resolve_to_hf_ids(name, expected):
    from nemo_retriever.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name(name, target="local") == expected


@pytest.mark.parametrize(
    "name",
    [
        OMNI_BF16,
        OMNI_FP8,
        OMNI_NVFP4,
        OMNI_REMOTE,
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4",
    ],
)
def test_omni_remote_names_resolve_to_hosted_model(name):
    from nemo_retriever.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name(name, target="remote") == OMNI_REMOTE


def test_unknown_remote_model_name_passes_through():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile, resolve_caption_model_name

    assert get_caption_model_profile("acme/custom-vlm", target="remote", strict=False) is None
    assert resolve_caption_model_name("acme/custom-vlm", target="remote") == "acme/custom-vlm"


def test_unknown_local_profile_raises_clear_error():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile

    with pytest.raises(ValueError) as exc_info:
        get_caption_model_profile("acme/custom-vlm", target="local")

    message = str(exc_info.value)
    assert "Unsupported caption model" in message
    assert "target='local'" in message
    assert NANO_BF16 in message
    assert OMNI_BF16 in message


def test_omni_profile_has_request_defaults_and_future_capabilities():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_FP8, target="local")

    assert profile.family == "nemotron-3-nano-omni"
    assert profile.variant == "FP8"
    assert profile.local_model_id == OMNI_FP8
    assert profile.remote_model_id == OMNI_REMOTE
    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.request_extras_for("remote") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.capabilities.image_captioning is True
    assert profile.capabilities.audio_input is True
    assert profile.capabilities.video_input is True
    assert profile.capabilities.document_intelligence is True
    assert profile.capabilities.reasoning_control is True


def test_request_extras_are_defensive_copies():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_BF16, target="local")
    extras = profile.request_extras_for("local")
    extras["chat_template_kwargs"]["enable_thinking"] = True

    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_merge_request_extras_deep_merges_with_user_values_winning():
    from nemo_retriever.caption.model_profiles import merge_request_extras

    defaults = {
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 0},
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
    }
    user = {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 1,
    }

    assert merge_request_extras(defaults, user) == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 0},
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
        "top_k": 1,
    }
    assert defaults["chat_template_kwargs"]["enable_thinking"] is False


def test_supported_local_names_include_ids_and_aliases():
    from nemo_retriever.caption.model_profiles import supported_caption_model_names

    names = supported_caption_model_names(target="local")

    assert NANO_BF16 in names
    assert OMNI_BF16 in names
    assert "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8" in names


def test_public_request_extra_fields_are_immutable():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_BF16, target="local")

    with pytest.raises(TypeError):
        profile.local_request_extras["chat_template_kwargs"]["enable_thinking"] = True
    with pytest.raises(TypeError):
        profile.remote_request_extras["chat_template_kwargs"]["enable_thinking"] = True

    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.request_extras_for("remote") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_public_fp8_engine_kwargs_are_immutable():
    from nemo_retriever.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_FP8, target="local")

    with pytest.raises(TypeError):
        profile.local_engine_kwargs["hf_overrides"]["quantization_config"]["activation_scheme"] = "dynamic"
    with pytest.raises(TypeError):
        profile.local_engine_kwargs["quantization"] = "modelopt"

    assert profile.engine_kwargs_for_local() == {
        "dtype": "auto",
        "quantization": "fp8",
        "hf_overrides": {"quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}},
    }
