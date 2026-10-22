# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only validation for the local HF image forward-batch limit."""

from __future__ import annotations

_UNRESOLVED = object()


def validate_local_hf_image_batch_size(value: object) -> int:
    """Validate the configured cap without coercing booleans or numeric strings."""
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("local_hf_image_batch_size must be an integer from 1 to 8 (not a boolean).")
    return value


def validate_local_hf_image_batch_scope(
    value: object,
    *,
    backend: str,
    endpoint: str | None = None,
    index_mode: str | None = None,
    model_type: object = _UNRESOLVED,
    local: bool = True,
) -> int:
    """Reject a nondefault cap on paths that cannot apply it, before loading.

    Default 2 is inert on other backends for compatibility. Callers that have
    resolved checkpoint metadata must pass model_type, including None for a
    checkpoint whose architecture does not support this option.
    """
    cap = validate_local_hf_image_batch_size(value)
    if cap == 2:
        return cap
    if (
        backend != "hf"
        or str(endpoint or "").strip()
        or index_mode == "sparse"
        or not local
        or (model_type is not _UNRESOLVED and model_type != "mistral3")
    ):
        raise ValueError(
            "Nondefault local_hf_image_batch_size requires dense local HF Nemotron 3.5 embedding "
            "(model_type='mistral3'); remote endpoints and other backends/models are unsupported."
        )
    return cap
