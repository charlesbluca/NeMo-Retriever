# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.ocr.gpu_ocr import OCRActor as _UnifiedOCRGPUActor


class OCRV2Actor(_UnifiedOCRGPUActor):
    """Compatibility wrapper for the unified GPU OCR actor."""
