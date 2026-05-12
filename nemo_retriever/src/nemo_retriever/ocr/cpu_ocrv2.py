# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.ocr.cpu_ocr import OCRCPUActor as _UnifiedOCRCPUActor


class OCRV2CPUActor(_UnifiedOCRCPUActor):
    """Compatibility wrapper for the unified CPU OCR actor."""
