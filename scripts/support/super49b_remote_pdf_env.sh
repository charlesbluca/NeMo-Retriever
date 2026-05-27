#!/usr/bin/env bash
# Source this from the NeMo-Retriever repo root:
#   source scripts/support/super49b_remote_pdf_env.sh
#
# It intentionally does not persist or print secret values. It reads ~/.env,
# maps the existing NGC/NVCF token names into the variables this repo expects,
# routes PDF extraction and embedding to hosted NVIDIA endpoints, and routes
# answer generation to the local Super-49B NIM on localhost:8000.

if [ -f "${HOME}/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${HOME}/.env"
  set +a
fi

# Hosted build.nvidia.com inference uses NVIDIA_API_KEY. The local ~/.env on
# this workstation has the NVCF developer token under NGC_NV_DEVELOPER_NVCF.
if [ -z "${NVIDIA_API_KEY:-}" ] && [ -n "${NGC_NV_DEVELOPER_NVCF:-}" ]; then
  export NVIDIA_API_KEY="${NGC_NV_DEVELOPER_NVCF}"
fi

# Self-hosted NIM model pulls/auth use NGC_API_KEY. Prefer the catalog/registry
# token that successfully authenticated to nvcr.io for the Super-49B image.
if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NGC_NVSTAGING_CATALOG_REGISTRY:-}" ]; then
  export NGC_API_KEY="${NGC_NVSTAGING_CATALOG_REGISTRY}"
fi

# Remote PDF extraction + embedding endpoints.
export PAGE_ELEMENTS_INVOKE_URL="${PAGE_ELEMENTS_INVOKE_URL:-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3}"
export OCR_INVOKE_URL="${OCR_INVOKE_URL:-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1}"
export OCR_VERSION="${OCR_VERSION:-v1}"
export GRAPHIC_ELEMENTS_INVOKE_URL="${GRAPHIC_ELEMENTS_INVOKE_URL:-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1}"
export TABLE_STRUCTURE_INVOKE_URL="${TABLE_STRUCTURE_INVOKE_URL:-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1}"
export EMBED_INVOKE_URL="${EMBED_INVOKE_URL:-https://integrate.api.nvidia.com/v1/embeddings}"
export EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-nvidia/llama-nemotron-embed-1b-v2}"

# Local Super-49B OpenAI-compatible answer-generation endpoint.
export RETRIEVER_LLM_MODEL="${RETRIEVER_LLM_MODEL:-openai/nvidia/llama-3.3-nemotron-super-49b-v1.5}"
export RETRIEVER_LLM_API_BASE="${RETRIEVER_LLM_API_BASE:-http://localhost:8000/v1}"
export RETRIEVER_LLM_API_KEY="${RETRIEVER_LLM_API_KEY:-not-needed}"
export RETRIEVER_LLM_RAG_SYSTEM_PROMPT_PREFIX="${RETRIEVER_LLM_RAG_SYSTEM_PROMPT_PREFIX:-/no_think}"
export RETRIEVER_LLM_MAX_TOKENS="${RETRIEVER_LLM_MAX_TOKENS:-512}"
export RETRIEVER_LLM_TIMEOUT="${RETRIEVER_LLM_TIMEOUT:-180}"

# LiteLLM/OpenAI local-provider compatibility when callers do not pass api_key.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${RETRIEVER_LLM_API_KEY}}"

# Batch-eval CLI compatibility names.
export GEN_MODEL="${GEN_MODEL:-${RETRIEVER_LLM_MODEL}}"
export GEN_API_BASE="${GEN_API_BASE:-${RETRIEVER_LLM_API_BASE}}"
export GEN_API_KEY="${GEN_API_KEY:-${RETRIEVER_LLM_API_KEY}}"
