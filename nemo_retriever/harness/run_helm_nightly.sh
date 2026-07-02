#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_CONFIG="${NEMO_RETRIEVER_HELM_NIGHTLY_CONFIG:-${SCRIPT_DIR}/helm_nightly_config.yaml}"

: "${HARNESS_HELM_SERVICE_IMAGE_REPOSITORY:?set the main/nightly service image repository}"
: "${HARNESS_HELM_SERVICE_IMAGE_TAG:?set the immutable main/nightly service image tag}"
: "${SLACK_WEBHOOK_URL:?set the channel-bound Slack webhook}"

exec uv run --project "${REPO_ROOT}/nemo_retriever" \
  retriever harness nightly --runs-config "${RUNS_CONFIG}" "$@"
