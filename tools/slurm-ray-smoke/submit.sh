#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage: tools/slurm-ray-smoke/submit.sh path/to/cluster.my-slurm.env

The profile is a shell file containing CLUSTER_ALIAS, SCRATCH_ROOT, NODES,
NTASKS_PER_NODE, CPUS_PER_TASK, TIME_LIMIT, optional Ray ports, and
SBATCH_SELECTOR.
USAGE
}

[[ $# -eq 1 ]] || {
  usage
  exit 2
}

profile=$1
[[ -f "$profile" ]] || die "profile not found: $profile"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel)"
kit_dir="$script_dir"
template="$kit_dir/templates/job.sbatch.sh"
[[ -f "$template" ]] || die "missing template: $template"

known_profile_vars=(
  CLUSTER_ALIAS
  SCRATCH_ROOT
  SBATCH_SELECTOR
  NODES
  NTASKS_PER_NODE
  CPUS_PER_TASK
  TIME_LIMIT
  RAY_PORT
  RAY_DASHBOARD_PORT
  RAY_CLIENT_SERVER_PORT
  SOURCE_ROOT
  GPUS_PER_NODE
  RUN_TEXT_SMOKES
  RUN_PDF_PAGE_ELEMENTS_SMOKE
  RUN_PDF_OCRV2_SMOKE
  RUN_PDF_TEXT_EMBED_VDB_SMOKE
  INSTALL_PAGE_ELEMENTS_EXTRAS
  INSTALL_OCRV2_EXTRAS
  INSTALL_LOCAL_EXTRAS
  INPUT_PDF
)

while IFS='=' read -r name _value; do
  [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || continue
  found=0
  for known in "${known_profile_vars[@]}"; do
    if [[ "$name" == "$known" ]]; then
      found=1
      break
    fi
  done
  [[ "$found" -eq 1 ]] || die "unsupported profile field: $name"
done < <(grep -E '^[A-Z][A-Z0-9_]*=' "$profile" || true)

# shellcheck source=/dev/null
source "$profile"

: "${CLUSTER_ALIAS:?profile must set CLUSTER_ALIAS}"
: "${SCRATCH_ROOT:?profile must set SCRATCH_ROOT}"
: "${NODES:?profile must set NODES}"
: "${NTASKS_PER_NODE:?profile must set NTASKS_PER_NODE}"
: "${CPUS_PER_TASK:?profile must set CPUS_PER_TASK}"
: "${TIME_LIMIT:?profile must set TIME_LIMIT}"

[[ "$CLUSTER_ALIAS" =~ ^[A-Za-z0-9_.-]+$ ]] || die "CLUSTER_ALIAS may contain only letters, numbers, dot, underscore, and dash"
[[ "$CLUSTER_ALIAS" != -* ]] || die "CLUSTER_ALIAS must not start with dash"
[[ "$SCRATCH_ROOT" != *[[:space:]]* ]] || die "SCRATCH_ROOT must not contain whitespace"
[[ "$SCRATCH_ROOT" != *"'"* ]] || die "SCRATCH_ROOT must not contain single quotes"
[[ "$NODES" =~ ^[0-9]+$ ]] || die "NODES must be an integer"
[[ "$NTASKS_PER_NODE" =~ ^[0-9]+$ ]] || die "NTASKS_PER_NODE must be an integer"
[[ "$CPUS_PER_TASK" =~ ^[0-9]+$ ]] || die "CPUS_PER_TASK must be an integer"
[[ "$NODES" -ge 2 ]] || die "NODES must be at least 2 for this two-node smoke"
[[ "$NTASKS_PER_NODE" -ge 1 ]] || die "NTASKS_PER_NODE must be at least 1"
[[ "$CPUS_PER_TASK" -ge 2 ]] || die "CPUS_PER_TASK must be at least 2"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}"
SBATCH_SELECTOR="${SBATCH_SELECTOR:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"
RUN_TEXT_SMOKES="${RUN_TEXT_SMOKES:-1}"
RUN_PDF_PAGE_ELEMENTS_SMOKE="${RUN_PDF_PAGE_ELEMENTS_SMOKE:-0}"
RUN_PDF_OCRV2_SMOKE="${RUN_PDF_OCRV2_SMOKE:-0}"
RUN_PDF_TEXT_EMBED_VDB_SMOKE="${RUN_PDF_TEXT_EMBED_VDB_SMOKE:-0}"
INSTALL_LOCAL_EXTRAS="${INSTALL_LOCAL_EXTRAS:-0}"
if [[ -z "${INSTALL_PAGE_ELEMENTS_EXTRAS+x}" ]]; then
  if [[ "$INSTALL_LOCAL_EXTRAS" == "1" || "$RUN_PDF_OCRV2_SMOKE" == "1" ]]; then
    INSTALL_PAGE_ELEMENTS_EXTRAS=0
  else
    INSTALL_PAGE_ELEMENTS_EXTRAS="$RUN_PDF_PAGE_ELEMENTS_SMOKE"
  fi
fi
INSTALL_PAGE_ELEMENTS_EXTRAS="${INSTALL_PAGE_ELEMENTS_EXTRAS:-0}"
if [[ -z "${INSTALL_OCRV2_EXTRAS+x}" ]]; then
  if [[ "$INSTALL_LOCAL_EXTRAS" == "1" ]]; then
    INSTALL_OCRV2_EXTRAS=0
  else
    INSTALL_OCRV2_EXTRAS="$RUN_PDF_OCRV2_SMOKE"
  fi
fi
INSTALL_OCRV2_EXTRAS="${INSTALL_OCRV2_EXTRAS:-0}"
INPUT_PDF="${INPUT_PDF:-}"

for port_name in RAY_PORT RAY_DASHBOARD_PORT RAY_CLIENT_SERVER_PORT; do
  port_value="${!port_name}"
  [[ "$port_value" =~ ^[0-9]+$ ]] || die "$port_name must be an integer"
done

[[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]] || die "GPUS_PER_NODE must be an integer"
for flag_name in RUN_TEXT_SMOKES RUN_PDF_PAGE_ELEMENTS_SMOKE RUN_PDF_OCRV2_SMOKE RUN_PDF_TEXT_EMBED_VDB_SMOKE INSTALL_PAGE_ELEMENTS_EXTRAS INSTALL_OCRV2_EXTRAS INSTALL_LOCAL_EXTRAS; do
  flag_value="${!flag_name}"
  [[ "$flag_value" =~ ^[01]$ ]] || die "$flag_name must be 0 or 1"
done
[[ "$RUN_TEXT_SMOKES" == "1" || "$RUN_PDF_PAGE_ELEMENTS_SMOKE" == "1" || "$RUN_PDF_OCRV2_SMOKE" == "1" || "$RUN_PDF_TEXT_EMBED_VDB_SMOKE" == "1" ]] || die "at least one smoke must be enabled"
[[ "$RUN_PDF_OCRV2_SMOKE" == "0" || "$RUN_PDF_PAGE_ELEMENTS_SMOKE" == "1" ]] || die "RUN_PDF_OCRV2_SMOKE requires RUN_PDF_PAGE_ELEMENTS_SMOKE=1"
[[ "$RUN_PDF_TEXT_EMBED_VDB_SMOKE" == "0" || "$RUN_PDF_PAGE_ELEMENTS_SMOKE" == "1" ]] || die "RUN_PDF_TEXT_EMBED_VDB_SMOKE requires RUN_PDF_PAGE_ELEMENTS_SMOKE=1"
install_extra_count=$((INSTALL_PAGE_ELEMENTS_EXTRAS + INSTALL_OCRV2_EXTRAS + INSTALL_LOCAL_EXTRAS))
[[ "$install_extra_count" -le 1 ]] || die "INSTALL_PAGE_ELEMENTS_EXTRAS, INSTALL_OCRV2_EXTRAS, and INSTALL_LOCAL_EXTRAS are mutually exclusive"

input_pdf_path=""
if [[ "$RUN_PDF_PAGE_ELEMENTS_SMOKE" == "1" ]]; then
  [[ "$GPUS_PER_NODE" -ge 1 ]] || die "RUN_PDF_PAGE_ELEMENTS_SMOKE requires GPUS_PER_NODE >= 1"
  [[ "$INSTALL_PAGE_ELEMENTS_EXTRAS" == "1" || "$INSTALL_OCRV2_EXTRAS" == "1" || "$INSTALL_LOCAL_EXTRAS" == "1" ]] || die "RUN_PDF_PAGE_ELEMENTS_SMOKE requires INSTALL_PAGE_ELEMENTS_EXTRAS=1, INSTALL_OCRV2_EXTRAS=1, or INSTALL_LOCAL_EXTRAS=1"
  [[ -n "$INPUT_PDF" ]] || die "RUN_PDF_PAGE_ELEMENTS_SMOKE requires INPUT_PDF"
  if [[ "$INPUT_PDF" == /* ]]; then
    [[ -f "$INPUT_PDF" ]] || die "INPUT_PDF not found: $INPUT_PDF"
    input_pdf_path="$INPUT_PDF"
  elif [[ -f "$INPUT_PDF" ]]; then
    input_pdf_path="$(cd "$(dirname "$INPUT_PDF")" && pwd)/$(basename "$INPUT_PDF")"
  elif [[ -f "$repo_root/$INPUT_PDF" ]]; then
    input_pdf_path="$(cd "$(dirname "$repo_root/$INPUT_PDF")" && pwd)/$(basename "$INPUT_PDF")"
  else
    die "INPUT_PDF not found: $INPUT_PDF"
  fi
fi
if [[ "$RUN_PDF_OCRV2_SMOKE" == "1" ]]; then
  [[ "$INSTALL_OCRV2_EXTRAS" == "1" || "$INSTALL_LOCAL_EXTRAS" == "1" ]] || die "RUN_PDF_OCRV2_SMOKE requires INSTALL_OCRV2_EXTRAS=1 or INSTALL_LOCAL_EXTRAS=1"
fi
if [[ "$RUN_PDF_TEXT_EMBED_VDB_SMOKE" == "1" ]]; then
  [[ "$INSTALL_PAGE_ELEMENTS_EXTRAS" == "1" || "$INSTALL_OCRV2_EXTRAS" == "1" || "$INSTALL_LOCAL_EXTRAS" == "1" ]] || die "RUN_PDF_TEXT_EMBED_VDB_SMOKE requires INSTALL_PAGE_ELEMENTS_EXTRAS=1, INSTALL_OCRV2_EXTRAS=1, or INSTALL_LOCAL_EXTRAS=1"
fi

source_root="${SOURCE_ROOT:-$repo_root/nemo_retriever}"
[[ -f "$source_root/pyproject.toml" ]] || die "nemo_retriever/pyproject.toml not found under source root: $source_root"
[[ -d "$kit_dir/smokes" ]] || die "missing smoke directory: $kit_dir/smokes"
source_root="$(cd "$source_root" && pwd)"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-${CLUSTER_ALIAS}-nemo-intracluster-ray"
run_dir="${SCRATCH_ROOT%/}/nemo-intracluster/runs/${run_id}"

tmp_dir="$(mktemp -d)"
cleanup_tmp() {
  rm -rf "$tmp_dir"
}
trap cleanup_tmp EXIT

source_tar="$tmp_dir/nemo_retriever-src.tar.gz"
smokes_tar="$tmp_dir/smokes.tar.gz"
local_job="$tmp_dir/job.sbatch"

tar -C "$source_root" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --transform='s|^\./|nemo_retriever/|' \
  --transform='s|^\.$|nemo_retriever|' \
  -czf "$source_tar" \
  .

tar -C "$kit_dir" -czf "$smokes_tar" smokes

export RUN_DIR="$run_dir"
export NODES NTASKS_PER_NODE CPUS_PER_TASK TIME_LIMIT
export RAY_PORT RAY_DASHBOARD_PORT RAY_CLIENT_SERVER_PORT
export SBATCH_SELECTOR
export GPUS_PER_NODE RUN_TEXT_SMOKES RUN_PDF_PAGE_ELEMENTS_SMOKE RUN_PDF_OCRV2_SMOKE RUN_PDF_TEXT_EMBED_VDB_SMOKE INSTALL_PAGE_ELEMENTS_EXTRAS INSTALL_OCRV2_EXTRAS INSTALL_LOCAL_EXTRAS
export TEMPLATE_PATH="$template"
export OUTPUT_PATH="$local_job"

python3 - <<'PY'
from pathlib import Path
import os

template = Path(os.environ["TEMPLATE_PATH"]).read_text(encoding="utf-8")
selector = os.environ.get("SBATCH_SELECTOR", "").strip()
gpus_per_node = int(os.environ["GPUS_PER_NODE"])
mapping = {
    "RUN_DIR": os.environ["RUN_DIR"],
    "NODES": os.environ["NODES"],
    "NTASKS_PER_NODE": os.environ["NTASKS_PER_NODE"],
    "CPUS_PER_TASK": os.environ["CPUS_PER_TASK"],
    "TIME_LIMIT": os.environ["TIME_LIMIT"],
    "SBATCH_GPUS_PER_NODE": f"#SBATCH --gpus-per-node={gpus_per_node}" if gpus_per_node > 0 else "",
    "RAY_PORT": os.environ["RAY_PORT"],
    "RAY_DASHBOARD_PORT": os.environ["RAY_DASHBOARD_PORT"],
    "RAY_CLIENT_SERVER_PORT": os.environ["RAY_CLIENT_SERVER_PORT"],
    "SBATCH_SELECTOR": selector,
    "GPUS_PER_NODE": os.environ["GPUS_PER_NODE"],
    "RUN_TEXT_SMOKES": os.environ["RUN_TEXT_SMOKES"],
    "RUN_PDF_PAGE_ELEMENTS_SMOKE": os.environ["RUN_PDF_PAGE_ELEMENTS_SMOKE"],
    "RUN_PDF_OCRV2_SMOKE": os.environ["RUN_PDF_OCRV2_SMOKE"],
    "RUN_PDF_TEXT_EMBED_VDB_SMOKE": os.environ["RUN_PDF_TEXT_EMBED_VDB_SMOKE"],
    "INSTALL_PAGE_ELEMENTS_EXTRAS": os.environ["INSTALL_PAGE_ELEMENTS_EXTRAS"],
    "INSTALL_OCRV2_EXTRAS": os.environ["INSTALL_OCRV2_EXTRAS"],
    "INSTALL_LOCAL_EXTRAS": os.environ["INSTALL_LOCAL_EXTRAS"],
}

rendered = template
for key, value in mapping.items():
    rendered = rendered.replace(f"@@{key}@@", value)

if "@@" in rendered:
    raise SystemExit("unrendered template marker remains in generated sbatch")

Path(os.environ["OUTPUT_PATH"]).write_text(rendered, encoding="utf-8")
PY

printf 'Staging run directory on %s:%s\n' "$CLUSTER_ALIAS" "$run_dir"
ssh "$CLUSTER_ALIAS" "mkdir -p '$run_dir/source' '$run_dir/logs' '$run_dir/input'"
scp "$source_tar" "$CLUSTER_ALIAS:$run_dir/source/nemo_retriever-src.tar.gz"
scp "$smokes_tar" "$CLUSTER_ALIAS:$run_dir/smokes.tar.gz"
scp "$local_job" "$CLUSTER_ALIAS:$run_dir/job.sbatch"
if [[ -n "$input_pdf_path" ]]; then
  scp "$input_pdf_path" "$CLUSTER_ALIAS:$run_dir/input/smoke.pdf"
fi

job_id="$(
  ssh "$CLUSTER_ALIAS" "cd '$run_dir' && tar -xzf smokes.tar.gz && sbatch --parsable job.sbatch"
)"

cat <<EOF
cluster=$CLUSTER_ALIAS
run_id=$run_id
run_dir=$run_dir
job_id=$job_id
EOF
