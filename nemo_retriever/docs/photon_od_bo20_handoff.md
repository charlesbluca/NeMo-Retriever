# Photon OD bo20 Handoff

This handoff captures the staged Photon OD dual-service bo20 setup from
2026-06-23. The machine used for the initial qualification attempt is unhealthy,
so treat the chart/harness work as the useful artifact and rerun the sweeps on a
different host.

## Objective

Qualify bo20 concurrent service ingest with two separate OD-backed NIMServices:

- page elements: `nemotron-page-elements-v3`
- table structure: `nemotron-table-structure-v1`

Both use the same staged OD runtime image but different
`NIM_ENGINE_MODEL_NAME` values and different HTTP routes. This preserves the
retriever chart's existing separate `pageElementsInvokeUrl` and
`tableStructureInvokeUrl` wiring while exercising the new OD runtime for both
failure-prone stages.

Prior local-NIM baseline cliffs to compare against:

- `return_results=False`: around `N=8`
- `return_results=True`: around `N=11`

## Chart Compatibility Changes

The staged Photon OD image did not reconcile correctly through the old chart
shape. The compatibility changes in this branch are:

- per-NIM `nimOperator.<key>.invokePath`, used by automatic in-cluster endpoint
  wiring
- optional per-NIM NIMCache bypass through
  `nimOperator.<key>.storage.nimCache.enabled=false`
- direct NIMService storage through `nimOperator.<key>.storage.service`
- optional NIMCache `modelPuller` and `modelEndpoint` overrides for Universal
  cache flows

The staged Photon OD image should use direct NIMService storage, not NIMCache.
The NIMCache path failed because this staged image does not include the
`create-model-store` entrypoint expected by the cache controller.

The OD routes are not `/v1/infer`:

- page elements: `/v1/page-elements`
- table structure: `/v1/table-structure`

Direct smoke probes against both routes returned HTTP 200 on this machine.

## Images

Use these staged images:

- page elements:
  `nvcr.io/nvstaging/nim/retriever-photon-od:2.0.0-rc.20260622003041-3decdfe158f0fe61`
- table structure:
  `nvcr.io/nvstaging/nim/retriever-photon-od:2.0.0-rc.20260622003041-3decdfe158f0fe61`
- OCR:
  `nvcr.io/nvstaging/nim/retriever-photon-ocr:2.0.0-rc.20260613213714-ab767ae3f81debff`
- VLM embed:
  `nvcr.io/nvstaging/nim/retriever-photon-embed:2.0.1-rc.20260619192608-be1601da70d066ef`

Required OD env:

- page elements:
  `NIM_ENGINE_MODEL_NAME=nvidia/nemotron-page-elements-v3`
- table structure:
  `NIM_ENGINE_MODEL_NAME=nvidia/nemotron-table-structure-v1`

Expected service URLs after rendering:

- `http://nemotron-page-elements-v3:8000/v1/page-elements`
- `http://nemotron-table-structure-v1:8000/v1/table-structure`
- `http://nemotron-ocr-v1:8000/v1/infer`
- `http://llama-nemotron-embed-vl-1b-v2:8000/v1/embeddings`

## Registry Auth

Use `~/.env:NGC_NVSTAGING_CATALOG_REGISTRY` for both the image pull secret and
the NIM auth secret. Do not print the value.

Example secret setup:

```bash
cd /localhome/charlesb/dev/NeMo-Retriever
set -a
source ~/.env
set +a
kubectl create namespace nemo-retriever-bo20-od
kubectl create secret docker-registry ngc-secret \
  --namespace nemo-retriever-bo20-od \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_NVSTAGING_CATALOG_REGISTRY}"
kubectl create secret generic ngc-api \
  --namespace nemo-retriever-bo20-od \
  --from-literal=NGC_API_KEY="${NGC_NVSTAGING_CATALOG_REGISTRY}"
```

If rerunning in an existing namespace, use `kubectl delete secret ...` first or
switch the `kubectl create secret` commands to a local apply flow.

## Suggested Sweep Command

Run from the package directory so the local `uv` environment and chart path
resolve consistently:

```bash
cd /localhome/charlesb/dev/NeMo-Retriever/nemo_retriever
uv run retriever-harness bo20-concurrency \
  --dataset-dir /localhome/charlesb/datasets/bo20 \
  --expected-pdfs 20 \
  --max-n 16 \
  --job-max-concurrency 8 \
  --return-results-mode both \
  --clean-page-elements-rerun \
  --nim-backend local \
  --helm-release nemo-retriever-bo20-od \
  --helm-namespace nemo-retriever-bo20-od \
  --helm-chart ./helm \
  --helm-timeout 1800 \
  --readiness-timeout 1800 \
  --idle-timeout-s 900 \
  --allow-non-main \
  --helm-set topology.mode=split \
  --helm-set serviceMonitor.enabled=false \
  --helm-set serviceMonitor.autoEnableInSplitMode=false \
  --helm-set prometheus-adapter.enabled=false \
  --helm-set autoscaling.metricType=cpu \
  --helm-set autoscaling.queueDepth.backend=cpu \
  --helm-set serviceConfig.vectordb.enabled=false \
  --helm-set serviceConfig.retrieverResults.enabled=false \
  --helm-set retrieverResults.enabled=false \
  --helm-set persistence.enabled=false \
  --helm-set batch.gpu.enabled=false \
  --helm-set topology.batch.gpu.enabled=false \
  --helm-set nimOperator.rerankqa.enabled=false \
  --helm-set nimOperator.nemotron_parse.enabled=false \
  --helm-set nimOperator.audio.enabled=false \
  --helm-set nimOperator.nimCache.keepOnUninstall=false \
  --helm-set nimOperator.page_elements.image.repository=nvcr.io/nvstaging/nim/retriever-photon-od \
  --helm-set nimOperator.page_elements.image.tag=2.0.0-rc.20260622003041-3decdfe158f0fe61 \
  --helm-set nimOperator.page_elements.invokePath=/v1/page-elements \
  --helm-set nimOperator.page_elements.storage.nimCache.enabled=false \
  --helm-set nimOperator.page_elements.storage.service.emptyDir.sizeLimit=25Gi \
  --helm-set 'nimOperator.page_elements.env=[{"name":"NIM_ENGINE_MODEL_NAME","value":"nvidia/nemotron-page-elements-v3"}]' \
  --helm-set nimOperator.table_structure.image.repository=nvcr.io/nvstaging/nim/retriever-photon-od \
  --helm-set nimOperator.table_structure.image.tag=2.0.0-rc.20260622003041-3decdfe158f0fe61 \
  --helm-set nimOperator.table_structure.invokePath=/v1/table-structure \
  --helm-set nimOperator.table_structure.storage.nimCache.enabled=false \
  --helm-set nimOperator.table_structure.storage.service.emptyDir.sizeLimit=25Gi \
  --helm-set 'nimOperator.table_structure.env=[{"name":"NIM_ENGINE_MODEL_NAME","value":"nvidia/nemotron-table-structure-v1"}]' \
  --helm-set nimOperator.ocr.image.repository=nvcr.io/nvstaging/nim/retriever-photon-ocr \
  --helm-set nimOperator.ocr.image.tag=2.0.0-rc.20260613213714-ab767ae3f81debff \
  --helm-set nimOperator.vlm_embed.image.repository=nvcr.io/nvstaging/nim/retriever-photon-embed \
  --helm-set nimOperator.vlm_embed.image.tag=2.0.1-rc.20260619192608-be1601da70d066ef
```

The harness should run two clean phases from fresh deploys:

- Phase A: `return_results=False`
- Phase B: `return_results=True`

For each phase it sweeps `N=1..16` with per-job
`max_concurrency=8`, stops after a hard failure, and reruns `N-1` and `N` once.
It waits for terminal jobs and idle queues between runs.

## Preflight Checklist

Before trusting a sweep:

- confirm `/localhome/charlesb/datasets/bo20` has exactly 20 PDFs
- confirm the inventory reports 496 pages
- confirm rendered NIMService images/tags match the staged Photon images
- confirm page and table NIMServices have the expected `NIM_ENGINE_MODEL_NAME`
- confirm page/table/OCR/embed pods are Ready with zero restarts before `N=1`
- run direct OD smoke probes if convenient:
  - `/v1/page-elements`
  - `/v1/table-structure`
- require all-NIM bo20 `N=1`, `return_results=False` success before the sweeps

## What Passed Here

All-NIM `N=1`, `return_results=False` confirmation passed on 2026-06-23:

- artifact:
  `nemo_retriever/artifacts/bo20_page_elements_recovery_20260623_150944_UTC/report.md`
- dataset: 20 PDFs, 496 pages
- health smoke: 20/20 docs completed, 0 failures, p95 70.013s, 7.073 pages/sec
- primary `N=1`: 20/20 docs completed, 0 failures, p95 56.687s, 8.734 pages/sec
- 0 retries/429s
- 0 gateway/worker 4xx/5xx observed
- 0 pod/NIM restarts or OOMs observed

Full sweep attempts were interrupted by host resets, not by captured Photon OD
service failures:

- Attempt 1, `20260623_153209_UTC`: Phase A health smoke, `N=1`, and `N=2`
  passed; `N=3` was in progress when the host reset.
- Attempt 2, `20260623_155541_UTC`: same progression, then the host reset.
  Stdout log:
  `nemo_retriever/artifacts/bo20_full_sweep_attempt2_stdout.log`
- Resume attempt from Phase A `N=3`: `N=3`, `N=4`, `N=5`, `N=6`, and
  `N=7` completed cleanly with 0 failures. `N=8` was in progress with 0 visible
  failures when the host reset. Stdout log:
  `nemo_retriever/artifacts/bo20_resume_from_n3_stdout.log`

No Phase B `return_results=True` sweep completed on this host.

## Host Health Caveat

Do not use `a4u8g-0114` for the qualification result. The host repeatedly reset
during clean sweeps. The kernel journal reported:

```text
Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event
```

After the last reset, Kubernetes showed app pods restarted and old NIM pods in
`UnexpectedAdmissionError` before replacements came up. The harness process was
gone after reboot, and the resume run did not produce structured artifacts.

## Attribution Rules

Use these rules when reviewing the next sweep:

- page/table 500s or timeouts: Photon OD stage failure
- OCR 500s or timeouts: Photon OCR stage failure
- all NIMs finish but client hangs while `return_results=True`: result
  materialization or proxy-client pressure
- service remains healthy while jobs fail: classify as per-document/job failure,
  not service crash

## Recommended Next Step

Rerun the full command above on a healthy machine. For final evidence, prefer a
clean `N=1..16` run for both `return_results` modes. If time is constrained, the
latest partial evidence suggests resuming Phase A at `N=8` is reasonable, but a
clean full sweep is the qualification-quality result.
