# Helm Autoscaling Lab Findings - 2026-06-04

This follow-up runs the autoscaling experiment against an actual Helm release on k3s, using the chart's current Prometheus-adapter HPA methodology.

## Lab Setup

- Release: `nrl-autoscale-lab` in namespace `nrl-super49b`
- Topology: split gateway/realtime/batch
- Realtime pool: `realtimeWorkers=1`, `realtimeQueueSize=4`
- Realtime HPA: queue ratio target `300m`, processing p95 target `10`, CPU disabled
- Prometheus: `kube-prometheus-stack` in namespace `monitoring`
- Load harness: `helm_load_probe.py`

The lab values are in `helm-autoscale-lab-values.yaml`.

## Adapter Wiring Finding

The chart rendered an HPA immediately, but it stayed at `<unknown>` until an external metrics API was registered.

Two setup gaps showed up:

1. The retriever chart's adapter ConfigMap uses key `rules.yaml`, while `prometheus-community/prometheus-adapter` with `rules.existing` expects `config.yaml` at `/etc/adapter/config.yaml`.
2. The adapter chart only renders `v1beta1.external.metrics.k8s.io` when `.Values.rules.external` is non-empty. `rules.existing` alone mounts the config but does not create the APIService/RBAC.

For the lab, I created a compatible ConfigMap with `config.yaml` and upgraded the adapter with a harmless `rules.external[0]` entry to trigger APIService/RBAC creation. After that, the HPA could read:

- `nemo_retriever_pool_queue_depth_ratio_avg`
- `nemo_retriever_pool_processing_duration_p95_seconds`

## Run 1: Gateway Burst, Current HPA Shape

Artifacts: `helm-runs/gateway-current-hpa/`

Command shape:

```bash
python3 helm_load_probe.py \
  --mode gateway \
  --base-url http://127.0.0.1:7670 \
  --input input/1598224.pdf \
  --count 32 \
  --concurrency 16 \
  --run-dir helm-runs/gateway-current-hpa
```

Result:

- Responses: 5 accepted, 27 rejected with HTTP 429.
- Gateway logged all 429s at about `15:22:52.66` through `15:22:52.83` UTC.
- Queue ratio reached 1.0 and queue depth reached 4 from sample elapsed `2s` through `14s`.
- HPA did scale realtime from 1 to 4, but only after the synchronous request burst had already been rejected.
- All 5 accepted items failed downstream on remote NIM auth errors: page-elements and embeddings returned HTTP 403.

Interpretation:

Queue depth can eventually signal saturation, but it does not protect bursty clients from immediate queue-full rejection. The current metric also does not preserve rejected demand after the queue is full; once the 429s are returned, the autoscaler only sees the few accepted items draining.

## Run 2: Direct One-Pod Skew

Artifacts: `helm-runs/direct-one-pod-skew/`

For this run I patched the lab HPA to `minReplicas=4`, `maxReplicas=8`, then sent all traffic directly to one realtime pod via port-forward.

Result:

- Responses: 5 accepted, 27 rejected with HTTP 429.
- `queue_ratio_max`: 1.0
- `queue_ratio_avg`: 0.25
- `queue_depth_max`: 4
- `queue_depth_sum`: 4
- HPA stayed at 4 replicas despite scale-out headroom to 8.

Interpretation:

This directly exposes the adapter aggregation problem. The hot pod was completely full, but the chart's `avg by(pool)` expression averaged that away across idle pods. With a `300m` target, one saturated pod out of four appears as `250m`, below target. A `max by(pool)` or high-percentile per-pod queue signal would have preserved the hot shard.

## Remote-Stage Observability Gap

The actual downstream failure mode in both runs was remote NIM authorization failure. Prometheus exposed only generic pool outcomes:

- `nemo_retriever_pool_processed_total{pool="realtime",outcome="failed"}` reached 10.
- No retriever metric names contained `remote`, `nim`, `error`, or HTTP status information.

The logs identify the cause as HTTP 403 from page-elements and embeddings, but the autoscaler cannot distinguish remote auth/throttle/failure from local worker slowness. Scaling local pods in response to that kind of failure is unlikely to help and can amplify pressure on the remote dependency.

## Harness Follow-Up On Merged Upstream Helm Code

After merging `upstream/main` into `codex/prometheus-autoscaling-helm-lab`, I reran the lab against the current chart using the managed Helm service harness. The harness config is `helm-autoscale-harness.yaml`.

### Run 3: Plain k3s Bootstrap Failure

Command shape:

```bash
nemo_retriever/.venv/bin/retriever harness run \
  --config experiments/prometheus-autoscaling-20260604/helm-autoscale-harness.yaml \
  --dataset lab_pdf \
  --preset lab \
  --run-name autoscale-crd-bootstrap \
  --tag autoscale-lab
```

Result:

- The cluster had no `servicemonitors.monitoring.coreos.com` CRD.
- Helm failed before creating the service because split mode auto-rendered three `ServiceMonitor` objects: gateway, realtime, and batch.
- The relevant error was `no matches for kind "ServiceMonitor" in version "monitoring.coreos.com/v1"`.

Interpretation:

Split mode currently defaults to the Prometheus-adapter autoscaling path and auto-renders ServiceMonitors, but the chart does not gate ServiceMonitor rendering on the Prometheus Operator CRD and does not fail early with a chart-level prerequisite message. On a plain k3s cluster, the first failure is Kubernetes object admission rather than an autoscaling diagnostic.

### Run 4: Service Up, HPA Still Inert Without External Metrics

Command shape:

```bash
nemo_retriever/.venv/bin/retriever harness run \
  --config experiments/prometheus-autoscaling-20260604/helm-autoscale-harness.yaml \
  --dataset lab_pdf \
  --preset lab \
  --run-name autoscale-no-servicemonitor \
  --tag autoscale-lab \
  --keep-up \
  --helm-namespace pr2202-super49b \
  --helm-timeout 300 \
  --readiness-timeout 300 \
  --helm-set serviceMonitor.autoEnableInSplitMode=false \
  --helm-set serviceMonitor.enabled=false
```

Result:

- Gateway, realtime, and batch pods became ready.
- Harness service ingestion reached the service but failed one PDF page downstream on remote NIM HTTP 403s.
- The realtime HPA stayed at one replica with both targets `<unknown>`:
  - `<unknown>/300m (avg)` for `nemo_retriever_pool_queue_depth_ratio_avg`
  - `<unknown>/10 (avg)` for `nemo_retriever_pool_processing_duration_p95_seconds`
- `kubectl describe hpa` reported `ScalingActive=False` / `FailedGetExternalMetric` because `v1beta1.external.metrics.k8s.io` was not registered.

Interpretation:

Disabling ServiceMonitor lets the service install, but it does not create a usable autoscaling deployment. The harness considers the service ready once health checks pass and then runs workload traffic, even though the HPA cannot read any autoscaling signal. A Helm/harness preflight for `ServiceMonitor` CRD, external metrics API registration, and HPA `ScalingActive=True` would catch this before load testing.

### Run 5: Gateway Burst Without Adapter

Artifacts: `helm-runs/harness-no-adapter-gateway-burst/`

Command shape:

```bash
kubectl port-forward -n pr2202-super49b \
  service/nrl-autoscale-lab-nemo-retriever-gateway 17670:7670

python3 experiments/prometheus-autoscaling-20260604/helm_load_probe.py \
  --mode gateway \
  --base-url http://127.0.0.1:17670 \
  --prom-url http://127.0.0.1:9090 \
  --input experiments/prometheus-autoscaling-20260604/input/1598224.pdf \
  --count 16 \
  --concurrency 8 \
  --sample-interval 2 \
  --post-sample-seconds 10 \
  --label harness-no-adapter-burst \
  --run-dir experiments/prometheus-autoscaling-20260604/helm-runs/harness-no-adapter-gateway-burst
```

Result:

- Responses: 5 accepted, 11 rejected with HTTP 429.
- HPA stayed at one realtime replica because both external metrics remained `<unknown>`.
- Probe Prometheus samples were empty because no Prometheus server was available at `127.0.0.1:9090`.
- The realtime pod raw `/metrics` endpoint did expose pool metrics after the run:
  - `nemo_retriever_pool_max_queue_size{pool="realtime"} 4`
  - `nemo_retriever_pool_workers{pool="realtime"} 1`
  - `nemo_retriever_pool_processed_total{pool="realtime",outcome="failed"} 5`
- The realtime pod did not expose a rejection or queue-wait metric. The gateway exposed only generic ingest request counters, including `status="4xx"` for the job-specific page endpoint.

Interpretation:

This separates publisher availability from autoscaler delivery. The service publishes useful raw pool metrics, but without Prometheus plus adapter registration those metrics never reach HPA. It also sharpens the rejected-work gap: 429 demand is visible only as generic gateway 4xx request traffic with a high-cardinality endpoint label, not as a pool-scoped rejection signal the autoscaler can use.

## Functional Autoscaling Risks Beyond Adapter Bootstrap

The ServiceMonitor/external-metrics findings above are bootstrap footguns. To move past those blockers, I kept the same Helm split-mode release running and manually changed realtime replicas when needed. The external metrics API was still absent in this plain k3s lab, so the manual replica change isolates what happens after capacity exists; it is not evidence that HPA itself made the scaling decision.

For these runs I installed a deliberately slow fake NIM dependency from `slow-fake-nim.yaml` and pointed page-elements, OCR, and embedding URLs at `http://nrl-autoscale-slow-nim:8000/...`. The fake service sleeps before returning HTTP 403, which keeps worker slots occupied long enough to observe the autoscaling signals.

### Run 6: Active Worker Invisible During Slow Dependency

Artifacts: `helm-runs/functional-active-worker-invisible/`

Command shape:

```bash
python3 experiments/prometheus-autoscaling-20260604/helm_functional_probe.py \
  --base-url http://127.0.0.1:17680 \
  --metric old=http://127.0.0.1:17681 \
  --input experiments/prometheus-autoscaling-20260604/input/1598224.pdf \
  --count 1 \
  --concurrency 1 \
  --post-sample-seconds 55 \
  --run-dir experiments/prometheus-autoscaling-20260604/helm-runs/functional-active-worker-invisible
```

Result:

- The one-page request was accepted with HTTP 202.
- While the worker was blocked on the slow fake NIM call, `nemo_retriever_pool_queue_depth` stayed at `0` and `nemo_retriever_pool_queue_depth_ratio` stayed at `0`.
- `nemo_retriever_pool_processing_duration_seconds_count` and `_sum` stayed at `0` until the worker finished; the final sample recorded `processing_count=1`, `processed_failed=1`, and `processing_sum=37.207783304009354`.

Interpretation:

The queue-depth publisher counts only waiting work. A request that is already being processed has left the queue, and the latency histogram is not observed until completion. A saturated worker pool can therefore look idle to the queue metric and silent to the latency metric during the exact window where scale-out would help.

This suggests adding a live occupancy signal, for example `nemo_retriever_pool_active_workers{pool}` or `nemo_retriever_pool_inflight_work_items{pool}`, and possibly an age gauge for the oldest in-flight item.

### Run 7: Scale-Out Does Not Move Existing Pod-Local Queue

Artifacts:

- `helm-runs/functional-queue-fill-before-scaleout/`
- `helm-runs/functional-direct-old-pod-local-queue/`

Command shape:

```bash
python3 experiments/prometheus-autoscaling-20260604/helm_functional_probe.py \
  --base-url http://127.0.0.1:17681 \
  --metric old=http://127.0.0.1:17681 \
  --metric new=http://127.0.0.1:17682 \
  --input experiments/prometheus-autoscaling-20260604/input/1598224.pdf \
  --count 5 \
  --concurrency 5 \
  --run-dir experiments/prometheus-autoscaling-20260604/helm-runs/functional-direct-old-pod-local-queue
```

Result:

- Before manual scale-out, a five-page burst through the gateway filled the single realtime pod: `queue_depth=4`, `queue_ratio=1.0`, with all five requests accepted.
- After scaling realtime to two replicas, a direct burst to the old pod reproduced the hot-shard state while sampling both pod-local metrics.
- The old pod reached `queue_depth=4` and `queue_ratio=1.0`.
- The new pod remained at `queue_depth=0`, `queue_ratio=0`, `processed_failed=0`, and `processing_count=0` throughout the same sample window.

Interpretation:

Accepted work lives in each worker pod-local `asyncio.Queue`. When capacity is added, the new pods start empty and cannot steal work that is already accepted by the hot pod. Scaling therefore helps future admissions, but not the backlog that triggered scale-out.

This is separate from the earlier `avg by(pool)` skew problem: even with a better `max` metric, the remediation path is still imperfect unless routing or queue ownership changes. Queue-aware routing, a shared durable queue, or explicit retry/resubmission would be needed to make scale-out relieve existing hot-pod backlog.

### Run 8: Gateway Routing Can Reject While Another Replica Has Room

Artifacts: `helm-runs/functional-gateway-routing-skew/`

Command shape:

```bash
python3 experiments/prometheus-autoscaling-20260604/helm_functional_probe.py \
  --base-url http://127.0.0.1:17680 \
  --metric old=http://127.0.0.1:17681 \
  --metric new=http://127.0.0.1:17682 \
  --input experiments/prometheus-autoscaling-20260604/input/1598224.pdf \
  --count 12 \
  --concurrency 12 \
  --run-dir experiments/prometheus-autoscaling-20260604/helm-runs/functional-gateway-routing-skew
```

Result:

- The gateway returned 5 HTTP 202 responses and 7 HTTP 429 responses.
- At the first sample, the old pod was already hot (`queue_depth=3`, `queue_ratio=0.75`) and the new pod was empty (`queue_depth=0`, `queue_ratio=0`).
- Within the next sample, the old pod was full (`queue_depth=4`, `queue_ratio=1.0`) while the new pod had accepted work but still had queue headroom (`queue_depth=3`, `queue_ratio=0.75`).
- The 429 responses all arrived in about 0.14 seconds, during the same burst that was filling the second replica.

Interpretation:

In gateway mode, the gateway forwards realtime traffic to the Kubernetes Service for the realtime pool. Kubernetes Service load balancing is not aware of each pod-local queue depth, so a request can land on a saturated backend and receive 429 while another backend has recently been empty or still has room.

This means scale-out is not enough by itself to make admission reliable under skew. A newly ready pod can help only the requests that route to it. Queue-aware routing, retries on backend 429, or moving admission/queue ownership out of pod-local memory would make the autoscaling story much less brittle.

### Normalized Queue Ratio Hides Absolute Backlog

The lab intentionally used `realtimeQueueSize=4` and target `300m`, so scale pressure appears at roughly 1.2 queued items per pod. The chart defaults are much deeper:

- Realtime default: `realtimeQueueSize=2048`, queue target `500m`, so the target corresponds to about 1024 waiting items per pod.
- Batch default: `batchQueueSize=8192`, queue target `700m`, so the target corresponds to about 5734 waiting items per pod.

A ratio is useful for normalizing between pools, but it can make low-latency behavior look healthy while the absolute backlog is already unacceptable. Queue-wait time would help here, but even before adding that metric, the chart may need workload-class guidance that ties queue target to latency SLO and service time, not just fill percentage.

### Metric Identity And Cardinality Still Need A Multi-Release Test

The raw worker pool metrics carry `pool` but not release, namespace, pod, or component labels. Those labels may be added by Prometheus scrape metadata, but the adapter query aggregates by `pool`, and the HPA selector matches only `pool=<role>`. A multi-release test should verify that namespace/release matchers survive all the way through the External Metrics API.

The gateway request metric has the opposite problem in one path: failed forwarded requests are recorded with `request.url.path`, which can include the job id, while successful page requests are also recorded under the normalized `/v1/ingest/job/page` endpoint. That high-cardinality 4xx signal is not suitable as a rejected-demand autoscaling metric without normalization.

## Low-Hanging Follow-Ups To Test

1. Add a rejected-work metric, for example `nemo_retriever_pool_enqueue_rejected_total{pool,reason}`, and drive scale-up from its short-window rate.
2. Add queue wait time, for example `nemo_retriever_pool_queue_wait_duration_seconds`, because current processing p95 excludes time spent waiting in the queue.
3. Replace or augment `avg by(pool)` with `max by(pool)` or a high percentile over per-pod queue ratios.
4. Revisit HPA target type: the adapter already returns a pool-level aggregate, but the HPA uses `AverageValue`, which treats the metric as a per-pod average target.
5. Add remote-stage metrics such as `nemo_retriever_remote_request_duration_seconds{stage,status}` and `nemo_retriever_remote_errors_total{stage,status}` so scaling logic can avoid treating remote 403/429/5xx as local capacity shortage.
6. Fix the adapter documentation/templates so `rules.existing` works without a hand-made `config.yaml` ConfigMap and dummy `rules.external` entry.
7. Gate `ServiceMonitor` rendering on the Prometheus Operator CRD or make the prerequisite explicit enough that a plain cluster fails before Kubernetes admission.
8. Add a harness/chart preflight that reports raw `/metrics`, Prometheus target discovery, External Metrics API registration, and HPA `ScalingActive` separately.
9. Normalize ingest request metrics before using them operationally; the current gateway `endpoint` label can include job IDs, while rejected queue demand needs a low-cardinality `{pool,reason}` shape.
10. Add active-worker/in-flight metrics so slow active work can trigger scale-up before queue depth or completion latency catches up.
11. Document that scale-out does not move accepted pod-local work; HPA only helps future admissions unless routing or queue ownership changes.
12. Add a queue-aware routing or retry experiment, including keepalive-enabled and keepalive-disabled gateway variants, because Kubernetes Service routing is not queue-aware.
13. Revisit default queue-size and queue-ratio targets as latency-SLO settings, not just normalized capacity settings.
14. Run a multi-release autoscaling test to prove adapter queries are isolated by namespace/release and cannot mix `pool="realtime"` series across tenants.
