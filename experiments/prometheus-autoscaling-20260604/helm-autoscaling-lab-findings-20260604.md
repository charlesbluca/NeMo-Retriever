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

## Low-Hanging Follow-Ups To Test

1. Add a rejected-work metric, for example `nemo_retriever_pool_enqueue_rejected_total{pool,reason}`, and drive scale-up from its short-window rate.
2. Add queue wait time, for example `nemo_retriever_pool_queue_wait_duration_seconds`, because current processing p95 excludes time spent waiting in the queue.
3. Replace or augment `avg by(pool)` with `max by(pool)` or a high percentile over per-pod queue ratios.
4. Revisit HPA target type: the adapter already returns a pool-level aggregate, but the HPA uses `AverageValue`, which treats the metric as a per-pod average target.
5. Add remote-stage metrics such as `nemo_retriever_remote_request_duration_seconds{stage,status}` and `nemo_retriever_remote_errors_total{stage,status}` so scaling logic can avoid treating remote 403/429/5xx as local capacity shortage.
6. Fix the adapter documentation/templates so `rules.existing` works without a hand-made `config.yaml` ConfigMap and dummy `rules.external` entry.
