# Disposable Autoscaling Lab Notes

This branch carries a pragmatic autoscaling lab for gathering evidence against the current Helm chart. It is intended for internal iteration, not as a stable customer-facing harness feature.

## Running The Lab

From the repository root:

```bash
uv --project nemo_retriever run python -m nemo_retriever.harness.autoscaling_lab \
  --config nemo_retriever/harness/autoscaling-lab.yaml \
  --dataset bo20
```

Use `--dry-run` first to confirm the selected dataset files, rendered Helm values, and artifact directory. The default config writes artifacts under `nemo_retriever/artifacts/autoscaling-lab/`.

## Auth Caveats

The Helm chart expects two NGC-related secrets in the release namespace:

- `ngc-secret`: `kubernetes.io/dockerconfigjson` for `nvcr.io` image pulls.
- `ngc-api`: opaque secret with `NGC_API_KEY` and `NGC_CLI_API_KEY`.

For the live smoke run, these already existed in `nrl-super49b`, so the safer path was copying the shaped Kubernetes secrets into `nrl-autoscale-lab` instead of passing raw key material on the command line. Avoid printing keys from `/home/charlesb/.env`; only inspect key names when confirming local auth material exists.

Example secret copy pattern:

```bash
kubectl create namespace nrl-autoscale-lab --dry-run=client -o yaml | kubectl apply -f -
for secret in ngc-secret ngc-api; do
  kubectl get secret "$secret" -n nrl-super49b -o json \
    | jq 'del(.metadata.annotations, .metadata.creationTimestamp, .metadata.resourceVersion, .metadata.uid, .metadata.managedFields, .metadata.ownerReferences) | .metadata.namespace = "nrl-autoscale-lab"' \
    | kubectl apply -f -
done
```

## GPU And NIM Caveats

The disposable config currently avoids provisioning in-chart NIMs:

- `nims.enabled: false`
- all core `nimOperator.*.enabled: false`
- external NVIDIA API endpoints under `serviceConfig.nimEndpoints.*`
- `serviceConfig.vectordb.enabled: false`

This keeps the lab focused on gateway and worker-pool autoscaling behavior. If the lab is changed to provision the four core NIMs in-cluster, the chart defaults each NIMService to one GPU allocation. On the observed A100 x3 host, that requires either fewer enabled NIMs, MIG, or time-slicing before the full stack can schedule.

The batch worker GPU request is explicitly disabled in this lab config via `topology.batch.gpu.enabled: false`; otherwise the chart default asks batch for one GPU and can block scheduling on a busy or unpartitioned node.

## Smoke Result

The `bo20` smoke run was already enough to reproduce the failure mode, so `bo767` should wait until the smoke path completes cleanly or until a branch specifically needs higher-pressure evidence.

Observed with 1 realtime worker, 1 batch worker, and queue size 4:

- `c8`: 13 accepted uploads, 7 retryable 429s, 7 terminal 409s after the job finalized failed.
- `c16`: 12 accepted uploads, 8 retryable 429s, then terminal 409/transport failures.
- Gateway logs showed both realtime and batch backends returning `pipeline is at capacity`.
- HPA remained at one replica; sampled queue metrics stayed at zero or unknown and latency remained below the configured target before dropping back to zero.

This is useful baseline evidence for iterating on autoscaling metrics, gateway routing, and queue ownership.
