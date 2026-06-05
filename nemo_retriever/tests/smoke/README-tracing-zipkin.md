# Manual tracing and Zipkin smoke test

This smoke test is manual and CI-optional because it requires a Kubernetes
cluster, Helm, reachable NIM images, and enough GPU capacity for the enabled NIMs.

## Steps

1. Install or upgrade the chart with OTel and Zipkin enabled:

   ```bash
   helm upgrade --install tracing-smoke nemo_retriever/helm \
     --set topology.otel.enabled=true \
     --set topology.zipkin.enabled=true
   ```

2. Port-forward the retriever service and create a one-document job:

   ```bash
   kubectl port-forward svc/tracing-smoke-nemo-retriever 7670:80

   curl -s -D headers.txt -o job.json \
     -X POST http://localhost:7670/v1/ingest/job \
     -H 'content-type: application/json' \
     -d '{"expected_documents":1,"retain_results":false}'

   JOB_ID=$(jq -r .job_id job.json)
   TRACE_ID=$(jq -r .trace_id job.json)
   grep -i x-trace-id headers.txt
   ```

3. Submit one document under that job:

   ```bash
   curl -s -X POST "http://localhost:7670/v1/ingest/job/${JOB_ID}/document" \
     -F 'metadata={}' \
     -F 'file=@/path/to/one-document.pdf;type=application/pdf'
   ```

4. Wait for the job to finish, then port-forward Zipkin:

   ```bash
   kubectl port-forward svc/tracing-smoke-nemo-retriever-zipkin 9411:9411
   ```

5. Query Zipkin for the returned trace id:

   ```bash
   curl "http://localhost:9411/api/v2/trace/${TRACE_ID}"
   ```

6. Verify the trace includes spans named `ingest.job`, `ingest.document.accept`,
   either `pool.realtime.process` or `pool.batch.process`, `pipeline.ingest`, and
   any NIM/Triton spans emitted by the enabled NIM images.

If `TRACE_ID` is empty, confirm `topology.otel.enabled=true` and that the service
pod has the rendered OTel environment. If Zipkin returns an empty trace, wait a
few seconds for collector export and retry the `/api/v2/trace/${TRACE_ID}` query.
