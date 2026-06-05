# Tracing and Zipkin Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Helm-deployed tracing parity with the older 26.1 non-NRL chart so an ingest job returns a Zipkin-compatible trace id and Zipkin contains useful service, queue, pipeline, and NIM spans.

**Architecture:** Keep the current chart-owned OpenTelemetry Collector as the telemetry hub, add chart-owned Zipkin by default, configure service pods and all chart-managed NIMServices to emit OTLP to the collector, and add explicit OpenTelemetry spans plus W3C context propagation through FastAPI, gateway forwarding, queues, worker processes, and remote NIM clients.

**Tech Stack:** Helm templates, Kubernetes `Deployment` and `Service`, OpenTelemetry Python SDK with OTLP gRPC exporter, FastAPI, `httpx`, `requests`, current `pytest` and Helm render tests.

---

## Preflight

- [ ] Confirm the branch and baseline state.

  ```bash
  git status --short --branch
  ```

  Expected output starts with:

  ```text
  ## codex/tracing-zipkin-parity-spec
  ```

- [ ] Read the approved spec before touching code:

  ```bash
  sed -n '1,260p' docs/superpowers/specs/2026-06-05-tracing-zipkin-parity-design.md
  ```

## Phase 1: Helm Render Parity

- [ ] Add a new Helm regression test file.

  Create `nemo_retriever/tests/test_helm_tracing_zipkin.py`.

  Use the existing `helm template` style from `test_helm_nimservice_resources.py`, but parse YAML documents so the tests can assert resource names and env blocks precisely.

  ```python
  from __future__ import annotations

  import shutil
  import subprocess
  from pathlib import Path
  from unittest import SkipTest

  import yaml


  def _repo_root() -> Path:
      return Path(__file__).resolve().parents[2]


  def _helm_template(extra_sets: list[str] | None = None) -> list[dict]:
      helm = shutil.which("helm")
      if helm is None:
          raise SkipTest("`helm` binary not available in this environment.")
      chart_path = _repo_root() / "nemo_retriever/helm"
      cmd = [
          helm,
          "template",
          "tracing-regression",
          str(chart_path),
          "--set",
          "ngcImagePullSecret.create=false",
          "--set",
          "ngcApiSecret.create=false",
          "--set",
          "nimOperator.rerankqa.enabled=true",
          "--set",
          "nimOperator.audio.enabled=true",
          "--set",
          "nimOperator.nemotron_parse.enabled=true",
          "--set",
          "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
          "--api-versions",
          "apps.nvidia.com/v1alpha1",
      ]
      for flag in extra_sets or []:
          cmd.extend(["--set", flag])
      proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
      if proc.returncode != 0:
          raise AssertionError(f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
      return [doc for doc in yaml.safe_load_all(proc.stdout) if doc]


  def _resource(docs: list[dict], kind: str, name: str) -> dict:
      for doc in docs:
          if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
              return doc
      raise AssertionError(f"Missing {kind}/{name}")


  def _env_by_name(container: dict) -> dict[str, dict]:
      return {entry["name"]: entry for entry in container.get("env", [])}
  ```

  Add these tests:

  - `test_default_renders_zipkin_deployment_and_service`
  - `test_zipkin_disabled_omits_zipkin_resources`
  - `test_otel_config_exports_traces_to_rendered_zipkin_endpoint`
  - `test_standalone_service_gets_otel_env_without_duplicate_user_overrides`
  - `test_split_roles_get_otel_env`
  - `test_all_enabled_nimservices_inherit_otel_env`
  - `test_per_nim_otel_opt_out_and_override`

  Key expected names for the default release are:

  ```text
  tracing-regression-nemo-retriever-zipkin
  tracing-regression-nemo-retriever-otel
  tracing-regression-nemo-retriever-otel-config
  ```

  Run the new test before implementation:

  ```bash
  python -m pytest nemo_retriever/tests/test_helm_tracing_zipkin.py
  ```

  Expected now: failures for missing Zipkin resources, missing Zipkin exporter, missing service OTel env, and missing NIM OTel env. If Helm is unavailable, tests may skip.

- [ ] Add Zipkin values, helpers, and templates.

  Edit `nemo_retriever/helm/values.yaml`.

  Add under `topology` next to `otel`:

  ```yaml
  zipkin:
    enabled: true
    image:
      repository: openzipkin/zipkin
      tag: "3.5.0"
      pullPolicy: IfNotPresent
    port: 9411
    javaOpts: "-Xms128m -Xmx512m -XX:+ExitOnOutOfMemoryError"
    resources:
      requests:
        cpu: "250m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
    nodeSelector: {}
    tolerations: []
    affinity: {}
    exporter:
      enabled: true
      endpoint: ""
  ```

  Add to `nemo_retriever/helm/templates/_helpers.tpl`:

  ```gotemplate
  {{- define "nemo-retriever.zipkin.fullname" -}}
  {{- printf "%s-zipkin" (include "nemo-retriever.fullname" .) | trunc 63 | trimSuffix "-" -}}
  {{- end -}}

  {{- define "nemo-retriever.otel.fullname" -}}
  {{- printf "%s-otel" (include "nemo-retriever.fullname" .) | trunc 63 | trimSuffix "-" -}}
  {{- end -}}

  {{- define "nemo-retriever.zipkin.endpoint" -}}
  {{- if .Values.topology.zipkin.exporter.endpoint -}}
  {{- tpl .Values.topology.zipkin.exporter.endpoint . -}}
  {{- else -}}
  {{- printf "http://%s:%v/api/v2/spans" (include "nemo-retriever.zipkin.fullname" .) .Values.topology.zipkin.port -}}
  {{- end -}}
  {{- end -}}
  ```

  Add `nemo_retriever/helm/templates/deployment-zipkin.yaml`:

  ```gotemplate
  {{- if and .Values.topology.otel.enabled .Values.topology.zipkin.enabled }}
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: {{ include "nemo-retriever.zipkin.fullname" . }}
    labels:
      {{- include "nemo-retriever.role.labels" (dict "context" $ "role" "zipkin") | nindent 4 }}
  spec:
    replicas: 1
    selector:
      matchLabels:
        {{- include "nemo-retriever.role.selectorLabels" (dict "context" $ "role" "zipkin") | nindent 6 }}
    template:
      metadata:
        labels:
          {{- include "nemo-retriever.role.selectorLabels" (dict "context" $ "role" "zipkin") | nindent 8 }}
      spec:
        {{- include "nemo-retriever.imagePullSecrets" . | nindent 6 }}
        {{- with .Values.topology.zipkin.nodeSelector }}
        nodeSelector:
          {{- toYaml . | nindent 8 }}
        {{- end }}
        {{- with .Values.topology.zipkin.tolerations }}
        tolerations:
          {{- toYaml . | nindent 8 }}
        {{- end }}
        {{- with .Values.topology.zipkin.affinity }}
        affinity:
          {{- toYaml . | nindent 8 }}
        {{- end }}
        containers:
          - name: zipkin
            image: "{{ .Values.topology.zipkin.image.repository }}:{{ .Values.topology.zipkin.image.tag }}"
            imagePullPolicy: {{ .Values.topology.zipkin.image.pullPolicy }}
            ports:
              - name: http
                containerPort: {{ .Values.topology.zipkin.port }}
                protocol: TCP
            env:
              - name: JAVA_OPTS
                value: {{ .Values.topology.zipkin.javaOpts | quote }}
            resources:
              {{- toYaml .Values.topology.zipkin.resources | nindent 14 }}
  {{- end }}
  ```

  Add `nemo_retriever/helm/templates/service-zipkin.yaml`:

  ```gotemplate
  {{- if and .Values.topology.otel.enabled .Values.topology.zipkin.enabled }}
  apiVersion: v1
  kind: Service
  metadata:
    name: {{ include "nemo-retriever.zipkin.fullname" . }}
    labels:
      {{- include "nemo-retriever.role.labels" (dict "context" $ "role" "zipkin") | nindent 4 }}
  spec:
    type: ClusterIP
    selector:
      {{- include "nemo-retriever.role.selectorLabels" (dict "context" $ "role" "zipkin") | nindent 6 }}
    ports:
      - name: http
        port: {{ .Values.topology.zipkin.port }}
        targetPort: http
        protocol: TCP
  {{- end }}
  ```

- [ ] Inject the Zipkin exporter into the collector config.

  Edit `nemo_retriever/helm/templates/configmap-otel.yaml`.

  Replace the direct `toYaml .Values.topology.otel.config` render with a helper-driven render that adds a Zipkin exporter only when `topology.zipkin.exporter.enabled=true`.

  Add this helper to `_helpers.tpl`:

  ```gotemplate
  {{- define "nemo-retriever.otel.config" -}}
  {{- $config := deepCopy .Values.topology.otel.config -}}
  {{- if .Values.topology.zipkin.exporter.enabled -}}
  {{- $exporters := get $config "exporters" | default dict -}}
  {{- $_ := set $exporters "zipkin" (dict "endpoint" (include "nemo-retriever.zipkin.endpoint" .)) -}}
  {{- $_ := set $config "exporters" $exporters -}}
  {{- $service := get $config "service" | default dict -}}
  {{- $pipelines := get $service "pipelines" | default dict -}}
  {{- $traces := get $pipelines "traces" | default dict -}}
  {{- $traceExporters := get $traces "exporters" | default (list) -}}
  {{- if not (has "zipkin" $traceExporters) -}}
  {{- $traceExporters = append $traceExporters "zipkin" -}}
  {{- end -}}
  {{- $_ := set $traces "exporters" $traceExporters -}}
  {{- $_ := set $pipelines "traces" $traces -}}
  {{- $_ := set $service "pipelines" $pipelines -}}
  {{- $_ := set $config "service" $service -}}
  {{- end -}}
  {{- tpl (toYaml $config) . -}}
  {{- end -}}
  ```

  Then `configmap-otel.yaml` should render:

  ```gotemplate
  data:
    config.yaml: |
      {{- include "nemo-retriever.otel.config" . | nindent 4 }}
  ```

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_helm_tracing_zipkin.py -k "zipkin or otel_config"
  ```

  Expected after this step: Zipkin resource and collector config tests pass.

- [ ] Add service OTel env defaults and wire them into standalone and split deployments.

  Edit `nemo_retriever/helm/values.yaml` under `service`:

  ```yaml
  otel:
    enabled: true
    serviceName: nemo-retriever-service
    env: {}
  ```

  Add to `_helpers.tpl`:

  ```gotemplate
  {{- define "nemo-retriever.serviceOtelEnv" -}}
  {{- $root := .context -}}
  {{- $role := .role | default $root.Values.topology.mode -}}
  {{- if and $root.Values.topology.otel.enabled $root.Values.service.otel.enabled -}}
  {{- $seen := dict -}}
  {{- range $env := $root.Values.service.env -}}
  {{- if hasKey $env "name" -}}{{- $_ := set $seen $env.name true -}}{{- end -}}
  {{- end -}}
  {{- $endpoint := printf "http://%s:%v" (include "nemo-retriever.otel.fullname" $root) $root.Values.topology.otel.ports.otlpGrpc -}}
  {{- $defaults := dict
      "OTEL_EXPORTER_OTLP_ENDPOINT" $endpoint
      "OTEL_SERVICE_NAME" $root.Values.service.otel.serviceName
      "OTEL_TRACES_EXPORTER" "otlp"
      "OTEL_METRICS_EXPORTER" "otlp"
      "OTEL_LOGS_EXPORTER" "none"
      "OTEL_PROPAGATORS" "tracecontext,baggage"
      "OTEL_RESOURCE_ATTRIBUTES" (printf "service.namespace=nemo-retriever,service.role=%s" $role)
      "OTEL_PYTHON_EXCLUDED_URLS" "health"
  -}}
  {{- $merged := mergeOverwrite $defaults ($root.Values.service.otel.env | default dict) -}}
  {{- range $name, $value := $merged -}}
  {{- if not (hasKey $seen $name) }}
  - name: {{ $name }}
    value: {{ $value | quote }}
  {{- end -}}
  {{- end -}}
  {{- end -}}
  {{- end -}}
  ```

  Edit `nemo_retriever/helm/templates/deployment.yaml`.

  In the standalone env block, add the helper before `service.env`:

  ```gotemplate
  {{- include "nemo-retriever.serviceOtelEnv" (dict "context" $ "role" "standalone") | nindent 12 }}
  ```

  In the split env block, add:

  ```gotemplate
  {{- include "nemo-retriever.serviceOtelEnv" (dict "context" $ "role" $role) | nindent 12 }}
  ```

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_helm_tracing_zipkin.py -k "service_gets_otel or split_roles"
  ```

  Expected after this step: service env tests pass and no env list contains duplicate `OTEL_SERVICE_NAME` when the user overrides it in `service.env`.

- [ ] Add generic NIM OTel defaults and wire every NIMService template.

  Edit `nemo_retriever/helm/values.yaml` under `nimOperator`:

  ```yaml
  otel:
    enabled: true
    endpoint: ""
    tritonPath: "/v1/traces"
    env:
      NIM_ENABLE_OTEL: "true"
      NIM_OTEL_TRACES_EXPORTER: "otlp"
      NIM_OTEL_METRICS_EXPORTER: "console"
      TRITON_OTEL_RATE: "1"
  ```

  Add this per-NIM block to all eight NIM value blocks:

  ```yaml
  otel:
    enabled: null
    serviceName: ""
    env: {}
  ```

  Add to `_helpers.tpl`:

  ```gotemplate
  {{- define "nemo-retriever.nimServiceOtelEnv" -}}
  {{- $root := .context -}}
  {{- $key := .key -}}
  {{- $nimName := .name -}}
  {{- $nim := index $root.Values.nimOperator $key -}}
  {{- $per := $nim.otel | default dict -}}
  {{- $enabled := $root.Values.nimOperator.otel.enabled -}}
  {{- if and (hasKey $per "enabled") (ne $per.enabled nil) -}}
  {{- $enabled = $per.enabled -}}
  {{- end -}}
  {{- if and $root.Values.topology.otel.enabled $enabled -}}
  {{- $seen := dict -}}
  {{- range $env := $nim.env -}}
  {{- if hasKey $env "name" -}}{{- $_ := set $seen $env.name true -}}{{- end -}}
  {{- end -}}
  {{- $endpoint := $root.Values.nimOperator.otel.endpoint -}}
  {{- if not $endpoint -}}
  {{- $endpoint = printf "http://%s:%v" (include "nemo-retriever.otel.fullname" $root) $root.Values.topology.otel.ports.otlpHttp -}}
  {{- end -}}
  {{- $serviceName := default $nimName $per.serviceName -}}
  {{- $defaults := dict
      "NIM_ENABLE_OTEL" "true"
      "NIM_OTEL_SERVICE_NAME" $serviceName
      "NIM_OTEL_TRACES_EXPORTER" "otlp"
      "NIM_OTEL_METRICS_EXPORTER" "console"
      "NIM_OTEL_EXPORTER_OTLP_ENDPOINT" $endpoint
      "TRITON_OTEL_URL" (printf "%s%s" $endpoint $root.Values.nimOperator.otel.tritonPath)
      "TRITON_OTEL_RATE" "1"
  -}}
  {{- $merged := mergeOverwrite $defaults ($root.Values.nimOperator.otel.env | default dict) ($per.env | default dict) -}}
  {{- range $name, $value := $merged -}}
  {{- if not (hasKey $seen $name) }}
  - name: {{ $name }}
    value: {{ $value | quote }}
  {{- end -}}
  {{- end -}}
  {{- end -}}
  {{- end -}}
  ```

  Edit each NIM template under `nemo_retriever/helm/templates/nims/`.

  Immediately below each `env:` line, add the helper before the existing per-NIM env list:

  ```gotemplate
  {{- include "nemo-retriever.nimServiceOtelEnv" (dict "context" $ "key" "page_elements" "name" "nemotron-page-elements-v3") | nindent 4 }}
  ```

  Use the correct key and rendered NIMService name in each file:

  ```text
  audio.yaml -> key audio, name parakeet-1-1b-ctc-en-us or the template's metadata.name
  llama-nemotron-embed-vl-1b-v2.yaml -> key vlm_embed, name $name
  llama-nemotron-rerank-vl-1b-v2.yaml -> key rerankqa, name llama-nemotron-rerank-vl-1b-v2
  nemotron-3-nano-omni-30b-a3b-reasoning.yaml -> key nemotron_3_nano_omni_30b_a3b_reasoning, name nemotron-3-nano-omni-30b-a3b-reasoning
  nemotron-ocr-v1.yaml -> key ocr, name .Values.nimOperator.ocr.nimServiceName
  nemotron-page-elements-v3.yaml -> key page_elements, name nemotron-page-elements-v3
  nemotron-parse.yaml -> key nemotron_parse, name nemotron-parse-v1.2
  nemotron-table-structure-v1.yaml -> key table_structure, name nemotron-table-structure-v1
  ```

  If a template already binds a `$name`, pass `"name" $name`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_helm_tracing_zipkin.py -k "nimservices"
  python -m pytest nemo_retriever/tests/test_helm_nimservice_resources.py
  ```

  Expected after this step: new NIM tracing tests pass and existing NIM resource tests still pass.

## Phase 2: Python OTel Runtime

- [ ] Add direct OpenTelemetry dependencies.

  Edit `nemo_retriever/pyproject.toml` under `[project].dependencies`:

  ```toml
  "opentelemetry-api>=1.41.1",
  "opentelemetry-sdk>=1.41.1",
  "opentelemetry-exporter-otlp-proto-grpc>=1.41.1",
  ```

  Update the lockfile:

  ```bash
  uv lock --project nemo_retriever
  ```

  Expected: `nemo_retriever/uv.lock` still resolves the existing `1.41.1` OpenTelemetry packages and the project package metadata lists the new direct dependencies.

- [ ] Add tracing unit tests first.

  Create `nemo_retriever/tests/test_service_tracing.py`.

  Test contracts:

  - `configure_tracing(service_role="standalone")` returns `False` when `OTEL_TRACES_EXPORTER` is unset or not `otlp`.
  - With `OTEL_TRACES_EXPORTER=otlp` and `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317`, a span started through the helper exposes a 32-character lowercase hex trace id.
  - `inject_trace_context()` and `extract_trace_context()` round-trip a `traceparent` carrier without leaking unrelated headers.
  - `span_attributes()` or equivalent sanitizer excludes authorization, API key, token, password, and request body values.

  Keep tests isolated from the global OTel provider by adding a private test reset helper in the implementation module, for example `_reset_tracing_for_tests()`.

  Run before implementation:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_tracing.py
  ```

  Expected now: import failure for `nemo_retriever.service.tracing`.

- [ ] Implement `nemo_retriever.service.tracing`.

  Add `nemo_retriever/src/nemo_retriever/service/tracing.py`.

  Public helpers:

  ```python
  TRACE_ID_HEADER = "x-trace-id"

  def tracing_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
      raise NotImplementedError

  def configure_tracing(*, service_role: str, service_name: str | None = None) -> bool:
      raise NotImplementedError

  def get_tracer(name: str = "nemo_retriever.service"):
      raise NotImplementedError

  def start_span(
      name: str,
      *,
      kind: Any | None = None,
      context: Any | None = None,
      attributes: Mapping[str, Any] | None = None,
  ):
      raise NotImplementedError

  def current_trace_id_hex() -> str | None:
      raise NotImplementedError

  def inject_trace_context(carrier: MutableMapping[str, str] | None = None) -> dict[str, str]:
      raise NotImplementedError

  def extract_trace_context(carrier: Mapping[str, str] | None) -> Any:
      raise NotImplementedError

  def force_flush(timeout_millis: int = 1000) -> None:
      raise NotImplementedError
  ```

  Implementation requirements:

  - Use `OTLPSpanExporter` from `opentelemetry.exporter.otlp.proto.grpc.trace_exporter`.
  - Use `BatchSpanProcessor`.
  - Use `Resource.create({"service.name": resolved_service_name, "service.role": service_role})`.
  - Catch exporter/provider setup exceptions, log a warning, and return `False`.
  - Make `configure_tracing()` idempotent for normal runtime.
  - Keep `_reset_tracing_for_tests()` private and only for tests.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_tracing.py
  ```

  Expected after this step: all tests in the file pass.

- [ ] Initialize tracing from the FastAPI app.

  Edit `nemo_retriever/src/nemo_retriever/service/app.py`.

  In `create_app(config)`, after logging and resource setup, add:

  ```python
  from nemo_retriever.service.tracing import configure_tracing

  configure_tracing(service_role=config.mode)
  ```

  Add a test to `test_service_tracing.py` that monkeypatches `configure_tracing`, calls `create_app(ServiceConfig(mode="gateway"))`, and asserts it was called with `service_role="gateway"`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_tracing.py
  ```

## Phase 3: Trace ID Contract and Propagation

- [ ] Add trace fields to job models and tracker tests.

  Edit `nemo_retriever/src/nemo_retriever/service/services/job_tracker.py`.

  Add `Field` to the imports if it is not already present:

  ```python
  from pydantic import Field
  ```

  Add to `JobAggregate`:

  ```python
  trace_id: str | None = None
  trace_context: dict[str, str] = Field(default_factory=dict)
  ```

  Extend `JobTracker.register_job()`:

  ```python
  def register_job(
      self,
      job_id: str,
      *,
      expected_documents: int,
      label: str | None = None,
      metadata: dict[str, Any] | None = None,
      retain_results: bool = False,
      trace_id: str | None = None,
      trace_context: dict[str, str] | None = None,
  ) -> JobAggregate:
  ```

  Store `trace_id=trace_id` and `trace_context=dict(trace_context or {})`.

  Update `_publish_job_event()` so `job_created`, `job_started`, progress, and terminal job events include `trace_id` when present.

  Add tests to `nemo_retriever/tests/test_service_job_tracker.py`:

  ```python
  def test_register_job_persists_trace_context_and_events_include_trace_id() -> None:
      tracker, bus = _make_tracker_with_bus()
      agg = tracker.register_job(
          "j",
          expected_documents=1,
          trace_id="0" * 31 + "1",
          trace_context={"traceparent": "00-" + "0" * 31 + "1" + "-" + "0" * 15 + "2-01"},
      )
      assert agg.trace_id == "0" * 31 + "1"
      assert agg.trace_context["traceparent"].startswith("00-")
      assert bus.events[0][1]["trace_id"] == agg.trace_id
  ```

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_job_tracker.py -k trace
  ```

- [ ] Return and expose trace ids from job creation.

  Edit `nemo_retriever/src/nemo_retriever/service/models/responses.py`.

  Add `trace_id: str | None = None` to `JobCreatedResponse` and `JobAggregateResponse`.

  Edit `nemo_retriever/src/nemo_retriever/service/routers/ingest.py`.

  Update `_aggregate_to_response()` to include `trace_id=agg.trace_id`.

  Update `create_job()` signature:

  ```python
  async def create_job(request: Request, response: Response, body: JobCreateRequest) -> JobCreatedResponse:
  ```

  Wrap registration in a root span and persist the carrier:

  ```python
  from nemo_retriever.service import tracing
  from opentelemetry.trace import SpanKind

  with tracing.start_span(
      "ingest.job",
      kind=SpanKind.SERVER,
      attributes={
          "service.role": _role(request),
          "job.expected_documents": body.expected_documents,
      },
  ):
      trace_id = tracing.current_trace_id_hex()
      trace_context = tracing.inject_trace_context()
      agg = tracker.register_job(
          job_id,
          expected_documents=body.expected_documents,
          label=body.label,
          metadata=body.metadata,
          retain_results=body.retain_results,
          trace_id=trace_id,
          trace_context=trace_context,
      )
  if trace_id:
      response.headers[tracing.TRACE_ID_HEADER] = trace_id
  ```

  Add tests to `nemo_retriever/tests/test_service_ingest_router.py`:

  - With tracing enabled through the test helper, `POST /v1/ingest/job` returns `trace_id`.
  - The `x-trace-id` response header matches the body.
  - `GET /v1/ingest/job/{job_id}` returns the same `trace_id`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_ingest_router.py -k trace
  ```

- [ ] Attach document, page, and whole submissions to the job trace.

  Edit `nemo_retriever/src/nemo_retriever/service/routers/ingest.py`.

  Add helpers:

  ```python
  def _job_trace_context(job_id: str | None):
      if not job_id:
          return None
      tracker = get_job_tracker()
      agg = tracker.get_job(job_id) if tracker is not None else None
      if agg is None or not agg.trace_context:
          return None
      return tracing.extract_trace_context(agg.trace_context)


  def _request_trace_context(request: Request, *, job_id: str | None):
      if request.headers.get("traceparent"):
          return tracing.extract_trace_context(request.headers)
      return _job_trace_context(job_id)
  ```

  In these route functions, start a span early and keep queue/proxy work inside it:

  ```text
  submit_document_to_job
  submit_page_to_job
  submit_whole_document_to_job
  ```

  Use span names:

  ```text
  ingest.document.accept
  ingest.page.accept
  ingest.whole.accept
  ```

  Attributes:

  ```python
  {
      "service.role": _role(request),
      "job.id": job_id,
      "route": request.url.path,
  }
  ```

  Add `trace_id` to `IngestAccepted`, `PageIngestAccepted`, and `DocumentIngestAccepted` only if the API owners are comfortable expanding those responses. If not, keep the body unchanged and rely on job-level `trace_id`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_ingest_router.py -k "trace or ingest_with_valid_spec"
  ```

- [ ] Propagate trace context through gateway proxy calls.

  Edit `nemo_retriever/src/nemo_retriever/service/services/proxy.py`.

  Import tracing and inject W3C headers after building `fwd_headers`:

  ```python
  from nemo_retriever.service import tracing

  tracing.inject_trace_context(fwd_headers)
  ```

  Do this in both `forward()` and `forward_get()`.

  Add a test that monkeypatches `httpx.AsyncClient.request`, calls `GatewayProxy.forward()` under a current span, and asserts the forwarded headers include `traceparent`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_tracing.py -k proxy
  ```

- [ ] Carry trace context in `WorkItem` and emit queue/pool spans.

  Edit `nemo_retriever/src/nemo_retriever/service/services/pipeline_pool.py`.

  Add `Field` to the existing Pydantic import:

  ```python
  from pydantic import ConfigDict, Field
  ```

  Add fields to `WorkItem`:

  ```python
  trace_context: dict[str, str] = Field(default_factory=dict)
  enqueued_at_monotonic_s: float | None = None
  ```

  In `_enqueue_or_reject()` in `ingest.py`, before building the `WorkItem`, capture current trace context:

  ```python
  trace_context=tracing.inject_trace_context()
  ```

  In `PipelinePool.submit()`, set queue timing before enqueue:

  ```python
  item.enqueued_at_monotonic_s = time.monotonic()
  ```

  In `_Pool._worker_loop()`, wrap item handling:

  ```python
  ctx = tracing.extract_trace_context(item.trace_context)
  with tracing.start_span(
      f"pool.{self._name}.process",
      context=ctx,
      attributes={
          "pool": self._name,
          "document.id": item.id,
          "job.id": item.job_id or "",
      },
  ) as span:
      if item.enqueued_at_monotonic_s is not None:
          span.set_attribute("queue.wait_ms", (time.monotonic() - item.enqueued_at_monotonic_s) * 1000.0)
      # Keep the existing worker success, failure, callback, tracker, and
      # Prometheus bookkeeping inside this span.
  ```

  Add tests to `nemo_retriever/tests/test_service_pool_metrics.py` or a new `test_service_pool_tracing.py`:

  - Submit a `WorkItem` with a trace context.
  - Let a stub work function complete.
  - Assert exported span names include `pool.rt-trace.process` or equivalent.
  - Assert attributes include `pool`, `document.id`, and `job.id`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_pool_metrics.py nemo_retriever/tests/test_service_pool_tracing.py
  ```

## Phase 4: Pipeline and NIM Spans

- [ ] Span the child-process pipeline execution.

  Edit `nemo_retriever/src/nemo_retriever/service/services/pipeline_executor.py`.

  Extend `_run_pipeline_in_process()` args:

  ```python
  trace_context: dict[str, str] | None = None
  pool_label: str | None = None
  service_role: str | None = None
  ```

  At the top of `_run_pipeline_in_process()`:

  ```python
  from nemo_retriever.service import tracing

  tracing.configure_tracing(service_role=service_role or "worker-process")
  ctx = tracing.extract_trace_context(trace_context or {})
  with tracing.start_span(
      "pipeline.ingest",
      context=ctx,
      attributes={
          "pool": (pool_label or "").lower(),
          "document.filename": filename,
      },
  ):
      ingestor, _extraction_mode, has_per_request_vdb = _build_graph_ingestor_from_spec(
          filename,
          payload,
          extract_params_dict,
          embed_params_dict,
          pipeline_spec,
          caption_params_dict,
          asr_params_dict,
      )
      result_df = ingestor.ingest()
  tracing.force_flush(timeout_millis=500)
  ```

  In `_make_work_fn._work()`, inject the current parent context immediately before `loop.run_in_executor()` and pass it to `_run_pipeline_in_process`.

  Add `nemo_retriever/tests/test_service_pipeline_tracing.py`.

  Test by monkeypatching `_build_graph_ingestor_from_spec()` to return a fake ingestor whose `.ingest()` returns a small pandas DataFrame. Call `_run_pipeline_in_process()` directly with a trace context and assert a `pipeline.ingest` span appears.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_pipeline_tracing.py
  ```

- [ ] Add spans and header propagation for the remote HTTP NIM client.

  Edit `nemo_retriever/src/nemo_retriever/nim/nim.py`.

  In `_post_with_retries()`, copy headers, inject trace context, and wrap each attempt in a span:

  ```python
  from nemo_retriever.service import tracing

  request_headers = dict(headers)
  tracing.inject_trace_context(request_headers)
  with tracing.start_span(
      "nim.http.post",
      attributes={
          "http.method": "POST",
          "nim.endpoint": invoke_url,
          "retry.attempt": attempt,
      },
  ) as span:
      response = requests.post(invoke_url, headers=request_headers, json=payload, timeout=float(timeout_s))
      span.set_attribute("http.status_code", response.status_code)
  ```

  Do not add payload, Authorization, API keys, or bearer tokens as attributes.

  Add `nemo_retriever/tests/test_nim_tracing.py`:

  - Monkeypatch `requests.post` to capture headers and return a fake 200 response.
  - Under a current span, call `_post_with_retries()`.
  - Assert captured headers include `traceparent`.
  - Assert exported span attributes do not include `Authorization` or request body values.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_nim_tracing.py
  ```

- [ ] Add spans for the internal `NimClient` HTTP and gRPC paths.

  Edit `nemo_retriever/src/nemo_retriever/api/internal/primitives/nim/nim_client.py`.

  In `_process_batch()`, wrap protocol dispatch in:

  ```python
  with tracing.start_span(
      "nim.infer",
      attributes={
          "nim.model": model_name,
          "nim.protocol": self.protocol,
          "nim.service": self.model_interface.name(),
      },
  ):
      # Keep the existing gRPC or HTTP batch execution here.
  ```

  In `_http_infer()`, inject `traceparent` into a copy of `self.headers`.

  In `_grpc_infer()`, build `metadata_headers = tracing.inject_trace_context()` and pass it to Triton if the installed client supports `headers=metadata_headers` on `infer()`. If the client does not support headers, keep the span and add a debug log; do not fail inference.

  Add tests to `test_nim_tracing.py` with fake HTTP and fake gRPC clients. Assert spans are emitted and HTTP/gRPC headers are passed when supported.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_nim_tracing.py
  ```

## Phase 5: Client Surfaces, Docs, and Smoke

- [ ] Surface trace ids in service client results without changing `return_traces=True`.

  Edit `nemo_retriever/src/nemo_retriever/service_ingestor.py`.

  Add to `ServiceIngestResult.__init__()`:

  ```python
  self.trace_id: str | None = None
  ```

  In `ServiceIngestor.ingest()`, when handling `event_type == "job_created"`:

  ```python
  result.trace_id = evt.get("trace_id") or result.trace_id
  ```

  Keep `return_traces=True` returning raw SSE event dicts for compatibility.

  Add tests to `nemo_retriever/tests/test_service_ingest_async.py` or a focused service ingestor test that feeds a `job_created` event with `trace_id` and asserts `result.trace_id`.

  Run:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_ingest_async.py -k trace
  ```

- [ ] Document the Helm tracing workflow.

  Edit `nemo_retriever/helm/README.md`.

  Add a section named `Tracing and Zipkin`.

  Include:

  ```text
  Default install:
    topology.otel.enabled=true
    topology.zipkin.enabled=true

  Submit a job:
    curl -s -D headers.txt -o job.json -X POST http://localhost:7670/v1/ingest/job -H 'content-type: application/json' -d '{"expected_documents":1}'

  Read the trace id:
    jq -r .trace_id job.json
    grep -i x-trace-id headers.txt

  Query Zipkin:
    kubectl port-forward svc/tracing-smoke-nemo-retriever-zipkin 9411:9411
    curl "http://localhost:9411/api/v2/trace/${TRACE_ID}"
  ```

  Document opt-out and override knobs:

  ```text
  topology.zipkin.enabled=false
  topology.zipkin.exporter.enabled=false
  topology.zipkin.exporter.endpoint=http://external-zipkin:9411/api/v2/spans
  service.otel.enabled=false
  nimOperator.otel.enabled=false
  nimOperator.page_elements.otel.enabled=false
  nimOperator.ocr.otel.env.TRITON_OTEL_RATE=10
  ```

- [ ] Add a CI-optional smoke script or documented command sequence.

  Add `nemo_retriever/tests/smoke/README-tracing-zipkin.md` or a Helm README appendix. Keep it manual if cluster and NIM availability make CI unrealistic.

  Required smoke steps:

  ```text
  1. helm upgrade --install tracing-smoke nemo_retriever/helm --set topology.otel.enabled=true --set topology.zipkin.enabled=true
  2. Submit a one-document ingest job.
  3. Capture the returned trace_id.
  4. kubectl port-forward svc/tracing-smoke-nemo-retriever-zipkin 9411:9411
  5. curl "http://localhost:9411/api/v2/trace/${TRACE_ID}"
  6. Verify spans include ingest.job, ingest.document.accept, pool.realtime.process or pool.batch.process, pipeline.ingest, and any emitted NIM/Triton spans.
  ```

## Final Verification

- [ ] Run the focused Helm tests:

  ```bash
  python -m pytest nemo_retriever/tests/test_helm_tracing_zipkin.py
  python -m pytest nemo_retriever/tests/test_helm_nimservice_resources.py
  python -m pytest nemo_retriever/tests/test_helm_optional_nims_disabled_by_default.py
  ```

- [ ] Run focused service and NIM tracing tests:

  ```bash
  python -m pytest nemo_retriever/tests/test_service_tracing.py
  python -m pytest nemo_retriever/tests/test_service_ingest_router.py -k "trace or create_job"
  python -m pytest nemo_retriever/tests/test_service_job_tracker.py -k "trace or register_job"
  python -m pytest nemo_retriever/tests/test_service_pool_metrics.py
  python -m pytest nemo_retriever/tests/test_service_pipeline_tracing.py
  python -m pytest nemo_retriever/tests/test_nim_tracing.py
  python -m pytest nemo_retriever/tests/test_service_ingest_async.py -k trace
  ```

- [ ] Render the chart manually:

  ```bash
  helm template tracing-regression nemo_retriever/helm --api-versions apps.nvidia.com/v1alpha1
  ```

  Expected rendered output contains:

  ```text
  kind: Deployment
  name: tracing-regression-nemo-retriever-zipkin
  kind: Service
  name: tracing-regression-nemo-retriever-zipkin
  endpoint: http://tracing-regression-nemo-retriever-zipkin:9411/api/v2/spans
  NIM_ENABLE_OTEL
  TRITON_OTEL_URL
  OTEL_EXPORTER_OTLP_ENDPOINT
  ```

- [ ] Run formatting or linting only if the repository has an established command in current branch docs or CI config.

## Acceptance Criteria

- [ ] Default Helm install renders an OTel collector, Zipkin Deployment, Zipkin Service, service OTel env, and generic NIM/Triton OTel env for every enabled chart-managed NIMService.
- [ ] Operators can disable chart-owned Zipkin, disable Zipkin export, or point the Zipkin exporter at an external endpoint.
- [ ] Operators can opt out of tracing for an individual NIM without disabling tracing for all NIMs.
- [ ] `POST /v1/ingest/job` returns a 32-character trace id and an `x-trace-id` header when tracing is enabled.
- [ ] Gateway-to-worker calls carry W3C `traceparent`.
- [ ] Queue, pool, pipeline, and remote NIM calls emit spans with non-secret attributes.
- [ ] Existing `ServiceIngestor(return_traces=True)` behavior remains raw SSE event collection, while `ServiceIngestResult.trace_id` exposes the Zipkin lookup key.
- [ ] Ingest continues if OTel or Zipkin export is unavailable.
