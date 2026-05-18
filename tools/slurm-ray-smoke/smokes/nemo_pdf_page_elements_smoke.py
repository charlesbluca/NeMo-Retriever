from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Any

import pandas as pd
import ray

from _common import assert_cluster_ready, ray_state, write_json
from nemo_retriever.graph import AbstractOperator, Graph, GPUOperator, Node, RayDataExecutor
from nemo_retriever.ocr.gpu_ocrv2 import OCRV2Actor as OCRV2GPUActor
from nemo_retriever.params import EmbedParams, ModelRuntimeParams
from nemo_retriever.page_elements.gpu_actor import PageElementDetectionActor as PageElementDetectionGPUActor
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.text_embed.gpu_operator import _BatchEmbedActor as BatchEmbedGPUActor
from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor
from nemo_retriever.vdb.operators import IngestVdbOperator


class HostAnnotatedPageElementDetectionGPUActor(AbstractOperator, GPUOperator):
    """Lazy local page-elements GPU actor that records the worker CUDA context."""

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)
        self.detect_kwargs = dict(detect_kwargs)
        self._delegate: PageElementDetectionGPUActor | None = None

    def _get_delegate(self) -> PageElementDetectionGPUActor:
        if self._delegate is None:
            self._delegate = PageElementDetectionGPUActor(**self.detect_kwargs)
        return self._delegate

    @staticmethod
    def _cuda_snapshot() -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        try:
            import torch

            snapshot["cuda_available"] = bool(torch.cuda.is_available())
            snapshot["cuda_device_count"] = int(torch.cuda.device_count())
            if snapshot["cuda_available"] and snapshot["cuda_device_count"]:
                snapshot["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        except BaseException as exc:
            snapshot["cuda_error"] = repr(exc)
        return snapshot

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        output = self._get_delegate()(data, **kwargs)
        if isinstance(output, pd.DataFrame):
            output = output.copy()
            snapshot = self._cuda_snapshot()
            output["page_elements_host"] = socket.gethostname()
            for key, value in snapshot.items():
                output[f"page_elements_{key}"] = value
        return output

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class HostAnnotatedOCRV2GPUActor(AbstractOperator, GPUOperator):
    """Lazy local OCR v2 GPU actor that records the worker CUDA context."""

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self.ocr_kwargs = dict(ocr_kwargs)
        self._delegate: OCRV2GPUActor | None = None

    def _get_delegate(self) -> OCRV2GPUActor:
        if self._delegate is None:
            self._delegate = OCRV2GPUActor(**self.ocr_kwargs)
        return self._delegate

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        output = self._get_delegate()(data, **kwargs)
        if isinstance(output, pd.DataFrame):
            output = output.copy()
            snapshot = HostAnnotatedPageElementDetectionGPUActor._cuda_snapshot()
            output["ocr_v2_host"] = socket.gethostname()
            for key, value in snapshot.items():
                output[f"ocr_v2_{key}"] = value
        return output

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class HostAnnotatedTextEmbedGPUActor(AbstractOperator, GPUOperator):
    """Lazy local text embedder GPU actor that records the worker CUDA context."""

    def __init__(self, params: EmbedParams) -> None:
        super().__init__(params=params)
        self.params = params
        self._delegate: BatchEmbedGPUActor | None = None

    def _get_delegate(self) -> BatchEmbedGPUActor:
        if self._delegate is None:
            self._delegate = BatchEmbedGPUActor(params=self.params)
        return self._delegate

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        output = self._get_delegate()(data)
        if isinstance(output, pd.DataFrame):
            output = output.copy()
            snapshot = HostAnnotatedPageElementDetectionGPUActor._cuda_snapshot()
            output["text_embed_host"] = socket.gethostname()
            for key, value in snapshot.items():
                output[f"text_embed_{key}"] = value
        return output

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PDF page-elements local GPU NeMo-Retriever smoke.")
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--enable-ocr-v2", action="store_true", help="Append local Nemotron OCR v2 after page-elements.")
    parser.add_argument(
        "--enable-text-embed-vdb",
        action="store_true",
        help="Append local text embedding and write a LanceDB artifact.",
    )
    parser.add_argument("--vdb-uri", default=None, help="LanceDB URI for --enable-text-embed-vdb.")
    parser.add_argument("--vdb-table", default="nv-ingest", help="LanceDB table name for --enable-text-embed-vdb.")
    return parser.parse_args()


def _build_graph(
    *,
    enable_ocr_v2: bool = False,
    enable_text_embed_vdb: bool = False,
    vdb_uri: str | None = None,
    vdb_table: str = "nv-ingest",
) -> Graph:
    graph = Graph()
    nodes = [
        Node(DocToPdfConversionActor(), name="DocToPdfConversionActor"),
        Node(PDFSplitActor(), name="PDFSplitActor"),
        Node(
            PDFExtractionActor(
                method="pdfium",
                extract_text=True,
                extract_images=False,
                extract_tables=False,
                extract_charts=False,
                extract_infographics=False,
                extract_page_as_image=True,
                render_mode="fit_to_model",
            ),
            name="PDFExtractionActor",
        ),
        Node(
            HostAnnotatedPageElementDetectionGPUActor(inference_batch_size=1),
            name="PageElementDetectionGPUActor",
        ),
    ]
    if enable_ocr_v2:
        nodes.append(
            Node(
                HostAnnotatedOCRV2GPUActor(
                    extract_text=False,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    inference_batch_size=1,
                    use_graphic_elements=False,
                    use_table_structure=False,
                ),
                name="OCRV2GPUActor",
            )
        )
    if enable_text_embed_vdb:
        if not vdb_uri:
            raise ValueError("vdb_uri is required when enable_text_embed_vdb=True")
        nodes.extend(
            [
                Node(
                    HostAnnotatedTextEmbedGPUActor(
                        params=EmbedParams(
                            model_name="nvidia/llama-nemotron-embed-1b-v2",
                            local_ingest_embed_backend="hf",
                            text_column="text",
                            inference_batch_size=1,
                            output_column="text_embeddings_1b_v2",
                            runtime=ModelRuntimeParams(max_length=512),
                        )
                    ),
                    name="TextEmbedGPUActor",
                ),
                Node(
                    IngestVdbOperator(
                        vdb_op="lancedb",
                        vdb_kwargs={
                            "uri": vdb_uri,
                            "table_name": vdb_table,
                            "overwrite": True,
                            "vector_dim": 2048,
                            "on_bad_vectors": "error",
                            "num_partitions": 2,
                        },
                    ),
                    name="IngestVdbOperator",
                ),
            ]
        )
    graph.add_chain(*nodes)
    return graph


def _node_overrides(*, enable_ocr_v2: bool = False, enable_text_embed_vdb: bool = False) -> dict[str, dict[str, Any]]:
    overrides = {
        "DocToPdfConversionActor": {
            "num_cpus": 0.25,
            "num_gpus": 0,
            "concurrency": 1,
        },
        "PDFSplitActor": {
            "num_cpus": 0.5,
            "num_gpus": 0,
            "concurrency": 1,
            "batch_size": 1,
        },
        "PDFExtractionActor": {
            "num_cpus": 1,
            "num_gpus": 0,
            "concurrency": 1,
            "batch_size": 1,
        },
        "PageElementDetectionGPUActor": {
            "resources": {"nemo_worker": 0.01},
            "num_cpus": 2,
            "num_gpus": 1,
            "concurrency": 1,
            "batch_size": 1,
        },
    }
    if enable_ocr_v2:
        overrides["OCRV2GPUActor"] = {
            "resources": {"nemo_head": 0.01},
            "num_cpus": 2,
            "num_gpus": 1,
            "concurrency": 1,
            "batch_size": 1,
        }
    if enable_text_embed_vdb:
        overrides["TextEmbedGPUActor"] = {
            "resources": {"nemo_worker": 0.01},
            "num_cpus": 4,
            "num_gpus": 1,
            "concurrency": 1,
            "batch_size": 1,
        }
        overrides["IngestVdbOperator"] = {
            "resources": {"nemo_head": 0.01},
            "num_cpus": 2,
            "num_gpus": 0,
            "concurrency": 1,
        }
    return overrides


def _payload_error(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload.get("error")
    return None


def _row_sample(df: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    columns = [
        "path",
        "page_number",
        "page_elements_v3_num_detections",
        "page_elements_v3_counts_by_label",
        "page_elements_host",
        "page_elements_cuda_available",
        "page_elements_cuda_device_count",
        "page_elements_cuda_device_name",
        "page_elements_cuda_visible_devices",
        "ocr_v1_num_detections",
        "ocr_v1_counts_by_label",
        "ocr_v2_host",
        "ocr_v2_cuda_available",
        "ocr_v2_cuda_device_count",
        "ocr_v2_cuda_device_name",
        "ocr_v2_cuda_visible_devices",
        "text_embeddings_1b_v2_dim",
        "text_embeddings_1b_v2_has_embedding",
        "text_embed_host",
        "text_embed_cuda_available",
        "text_embed_cuda_device_count",
        "text_embed_cuda_device_name",
        "text_embed_cuda_visible_devices",
    ]
    sample: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        item = {column: row.get(column) for column in columns if column in df.columns}
        page_image = row.get("page_image")
        if isinstance(page_image, dict):
            image_b64 = str(page_image.get("image_b64") or "")
            item["page_image"] = {
                "encoding": page_image.get("encoding"),
                "orig_shape_hw": page_image.get("orig_shape_hw"),
                "image_b64_len": len(image_b64),
            }
        payload = row.get("page_elements_v3")
        if isinstance(payload, dict):
            detections = payload.get("detections") or []
            item["page_elements_v3"] = {
                "num_detections": len(detections),
                "timing": payload.get("timing"),
                "error": payload.get("error"),
            }
        ocr_payload = row.get("ocr")
        if isinstance(ocr_payload, dict):
            item["ocr"] = {
                "num_detections": ocr_payload.get("num_detections"),
                "counts_by_label": ocr_payload.get("counts_by_label"),
                "timing": ocr_payload.get("timing"),
                "error": ocr_payload.get("error"),
            }
        for column in ("table", "chart", "infographic"):
            value = row.get(column)
            if isinstance(value, list):
                item[f"{column}_items"] = len(value)
        embedding_payload = row.get("text_embeddings_1b_v2")
        if isinstance(embedding_payload, dict):
            embedding = embedding_payload.get("embedding")
            item["text_embeddings_1b_v2"] = {
                "embedding_dim": len(embedding) if isinstance(embedding, list) else 0,
                "error": embedding_payload.get("error"),
                "info_msg": embedding_payload.get("info_msg"),
            }
        sample.append(item)
    return sample


def _assert_gpu_ready(state: dict[str, Any], expected_gpus: int) -> None:
    resources = state.get("cluster_resources", {})
    gpu_count = resources.get("GPU", 0)
    if expected_gpus > 0 and gpu_count < expected_gpus:
        raise AssertionError(f"expected at least {expected_gpus} Ray GPUs, found {gpu_count!r}")


def _lancedb_row_count(uri: Path, table_name: str) -> int:
    import lancedb

    table = lancedb.connect(str(uri)).open_table(table_name)
    return int(table.count_rows())


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_pdf = Path(args.input_pdf)
    if not input_pdf.is_file():
        raise FileNotFoundError(f"input PDF not found: {input_pdf}")
    vdb_uri = Path(args.vdb_uri) if args.vdb_uri else Path(args.output_json).resolve().parent / "lancedb"

    executor = RayDataExecutor(
        _build_graph(
            enable_ocr_v2=args.enable_ocr_v2,
            enable_text_embed_vdb=args.enable_text_embed_vdb,
            vdb_uri=str(vdb_uri),
            vdb_table=args.vdb_table,
        ),
        ray_address=args.ray_address,
        batch_size=1,
        num_cpus=0.25,
        num_gpus=0,
        node_overrides=_node_overrides(
            enable_ocr_v2=args.enable_ocr_v2,
            enable_text_embed_vdb=args.enable_text_embed_vdb,
        ),
    )
    result = executor.ingest(str(input_pdf))
    state = ray_state(ray)

    if not isinstance(result, pd.DataFrame):
        raise AssertionError(f"expected pandas DataFrame result, got {type(result).__name__}")
    if result.empty:
        raise AssertionError("expected non-empty result DataFrame")

    assert_cluster_ready(state, expected_nodes=args.expected_nodes)
    _assert_gpu_ready(state, expected_gpus=args.expected_gpus)

    required_columns = (
        "page_image",
        "page_elements_v3",
        "page_elements_v3_num_detections",
        "page_elements_host",
        "page_elements_cuda_available",
        "page_elements_cuda_device_count",
    )
    for column in required_columns:
        if column not in result.columns:
            raise AssertionError(f"expected result column {column!r}")

    errors = [_payload_error(payload) for payload in result["page_elements_v3"]]
    errors = [error for error in errors if error]
    if errors:
        raise AssertionError(f"page-elements inference returned errors: {errors!r}")

    cuda_available = [bool(value) for value in result["page_elements_cuda_available"].dropna().tolist()]
    if not cuda_available or not all(cuda_available):
        raise AssertionError(f"expected page-elements actor to see CUDA, got {cuda_available!r}")

    device_counts = [int(value) for value in result["page_elements_cuda_device_count"].dropna().tolist()]
    if not device_counts or max(device_counts) < 1:
        raise AssertionError(f"expected at least one CUDA device in actor, got {device_counts!r}")

    detection_counts = [int(value) for value in result["page_elements_v3_num_detections"].fillna(0).tolist()]
    total_detections = sum(detection_counts)
    if total_detections <= 0:
        raise AssertionError("expected at least one page-elements detection")

    page_elements_hosts = sorted(
        {str(host) for host in result["page_elements_host"].dropna().unique() if str(host)}
    )
    if not page_elements_hosts:
        raise AssertionError("expected at least one page_elements_host value")

    payload = {
        "status": "succeeded",
        "ray_address": args.ray_address,
        "input_pdf": str(input_pdf),
        "ocr_v2_enabled": bool(args.enable_ocr_v2),
        "text_embed_vdb_enabled": bool(args.enable_text_embed_vdb),
        "rows": int(len(result)),
        "columns": list(result.columns),
        "page_elements_hosts": page_elements_hosts,
        "detection_counts": detection_counts,
        "total_detections": int(total_detections),
        "cuda": {
            "available": sorted({bool(value) for value in result["page_elements_cuda_available"].tolist()}),
            "device_counts": sorted({int(value) for value in result["page_elements_cuda_device_count"].tolist()}),
            "device_names": sorted(
                {str(value) for value in result["page_elements_cuda_device_name"].dropna().unique() if str(value)}
            ),
            "visible_devices": sorted(
                {
                    str(value)
                    for value in result["page_elements_cuda_visible_devices"].dropna().unique()
                    if str(value)
                }
            ),
        },
        "ray": state,
        "sample": _row_sample(result),
    }
    if args.enable_ocr_v2:
        required_ocr_columns = (
            "ocr",
            "ocr_v1_num_detections",
            "ocr_v1_counts_by_label",
            "ocr_v2_host",
            "ocr_v2_cuda_available",
            "ocr_v2_cuda_device_count",
        )
        for column in required_ocr_columns:
            if column not in result.columns:
                raise AssertionError(f"expected result column {column!r}")

        ocr_errors = [_payload_error(payload) for payload in result["ocr"]]
        ocr_errors = [error for error in ocr_errors if error]
        if ocr_errors:
            raise AssertionError(f"OCR v2 inference returned errors: {ocr_errors!r}")

        ocr_cuda_available = [bool(value) for value in result["ocr_v2_cuda_available"].dropna().tolist()]
        if not ocr_cuda_available or not all(ocr_cuda_available):
            raise AssertionError(f"expected OCR v2 actor to see CUDA, got {ocr_cuda_available!r}")

        ocr_device_counts = [int(value) for value in result["ocr_v2_cuda_device_count"].dropna().tolist()]
        if not ocr_device_counts or max(ocr_device_counts) < 1:
            raise AssertionError(f"expected at least one CUDA device in OCR v2 actor, got {ocr_device_counts!r}")

        ocr_detection_counts = [int(value) for value in result["ocr_v1_num_detections"].fillna(0).tolist()]
        ocr_total_detections = sum(ocr_detection_counts)
        if ocr_total_detections <= 0:
            raise AssertionError("expected at least one OCR v2 detection")

        ocr_hosts = sorted({str(host) for host in result["ocr_v2_host"].dropna().unique() if str(host)})
        if not ocr_hosts:
            raise AssertionError("expected at least one ocr_v2_host value")

        payload.update(
            {
                "ocr_v2_hosts": ocr_hosts,
                "ocr_v2_detection_counts": ocr_detection_counts,
                "ocr_v2_total_detections": int(ocr_total_detections),
                "ocr_v2_cuda": {
                    "available": sorted({bool(value) for value in result["ocr_v2_cuda_available"].tolist()}),
                    "device_counts": sorted({int(value) for value in result["ocr_v2_cuda_device_count"].tolist()}),
                    "device_names": sorted(
                        {
                            str(value)
                            for value in result["ocr_v2_cuda_device_name"].dropna().unique()
                            if str(value)
                        }
                    ),
                    "visible_devices": sorted(
                        {
                            str(value)
                            for value in result["ocr_v2_cuda_visible_devices"].dropna().unique()
                            if str(value)
                        }
                    ),
                },
            }
        )
    if args.enable_text_embed_vdb:
        required_embed_columns = (
            "text_embeddings_1b_v2",
            "text_embeddings_1b_v2_dim",
            "text_embeddings_1b_v2_has_embedding",
            "text_embed_host",
            "text_embed_cuda_available",
            "text_embed_cuda_device_count",
        )
        for column in required_embed_columns:
            if column not in result.columns:
                raise AssertionError(f"expected result column {column!r}")

        embedding_payload_errors = []
        for embedding_payload in result["text_embeddings_1b_v2"]:
            if isinstance(embedding_payload, dict) and embedding_payload.get("error"):
                embedding_payload_errors.append(embedding_payload["error"])
        if embedding_payload_errors:
            raise AssertionError(f"text embedding returned errors: {embedding_payload_errors!r}")

        has_embeddings = [bool(value) for value in result["text_embeddings_1b_v2_has_embedding"].tolist()]
        if not has_embeddings or not all(has_embeddings):
            raise AssertionError(f"expected every row to contain a text embedding, got {has_embeddings!r}")

        embedding_dims = [int(value) for value in result["text_embeddings_1b_v2_dim"].fillna(0).tolist()]
        if not embedding_dims or any(dim != 2048 for dim in embedding_dims):
            raise AssertionError(f"expected 2048-d text embeddings, got {embedding_dims!r}")

        text_embed_cuda_available = [bool(value) for value in result["text_embed_cuda_available"].dropna().tolist()]
        if not text_embed_cuda_available or not all(text_embed_cuda_available):
            raise AssertionError(f"expected text embed actor to see CUDA, got {text_embed_cuda_available!r}")

        text_embed_device_counts = [int(value) for value in result["text_embed_cuda_device_count"].dropna().tolist()]
        if not text_embed_device_counts or max(text_embed_device_counts) < 1:
            raise AssertionError(f"expected at least one CUDA device in text embed actor, got {text_embed_device_counts!r}")

        text_embed_hosts = sorted({str(host) for host in result["text_embed_host"].dropna().unique() if str(host)})
        if not text_embed_hosts:
            raise AssertionError("expected at least one text_embed_host value")

        vdb_row_count = _lancedb_row_count(vdb_uri, args.vdb_table)
        if vdb_row_count != len(result):
            raise AssertionError(f"expected LanceDB row count {len(result)}, got {vdb_row_count}")

        payload.update(
            {
                "text_embed_hosts": text_embed_hosts,
                "text_embedding_dims": embedding_dims,
                "text_embed_cuda": {
                    "available": sorted({bool(value) for value in result["text_embed_cuda_available"].tolist()}),
                    "device_counts": sorted({int(value) for value in result["text_embed_cuda_device_count"].tolist()}),
                    "device_names": sorted(
                        {
                            str(value)
                            for value in result["text_embed_cuda_device_name"].dropna().unique()
                            if str(value)
                        }
                    ),
                    "visible_devices": sorted(
                        {
                            str(value)
                            for value in result["text_embed_cuda_visible_devices"].dropna().unique()
                            if str(value)
                        }
                    ),
                },
                "vdb": {
                    "op": "lancedb",
                    "uri": str(vdb_uri),
                    "table": args.vdb_table,
                    "row_count": int(vdb_row_count),
                },
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        write_json(args.output_json, {"status": "failed", "error": repr(exc)})
        raise
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
