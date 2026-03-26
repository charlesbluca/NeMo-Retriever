"""BEIR Evaluator — Designer component for running BEIR evaluation against LanceDB."""

from __future__ import annotations

from typing import Annotated

from nemo_retriever.graph.designer import Param, designer_component


@designer_component(
    name="BEIR Evaluator",
    category="Evaluation",
    compute="cpu",
    description="Runs BEIR evaluation against a LanceDB table using a HuggingFace BEIR dataset",
    category_color="#42d6a4",
    component_type="pipeline_evaluator",
)
class BEIREvaluatorActor:
    """Evaluation node placed after a LanceDB Writer in the Designer pipeline.

    Parameters are read by the code-generator to emit an inline call to
    ``nemo_retriever.recall.beir.evaluate_lancedb_beir``.
    """

    def __init__(
        self,
        lancedb_uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        lancedb_table: Annotated[str, Param(label="Table Name")] = "nv-ingest",
        embedding_model: Annotated[
            str, Param(label="Embedding Model")
        ] = "nvidia/llama-nemotron-embed-1b-v2",
        beir_loader: Annotated[
            str, Param(label="BEIR Loader", choices=["vidore_hf"])
        ] = "vidore_hf",
        beir_dataset_name: Annotated[
            str, Param(label="BEIR Dataset Name", placeholder="e.g. vidore_v3_computer_science")
        ] = "",
        beir_split: Annotated[str, Param(label="BEIR Split")] = "test",
        beir_query_language: Annotated[
            str, Param(label="Query Language", placeholder="Optional (e.g. en, fr)")
        ] = "",
        beir_doc_id_field: Annotated[
            str,
            Param(label="Doc ID Field", choices=["pdf_basename", "pdf_page", "source_id", "path"]),
        ] = "pdf_basename",
        beir_ks: Annotated[str, Param(label="K Values", placeholder="1,3,5,10")] = "1,3,5,10",
        hybrid: Annotated[bool, Param(label="Hybrid Search")] = False,
    ) -> None:
        self.lancedb_uri = lancedb_uri
        self.lancedb_table = lancedb_table
        self.embedding_model = embedding_model
        self.beir_loader = beir_loader
        self.beir_dataset_name = beir_dataset_name
        self.beir_split = beir_split
        self.beir_query_language = beir_query_language
        self.beir_doc_id_field = beir_doc_id_field
        self.beir_ks = beir_ks
        self.hybrid = hybrid
