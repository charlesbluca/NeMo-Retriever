"""Recall Evaluator — Designer component for running recall evaluation against LanceDB."""

from __future__ import annotations

from typing import Annotated

from nemo_retriever.graph.designer import Param, designer_component


@designer_component(
    name="Recall Evaluator",
    category="Evaluation",
    compute="cpu",
    description="Runs recall evaluation against a LanceDB table using a ground-truth query CSV",
    category_color="#42d6a4",
    component_type="pipeline_evaluator",
)
class RecallEvaluatorActor:
    """Evaluation node placed after a LanceDB Writer in the Designer pipeline.

    Parameters are read by the code-generator to emit an inline call to
    ``nemo_retriever.recall.core.retrieve_and_score``.
    """

    def __init__(
        self,
        lancedb_uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        lancedb_table: Annotated[str, Param(label="Table Name")] = "nv-ingest",
        query_csv: Annotated[str, Param(label="Query CSV", placeholder="/path/to/query_gt.csv")] = "",
        embedding_model: Annotated[
            str, Param(label="Embedding Model")
        ] = "nvidia/llama-nemotron-embed-1b-v2",
        match_mode: Annotated[
            str, Param(label="Match Mode", choices=["pdf_page", "pdf_only"])
        ] = "pdf_page",
        ks: Annotated[str, Param(label="K Values", placeholder="1,3,5,10")] = "1,3,5,10",
        hybrid: Annotated[bool, Param(label="Hybrid Search")] = False,
    ) -> None:
        self.lancedb_uri = lancedb_uri
        self.lancedb_table = lancedb_table
        self.query_csv = query_csv
        self.embedding_model = embedding_model
        self.match_mode = match_mode
        self.ks = ks
        self.hybrid = hybrid
