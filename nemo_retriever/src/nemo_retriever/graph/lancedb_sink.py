"""LanceDB Writer — Designer sink component for writing Ray Datasets to LanceDB."""

from __future__ import annotations

from typing import Annotated

from nemo_retriever.graph.designer import Param, designer_component


@designer_component(
    name="LanceDB Writer",
    category="Data Sinks",
    compute="cpu",
    description="Materializes a Ray Dataset and writes rows with a 'vector' column to a LanceDB table",
    category_color="#ff9f43",
    component_type="pipeline_sink",
)
class LanceDBWriterActor:
    """Sink node: materializes the Ray Dataset, concatenates Arrow batches,
    clears any stale LanceDB directory, and writes to a new table.

    This class is **not** used at runtime via ``map_batches``; its parameters
    are read by the Designer code-generator to emit inline write logic.
    """

    def __init__(
        self,
        uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        table_name: Annotated[str, Param(label="Table Name")] = "nv-ingest",
    ) -> None:
        self.uri = uri
        self.table_name = table_name
