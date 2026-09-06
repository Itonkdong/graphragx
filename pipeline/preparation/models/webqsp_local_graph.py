"""Prepared WebQSP graph dataset models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any


class WebQSPProcessedInstance(BaseModel):
    """Prepared graph representation for one WebQSP dataset instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(..., description="Topic/question entities.")
    a_entity: list[str] = Field(..., description="Gold answer entities.")
    nodes: list[str] = Field(..., description="Local graph node names.")
    node2id: dict[str, int] = Field(..., description="Local node-to-id mapping.")
    edge_index: TorchTensor = Field(..., description="Directed edge tensor.")
    edge_relations: list[str] = Field(..., description="Relation text for each edge.")
    node_labels: TorchTensor = Field(..., description="Binary answer-node labels.")


class WebQSPVocabularyStore(BaseModel):
    """Reusable text vocabularies collected from WebQSP graphs."""

    nodes: dict[str, int] = Field(default_factory=dict)
    relations: dict[str, int] = Field(default_factory=dict)
    questions: dict[str, int] = Field(default_factory=dict)


class WebQSPEntityMappingSummary(BaseModel):
    """Summary of MID-to-readable-name mapping during WebQSP processing."""

    total_entity_references: int = Field(default=0)
    mapped_entity_references: int = Field(default=0)
    disambiguated_entity_references: int = Field(default=0)
    unmapped_mid_entity_references: int = Field(default=0)
    unique_mapped_mid_count: int = Field(default=0)
    unique_disambiguated_mid_count: int = Field(default=0)
    unique_unmapped_mid_count: int = Field(default=0)
    unmapped_mid_samples: list[str] = Field(default_factory=list)


class PreparedWebQSPGraphDataset(StepResult):
    """Prepared WebQSP graph dataset artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    processing_version: str = Field(..., description="Processing version.")
    use_reverse_edges: bool = Field(
        default=False,
        description="Whether reverse edges were materialized during processing.",
    )
    train_instances: list[WebQSPProcessedInstance] = Field(default_factory=list)
    test_instances: list[WebQSPProcessedInstance] = Field(default_factory=list)
    vocabulary_store: WebQSPVocabularyStore = Field(...)
    entity_mapping_summary: WebQSPEntityMappingSummary = Field(
        default_factory=WebQSPEntityMappingSummary
    )
    source_fingerprints: dict[str, str] = Field(default_factory=dict)
    entity_mapping_sha256: str | None = Field(default=None)
    cache_directory: Path = Field(..., description="Processed dataset cache directory.")

    @property
    def train_size(self) -> int:
        """Return the number of prepared training instances."""
        return len(self.train_instances)

    @property
    def test_size(self) -> int:
        """Return the number of prepared test instances."""
        return len(self.test_instances)
