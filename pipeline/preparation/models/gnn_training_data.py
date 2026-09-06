"""Prepared tensors used by GNN answer-retriever training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.abstract import StepResult
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any


class PreparedGnnTrainingInstance(BaseModel):
    """Indexed graph tensors for one selected training instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_instance_index: int = Field(..., description="Index in the train split.")
    node_embedding_indices: TorchTensor | None = Field(default=None)
    relation_embedding_indices: TorchTensor | None = Field(default=None)
    question_embedding_index: int | None = Field(default=None)
    edge_index: TorchTensor = Field(...)
    edge_type: TorchTensor | None = Field(default=None)
    edge_norm: TorchTensor | None = Field(default=None)
    active_relation_ids: TorchTensor | None = Field(default=None)
    edge_relation_index: TorchTensor | None = Field(default=None)
    active_relation_offsets: TorchTensor | None = Field(default=None)
    node_labels: TorchTensor = Field(...)
    gold_answer_count: int | None = Field(
        default=None,
        description="Number of unique gold answers used by preparation filtering.",
    )
    question_input_ids: TorchTensor | None = Field(default=None)
    question_attention_mask: TorchTensor | None = Field(default=None)
    seed_distribution: TorchTensor | None = Field(default=None)
    seed_mask: TorchTensor | None = Field(default=None)
    initialization_edge_index: TorchTensor | None = Field(default=None)
    initialization_edge_type: TorchTensor | None = Field(default=None)
    seed_node_indices: TorchTensor | None = Field(default=None)
    skip_reason: str | None = Field(default=None)


class PreparedGnnTrainingData(StepResult):
    """Compact frozen embedding matrices and indexed graphs ready for training."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    built_retriever: BuiltGnnAnswerRetriever = Field(...)
    instances: list[PreparedGnnTrainingInstance] = Field(default_factory=list)
    node_embeddings: TorchTensor | None = Field(default=None)
    relation_embeddings: TorchTensor | None = Field(default=None)
    question_embeddings: TorchTensor | None = Field(default=None)
    training_start_instance: int = Field(...)
    training_end_instance: int = Field(...)
    selected_device: str = Field(...)
    embedding_cache_device: str = Field(...)
    embedding_cache_dtype: str = Field(...)
    entity_embedding_model: str | None = Field(default=None)
    question_embedding_model: str | None = Field(default=None)
    relation_embedding_model: str | None = Field(default=None)
    relation_input_ids: TorchTensor | None = Field(default=None)
    relation_attention_mask: TorchTensor | None = Field(default=None)
    runtime_strategy: str = Field(default="default")
    autocast_dtype: str = Field(default="float32")
    cache_root: Path = Field(...)
    skipped_missing_gold_in_graph_count: int = Field(default=0)

    @property
    def uses_bfloat16(self) -> bool:
        """Return whether frozen embeddings use BF16 storage."""
        return self.autocast_dtype == "bfloat16" or self.embedding_cache_dtype == "bfloat16"
