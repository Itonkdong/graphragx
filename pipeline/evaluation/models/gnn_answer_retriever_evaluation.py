"""Models for GNN answer-retriever evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from helpers.constants import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_CANDIDATE_TOP_K,
    MIN_CANDIDATE_TOP_K,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE,
    DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB,
    DEFAULT_EVALUATION_LOG_EVERY,
    DEFAULT_EVALUATION_PROFILE,
)
from pipeline.abstract import StepResult
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any


class GnnAnswerRetrieverEvaluationConfig(BaseModel):
    """Runtime settings for evaluating a saved GNN answer-retriever run."""

    model_run_name: str | None = Field(default=None)
    model_run_number: int | None = Field(default=None)
    answer_threshold: float = Field(default=DEFAULT_ANSWER_THRESHOLD)
    candidate_top_k: int = Field(
        default=DEFAULT_CANDIDATE_TOP_K,
        ge=MIN_CANDIDATE_TOP_K,
    )
    candidate_limit: int = Field(default=DEFAULT_CANDIDATE_LIMIT)
    run_name: str | None = Field(default=None)
    max_instances: int | None = Field(default=None)
    skip_missing_gold_in_graph: bool = Field(default=True)
    log_every: int = Field(default=DEFAULT_EVALUATION_LOG_EVERY)
    profile: bool = Field(default=DEFAULT_EVALUATION_PROFILE)
    embedding_cache_device: Literal["auto", "gpu", "cpu"] = Field(
        default=DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE
    )
    embedding_cache_dtype: Literal["auto", "float32", "bfloat16"] = Field(
        default=DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE
    )
    gpu_cache_reserve_gb: float = Field(
        default=DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB
    )


class PreparedGnnEvaluationInstance(BaseModel):
    """One evaluation graph indexed into compact embedding matrices."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_instance_index: int = Field(...)
    instance: WebQSPProcessedInstance = Field(...)
    node_embedding_indices: TorchTensor | None = Field(default=None)
    relation_embedding_indices: TorchTensor | None = Field(default=None)
    question_embedding_index: int | None = Field(default=None)
    edge_index: TorchTensor = Field(...)
    edge_type: TorchTensor | None = Field(default=None)
    edge_norm: TorchTensor | None = Field(default=None)
    active_relation_ids: TorchTensor | None = Field(default=None)
    edge_relation_index: TorchTensor | None = Field(default=None)
    active_relation_offsets: TorchTensor | None = Field(default=None)
    question_input_ids: TorchTensor | None = Field(default=None)
    question_attention_mask: TorchTensor | None = Field(default=None)
    seed_distribution: TorchTensor | None = Field(default=None)
    seed_mask: TorchTensor | None = Field(default=None)
    initialization_edge_index: TorchTensor | None = Field(default=None)
    initialization_edge_type: TorchTensor | None = Field(default=None)
    seed_node_indices: TorchTensor | None = Field(default=None)
    skip_reason: str | None = Field(default=None)


class PreparedGnnEvaluationData(BaseModel):
    """Compact frozen embeddings and indexed graphs used during evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instances: list[PreparedGnnEvaluationInstance] = Field(default_factory=list)
    node_embeddings: TorchTensor | None = Field(default=None)
    relation_embeddings: TorchTensor | None = Field(default=None)
    question_embeddings: TorchTensor | None = Field(default=None)
    selected_device: str = Field(...)
    embedding_cache_device: str = Field(...)
    embedding_cache_dtype: str = Field(...)
    relation_input_ids: TorchTensor | None = Field(default=None)
    relation_attention_mask: TorchTensor | None = Field(default=None)
    runtime_strategy: str = Field(default="default")
    autocast_dtype: str = Field(default="float32")
    skipped_missing_gold_in_graph_count: int = Field(default=0)

    @property
    def uses_bfloat16(self) -> bool:
        """Return whether compact evaluation embeddings use BF16 storage."""
        return self.autocast_dtype == "bfloat16" or self.embedding_cache_dtype == "bfloat16"


class AnswerCandidateScore(BaseModel):
    """Score for one selected answer-candidate node."""

    node: str = Field(..., description="Candidate node text.")
    local_node_id: int = Field(..., description="Candidate local graph node id.")
    global_node_id: int = Field(..., description="Global nodes.json vocabulary id.")
    logit: float = Field(..., description="Raw classifier logit.")
    probability: float = Field(..., description="Architecture-normalized node probability.")
    is_gold_answer: bool = Field(..., description="Whether the candidate is gold.")
    selection_reason: str = Field(
        ...,
        description="Whether the node was selected by threshold or fallback top-k.",
    )


class GoldAnswerScore(BaseModel):
    """Score for a gold answer node, even when it was not selected."""

    node: str = Field(..., description="Gold answer node text.")
    local_node_id: int | None = Field(
        default=None,
        description="Local node id when the gold answer is present in the graph.",
    )
    global_node_id: int | None = Field(
        default=None,
        description="Global nodes.json vocabulary id when the node is known.",
    )
    logit: float | None = Field(default=None, description="Raw classifier logit.")
    probability: float | None = Field(default=None, description="Normalized node probability.")
    present_in_graph: bool = Field(
        ...,
        description="Whether the gold answer appears in the local graph.",
    )


class EvaluatedAnswerRetrievalInstance(BaseModel):
    """Persisted retrieval evaluation output for one WebQSP test instance."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(..., description="Question/topic entities.")
    a_entity: list[str] = Field(..., description="Gold answer entities.")
    answer_candidates: list[AnswerCandidateScore] = Field(default_factory=list)
    gold_answer_scores: list[GoldAnswerScore] = Field(default_factory=list)
    hit_at_1: bool = Field(..., description="Whether top scored node is gold.")
    hit_at_5: bool = Field(
        default=False,
        description="Whether any top-5 selected answer candidate is gold.",
    )
    hit_at_10: bool = Field(
        default=False,
        description="Whether any top-10 selected answer candidate is gold.",
    )
    hit_at_candidate_limit: bool = Field(
        default=False,
        description="Whether any selected answer candidate is gold.",
    )
    missing_gold_in_graph: bool = Field(
        ...,
        description="Whether at least one gold answer is absent from the local graph.",
    )


class GnnAnswerRetrieverMetrics(BaseModel):
    """Aggregate metrics for one persisted answer-retriever evaluation."""

    dataset_id: str
    model_run_name: str
    model_run_number: int
    evaluation_run_name: str | None = None
    evaluation_run_number: int | None = None
    evaluated_instances: int
    hits_at_1: float
    hits_at_1_count: int
    hits_at_5: float
    hits_at_5_count: int
    hits_at_10: float
    hits_at_10_count: int
    hits_at_candidate_limit: float
    hits_at_candidate_limit_count: int
    ndcg_at_1: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_candidate_limit: float = 0.0
    conditioned_evaluated_instances: int = 0
    retrieval_gold_coverage: float = 0.0
    retrieval_full_gold_coverage_count: int = 0
    retrieval_full_gold_coverage_rate: float = 0.0
    retrieved_gold_answer_count: int = 0
    candidate_limit: int
    average_candidate_count: float
    missing_gold_in_graph_count: int
    skipped_missing_gold_in_graph_count: int = 0


class GnnAnswerRetrieverEvaluationResult(StepResult):
    """Saved evaluation result for a GNN answer-retriever run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Evaluated dataset identifier.")
    gnn_architecture: str = Field(default="graphsage")
    model_run_directory: Path = Field(..., description="Selected model run directory.")
    model_run_name: str = Field(..., description="Selected model run folder name.")
    model_run_number: int = Field(..., description="Selected model run number.")
    evaluation_run_directory: Path = Field(
        ...,
        description="Created evaluation run directory.",
    )
    evaluation_run_name: str = Field(..., description="Created evaluation run folder name.")
    evaluation_run_number: int = Field(..., description="Created evaluation run number.")
    evaluated_instances: int = Field(..., description="Number of evaluated instances.")
    hits_at_1: float = Field(..., description="Hits@1 rate.")
    hits_at_1_count: int = Field(..., description="Hits@1 count.")
    hits_at_5: float = Field(default=0.0, description="Hits@5 rate.")
    hits_at_5_count: int = Field(default=0, description="Hits@5 count.")
    hits_at_10: float = Field(default=0.0, description="Hits@10 rate.")
    hits_at_10_count: int = Field(default=0, description="Hits@10 count.")
    hits_at_candidate_limit: float = Field(
        default=0.0,
        description="Hits at configured candidate limit rate.",
    )
    hits_at_candidate_limit_count: int = Field(
        default=0,
        description="Hits at configured candidate limit count.",
    )
    ndcg_at_1: float = Field(default=0.0, description="Mean retrieval nDCG@1.")
    ndcg_at_5: float = Field(default=0.0, description="Mean retrieval nDCG@5.")
    ndcg_at_10: float = Field(default=0.0, description="Mean retrieval nDCG@10.")
    ndcg_at_candidate_limit: float = Field(
        default=0.0,
        description="Mean retrieval nDCG at the configured candidate limit.",
    )
    conditioned_evaluated_instances: int = Field(
        default=0,
        description="Evaluated predictions containing normalized gold answers.",
    )
    retrieval_gold_coverage: float = Field(
        default=0.0,
        description="Macro mean share of gold answers among retrieved candidates.",
    )
    retrieval_full_gold_coverage_count: int = Field(default=0)
    retrieval_full_gold_coverage_rate: float = Field(default=0.0)
    retrieved_gold_answer_count: int = Field(default=0)
    average_candidate_count: float = Field(
        ...,
        description="Average number of selected candidates.",
    )
    missing_gold_in_graph_count: int = Field(
        ...,
        description="Instances where no gold answer appears in the local graph.",
    )
    skipped_missing_gold_in_graph_count: int = Field(
        default=0,
        description="Instances excluded before evaluation for incomplete gold coverage.",
    )
    predictions_path: Path = Field(..., description="Saved JSONL predictions path.")
    evaluation_config_path: Path = Field(..., description="Saved evaluation config path.")
    retrieval_metrics_path: Path | None = Field(
        default=None,
        description="Saved aggregate retrieval metrics path when available.",
    )
    wandb_status: str | None = None
    wandb_run_id: str | None = None
    wandb_run_url: str | None = None
    wandb_error_message: str | None = None
