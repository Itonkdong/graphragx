"""Batch models for post-retrieval LLM inference runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.abstract import StepResult
from pipeline.evaluation.models.gnn_answer_retriever_evaluation import (
    EvaluatedAnswerRetrievalInstance,
)
from pipeline.evaluation.models.path_extraction import (
    CandidateNodeScores,
    EvidenceSubgraphConstruction,
    ExtractedReasoningPaths,
    GraphTriple,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance


class ReasoningSampleForPrediction(BaseModel):
    """One GNN prediction converted into a graph reasoning sample."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    prediction: EvaluatedAnswerRetrievalInstance = Field(
        ...,
        description="GNN answer-retriever prediction for the instance.",
    )
    candidate_scores: CandidateNodeScores = Field(
        ...,
        description="Candidate scores and local graph sample for path extraction.",
    )
    graph_instance: WebQSPProcessedInstance | None = Field(
        default=None,
        exclude=True,
        description="In-memory processed graph used by optimized path extraction.",
    )


class BuiltReasoningSamples(StepResult):
    """Batch of GNN predictions converted into path-extraction inputs."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    evaluation_run_directory: Path = Field(
        ...,
        description="Source GNN evaluation run directory.",
    )
    samples: list[ReasoningSampleForPrediction] = Field(default_factory=list)


class ReasoningPathsForPrediction(BaseModel):
    """Extracted shortest paths for one GNN prediction."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    prediction: EvaluatedAnswerRetrievalInstance = Field(
        ...,
        description="Source GNN answer-retriever prediction.",
    )
    extracted_paths: ExtractedReasoningPaths = Field(
        ...,
        description="Shortest paths and deduplicated reasoning subgraph.",
    )


class ExtractedReasoningPathsBatch(StepResult):
    """Batch output of shortest-path extraction over GNN candidates."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    items: list[ReasoningPathsForPrediction] = Field(default_factory=list)


class SavedEvidenceSubgraphRun(StepResult):
    """Persisted evidence construction run without LLM answer generation."""

    dataset_id: str
    gnn_architecture: str
    evaluation_run_name: str
    evidence_run_directory: Path
    evidence_run_name: str
    evidence_run_number: int
    evaluated_instances: int
    subgraph_algorithm: str
    evidence_configuration: dict[str, object] = Field(default_factory=dict)
    evidence_metrics: dict[str, float | int] = Field(default_factory=dict)
    evidence_config_path: Path
    evidence_subgraphs_path: Path
    evidence_metrics_path: Path
    wandb_status: str | None = None
    wandb_run_id: str | None = None
    wandb_run_url: str | None = None
    wandb_error_message: str | None = None


class GeneratedAnswerForPrediction(BaseModel):
    """Generated LLM answer for one path-extracted prediction."""

    instance_index: int = Field(..., description="Index inside the prepared test split.")
    question: str = Field(..., description="Natural language question.")
    q_entity: list[str] = Field(default_factory=list)
    a_entity: list[str] = Field(default_factory=list)
    answer_candidates: list[str] = Field(default_factory=list)
    reasoning_subgraph_triples: list[GraphTriple] = Field(default_factory=list)
    reasoning_path_lengths: list[int] = Field(default_factory=list)
    found_reasoning_paths: int = Field(default=0)
    missing_reasoning_paths: int = Field(default=0)
    reasoning_paths_text: str = Field(default="")
    evidence_construction: EvidenceSubgraphConstruction = Field(
        default_factory=EvidenceSubgraphConstruction
    )
    model_id: str = Field(..., description="LLM model used for answer generation.")
    llm_provider: str = Field(default="openai", description="LLM provider used.")
    answers: list[str] = Field(
        default_factory=list,
        description="Generated answer entities with atomic entity boundaries.",
    )
    explanation: str = Field(
        default="",
        description="Explanation of which reasoning paths supported the answer.",
    )
    raw_response: str = Field(default="", description="Raw LLM response text.")
    prompt_tokens: int = Field(default=0, description="Prompt/input tokens used.")
    completion_tokens: int = Field(default=0, description="Completion/output tokens used.")
    total_tokens: int = Field(default=0, description="Total tokens used.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated generation cost.")
    error_message: str | None = Field(
        default=None,
        description="Generation error when this instance failed.",
    )


class GeneratedFinalAnswersBatch(StepResult):
    """Batch of LLM-generated answers for extracted reasoning paths."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    llm_provider: str = Field(default="openai", description="LLM provider used.")
    reasoning_effort: str | None = Field(default=None)
    generate_explanation: bool = Field(
        default=True,
        description="Whether this run requested LLM-generated explanations.",
    )
    evidence_subgraph: dict[str, object] = Field(default_factory=dict)
    inference_batch_size: int | None = Field(
        default=None,
        description="Number of answers persisted together during batched inference.",
    )
    inference_parallel_calls: int = Field(
        default=1,
        description="Maximum simultaneous LLM API calls used for this run.",
    )
    items: list[GeneratedAnswerForPrediction] = Field(default_factory=list)

    @property
    def successful_answers(self) -> int:
        """Return the number of successfully generated answers."""
        return sum(1 for item in self.items if item.error_message is None)

    @property
    def failed_answers(self) -> int:
        """Return the number of failed answer generations."""
        return sum(1 for item in self.items if item.error_message is not None)


class SavedLlmInferenceRun(StepResult):
    """Persisted inference run metadata."""

    dataset_id: str = Field(..., description="Dataset identifier.")
    evaluation_run_name: str = Field(..., description="Source GNN evaluation run name.")
    inference_run_directory: Path = Field(..., description="Created inference directory.")
    inference_run_name: str = Field(..., description="Created inference run folder name.")
    inference_run_number: int = Field(..., description="Created inference run number.")
    model_id: str = Field(..., description="LLM model used for answer generation.")
    llm_provider: str = Field(default="openai", description="LLM provider used.")
    reasoning_effort: str | None = Field(default=None)
    generate_explanation: bool = Field(
        default=True,
        description="Whether this run requested LLM-generated explanations.",
    )
    evidence_subgraph: dict[str, object] = Field(default_factory=dict)
    inference_batch_size: int | None = None
    inference_parallel_calls: int = 1
    total_instances: int = Field(..., description="Number of instances processed.")
    successful_answers: int = Field(..., description="Number of successful generations.")
    failed_answers: int = Field(..., description="Number of failed generations.")
    reasoning_path: Path = Field(..., description="Saved reasoning path.")
    answers_path: Path = Field(..., description="Saved answers path.")
    inference_config_path: Path = Field(..., description="Saved inference config path.")
    wandb_status: str | None = None
    wandb_run_id: str | None = None
    wandb_run_url: str | None = None
    wandb_error_message: str | None = None
