"""Typed models for WebQSP path extraction and LLM subgraph context."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pipeline.abstract import StepResult
from pipeline.evaluation.exceptions import InvalidEvaluationSampleException


class GraphTriple(BaseModel):
    """One directed relation triple from a WebQSP local graph."""

    source: str = Field(..., description="Head/source entity.")
    relation: str = Field(..., description="Relation label.")
    target: str = Field(..., description="Tail/target entity.")


class EvaluationSample(StepResult):
    """Normalized WebQSP sample used by evaluation steps."""

    sample_id: str | None = Field(default=None, description="Dataset sample identifier.")
    question: str = Field(..., description="Original question text.")
    q_entities: list[str] = Field(..., description="Question/topic entities.")
    a_entities: list[str] = Field(default_factory=list, description="Gold answer entities.")
    graph_triples: list[GraphTriple] = Field(..., description="Local graph triples.")

    @classmethod
    def from_webqsp_row(cls, raw_sample: dict[str, Any]) -> "EvaluationSample":
        """Build an evaluation sample from a WebQSP-shaped row."""
        question = raw_sample.get("question")
        if not isinstance(question, str) or not question:
            raise InvalidEvaluationSampleException(
                "Evaluation sample must contain a non-empty 'question' string."
            )

        q_entities = cls._normalize_string_list(raw_sample.get("q_entity"), "q_entity")
        a_entities = cls._normalize_string_list(
            raw_sample.get("a_entity", []),
            "a_entity",
            allow_empty=True,
        )
        graph_triples = cls._normalize_graph_triples(raw_sample.get("graph"))
        sample_id = raw_sample.get("id")

        return cls(
            sample_id=str(sample_id) if sample_id is not None else None,
            question=question,
            q_entities=q_entities,
            a_entities=a_entities,
            graph_triples=graph_triples,
        )

    @staticmethod
    def _normalize_string_list(
        value: Any,
        field_name: str,
        allow_empty: bool = False,
    ) -> list[str]:
        """Normalize WebQSP string-or-list fields into a non-empty string list."""
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = [str(item) for item in value if str(item)]
        else:
            values = []

        if not values and not allow_empty:
            raise InvalidEvaluationSampleException(
                f"Evaluation sample must contain a non-empty '{field_name}' field."
            )

        return values

    @staticmethod
    def _normalize_graph_triples(value: Any) -> list[GraphTriple]:
        """Normalize WebQSP graph rows into typed triples."""
        if not isinstance(value, list) or not value:
            raise InvalidEvaluationSampleException(
                "Evaluation sample must contain a non-empty 'graph' list."
            )

        triples: list[GraphTriple] = []
        for item in value:
            if not isinstance(item, list) or len(item) != 3:
                raise InvalidEvaluationSampleException(
                    "Each graph item must be a [head, relation, tail] triple."
                )
            head, relation, tail = item
            triples.append(
                GraphTriple(
                    source=str(head),
                    relation=str(relation),
                    target=str(tail),
                )
            )

        return triples


class CandidateNodeScore(BaseModel):
    """Score assigned to one candidate answer node."""

    node_id: str = Field(..., description="Candidate node identifier or label.")
    score: float = Field(..., description="Generic candidate ranking score.")
    local_node_id: int | None = Field(
        default=None,
        description="Candidate local graph node id for persisted GNN predictions.",
    )
    global_node_id: int | None = Field(
        default=None,
        description="Candidate global vocabulary node id for persisted GNN predictions.",
    )
    logit: float | None = Field(
        default=None,
        description="Raw GNN classifier logit for persisted predictions.",
    )
    probability: float | None = Field(
        default=None,
        description="Sigmoid GNN classifier probability for persisted predictions.",
    )
    is_gold_answer: bool | None = Field(
        default=None,
        description="Whether the candidate is a gold answer when known.",
    )
    selection_reason: str | None = Field(
        default=None,
        description="Why the GNN evaluation selected this candidate.",
    )


class CandidateNodeScores(StepResult):
    """Ranked candidate answer nodes for one evaluation sample."""

    sample: EvaluationSample = Field(..., description="Evaluation sample being scored.")
    candidates: list[CandidateNodeScore] = Field(
        default_factory=list,
        description="Ranked candidate answer nodes.",
    )
    top_k: int = Field(..., description="Requested top-K candidate count.")


class ReasoningPath(BaseModel):
    """Shortest reasoning path for one candidate answer node."""

    candidate_node: str = Field(..., description="Candidate node targeted by the path.")
    candidate_score: float = Field(..., description="Candidate answer score.")
    path_found: bool = Field(..., description="Whether a path was found.")
    triples: list[GraphTriple] = Field(
        default_factory=list,
        description="Triples along the first shortest path, kept for simple inspection.",
    )
    shortest_paths: list[list[GraphTriple]] = Field(
        default_factory=list,
        description="All equal-length shortest paths to the candidate.",
    )


class EvidenceSubgraphConstruction(BaseModel):
    """Per-instance evidence-subgraph construction metadata."""

    strategy: str = "shortest_path"
    edge_cost_strategy: str | None = None
    edge_cost_lambda: float | None = None
    semantic_embedding_model: str | None = None
    input_candidate_count: int = 0
    valid_candidate_count: int = 0
    selected_candidate_count: int = 0
    selected_candidate_ranks: list[int] = Field(default_factory=list)
    collected_prize: float = 0.0
    selected_node_count: int = 0
    selected_triple_count: int = 0
    total_edge_cost: float = 0.0
    objective: float = 0.0
    valid_seed_count: int = 0
    missing_seed_count: int = 0
    construction_time_ms: float = 0.0
    empty_result_reason: str | None = None

    @property
    def candidate_evidence_coverage(self) -> float:
        """Return the fraction of candidates represented in the evidence."""
        if self.input_candidate_count == 0:
            return 0.0
        return self.selected_candidate_count / self.input_candidate_count


class ExtractedReasoningPaths(StepResult):
    """Structured paths and deduplicated reasoning subgraph for LLM context."""

    sample: EvaluationSample = Field(..., description="Evaluation sample.")
    paths: list[ReasoningPath] = Field(
        default_factory=list,
        description="Candidate reasoning paths.",
    )
    reasoning_subgraph_triples: list[GraphTriple] = Field(
        default_factory=list,
        description="Deduplicated triples from all found shortest candidate paths.",
    )
    reasoning_paths_text: str = Field(
        default="",
        description="Human-readable deduplicated reasoning subgraph for LLM context.",
    )
    found_paths: int = Field(default=0, description="Number of candidates with paths.")
    missing_paths: int = Field(default=0, description="Number of candidates without paths.")
    construction: EvidenceSubgraphConstruction = Field(
        default_factory=EvidenceSubgraphConstruction,
        description="Evidence strategy and per-instance construction analytics.",
    )
