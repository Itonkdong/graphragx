"""Models for final LLM answer generation."""

from __future__ import annotations

from pydantic import Field

from pipeline.abstract import StepResult
from pipeline.evaluation.models.path_extraction import ExtractedReasoningPaths


class GeneratedFinalAnswer(StepResult):
    """Final answer generated from the question and reasoning paths."""

    extracted_paths: ExtractedReasoningPaths = Field(
        ...,
        description="Reasoning paths/subgraph used as answer context.",
    )
    model_id: str = Field(..., description="LLM model used for answer generation.")
    prompt: str = Field(..., description="Prompt sent to the LLM.")
    answers: list[str] = Field(
        default_factory=list,
        description="Generated answer entities with atomic entity boundaries.",
    )
