"""Evaluation-specific exceptions for Stage 2 inference."""

from pipeline.exceptions import PipelineException


class InvalidEvaluationSampleException(PipelineException):
    """Raised when a WebQSP evaluation sample has invalid or missing fields."""


class ShortestPathExtractionException(PipelineException):
    """Raised when shortest path extraction cannot run for a valid sample."""


class LlmAnswerGenerationException(PipelineException):
    """Raised when final LLM answer generation fails."""


class InsufficientLlmCreditsException(LlmAnswerGenerationException):
    """Raised when an LLM provider rejects a request for insufficient credit."""


class FinalResultsEvaluationException(PipelineException):
    """Raised when final results evaluation cannot be computed or stored."""
