"""Pipeline step for final LLM answer generation."""

from __future__ import annotations

from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.exceptions import LlmAnswerGenerationException
from pipeline.evaluation.models import ExtractedReasoningPaths, GeneratedFinalAnswer
from pipeline.evaluation.services import LangChainOpenAiAnswerGenerationService


class GenerateFinalAnswerStep(
    AbstractStep[GeneratedFinalAnswer, ExtractedReasoningPaths]
):
    """Generate a final answer from extracted reasoning paths."""

    def __init__(
        self,
        model_id: str = "gpt-4.1-mini",
        answer_generation_service: LangChainOpenAiAnswerGenerationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.model_id = model_id
        self.answer_generation_service = (
            answer_generation_service or LangChainOpenAiAnswerGenerationService()
        )

    def execute_default(
        self,
        context: StepContext[ExtractedReasoningPaths],
    ) -> GeneratedFinalAnswer:
        extracted_paths = context.result
        if extracted_paths is None:
            raise LlmAnswerGenerationException(
                "Final answer generation requires extracted reasoning paths."
            )

        answers, prompt = self.answer_generation_service.generate_answer(
            question=extracted_paths.sample.question,
            reasoning_paths_text=extracted_paths.reasoning_paths_text,
            model_id=self.model_id,
        )
        return GeneratedFinalAnswer(
            extracted_paths=extracted_paths,
            model_id=self.model_id,
            prompt=prompt,
            answers=answers,
        )
