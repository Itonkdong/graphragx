"""Final results step for combining retrieval and LLM reasoning metrics."""

from __future__ import annotations

from pydantic import Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.exceptions import FinalResultsEvaluationException
from pipeline.evaluation.models import (
    FinalResultsEvaluationResult,
    GnnAnswerRetrieverEvaluationResult,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.services import FinalResultsEvaluationService

logger = get_logger(__name__)


class ComputeFinalResultsContext(StepContext[SavedLlmInferenceRun]):
    """Context for computing final metrics from saved upstream artifacts."""

    gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult = Field(
        ...,
        description="Earlier GNN answer-retriever evaluation result.",
    )


class ComputeFinalResultsStep(
    AbstractStep[FinalResultsEvaluationResult, SavedLlmInferenceRun]
):
    """Compute final answer, grounding, and ranking metrics."""

    def __init__(
        self,
        evaluation_service: FinalResultsEvaluationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.evaluation_service = evaluation_service or FinalResultsEvaluationService()

    def execute_default(
        self,
        context: ComputeFinalResultsContext,
    ) -> FinalResultsEvaluationResult:
        inference_run = context.result
        if inference_run is None:
            raise FinalResultsEvaluationException(
                "Final results evaluation requires a saved LLM inference run."
            )

        logger.info(
            f"Computing final results: evaluation_run="
            f"{context.gnn_evaluation_result.evaluation_run_name} "
            f"inference_run={inference_run.inference_run_name}"
        )
        outcome = self.evaluation_service.evaluate(
            gnn_evaluation_result=context.gnn_evaluation_result,
            llm_inference_run=inference_run,
        )
        answer_metrics = outcome.reasoning_metrics.answer_metrics
        grounding_metrics = outcome.reasoning_metrics.explanation_grounding_metrics
        ranking_metrics = outcome.reasoning_metrics.ranking_metrics
        logger.info(
            f"Finished final results: run={outcome.storage_result.results_run_name} "
            f"accuracy={answer_metrics.accuracy:.4f} "
            f"precision={answer_metrics.precision:.4f} "
            f"recall={answer_metrics.recall:.4f} "
            f"f1={answer_metrics.f1:.4f}"
        )
        return FinalResultsEvaluationResult(
            dataset_id=inference_run.dataset_id,
            gnn_architecture=context.gnn_evaluation_result.gnn_architecture,
            subgraph_algorithm=str(
                inference_run.evidence_subgraph.get("algorithm", "shortest_path")
            ),
            model_run_name=context.gnn_evaluation_result.model_run_name,
            evaluation_run_name=context.gnn_evaluation_result.evaluation_run_name,
            inference_run_name=inference_run.inference_run_name,
            results_run_directory=outcome.storage_result.results_run_directory,
            results_run_name=outcome.storage_result.results_run_name,
            results_run_number=outcome.storage_result.results_run_number,
            results_config_path=outcome.storage_result.results_config_path,
            retrieval_metrics_path=outcome.storage_result.retrieval_metrics_path,
            reasoning_metrics_path=outcome.storage_result.reasoning_metrics_path,
            per_instance_results_path=outcome.storage_result.per_instance_results_path,
            evaluated_instances=answer_metrics.evaluated_instances,
            accuracy=answer_metrics.accuracy,
            hit_rate=answer_metrics.hit_rate,
            hits_at_1=answer_metrics.hits_at_1,
            precision=answer_metrics.precision,
            recall=answer_metrics.recall,
            f1=answer_metrics.f1,
            grounded_explanation_rate=(
                grounding_metrics.grounded_explanation_rate
            ),
            ndcg_at_10=ranking_metrics.ndcg_at_10,
        )
