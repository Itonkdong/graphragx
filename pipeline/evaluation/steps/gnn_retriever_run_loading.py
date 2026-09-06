"""Pipeline step for loading an existing retriever evaluation run."""

from __future__ import annotations

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.models import GnnAnswerRetrieverEvaluationResult
from pipeline.evaluation.services.gnn_retriever_results import GnnRetrieverResultsService
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset

logger = get_logger(__name__)


class LoadGnnAnswerRetrieverRunStep(
    AbstractStep[GnnAnswerRetrieverEvaluationResult, PreparedWebQSPGraphDataset]
):
    """Load saved retriever predictions and metrics for downstream inference."""

    def __init__(
        self,
        run_name: str | None = None,
        run_number: int | None = None,
        results_service: GnnRetrieverResultsService | None = None,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.run_name = run_name
        self.run_number = run_number
        self.results_service = results_service or GnnRetrieverResultsService()

    def execute_default(
        self,
        context: StepContext[PreparedWebQSPGraphDataset],
    ) -> GnnAnswerRetrieverEvaluationResult:
        prepared_dataset = context.result
        if prepared_dataset is None:
            from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException

            raise GnnAnswerRetrieverEvaluationException(
                "Loading a retriever run requires a prepared dataset."
            )
        result = self.results_service.load_run(
            evaluation_root=prepared_dataset.cache_directory.parent / "evaluations",
            dataset_id=prepared_dataset.dataset_id,
            run_name=self.run_name,
            run_number=self.run_number,
        )
        logger.info(
            f"Loaded retriever run: run={result.evaluation_run_name} "
            f"instances={result.evaluated_instances}"
        )
        return result
