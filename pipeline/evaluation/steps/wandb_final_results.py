"""Optional WandB logging step for final results."""

from __future__ import annotations

import json

from helpers.env_variables import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT
from helpers.logging_config import get_logger
from helpers.path_serialization import project_absolute_path
from pipeline.abstract import AbstractStep, StepContext
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.evaluation.services import (
    WandbFinalResultsConfig,
    WandbFinalResultsLoggingService,
    WandbFinalResultsLogResult,
)
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator

logger = get_logger(__name__)


class LogFinalResultsToWandbStep(
    AbstractStep[FinalResultsEvaluationResult, FinalResultsEvaluationResult]
):
    """Append final metrics to a shared or standalone W&B experiment."""

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        logging_service: WandbFinalResultsLoggingService | None = None,
        coordinator: WandbExperimentCoordinator | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.project = project or WANDB_PROJECT
        self.entity = entity if entity is not None else WANDB_ENTITY
        self.mode = mode or WANDB_MODE
        self.logging_service = logging_service or WandbFinalResultsLoggingService()
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[FinalResultsEvaluationResult],
    ) -> FinalResultsEvaluationResult:
        final_result = context.result
        if final_result is None:
            logger.warning("Skipping WandB logging because final results are missing.")
            return FinalResultsEvaluationResult(
                dataset_id="",
                results_run_directory=".",
                results_run_name="",
                results_run_number=0,
                results_config_path=".",
                retrieval_metrics_path=".",
                reasoning_metrics_path=".",
                per_instance_results_path=".",
                evaluated_instances=0,
                accuracy=0.0,
                hit_rate=0.0,
                hits_at_1=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                grounded_explanation_rate=0.0,
                ndcg_at_10=0.0,
                wandb_status="skipped",
                wandb_error_message="Final results are missing.",
            )

        if self.coordinator is not None:
            return self._log_with_coordinator(final_result)

        logger.info(
            f"Logging final results to WandB: project={self.project} "
            f"entity={self.entity} mode={self.mode} "
            f"results_run={final_result.results_run_name}"
        )
        try:
            outcome = self.logging_service.log_final_results(
                final_result=final_result,
                config=WandbFinalResultsConfig(
                    project=self.project,
                    entity=self.entity,
                    mode=self.mode,
                ),
            )
        except Exception as error:
            logger.warning(f"WandB final results logging failed: {error}")
            outcome = WandbFinalResultsLogResult(
                status="failed",
                error_message=str(error),
            )
        return final_result.model_copy(
            update={
                "wandb_status": outcome.status,
                "wandb_run_id": outcome.run_id,
                "wandb_run_url": outcome.run_url,
                "wandb_error_message": outcome.error_message,
            }
        )

    def _log_with_coordinator(
        self,
        final_result: FinalResultsEvaluationResult,
    ) -> FinalResultsEvaluationResult:
        """Append namespaced inference metrics to the shared pipeline experiment."""
        try:
            results_config = json.loads(
                final_result.results_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            results_config = {}
        artifacts = results_config.get("artifacts", {})
        evaluation_ref = artifacts.get("evaluation", {}) if isinstance(artifacts, dict) else {}
        evaluation_config_value = evaluation_ref.get("evaluation_config_path")
        source_config_path = (
            project_absolute_path(evaluation_config_value)
            if isinstance(evaluation_config_value, str)
            else None
        )
        reasoning_metrics = self.logging_service._load_json_object(
            final_result.reasoning_metrics_path
        )
        scalar_metrics = self.logging_service.build_scalar_metrics(
            retrieval_metrics={},
            reasoning_metrics=reasoning_metrics,
        )
        self.coordinator.update_config(
            self.logging_service.build_wandb_config(
                final_result=final_result,
                results_config=results_config,
            ),
            source_config_path=source_config_path,
        )
        payload: dict[str, float | int] = {}
        payload.update(
            self.logging_service.build_run_summary_plot_metrics(
                scalar_metrics=scalar_metrics,
                wandb_config={},
            )
        )
        payload.update(
            self.logging_service.build_summary_plot_metrics(scalar_metrics)
        )
        self.coordinator.log_aggregate_metrics(
            payload,
            source_config_path=source_config_path,
        )
        artifact_paths = [
            final_result.results_config_path,
            final_result.retrieval_metrics_path,
            final_result.reasoning_metrics_path,
            final_result.per_instance_results_path,
        ]
        self.coordinator.persist_metadata(final_result.results_config_path)
        self.coordinator.log_artifact(
            name=f"results-{final_result.results_run_name}",
            artifact_type="final-results",
            paths=artifact_paths,
            source_config_path=source_config_path,
        )
        self.coordinator.persist_metadata(final_result.results_config_path)
        metadata = self.coordinator.metadata
        return final_result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )
