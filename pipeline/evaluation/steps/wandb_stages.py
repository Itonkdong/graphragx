"""Stage-aware W&B logging steps for training and retrieval artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helpers.logging_config import get_logger
from helpers.constants import GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
from pipeline.preparation.services.gnn_relation_vocabulary import (
    RGCN_RELATION_VOCABULARY_FILENAME,
)
from pipeline.abstract import AbstractStep, StepContext
from pipeline.exceptions import PipelineException
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationResult,
    SavedEvidenceSubgraphRun,
    SavedLlmInferenceRun,
)
from helpers.path_serialization import project_absolute_path
from helpers.path_serialization import make_project_paths_relative
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.evaluation.services.wandb_final_results import (
    WandbFinalResultsLoggingService,
)
from pipeline.evaluation.services.model_config_normalization import (
    normalize_model_config,
)
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainedGnnAnswerRetriever,
)
logger = get_logger(__name__)
LEGACY_GNN_CONFIG_FILENAME = "gnn_answer_retriever_config.json"


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    payload.pop("wandb", None)
    return payload


def _run_reference(config: dict[str, Any]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    if config.get("run_name") is not None:
        reference["name"] = config["run_name"]
    if config.get("run_number") is not None:
        reference["number"] = config["run_number"]
    return reference


def _build_available_wandb_config(
    *,
    model_config_path: Path | None = None,
    model_weights_path: Path | None = None,
    evaluation_config_path: Path | None = None,
    predictions_path: Path | None = None,
    retrieval_metrics_path: Path | None = None,
    inference_config_path: Path | None = None,
    answers_path: Path | None = None,
    reasoning_path: Path | None = None,
) -> dict[str, Any]:
    """Build the legacy Config-tab shape from all currently available stages."""
    inference_config = _load_config(inference_config_path)
    inference_evaluation_ref = inference_config.get("evaluation_config", {})
    if evaluation_config_path is None and isinstance(inference_evaluation_ref, dict):
        value = inference_evaluation_ref.get("full_config_path")
        if isinstance(value, str):
            evaluation_config_path = project_absolute_path(value)
        predictions_value = inference_evaluation_ref.get("predictions_path")
        if predictions_path is None and isinstance(predictions_value, str):
            predictions_path = project_absolute_path(predictions_value)

    evaluation_config = _load_config(evaluation_config_path)
    evaluation_model_ref = evaluation_config.get("model_config", {})
    if isinstance(evaluation_model_ref, dict):
        if model_config_path is None:
            value = evaluation_model_ref.get("full_config_path")
            if isinstance(value, str):
                model_config_path = project_absolute_path(value)
        if model_weights_path is None:
            value = evaluation_model_ref.get("weights_path")
            if isinstance(value, str):
                model_weights_path = project_absolute_path(value)
    evaluation_artifacts = evaluation_config.get("artifacts", {})
    if isinstance(evaluation_artifacts, dict):
        if predictions_path is None:
            value = evaluation_artifacts.get("predictions_path")
            if isinstance(value, str):
                predictions_path = project_absolute_path(value)
        if retrieval_metrics_path is None:
            value = evaluation_artifacts.get("retrieval_metrics_path")
            if isinstance(value, str):
                retrieval_metrics_path = project_absolute_path(value)

    model_config = _load_config(model_config_path)
    normalized_model_config = normalize_model_config(model_config)
    runs: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    source_paths: dict[str, Any] = {}
    payload: dict[str, Any] = {}

    dataset_id = (
        inference_config.get("dataset_id")
        or evaluation_config.get("dataset_id")
        or model_config.get("dataset_id")
    )
    if dataset_id is not None:
        payload["dataset_id"] = dataset_id

    if normalized_model_config:
        runs["model"] = _run_reference(normalized_model_config)
        configs["model"] = normalized_model_config
    if model_config_path is not None:
        source_paths["model_config_path"] = model_config_path
        source_paths["training_model_config_path"] = model_config_path
    if model_weights_path is not None:
        source_paths["training_weights_path"] = model_weights_path
    if model_config_path is not None:
        relation_vocabulary_path = (
            model_config_path.parent / RGCN_RELATION_VOCABULARY_FILENAME
        )
        if relation_vocabulary_path.exists():
            source_paths["training_relation_vocabulary_path"] = (
                relation_vocabulary_path
            )

    if evaluation_config:
        runs["evaluation"] = _run_reference(evaluation_config)
        evaluation_payload = evaluation_config.get("evaluation")
        if isinstance(evaluation_payload, dict):
            configs["evaluation"] = evaluation_payload
    if evaluation_config_path is not None:
        source_paths["evaluation_config_path"] = evaluation_config_path
        source_paths["evaluation_evaluation_config_path"] = evaluation_config_path
    if predictions_path is not None:
        source_paths["evaluation_predictions_path"] = predictions_path
    if retrieval_metrics_path is not None:
        source_paths["retrieval_metrics_path"] = retrieval_metrics_path

    if inference_config:
        runs["inference"] = _run_reference(inference_config)
        inference_payload = inference_config.get("inference")
        if isinstance(inference_payload, dict):
            evidence_payload = inference_payload.get("evidence_subgraph")
            configs["inference"] = {
                key: value
                for key, value in inference_payload.items()
                if key not in {"evidence_metrics", "evidence_subgraph"}
            }
            if isinstance(evidence_payload, dict):
                configs["evidence"] = evidence_payload
            if inference_payload.get("model_id") is not None:
                payload["model_id"] = inference_payload["model_id"]
            if inference_payload.get("llm_provider") is not None:
                payload["llm_provider"] = inference_payload["llm_provider"]
    if inference_config_path is not None:
        source_paths["inference_config_path"] = inference_config_path
        source_paths["inference_inference_config_path"] = inference_config_path
    if answers_path is not None:
        source_paths["inference_answers_path"] = answers_path
    if reasoning_path is not None:
        source_paths["inference_reasoning_path"] = reasoning_path

    if runs:
        payload["runs"] = runs
    if configs:
        payload["configs"] = configs
    if source_paths:
        payload["source_paths"] = source_paths
    return make_project_paths_relative(payload)


class LogTrainingToWandbStep(
    AbstractStep[TrainedGnnAnswerRetriever, TrainedGnnAnswerRetriever]
):
    """Log and annotate one completed GNN training artifact."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
        upload_retriever: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator
        self.upload_retriever = upload_retriever

    def execute_default(
        self,
        context: StepContext[TrainedGnnAnswerRetriever],
    ) -> TrainedGnnAnswerRetriever:
        result = context.result
        if result is None:
            raise PipelineException("Training W&B logging requires a training result.")
        self.coordinator.ensure_run(architecture_name=result.gnn_architecture)
        self.coordinator.update_config(
            _build_available_wandb_config(
                model_config_path=result.model_config_path,
                model_weights_path=result.model_artifact_path,
            )
        )
        for point in result.loss_history:
            epoch = int(point["epoch"])
            payload = {
                "average_loss": float(point["average_loss"]),
                "epoch": epoch,
            }
            log_training_epoch = getattr(
                self.coordinator,
                "log_training_epoch",
                None,
            )
            if callable(log_training_epoch):
                epoch_payload = {
                    "average_loss": payload["average_loss"],
                    "epoch": epoch,
                    "gnn_architecture": result.gnn_architecture,
                }
                log_training_epoch(epoch_payload)
            else:
                self.coordinator.log(
                    {
                        "Training/gnn_training_loss": payload["average_loss"],
                        "Training/epoch": epoch,
                    }
                )
        self.coordinator.persist_metadata(result.model_config_path)
        model_artifact_paths = [result.model_config_path]
        if self.upload_retriever:
            model_artifact_paths.append(result.model_artifact_path)
        if result.relation_vocabulary_path is not None:
            model_artifact_paths.append(result.relation_vocabulary_path)
        self.coordinator.log_artifact(
            name=f"model-{result.model_run_name}",
            artifact_type="gnn-model",
            paths=model_artifact_paths,
        )
        self.coordinator.persist_metadata(result.model_config_path)
        metadata = self.coordinator.metadata
        logger.info(
            f"Recorded training W&B stage: run={result.model_run_name} "
            f"status={metadata.status}"
        )
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )


class LogRetrieverToWandbStep(
    AbstractStep[GnnAnswerRetrieverEvaluationResult, GnnAnswerRetrieverEvaluationResult]
):
    """Log retriever metrics and predictions to the shared experiment."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        copy_to_new_experiment: bool = False,
        force_default: bool = False,
        upload_retriever: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator
        self.upload_retriever = upload_retriever
        self.copy_to_new_experiment = copy_to_new_experiment

    def execute_default(
        self,
        context: StepContext[GnnAnswerRetrieverEvaluationResult],
    ) -> GnnAnswerRetrieverEvaluationResult:
        result = context.result
        if result is None:
            raise PipelineException("Retriever W&B logging requires an evaluation result.")
        had_active_run = self.coordinator.has_active_run
        model_config_path = result.model_run_directory / "model_config.json"
        if not model_config_path.exists():
            model_config_path = (
                result.model_run_directory / LEGACY_GNN_CONFIG_FILENAME
            )
        source_config_path = (
            result.evaluation_config_path
            if self._has_lineage(result.evaluation_config_path)
            else model_config_path
        )
        self.coordinator.ensure_run(
            source_config_path=source_config_path,
            architecture_name=result.gnn_architecture,
        )
        if not had_active_run and result.wandb_run_id is None:
            self._backfill_training(
                model_config_path,
                upload_retriever=self.upload_retriever,
            )
        self.coordinator.update_config(
            _build_available_wandb_config(
                model_config_path=model_config_path,
                evaluation_config_path=result.evaluation_config_path,
                predictions_path=result.predictions_path,
                retrieval_metrics_path=result.retrieval_metrics_path,
            )
        )
        self.coordinator.update_tags(
            [f"evaluated_instances:{result.evaluated_instances}"]
        )
        is_logged_continuation = (
            result.wandb_run_id is not None and not self.copy_to_new_experiment
        )
        if is_logged_continuation:
            logger.info(
                "Preserving previously logged retriever metrics during W&B "
                f"continuation: run={result.evaluation_run_name}"
            )
        if not is_logged_continuation:
            scalar_metrics = WandbFinalResultsLoggingService.build_scalar_metrics(
                retrieval_metrics={
                    "evaluated_instances": result.evaluated_instances,
                    "hits_at_1": result.hits_at_1,
                    "hits_at_1_count": result.hits_at_1_count,
                    "hits_at_5": result.hits_at_5,
                    "hits_at_5_count": result.hits_at_5_count,
                    "hits_at_10": result.hits_at_10,
                    "hits_at_10_count": result.hits_at_10_count,
                    "hits_at_candidate_limit": result.hits_at_candidate_limit,
                    "hits_at_candidate_limit_count": (
                        result.hits_at_candidate_limit_count
                    ),
                    "ndcg_at_1": result.ndcg_at_1,
                    "ndcg_at_5": result.ndcg_at_5,
                    "ndcg_at_10": result.ndcg_at_10,
                    "ndcg_at_candidate_limit": result.ndcg_at_candidate_limit,
                    "conditioned_evaluated_instances": (
                        result.conditioned_evaluated_instances
                    ),
                    "retrieval_gold_coverage": result.retrieval_gold_coverage,
                    "retrieval_full_gold_coverage_count": (
                        result.retrieval_full_gold_coverage_count
                    ),
                    "retrieval_full_gold_coverage_rate": (
                        result.retrieval_full_gold_coverage_rate
                    ),
                    "retrieved_gold_answer_count": (
                        result.retrieved_gold_answer_count
                    ),
                    "average_candidate_count": result.average_candidate_count,
                    "missing_gold_in_graph_count": (
                        result.missing_gold_in_graph_count
                    ),
                    "skipped_missing_gold_in_graph_count": (
                        result.skipped_missing_gold_in_graph_count
                    ),
                },
                reasoning_metrics={},
            )
            metrics = WandbFinalResultsLoggingService.build_run_summary_plot_metrics(
                scalar_metrics=scalar_metrics,
                wandb_config={},
            )
            metrics.update(
                WandbFinalResultsLoggingService.build_summary_plot_metrics(
                    scalar_metrics
                )
            )
            self.coordinator.log_aggregate_metrics(metrics)
        if not self.copy_to_new_experiment:
            self.coordinator.persist_metadata(result.evaluation_config_path)
        if not is_logged_continuation:
            artifact_paths = [result.evaluation_config_path, result.predictions_path]
            if result.retrieval_metrics_path is not None:
                artifact_paths.append(result.retrieval_metrics_path)
            self.coordinator.log_artifact(
                name=f"retriever-{result.evaluation_run_name}",
                artifact_type="retriever-results",
                paths=artifact_paths,
            )
        if not self.copy_to_new_experiment:
            self.coordinator.persist_metadata(result.evaluation_config_path)
        metadata = self.coordinator.metadata
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )

    def _backfill_training(
        self,
        model_config_path: Path,
        *,
        upload_retriever: bool,
    ) -> None:
        """Log saved training history when an upstream run has no W&B lineage."""
        try:
            payload: dict[str, Any] = json.loads(
                model_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return
        training_payload = payload.get("training", {})
        if not isinstance(training_payload, dict):
            training_payload = {}
        for point in (
            training_payload.get("loss_history")
            or payload.get("loss_history", [])
        ):
            if not isinstance(point, dict):
                continue
            epoch = point.get("epoch")
            average_loss = point.get("average_loss")
            if isinstance(epoch, int) and isinstance(average_loss, int | float):
                epoch_payload = {
                    "epoch": epoch,
                    "average_loss": float(average_loss),
                }
                log_training_epoch = getattr(
                    self.coordinator,
                    "log_training_epoch",
                    None,
                )
                if callable(log_training_epoch):
                    log_training_epoch(epoch_payload)
                else:
                    self.coordinator.log(
                        {
                            "Training/epoch": epoch,
                            "Training/gnn_training_loss": float(average_loss),
                        }
                    )
        artifact_paths = [model_config_path]
        if upload_retriever:
            artifact_paths.append(
                model_config_path.parent / GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
            )
        relation_vocabulary_path = (
            model_config_path.parent / RGCN_RELATION_VOCABULARY_FILENAME
        )
        if relation_vocabulary_path.exists():
            artifact_paths.append(relation_vocabulary_path)
        self.coordinator.log_artifact(
            name=f"model-{model_config_path.parent.name}",
            artifact_type="gnn-model",
            paths=artifact_paths,
        )

    @staticmethod
    def _has_lineage(config_path: Path) -> bool:
        """Return whether a JSON config contains a resumable W&B run id."""
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        tracking = payload.get("wandb") if isinstance(payload, dict) else None
        return isinstance(tracking, dict) and bool(tracking.get("run_id"))


class LogInferenceToWandbStep(
    AbstractStep[SavedLlmInferenceRun, SavedLlmInferenceRun]
):
    """Log a persisted inference run before combined metric computation."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[SavedLlmInferenceRun],
    ) -> SavedLlmInferenceRun:
        result = context.result
        if result is None:
            raise PipelineException("Inference W&B logging requires an inference run.")
        try:
            config = json.loads(result.inference_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        evaluation_ref = config.get("evaluation_config", {})
        evaluation_config_value = (
            evaluation_ref.get("full_config_path")
            if isinstance(evaluation_ref, dict)
            else None
        )
        source_config_path = (
            project_absolute_path(evaluation_config_value)
            if isinstance(evaluation_config_value, str)
            else None
        )
        architecture_name = None
        if source_config_path is not None:
            try:
                source_config = json.loads(
                    source_config_path.read_text(encoding="utf-8")
                )
                architecture_name = source_config.get("gnn_architecture")
            except (OSError, json.JSONDecodeError):
                pass
        self.coordinator.ensure_run(
            source_config_path=source_config_path,
            architecture_name=(
                str(architecture_name) if architecture_name is not None else None
            ),
        )
        self.coordinator.update_config(
            _build_available_wandb_config(
                evaluation_config_path=source_config_path,
                inference_config_path=result.inference_config_path,
                answers_path=result.answers_path,
                reasoning_path=result.reasoning_path,
            ),
            source_config_path=source_config_path,
        )
        inference_payload = config.get("inference", {})
        if not isinstance(inference_payload, dict):
            inference_payload = {}
        evidence_configuration = inference_payload.get("evidence_subgraph", {})
        evidence_algorithm = (
            evidence_configuration.get("algorithm")
            if isinstance(evidence_configuration, dict)
            else None
        )
        model_id = inference_payload.get("model_id")
        self.coordinator.update_inference_run_name(
            evidence_algorithm=(
                str(evidence_algorithm)
                if isinstance(evidence_algorithm, str)
                else None
            ),
            model_id=str(model_id) if isinstance(model_id, str) else None,
        )
        evidence_metrics = (
            inference_payload.get("evidence_metrics", {})
        )
        if isinstance(evidence_metrics, dict):
            evidence_payload = {
                f"Summary_Plots/evidence_{name}": value
                for name, value in evidence_metrics.items()
                if isinstance(value, int | float)
            }
            candidate_reduction_percentage = evidence_metrics.get(
                "candidate_reduction_percentage"
            )
            if isinstance(candidate_reduction_percentage, int | float):
                evidence_payload[
                    "Run_Summary/evidence_candidate_reduction_percentage"
                ] = candidate_reduction_percentage
            self.coordinator.log_aggregate_metrics(
                evidence_payload,
                source_config_path=source_config_path,
                architecture_name=(
                    str(architecture_name) if architecture_name is not None else None
                ),
            )
        self.coordinator.persist_metadata(result.inference_config_path)
        self.coordinator.log_artifact(
            name=f"inference-raw-{result.inference_run_name}",
            artifact_type="inference-predictions",
            paths=[
                result.inference_config_path,
                result.answers_path,
                result.reasoning_path,
            ],
            source_config_path=source_config_path,
        )
        self.coordinator.persist_metadata(result.inference_config_path)
        metadata = self.coordinator.metadata
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )


class LogEvidenceToWandbStep(
    AbstractStep[SavedEvidenceSubgraphRun, SavedEvidenceSubgraphRun]
):
    """Log a saved evidence-only run into its new copied W&B experiment."""

    def __init__(
        self,
        coordinator: WandbExperimentCoordinator,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.coordinator = coordinator

    def execute_default(
        self,
        context: StepContext[SavedEvidenceSubgraphRun],
    ) -> SavedEvidenceSubgraphRun:
        result = context.result
        if result is None:
            raise PipelineException("Evidence W&B logging requires an evidence run.")
        try:
            config = json.loads(result.evidence_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        evaluation_ref = config.get("evaluation_config", {})
        evaluation_path_value = (
            evaluation_ref.get("full_config_path")
            if isinstance(evaluation_ref, dict)
            else None
        )
        evaluation_config_path = (
            project_absolute_path(evaluation_path_value)
            if isinstance(evaluation_path_value, str)
            else None
        )
        self.coordinator.ensure_run(
            source_config_path=evaluation_config_path,
            architecture_name=result.gnn_architecture,
        )
        evidence_config = config.get("evidence", {})
        if not isinstance(evidence_config, dict):
            evidence_config = {}
        self.coordinator.update_config(
            make_project_paths_relative(
                {
                    "dataset_id": result.dataset_id,
                    "gnn_architecture": result.gnn_architecture,
                    "runs": {
                        "evidence": {
                            "name": result.evidence_run_name,
                            "number": result.evidence_run_number,
                        }
                    },
                    "configs": {
                        "evidence": {
                            key: value
                            for key, value in evidence_config.items()
                            if key != "evidence_metrics"
                        }
                    },
                    "source_paths": {
                        "evidence_config_path": result.evidence_config_path,
                        "evidence_subgraphs_path": result.evidence_subgraphs_path,
                        "evidence_metrics_path": result.evidence_metrics_path,
                    },
                }
            ),
            source_config_path=evaluation_config_path,
        )
        strategy_tag = (
            "sp" if result.subgraph_algorithm == "shortest_path" else "pcst"
        )
        tags = [result.subgraph_algorithm]
        pcst = result.evidence_configuration.get("pcst")
        if isinstance(pcst, dict) and pcst.get("edge_cost_strategy"):
            tags.append(f"pcst-{pcst['edge_cost_strategy']}")
        self.coordinator.update_tags(tags)
        self.coordinator.update_inference_run_name(
            evidence_algorithm=result.subgraph_algorithm,
            model_id=None,
        )
        metrics = result.evidence_metrics
        payload = {
            **{
                (
                    f"Summary_Plots/{name}"
                    if name.startswith("reasoning_context_")
                    else f"Summary_Plots/evidence_{name}"
                ): value
                for name, value in metrics.items()
                if isinstance(value, int | float)
            },
            "Run_Summary/evidence_candidate_reduction_percentage": metrics.get(
                "candidate_reduction_percentage", 0.0
            ),
            "Run_Summary/reasoning_context_gold_coverage": metrics.get(
                "reasoning_context_gold_coverage", 0.0
            ),
            "Run_Summary/reasoning_context_full_gold_coverage": metrics.get(
                "reasoning_context_full_gold_coverage_rate", 0.0
            ),
            "Run_Summary/evidence_empty_subgraph_rate": metrics.get(
                "empty_subgraph_rate", 0.0
            ),
        }
        self.coordinator.log_aggregate_metrics(
            payload,
            source_config_path=evaluation_config_path,
            architecture_name=result.gnn_architecture,
        )
        self.coordinator.log_artifact(
            name=f"evidence-{result.evidence_run_name}",
            artifact_type="evidence-subgraphs",
            paths=[
                result.evidence_config_path,
                result.evidence_subgraphs_path,
                result.evidence_metrics_path,
            ],
            source_config_path=evaluation_config_path,
        )
        self.coordinator.persist_metadata(result.evidence_config_path)
        metadata = self.coordinator.metadata
        logger.info(
            f"Logged evidence run to W&B: run={result.evidence_run_name} "
            f"strategy={strategy_tag} status={metadata.status}"
        )
        return result.model_copy(
            update={
                "wandb_status": metadata.status,
                "wandb_run_id": metadata.run_id,
                "wandb_run_url": metadata.run_url,
                "wandb_error_message": metadata.error_message,
            }
        )
