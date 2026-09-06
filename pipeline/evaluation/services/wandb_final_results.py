"""Optional WandB logging for final results runs."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from helpers.logging_config import get_logger
from helpers.path_serialization import project_absolute_path
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.preparation.helpers.gnn_architecture import infer_gnn_architecture
from pipeline.evaluation.services.model_config_normalization import (
    normalize_model_config,
)
from pipeline.evaluation.services.wandb_experiment import (
    WandbExperimentCoordinator,
)
from pipeline.services import AbstractService

logger = get_logger(__name__)


class WandbFinalResultsConfig(BaseModel):
    """Runtime WandB settings for final results logging."""

    project: str = Field(..., description="WandB project name.")
    entity: str | None = Field(default=None, description="Optional WandB entity.")
    mode: str = Field(default="online", description="WandB mode.")


class WandbFinalResultsLogResult(BaseModel):
    """Outcome of optional WandB logging."""

    status: Literal["logged", "skipped", "failed"] = Field(
        ...,
        description="WandB logging status.",
    )
    run_id: str | None = Field(default=None, description="WandB run id.")
    run_url: str | None = Field(default=None, description="WandB run URL.")
    error_message: str | None = Field(default=None, description="Logging error.")


class WandbFinalResultsLoggingService(AbstractService):
    """Upload final result metrics, tables, and artifacts to WandB."""

    run_summary_prefix = "Run_Summary"
    loss_metric_key = "Training/gnn_training_loss"
    table_key = "Per_Instance_Metrics/per_instance_results"
    aggregate_table_key = "Summary_Metrics/aggregate_metrics"
    summary_plot_prefix = "Summary_Plots"
    artifact_type = "final-results"
    retrieval_conditioned_metric_keys = (
        "conditioned_evaluated_instances",
        "retrieval_gold_coverage",
        "retrieval_full_gold_coverage_count",
        "retrieval_full_gold_coverage_rate",
        "reasoning_context_gold_coverage",
        "reasoning_context_full_gold_coverage_count",
        "reasoning_context_full_gold_coverage_rate",
        "retrieved_gold_answer_count",
        "answered_retrieved_gold_count",
        "llm_retrieved_gold_utilization",
        "llm_omission_given_full_retrieval_count",
        "llm_omission_given_full_retrieval_rate",
        "llm_exact_match_given_full_retrieval_count",
        "llm_exact_match_given_full_retrieval",
        "llm_omission_given_full_context_count",
        "llm_omission_given_full_context_rate",
        "llm_exact_match_given_full_context_count",
        "llm_exact_match_given_full_context",
        "full_retrieval_complete_answer_count",
        "full_retrieval_complete_answer_rate",
        "full_retrieval_llm_omission_count",
        "full_retrieval_llm_omission_rate",
        "partial_retrieval_fully_utilized_count",
        "partial_retrieval_fully_utilized_rate",
        "partial_retrieval_underutilized_count",
        "partial_retrieval_underutilized_rate",
        "full_context_complete_answer_count",
        "full_context_complete_answer_rate",
        "full_context_llm_omission_count",
        "full_context_llm_omission_rate",
        "partial_context_fully_utilized_count",
        "partial_context_fully_utilized_rate",
        "partial_context_underutilized_count",
        "partial_context_underutilized_rate",
        "no_gold_retrieved_no_gold_answered_count",
        "no_gold_retrieved_no_gold_answered_rate",
        "correct_without_gold_retrieval_count",
        "correct_without_gold_retrieval_rate",
    )
    source_path_keys = {
        "answers_path",
        "reasoning_path",
        "predictions_path",
        "inference_config_path",
        "evaluation_config_path",
    }
    table_columns = [
        "instance_index",
        "question",
        "q_entity",
        "gold_answers",
        "predicted_answers",
        "explanation",
        "exact_match",
        "hit",
        "hits_at_1",
        "precision",
        "recall",
        "f1",
        "mentioned_triple_count",
        "grounded_mentioned_triple_count",
        "grounded_explanation",
        "fully_grounded_explanation",
        "ndcg_at_1",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_candidate_limit",
        "answer_error_message",
        "retrieval_gold_coverage",
        "reasoning_context_gold_coverage",
        "llm_retrieved_gold_utilization",
        "retrieval_generation_outcome",
        "retrieved_candidates",
    ]

    def log_final_results(
        self,
        final_result: FinalResultsEvaluationResult,
        config: WandbFinalResultsConfig,
    ) -> WandbFinalResultsLogResult:
        """Best-effort upload of local final results to WandB."""
        try:
            wandb = importlib.import_module("wandb")
            results_config = self._load_json_object(final_result.results_config_path)
            retrieval_metrics = self._load_json_object(
                final_result.retrieval_metrics_path
            )
            # Legacy retrieval metric files predate this scalar; the final
            # result still carries the same evaluated-instance denominator.
            retrieval_metrics.setdefault(
                "evaluated_instances",
                final_result.evaluated_instances,
            )
            reasoning_metrics = self._load_json_object(
                final_result.reasoning_metrics_path
            )
            per_instance_rows = self._load_jsonl_objects(
                final_result.per_instance_results_path
            )
            scalar_metrics = self.build_scalar_metrics(
                retrieval_metrics=retrieval_metrics,
                reasoning_metrics=reasoning_metrics,
            )
            aggregate_metric_rows = self.build_aggregate_metric_rows(scalar_metrics)
            table_rows = self.build_table_rows(
                results_config=results_config,
                per_instance_rows=per_instance_rows,
            )
            wandb_config = self.build_wandb_config(
                final_result=final_result,
                results_config=results_config,
            )
            run_name = final_result.results_run_name
            architecture_name = results_config.get("gnn_architecture")
            if (
                isinstance(architecture_name, str)
                and architecture_name
                and not run_name.endswith(f"_{architecture_name}")
            ):
                run_name = f"{run_name}_{architecture_name}"
            inference_config = wandb_config.get("configs", {}).get("inference", {})
            if not isinstance(inference_config, dict):
                inference_config = {}
            evidence_configuration = wandb_config.get("configs", {}).get(
                "evidence", {}
            )
            evidence_algorithm = (
                evidence_configuration.get("algorithm")
                if isinstance(evidence_configuration, dict)
                else None
            )
            run_name = WandbExperimentCoordinator.build_inference_run_name(
                run_name,
                evidence_algorithm=(
                    str(evidence_algorithm)
                    if isinstance(evidence_algorithm, str)
                    else None
                ),
                model_id=(
                    str(inference_config["model_id"])
                    if isinstance(inference_config.get("model_id"), str)
                    else None
                ),
            )
            tags = self._build_tags(results_config)

            with wandb.init(
                project=config.project,
                entity=config.entity,
                mode=config.mode,
                name=run_name,
                tags=tags,
                config=wandb_config,
                job_type="final-results",
            ) as run:
                aggregate_table = wandb.Table(
                    columns=["group", "metric", "value"],
                    data=aggregate_metric_rows,
                )
                table = wandb.Table(columns=self.table_columns, data=table_rows)
                payload = {
                    self.table_key: table,
                    self.aggregate_table_key: aggregate_table,
                }
                summary_metrics = self.build_summary_plot_metrics(scalar_metrics)
                run_summary_plot_metrics = self.build_run_summary_plot_metrics(
                    scalar_metrics=scalar_metrics,
                    wandb_config=wandb_config,
                )
                loss_points = self.build_training_loss_points(
                    wandb_config.get("configs", {}).get("model", {})
                )
                aggregate_metrics = {
                    **run_summary_plot_metrics,
                    **summary_metrics,
                }
                if aggregate_metrics:
                    run.log(aggregate_metrics)
                if loss_points:
                    for point in loss_points:
                        run.log(
                            {self.loss_metric_key: point["average_loss"]},
                            step=point["epoch"],
                        )
                run.log(payload)
                artifact = wandb.Artifact(
                    self._artifact_name(final_result.results_run_name),
                    type=self.artifact_type,
                )
                self._add_artifact_files(
                    artifact=artifact,
                    final_result=final_result,
                )
                run.log_artifact(artifact)
                return WandbFinalResultsLogResult(
                    status="logged",
                    run_id=getattr(run, "id", None),
                    run_url=getattr(run, "url", None),
                )
        except Exception as error:
            logger.warning(f"WandB final results logging failed: {error}")
            return WandbFinalResultsLogResult(
                status="failed",
                error_message=str(error),
            )

    @classmethod
    def build_scalar_metrics(
        cls,
        retrieval_metrics: dict[str, Any],
        reasoning_metrics: dict[str, Any],
    ) -> dict[str, float | int]:
        """Build WandB-safe scalar metric names."""
        mappings = {
            "retrieval_evaluated_instances": retrieval_metrics.get(
                "evaluated_instances"
            ),
            "retrieval_hits_at_1": retrieval_metrics.get("hits_at_1"),
            "retrieval_hits_at_1_count": retrieval_metrics.get("hits_at_1_count"),
            "retrieval_hits_at_5": retrieval_metrics.get("hits_at_5"),
            "retrieval_hits_at_5_count": retrieval_metrics.get("hits_at_5_count"),
            "retrieval_hits_at_10": retrieval_metrics.get("hits_at_10"),
            "retrieval_hits_at_10_count": retrieval_metrics.get("hits_at_10_count"),
            "retrieval_hits_at_candidate_limit": retrieval_metrics.get(
                "hits_at_candidate_limit"
            ),
            "retrieval_hits_at_candidate_limit_count": retrieval_metrics.get(
                "hits_at_candidate_limit_count"
            ),
            "retrieval_average_candidate_count": retrieval_metrics.get(
                "average_candidate_count"
            ),
            "retrieval_missing_gold_in_graph_count": retrieval_metrics.get(
                "missing_gold_in_graph_count"
            ),
            "retrieval_skipped_missing_gold_in_graph_count": retrieval_metrics.get(
                "skipped_missing_gold_in_graph_count"
            ),
            "answer_accuracy": reasoning_metrics.get("accuracy"),
            "answer_hit_rate": reasoning_metrics.get("hit_rate"),
            "answer_hits_at_1": reasoning_metrics.get("hits_at_1"),
            "answer_precision": reasoning_metrics.get("precision"),
            "answer_recall": reasoning_metrics.get("recall"),
            "answer_f1": reasoning_metrics.get("f1"),
            "grounding_grounded_explanation_rate": reasoning_metrics.get(
                "grounded_explanation_rate"
            ),
            "grounding_fully_grounded_explanation_rate": reasoning_metrics.get(
                "fully_grounded_explanation_rate"
            ),
            "ranking_ndcg_at_1": retrieval_metrics.get(
                "ndcg_at_1", reasoning_metrics.get("ndcg_at_1")
            ),
            "ranking_ndcg_at_5": retrieval_metrics.get(
                "ndcg_at_5", reasoning_metrics.get("ndcg_at_5")
            ),
            "ranking_ndcg_at_10": retrieval_metrics.get(
                "ndcg_at_10", reasoning_metrics.get("ndcg_at_10")
            ),
            "ranking_ndcg_at_candidate_limit": retrieval_metrics.get(
                "ndcg_at_candidate_limit",
                reasoning_metrics.get("ndcg_at_candidate_limit"),
            ),
        }
        for key in cls.retrieval_conditioned_metric_keys:
            value = reasoning_metrics.get(key)
            if not isinstance(value, int | float):
                value = retrieval_metrics.get(key)
            mappings[key] = value
        return {
            key: value
            for key, value in mappings.items()
            if isinstance(value, int | float)
        }

    @staticmethod
    def build_aggregate_metric_rows(
        scalar_metrics: dict[str, float | int],
    ) -> list[list[Any]]:
        """Convert aggregate scalar metrics into table rows."""
        rows: list[list[Any]] = []
        for metric_name, metric_value in scalar_metrics.items():
            group, _, short_name = metric_name.partition("_")
            rows.append([group or "metric", short_name or metric_name, metric_value])
        return rows

    @classmethod
    def build_summary_plot_metrics(
        cls,
        scalar_metrics: dict[str, float | int],
    ) -> dict[str, float | int]:
        """Build one-value summary metrics for WandB history plots."""
        return {
            f"{cls.summary_plot_prefix}/{metric_name}": metric_value
            for metric_name, metric_value in scalar_metrics.items()
        }

    @classmethod
    def build_run_summary_plot_metrics(
        cls,
        scalar_metrics: dict[str, float | int],
        wandb_config: dict[str, Any],
    ) -> dict[str, float | int]:
        """Build curated run-summary metrics for WandB history plots."""
        run_summary_keys = {
            "retrieval_evaluated_instances": "retrieval_evaluated_instances",
            "retrieval_hits_at_1": "retrieval_hits_at_1",
            "retrieval_hits_at_10": "retrieval_hits_at_10",
            "retrieval_hits_at_candidate_limit": "retrieval_hits_at_candidate_limit",
            "answer_hit_rate": "answer_hit_rate",
            "answer_f1": "answer_f1",
            "ranking_ndcg_at_10": "ranking_ndcg_at_10",
            "grounding_grounded_explanation_rate": "grounded_explanation_rate",
        }
        run_summary_keys.update(
            {
                "retrieval_gold_coverage": "retrieval_gold_coverage",
                "retrieval_full_gold_coverage_rate": (
                    "retrieval_full_gold_coverage"
                ),
                "reasoning_context_gold_coverage": (
                    "reasoning_context_gold_coverage"
                ),
                "reasoning_context_full_gold_coverage_rate": (
                    "reasoning_context_full_gold_coverage"
                ),
                "llm_exact_match_given_full_context": (
                    "llm_exact_match_given_full_context"
                ),
                "llm_omission_given_full_context_rate": (
                    "llm_omission_given_full_context"
                ),
                "llm_exact_match_given_full_retrieval": (
                    "llm_exact_match_given_full_retrieval"
                ),
                "llm_omission_given_full_retrieval_rate": (
                    "llm_omission_given_full_retrieval"
                ),
                "full_retrieval_complete_answer_rate": (
                    "full_retrieval_complete_answer"
                ),
                "full_retrieval_llm_omission_rate": (
                    "full_retrieval_llm_omission"
                ),
                "partial_retrieval_fully_utilized_rate": (
                    "partial_retrieval_fully_utilized"
                ),
                "partial_retrieval_underutilized_rate": (
                    "partial_retrieval_underutilized"
                ),
                "full_context_complete_answer_rate": (
                    "full_context_complete_answer"
                ),
                "full_context_llm_omission_rate": (
                    "full_context_llm_omission"
                ),
                "partial_context_fully_utilized_rate": (
                    "partial_context_fully_utilized"
                ),
                "partial_context_underutilized_rate": (
                    "partial_context_underutilized"
                ),
                "no_gold_retrieved_no_gold_answered_rate": (
                    "no_gold_retrieved_no_gold_answered"
                ),
                "correct_without_gold_retrieval_rate": (
                    "correct_without_gold_retrieval"
                ),
            }
        )
        return {
            f"{cls.run_summary_prefix}/{target_key}": scalar_metrics[source_key]
            for source_key, target_key in run_summary_keys.items()
            if source_key in scalar_metrics
        }

    @staticmethod
    def build_training_loss_points(
        model_config: dict[str, Any],
    ) -> list[dict[str, float | int]]:
        """Build scalar loss history points for WandB logging."""
        training = model_config.get("training", {})
        if not isinstance(training, dict):
            training = {}
        loss_history = training.get("loss_history") or model_config.get("loss_history")
        if not isinstance(loss_history, list):
            return []

        points: list[dict[str, float | int]] = []
        for item in loss_history:
            if not isinstance(item, dict):
                continue
            epoch = item.get("epoch")
            average_loss = item.get("average_loss")
            if isinstance(epoch, int) and isinstance(average_loss, int | float):
                points.append(
                    {
                        "epoch": epoch,
                        "average_loss": average_loss,
                    }
                )
        return points

    def build_table_rows(
        self,
        results_config: dict[str, Any],
        per_instance_rows: list[dict[str, Any]],
    ) -> list[list[Any]]:
        """Build final-result rows with the GNN candidates from predictions."""
        answers_by_index = self._load_answers_by_index(results_config)
        retrieved_candidates_by_index = self._load_retrieved_candidates_by_index(
            results_config
        )
        table_rows: list[list[Any]] = []
        for row in per_instance_rows:
            instance_index = row.get("instance_index")
            answer_row = answers_by_index.get(instance_index, {})
            table_rows.append(
                [
                    instance_index,
                    row.get("question", ""),
                    self._format_table_cell(row.get("q_entity", [])),
                    self._format_table_cell(row.get("gold_answers", [])),
                    self._format_table_cell(row.get("predicted_answers", [])),
                    answer_row.get("explanation", ""),
                    row.get("exact_match", False),
                    row.get("hit", False),
                    row.get("hits_at_1", False),
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1", 0.0),
                    row.get("mentioned_triple_count", 0),
                    row.get("grounded_mentioned_triple_count", 0),
                    row.get("grounded_explanation", False),
                    row.get("fully_grounded_explanation", False),
                    row.get("ndcg_at_1", 0.0),
                    row.get("ndcg_at_5", 0.0),
                    row.get("ndcg_at_10", 0.0),
                    row.get("ndcg_at_candidate_limit", 0.0),
                    row.get("answer_error_message"),
                    row.get("retrieval_gold_coverage", 0.0),
                    row.get("reasoning_context_gold_coverage", 0.0),
                    row.get("llm_retrieved_gold_utilization"),
                    row.get("retrieval_generation_outcome", ""),
                    self._format_table_cell(
                        retrieved_candidates_by_index.get(instance_index, [])
                    ),
                ]
            )
        return table_rows

    @classmethod
    def _load_retrieved_candidates_by_index(
        cls,
        results_config: dict[str, Any],
    ) -> dict[int, list[str]]:
        """Load only ranked candidate node names from the retriever predictions."""
        predictions_path_value = cls._result_artifact_path(
            results_config,
            "predictions_path",
        )
        if not isinstance(predictions_path_value, str):
            return {}
        predictions_path = project_absolute_path(predictions_path_value)
        if not predictions_path.exists():
            return {}

        candidates_by_index: dict[int, list[str]] = {}
        try:
            prediction_rows = cls._load_jsonl_objects(predictions_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        for prediction in prediction_rows:
            instance_index = prediction.get("instance_index")
            raw_candidates = prediction.get("answer_candidates", [])
            if not isinstance(instance_index, int) or not isinstance(raw_candidates, list):
                continue
            candidates_by_index[instance_index] = [
                str(candidate["node"])
                for candidate in raw_candidates
                if isinstance(candidate, dict) and candidate.get("node") is not None
            ]
        return candidates_by_index

    @classmethod
    def _format_table_cell(cls, value: Any) -> Any:
        """Convert nested values to readable table text for WandB."""
        if isinstance(value, list):
            return ", ".join(cls._format_table_cell(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    def build_wandb_config(
        self,
        final_result: FinalResultsEvaluationResult,
        results_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one deduplicated WandB config payload for run filtering."""
        evaluation_config = self._load_optional_json_object(
            self._result_config_path(results_config, "evaluation_config_path")
        )
        inference_config = self._load_optional_json_object(
            self._result_config_path(results_config, "inference_config_path")
        )
        model_config = self._load_model_config(
            results_config=results_config,
            evaluation_config=evaluation_config,
        )
        model_ref = evaluation_config.get("model_config", {})
        if not isinstance(model_ref, dict):
            model_ref = {}
        inference_ref = inference_config.get("evaluation_config", {})
        if not isinstance(inference_ref, dict):
            inference_ref = {}
        inference_payload = inference_config.get("inference", {})
        if not isinstance(inference_payload, dict):
            inference_payload = inference_config
        elif "total_instances" in inference_config and "total_instances" not in inference_payload:
            inference_payload = {
                **inference_payload,
                "total_instances": inference_config["total_instances"],
            }
        evidence_payload = inference_payload.get("evidence_subgraph")
        inference_payload = {
            key: value
            for key, value in inference_payload.items()
            if key not in {"evidence_metrics", "evidence_subgraph"}
        }

        model_run_name = model_ref.get("model_run_name") or results_config.get("model_run_name")
        model_run_number = self._int_or_none(model_ref.get("model_run_number")) or self._extract_run_number(model_run_name)
        evaluation_run_name = inference_ref.get("evaluation_run_name") or results_config.get("evaluation_run_name")
        evaluation_run_number = self._int_or_none(inference_ref.get("evaluation_run_number")) or self._extract_run_number(evaluation_run_name)
        inference_run_name = inference_config.get("run_name") or results_config.get("inference_run_name")
        inference_run_number = self._int_or_none(inference_config.get("run_number")) or self._extract_run_number(inference_run_name)
        source_paths = self._build_source_paths(results_config)
        return self._stringify_paths(
            {
                "dataset_id": results_config.get("dataset_id"),
                **(
                    {"llm_provider": results_config["llm_provider"]}
                    if results_config.get("llm_provider")
                    else {}
                ),
                "model_id": results_config.get("model_id"),
                "runs": {
                    "model": {
                        "name": model_run_name,
                        "number": model_run_number,
                    },
                    "evaluation": {
                        "name": evaluation_run_name,
                        "number": evaluation_run_number,
                    },
                    "inference": {
                        "name": inference_run_name,
                        "number": inference_run_number,
                    },
                    "results": {
                        "name": final_result.results_run_name,
                        "number": final_result.results_run_number,
                    },
                },
                "configs": {
                    "model": model_config,
                    "evaluation": evaluation_config.get("evaluation", {}),
                    **(
                        {"evidence": evidence_payload}
                        if isinstance(evidence_payload, dict)
                        else {}
                    ),
                    "inference": inference_payload,
                },
                "source_paths": source_paths,
            }
        )

    def _load_answers_by_index(
        self,
        results_config: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        answers_path = self._result_artifact_path(results_config, "answers_path")
        if not isinstance(answers_path, str):
            return {}
        path = project_absolute_path(answers_path)
        if not path.exists():
            return {}
        return {
            row["instance_index"]: row
            for row in self._load_jsonl_objects(path)
            if isinstance(row.get("instance_index"), int)
        }

    @staticmethod
    def _result_config_path(results_config: dict[str, Any], key: str) -> str | None:
        configs = results_config.get("configs")
        if isinstance(configs, dict) and isinstance(configs.get(key), str):
            return configs[key]
        value = results_config.get(key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _result_artifact_path(results_config: dict[str, Any], key: str) -> str | None:
        artifacts = results_config.get("artifacts")
        if isinstance(artifacts, dict) and isinstance(artifacts.get(key), str):
            return artifacts[key]
        if isinstance(artifacts, dict):
            for stage_artifacts in artifacts.values():
                if (
                    isinstance(stage_artifacts, dict)
                    and isinstance(stage_artifacts.get(key), str)
                ):
                    return stage_artifacts[key]
        value = results_config.get(key)
        return value if isinstance(value, str) else None

    def _load_model_config(
        self,
        results_config: dict[str, Any],
        evaluation_config: dict[str, Any],
    ) -> dict[str, Any]:
        model_config_path_value = self._result_config_path(results_config, "model_config_path")
        if isinstance(model_config_path_value, str):
            model_config_path = project_absolute_path(model_config_path_value)
            if model_config_path.exists():
                return normalize_model_config(
                    self._without_wandb_tracking(
                        self._load_json_object(model_config_path)
                    )
                )

        model_run_directory = results_config.get("model_run_directory")
        if isinstance(model_run_directory, str):
            for filename in [
                "model_config.json",
                "model.config",
                "gnn_answer_retriever_config.json",
            ]:
                model_config_path = project_absolute_path(model_run_directory) / filename
                if model_config_path.exists():
                    return normalize_model_config(
                        self._without_wandb_tracking(
                            self._load_json_object(model_config_path)
                        )
                    )

        model_configuration = evaluation_config.get("model_configuration")
        if isinstance(model_configuration, dict):
            return normalize_model_config(
                self._without_wandb_tracking(model_configuration)
            )

        return {}

    @staticmethod
    def _without_wandb_tracking(config: dict[str, Any]) -> dict[str, Any]:
        """Keep persisted lineage metadata out of the user-facing Config tab."""
        cleaned = dict(config)
        cleaned.pop("wandb", None)
        return cleaned

    @classmethod
    def _build_source_paths(cls, results_config: dict[str, Any]) -> dict[str, str]:
        path_keys = {
            *cls.source_path_keys,
            "model_run_directory",
            "evaluation_run_directory",
            "inference_run_directory",
            "answers_path",
            "reasoning_path",
            "predictions_path",
        }
        paths: dict[str, str] = {}
        for collection_key in ["configs", "artifacts"]:
            collection = results_config.get(collection_key)
            if isinstance(collection, dict):
                for key, value in collection.items():
                    if isinstance(value, str) and cls._is_path_key(key):
                        paths[key] = value
                    elif isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if (
                                isinstance(nested_value, str)
                                and cls._is_path_key(nested_key)
                            ):
                                paths[f"{key}_{nested_key}"] = nested_value
        for key in sorted(path_keys):
            value = results_config.get(key)
            if isinstance(value, str):
                paths[key] = value
        return paths

    @staticmethod
    def _is_path_key(key: str) -> bool:
        return key.endswith("_path") or key.endswith("_directory")

    @classmethod
    def _build_tags(cls, results_config: dict[str, Any]) -> list[str]:
        tags = ["graphragx"]
        for key in ["dataset_id", "llm_provider", "model_id"]:
            value = results_config.get(key)
            if value:
                tags.append(str(value))
        architecture = results_config.get("gnn_architecture")
        if architecture:
            tags.append(str(architecture))

        model_config_path = cls._result_config_path(results_config, "model_config_path")
        if isinstance(model_config_path, str):
            try:
                model_config = normalize_model_config(
                    cls._load_json_object(project_absolute_path(model_config_path))
                )
                if not architecture:
                    tags.append(infer_gnn_architecture(model_config))
                embedding_model_id = model_config.get("embedding_model")
                if embedding_model_id:
                    tags.append(str(embedding_model_id))
                training = model_config.get("training", {})
                trained_instances = (
                    training.get("trained_instances")
                    if isinstance(training, dict)
                    else None
                )
                if isinstance(trained_instances, dict):
                    trained_instances = trained_instances.get("count")
                if trained_instances is None:
                    trained_instances = model_config.get("trained_instances")
                if trained_instances is not None:
                    tags.append(f"trained_instances:{trained_instances}")
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        runs = results_config.get("runs")
        if isinstance(runs, dict):
            for stage, run_payload in runs.items():
                if not isinstance(run_payload, dict):
                    continue
                run_number = run_payload.get("number")
                if run_number is not None:
                    tags.append(f"{stage}_run_number:{run_number}")

        evaluation_config_path = cls._result_config_path(
            results_config,
            "evaluation_config_path",
        )
        if isinstance(evaluation_config_path, str):
            try:
                evaluation_config = cls._load_json_object(
                    project_absolute_path(evaluation_config_path)
                )
                model_config = evaluation_config.get("model_config", {})
                if (
                    isinstance(model_config, dict)
                    and model_config.get("model_run_number") is not None
                ):
                    tags.append(f"model_run_number:{model_config['model_run_number']}")
                model_run = evaluation_config.get("model_run", {})
                if isinstance(model_run, dict) and model_run.get("number") is not None:
                    tags.append(f"model_run_number:{model_run['number']}")
                if evaluation_config.get("run_number") is not None:
                    tags.append(f"evaluation_run_number:{evaluation_config['run_number']}")
                if evaluation_config.get("evaluated_instances") is not None:
                    tags.append(
                        f"evaluated_instances:{evaluation_config['evaluated_instances']}"
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        inference_config_path = cls._result_config_path(
            results_config,
            "inference_config_path",
        )
        if isinstance(inference_config_path, str):
            try:
                inference_config = cls._load_json_object(
                    project_absolute_path(inference_config_path)
                )
                if inference_config.get("run_number") is not None:
                    tags.append(f"inference_run_number:{inference_config['run_number']}")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return list(dict.fromkeys(tags))

    @classmethod
    def _add_artifact_files(
        cls,
        artifact: Any,
        final_result: FinalResultsEvaluationResult,
    ) -> None:
        """Upload only final-result-owned files; upstream artifacts remain references."""
        for path in sorted(final_result.results_run_directory.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=f"results/{path.name}")

    @staticmethod
    def _artifact_name(results_run_name: str) -> str:
        sanitized = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in results_run_name
        ).strip("._-")
        return f"results-{sanitized or 'run'}"

    @classmethod
    def _stringify_paths(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._stringify_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._stringify_paths(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON file {path} must contain an object.")
        return value

    @classmethod
    def _load_optional_json_object(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, str):
            return {}
        path = project_absolute_path(value)
        if not path.exists():
            return {}
        return cls._load_json_object(path)

    @staticmethod
    def _extract_run_number(value: Any) -> int | None:
        if not isinstance(value, str):
            return None
        match = re.match(r"^(\d+)_", value)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        return None

    @staticmethod
    def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} must be an object.")
            rows.append(value)
        return rows
