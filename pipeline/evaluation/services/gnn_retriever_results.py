"""Persistence-independent metrics and loading for retriever evaluation runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from string import punctuation
from typing import Any

from helpers.constants import (
    GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME,
    GNN_ANSWER_RETRIEVER_EVALUATION_PREDICTIONS_FILENAME,
)
from helpers.path_serialization import project_absolute_path
from pipeline.evaluation.models import (
    EvaluatedAnswerRetrievalInstance,
    GnnAnswerRetrieverEvaluationResult,
    GnnAnswerRetrieverMetrics,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    SavedGnnAnswerRetrieverConfig,
)
from pipeline.preparation.helpers.gnn_architecture import infer_gnn_architecture
from pipeline.services import AbstractService


class GnnRetrieverResultsService(AbstractService):
    """Compute metrics and resolve persisted GNN retriever evaluation runs."""

    whitespace_pattern = re.compile(r"\s+")
    article_pattern = re.compile(r"\b(a|an|the)\b")

    def build_metrics(
        self,
        *,
        dataset_id: str,
        model_run_name: str,
        model_run_number: int,
        predictions: list[EvaluatedAnswerRetrievalInstance],
        candidate_limit: int,
        missing_gold_in_graph_count: int | None = None,
        skipped_missing_gold_in_graph_count: int = 0,
        evaluation_run_name: str | None = None,
        evaluation_run_number: int | None = None,
    ) -> GnnAnswerRetrieverMetrics:
        """Compute aggregate retrieval metrics from persisted prediction models."""
        evaluated_instances = len(predictions)
        if evaluated_instances == 0:
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever metrics require at least one prediction."
            )
        if candidate_limit <= 0:
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever candidate limit must be greater than zero."
            )

        def hit_count(limit: int) -> int:
            return sum(
                any(candidate.is_gold_answer for candidate in item.answer_candidates[:limit])
                for item in predictions
            )

        hits_at_1_count = hit_count(1)
        hits_at_5_count = hit_count(5)
        hits_at_10_count = hit_count(10)
        hits_at_candidate_limit_count = hit_count(candidate_limit)
        total_candidates = sum(len(item.answer_candidates) for item in predictions)
        missing_gold_count = (
            missing_gold_in_graph_count
            if missing_gold_in_graph_count is not None
            else sum(item.missing_gold_in_graph for item in predictions)
        )
        ndcg_at_1 = self._mean_ndcg(predictions, 1)
        ndcg_at_5 = self._mean_ndcg(predictions, 5)
        ndcg_at_10 = self._mean_ndcg(predictions, 10)
        ndcg_at_candidate_limit = self._mean_ndcg(predictions, candidate_limit)
        coverage_values: list[float] = []
        full_gold_coverage_count = 0
        retrieved_gold_answer_count = 0
        for prediction in predictions:
            gold_answers = self._normalize_answer_set(prediction.a_entity)
            if not gold_answers:
                continue
            retrieved_candidates = self._normalize_answer_set(
                [candidate.node for candidate in prediction.answer_candidates]
            )
            retrieved_gold = gold_answers & retrieved_candidates
            coverage_values.append(len(retrieved_gold) / len(gold_answers))
            full_gold_coverage_count += int(retrieved_gold == gold_answers)
            retrieved_gold_answer_count += len(retrieved_gold)
        conditioned_evaluated_instances = len(coverage_values)
        return GnnAnswerRetrieverMetrics(
            dataset_id=dataset_id,
            model_run_name=model_run_name,
            model_run_number=model_run_number,
            evaluation_run_name=evaluation_run_name,
            evaluation_run_number=evaluation_run_number,
            evaluated_instances=evaluated_instances,
            hits_at_1=hits_at_1_count / evaluated_instances,
            hits_at_1_count=hits_at_1_count,
            hits_at_5=hits_at_5_count / evaluated_instances,
            hits_at_5_count=hits_at_5_count,
            hits_at_10=hits_at_10_count / evaluated_instances,
            hits_at_10_count=hits_at_10_count,
            hits_at_candidate_limit=(
                hits_at_candidate_limit_count / evaluated_instances
            ),
            hits_at_candidate_limit_count=hits_at_candidate_limit_count,
            ndcg_at_1=ndcg_at_1,
            ndcg_at_5=ndcg_at_5,
            ndcg_at_10=ndcg_at_10,
            ndcg_at_candidate_limit=ndcg_at_candidate_limit,
            conditioned_evaluated_instances=conditioned_evaluated_instances,
            retrieval_gold_coverage=(
                sum(coverage_values) / conditioned_evaluated_instances
                if conditioned_evaluated_instances
                else 0.0
            ),
            retrieval_full_gold_coverage_count=full_gold_coverage_count,
            retrieval_full_gold_coverage_rate=(
                full_gold_coverage_count / conditioned_evaluated_instances
                if conditioned_evaluated_instances
                else 0.0
            ),
            retrieved_gold_answer_count=retrieved_gold_answer_count,
            candidate_limit=candidate_limit,
            average_candidate_count=total_candidates / evaluated_instances,
            missing_gold_in_graph_count=missing_gold_count,
            skipped_missing_gold_in_graph_count=(
                skipped_missing_gold_in_graph_count
            ),
        )

    @classmethod
    def _normalize_answer_set(cls, answers: list[str]) -> set[str]:
        """Normalize answer identities exactly like final-results evaluation."""
        normalized: set[str] = set()
        for answer in answers:
            value = answer.lower().strip()
            value = value.translate(str.maketrans("", "", punctuation))
            value = cls.article_pattern.sub(" ", value)
            value = cls.whitespace_pattern.sub(" ", value).strip()
            if value:
                normalized.add(value)
        return normalized

    @classmethod
    def ndcg_at_k(
        cls,
        prediction: EvaluatedAnswerRetrievalInstance,
        k: int,
    ) -> float:
        """Calculate binary-relevance nDCG from persisted retriever rank order."""
        if k <= 0 or not prediction.answer_candidates:
            return 0.0
        relevances = [
            1.0 if candidate.is_gold_answer else 0.0
            for candidate in prediction.answer_candidates[:k]
        ]
        dcg = sum(
            relevance / math.log2(rank + 2)
            for rank, relevance in enumerate(relevances)
        )
        ideal_relevant_count = min(len(set(prediction.a_entity)), k)
        if ideal_relevant_count <= 0:
            return 0.0
        idcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(ideal_relevant_count)
        )
        return dcg / idcg

    @classmethod
    def _mean_ndcg(
        cls,
        predictions: list[EvaluatedAnswerRetrievalInstance],
        k: int,
    ) -> float:
        return sum(cls.ndcg_at_k(prediction, k) for prediction in predictions) / len(
            predictions
        )

    def load_run(
        self,
        *,
        evaluation_root: Path,
        dataset_id: str,
        run_name: str | None,
        run_number: int | None,
    ) -> GnnAnswerRetrieverEvaluationResult:
        """Resolve and validate a persisted retriever run by name or number."""
        run_directory = self.resolve_run_directory(
            evaluation_root=evaluation_root,
            run_name=run_name,
            run_number=run_number,
        )
        config_path = run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME
        predictions_path = (
            run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_PREDICTIONS_FILENAME
        )
        metrics_path = run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME
        config = self._load_json(config_path, "retriever evaluation config")
        predictions = self._load_predictions(predictions_path)
        if config.get("dataset_id") != dataset_id:
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever run dataset {config.get('dataset_id')} does not match {dataset_id}."
            )

        model_ref = config.get("model_config")
        if not isinstance(model_ref, dict):
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run config is missing its model reference."
            )
        model_run_name = str(model_ref.get("model_run_name") or "")
        try:
            model_run_number = int(model_ref.get("model_run_number") or 0)
        except (TypeError, ValueError) as error:
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run contains an invalid model run number."
            ) from error
        model_config_path_value = model_ref.get("full_config_path")
        if not model_run_name or not isinstance(model_config_path_value, str):
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run contains an invalid model reference."
            )
        model_config_path = project_absolute_path(model_config_path_value)
        if not model_config_path.exists():
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever model config does not exist: {model_config_path}"
            )
        model_run_directory = model_config_path.parent
        evaluation = config.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run contains an invalid evaluation configuration."
            )
        try:
            candidate_limit = int(
                evaluation.get(
                    "candidate_limit",
                    len(predictions[0].answer_candidates),
                )
            )
        except (TypeError, ValueError) as error:
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run contains an invalid candidate limit."
            ) from error
        if metrics_path.exists():
            try:
                persisted_metrics_payload = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                metrics = GnnAnswerRetrieverMetrics.model_validate(
                    persisted_metrics_payload
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise GnnAnswerRetrieverEvaluationException(
                    f"Retriever metrics are invalid: {metrics_path}"
                ) from error
            if (
                metrics.dataset_id != dataset_id
                or metrics.model_run_name != model_run_name
                or metrics.evaluated_instances != len(predictions)
            ):
                raise GnnAnswerRetrieverEvaluationException(
                    "Retriever metrics do not match the run configuration and predictions."
                )
            metrics = self.build_metrics(
                dataset_id=dataset_id,
                model_run_name=model_run_name,
                model_run_number=model_run_number,
                predictions=predictions,
                candidate_limit=candidate_limit,
                missing_gold_in_graph_count=metrics.missing_gold_in_graph_count,
                skipped_missing_gold_in_graph_count=(
                    metrics.skipped_missing_gold_in_graph_count
                    if "skipped_missing_gold_in_graph_count"
                    in persisted_metrics_payload
                    else (
                        metrics.missing_gold_in_graph_count
                        if bool(evaluation.get("skip_missing_gold_in_graph", True))
                        else 0
                    )
                ),
                evaluation_run_name=run_directory.name,
                evaluation_run_number=self._extract_run_number(run_directory.name),
            )
            persisted_metrics_path: Path | None = metrics_path
        else:
            metrics = self.build_metrics(
                dataset_id=dataset_id,
                model_run_name=model_run_name,
                model_run_number=model_run_number,
                predictions=predictions,
                candidate_limit=candidate_limit,
                evaluation_run_name=run_directory.name,
                evaluation_run_number=self._extract_run_number(run_directory.name),
            )
            persisted_metrics_path = None

        tracking = config.get("wandb", {})
        if not isinstance(tracking, dict):
            tracking = {}
        model_config_payload = self._load_json(
            model_config_path,
            "retriever model config",
        )
        model_architecture = infer_gnn_architecture(model_config_payload)
        if model_architecture == "rgcn":
            # Inference-only does not load the GNN weights, but its upstream
            # R-GCN lineage must still include a valid authoritative vocabulary.
            GnnAnswerRetrieverModelRunService().resolve_run(
                model_root=model_run_directory.parent,
                run_name=model_run_directory.name,
                run_number=None,
            )
        return GnnAnswerRetrieverEvaluationResult(
            dataset_id=dataset_id,
            gnn_architecture=str(model_architecture),
            model_run_directory=model_run_directory,
            model_run_name=model_run_name,
            model_run_number=model_run_number,
            evaluation_run_directory=run_directory,
            evaluation_run_name=run_directory.name,
            evaluation_run_number=self._extract_run_number(run_directory.name),
            evaluated_instances=metrics.evaluated_instances,
            hits_at_1=metrics.hits_at_1,
            hits_at_1_count=metrics.hits_at_1_count,
            hits_at_5=metrics.hits_at_5,
            hits_at_5_count=metrics.hits_at_5_count,
            hits_at_10=metrics.hits_at_10,
            hits_at_10_count=metrics.hits_at_10_count,
            hits_at_candidate_limit=metrics.hits_at_candidate_limit,
            hits_at_candidate_limit_count=metrics.hits_at_candidate_limit_count,
            ndcg_at_1=metrics.ndcg_at_1,
            ndcg_at_5=metrics.ndcg_at_5,
            ndcg_at_10=metrics.ndcg_at_10,
            ndcg_at_candidate_limit=metrics.ndcg_at_candidate_limit,
            conditioned_evaluated_instances=(
                metrics.conditioned_evaluated_instances
            ),
            retrieval_gold_coverage=metrics.retrieval_gold_coverage,
            retrieval_full_gold_coverage_count=(
                metrics.retrieval_full_gold_coverage_count
            ),
            retrieval_full_gold_coverage_rate=(
                metrics.retrieval_full_gold_coverage_rate
            ),
            retrieved_gold_answer_count=metrics.retrieved_gold_answer_count,
            average_candidate_count=metrics.average_candidate_count,
            missing_gold_in_graph_count=metrics.missing_gold_in_graph_count,
            skipped_missing_gold_in_graph_count=(
                metrics.skipped_missing_gold_in_graph_count
            ),
            predictions_path=predictions_path,
            evaluation_config_path=config_path,
            retrieval_metrics_path=persisted_metrics_path,
            wandb_status=tracking.get("status"),
            wandb_run_id=tracking.get("run_id"),
            wandb_run_url=tracking.get("run_url"),
            wandb_error_message=tracking.get("error_message"),
        )

    def load_model_config(
        self,
        *,
        evaluation_root: Path,
        run_name: str | None,
        run_number: int | None,
    ) -> SavedGnnAnswerRetrieverConfig:
        """Load the model configuration referenced by a retriever run."""
        run_directory = self.resolve_run_directory(
            evaluation_root=evaluation_root,
            run_name=run_name,
            run_number=run_number,
        )
        evaluation_config = self._load_json(
            run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME,
            "retriever evaluation config",
        )
        model_ref = evaluation_config.get("model_config")
        if not isinstance(model_ref, dict):
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run config is missing its model reference."
            )
        model_config_value = model_ref.get("full_config_path")
        if not isinstance(model_config_value, str):
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever run contains an invalid model config path."
            )
        model_config_path = project_absolute_path(model_config_value)
        try:
            model_config = SavedGnnAnswerRetrieverConfig.model_validate_json(
                model_config_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever model config is invalid: {model_config_path}"
            ) from error
        evaluation_dataset_id = evaluation_config.get("dataset_id")
        if evaluation_dataset_id != model_config.dataset_id:
            raise GnnAnswerRetrieverEvaluationException(
                "Retriever evaluation and model configurations reference different datasets."
            )
        return model_config

    def resolve_run_directory(
        self,
        *,
        evaluation_root: Path,
        run_name: str | None,
        run_number: int | None,
    ) -> Path:
        """Resolve a retriever run directory with model-run-compatible selectors."""
        if run_name is not None and run_number is not None:
            raise GnnAnswerRetrieverEvaluationException(
                "Select a retriever run by name or number, not both."
            )
        if run_name is None and run_number is None:
            raise GnnAnswerRetrieverEvaluationException(
                "A retriever run name or number is required."
            )
        if not evaluation_root.exists():
            raise GnnAnswerRetrieverEvaluationException(
                f"No retriever runs exist under {evaluation_root}."
            )
        candidates = [path for path in evaluation_root.iterdir() if path.is_dir()]
        if run_number is not None:
            matches = [
                path for path in candidates
                if self._extract_run_number(path.name) == run_number
            ]
        else:
            matches = [
                path for path in candidates
                if path.name == run_name or self._extract_run_label(path.name) == run_name
            ]
        if not matches:
            selector = f"number {run_number}" if run_number is not None else f"name '{run_name}'"
            raise GnnAnswerRetrieverEvaluationException(
                f"No retriever run matched {selector}."
            )
        return max(matches, key=lambda path: self._extract_run_number(path.name))

    @staticmethod
    def _load_json(path: Path, label: str) -> dict[str, Any]:
        if not path.exists():
            raise GnnAnswerRetrieverEvaluationException(f"Missing {label}: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Invalid {label}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise GnnAnswerRetrieverEvaluationException(f"Invalid {label}: {path}")
        return payload

    @staticmethod
    def _load_predictions(path: Path) -> list[EvaluatedAnswerRetrievalInstance]:
        if not path.exists():
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever run is missing predictions: {path}"
            )
        try:
            predictions = [
                EvaluatedAnswerRetrievalInstance.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever predictions are invalid: {path}"
            ) from error
        if not predictions:
            raise GnnAnswerRetrieverEvaluationException(
                f"Retriever predictions are empty: {path}"
            )
        return predictions

    @staticmethod
    def _extract_run_number(name: str) -> int:
        match = re.match(r"^(\d+)_", name)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_run_label(name: str) -> str:
        match = re.match(r"^\d+_(.+)$", name)
        return match.group(1) if match else name
