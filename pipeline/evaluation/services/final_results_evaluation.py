"""Final results computation and storage for retrieval plus reasoning runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from string import punctuation
from typing import Any

from pydantic import BaseModel, Field

from helpers.constants import (
    FINAL_RESULTS_CONFIG_FILENAME,
    FINAL_RESULTS_PER_INSTANCE_FILENAME,
    FINAL_RESULTS_REASONING_METRICS_FILENAME,
    FINAL_RESULTS_RETRIEVAL_METRICS_FILENAME,
)
from helpers.path_serialization import make_project_paths_relative
from pipeline.evaluation.exceptions import FinalResultsEvaluationException
from pipeline.evaluation.models import (
    EvaluatedAnswerRetrievalInstance,
    FinalAnswerMetrics,
    FinalReasoningMetrics,
    FinalResultsConfig,
    GnnAnswerRetrieverEvaluationResult,
    PerInstanceFinalResult,
    RankingMetrics,
    RetrievalConditionedAnswerMetrics,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.models.final_results import ExplanationGroundingMetrics
from pipeline.evaluation.services.gnn_retriever_results import (
    GnnRetrieverResultsService,
)
from pipeline.preparation.helpers.gnn_architecture import infer_gnn_architecture
from pipeline.services import AbstractService


class FinalResultsStorageResult(BaseModel):
    """Paths and version metadata produced by final results storage."""

    results_run_directory: Path
    results_run_name: str
    results_run_number: int
    results_config_path: Path
    retrieval_metrics_path: Path
    reasoning_metrics_path: Path
    per_instance_results_path: Path


class FinalResultsEvaluationOutcome(BaseModel):
    """Computed final metrics and persisted results metadata."""

    storage_result: FinalResultsStorageResult
    reasoning_metrics: FinalReasoningMetrics
    per_instance_results: list[PerInstanceFinalResult] = Field(default_factory=list)


class FinalResultsEvaluationService(AbstractService):
    """Compute and persist final WebQSP retrieval and reasoning metrics."""

    results_config_filename = FINAL_RESULTS_CONFIG_FILENAME
    retrieval_metrics_filename = FINAL_RESULTS_RETRIEVAL_METRICS_FILENAME
    reasoning_metrics_filename = FINAL_RESULTS_REASONING_METRICS_FILENAME
    per_instance_results_filename = FINAL_RESULTS_PER_INSTANCE_FILENAME
    unknown_answer_values = {"", "unknown", "n/a", "none", "null"}
    article_pattern = re.compile(r"\b(a|an|the)\b")
    whitespace_pattern = re.compile(r"\s+")
    arrow_triple_pattern = re.compile(
        r"([^.;\n]+?)\s*->\s*([A-Za-z0-9_./-]+)\s*->\s*([^.;\n]+)"
    )

    def __init__(
        self,
        retriever_results_service: GnnRetrieverResultsService | None = None,
    ) -> None:
        self.retriever_results_service = (
            retriever_results_service or GnnRetrieverResultsService()
        )

    def evaluate(
        self,
        gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult,
        llm_inference_run: SavedLlmInferenceRun,
    ) -> FinalResultsEvaluationOutcome:
        """Compute final results from saved retrieval and inference artifacts."""
        answers = self._load_jsonl_objects(llm_inference_run.answers_path)
        reasoning_rows = self._load_jsonl_objects(llm_inference_run.reasoning_path)
        predictions = [
            EvaluatedAnswerRetrievalInstance.model_validate(item)
            for item in self._load_jsonl_objects(gnn_evaluation_result.predictions_path)
        ]
        evaluation_config = self._load_json_object(
            gnn_evaluation_result.evaluation_config_path
        )
        candidate_limit = self._extract_candidate_limit(evaluation_config)

        answers_by_index = self._index_rows(answers, "answers")
        reasoning_by_index = self._index_rows(reasoning_rows, "reasoning")
        predictions_by_index = {
            prediction.instance_index: prediction
            for prediction in predictions
        }
        self._validate_index_sets(
            answers_by_index=answers_by_index,
            reasoning_by_index=reasoning_by_index,
            predictions_by_index=predictions_by_index,
        )

        per_instance_results = [
            self._build_per_instance_result(
                instance_index=instance_index,
                answer_row=answers_by_index[instance_index],
                reasoning_row=reasoning_by_index[instance_index],
                prediction=predictions_by_index[instance_index],
                candidate_limit=candidate_limit,
            )
            for instance_index in sorted(answers_by_index)
        ]
        retrieval_metrics = self.retriever_results_service.build_metrics(
            dataset_id=gnn_evaluation_result.dataset_id,
            model_run_name=gnn_evaluation_result.model_run_name,
            model_run_number=gnn_evaluation_result.model_run_number,
            predictions=predictions,
            candidate_limit=candidate_limit,
            missing_gold_in_graph_count=(
                gnn_evaluation_result.missing_gold_in_graph_count
            ),
            skipped_missing_gold_in_graph_count=(
                gnn_evaluation_result.skipped_missing_gold_in_graph_count
            ),
            evaluation_run_name=gnn_evaluation_result.evaluation_run_name,
            evaluation_run_number=gnn_evaluation_result.evaluation_run_number,
        ).model_dump(mode="json")
        reasoning_metrics = self._build_reasoning_metrics(
            gnn_evaluation_result=gnn_evaluation_result,
            llm_inference_run=llm_inference_run,
            per_instance_results=per_instance_results,
            retrieval_metrics=retrieval_metrics,
            candidate_limit=candidate_limit,
        )
        model_config_path = (
            gnn_evaluation_result.model_run_directory / "model_config.json"
        )
        gnn_architecture = self._build_gnn_architecture(model_config_path)
        results_config = FinalResultsConfig(
            dataset_id=llm_inference_run.dataset_id,
            model_run_name=gnn_evaluation_result.model_run_name,
            evaluation_run_name=gnn_evaluation_result.evaluation_run_name,
            inference_run_name=llm_inference_run.inference_run_name,
            model_id=llm_inference_run.model_id,
            llm_provider=llm_inference_run.llm_provider,
            gnn_architecture=gnn_architecture,
            configs={
                "model_config_path": model_config_path,
                "evaluation_config_path": gnn_evaluation_result.evaluation_config_path,
                "inference_config_path": llm_inference_run.inference_config_path,
            },
            artifacts={
                "training": {
                    "name": gnn_evaluation_result.model_run_name,
                    "model_config_path": str(model_config_path),
                    "weights_path": str(
                        gnn_evaluation_result.model_run_directory
                        / "gnn_answer_retriever.pt"
                    ),
                    **(
                        {
                            "relation_vocabulary_path": str(
                                gnn_evaluation_result.model_run_directory
                                / "relation_vocabulary.json"
                            )
                        }
                        if (
                            gnn_evaluation_result.model_run_directory
                            / "relation_vocabulary.json"
                        ).exists()
                        else {}
                    ),
                },
                "evaluation": {
                    "name": gnn_evaluation_result.evaluation_run_name,
                    "evaluation_config_path": str(
                        gnn_evaluation_result.evaluation_config_path
                    ),
                    "predictions_path": str(gnn_evaluation_result.predictions_path),
                },
                "inference": {
                    "name": llm_inference_run.inference_run_name,
                    "evidence_subgraph": llm_inference_run.evidence_subgraph,
                    "inference_config_path": str(
                        llm_inference_run.inference_config_path
                    ),
                    "answers_path": str(llm_inference_run.answers_path),
                    "reasoning_path": str(llm_inference_run.reasoning_path),
                },
            },
        )
        storage_result = self._save_results_run(
            results_root=llm_inference_run.inference_run_directory.parent.parent / "results",
            results_config=results_config,
            retrieval_metrics=retrieval_metrics,
            reasoning_metrics=reasoning_metrics,
            per_instance_results=per_instance_results,
        )
        return FinalResultsEvaluationOutcome(
            storage_result=storage_result,
            reasoning_metrics=reasoning_metrics,
            per_instance_results=per_instance_results,
        )

    def _build_per_instance_result(
        self,
        instance_index: int,
        answer_row: dict[str, Any],
        reasoning_row: dict[str, Any],
        prediction: EvaluatedAnswerRetrievalInstance,
        candidate_limit: int,
    ) -> PerInstanceFinalResult:
        gold_answers = [str(item) for item in answer_row.get("gold_answers", [])]
        predicted_answers = self._prediction_answers(answer_row)
        normalized_gold_answers = self._normalize_answer_set(gold_answers)
        normalized_predicted_answers = self._normalize_answer_list(predicted_answers)
        gold_set = set(normalized_gold_answers)
        predicted_set = set(normalized_predicted_answers)
        retrieved_candidate_set = set(
            self._normalize_answer_list(
                [candidate.node for candidate in prediction.answer_candidates]
            )
        )
        context_entity_set = self._normalized_context_entities(reasoning_row)
        retrieved_gold_set = gold_set & retrieved_candidate_set
        context_visible_gold_set = gold_set & context_entity_set
        retrieval_gold_coverage = self._safe_divide(
            len(retrieved_gold_set),
            len(gold_set),
        )
        reasoning_context_gold_coverage = self._safe_divide(
            len(context_visible_gold_set),
            len(gold_set),
        )
        full_gold_retrieval = bool(gold_set) and retrieved_gold_set == gold_set
        full_gold_context = bool(gold_set) and context_visible_gold_set == gold_set
        llm_retrieved_gold_utilization = self._optional_divide(
            len(predicted_set & retrieved_gold_set),
            len(retrieved_gold_set),
        )
        true_positive_count = len(gold_set & predicted_set)
        false_positive_count = len(predicted_set - gold_set)
        false_negative_count = len(gold_set - predicted_set)
        precision = self._safe_divide(true_positive_count, true_positive_count + false_positive_count)
        recall = self._safe_divide(true_positive_count, true_positive_count + false_negative_count)
        f1 = self._f1(precision, recall)
        grounding = self._compute_grounding(
            explanation=str(answer_row.get("explanation", "")),
            reasoning_row=reasoning_row,
        )
        answer_failed = answer_row.get("error_message") is not None
        return PerInstanceFinalResult(
            instance_index=instance_index,
            question=str(answer_row.get("question", "")),
            q_entity=[str(item) for item in answer_row.get("q_entity", [])],
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            normalized_gold_answers=normalized_gold_answers,
            normalized_predicted_answers=normalized_predicted_answers,
            exact_match=not answer_failed and gold_set == predicted_set,
            hit=true_positive_count > 0,
            hits_at_1=(
                bool(normalized_predicted_answers)
                and normalized_predicted_answers[0] in gold_set
            ),
            true_positive_count=true_positive_count,
            false_positive_count=false_positive_count,
            false_negative_count=false_negative_count,
            precision=precision,
            recall=recall,
            f1=f1,
            answer_error_message=answer_row.get("error_message"),
            mentioned_triple_count=grounding["mentioned_triple_count"],
            grounded_mentioned_triple_count=grounding["grounded_mentioned_triple_count"],
            grounded_explanation=grounding["grounded_explanation"],
            fully_grounded_explanation=grounding["fully_grounded_explanation"],
            ndcg_at_1=self.retriever_results_service.ndcg_at_k(prediction, 1),
            ndcg_at_5=self.retriever_results_service.ndcg_at_k(prediction, 5),
            ndcg_at_10=self.retriever_results_service.ndcg_at_k(prediction, 10),
            ndcg_at_candidate_limit=self.retriever_results_service.ndcg_at_k(
                prediction,
                candidate_limit,
            ),
            retrieved_gold_answers=sorted(retrieved_gold_set),
            context_visible_gold_answers=sorted(context_visible_gold_set),
            retrieval_gold_coverage=retrieval_gold_coverage,
            reasoning_context_gold_coverage=reasoning_context_gold_coverage,
            llm_retrieved_gold_utilization=llm_retrieved_gold_utilization,
            full_gold_retrieval=full_gold_retrieval,
            full_gold_context=full_gold_context,
            retrieval_generation_outcome=self._retrieval_generation_outcome(
                gold_set=gold_set,
                retrieved_gold_set=retrieved_gold_set,
                predicted_set=predicted_set,
            ),
        )

    def _build_reasoning_metrics(
        self,
        gnn_evaluation_result: GnnAnswerRetrieverEvaluationResult,
        llm_inference_run: SavedLlmInferenceRun,
        per_instance_results: list[PerInstanceFinalResult],
        retrieval_metrics: dict[str, Any],
        candidate_limit: int,
    ) -> FinalReasoningMetrics:
        evaluated_instances = len(per_instance_results)
        exact_match_count = sum(1 for item in per_instance_results if item.exact_match)
        hit_count = sum(1 for item in per_instance_results if item.hit)
        hits_at_1_count = sum(1 for item in per_instance_results if item.hits_at_1)
        true_positive_count = sum(
            item.true_positive_count for item in per_instance_results
        )
        false_positive_count = sum(
            item.false_positive_count for item in per_instance_results
        )
        false_negative_count = sum(
            item.false_negative_count for item in per_instance_results
        )
        precision = self._safe_divide(true_positive_count, true_positive_count + false_positive_count)
        recall = self._safe_divide(true_positive_count, true_positive_count + false_negative_count)
        grounded_count = sum(
            1 for item in per_instance_results if item.grounded_explanation
        )
        fully_grounded_count = sum(
            1 for item in per_instance_results if item.fully_grounded_explanation
        )
        mentioned_triple_count = sum(
            item.mentioned_triple_count for item in per_instance_results
        )
        grounded_mentioned_triple_count = sum(
            item.grounded_mentioned_triple_count for item in per_instance_results
        )
        return FinalReasoningMetrics(
            dataset_id=llm_inference_run.dataset_id,
            evaluation_run_name=gnn_evaluation_result.evaluation_run_name,
            inference_run_name=llm_inference_run.inference_run_name,
            model_run_name=gnn_evaluation_result.model_run_name,
            model_id=llm_inference_run.model_id,
            llm_provider=llm_inference_run.llm_provider,
            answer_metrics=FinalAnswerMetrics(
                evaluated_instances=evaluated_instances,
                successful_answers=sum(
                    1 for item in per_instance_results if item.answer_error_message is None
                ),
                failed_answers=sum(
                    1 for item in per_instance_results if item.answer_error_message is not None
                ),
                exact_match_count=exact_match_count,
                accuracy=self._safe_divide(exact_match_count, evaluated_instances),
                hit_count=hit_count,
                hit_rate=self._safe_divide(hit_count, evaluated_instances),
                hits_at_1_count=hits_at_1_count,
                hits_at_1=self._safe_divide(
                    hits_at_1_count,
                    evaluated_instances,
                ),
                true_positive_count=true_positive_count,
                false_positive_count=false_positive_count,
                false_negative_count=false_negative_count,
                precision=precision,
                recall=recall,
                f1=self._f1(precision, recall),
            ),
            explanation_grounding_metrics=ExplanationGroundingMetrics(
                grounded_explanation_count=grounded_count,
                fully_grounded_explanation_count=fully_grounded_count,
                grounded_explanation_rate=self._safe_divide(
                    grounded_count,
                    evaluated_instances,
                ),
                fully_grounded_explanation_rate=self._safe_divide(
                    fully_grounded_count,
                    evaluated_instances,
                ),
                mentioned_triple_count=mentioned_triple_count,
                grounded_mentioned_triple_count=grounded_mentioned_triple_count,
            ),
            ranking_metrics=RankingMetrics(
                ndcg_at_1=float(retrieval_metrics["ndcg_at_1"]),
                ndcg_at_5=float(retrieval_metrics["ndcg_at_5"]),
                ndcg_at_10=float(retrieval_metrics["ndcg_at_10"]),
                ndcg_at_candidate_limit=float(
                    retrieval_metrics["ndcg_at_candidate_limit"]
                ),
                candidate_limit=candidate_limit,
            ),
            retrieval_conditioned_answer_metrics=(
                self._build_retrieval_conditioned_answer_metrics(
                    per_instance_results
                )
            ),
        )

    def _build_retrieval_conditioned_answer_metrics(
        self,
        per_instance_results: list[PerInstanceFinalResult],
    ) -> RetrievalConditionedAnswerMetrics:
        """Aggregate retrieval availability and downstream answer utilization."""
        eligible_results = [
            item for item in per_instance_results if item.normalized_gold_answers
        ]
        eligible_count = len(eligible_results)
        full_retrieval_results = [
            item for item in eligible_results if item.full_gold_retrieval
        ]
        full_context_results = [
            item for item in eligible_results if item.full_gold_context
        ]
        retrieved_gold_answer_count = sum(
            len(item.retrieved_gold_answers) for item in eligible_results
        )
        answered_retrieved_gold_count = sum(
            len(
                set(item.normalized_predicted_answers)
                & set(item.retrieved_gold_answers)
            )
            for item in eligible_results
        )
        full_retrieval_omissions = sum(
            not set(item.normalized_gold_answers).issubset(
                item.normalized_predicted_answers
            )
            for item in full_retrieval_results
        )
        full_retrieval_exact_matches = sum(
            item.exact_match for item in full_retrieval_results
        )
        full_context_omissions = sum(
            not set(item.normalized_gold_answers).issubset(
                item.normalized_predicted_answers
            )
            for item in full_context_results
        )
        full_context_exact_matches = sum(
            item.exact_match for item in full_context_results
        )
        full_context_complete_answers = (
            len(full_context_results) - full_context_omissions
        )
        partial_context_results = [
            item
            for item in eligible_results
            if item.context_visible_gold_answers and not item.full_gold_context
        ]
        partial_context_fully_utilized = sum(
            set(item.context_visible_gold_answers).issubset(
                item.normalized_predicted_answers
            )
            for item in partial_context_results
        )
        partial_context_underutilized = (
            len(partial_context_results) - partial_context_fully_utilized
        )
        outcome_names = [
            "full_retrieval_complete_answer",
            "full_retrieval_llm_omission",
            "partial_retrieval_fully_utilized",
            "partial_retrieval_underutilized",
            "no_gold_retrieved_no_gold_answered",
            "correct_without_gold_retrieval",
        ]
        outcome_counts = {
            name: sum(
                item.retrieval_generation_outcome == name
                for item in eligible_results
            )
            for name in outcome_names
        }

        return RetrievalConditionedAnswerMetrics(
            conditioned_evaluated_instances=eligible_count,
            retrieval_gold_coverage=self._mean(
                [item.retrieval_gold_coverage for item in eligible_results]
            ),
            retrieval_full_gold_coverage_count=len(full_retrieval_results),
            retrieval_full_gold_coverage_rate=self._safe_divide(
                len(full_retrieval_results), eligible_count
            ),
            reasoning_context_gold_coverage=self._mean(
                [item.reasoning_context_gold_coverage for item in eligible_results]
            ),
            reasoning_context_full_gold_coverage_count=len(full_context_results),
            reasoning_context_full_gold_coverage_rate=self._safe_divide(
                len(full_context_results), eligible_count
            ),
            retrieved_gold_answer_count=retrieved_gold_answer_count,
            answered_retrieved_gold_count=answered_retrieved_gold_count,
            llm_retrieved_gold_utilization=self._optional_divide(
                answered_retrieved_gold_count,
                retrieved_gold_answer_count,
            ),
            llm_omission_given_full_retrieval_count=full_retrieval_omissions,
            llm_omission_given_full_retrieval_rate=self._optional_divide(
                full_retrieval_omissions,
                len(full_retrieval_results),
            ),
            llm_exact_match_given_full_retrieval_count=(
                full_retrieval_exact_matches
            ),
            llm_exact_match_given_full_retrieval=self._optional_divide(
                full_retrieval_exact_matches,
                len(full_retrieval_results),
            ),
            llm_omission_given_full_context_count=full_context_omissions,
            llm_omission_given_full_context_rate=self._optional_divide(
                full_context_omissions,
                len(full_context_results),
            ),
            llm_exact_match_given_full_context_count=full_context_exact_matches,
            llm_exact_match_given_full_context=self._optional_divide(
                full_context_exact_matches,
                len(full_context_results),
            ),
            full_retrieval_complete_answer_count=outcome_counts[
                "full_retrieval_complete_answer"
            ],
            full_retrieval_complete_answer_rate=self._safe_divide(
                outcome_counts["full_retrieval_complete_answer"], eligible_count
            ),
            full_retrieval_llm_omission_count=outcome_counts[
                "full_retrieval_llm_omission"
            ],
            full_retrieval_llm_omission_rate=self._safe_divide(
                outcome_counts["full_retrieval_llm_omission"], eligible_count
            ),
            partial_retrieval_fully_utilized_count=outcome_counts[
                "partial_retrieval_fully_utilized"
            ],
            partial_retrieval_fully_utilized_rate=self._safe_divide(
                outcome_counts["partial_retrieval_fully_utilized"], eligible_count
            ),
            partial_retrieval_underutilized_count=outcome_counts[
                "partial_retrieval_underutilized"
            ],
            partial_retrieval_underutilized_rate=self._safe_divide(
                outcome_counts["partial_retrieval_underutilized"], eligible_count
            ),
            full_context_complete_answer_count=full_context_complete_answers,
            full_context_complete_answer_rate=self._safe_divide(
                full_context_complete_answers,
                eligible_count,
            ),
            full_context_llm_omission_count=full_context_omissions,
            full_context_llm_omission_rate=self._safe_divide(
                full_context_omissions,
                eligible_count,
            ),
            partial_context_fully_utilized_count=(
                partial_context_fully_utilized
            ),
            partial_context_fully_utilized_rate=self._safe_divide(
                partial_context_fully_utilized,
                eligible_count,
            ),
            partial_context_underutilized_count=partial_context_underutilized,
            partial_context_underutilized_rate=self._safe_divide(
                partial_context_underutilized,
                eligible_count,
            ),
            no_gold_retrieved_no_gold_answered_count=outcome_counts[
                "no_gold_retrieved_no_gold_answered"
            ],
            no_gold_retrieved_no_gold_answered_rate=self._safe_divide(
                outcome_counts["no_gold_retrieved_no_gold_answered"], eligible_count
            ),
            correct_without_gold_retrieval_count=outcome_counts[
                "correct_without_gold_retrieval"
            ],
            correct_without_gold_retrieval_rate=self._safe_divide(
                outcome_counts["correct_without_gold_retrieval"], eligible_count
            ),
        )

    def _save_results_run(
        self,
        results_root: Path,
        results_config: FinalResultsConfig,
        retrieval_metrics: dict[str, Any],
        reasoning_metrics: FinalReasoningMetrics,
        per_instance_results: list[PerInstanceFinalResult],
    ) -> FinalResultsStorageResult:
        try:
            results_run_directory = self._create_results_run_directory(results_root)
            results_config_path = results_run_directory / self.results_config_filename
            retrieval_metrics_path = results_run_directory / self.retrieval_metrics_filename
            reasoning_metrics_path = results_run_directory / self.reasoning_metrics_filename
            per_instance_results_path = (
                results_run_directory / self.per_instance_results_filename
            )
            results_config = results_config.model_copy(
                update={
                    "run_name": results_run_directory.name,
                    "run_number": self._extract_run_number(results_run_directory.name),
                    "configs": {
                        **results_config.configs,
                        "results_config_path": results_config_path,
                    },
                    "artifacts": {
                        **results_config.artifacts,
                        "retrieval_metrics_path": str(retrieval_metrics_path),
                        "reasoning_metrics_path": str(reasoning_metrics_path),
                        "per_instance_results_path": str(per_instance_results_path),
                    },
                }
            )
            results_config_path.write_text(
                json.dumps(
                    make_project_paths_relative(results_config.model_dump(mode="json")),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            retrieval_metrics_path.write_text(
                json.dumps(retrieval_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            reasoning_metrics_path.write_text(
                json.dumps(reasoning_metrics.flattened_payload(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with per_instance_results_path.open("w", encoding="utf-8") as output_file:
                for item in per_instance_results:
                    output_file.write(item.model_dump_json())
                    output_file.write("\n")
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not save final results run: {error}"
            ) from error

        return FinalResultsStorageResult(
            results_run_directory=results_run_directory,
            results_run_name=results_run_directory.name,
            results_run_number=self._extract_run_number(results_run_directory.name),
            results_config_path=results_config_path,
            retrieval_metrics_path=retrieval_metrics_path,
            reasoning_metrics_path=reasoning_metrics_path,
            per_instance_results_path=per_instance_results_path,
        )

    def _compute_grounding(
        self,
        explanation: str,
        reasoning_row: dict[str, Any],
    ) -> dict[str, Any]:
        subgraph_triples = self._normalized_subgraph_triples(reasoning_row)
        mentioned_triples = self._extract_explanation_triples(explanation)
        if not mentioned_triples:
            mentioned_triples = {
                triple
                for triple in subgraph_triples
                if self._triple_to_text(triple) in self._normalize_text(explanation)
            }

        grounded_triples = mentioned_triples & subgraph_triples
        mentioned_count = len(mentioned_triples)
        grounded_count = len(grounded_triples)
        return {
            "mentioned_triple_count": mentioned_count,
            "grounded_mentioned_triple_count": grounded_count,
            "grounded_explanation": grounded_count > 0,
            "fully_grounded_explanation": mentioned_count > 0 and grounded_count == mentioned_count,
        }

    def _normalized_subgraph_triples(
        self,
        reasoning_row: dict[str, Any],
    ) -> set[tuple[str, str, str]]:
        subgraph = reasoning_row.get("subgraph")
        if not isinstance(subgraph, list):
            raise FinalResultsEvaluationException(
                f"Reasoning row {reasoning_row.get('instance_index')} is missing subgraph."
            )

        triples: set[tuple[str, str, str]] = set()
        for triple in subgraph:
            if not isinstance(triple, dict):
                continue
            triples.add(
                (
                    self._normalize_text(str(triple.get("source", ""))),
                    self._normalize_text(str(triple.get("relation", ""))),
                    self._normalize_text(str(triple.get("target", ""))),
                )
            )
        return triples

    def _normalized_context_entities(
        self,
        reasoning_row: dict[str, Any],
    ) -> set[str]:
        """Return normalized entity names actually exposed in reasoning paths."""
        subgraph = reasoning_row.get("subgraph")
        if not isinstance(subgraph, list):
            raise FinalResultsEvaluationException(
                f"Reasoning row {reasoning_row.get('instance_index')} is missing subgraph."
            )
        entities: set[str] = set()
        for triple in subgraph:
            if not isinstance(triple, dict):
                continue
            for key in ("source", "target"):
                entity = self._normalize_answer(str(triple.get(key, "")))
                if entity:
                    entities.add(entity)
        return entities

    @staticmethod
    def _retrieval_generation_outcome(
        *,
        gold_set: set[str],
        retrieved_gold_set: set[str],
        predicted_set: set[str],
    ) -> str:
        """Classify one instance into a mutually exclusive pipeline outcome."""
        if not gold_set:
            return "no_gold_answers"
        if retrieved_gold_set == gold_set:
            if gold_set.issubset(predicted_set):
                return "full_retrieval_complete_answer"
            return "full_retrieval_llm_omission"
        if retrieved_gold_set:
            if retrieved_gold_set.issubset(predicted_set):
                return "partial_retrieval_fully_utilized"
            return "partial_retrieval_underutilized"
        if predicted_set & gold_set:
            return "correct_without_gold_retrieval"
        return "no_gold_retrieved_no_gold_answered"

    def _extract_explanation_triples(self, explanation: str) -> set[tuple[str, str, str]]:
        triples: set[tuple[str, str, str]] = set()
        for match in self.arrow_triple_pattern.finditer(explanation):
            triples.add(
                (
                    self._normalize_text(match.group(1)),
                    self._normalize_text(match.group(2)),
                    self._normalize_text(match.group(3)),
                )
            )
        return triples

    @staticmethod
    def _triple_to_text(triple: tuple[str, str, str]) -> str:
        return " ".join(triple)

    def _prediction_answers(self, answer_row: dict[str, Any]) -> list[str]:
        structured_answers = answer_row.get("answers")
        if not isinstance(structured_answers, list) or any(
            not isinstance(item, str) for item in structured_answers
        ):
            raise FinalResultsEvaluationException(
                "Inference answer row must contain an 'answers' array of strings. "
                "Rerun inference for artifacts created with the old string format."
            )
        if answer_row.get("error_message") is not None:
            return []
        answers = [item.strip() for item in structured_answers if item.strip()]
        return [
            answer
            for answer in answers
            if self._normalize_text(answer) not in self.unknown_answer_values
        ]

    def _normalize_answer_set(self, answers: list[str]) -> list[str]:
        return sorted(set(self._normalize_answer_list(answers)))

    def _normalize_answer_list(self, answers: list[str]) -> list[str]:
        normalized_answers: list[str] = []
        seen_answers: set[str] = set()
        for answer in answers:
            normalized_answer = self._normalize_answer(answer)
            if not normalized_answer or normalized_answer in seen_answers:
                continue
            normalized_answers.append(normalized_answer)
            seen_answers.add(normalized_answer)
        return normalized_answers

    def _normalize_answer(self, answer: str) -> str:
        normalized = answer.lower().strip()
        normalized = normalized.translate(str.maketrans("", "", punctuation))
        normalized = self.article_pattern.sub(" ", normalized)
        return self.whitespace_pattern.sub(" ", normalized).strip()

    def _normalize_text(self, value: str) -> str:
        normalized = value.lower().strip()
        normalized = normalized.translate(str.maketrans("", "", punctuation.replace(".", "")))
        return self.whitespace_pattern.sub(" ", normalized).strip()

    @staticmethod
    def _ndcg_at_k(
        prediction: EvaluatedAnswerRetrievalInstance,
        k: int,
    ) -> float:
        return GnnRetrieverResultsService.ndcg_at_k(prediction, k)

    @staticmethod
    def _safe_divide(numerator: int | float, denominator: int | float) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _optional_divide(
        numerator: int | float,
        denominator: int | float,
    ) -> float | None:
        """Divide conditional metrics, preserving an unavailable denominator."""
        if denominator == 0:
            return None
        return numerator / denominator

    @classmethod
    def _f1(cls, precision: float, recall: float) -> float:
        return cls._safe_divide(2 * precision * recall, precision + recall)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
        try:
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FinalResultsEvaluationException(
                        f"JSONL row {line_number} in {path} must be an object."
                    )
                rows.append(value)
            return rows
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not read JSONL file {path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise FinalResultsEvaluationException(
                f"Invalid JSONL content in {path}: {error}"
            ) from error

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise FinalResultsEvaluationException(
                f"Could not read JSON file {path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise FinalResultsEvaluationException(
                f"Invalid JSON content in {path}: {error}"
            ) from error

        if not isinstance(value, dict):
            raise FinalResultsEvaluationException(f"JSON file {path} must contain an object.")
        return value

    @staticmethod
    def _index_rows(
        rows: list[dict[str, Any]],
        row_kind: str,
    ) -> dict[int, dict[str, Any]]:
        indexed_rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            instance_index = row.get("instance_index")
            if not isinstance(instance_index, int):
                raise FinalResultsEvaluationException(
                    f"{row_kind} row is missing integer instance_index."
                )
            if instance_index in indexed_rows:
                raise FinalResultsEvaluationException(
                    f"{row_kind} contains duplicate instance_index={instance_index}."
                )
            indexed_rows[instance_index] = row
        return indexed_rows

    @staticmethod
    def _validate_index_sets(
        answers_by_index: dict[int, dict[str, Any]],
        reasoning_by_index: dict[int, dict[str, Any]],
        predictions_by_index: dict[int, EvaluatedAnswerRetrievalInstance],
    ) -> None:
        answer_indexes = set(answers_by_index)
        reasoning_indexes = set(reasoning_by_index)
        prediction_indexes = set(predictions_by_index)
        if answer_indexes != reasoning_indexes or answer_indexes != prediction_indexes:
            raise FinalResultsEvaluationException(
                "Final results source files must contain the same instance indexes. "
                f"answers={sorted(answer_indexes)} reasoning={sorted(reasoning_indexes)} "
                f"predictions={sorted(prediction_indexes)}"
            )

    @staticmethod
    def _extract_candidate_limit(evaluation_config: dict[str, Any]) -> int:
        raw_candidate_limit = evaluation_config.get("evaluation", {}).get("candidate_limit")
        if isinstance(raw_candidate_limit, int) and raw_candidate_limit > 0:
            return raw_candidate_limit

        return 10

    @classmethod
    def _build_gnn_architecture(cls, model_config_path: Path) -> str:
        if not model_config_path.exists():
            for legacy_filename in ["model.config", "gnn_answer_retriever_config.json"]:
                legacy_config_path = model_config_path.with_name(legacy_filename)
                if legacy_config_path.exists():
                    model_config_path = legacy_config_path
                    break
        if model_config_path.exists():
            try:
                model_config = cls._load_json_object(model_config_path)
                return infer_gnn_architecture(model_config)
            except FinalResultsEvaluationException:
                pass
        return "graphsage"

    def _create_results_run_directory(self, results_root: Path) -> Path:
        results_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(results_root)
        run_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_run_directory = results_root / f"{run_number}_{run_label}"
        results_run_directory.mkdir(parents=True, exist_ok=False)
        return results_run_directory

    @classmethod
    def _next_run_number(cls, results_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in results_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))
