"""Tests for final retrieval plus reasoning result evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pipeline.evaluation.exceptions import FinalResultsEvaluationException
from pipeline.evaluation.models import (
    AnswerCandidateScore,
    EvaluatedAnswerRetrievalInstance,
    GnnAnswerRetrieverEvaluationResult,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.services.final_results_evaluation import (
    FinalResultsEvaluationService,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(node: str, probability: float, is_gold: bool) -> AnswerCandidateScore:
    return AnswerCandidateScore(
        node=node,
        local_node_id=1,
        global_node_id=1,
        logit=0.0,
        probability=probability,
        is_gold_answer=is_gold,
        selection_reason="threshold",
    )


def _prediction(
    candidates: list[AnswerCandidateScore],
    gold_answers: list[str] | None = None,
) -> EvaluatedAnswerRetrievalInstance:
    resolved_gold_answers = ["gold"] if gold_answers is None else gold_answers
    return EvaluatedAnswerRetrievalInstance(
        instance_index=0,
        question="question?",
        q_entity=["topic"],
        a_entity=resolved_gold_answers,
        answer_candidates=candidates,
        gold_answer_scores=[],
        hit_at_1=bool(candidates and candidates[0].is_gold_answer),
        hit_at_5=any(candidate.is_gold_answer for candidate in candidates[:5]),
        hit_at_10=any(candidate.is_gold_answer for candidate in candidates[:10]),
        hit_at_candidate_limit=any(candidate.is_gold_answer for candidate in candidates),
        missing_gold_in_graph=not any(candidate.is_gold_answer for candidate in candidates),
    )


def test_answer_normalization_and_set_metrics() -> None:
    service = FinalResultsEvaluationService()
    answer_row = {
        "question": "q",
        "q_entity": ["e"],
        "gold_answers": ["The Moon", "Earth"],
        "answers": ["moon", "the earth"],
        "explanation": "",
        "error_message": None,
    }
    result = service._build_per_instance_result(
        instance_index=0,
        answer_row=answer_row,
        reasoning_row={"instance_index": 0, "subgraph": []},
        prediction=_prediction([]),
        candidate_limit=10,
    )

    assert result.normalized_gold_answers == ["earth", "moon"]
    assert result.normalized_predicted_answers == ["moon", "earth"]
    assert result.exact_match is True
    assert result.hit is True
    assert result.hits_at_1 is True
    assert result.precision == 1.0
    assert result.recall == 1.0

    wrong_first_row = answer_row | {"answers": ["Berlin", "Earth"]}
    wrong_first_result = service._build_per_instance_result(
        instance_index=1,
        answer_row=wrong_first_row,
        reasoning_row={"instance_index": 1, "subgraph": []},
        prediction=_prediction([]),
        candidate_limit=10,
    )
    assert wrong_first_result.hit is True
    assert wrong_first_result.hits_at_1 is False
    assert wrong_first_result.precision == 0.5
    assert wrong_first_result.recall == 0.5
    assert wrong_first_result.f1 == 0.5

    unknown_row = answer_row | {"answers": ["Unknown"]}
    unknown_result = service._build_per_instance_result(
        instance_index=2,
        answer_row=unknown_row,
        reasoning_row={"instance_index": 2, "subgraph": []},
        prediction=_prediction([]),
        candidate_limit=10,
    )
    assert unknown_result.predicted_answers == []
    assert unknown_result.false_negative_count == 2
    assert unknown_result.exact_match is False
    assert unknown_result.hit is False
    assert unknown_result.hits_at_1 is False

    failed_row = answer_row | {"answers": ["Moon"], "error_message": "timeout"}
    failed_result = service._build_per_instance_result(
        instance_index=3,
        answer_row=failed_row,
        reasoning_row={"instance_index": 3, "subgraph": []},
        prediction=_prediction([]),
        candidate_limit=10,
    )
    assert failed_result.predicted_answers == []
    assert failed_result.exact_match is False
    assert failed_result.hit is False
    assert failed_result.false_negative_count == 2


def test_comma_bearing_entities_remain_atomic_across_all_metrics() -> None:
    service = FinalResultsEvaluationService()
    candidates = [
        _candidate("Washington, D.C.", 0.9, True),
        _candidate("Paris, Texas", 0.8, True),
    ]
    answer_row = {
        "question": "q",
        "q_entity": ["topic"],
        "gold_answers": ["Washington, D.C.", "Paris, Texas"],
        "answers": ["Paris, Texas", "Washington, D.C."],
        "explanation": "",
        "error_message": None,
    }
    reasoning_row = {
        "instance_index": 0,
        "subgraph": [
            {"source": "topic", "relation": "located_in", "target": "Washington, D.C."},
            {"source": "topic", "relation": "located_in", "target": "Paris, Texas"},
        ],
    }

    result = service._build_per_instance_result(
        instance_index=0,
        answer_row=answer_row,
        reasoning_row=reasoning_row,
        prediction=_prediction(candidates),
        candidate_limit=10,
    )

    assert result.normalized_gold_answers == ["paris texas", "washington dc"]
    assert result.normalized_predicted_answers == ["paris texas", "washington dc"]
    assert result.true_positive_count == 2
    assert result.false_positive_count == 0
    assert result.false_negative_count == 0
    assert result.exact_match is True
    assert result.retrieval_gold_coverage == 1.0
    assert result.reasoning_context_gold_coverage == 1.0
    assert result.full_gold_retrieval is True
    assert result.full_gold_context is True


def test_separate_array_items_cannot_impersonate_one_comma_bearing_entity() -> None:
    service = FinalResultsEvaluationService()
    result = service._build_per_instance_result(
        instance_index=0,
        answer_row={
            "question": "q",
            "q_entity": ["topic"],
            "gold_answers": ["Washington, D.C."],
            "answers": ["Washington", "D.C."],
            "explanation": "",
            "error_message": None,
        },
        reasoning_row={"instance_index": 0, "subgraph": []},
        prediction=_prediction([]),
        candidate_limit=10,
    )

    assert result.normalized_gold_answers == ["washington dc"]
    assert result.normalized_predicted_answers == ["washington", "dc"]
    assert result.exact_match is False
    assert result.true_positive_count == 0


def test_singular_answer_artifact_is_rejected() -> None:
    service = FinalResultsEvaluationService()

    with pytest.raises(FinalResultsEvaluationException, match="Rerun inference"):
        service._prediction_answers(
            {
                "answer": "Washington, D.C., Paris, Texas",
                "answer_candidates": ["Washington, D.C.", "Paris, Texas"],
                "error_message": None,
            }
        )


def test_retrieval_conditioned_answer_metrics_cover_all_outcomes() -> None:
    service = FinalResultsEvaluationService()

    def build_result(
        instance_index: int,
        retrieved: list[str],
        predicted: list[str],
        visible: list[str],
    ):
        prediction = EvaluatedAnswerRetrievalInstance(
            instance_index=instance_index,
            question="question?",
            q_entity=["topic"],
            a_entity=["Alpha", "Beta"],
            answer_candidates=[
                _candidate(node, 1.0 - rank * 0.1, node in {"Alpha", "Beta"})
                for rank, node in enumerate(retrieved)
            ],
            gold_answer_scores=[],
            hit_at_1=bool(retrieved and retrieved[0] in {"Alpha", "Beta"}),
            hit_at_5=any(node in {"Alpha", "Beta"} for node in retrieved[:5]),
            hit_at_10=any(node in {"Alpha", "Beta"} for node in retrieved[:10]),
            hit_at_candidate_limit=any(
                node in {"Alpha", "Beta"} for node in retrieved
            ),
            missing_gold_in_graph=False,
        )
        return service._build_per_instance_result(
            instance_index=instance_index,
            answer_row={
                "question": "question?",
                "q_entity": ["topic"],
                "gold_answers": ["Alpha", "Beta"],
                "answers": predicted,
                "explanation": "",
                "error_message": None,
            },
            reasoning_row={
                "instance_index": instance_index,
                "subgraph": [
                    {"source": "topic", "relation": "related", "target": node}
                    for node in visible
                ],
            },
            prediction=prediction,
            candidate_limit=10,
        )

    results = [
        build_result(0, ["Alpha", "Beta"], ["Alpha", "Beta"], ["Alpha", "Beta"]),
        build_result(1, ["Alpha", "Beta"], ["Alpha"], ["Alpha", "Beta"]),
        build_result(2, ["Alpha"], ["Alpha"], ["Alpha"]),
        build_result(3, ["Alpha"], [], ["Alpha"]),
        build_result(4, ["Wrong"], [], []),
        build_result(5, ["Wrong"], ["Alpha"], []),
    ]

    assert [item.retrieval_generation_outcome for item in results] == [
        "full_retrieval_complete_answer",
        "full_retrieval_llm_omission",
        "partial_retrieval_fully_utilized",
        "partial_retrieval_underutilized",
        "no_gold_retrieved_no_gold_answered",
        "correct_without_gold_retrieval",
    ]
    metrics = service._build_retrieval_conditioned_answer_metrics(results)
    assert metrics.conditioned_evaluated_instances == 6
    assert metrics.retrieval_gold_coverage == 0.5
    assert metrics.retrieval_full_gold_coverage_count == 2
    assert math.isclose(metrics.retrieval_full_gold_coverage_rate, 1 / 3)
    assert metrics.reasoning_context_gold_coverage == 0.5
    assert metrics.retrieved_gold_answer_count == 6
    assert metrics.answered_retrieved_gold_count == 4
    assert metrics.llm_retrieved_gold_utilization is not None
    assert math.isclose(metrics.llm_retrieved_gold_utilization, 2 / 3)
    assert metrics.llm_omission_given_full_retrieval_count == 1
    assert metrics.llm_omission_given_full_retrieval_rate == 0.5
    assert metrics.llm_exact_match_given_full_retrieval == 0.5
    assert metrics.llm_omission_given_full_context_rate == 0.5
    assert metrics.llm_exact_match_given_full_context == 0.5
    assert metrics.full_retrieval_complete_answer_count == 1
    assert metrics.full_retrieval_llm_omission_count == 1
    assert metrics.partial_retrieval_fully_utilized_count == 1
    assert metrics.partial_retrieval_underutilized_count == 1
    assert metrics.full_context_complete_answer_count == 1
    assert metrics.full_context_llm_omission_count == 1
    assert metrics.partial_context_fully_utilized_count == 1
    assert metrics.partial_context_underutilized_count == 1
    assert math.isclose(metrics.full_context_complete_answer_rate, 1 / 6)
    assert math.isclose(metrics.full_context_llm_omission_rate, 1 / 6)
    assert math.isclose(metrics.partial_context_fully_utilized_rate, 1 / 6)
    assert math.isclose(metrics.partial_context_underutilized_rate, 1 / 6)
    assert metrics.no_gold_retrieved_no_gold_answered_count == 1
    assert metrics.correct_without_gold_retrieval_count == 1
    assert math.isclose(metrics.correct_without_gold_retrieval_rate, 1 / 6)


def test_conditional_metrics_are_none_when_no_answers_are_available() -> None:
    service = FinalResultsEvaluationService()
    result = service._build_per_instance_result(
        instance_index=0,
        answer_row={
            "question": "question?",
            "q_entity": ["topic"],
            "gold_answers": ["Gold"],
            "answers": ["Unknown"],
            "explanation": "",
            "error_message": None,
        },
        reasoning_row={"instance_index": 0, "subgraph": []},
        prediction=_prediction([_candidate("Wrong", 0.9, False)]),
        candidate_limit=10,
    )

    metrics = service._build_retrieval_conditioned_answer_metrics([result])
    assert metrics.llm_retrieved_gold_utilization is None
    assert metrics.llm_omission_given_full_retrieval_rate is None
    assert metrics.llm_exact_match_given_full_retrieval is None
    assert metrics.llm_omission_given_full_context_rate is None
    assert metrics.llm_exact_match_given_full_context is None


def test_explanation_grounding_exact_missing_and_empty() -> None:
    service = FinalResultsEvaluationService()
    reasoning_row = {
        "instance_index": 0,
        "subgraph": [
            {"source": "A", "relation": "related.to", "target": "B"},
        ],
    }

    exact = service._compute_grounding("A -> related.to -> B", reasoning_row)
    assert exact["mentioned_triple_count"] == 1
    assert exact["grounded_mentioned_triple_count"] == 1
    assert exact["grounded_explanation"] is True
    assert exact["fully_grounded_explanation"] is True

    missing = service._compute_grounding("A -> related.to -> C", reasoning_row)
    assert missing["mentioned_triple_count"] == 1
    assert missing["grounded_mentioned_triple_count"] == 0
    assert missing["grounded_explanation"] is False
    assert missing["fully_grounded_explanation"] is False

    empty = service._compute_grounding("No path mentioned.", reasoning_row)
    assert empty["mentioned_triple_count"] == 0
    assert empty["grounded_explanation"] is False


def test_ndcg_variants() -> None:
    service = FinalResultsEvaluationService()

    perfect = _prediction([
        _candidate("gold", 0.9, True),
        _candidate("wrong", 0.1, False),
    ])
    assert service._ndcg_at_k(perfect, 1) == 1.0
    assert service._ndcg_at_k(perfect, 5) == 1.0

    gold_second = _prediction([
        _candidate("wrong", 0.9, False),
        _candidate("gold", 0.1, True),
    ])
    assert service._ndcg_at_k(gold_second, 1) == 0.0
    assert math.isclose(service._ndcg_at_k(gold_second, 5), 1 / math.log2(3))

    no_gold = _prediction([_candidate("wrong", 0.9, False)])
    empty = _prediction([])
    assert service._ndcg_at_k(no_gold, 10) == 0.0
    assert service._ndcg_at_k(empty, 10) == 0.0

    incomplete_retrieval = _prediction(
        [_candidate("gold-1", 0.9, True)],
        gold_answers=["gold-1", "gold-2", "gold-3"],
    )
    ideal_three = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(
        service._ndcg_at_k(incomplete_retrieval, 5),
        1.0 / ideal_three,
    )

    persisted_order = _prediction([
        _candidate("wrong", 0.1, False),
        _candidate("gold", 0.9, True),
    ])
    assert math.isclose(
        service._ndcg_at_k(persisted_order, 5),
        1 / math.log2(3),
    )


def test_mismatched_source_rows_raise_domain_exception() -> None:
    with pytest.raises(FinalResultsEvaluationException):
        FinalResultsEvaluationService._validate_index_sets(
            answers_by_index={0: {"instance_index": 0}},
            reasoning_by_index={0: {"instance_index": 0}},
            predictions_by_index={1: _prediction([])},
        )


def test_final_results_storage_integration(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "data" / "webqsp" / "evaluations" / "1_eval"
    inference_dir = tmp_path / "data" / "webqsp" / "inference" / "1_inference"
    predictions_path = evaluation_dir / "predictions.jsonl"
    evaluation_config_path = evaluation_dir / "evaluation_config.json"
    answers_path = inference_dir / "answers.jsonl"
    reasoning_path = inference_dir / "reasoning.jsonl"
    inference_config_path = inference_dir / "inference_config.json"

    prediction_rows = [
        {
            "instance_index": 0,
            "question": "What does the Moon orbit?",
            "q_entity": ["Moon"],
            "a_entity": ["Earth"],
            "answer_candidates": [
                _candidate("Earth", 0.95, True).model_dump(mode="json"),
                _candidate("Venus", 0.1, False).model_dump(mode="json"),
            ],
            "gold_answer_scores": [],
            "hit_at_1": True,
            "hit_at_5": True,
            "hit_at_10": True,
            "missing_gold_in_graph": False,
        },
        {
            "instance_index": 1,
            "question": "What planet is Olympus Mons on?",
            "q_entity": ["Olympus Mons"],
            "a_entity": ["Mars"],
            "answer_candidates": [
                _candidate("Phobos", 0.8, False).model_dump(mode="json"),
                _candidate("Mars", 0.1, True).model_dump(mode="json"),
            ],
            "gold_answer_scores": [],
            "hit_at_1": False,
            "hit_at_5": True,
            "hit_at_10": True,
            "missing_gold_in_graph": False,
        },
    ]
    _write_jsonl(predictions_path, prediction_rows)
    _write_json(evaluation_config_path, {"evaluation": {"candidate_limit": 5}})
    _write_json(inference_config_path, {"model_id": "test-model"})
    model_run_directory = tmp_path / "models" / "1_model"
    _write_json(
        model_run_directory / "model_config.json",
        {
            "training": {
                "gnn_layer_count": 2,
                "hidden_dimension": 128,
            },
        },
    )
    _write_jsonl(
        answers_path,
        [
            {
                "instance_index": 0,
                "question": "What does the Moon orbit?",
                "q_entity": ["Moon"],
                "gold_answers": ["Earth"],
                "answer_candidates": ["Earth", "Venus"],
                "model_id": "test-model",
                "answers": ["the earth"],
                "explanation": "Moon -> orbits -> Earth",
                "raw_response": "",
                "error_message": None,
            },
            {
                "instance_index": 1,
                "question": "What planet is Olympus Mons on?",
                "q_entity": ["Olympus Mons"],
                "gold_answers": ["Mars"],
                "answer_candidates": ["Phobos", "Mars"],
                "model_id": "test-model",
                "answers": ["Unknown"],
                "explanation": "",
                "raw_response": "",
                "error_message": "LLM request failed",
            },
        ],
    )
    _write_jsonl(
        reasoning_path,
        [
            {
                "instance_index": 0,
                "question": "What does the Moon orbit?",
                "q_entity": ["Moon"],
                "subgraph": [
                    {"source": "Moon", "relation": "orbits", "target": "Earth"},
                ],
                "analytics": {},
            },
            {
                "instance_index": 1,
                "question": "What planet is Olympus Mons on?",
                "q_entity": ["Olympus Mons"],
                "subgraph": [
                    {"source": "Olympus Mons", "relation": "located.on", "target": "Mars"},
                ],
                "analytics": {},
            },
        ],
    )
    gnn_result = GnnAnswerRetrieverEvaluationResult(
        dataset_id="webqsp",
        model_run_directory=model_run_directory,
        model_run_name="1_model",
        model_run_number=1,
        evaluation_run_directory=evaluation_dir,
        evaluation_run_name="1_eval",
        evaluation_run_number=1,
        evaluated_instances=2,
        hits_at_1=0.5,
        hits_at_1_count=1,
        hits_at_5=1.0,
        hits_at_5_count=2,
        hits_at_10=1.0,
        hits_at_10_count=2,
        average_candidate_count=2,
        missing_gold_in_graph_count=0,
        predictions_path=predictions_path,
        evaluation_config_path=evaluation_config_path,
    )
    inference_run = SavedLlmInferenceRun(
        dataset_id="webqsp",
        evaluation_run_name="1_eval",
        inference_run_directory=inference_dir,
        inference_run_name="1_inference",
        inference_run_number=1,
        model_id="test-model",
        total_instances=2,
        successful_answers=1,
        failed_answers=1,
        reasoning_path=reasoning_path,
        answers_path=answers_path,
        inference_config_path=inference_config_path,
    )

    outcome = FinalResultsEvaluationService().evaluate(
        gnn_evaluation_result=gnn_result,
        llm_inference_run=inference_run,
    )

    results_dir = tmp_path / "data" / "webqsp" / "results" / outcome.storage_result.results_run_name
    assert outcome.storage_result.results_run_directory == results_dir
    assert outcome.storage_result.retrieval_metrics_path.exists()
    assert outcome.storage_result.reasoning_metrics_path.exists()
    assert outcome.storage_result.per_instance_results_path.exists()
    assert not (evaluation_dir / "summary_metrics.json").exists()
    results_config = json.loads(outcome.storage_result.results_config_path.read_text())
    assert results_config["dataset_id"] == "webqsp"
    assert results_config["model_id"] == "test-model"
    assert results_config["gnn_architecture"] == "graphsage"
    assert "gnn_id" not in results_config
    assert results_config["run_number"] == outcome.storage_result.results_run_number
    assert "model_run_directory" not in results_config
    assert "evaluation_run_directory" not in results_config
    assert "inference_run_directory" not in results_config
    assert "model_config_path" in results_config["configs"]
    assert results_config["artifacts"]["training"]["name"] == "1_model"
    assert "model_config_path" in results_config["artifacts"]["training"]
    assert results_config["artifacts"]["evaluation"]["name"] == "1_eval"
    assert "predictions_path" in results_config["artifacts"]["evaluation"]
    assert results_config["artifacts"]["inference"]["name"] == "1_inference"
    assert "answers_path" in results_config["artifacts"]["inference"]
    retrieval_metrics = json.loads(
        outcome.storage_result.retrieval_metrics_path.read_text()
    )
    assert retrieval_metrics["hits_at_1"] == 0.5
    assert retrieval_metrics["hits_at_5"] == 1.0
    assert retrieval_metrics["hits_at_10"] == 1.0
    assert retrieval_metrics["hits_at_candidate_limit"] == 1.0
    assert retrieval_metrics["candidate_limit"] == 5
    assert retrieval_metrics["average_candidate_count"] == 2.0
    assert retrieval_metrics["ndcg_at_1"] == 0.5
    assert math.isclose(
        retrieval_metrics["ndcg_at_5"],
        (1.0 + 1 / math.log2(3)) / 2,
    )

    metrics = json.loads(outcome.storage_result.reasoning_metrics_path.read_text())
    assert metrics["evaluated_instances"] == 2
    assert metrics["successful_answers"] == 1
    assert metrics["failed_answers"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["hit_count"] == 1
    assert metrics["hit_rate"] == 0.5
    assert metrics["hits_at_1_count"] == 1
    assert metrics["hits_at_1"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert math.isclose(metrics["f1"], 2 / 3)
    assert metrics["grounded_explanation_rate"] == 0.5
    assert metrics["candidate_limit"] == 5
    assert math.isclose(metrics["ndcg_at_5"], (1.0 + 1 / math.log2(3)) / 2)
    assert metrics["retrieval_gold_coverage"] == 1.0
    assert metrics["retrieval_full_gold_coverage_rate"] == 1.0
    assert metrics["reasoning_context_gold_coverage"] == 1.0
    assert metrics["reasoning_context_full_gold_coverage_rate"] == 1.0
    assert metrics["llm_retrieved_gold_utilization"] == 0.5
    assert metrics["llm_omission_given_full_retrieval_rate"] == 0.5
    assert metrics["llm_exact_match_given_full_retrieval"] == 0.5
    assert metrics["llm_omission_given_full_context_rate"] == 0.5
    assert metrics["llm_exact_match_given_full_context"] == 0.5
    assert metrics["full_retrieval_complete_answer_rate"] == 0.5
    assert metrics["full_context_complete_answer_rate"] == 0.5
    assert metrics["full_retrieval_llm_omission_rate"] == 0.5

    rows = outcome.storage_result.per_instance_results_path.read_text().splitlines()
    assert len(rows) == 2
    first_row = json.loads(rows[0])
    second_row = json.loads(rows[1])
    assert first_row["hit"] is True
    assert first_row["hits_at_1"] is True
    assert first_row["retrieval_generation_outcome"] == (
        "full_retrieval_complete_answer"
    )
    assert first_row["retrieval_gold_coverage"] == 1.0
    assert first_row["reasoning_context_gold_coverage"] == 1.0
    assert second_row["hit"] is False
    assert second_row["hits_at_1"] is False
    assert second_row["retrieval_generation_outcome"] == (
        "full_retrieval_llm_omission"
    )
