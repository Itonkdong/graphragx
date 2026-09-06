"""Tests for durable retriever metrics and run loading."""

from __future__ import annotations

import json

import pytest

from pipeline.evaluation.models import (
    AnswerCandidateScore,
    EvaluatedAnswerRetrievalInstance,
)
from pipeline.evaluation.services.gnn_retriever_results import (
    GnnRetrieverResultsService,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException


def _prediction(*, hit: bool) -> EvaluatedAnswerRetrievalInstance:
    return EvaluatedAnswerRetrievalInstance(
        instance_index=0,
        question="question",
        q_entity=["q"],
        a_entity=["answer"],
        answer_candidates=[
            AnswerCandidateScore(
                node="answer" if hit else "wrong",
                local_node_id=0,
                global_node_id=0,
                logit=1.0,
                probability=0.8,
                is_gold_answer=hit,
                selection_reason="threshold",
            )
        ],
        gold_answer_scores=[],
        hit_at_1=hit,
        hit_at_5=hit,
        hit_at_10=hit,
        hit_at_candidate_limit=hit,
        missing_gold_in_graph=False,
    )


def _write_run(tmp_path, *, with_metrics: bool = False):
    model_dir = tmp_path / "models" / "3_model"
    model_dir.mkdir(parents=True)
    model_config = {
        "dataset_id": "WebQSP",
        "entity_embedding_model": "text-embedding-3-small",
        "question_embedding_model": "text-embedding-3-small",
        "relation_embedding_model": "text-embedding-3-small",
        "entity_embedding_dimension": 1536,
        "question_embedding_dimension": 1536,
        "relation_embedding_dimension": 1536,
        "hidden_dimension": 256,
        "gnn_layer_count": 2,
        "node_classifier": "mlp",
        "use_reverse_edges": True,
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "log_every": 3,
            "device": "cpu",
        },
        "final_loss": 0.5,
        "trained_instances": 1,
    }
    model_config_path = model_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")

    run_dir = tmp_path / "evaluations" / "7_eval"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "model_config": {
                    "model_run_name": "3_model",
                    "model_run_number": 3,
                    "full_config_path": str(model_config_path),
                },
                "evaluation": {"candidate_limit": 15},
            }
        ),
        encoding="utf-8",
    )
    prediction = _prediction(hit=True)
    (run_dir / "predictions.jsonl").write_text(
        prediction.model_dump_json() + "\n",
        encoding="utf-8",
    )
    if with_metrics:
        metrics = GnnRetrieverResultsService().build_metrics(
            dataset_id="WebQSP",
            model_run_name="3_model",
            model_run_number=3,
            predictions=[prediction],
            candidate_limit=15,
            evaluation_run_name="7_eval",
            evaluation_run_number=7,
        )
        (run_dir / "retrieval_metrics.json").write_text(
            metrics.model_dump_json(),
            encoding="utf-8",
        )
    return run_dir


def test_load_legacy_retriever_run_computes_metrics_without_mutating(tmp_path) -> None:
    run_dir = _write_run(tmp_path)

    result = GnnRetrieverResultsService().load_run(
        evaluation_root=tmp_path / "evaluations",
        dataset_id="WebQSP",
        run_name=None,
        run_number=7,
    )

    assert result.evaluation_run_name == "7_eval"
    assert result.hits_at_1 == 1.0
    assert result.ndcg_at_10 == 1.0
    assert result.retrieval_metrics_path is None
    assert not (run_dir / "retrieval_metrics.json").exists()


def test_metrics_retain_missing_gold_count_for_skipped_predictions() -> None:
    metrics = GnnRetrieverResultsService().build_metrics(
        dataset_id="WebQSP",
        model_run_name="3_model",
        model_run_number=3,
        predictions=[_prediction(hit=True)],
        candidate_limit=10,
        missing_gold_in_graph_count=4,
        skipped_missing_gold_in_graph_count=3,
    )

    assert metrics.evaluated_instances == 1
    assert metrics.missing_gold_in_graph_count == 4
    assert metrics.skipped_missing_gold_in_graph_count == 3


def test_retriever_metrics_include_ndcg_from_persisted_candidate_order() -> None:
    prediction = _prediction(hit=False).model_copy(
        update={
            "a_entity": ["answer"],
            "answer_candidates": [
                AnswerCandidateScore(
                    node="wrong",
                    local_node_id=0,
                    global_node_id=0,
                    logit=2.0,
                    probability=0.9,
                    is_gold_answer=False,
                    selection_reason="threshold",
                ),
                AnswerCandidateScore(
                    node="answer",
                    local_node_id=1,
                    global_node_id=1,
                    logit=1.0,
                    probability=0.8,
                    is_gold_answer=True,
                    selection_reason="threshold",
                ),
            ],
        }
    )

    metrics = GnnRetrieverResultsService().build_metrics(
        dataset_id="WebQSP",
        model_run_name="3_model",
        model_run_number=3,
        predictions=[prediction],
        candidate_limit=5,
    )

    expected = 1 / 1.584962500721156
    assert metrics.ndcg_at_1 == 0.0
    assert metrics.ndcg_at_5 == pytest.approx(expected)
    assert metrics.ndcg_at_10 == pytest.approx(expected)
    assert metrics.ndcg_at_candidate_limit == pytest.approx(expected)


def test_retriever_metrics_include_multi_answer_gold_coverage() -> None:
    partial = _prediction(hit=True).model_copy(
        update={
            "a_entity": ["The Answer", "Second Answer"],
            "answer_candidates": [
                AnswerCandidateScore(
                    node="Answer",
                    local_node_id=0,
                    global_node_id=0,
                    logit=2.0,
                    probability=0.9,
                    is_gold_answer=True,
                    selection_reason="threshold",
                )
            ],
        }
    )
    complete = _prediction(hit=True).model_copy(
        update={
            "a_entity": ["Gold"],
            "answer_candidates": [
                AnswerCandidateScore(
                    node="Gold",
                    local_node_id=0,
                    global_node_id=0,
                    logit=2.0,
                    probability=0.9,
                    is_gold_answer=True,
                    selection_reason="threshold",
                )
            ],
        }
    )

    metrics = GnnRetrieverResultsService().build_metrics(
        dataset_id="WebQSP",
        model_run_name="3_model",
        model_run_number=3,
        predictions=[partial, complete],
        candidate_limit=10,
    )

    assert metrics.conditioned_evaluated_instances == 2
    assert metrics.retrieval_gold_coverage == 0.75
    assert metrics.retrieval_full_gold_coverage_count == 1
    assert metrics.retrieval_full_gold_coverage_rate == 0.5
    assert metrics.retrieved_gold_answer_count == 2


def test_load_retriever_run_uses_persisted_metrics(tmp_path) -> None:
    run_dir = _write_run(tmp_path, with_metrics=True)

    result = GnnRetrieverResultsService().load_run(
        evaluation_root=tmp_path / "evaluations",
        dataset_id="WebQSP",
        run_name="eval",
        run_number=None,
    )

    assert result.retrieval_metrics_path == run_dir / "retrieval_metrics.json"
    assert result.model_run_name == "3_model"
    assert result.ndcg_at_10 == 1.0
    assert result.retrieval_gold_coverage == 1.0
    assert result.retrieval_full_gold_coverage_rate == 1.0


def test_load_run_recovers_skipped_count_from_pre_field_metrics(tmp_path) -> None:
    run_dir = _write_run(tmp_path, with_metrics=True)
    metrics_path = run_dir / "retrieval_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.pop("skipped_missing_gold_in_graph_count")
    payload["missing_gold_in_graph_count"] = 3
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    result = GnnRetrieverResultsService().load_run(
        evaluation_root=tmp_path / "evaluations",
        dataset_id="WebQSP",
        run_name="eval",
        run_number=None,
    )

    assert result.missing_gold_in_graph_count == 3
    assert result.skipped_missing_gold_in_graph_count == 3


def test_load_retriever_run_rejects_dataset_mismatch(tmp_path) -> None:
    _write_run(tmp_path)

    with pytest.raises(GnnAnswerRetrieverEvaluationException, match="does not match"):
        GnnRetrieverResultsService().load_run(
            evaluation_root=tmp_path / "evaluations",
            dataset_id="Other",
            run_name="eval",
            run_number=None,
        )


def test_load_model_config_exposes_authoritative_graph_settings(tmp_path) -> None:
    _write_run(tmp_path)

    config = GnnRetrieverResultsService().load_model_config(
        evaluation_root=tmp_path / "evaluations",
        run_name="7_eval",
        run_number=None,
    )

    assert config.use_reverse_edges is True
    assert config.resolved_gnn_layer_count == 2
