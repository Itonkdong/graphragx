"""Tests for GNN evaluation storage payload shapes."""

from pipeline.preparation.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStoragePayload,
)
from pipeline.evaluation.models import GnnAnswerRetrieverMetrics


def test_evaluation_storage_payload_accepts_nested_model_config_reference() -> None:
    payload = GnnAnswerRetrieverEvaluationStoragePayload(
        evaluation_config={
            "dataset_id": "WebQSP",
            "model_config": {
                "model_run_name": "1_model",
                "model_run_number": 1,
                "full_config_path": "/tmp/models/1_model/model_config.json",
                "weights_path": "/tmp/models/1_model/gnn_answer_retriever.pt",
            },
            "evaluation": {
                "candidate_limit": 15,
            },
        },
        metrics=GnnAnswerRetrieverMetrics(
            dataset_id="WebQSP",
            model_run_name="1_model",
            model_run_number=1,
            evaluated_instances=1,
            hits_at_1=1.0,
            hits_at_1_count=1,
            hits_at_5=1.0,
            hits_at_5_count=1,
            hits_at_10=1.0,
            hits_at_10_count=1,
            hits_at_candidate_limit=1.0,
            hits_at_candidate_limit_count=1,
            candidate_limit=15,
            average_candidate_count=1.0,
            missing_gold_in_graph_count=0,
        ),
    )

    assert payload.evaluation_config["model_config"]["model_run_number"] == 1
