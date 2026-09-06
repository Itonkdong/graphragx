"""Tests for optional WandB final result logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.abstract import StepContext
from pipeline.evaluation.models import FinalResultsEvaluationResult
from pipeline.evaluation.services import (
    WandbFinalResultsConfig,
    WandbFinalResultsLoggingService,
    WandbFinalResultsLogResult,
)
from pipeline.evaluation.steps.wandb_final_results import LogFinalResultsToWandbStep


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _fake_final_result(tmp_path: Path) -> FinalResultsEvaluationResult:
    results_dir = tmp_path / "data" / "webqsp" / "results" / "1_test"
    sources_dir = tmp_path / "sources"
    model_run_dir = sources_dir / "models" / "7_model"
    answers_path = sources_dir / "answers.jsonl"
    reasoning_path = sources_dir / "reasoning.jsonl"
    predictions_path = sources_dir / "predictions.jsonl"
    inference_config_path = sources_dir / "inference_config.json"
    evaluation_config_path = sources_dir / "evaluation_config.json"
    for path in [reasoning_path, predictions_path]:
        _write_json(path, {"source": path.name})
    _write_json(
        model_run_dir / "model_config.json",
        {
            "dataset_id": "webqsp",
            "run_name": "7_model",
            "run_number": 7,
            "entity_embedding_model": "text-embedding-3-small",
            "question_embedding_model": "text-embedding-3-small",
            "relation_embedding_model": "text-embedding-3-small",
            "trained_instances": 123,
            "loss_history": [
                {"epoch": 1, "average_loss": 0.8},
                {"epoch": 2, "average_loss": 0.4},
            ],
            "training": {
                "epochs": 3,
                "learning_rate": 0.001,
                "gnn_layer_count": 2,
                "hidden_dimension": 256,
                "loss_function": "BCEWithLogitsLoss",
            },
        },
    )
    _write_json(
        evaluation_config_path,
        {
            "dataset_id": "webqsp",
            "run_name": "1_eval",
            "run_number": 1,
            "evaluated_instances": 1,
            "selected_device": "cpu",
            "model_config": {
                "model_run_name": "7_model",
                "model_run_number": 7,
                "full_config_path": str(model_run_dir / "model_config.json"),
                "weights_path": str(model_run_dir / "gnn_answer_retriever.pt"),
            },
            "evaluation": {"candidate_limit": 15},
        },
    )
    _write_json(
        inference_config_path,
        {
            "run_name": "1_inference",
            "run_number": 1,
            "evaluation_config": {
                "evaluation_run_name": "1_eval",
                "evaluation_run_number": 1,
                "full_config_path": str(evaluation_config_path),
                "predictions_path": str(predictions_path),
            },
            "inference": {
                "model_id": "test-model",
                "evidence_subgraph": {"algorithm": "pcst"},
                "total_requests": 1,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            },
        },
    )
    _write_jsonl(
        answers_path,
        [
            {
                "instance_index": 0,
                "explanation": "Moon -> orbits -> Earth",
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "instance_index": 0,
                "answer_candidates": [
                    {"node": "Earth"},
                    {"node": "Mars"},
                ],
            }
        ],
    )
    results_config_path = results_dir / "results_config.json"
    retrieval_metrics_path = results_dir / "retrieval_metrics.json"
    reasoning_metrics_path = results_dir / "reasoning_metrics.json"
    per_instance_results_path = results_dir / "per_instance_results.jsonl"
    _write_json(
        results_config_path,
        {
            "dataset_id": "webqsp",
            "model_id": "test-model",
            "gnn_architecture": "graphsage",
            "run_name": "1_test",
            "run_number": 1,
            "configs": {
                "model_config_path": str(model_run_dir / "model_config.json"),
                "evaluation_config_path": str(evaluation_config_path),
                "inference_config_path": str(inference_config_path),
                "results_config_path": str(results_config_path),
            },
            "artifacts": {
                "training": {
                    "name": "7_model",
                    "model_config_path": str(model_run_dir / "model_config.json"),
                    "weights_path": str(model_run_dir / "gnn_answer_retriever.pt"),
                },
                "evaluation": {
                    "name": "1_eval",
                    "evaluation_config_path": str(evaluation_config_path),
                    "predictions_path": str(predictions_path),
                },
                "inference": {
                    "name": "1_inference",
                    "inference_config_path": str(inference_config_path),
                    "answers_path": str(answers_path),
                    "reasoning_path": str(reasoning_path),
                },
            },
        },
    )
    _write_json(
        retrieval_metrics_path,
        {
            "evaluated_instances": 1,
            "hits_at_1": 0.5,
            "hits_at_5": 1.0,
            "hits_at_10": 1.0,
            "hits_at_candidate_limit": 1.0,
            "average_candidate_count": 2.0,
            "missing_gold_in_graph_count": 0,
        },
    )
    _write_json(
        reasoning_metrics_path,
        {
            "accuracy": 0.5,
            "hit_rate": 0.5,
            "hits_at_1": 0.5,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "grounded_explanation_rate": 0.5,
            "fully_grounded_explanation_rate": 0.5,
            "ndcg_at_1": 0.5,
            "ndcg_at_5": 0.75,
            "ndcg_at_10": 0.75,
            "ndcg_at_candidate_limit": 0.75,
            "conditioned_evaluated_instances": 20,
            "retrieval_gold_coverage": 0.8,
            "retrieval_full_gold_coverage_count": 12,
            "retrieval_full_gold_coverage_rate": 0.6,
            "reasoning_context_gold_coverage": 0.7,
            "reasoning_context_full_gold_coverage_rate": 0.5,
            "llm_retrieved_gold_utilization": 0.75,
            "llm_omission_given_full_retrieval_rate": 0.25,
            "llm_omission_given_full_retrieval_count": 3,
            "llm_exact_match_given_full_retrieval": 0.7,
            "llm_omission_given_full_context_rate": 0.2,
            "llm_exact_match_given_full_context": 0.75,
            "full_retrieval_complete_answer_rate": 0.4,
            "full_retrieval_llm_omission_rate": 0.2,
            "partial_retrieval_fully_utilized_rate": 0.15,
            "partial_retrieval_underutilized_rate": 0.1,
            "full_context_complete_answer_rate": 0.35,
            "full_context_llm_omission_rate": 0.15,
            "partial_context_fully_utilized_rate": 0.2,
            "partial_context_underutilized_rate": 0.1,
            "no_gold_retrieved_no_gold_answered_rate": 0.1,
            "correct_without_gold_retrieval_rate": 0.05,
        },
    )
    _write_jsonl(
        per_instance_results_path,
        [
            {
                "instance_index": 0,
                "question": "What does the Moon orbit?",
                "q_entity": ["Moon"],
                "gold_answers": ["Earth"],
                "predicted_answers": ["Earth"],
                "exact_match": True,
                "hit": True,
                "hits_at_1": True,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "mentioned_triple_count": 1,
                "grounded_mentioned_triple_count": 1,
                "grounded_explanation": True,
                "fully_grounded_explanation": True,
                "ndcg_at_1": 1.0,
                "ndcg_at_5": 1.0,
                "ndcg_at_10": 1.0,
                "ndcg_at_candidate_limit": 1.0,
                "answer_error_message": None,
                "retrieval_gold_coverage": 1.0,
                "reasoning_context_gold_coverage": 1.0,
                "llm_retrieved_gold_utilization": 1.0,
                "retrieval_generation_outcome": (
                    "full_retrieval_complete_answer"
                ),
            }
        ],
    )
    return FinalResultsEvaluationResult(
        dataset_id="webqsp",
        results_run_directory=results_dir,
        results_run_name="1_test",
        results_run_number=1,
        results_config_path=results_config_path,
        retrieval_metrics_path=retrieval_metrics_path,
        reasoning_metrics_path=reasoning_metrics_path,
        per_instance_results_path=per_instance_results_path,
        evaluated_instances=1,
        accuracy=0.5,
        hit_rate=0.5,
        hits_at_1=0.5,
        precision=1.0,
        recall=0.5,
        f1=2 / 3,
        grounded_explanation_rate=0.5,
        ndcg_at_10=0.75,
    )


def test_wandb_payload_construction(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)
    service = WandbFinalResultsLoggingService()
    retrieval_metrics = service._load_json_object(final_result.retrieval_metrics_path)
    reasoning_metrics = service._load_json_object(final_result.reasoning_metrics_path)
    results_config = service._load_json_object(final_result.results_config_path)
    per_instance_rows = service._load_jsonl_objects(
        final_result.per_instance_results_path
    )

    scalars = service.build_scalar_metrics(retrieval_metrics, reasoning_metrics)
    aggregate_rows = service.build_aggregate_metric_rows(scalars)
    summary_plot_metrics = service.build_summary_plot_metrics(scalars)
    table_rows = service.build_table_rows(results_config, per_instance_rows)
    wandb_config = service.build_wandb_config(final_result, results_config)
    run_summary_metrics = service.build_run_summary_plot_metrics(
        scalar_metrics=scalars,
        wandb_config=wandb_config,
    )

    assert scalars["retrieval_hits_at_1"] == 0.5
    assert scalars["retrieval_evaluated_instances"] == 1
    assert scalars["retrieval_hits_at_candidate_limit"] == 1.0
    assert scalars["answer_f1"] == 2 / 3
    assert scalars["ranking_ndcg_at_10"] == 0.75
    assert scalars["retrieval_gold_coverage"] == 0.8
    assert scalars["conditioned_evaluated_instances"] == 20
    assert scalars["llm_omission_given_full_retrieval_rate"] == 0.25
    assert scalars["llm_exact_match_given_full_context"] == 0.75
    assert ["answer", "f1", 2 / 3] in aggregate_rows
    assert ["retrieval", "hits_at_1", 0.5] in aggregate_rows
    assert ["retrieval", "hits_at_candidate_limit", 1.0] in aggregate_rows
    assert summary_plot_metrics["Summary_Plots/answer_f1"] == 2 / 3
    assert summary_plot_metrics["Summary_Plots/retrieval_hits_at_1"] == 0.5
    assert summary_plot_metrics["Summary_Plots/retrieval_evaluated_instances"] == 1
    assert summary_plot_metrics["Summary_Plots/retrieval_gold_coverage"] == 0.8
    assert (
        summary_plot_metrics[
            "Summary_Plots/llm_omission_given_full_retrieval_rate"
        ]
        == 0.25
    )
    assert (
        summary_plot_metrics["Summary_Plots/retrieval_hits_at_candidate_limit"]
        == 1.0
    )
    assert run_summary_metrics["Run_Summary/retrieval_hits_at_1"] == 0.5
    assert run_summary_metrics["Run_Summary/retrieval_evaluated_instances"] == 1
    assert run_summary_metrics["Run_Summary/retrieval_gold_coverage"] == 0.8
    assert run_summary_metrics["Run_Summary/retrieval_full_gold_coverage"] == 0.6
    assert run_summary_metrics["Run_Summary/reasoning_context_gold_coverage"] == 0.7
    assert (
        run_summary_metrics["Run_Summary/reasoning_context_full_gold_coverage"]
        == 0.5
    )
    assert run_summary_metrics["Run_Summary/llm_exact_match_given_full_context"] == 0.75
    assert run_summary_metrics["Run_Summary/llm_omission_given_full_context"] == 0.2
    assert (
        run_summary_metrics["Run_Summary/llm_exact_match_given_full_retrieval"]
        == 0.7
    )
    assert (
        run_summary_metrics["Run_Summary/llm_omission_given_full_retrieval"]
        == 0.25
    )
    assert "Run_Summary/conditioned_evaluated_instances" not in run_summary_metrics
    assert "Run_Summary/llm_omission_given_full_retrieval_rate" not in run_summary_metrics
    assert run_summary_metrics["Run_Summary/full_retrieval_complete_answer"] == 0.4
    assert run_summary_metrics["Run_Summary/full_retrieval_llm_omission"] == 0.2
    assert run_summary_metrics["Run_Summary/partial_retrieval_fully_utilized"] == 0.15
    assert run_summary_metrics["Run_Summary/partial_retrieval_underutilized"] == 0.1
    assert run_summary_metrics["Run_Summary/full_context_complete_answer"] == 0.35
    assert run_summary_metrics["Run_Summary/full_context_llm_omission"] == 0.15
    assert run_summary_metrics["Run_Summary/partial_context_fully_utilized"] == 0.2
    assert run_summary_metrics["Run_Summary/partial_context_underutilized"] == 0.1
    assert (
        summary_plot_metrics["Summary_Plots/full_context_complete_answer_rate"]
        == 0.35
    )
    assert (
        summary_plot_metrics["Summary_Plots/full_context_llm_omission_rate"]
        == 0.15
    )
    assert (
        summary_plot_metrics["Summary_Plots/partial_context_fully_utilized_rate"]
        == 0.2
    )
    assert (
        summary_plot_metrics["Summary_Plots/partial_context_underutilized_rate"]
        == 0.1
    )
    assert run_summary_metrics["Run_Summary/no_gold_retrieved_no_gold_answered"] == 0.1
    assert run_summary_metrics["Run_Summary/correct_without_gold_retrieval"] == 0.05
    assert "Run_Summary/retrieval_hits_at_5" not in run_summary_metrics
    assert run_summary_metrics["Run_Summary/retrieval_hits_at_10"] == 1.0
    assert (
        run_summary_metrics["Run_Summary/retrieval_hits_at_candidate_limit"]
        == 1.0
    )
    assert run_summary_metrics["Run_Summary/answer_hit_rate"] == 0.5
    assert run_summary_metrics["Run_Summary/answer_f1"] == 2 / 3
    assert run_summary_metrics["Run_Summary/ranking_ndcg_at_10"] == 0.75
    assert (
        run_summary_metrics["Run_Summary/grounded_explanation_rate"]
        == 0.5
    )
    assert service.table_columns[0] == "instance_index"
    assert len(table_rows) == 1
    assert table_rows[0][2] == "Moon"
    assert table_rows[0][3] == "Earth"
    assert table_rows[0][4] == "Earth"
    assert table_rows[0][5] == "Moon -> orbits -> Earth"
    assert table_rows[0][-5] == 1.0
    assert table_rows[0][-4] == 1.0
    assert table_rows[0][-3] == 1.0
    assert table_rows[0][-2] == "full_retrieval_complete_answer"
    assert table_rows[0][-1] == "Earth, Mars"
    assert set(wandb_config) == {
        "configs",
        "dataset_id",
        "model_id",
        "runs",
        "source_paths",
    }
    assert wandb_config["runs"]["model"]["number"] == 7
    assert wandb_config["runs"]["evaluation"]["number"] == 1
    assert set(wandb_config["source_paths"]) == {
        "model_config_path",
        "evaluation_config_path",
        "inference_config_path",
        "results_config_path",
        "training_model_config_path",
        "training_weights_path",
        "evaluation_evaluation_config_path",
        "evaluation_predictions_path",
        "inference_inference_config_path",
        "inference_answers_path",
        "inference_reasoning_path",
    }
    assert "training_name" not in wandb_config["source_paths"]
    assert "evaluation_name" not in wandb_config["source_paths"]
    assert "inference_name" not in wandb_config["source_paths"]
    assert wandb_config["configs"]["model"]["training"]["epochs"] == 3
    assert (
        wandb_config["configs"]["model"]["training"]["loss_function"]
        == "BCEWithLogitsLoss"
    )
    assert wandb_config["configs"]["evaluation"]["candidate_limit"] == 15
    assert wandb_config["configs"]["inference"]["total_requests"] == 1
    assert wandb_config["configs"]["evidence"] == {"algorithm": "pcst"}
    assert "evidence_subgraph" not in wandb_config["configs"]["inference"]
    loss_points = service.build_training_loss_points(wandb_config["configs"]["model"])
    assert loss_points == [
        {"epoch": 1, "average_loss": 0.8},
        {"epoch": 2, "average_loss": 0.4},
    ]
    assert (
        "should_not"
        not in wandb_config["configs"]["model"]
    )


def test_wandb_logging_success_with_fake_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_result = _fake_final_result(tmp_path)
    captured: dict[str, object] = {}

    class FakeRun:
        id = "run-1"
        url = "https://wandb.test/run-1"

        def __init__(self):
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def log(self, payload, step=None):
            captured.setdefault("logs", []).append(
                {
                    "payload": payload,
                    "step": step,
                }
            )
            self.summary.update(
                {
                    key: value
                    for key, value in payload.items()
                    if isinstance(value, int | float | str)
                }
            )

        def log_artifact(self, artifact):
            captured["artifact_files"] = artifact.files

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    def fake_init(**kwargs):
        captured["init"] = kwargs
        run = FakeRun()
        captured["run"] = run
        return run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Table=FakeTable,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    result = WandbFinalResultsLoggingService().log_final_results(
        final_result=final_result,
        config=WandbFinalResultsConfig(
            project="project",
            entity="entity",
            mode="disabled",
        ),
    )

    assert result.status == "logged"
    assert result.run_id == "run-1"
    assert captured["init"]["project"] == "project"
    assert captured["init"]["entity"] == "entity"
    assert captured["init"]["mode"] == "disabled"
    assert captured["init"]["name"] == "1_test_graphsage_pcst_test-model"
    assert captured["init"]["config"]["runs"]["model"]["number"] == 7
    assert captured["init"]["config"]["configs"]["model"]["training"]["epochs"] == 3
    assert "model_run_number:7" in captured["init"]["tags"]
    assert "evaluation_run_number:1" in captured["init"]["tags"]
    assert "inference_run_number:1" in captured["init"]["tags"]
    assert "results_run_number:1" not in captured["init"]["tags"]
    assert "trained_instances:123" in captured["init"]["tags"]
    assert "evaluated_instances:1" in captured["init"]["tags"]
    assert "text-embedding-3-small" in captured["init"]["tags"]
    assert "graphsage" in captured["init"]["tags"]
    assert "7_model" not in captured["init"]["tags"]
    assert "1_eval" not in captured["init"]["tags"]
    assert "1_inference" not in captured["init"]["tags"]
    run_summary_payload = captured["run"].summary
    assert run_summary_payload["Run_Summary/retrieval_hits_at_1"] == 0.5
    assert run_summary_payload["Run_Summary/retrieval_gold_coverage"] == 0.8
    assert run_summary_payload["Run_Summary/retrieval_full_gold_coverage"] == 0.6
    assert run_summary_payload["Run_Summary/reasoning_context_gold_coverage"] == 0.7
    assert (
        run_summary_payload["Run_Summary/reasoning_context_full_gold_coverage"]
        == 0.5
    )
    assert run_summary_payload["Run_Summary/llm_exact_match_given_full_context"] == 0.75
    assert run_summary_payload["Run_Summary/llm_omission_given_full_context"] == 0.2
    assert (
        run_summary_payload["Run_Summary/llm_exact_match_given_full_retrieval"]
        == 0.7
    )
    assert (
        run_summary_payload["Run_Summary/llm_omission_given_full_retrieval"]
        == 0.25
    )
    assert "Run_Summary/retrieval_hits_at_5" not in run_summary_payload
    assert run_summary_payload["Run_Summary/retrieval_hits_at_10"] == 1.0
    assert (
        run_summary_payload["Run_Summary/retrieval_hits_at_candidate_limit"]
        == 1.0
    )
    assert run_summary_payload["Run_Summary/answer_hit_rate"] == 0.5
    assert run_summary_payload["Run_Summary/answer_f1"] == 2 / 3
    assert run_summary_payload["Run_Summary/ranking_ndcg_at_10"] == 0.75
    assert run_summary_payload["Run_Summary/grounded_explanation_rate"] == 0.5
    aggregate_log = captured["logs"][0]
    assert aggregate_log["step"] is None
    assert aggregate_log["payload"]["Run_Summary/retrieval_hits_at_1"] == 0.5
    assert aggregate_log["payload"]["Summary_Plots/answer_f1"] == 2 / 3
    assert captured["logs"][1] == {
        "payload": {"Training/gnn_training_loss": 0.8},
        "step": 1,
    }
    assert captured["logs"][2] == {
        "payload": {"Training/gnn_training_loss": 0.4},
        "step": 2,
    }
    logged_payload = captured["logs"][3]["payload"]
    assert "answer_f1" not in logged_payload
    assert "Summary_Metrics/aggregate_metrics" in logged_payload
    assert "Per_Instance_Metrics/per_instance_results" in logged_payload
    assert run_summary_payload["Summary_Plots/answer_f1"] == 2 / 3
    assert run_summary_payload["Summary_Plots/retrieval_hits_at_1"] == 0.5
    assert "Run_Summary/answer_f1" not in logged_payload
    assert "gnn_training_loss" not in logged_payload
    aggregate_table = logged_payload["Summary_Metrics/aggregate_metrics"]
    assert aggregate_table.columns == ["group", "metric", "value"]
    assert ["answer", "f1", 2 / 3] in aggregate_table.data
    assert len(captured["logs"]) == 4
    assert any(service_name == "results/results_config.json" for _, service_name in captured["artifact_files"])
    assert not any(service_name.startswith("sources/") for _, service_name in captured["artifact_files"])


def test_wandb_step_handles_service_failure(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)

    class FailingService:
        def log_final_results(self, final_result, config):
            raise RuntimeError("boom")

    result = LogFinalResultsToWandbStep(
        project="project",
        mode="disabled",
        logging_service=FailingService(),
    ).execute(StepContext(result=final_result))

    assert result.wandb_status == "failed"
    assert result.wandb_error_message == "boom"


def test_wandb_step_records_success(tmp_path: Path) -> None:
    final_result = _fake_final_result(tmp_path)

    class SuccessfulService:
        def log_final_results(self, final_result, config):
            return WandbFinalResultsLogResult(
                status="logged",
                run_id="run-1",
                run_url="https://wandb.test/run-1",
            )

    result = LogFinalResultsToWandbStep(
        project="project",
        mode="disabled",
        logging_service=SuccessfulService(),
    ).execute(StepContext(result=final_result))

    assert result.wandb_status == "logged"
    assert result.wandb_run_id == "run-1"
    assert result.wandb_run_url == "https://wandb.test/run-1"


def test_shared_experiment_restores_legacy_run_summary_metrics(
    tmp_path: Path,
) -> None:
    final_result = _fake_final_result(tmp_path)

    class CapturingCoordinator:
        def __init__(self) -> None:
            self.logged: list[dict] = []
            self.summaries: list[dict] = []
            self.config_updates: list[dict] = []
            self.artifact_calls: list[dict] = []
            self.metadata = SimpleNamespace(
                status="logged",
                run_id="run-1",
                run_url="https://wandb.test/run-1",
                error_message=None,
            )

        def log(self, payload, **kwargs) -> None:
            self.logged.append(payload)

        def log_aggregate_metrics(self, payload, **kwargs) -> None:
            self.summaries.append(payload)

        def update_config(self, payload, **kwargs) -> None:
            self.config_updates.append(payload)

        def persist_metadata(self, path) -> None:
            return None

        def log_artifact(self, **kwargs) -> None:
            self.artifact_calls.append(kwargs)

    coordinator = CapturingCoordinator()
    result = LogFinalResultsToWandbStep(
        coordinator=coordinator,
    ).execute_default(StepContext(result=final_result))

    payload = coordinator.summaries[0]
    assert "Run_Summary/retrieval_hits_at_1" not in payload
    assert "Run_Summary/retrieval_hits_at_10" not in payload
    assert payload["Run_Summary/retrieval_gold_coverage"] == 0.8
    assert payload["Run_Summary/retrieval_full_gold_coverage"] == 0.6
    assert payload["Run_Summary/reasoning_context_gold_coverage"] == 0.7
    assert payload["Run_Summary/reasoning_context_full_gold_coverage"] == 0.5
    assert payload["Run_Summary/llm_exact_match_given_full_context"] == 0.75
    assert payload["Run_Summary/llm_omission_given_full_context"] == 0.2
    assert payload["Run_Summary/llm_exact_match_given_full_retrieval"] == 0.7
    assert payload["Run_Summary/llm_omission_given_full_retrieval"] == 0.25
    assert payload["Run_Summary/full_retrieval_complete_answer"] == 0.4
    assert payload["Run_Summary/answer_hit_rate"] == 0.5
    assert payload["Run_Summary/answer_f1"] == 2 / 3
    assert payload["Run_Summary/ranking_ndcg_at_10"] == 0.75
    assert payload["Run_Summary/grounded_explanation_rate"] == 0.5
    assert "Summary_Plots/retrieval_hits_at_1" not in payload
    assert payload["Summary_Plots/retrieval_gold_coverage"] == 0.8
    assert payload["Summary_Plots/answer_accuracy"] == 0.5
    assert payload["Summary_Plots/answer_f1"] == 2 / 3
    assert payload["Summary_Plots/grounding_fully_grounded_explanation_rate"] == 0.5
    assert payload["Summary_Plots/ranking_ndcg_at_1"] == 0.5
    assert payload["Summary_Plots/ranking_ndcg_at_candidate_limit"] == 0.75
    assert not any(key.startswith("Inference/") for key in payload)
    config_payload = coordinator.config_updates[0]
    assert set(config_payload["configs"]) == {
        "model",
        "evaluation",
        "evidence",
        "inference",
    }
    assert set(config_payload["runs"]) == {
        "model",
        "evaluation",
        "inference",
        "results",
    }
    assert coordinator.artifact_calls == [
        {
            "name": f"results-{final_result.results_run_name}",
            "artifact_type": "final-results",
            "paths": [
                final_result.results_config_path,
                final_result.retrieval_metrics_path,
                final_result.reasoning_metrics_path,
                final_result.per_instance_results_path,
            ],
            "source_config_path": tmp_path / "sources" / "evaluation_config.json",
        }
    ]
    assert result.wandb_status == "logged"
