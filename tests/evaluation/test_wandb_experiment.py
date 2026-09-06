"""Tests for the resumable stage-aware W&B experiment lifecycle."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pipeline.evaluation.services.wandb_experiment import (
    WandbExperimentCoordinator,
    WandbRunIdentifierService,
)


def test_wandb_tags_include_pcst_strategy_and_semantic_embedding() -> None:
    tags = WandbExperimentCoordinator._build_tags_from_config(
        {
            "configs": {
                "evidence": {
                    "algorithm": "pcst",
                    "pcst": {
                        "edge_cost_strategy": "semantic",
                        "semantic_embedding_model": "text-embedding-3-small",
                    },
                }
            }
        }
    )
    assert "pcst" in tags
    assert "pcst-semantic" in tags
    assert "text-embedding-3-small" in tags
from pipeline.exceptions import PipelineException
from pipeline.abstract import StepContext
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationResult,
    SavedLlmInferenceRun,
)
from pipeline.evaluation.steps.wandb_stages import (
    LogInferenceToWandbStep,
    LogRetrieverToWandbStep,
    LogTrainingToWandbStep,
)
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainedGnnAnswerRetriever,
)


class FakeConfig(dict):
    def update(self, payload=None, *, allow_val_change=False, **kwargs) -> None:
        super().update(payload or {})
        super().update(kwargs)


class FakeRun:
    def __init__(self) -> None:
        self.id = "wandb-run-id"
        self.url = "https://wandb.test/run"
        self.logged: list[tuple[dict, int | None]] = []
        self.artifacts = []
        self.finished = False
        self.defined_metrics: list[tuple[str, dict]] = []
        self.config = FakeConfig()
        self.summary = FakeConfig()
        self.tags: tuple[str, ...] = ()

    def save(self, glob_str) -> None:
        raise AssertionError("Renaming a W&B run must not call Run.save().")

    def define_metric(self, name, **kwargs) -> None:
        self.defined_metrics.append((name, kwargs))

    def log(self, payload, step=None) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class CapturingCoordinator:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summaries: list[dict] = []
        self.config_updates: list[dict] = []
        self.artifact_calls: list[dict] = []
        self.persisted_metadata_paths: list = []
        self.tag_updates: list[list[str]] = []
        self.has_active_run = True
        self.metadata = type(
            "Metadata",
            (),
            {
                "status": "logged",
                "run_id": "wandb-run-id",
                "run_url": "https://wandb.test/run",
                "error_message": None,
            },
        )()

    def ensure_run(self, **kwargs) -> None:
        return None

    def log(self, payload, **kwargs) -> None:
        self.logged.append(payload)

    def log_aggregate_metrics(self, payload, **kwargs) -> None:
        self.summaries.append(payload)

    def update_config(self, payload, **kwargs) -> None:
        self.config_updates.append(payload)

    def update_tags(self, tags) -> None:
        self.tag_updates.append(tags)

    def persist_metadata(self, path) -> None:
        self.persisted_metadata_paths.append(path)

    def log_artifact(self, **kwargs) -> None:
        self.artifact_calls.append(kwargs)


def test_coordinator_uses_one_run_and_persists_lineage(tmp_path) -> None:
    fake_wandb = FakeWandb()
    config_path = tmp_path / "model_config.json"
    config_path.write_text("{}", encoding="utf-8")
    coordinator = WandbExperimentCoordinator(
        project="project",
        entity="team",
        run_root=tmp_path / "wandb_runs",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.log_training_progress(
            {"epoch": 1, "instance": 3, "global_step": 3, "loss": 0.4}
        )
        coordinator.log_training_epoch({"epoch": 1, "average_loss": 0.35})
        coordinator.log_training_epoch({"epoch": 1, "average_loss": 0.35})
        coordinator.update_config({"configs": {"model": {"epochs": 2}}})
        coordinator.update_config(
            {"configs": {"evaluation": {"candidate_limit": 10}}}
        )
        coordinator.log_aggregate_metrics(
            {"Run_Summary/retrieval_hits_at_1": 0.8}
        )
        coordinator.log_aggregate_metrics(
            {"Run_Summary/retrieval_hits_at_1": 0.8}
        )
        coordinator.persist_metadata(config_path)
        coordinator.finish()

    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.init_calls[0]["name"].startswith("1_")
    assert fake_wandb.init_calls[0]["name"] != "gnn-training"
    assert fake_wandb.run.logged[0][0]["Training/global_step"] == 3
    epoch_average_events = [
        payload
        for payload, _ in fake_wandb.run.logged
        if "Training/gnn_training_loss" in payload
    ]
    assert epoch_average_events == [
        {
            "Training/gnn_training_loss": 0.35,
            "Training/epoch": 1,
        }
    ]
    assert fake_wandb.run.config["configs"] == {
        "model": {"epochs": 2},
        "evaluation": {"candidate_limit": 10},
    }
    aggregate_events = [
        payload
        for payload, _ in fake_wandb.run.logged
        if "Run_Summary/retrieval_hits_at_1" in payload
    ]
    assert aggregate_events == [{"Run_Summary/retrieval_hits_at_1": 0.8}]
    assert "graphragx" in fake_wandb.run.tags
    assert ("Training/global_step", {}) in fake_wandb.run.defined_metrics
    assert (
        "Training/loss",
        {"step_metric": "Training/global_step"},
    ) in fake_wandb.run.defined_metrics
    assert ("Training/epoch", {}) in fake_wandb.run.defined_metrics
    assert (
        "Training/gnn_training_loss",
        {"step_metric": "Training/epoch"},
    ) in fake_wandb.run.defined_metrics
    assert fake_wandb.run.finished is True
    tracking = json.loads(config_path.read_text(encoding="utf-8"))["wandb"]
    assert tracking["run_id"] == "wandb-run-id"
    assert tracking["run_name"] == fake_wandb.init_calls[0]["name"]
    assert tracking["project"] == "project"


def test_coordinator_appends_architecture_to_new_run_name(tmp_path) -> None:
    fake_wandb = FakeWandb()
    coordinator = WandbExperimentCoordinator(
        project="project",
        run_root=tmp_path / "wandb_runs",
        architecture_name="aa-graphsage",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.ensure_run()

    run_name = fake_wandb.init_calls[0]["name"]
    assert run_name.startswith("1_")
    assert run_name.endswith("_aa-graphsage")
    assert (tmp_path / "wandb_runs" / run_name.removesuffix("_aa-graphsage")).is_dir()


def test_inference_run_name_appends_available_algorithm_and_model() -> None:
    assert WandbExperimentCoordinator.build_inference_run_name(
        "1_20260829_120000_hgt",
        evidence_algorithm="shortest_path",
        model_id="provider/model 1",
    ) == "1_20260829_120000_hgt_sp_provider-model-1"
    assert WandbExperimentCoordinator.build_inference_run_name(
        "1_20260829_120000_hgt",
        evidence_algorithm=None,
        model_id=None,
    ) == "1_20260829_120000_hgt"
    assert WandbExperimentCoordinator.build_inference_run_name(
        "1_20260829_120000_hgt",
        evidence_algorithm="pcst",
        model_id=None,
    ) == "1_20260829_120000_hgt_pcst"


def test_coordinator_preserves_persisted_lineage_run_name(tmp_path) -> None:
    fake_wandb = FakeWandb()
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "gnn_architecture": "hgt",
                "wandb": {
                    "status": "logged",
                    "run_id": "existing-id",
                    "run_name": "17_20260826_120000_hgt",
                    "project": "project",
                },
            }
        ),
        encoding="utf-8",
    )
    coordinator = WandbExperimentCoordinator(
        project="project",
        run_root=tmp_path / "wandb_runs",
        architecture_name="hgt",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.ensure_run(source_config_path=config_path)

    assert fake_wandb.init_calls[0]["name"] == "17_20260826_120000_hgt"


def test_coordinator_resumes_persisted_run(tmp_path) -> None:
    fake_wandb = FakeWandb()
    fake_wandb.run.config.update(
        {"configs": {"model": {"training": {"epochs": 3}}}}
    )
    config_path = tmp_path / "evaluation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "status": "logged",
                    "run_id": "existing-id",
                    "run_url": "https://wandb.test/existing",
                    "project": "project",
                    "entity": "team",
                }
            }
        ),
        encoding="utf-8",
    )
    coordinator = WandbExperimentCoordinator(project="project", entity="team")

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.ensure_run(source_config_path=config_path)
        coordinator.update_config(
            {"configs": {"evaluation": {"candidate_limit": 20}}}
        )

    assert fake_wandb.init_calls[0]["id"] == "existing-id"
    assert fake_wandb.init_calls[0]["resume"] == "allow"
    assert fake_wandb.run.config["configs"] == {
        "model": {"training": {"epochs": 3}},
        "evaluation": {"candidate_limit": 20},
    }


def test_coordinator_can_start_new_run_without_resuming_lineage(tmp_path) -> None:
    fake_wandb = FakeWandb()
    config_path = tmp_path / "evaluation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "status": "logged",
                    "run_id": "retriever-run-id",
                    "run_name": "7_retriever",
                    "project": "project",
                }
            }
        ),
        encoding="utf-8",
    )
    coordinator = WandbExperimentCoordinator(
        project="project",
        resume_from_lineage=False,
        run_root=tmp_path / "wandb_runs",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.ensure_run(source_config_path=config_path)

    init_payload = fake_wandb.init_calls[0]
    assert "id" not in init_payload
    assert "resume" not in init_payload
    assert init_payload["name"].startswith("1_")


def test_config_updates_add_all_available_stage_tags(tmp_path) -> None:
    fake_wandb = FakeWandb()
    coordinator = WandbExperimentCoordinator(
        project="project",
        run_root=tmp_path / "wandb_runs",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        coordinator.update_config(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "aa-graphsage",
                "model_id": "gpt-5.4-mini",
                "runs": {
                    "model": {"number": 59},
                    "evaluation": {"number": 64},
                    "inference": {"number": 55},
                },
                "configs": {
                    "model": {
                        "entity_embedding_model": "text-embedding-3-small",
                        "question_embedding_model": "text-embedding-3-small",
                        "relation_embedding_model": "text-embedding-3-small",
                        "trained_instances": 100,
                    }
                },
            }
        )
        coordinator.update_tags(["evaluated_instances:100"])

    assert set(fake_wandb.run.tags) == {
        "graphragx",
        "WebQSP",
        "aa-graphsage",
        "gpt-5.4-mini",
        "text-embedding-3-small",
        "trained_instances:100",
        "model_run_number:59",
        "evaluation_run_number:64",
        "inference_run_number:55",
        "evaluated_instances:100",
    }


def test_coordinator_rejects_conflicting_persisted_project(tmp_path) -> None:
    config_path = tmp_path / "evaluation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "status": "logged",
                    "run_id": "existing-id",
                    "project": "old-project",
                    "entity": None,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PipelineException, match="project"):
        WandbExperimentCoordinator(project="new-project").ensure_run(
            source_config_path=config_path
        )


def test_operational_initialization_failure_is_non_fatal(tmp_path) -> None:
    coordinator = WandbExperimentCoordinator(
        project="project",
        run_root=tmp_path / "wandb_runs",
    )

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        side_effect=RuntimeError("offline"),
    ):
        coordinator.log({"Run_Summary/retrieval_hits_at_1": 0.5})

    assert coordinator.metadata.status == "failed"
    assert coordinator.metadata.error_message == "offline"


def test_inference_stage_does_not_log_raw_scalar_reports(tmp_path) -> None:
    fake_wandb = FakeWandb()
    coordinator = WandbExperimentCoordinator(
        project="project",
        run_root=tmp_path / "wandb_runs",
    )
    step = LogInferenceToWandbStep(coordinator=coordinator)

    with patch(
        "pipeline.evaluation.services.wandb_experiment.importlib.import_module",
        return_value=fake_wandb,
    ):
        for run_number, run_name in [(1, "first"), (2, "second")]:
            run_dir = tmp_path / run_name
            run_dir.mkdir()
            config_path = run_dir / "inference_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "WebQSP",
                        "run_name": run_name,
                        "run_number": run_number,
                        "evaluation_config": {},
                        "inference": {
                            "model_id": "gpt-test",
                            "total_tokens": run_number * 10,
                            "evidence_subgraph": {
                                "algorithm": "shortest_path",
                            },
                            "evidence_metrics": {
                                "candidate_reduction_percentage": (
                                    run_number * 10.0
                                ),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            answers_path = run_dir / "answers.jsonl"
            reasoning_path = run_dir / "reasoning.jsonl"
            answers_path.write_text("", encoding="utf-8")
            reasoning_path.write_text("", encoding="utf-8")
            step.execute_default(
                StepContext(
                    result=SavedLlmInferenceRun(
                        dataset_id="WebQSP",
                        evaluation_run_name="1_eval",
                        inference_run_directory=run_dir,
                        inference_run_name=run_name,
                        inference_run_number=run_number,
                        model_id="gpt-test",
                        total_instances=1,
                        successful_answers=1,
                        failed_answers=0,
                        reasoning_path=reasoning_path,
                        answers_path=answers_path,
                        inference_config_path=config_path,
                    )
                )
            )

    assert len(fake_wandb.init_calls) == 1
    logged_keys = {
        key for payload, _ in fake_wandb.run.logged for key in payload
    }
    assert not any(key.startswith("Inference/") for key in logged_keys)
    assert "Summary_Plots/evidence_candidate_reduction_percentage" in logged_keys
    assert "Run_Summary/evidence_candidate_reduction_percentage" in logged_keys
    aggregate_events = [
        payload
        for payload, _ in fake_wandb.run.logged
        if "Summary_Plots/evidence_candidate_reduction_percentage" in payload
    ]
    assert len(aggregate_events) == 1
    assert aggregate_events[0][
        "Summary_Plots/evidence_candidate_reduction_percentage"
    ] == 10.0
    assert aggregate_events[0][
        "Run_Summary/evidence_candidate_reduction_percentage"
    ] == 10.0
    assert fake_wandb.run.summary[
        "Summary_Plots/evidence_candidate_reduction_percentage"
    ] == 20.0
    assert fake_wandb.run.summary[
        "Run_Summary/evidence_candidate_reduction_percentage"
    ] == 20.0
    assert fake_wandb.run.config["dataset_id"] == "WebQSP"
    assert fake_wandb.run.config["model_id"] == "gpt-test"
    assert fake_wandb.run.config["runs"]["inference"] == {
        "name": "second",
        "number": 2,
    }
    assert fake_wandb.run.config["configs"]["inference"]["total_tokens"] == 20
    assert fake_wandb.run.config["configs"]["evidence"] == {
        "algorithm": "shortest_path"
    }
    assert "evidence_subgraph" not in fake_wandb.run.config["configs"]["inference"]
    assert "WebQSP" in fake_wandb.run.tags
    assert "gpt-test" in fake_wandb.run.tags
    assert "inference_run_number:2" in fake_wandb.run.tags
    assert fake_wandb.run.name.endswith("_sp_gpt-test")
    assert coordinator.metadata.run_name == fake_wandb.run.name


def test_run_identifier_service_uses_one_global_counter(tmp_path) -> None:
    service = WandbRunIdentifierService()
    run_root = tmp_path / "wandb_runs"

    first = service.allocate(run_root)
    second = service.allocate(run_root)

    assert first.startswith("1_")
    assert second.startswith("2_")
    assert (run_root / first).is_dir()
    assert (run_root / second).is_dir()


def test_training_epoch_average_uses_legacy_metric_name(tmp_path) -> None:
    coordinator = CapturingCoordinator()
    config_path = tmp_path / "model_config.json"
    weights_path = tmp_path / "gnn_answer_retriever.pt"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "run_name": "1_model",
                "run_number": 1,
                "gnn_layer_count": 2,
                "hidden_dimension": 256,
                "entity_embedding_model": "text-embedding-3-small",
                "trained_instances": 10,
                "training": {"epochs": 1, "random_seed": 42},
            }
        ),
        encoding="utf-8",
    )
    weights_path.write_bytes(b"")
    result = TrainedGnnAnswerRetriever.model_construct(
        loss_history=[{"epoch": 1, "average_loss": 0.75}],
        model_config_path=config_path,
        model_artifact_path=weights_path,
        model_run_name="1_model",
    )

    LogTrainingToWandbStep(coordinator=coordinator).execute_default(
        StepContext(result=result)
    )

    assert coordinator.logged == [
        {"Training/gnn_training_loss": 0.75, "Training/epoch": 1}
    ]
    assert coordinator.config_updates[0]["dataset_id"] == "WebQSP"
    assert "gnn_architecture" not in coordinator.config_updates[0]
    assert "gnn_id" not in coordinator.config_updates[0]
    model_config = coordinator.config_updates[0]["configs"]["model"]
    assert "dataset_id" not in model_config
    assert model_config["training"]["epochs"] == 1
    assert model_config["training"]["random_seed"] == 42
    assert model_config["training"]["trained_instances"] == {
        "start": 0,
        "end": 10,
        "count": 10,
    }
    assert coordinator.artifact_calls[0]["paths"] == [config_path]


def test_training_wandb_upload_retriever_includes_weights(tmp_path) -> None:
    coordinator = CapturingCoordinator()
    config_path = tmp_path / "model_config.json"
    weights_path = tmp_path / "gnn_answer_retriever.pt"
    config_path.write_text(json.dumps({"dataset_id": "WebQSP"}), encoding="utf-8")
    weights_path.write_bytes(b"weights")
    result = TrainedGnnAnswerRetriever.model_construct(
        loss_history=[],
        model_config_path=config_path,
        model_artifact_path=weights_path,
        model_run_name="1_model",
    )

    LogTrainingToWandbStep(
        coordinator=coordinator,
        upload_retriever=True,
    ).execute_default(StepContext(result=result))

    assert coordinator.artifact_calls[0]["paths"] == [config_path, weights_path]


def test_retriever_stage_logs_legacy_run_summary_metrics(tmp_path) -> None:
    coordinator = CapturingCoordinator()
    model_dir = tmp_path / "model"
    evaluation_dir = tmp_path / "evaluation"
    model_dir.mkdir()
    evaluation_dir.mkdir()
    (model_dir / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "run_name": "1_model",
                "run_number": 1,
                "gnn_layer_count": 2,
                "hidden_dimension": 256,
                "loss_history": [{"epoch": 1, "average_loss": 0.6}],
            }
        ),
        encoding="utf-8",
    )
    evaluation_config_path = evaluation_dir / "evaluation_config.json"
    predictions_path = evaluation_dir / "predictions.jsonl"
    retrieval_metrics_path = evaluation_dir / "retrieval_metrics.json"
    evaluation_config_path.write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "run_name": "1_evaluation",
                "run_number": 1,
                "evaluation": {"candidate_limit": 10},
            }
        ),
        encoding="utf-8",
    )
    predictions_path.write_text("", encoding="utf-8")
    retrieval_metrics_path.write_text("{}", encoding="utf-8")
    result = GnnAnswerRetrieverEvaluationResult(
        dataset_id="webqsp",
        model_run_directory=model_dir,
        model_run_name="1_model",
        model_run_number=1,
        evaluation_run_directory=evaluation_dir,
        evaluation_run_name="1_evaluation",
        evaluation_run_number=1,
        evaluated_instances=10,
        hits_at_1=0.4,
        hits_at_1_count=4,
        hits_at_5=0.8,
        hits_at_5_count=8,
        hits_at_10=0.9,
        hits_at_10_count=9,
        hits_at_candidate_limit=1.0,
        hits_at_candidate_limit_count=10,
        ndcg_at_1=0.4,
        ndcg_at_5=0.7,
        ndcg_at_10=0.75,
        ndcg_at_candidate_limit=0.8,
        conditioned_evaluated_instances=10,
        retrieval_gold_coverage=0.65,
        retrieval_full_gold_coverage_count=5,
        retrieval_full_gold_coverage_rate=0.5,
        retrieved_gold_answer_count=8,
        average_candidate_count=12.0,
        missing_gold_in_graph_count=2,
        skipped_missing_gold_in_graph_count=2,
        predictions_path=predictions_path,
        evaluation_config_path=evaluation_config_path,
        retrieval_metrics_path=retrieval_metrics_path,
    )

    LogRetrieverToWandbStep(coordinator=coordinator).execute_default(
        StepContext(result=result)
    )

    payload = coordinator.summaries[0]
    assert payload["Run_Summary/retrieval_hits_at_1"] == 0.4
    assert payload["Run_Summary/retrieval_evaluated_instances"] == 10
    assert payload["Run_Summary/retrieval_hits_at_10"] == 0.9
    assert payload["Run_Summary/retrieval_hits_at_candidate_limit"] == 1.0
    assert payload["Summary_Plots/retrieval_hits_at_1"] == 0.4
    assert payload["Summary_Plots/retrieval_evaluated_instances"] == 10
    assert payload["Summary_Plots/retrieval_hits_at_5"] == 0.8
    assert payload["Summary_Plots/retrieval_hits_at_10"] == 0.9
    assert payload["Summary_Plots/retrieval_hits_at_candidate_limit"] == 1.0
    assert payload["Summary_Plots/retrieval_average_candidate_count"] == 12.0
    assert payload["Summary_Plots/retrieval_missing_gold_in_graph_count"] == 2
    assert payload["Summary_Plots/retrieval_skipped_missing_gold_in_graph_count"] == 2
    assert payload["Summary_Plots/retrieval_hits_at_1_count"] == 4
    assert payload["Summary_Plots/retrieval_hits_at_5_count"] == 8
    assert payload["Summary_Plots/retrieval_hits_at_10_count"] == 9
    assert payload["Summary_Plots/retrieval_hits_at_candidate_limit_count"] == 10
    assert payload["Summary_Plots/ranking_ndcg_at_1"] == 0.4
    assert payload["Summary_Plots/ranking_ndcg_at_5"] == 0.7
    assert payload["Summary_Plots/ranking_ndcg_at_10"] == 0.75
    assert payload["Summary_Plots/ranking_ndcg_at_candidate_limit"] == 0.8
    assert payload["Summary_Plots/conditioned_evaluated_instances"] == 10
    assert payload["Summary_Plots/retrieval_gold_coverage"] == 0.65
    assert payload["Summary_Plots/retrieval_full_gold_coverage_count"] == 5
    assert payload["Summary_Plots/retrieval_full_gold_coverage_rate"] == 0.5
    assert payload["Summary_Plots/retrieved_gold_answer_count"] == 8
    assert payload["Run_Summary/retrieval_gold_coverage"] == 0.65
    assert payload["Run_Summary/retrieval_full_gold_coverage"] == 0.5
    assert payload["Run_Summary/ranking_ndcg_at_10"] == 0.75
    assert not any(key.startswith("Retriever/") for key in payload)
    config_payload = coordinator.config_updates[0]
    assert config_payload["runs"]["model"] == {"name": "1_model", "number": 1}
    assert config_payload["runs"]["evaluation"] == {
        "name": "1_evaluation",
        "number": 1,
    }
    assert config_payload["configs"]["evaluation"] == {"candidate_limit": 10}
    assert coordinator.tag_updates == [["evaluated_instances:10"]]

    continuation_coordinator = CapturingCoordinator()
    continuation_result = result.model_copy(
        update={"wandb_run_id": "existing-lineage", "wandb_status": "logged"}
    )
    LogRetrieverToWandbStep(
        coordinator=continuation_coordinator
    ).execute_default(StepContext(result=continuation_result))

    assert continuation_coordinator.logged == []
    assert continuation_coordinator.summaries == []
    assert continuation_coordinator.artifact_calls == []
    assert len(continuation_coordinator.config_updates) == 1
    assert continuation_coordinator.tag_updates == [["evaluated_instances:10"]]

    copied_coordinator = CapturingCoordinator()
    LogRetrieverToWandbStep(
        coordinator=copied_coordinator,
        copy_to_new_experiment=True,
    ).execute_default(StepContext(result=continuation_result))

    assert copied_coordinator.summaries[0]["Run_Summary/retrieval_hits_at_1"] == 0.4
    assert copied_coordinator.summaries[0]["Summary_Plots/retrieval_hits_at_5"] == 0.8
    assert len(copied_coordinator.artifact_calls) == 1
    assert copied_coordinator.persisted_metadata_paths == []
    assert copied_coordinator.tag_updates == [["evaluated_instances:10"]]

    evaluation_only_coordinator = CapturingCoordinator()
    evaluation_only_coordinator.has_active_run = False
    LogRetrieverToWandbStep(
        coordinator=evaluation_only_coordinator,
    ).execute_default(StepContext(result=result))

    assert evaluation_only_coordinator.logged[0] == {
        "Training/epoch": 1,
        "Training/gnn_training_loss": 0.6,
    }
    assert (
        evaluation_only_coordinator.summaries[0][
            "Run_Summary/retrieval_hits_at_1"
        ]
        == 0.4
    )
    assert [call["artifact_type"] for call in evaluation_only_coordinator.artifact_calls] == [
        "gnn-model",
        "retriever-results",
    ]
