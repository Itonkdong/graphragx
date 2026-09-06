"""Tests for evidence-only artifact persistence and aggregate metrics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from pipeline.evaluation.models import (
    AnswerCandidateScore,
    EvaluatedAnswerRetrievalInstance,
    EvidenceSubgraphConstruction,
    ExtractedReasoningPaths,
    ExtractedReasoningPathsBatch,
    GnnAnswerRetrieverEvaluationResult,
    GraphTriple,
    ReasoningPathsForPrediction,
    SavedEvidenceSubgraphRun,
)
from pipeline.evaluation.models.path_extraction import EvaluationSample
from pipeline.evaluation.services.evidence_run_storage import (
    EvidenceRunStorageService,
)
from pipeline.evaluation.steps.wandb_stages import LogEvidenceToWandbStep
from pipeline.abstract import StepContext
from pipeline.preparation.steps.configuration_building import (
    BuiltPipelineConfiguration,
)


def test_evidence_run_saves_rows_metrics_and_lineage(tmp_path) -> None:
    evaluation_directory = tmp_path / "evaluations" / "7_eval"
    evaluation_directory.mkdir(parents=True)
    predictions_path = evaluation_directory / "predictions.jsonl"
    config_path = evaluation_directory / "evaluation_config.json"
    predictions_path.write_text("", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")
    prediction = EvaluatedAnswerRetrievalInstance(
        instance_index=0,
        question="Where was The Answer born?",
        q_entity=["Seed"],
        a_entity=["The Answer"],
        answer_candidates=[
            AnswerCandidateScore(
                node="The Answer",
                local_node_id=1,
                global_node_id=1,
                logit=3.0,
                probability=0.95,
                is_gold_answer=True,
                selection_reason="threshold",
            ),
            AnswerCandidateScore(
                node="Distractor",
                local_node_id=2,
                global_node_id=2,
                logit=1.0,
                probability=0.7,
                is_gold_answer=False,
                selection_reason="threshold",
            ),
        ],
        gold_answer_scores=[],
        hit_at_1=True,
        hit_at_5=True,
        hit_at_10=True,
        hit_at_candidate_limit=True,
        missing_gold_in_graph=False,
    )
    extracted = ExtractedReasoningPaths(
        sample=EvaluationSample(
            question=prediction.question,
            q_entities=prediction.q_entity,
            a_entities=prediction.a_entity,
            graph_triples=[GraphTriple(source="Seed", relation="r", target="Answer")],
        ),
        reasoning_subgraph_triples=[
            GraphTriple(source="Seed", relation="r", target="Answer")
        ],
        found_paths=1,
        missing_paths=1,
        construction=EvidenceSubgraphConstruction(
            strategy="pcst",
            edge_cost_strategy="constant",
            edge_cost_lambda=1.0,
            input_candidate_count=2,
            valid_candidate_count=2,
            selected_candidate_count=1,
            selected_candidate_ranks=[1],
            collected_prize=2.0,
            selected_node_count=2,
            selected_triple_count=1,
            total_edge_cost=1.0,
            objective=1.0,
            valid_seed_count=1,
            construction_time_ms=4.0,
        ),
    )
    batch = ExtractedReasoningPathsBatch(
        dataset_id="WebQSP",
        evaluation_run_name="7_eval",
        items=[
            ReasoningPathsForPrediction(
                instance_index=0,
                prediction=prediction,
                extracted_paths=extracted,
            )
        ],
    )
    evaluation_result = GnnAnswerRetrieverEvaluationResult(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        model_run_directory=tmp_path / "models" / "3_model",
        model_run_name="3_model",
        model_run_number=3,
        evaluation_run_directory=evaluation_directory,
        evaluation_run_name="7_eval",
        evaluation_run_number=7,
        evaluated_instances=1,
        hits_at_1=1.0,
        hits_at_1_count=1,
        average_candidate_count=2.0,
        missing_gold_in_graph_count=0,
        predictions_path=predictions_path,
        evaluation_config_path=config_path,
    )
    configuration = BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        main_llm_model="unused",
        subgraph_construction_algorithm="pcst",
        pcst_edge_cost_strategy="constant",
        pcst_edge_cost=1.0,
        context_construction_strategy="structured_triples",
    )

    saved = EvidenceRunStorageService().save(
        paths_batch=batch,
        evaluation_result=evaluation_result,
        configuration=configuration,
        evidence_root=tmp_path / "evidence",
        run_name="experiment-1a",
    )

    assert saved.evidence_run_name == "1_experiment-1a"
    assert saved.subgraph_algorithm == "pcst"
    assert saved.evidence_metrics["average_candidate_evidence_coverage"] == 0.5
    assert saved.evidence_metrics["candidate_reduction_percentage"] == 50.0
    assert saved.evidence_metrics["reasoning_context_gold_coverage"] == 1.0
    assert saved.evidence_metrics["reasoning_context_full_gold_coverage_rate"] == 1.0
    assert saved.evidence_metrics["average_objective"] == 1.0

    row = json.loads(saved.evidence_subgraphs_path.read_text(encoding="utf-8"))
    assert row["answer_candidates"] == ["The Answer", "Distractor"]
    assert row["analytics"]["context_visible_gold_answers"] == ["answer"]
    persisted_config = json.loads(saved.evidence_config_path.read_text(encoding="utf-8"))
    assert persisted_config["gnn_architecture"] == "hgt"
    assert persisted_config["evaluation_config"]["evaluation_run_name"] == "7_eval"
    assert persisted_config["evidence"]["algorithm"] == "pcst"


def test_empty_evidence_metrics_are_zero() -> None:
    metrics = EvidenceRunStorageService.build_metrics([])

    assert metrics["empty_subgraph_count"] == 0
    assert metrics["reasoning_context_gold_coverage"] == 0.0
    assert metrics["reasoning_context_full_gold_coverage_rate"] == 0.0


def test_evidence_wandb_logs_summary_metrics_and_pcst_title(tmp_path) -> None:
    evaluation_config_path = tmp_path / "evaluation_config.json"
    evaluation_config_path.write_text("{}", encoding="utf-8")
    evidence_config_path = tmp_path / "evidence_config.json"
    evidence_rows_path = tmp_path / "evidence_subgraphs.jsonl"
    evidence_metrics_path = tmp_path / "evidence_metrics.json"
    evidence_rows_path.write_text("", encoding="utf-8")
    evidence_metrics_path.write_text("{}", encoding="utf-8")
    evidence_config_path.write_text(
        json.dumps(
            {
                "evaluation_config": {
                    "full_config_path": str(evaluation_config_path),
                },
                "evidence": {
                    "algorithm": "pcst",
                    "pcst": {"edge_cost_strategy": "constant"},
                },
            }
        ),
        encoding="utf-8",
    )
    result = SavedEvidenceSubgraphRun(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        evaluation_run_name="7_eval",
        evidence_run_directory=tmp_path,
        evidence_run_name="1_pcst",
        evidence_run_number=1,
        evaluated_instances=10,
        subgraph_algorithm="pcst",
        evidence_configuration={
            "algorithm": "pcst",
            "pcst": {"edge_cost_strategy": "constant"},
        },
        evidence_metrics={
            "average_subgraph_triples": 8.0,
            "average_candidate_evidence_coverage": 0.75,
            "candidate_reduction_percentage": 25.0,
            "empty_subgraph_rate": 0.1,
            "reasoning_context_gold_coverage": 0.8,
            "reasoning_context_full_gold_coverage_rate": 0.6,
        },
        evidence_config_path=evidence_config_path,
        evidence_subgraphs_path=evidence_rows_path,
        evidence_metrics_path=evidence_metrics_path,
    )
    coordinator = MagicMock()
    coordinator.metadata.status = "logged"
    coordinator.metadata.run_id = "wandb-id"
    coordinator.metadata.run_url = "https://wandb.test/run"
    coordinator.metadata.error_message = None

    logged = LogEvidenceToWandbStep(coordinator=coordinator).execute_default(
        StepContext(result=result)
    )

    coordinator.update_inference_run_name.assert_called_once_with(
        evidence_algorithm="pcst",
        model_id=None,
    )
    coordinator.update_tags.assert_called_once_with(["pcst", "pcst-constant"])
    payload = coordinator.log_aggregate_metrics.call_args.args[0]
    assert payload["Summary_Plots/evidence_average_subgraph_triples"] == 8.0
    assert payload["Summary_Plots/reasoning_context_gold_coverage"] == 0.8
    assert payload["Run_Summary/evidence_candidate_reduction_percentage"] == 25.0
    assert payload["Run_Summary/reasoning_context_gold_coverage"] == 0.8
    assert payload["Run_Summary/reasoning_context_full_gold_coverage"] == 0.6
    assert logged.wandb_status == "logged"
