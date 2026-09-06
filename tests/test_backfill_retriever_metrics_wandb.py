"""Tests for the one-time complete retriever-metric backfill script."""

from types import SimpleNamespace

from scripts.one_time.backfill_retriever_metrics_wandb import (
    _retrieval_metrics_payload,
    _selected_run_directories,
    _wandb_payload,
    build_parser,
)


def test_parser_accepts_inclusive_retriever_run_range() -> None:
    args = build_parser().parse_args(
        ["--retriever-run-range", "98", "114", "--apply"]
    )

    assert args.retriever_run_range == [98, 114]
    assert args.apply is True


def test_range_selection_resolves_every_run_inclusively(tmp_path) -> None:
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    for number in range(3, 6):
        (evaluation_root / f"{number}_run").mkdir()

    selected = _selected_run_directories(
        evaluation_root=evaluation_root,
        run_numbers=None,
        run_names=None,
        run_range=[3, 5],
        all_missing_local_metrics=False,
    )

    assert [path.name for path in selected] == ["3_run", "4_run", "5_run"]


def test_payload_contains_complete_summary_and_curated_run_metrics() -> None:
    result = SimpleNamespace(
        dataset_id="WebQSP",
        model_run_name="1_model",
        model_run_number=1,
        evaluation_run_name="2_eval",
        evaluation_run_number=2,
        evaluated_instances=10,
        conditioned_evaluated_instances=10,
        hits_at_1=0.2,
        hits_at_1_count=2,
        hits_at_5=0.5,
        hits_at_5_count=5,
        hits_at_10=0.7,
        hits_at_10_count=7,
        hits_at_candidate_limit=0.8,
        hits_at_candidate_limit_count=8,
        ndcg_at_1=0.2,
        ndcg_at_5=0.4,
        ndcg_at_10=0.5,
        ndcg_at_candidate_limit=0.6,
        average_candidate_count=10.0,
        missing_gold_in_graph_count=2,
        skipped_missing_gold_in_graph_count=2,
        retrieval_gold_coverage=0.65,
        retrieval_full_gold_coverage_count=4,
        retrieval_full_gold_coverage_rate=0.4,
        retrieved_gold_answer_count=9,
    )

    metrics = _retrieval_metrics_payload(result, candidate_limit=10)
    wandb = _wandb_payload(metrics)

    assert metrics["retrieval_gold_coverage"] == 0.65
    assert metrics["skipped_missing_gold_in_graph_count"] == 2
    assert wandb["Summary_Plots/retrieval_gold_coverage"] == 0.65
    assert wandb["Summary_Plots/retrieval_hits_at_5_count"] == 5
    assert wandb["Run_Summary/retrieval_gold_coverage"] == 0.65
    assert wandb["Run_Summary/retrieval_full_gold_coverage"] == 0.4
    assert "Summary_Plots/retrieval_hits_at_5" not in wandb
    assert "Summary_Plots/ranking_ndcg_at_10" not in wandb
