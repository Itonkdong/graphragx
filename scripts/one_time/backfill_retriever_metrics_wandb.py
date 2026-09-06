"""One-time local and W&B backfill for complete retriever metrics.

The script recomputes metrics from each run's persisted predictions, updates
``retrieval_metrics.json``, and resumes the exact W&B run recorded in the
evaluation configuration. It never loads GNN weights or executes evaluation.
Without ``--apply`` it performs a read-only dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.constants import (  # noqa: E402
    GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME,
)
from pipeline.evaluation.services.gnn_retriever_results import (  # noqa: E402
    GnnRetrieverResultsService,
)
from pipeline.evaluation.services.wandb_experiment import (  # noqa: E402
    WandbExperimentCoordinator,
)
from pipeline.evaluation.services.wandb_final_results import (  # noqa: E402
    WandbFinalResultsLoggingService,
)
from pipeline.preparation.helpers.dataset_definitions import (  # noqa: E402
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)


BACKFILL_KEYS = (
    "conditioned_evaluated_instances",
    "retrieval_gold_coverage",
    "retrieval_full_gold_coverage_count",
    "retrieval_full_gold_coverage_rate",
    "retrieved_gold_answer_count",
    "skipped_missing_gold_in_graph_count",
)
WANDB_BACKFILL_KEYS = {
    "Summary_Plots/retrieval_hits_at_1_count",
    "Summary_Plots/retrieval_hits_at_5_count",
    "Summary_Plots/retrieval_hits_at_10_count",
    "Summary_Plots/retrieval_hits_at_candidate_limit_count",
    "Summary_Plots/retrieval_skipped_missing_gold_in_graph_count",
    "Summary_Plots/conditioned_evaluated_instances",
    "Summary_Plots/retrieval_gold_coverage",
    "Summary_Plots/retrieval_full_gold_coverage_count",
    "Summary_Plots/retrieval_full_gold_coverage_rate",
    "Summary_Plots/retrieved_gold_answer_count",
    "Run_Summary/retrieval_gold_coverage",
    "Run_Summary/retrieval_full_gold_coverage",
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _has_complete_local_metrics(run_directory: Path) -> bool:
    metrics_path = (
        run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME
    )
    if not metrics_path.exists():
        return False
    try:
        metrics = _load_object(metrics_path)
    except ValueError:
        return False
    return all(isinstance(metrics.get(key), int | float) for key in BACKFILL_KEYS)


def _wandb_lineage(config_path: Path) -> dict[str, Any]:
    tracking = _load_object(config_path).get("wandb")
    if not isinstance(tracking, dict) or not tracking.get("run_id"):
        raise ValueError(
            f"Retriever config has no resumable W&B run ID: {config_path}"
        )
    return tracking


def _candidate_limit(config_path: Path) -> int:
    evaluation = _load_object(config_path).get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Retriever config has no evaluation block: {config_path}")
    try:
        candidate_limit = int(evaluation["candidate_limit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Retriever config has an invalid candidate limit: {config_path}"
        ) from error
    if candidate_limit <= 0:
        raise ValueError(
            f"Retriever candidate limit must be positive: {config_path}"
        )
    return candidate_limit


def _retrieval_metrics_payload(result, *, candidate_limit: int) -> dict[str, Any]:
    return {
        "dataset_id": result.dataset_id,
        "model_run_name": result.model_run_name,
        "model_run_number": result.model_run_number,
        "evaluation_run_name": result.evaluation_run_name,
        "evaluation_run_number": result.evaluation_run_number,
        "evaluated_instances": result.evaluated_instances,
        "conditioned_evaluated_instances": result.conditioned_evaluated_instances,
        "hits_at_1": result.hits_at_1,
        "hits_at_1_count": result.hits_at_1_count,
        "hits_at_5": result.hits_at_5,
        "hits_at_5_count": result.hits_at_5_count,
        "hits_at_10": result.hits_at_10,
        "hits_at_10_count": result.hits_at_10_count,
        "hits_at_candidate_limit": result.hits_at_candidate_limit,
        "hits_at_candidate_limit_count": result.hits_at_candidate_limit_count,
        "ndcg_at_1": result.ndcg_at_1,
        "ndcg_at_5": result.ndcg_at_5,
        "ndcg_at_10": result.ndcg_at_10,
        "ndcg_at_candidate_limit": result.ndcg_at_candidate_limit,
        "candidate_limit": candidate_limit,
        "average_candidate_count": result.average_candidate_count,
        "missing_gold_in_graph_count": result.missing_gold_in_graph_count,
        "skipped_missing_gold_in_graph_count": (
            result.skipped_missing_gold_in_graph_count
        ),
        "retrieval_gold_coverage": result.retrieval_gold_coverage,
        "retrieval_full_gold_coverage_count": (
            result.retrieval_full_gold_coverage_count
        ),
        "retrieval_full_gold_coverage_rate": (
            result.retrieval_full_gold_coverage_rate
        ),
        "retrieved_gold_answer_count": result.retrieved_gold_answer_count,
    }


def _wandb_payload(metrics: dict[str, Any]) -> dict[str, float | int]:
    scalar_metrics = WandbFinalResultsLoggingService.build_scalar_metrics(
        retrieval_metrics=metrics,
        reasoning_metrics={},
    )
    payload = WandbFinalResultsLoggingService.build_summary_plot_metrics(
        scalar_metrics
    )
    payload.update(
        WandbFinalResultsLoggingService.build_run_summary_plot_metrics(
            scalar_metrics=scalar_metrics,
            wandb_config={},
        )
    )
    # Do not re-log existing Hits/nDCG rate keys at a later W&B step. Repeated
    # points would distort their history axes. This patch uploads only metrics
    # absent from the completed experiment runs.
    return {
        key: value
        for key, value in payload.items()
        if key in WANDB_BACKFILL_KEYS
    }


def _selected_run_directories(
    *,
    evaluation_root: Path,
    run_numbers: list[int] | None,
    run_names: list[str] | None,
    run_range: list[int] | None,
    all_missing_local_metrics: bool,
) -> list[Path]:
    service = GnnRetrieverResultsService()
    if all_missing_local_metrics:
        if not evaluation_root.exists():
            raise ValueError(f"No retriever runs exist under {evaluation_root}.")
        return [
            path
            for path in sorted(evaluation_root.iterdir(), key=lambda item: item.name)
            if path.is_dir() and not _has_complete_local_metrics(path)
        ]
    if run_range is not None:
        start, end = run_range
        if start < 1 or end < start:
            raise ValueError("Retriever run range must satisfy 1 <= START <= END.")
        run_numbers = list(range(start, end + 1))
    if run_numbers:
        return [
            service.resolve_run_directory(
                evaluation_root=evaluation_root,
                run_name=None,
                run_number=run_number,
            )
            for run_number in run_numbers
        ]
    return [
        service.resolve_run_directory(
            evaluation_root=evaluation_root,
            run_name=run_name,
            run_number=None,
        )
        for run_name in run_names or []
    ]


def backfill(
    *,
    dataset_id: str,
    run_directories: list[Path],
    apply: bool,
) -> int:
    """Dry-run or persist complete metrics for selected retriever runs."""
    service = GnnRetrieverResultsService()
    failures = 0
    if not run_directories:
        print("No matching retriever runs require a metric backfill.")
        return 0

    for run_directory in run_directories:
        config_path = (
            run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME
        )
        metrics_path = (
            run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME
        )
        coordinator: WandbExperimentCoordinator | None = None
        try:
            lineage = _wandb_lineage(config_path)
            result = service.load_run(
                evaluation_root=run_directory.parent,
                dataset_id=dataset_id,
                run_name=run_directory.name,
                run_number=None,
            )
            metrics = _retrieval_metrics_payload(
                result,
                candidate_limit=_candidate_limit(config_path),
            )
            wandb_payload = _wandb_payload(metrics)
            print(
                f"{'APPLY' if apply else 'DRY RUN'} {run_directory.name}: "
                f"wandb_run_id={lineage['run_id']} "
                f"gold_coverage={result.retrieval_gold_coverage:.6f} "
                f"full_gold_coverage="
                f"{result.retrieval_full_gold_coverage_rate:.6f} "
                f"retrieved_gold_answers={result.retrieved_gold_answer_count} "
                f"summary_metrics={len(wandb_payload)}"
            )
            if not apply:
                continue

            coordinator = WandbExperimentCoordinator(
                mode="online",
                enabled=True,
                resume_from_lineage=True,
            )
            metadata = coordinator.ensure_run(source_config_path=config_path)
            if metadata.status != "logged":
                raise RuntimeError(
                    metadata.error_message or "Could not resume the W&B run."
                )
            coordinator.log(wandb_payload)
            metadata = coordinator.metadata
            if metadata.status != "logged":
                raise RuntimeError(
                    metadata.error_message or "Could not upload retriever metrics."
                )
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"UPDATED {run_directory.name}: "
                f"{metadata.run_url or metadata.run_id} local={metrics_path}"
            )
        except Exception as error:
            failures += 1
            print(f"FAILED {run_directory.name}: {error}", file=sys.stderr)
        finally:
            if coordinator is not None:
                coordinator.finish()

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill of complete local retriever metrics and missing "
            "retriever metrics in existing W&B runs. Without --apply, nothing "
            "is changed."
        )
    )
    parser.add_argument("--dataset", default=WEBQSP_DATASET_ID)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument(
        "--retriever-run-number",
        type=int,
        action="append",
        help="Retriever run number to patch; repeat for multiple runs.",
    )
    selectors.add_argument(
        "--retriever-run-name",
        action="append",
        help="Retriever run name/label to patch; repeat for multiple runs.",
    )
    selectors.add_argument(
        "--retriever-run-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive retriever run-number range to patch.",
    )
    selectors.add_argument(
        "--all-missing-local-metrics",
        action="store_true",
        help="Patch every local run missing any newly introduced metric.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update local metrics and existing W&B runs. Omit for a dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = DATASET_LOADERS.get(args.dataset)
    if loader is None:
        print(f"Unsupported dataset: {args.dataset}", file=sys.stderr)
        return 2
    try:
        run_directories = _selected_run_directories(
            evaluation_root=loader.cache_root / "evaluations",
            run_numbers=args.retriever_run_number,
            run_names=args.retriever_run_name,
            run_range=args.retriever_run_range,
            all_missing_local_metrics=args.all_missing_local_metrics,
        )
    except Exception as error:
        print(f"Could not select retriever runs: {error}", file=sys.stderr)
        return 2
    return backfill(
        dataset_id=args.dataset,
        run_directories=run_directories,
        apply=args.apply,
    )


if __name__ == "__main__":
    raise SystemExit(main())
