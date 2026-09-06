"""One-time W&B backfill for retriever nDCG metrics.

This maintenance script reads existing local retriever predictions, recomputes
nDCG with the production retrieval-results service, and resumes the exact W&B
run recorded in each evaluation configuration. It never loads a GNN and does
not modify local retriever artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


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
from pipeline.preparation.helpers.dataset_definitions import (  # noqa: E402
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)


NDCG_KEYS = (
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_candidate_limit",
)


def _load_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _has_persisted_ndcg(run_directory: Path) -> bool:
    metrics_path = (
        run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_METRICS_FILENAME
    )
    if not metrics_path.exists():
        return False
    try:
        metrics = _load_object(metrics_path)
    except ValueError:
        return False
    return all(isinstance(metrics.get(key), int | float) for key in NDCG_KEYS)


def _wandb_lineage(config_path: Path) -> dict:
    tracking = _load_object(config_path).get("wandb")
    if not isinstance(tracking, dict) or not tracking.get("run_id"):
        raise ValueError(
            f"Retriever config has no resumable W&B run ID: {config_path}"
        )
    return tracking


def _ndcg_payload(result) -> dict[str, float]:
    return {
        "Summary_Plots/ranking_ndcg_at_1": result.ndcg_at_1,
        "Summary_Plots/ranking_ndcg_at_5": result.ndcg_at_5,
        "Summary_Plots/ranking_ndcg_at_10": result.ndcg_at_10,
        "Summary_Plots/ranking_ndcg_at_candidate_limit": (
            result.ndcg_at_candidate_limit
        ),
        "Run_Summary/ranking_ndcg_at_10": result.ndcg_at_10,
    }


def _selected_run_directories(
    *,
    evaluation_root: Path,
    run_numbers: list[int] | None,
    run_names: list[str] | None,
    all_missing_local_ndcg: bool,
) -> list[Path]:
    service = GnnRetrieverResultsService()
    if all_missing_local_ndcg:
        if not evaluation_root.exists():
            raise ValueError(f"No retriever runs exist under {evaluation_root}.")
        return [
            path
            for path in sorted(evaluation_root.iterdir(), key=lambda item: item.name)
            if path.is_dir() and not _has_persisted_ndcg(path)
        ]
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
    """Print or upload nDCG for selected local retriever runs."""
    service = GnnRetrieverResultsService()
    failures = 0
    if not run_directories:
        print("No matching retriever runs require an nDCG backfill.")
        return 0

    for run_directory in run_directories:
        config_path = (
            run_directory / GNN_ANSWER_RETRIEVER_EVALUATION_CONFIG_FILENAME
        )
        try:
            lineage = _wandb_lineage(config_path)
            result = service.load_run(
                evaluation_root=run_directory.parent,
                dataset_id=dataset_id,
                run_name=run_directory.name,
                run_number=None,
            )
            payload = _ndcg_payload(result)
            print(
                f"{'APPLY' if apply else 'DRY RUN'} {run_directory.name}: "
                f"wandb_run_id={lineage['run_id']} "
                f"ndcg@1={result.ndcg_at_1:.6f} "
                f"ndcg@5={result.ndcg_at_5:.6f} "
                f"ndcg@10={result.ndcg_at_10:.6f} "
                f"ndcg@candidate_limit={result.ndcg_at_candidate_limit:.6f}"
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
            coordinator.log(payload)
            metadata = coordinator.metadata
            coordinator.finish()
            if metadata.status != "logged":
                raise RuntimeError(
                    metadata.error_message or "Could not upload nDCG metrics."
                )
            print(f"UPDATED {run_directory.name}: {metadata.run_url or metadata.run_id}")
        except Exception as error:
            failures += 1
            print(f"FAILED {run_directory.name}: {error}", file=sys.stderr)

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill of retriever nDCG metrics into existing W&B runs. "
            "Without --apply, no remote state is changed."
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
        "--all-missing-local-ndcg",
        action="store_true",
        help="Patch every local retriever run whose metrics file lacks nDCG.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Resume and update W&B runs. Omit for a local dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = DATASET_LOADERS.get(args.dataset)
    if loader is None:
        print(f"Unsupported dataset: {args.dataset}", file=sys.stderr)
        return 2
    evaluation_root = loader.cache_root / "evaluations"
    try:
        run_directories = _selected_run_directories(
            evaluation_root=evaluation_root,
            run_numbers=args.retriever_run_number,
            run_names=args.retriever_run_name,
            all_missing_local_ndcg=args.all_missing_local_ndcg,
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
