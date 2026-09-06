#!/usr/bin/env python3
"""Build thesis tables and figures for the GNN architecture experiment.

The script is deliberately read-only with respect to W&B. It resolves the runs
declared in ``experiments/experiment_0_gnn_architectures.toml`` by their saved
evaluation lineage, downloads scalar summaries, and writes auditable local
outputs. No W&B run, panel, summary, or configuration is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments/experiment_0_gnn_architectures.toml"
DEFAULT_FIGURES = PROJECT_ROOT / "metadata/figures/architecture_retrieval"
DEFAULT_TABLES = PROJECT_ROOT / "metadata/tables/architecture_retrieval"
DEFAULT_RESULTS_METADATA = (
    PROJECT_ROOT / "metadata/results_metadata/architecture_retrieval"
)

ARCHITECTURE_ORDER = (
    "graphsage",
    "aa-graphsage",
    "rgcn",
    "hgt",
    "rearev",
    "nbfnet",
)
ARCHITECTURE_LABELS = {
    "graphsage": "GraphSAGE",
    "aa-graphsage": "Advance GraphSAGE",
    "rgcn": "R-GCN",
    "hgt": "HGT",
    "rearev": "ReaRev",
    "nbfnet": "NBFNet",
}

METRICS = {
    "hits_at_1": "Summary_Plots/retrieval_hits_at_1",
    "hits_at_5": "Summary_Plots/retrieval_hits_at_5",
    "hits_at_10": "Summary_Plots/retrieval_hits_at_10",
    "hits_at_candidate_limit": "Summary_Plots/retrieval_hits_at_candidate_limit",
    "ndcg_at_1": "Summary_Plots/ranking_ndcg_at_1",
    "ndcg_at_5": "Summary_Plots/ranking_ndcg_at_5",
    "ndcg_at_10": "Summary_Plots/ranking_ndcg_at_10",
    "ndcg_at_candidate_limit": "Summary_Plots/ranking_ndcg_at_candidate_limit",
    "retrieval_gold_coverage": "Summary_Plots/retrieval_gold_coverage",
    "retrieval_full_gold_coverage": (
        "Summary_Plots/retrieval_full_gold_coverage_rate"
    ),
    "average_candidate_count": "Summary_Plots/retrieval_average_candidate_count",
    "evaluated_instances": "Summary_Plots/retrieval_evaluated_instances",
}


@dataclass(frozen=True)
class ExpectedRun:
    experiment_id: str
    evaluation_name: str
    architecture: str
    seed: int


@dataclass(frozen=True)
class RunRecord:
    expected: ExpectedRun
    wandb_run_id: str
    wandb_run_name: str
    wandb_run_url: str
    evaluation_lineage_name: str
    wandb_group: str
    state: str
    metrics: dict[str, float]


def _argument_value(arguments: list[str], option: str) -> str:
    try:
        index = arguments.index(option)
        return arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Run arguments are missing {option}.") from error


def load_expected_runs(path: Path) -> list[ExpectedRun]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    expected: list[ExpectedRun] = []
    for run in payload.get("runs", []):
        arguments = [str(value) for value in run.get("args", [])]
        expected.append(
            ExpectedRun(
                experiment_id=str(run["id"]),
                evaluation_name=_argument_value(arguments, "--evaluation-run-name"),
                architecture=_argument_value(arguments, "--gnn-architecture"),
                seed=int(_argument_value(arguments, "--seed")),
            )
        )
    if not expected:
        raise ValueError(f"No runs were found in {path}.")
    return expected


def _lineage_name(config: dict[str, Any]) -> str:
    runs = config.get("runs")
    evaluation = runs.get("evaluation") if isinstance(runs, dict) else None
    name = evaluation.get("name") if isinstance(evaluation, dict) else None
    return str(name or "")


def _matches_lineage(actual: str, expected: str) -> bool:
    return actual == expected or actual.endswith(f"_{expected}")


def _numeric_summary(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def fetch_wandb_records(
    *,
    entity: str,
    project: str,
    group_prefix: str,
    expected_runs: list[ExpectedRun],
) -> list[RunRecord]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("The W&B package is required to fetch experiment runs.") from error

    api = wandb.Api()
    allowed_groups = {
        group_prefix,
        f"{group_prefix}_seed_winners",
        f"{group_prefix}_winners",
    }
    project_runs = list(api.runs(f"{entity}/{project}", per_page=300))
    indexed: list[tuple[Any, str, dict[str, Any]]] = []
    for run in project_runs:
        if str(run.group or "") not in allowed_groups:
            continue
        config = dict(run.config or {})
        indexed.append((run, _lineage_name(config), config))

    records: list[RunRecord] = []
    for expected in expected_runs:
        candidates: list[RunRecord] = []
        rejected: list[str] = []
        for run, lineage_name, config in indexed:
            if not _matches_lineage(lineage_name, expected.evaluation_name):
                continue
            summary = dict(run.summary or {})
            values: dict[str, float] = {}
            missing: list[str] = []
            for metric_name, summary_key in METRICS.items():
                value = _numeric_summary(summary, summary_key)
                if value is None:
                    missing.append(summary_key)
                else:
                    values[metric_name] = value
            if missing:
                rejected.append(f"{run.id} (missing {', '.join(missing)})")
                continue

            model_config = (config.get("configs") or {}).get("model") or {}
            saved_architecture = str(model_config.get("gnn_architecture") or "")
            training = model_config.get("training") or {}
            saved_seed = training.get("random_seed")
            if saved_architecture != expected.architecture or int(saved_seed) != expected.seed:
                rejected.append(
                    f"{run.id} (saved architecture/seed={saved_architecture}/{saved_seed})"
                )
                continue
            candidates.append(
                RunRecord(
                    expected=expected,
                    wandb_run_id=str(run.id),
                    wandb_run_name=str(run.name),
                    wandb_run_url=str(run.url or ""),
                    evaluation_lineage_name=lineage_name,
                    wandb_group=str(run.group or ""),
                    state=str(run.state or ""),
                    metrics=values,
                )
            )

        if len(candidates) != 1:
            detail = "; ".join(rejected) if rejected else "none"
            raise RuntimeError(
                f"Expected exactly one complete W&B run for {expected.evaluation_name}, "
                f"found {len(candidates)}. Rejected matches: {detail}"
            )
        records.append(candidates[0])
    return records


def aggregate_records(records: list[RunRecord]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for architecture in ARCHITECTURE_ORDER:
        selected = [item for item in records if item.expected.architecture == architecture]
        if not selected:
            continue
        row: dict[str, float | str | int] = {
            "architecture": architecture,
            "architecture_label": ARCHITECTURE_LABELS[architecture],
            "seed_count": len(selected),
        }
        for metric_name in METRICS:
            values = [item.metrics[metric_name] for item in selected]
            row[f"{metric_name}_mean"] = statistics.fmean(values)
            row[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_raw_csv(path: Path, records: list[RunRecord]) -> None:
    fields = [
        "experiment_id",
        "architecture",
        "architecture_label",
        "seed",
        "wandb_run_id",
        "wandb_run_name",
        "wandb_run_url",
        "evaluation_lineage_name",
        "wandb_group",
        "state",
        *METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "experiment_id": record.expected.experiment_id,
                    "architecture": record.expected.architecture,
                    "architecture_label": ARCHITECTURE_LABELS[
                        record.expected.architecture
                    ],
                    "seed": record.expected.seed,
                    "wandb_run_id": record.wandb_run_id,
                    "wandb_run_name": record.wandb_run_name,
                    "wandb_run_url": record.wandb_run_url,
                    "evaluation_lineage_name": record.evaluation_lineage_name,
                    "wandb_group": record.wandb_group,
                    "state": record.state,
                    **record.metrics,
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fields = ["architecture", "architecture_label", "seed_count"]
    for metric_name in METRICS:
        fields.extend((f"{metric_name}_mean", f"{metric_name}_std"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_value(row: dict[str, float | str | int], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"${mean:.3f} \\pm {std:.3f}$"


def write_latex_tables(output_dir: Path, rows: list[dict[str, float | str | int]]) -> None:
    primary_metrics = (
        "hits_at_1",
        "hits_at_10",
        "ndcg_at_10",
        "retrieval_gold_coverage",
        "retrieval_full_gold_coverage",
    )
    best_primary = {
        metric: max(float(row[f"{metric}_mean"]) for row in rows)
        for metric in primary_metrics
    }

    def primary_value(row: dict[str, float | str | int], metric: str) -> str:
        value = _latex_value(row, metric)
        if math.isclose(
            float(row[f"{metric}_mean"]),
            best_primary[metric],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return rf"{{\boldmath {value}}}"
        return value

    primary_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (
            r"Архитектура & Hits@1 & Hits@10 & nDCG@10 & "
            r"RetrievalGoldCoverage & RetrievalFullGoldCoverage \\"
        ),
        r"\midrule",
    ]
    ranking_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Архитектура & Hits@1 & Hits@5 & Hits@10 & nDCG@10 \\",
        r"\midrule",
    ]
    coverage_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Архитектура & Hits@CandidateLimit & RetrievalGoldCoverage & RetrievalFullGoldCoverage \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["architecture_label"])
        primary_lines.append(
            " & ".join(
                [
                    label,
                    *[primary_value(row, metric) for metric in primary_metrics],
                ]
            )
            + r" \\"
        )
        ranking_lines.append(
            " & ".join(
                [
                    label,
                    _latex_value(row, "hits_at_1"),
                    _latex_value(row, "hits_at_5"),
                    _latex_value(row, "hits_at_10"),
                    _latex_value(row, "ndcg_at_10"),
                ]
            )
            + r" \\"
        )
        coverage_lines.append(
            " & ".join(
                [
                    label,
                    _latex_value(row, "hits_at_candidate_limit"),
                    _latex_value(row, "retrieval_gold_coverage"),
                    _latex_value(row, "retrieval_full_gold_coverage"),
                ]
            )
            + r" \\"
        )
    primary_lines.extend((r"\bottomrule", r"\end{tabular}"))
    ranking_lines.extend((r"\bottomrule", r"\end{tabular}"))
    coverage_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output_dir / "architecture_primary_table.tex").write_text(
        "\n".join(primary_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "architecture_ranking_table.tex").write_text(
        "\n".join(ranking_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "architecture_coverage_table.tex").write_text(
        "\n".join(coverage_lines) + "\n", encoding="utf-8"
    )


def _plot_grouped_bars(
    *,
    output_dir: Path,
    rows: list[dict[str, float | str | int]],
    filename: str,
    series: Iterable[tuple[str, str]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Figure generation requires Matplotlib (for example: "
            "uv run --with matplotlib python scripts/results/architecture_retrieval_results.py)."
        ) from error

    series = list(series)
    x = np.arange(len(rows), dtype=float)
    width = 0.8 / len(series)
    fig, axis = plt.subplots(figsize=(11.2 if len(series) == 5 else 10.4, 5.2))
    metric_colors = {
        "hits_at_1": "#8064A2",
        "hits_at_10": "#4472C4",
        "ndcg_at_10": "#70AD47",
        "retrieval_gold_coverage": "#ED7D31",
        "retrieval_full_gold_coverage": "#A5A5A5",
    }
    for index, (metric, label) in enumerate(series):
        means = [float(row[f"{metric}_mean"]) for row in rows]
        errors = [float(row[f"{metric}_std"]) for row in rows]
        positions = x - 0.4 + width / 2 + index * width
        axis.bar(
            positions,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=metric_colors.get(metric, "#4472C4"),
            edgecolor="black",
            linewidth=0.35,
        )
    axis.set_xticks(x, [str(row["architecture_label"]) for row in rows])
    axis.set_ylabel("Средна вредност")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        ncol=min(2, len(series)),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_figures(output_dir: Path, rows: list[dict[str, float | str | int]]) -> None:
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_primary_metrics",
        series=(
            ("hits_at_10", "Hits@10"),
            ("ndcg_at_10", "nDCG@10"),
            ("retrieval_gold_coverage", "RetrievalGoldCoverage"),
            (
                "retrieval_full_gold_coverage",
                "RetrievalFullGoldCoverage",
            ),
        ),
    )
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_primary_metrics_hits1_no_full_coverage",
        series=(
            ("hits_at_1", "Hits@1"),
            ("hits_at_10", "Hits@10"),
            ("ndcg_at_10", "nDCG@10"),
            ("retrieval_gold_coverage", "RetrievalGoldCoverage"),
        ),
    )
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_primary_metrics_hits1_all_metrics",
        series=(
            ("hits_at_1", "Hits@1"),
            ("hits_at_10", "Hits@10"),
            ("ndcg_at_10", "nDCG@10"),
            ("retrieval_gold_coverage", "RetrievalGoldCoverage"),
            (
                "retrieval_full_gold_coverage",
                "RetrievalFullGoldCoverage",
            ),
        ),
    )
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_hits",
        series=(
            ("hits_at_1", "Hits@1"),
            ("hits_at_5", "Hits@5"),
            ("hits_at_10", "Hits@10"),
            ("hits_at_candidate_limit", "Hits@CandidateLimit"),
        ),
    )
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_ndcg",
        series=(
            ("ndcg_at_1", "nDCG@1"),
            ("ndcg_at_5", "nDCG@5"),
            ("ndcg_at_10", "nDCG@10"),
            ("ndcg_at_candidate_limit", "nDCG@CandidateLimit"),
        ),
    )
    _plot_grouped_bars(
        output_dir=output_dir,
        rows=rows,
        filename="architecture_gold_coverage",
        series=(
            ("retrieval_gold_coverage", "RetrievalGoldCoverage"),
            (
                "retrieval_full_gold_coverage",
                "RetrievalFullGoldCoverage",
            ),
        ),
    )


def write_manifest(
    path: Path,
    *,
    entity: str,
    project: str,
    group_prefix: str,
    experiment_path: Path,
    records: list[RunRecord],
) -> None:
    payload = {
        "wandb_project": f"{entity}/{project}",
        "wandb_groups": [
            group_prefix,
            f"{group_prefix}_seed_winners",
            f"{group_prefix}_winners",
        ],
        "experiment_file": str(experiment_path.resolve()),
        "selection_rule": (
            "Unique complete run whose configs.runs.evaluation.name equals or ends "
            "with the experiment-defined evaluation run name, with matching saved "
            "architecture and random seed."
        ),
        "metrics": METRICS,
        "runs": [
            {
                "experiment_id": record.expected.experiment_id,
                "architecture": record.expected.architecture,
                "seed": record.expected.seed,
                "wandb_run_id": record.wandb_run_id,
                "wandb_run_name": record.wandb_run_name,
                "wandb_run_url": record.wandb_run_url,
                "evaluation_lineage_name": record.evaluation_lineage_name,
                "wandb_group": record.wandb_group,
                "state": record.state,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="itonkdong-org")
    parser.add_argument("--project", default="graphragx")
    parser.add_argument(
        "--wandb-group-prefix",
        default="experiment0",
        help=(
            "Base W&B group. The selected union is BASE, "
            "BASE_seed_winners, and BASE_winners."
        ),
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument(
        "--results-metadata-dir",
        type=Path,
        default=DEFAULT_RESULTS_METADATA,
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write CSV, LaTeX, and provenance outputs without importing Matplotlib.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    expected = load_expected_runs(arguments.experiment)
    records = fetch_wandb_records(
        entity=arguments.entity,
        project=arguments.project,
        group_prefix=arguments.wandb_group_prefix,
        expected_runs=expected,
    )
    figures_dir = arguments.figures_dir.resolve()
    tables_dir = arguments.tables_dir.resolve()
    results_metadata_dir = arguments.results_metadata_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_metadata_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_records(records)
    write_raw_csv(results_metadata_dir / "architecture_runs.csv", records)
    write_summary_csv(results_metadata_dir / "architecture_summary.csv", summary)
    write_latex_tables(tables_dir, summary)
    write_manifest(
        results_metadata_dir / "provenance.json",
        entity=arguments.entity,
        project=arguments.project,
        group_prefix=arguments.wandb_group_prefix,
        experiment_path=arguments.experiment,
        records=records,
    )
    if not arguments.no_figures:
        write_figures(figures_dir, summary)
    print(
        "Wrote architecture retrieval results to "
        f"figures={figures_dir}, tables={tables_dir}, "
        f"metadata={results_metadata_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
