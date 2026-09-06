#!/usr/bin/env python3
"""Build thesis outputs for the evidence-subgraph experiment.

Runs are resolved by the evidence lineage names declared in
``experiments/experiment_1_evidence_subgraphs.toml``. W&B is read only.
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
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments/experiment_1_evidence_subgraphs.toml"
DEFAULT_FIGURES = PROJECT_ROOT / "metadata/figures/evidence_subgraphs"
DEFAULT_TABLES = PROJECT_ROOT / "metadata/tables/evidence_subgraphs"
DEFAULT_RESULTS_METADATA = PROJECT_ROOT / "metadata/results_metadata/evidence_subgraphs"

METRICS = {
    "average_subgraph_triples": "Summary_Plots/evidence_average_subgraph_triples",
    "average_distinct_nodes": "Summary_Plots/evidence_average_distinct_nodes",
    "context_candidate_coverage": (
        "Summary_Plots/evidence_average_candidate_evidence_coverage"
    ),
    "candidate_reduction_percentage": (
        "Summary_Plots/evidence_candidate_reduction_percentage"
    ),
    "context_gold_coverage": "Summary_Plots/reasoning_context_gold_coverage",
    "context_full_gold_coverage": (
        "Summary_Plots/reasoning_context_full_gold_coverage_rate"
    ),
}
PCST_METRICS = {
    "average_collected_prize": "Summary_Plots/evidence_average_collected_prize",
    "average_edge_cost": "Summary_Plots/evidence_average_edge_cost",
}


@dataclass(frozen=True)
class ExpectedRun:
    experiment_id: str
    evidence_name: str
    seed: int
    algorithm: str
    cost_strategy: str | None
    edge_cost_lambda: float | None

    @property
    def configuration_id(self) -> str:
        if self.algorithm == "shortest_path":
            return "shortest_path"
        return f"pcst_{self.cost_strategy}_{self.edge_cost_lambda:g}"

    @property
    def configuration_label(self) -> str:
        if self.algorithm == "shortest_path":
            return "Унија на најкратки патеки"
        strategy = "константна" if self.cost_strategy == "constant" else "семантичка"
        return rf"PCST, {strategy}, $\lambda={self.edge_cost_lambda:g}$"


@dataclass(frozen=True)
class RunRecord:
    expected: ExpectedRun
    wandb_run_id: str
    wandb_run_name: str
    wandb_run_url: str
    wandb_group: str
    evidence_lineage_name: str
    state: str
    metrics: dict[str, float]


def _argument_value(arguments: list[str], option: str) -> str:
    try:
        position = arguments.index(option)
        return arguments[position + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Run arguments are missing {option}.") from error


def _optional_argument_value(arguments: list[str], option: str) -> str | None:
    try:
        return _argument_value(arguments, option)
    except ValueError:
        return None


def load_expected_runs(path: Path) -> list[ExpectedRun]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    expected: list[ExpectedRun] = []
    for run in payload.get("runs", []):
        arguments = [str(value) for value in run.get("args", [])]
        algorithm = _argument_value(arguments, "--subgraph-algorithm")
        cost_strategy = _optional_argument_value(
            arguments, "--pcst-edge-cost-strategy"
        )
        lambda_value = _optional_argument_value(arguments, "--pcst-edge-cost")
        expected.append(
            ExpectedRun(
                experiment_id=str(run["id"]),
                evidence_name=str(
                    run.get("wandb_lineage_name")
                    or _argument_value(arguments, "--evidence-run-name")
                ),
                seed=int(_argument_value(arguments, "--seed")),
                algorithm=algorithm,
                cost_strategy=cost_strategy,
                edge_cost_lambda=float(lambda_value) if lambda_value else None,
            )
        )
    if not expected:
        raise ValueError(f"No runs were found in {path}.")
    return expected


def _lineage_name(config: dict[str, Any]) -> str:
    runs = config.get("runs")
    evidence = runs.get("evidence") if isinstance(runs, dict) else None
    name = evidence.get("name") if isinstance(evidence, dict) else None
    return str(name or "")


def _matches_lineage(actual: str, expected: str) -> bool:
    return actual == expected or actual.endswith(f"_{expected}")


def _numeric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def fetch_records(
    *,
    entity: str,
    project: str,
    expected_runs: list[ExpectedRun],
    base_group: str,
) -> list[RunRecord]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("The W&B package is required to fetch results.") from error

    authoritative_groups = {
        base_group,
        f"{base_group}_seed_winners",
        f"{base_group}_winners",
    }
    project_runs = list(wandb.Api().runs(f"{entity}/{project}", per_page=300))
    indexed: list[tuple[Any, str, dict[str, Any]]] = []
    for run in project_runs:
        if str(run.group or "") not in authoritative_groups:
            continue
        config = dict(run.config or {})
        indexed.append((run, _lineage_name(config), config))

    records: list[RunRecord] = []
    for expected in expected_runs:
        matches: list[RunRecord] = []
        rejected: list[str] = []
        for run, lineage_name, config in indexed:
            if not _matches_lineage(lineage_name, expected.evidence_name):
                continue
            evidence_config = (config.get("configs") or {}).get("evidence") or {}
            algorithm = str(evidence_config.get("algorithm") or "")
            pcst = evidence_config.get("pcst") or {}
            strategy = str(pcst.get("edge_cost_strategy") or "") or None
            lambda_value = pcst.get("edge_cost_lambda")
            if algorithm != expected.algorithm:
                rejected.append(f"{run.id} (algorithm={algorithm})")
                continue
            if expected.algorithm == "pcst" and (
                strategy != expected.cost_strategy
                or not math.isclose(
                    float(lambda_value),
                    float(expected.edge_cost_lambda),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                rejected.append(
                    f"{run.id} (strategy/lambda={strategy}/{lambda_value})"
                )
                continue
            summary = dict(run.summary or {})
            requested_metrics = dict(METRICS)
            if expected.algorithm == "pcst":
                requested_metrics.update(PCST_METRICS)
            values: dict[str, float] = {}
            missing: list[str] = []
            for metric_name, summary_key in requested_metrics.items():
                value = _numeric(summary, summary_key)
                if value is None:
                    missing.append(summary_key)
                else:
                    values[metric_name] = value
            if missing:
                rejected.append(f"{run.id} (missing {', '.join(missing)})")
                continue
            matches.append(
                RunRecord(
                    expected=expected,
                    wandb_run_id=str(run.id),
                    wandb_run_name=str(run.name),
                    wandb_run_url=str(run.url or ""),
                    wandb_group=str(run.group or ""),
                    evidence_lineage_name=lineage_name,
                    state=str(run.state or ""),
                    metrics=values,
                )
            )
        if len(matches) != 1:
            detail = "; ".join(rejected) if rejected else "none"
            raise RuntimeError(
                f"Expected one complete run for {expected.evidence_name}, found "
                f"{len(matches)}. Rejected matches: {detail}"
            )
        records.append(matches[0])
    return records


def configuration_order(configuration_id: str) -> tuple[int, int, float]:
    if configuration_id == "shortest_path":
        return (0, 0, 0.0)
    _, strategy, value = configuration_id.split("_", 2)
    return (1, 0 if strategy == "constant" else 1, float(value))


def aggregate(records: list[RunRecord]) -> list[dict[str, Any]]:
    configuration_ids = sorted(
        {record.expected.configuration_id for record in records},
        key=configuration_order,
    )
    rows: list[dict[str, Any]] = []
    for configuration_id in configuration_ids:
        selected = [
            record
            for record in records
            if record.expected.configuration_id == configuration_id
        ]
        if len(selected) != 3:
            raise RuntimeError(
                f"Configuration {configuration_id} has {len(selected)} seeds instead of 3."
            )
        first = selected[0].expected
        row: dict[str, Any] = {
            "configuration_id": configuration_id,
            "configuration_label": first.configuration_label,
            "algorithm": first.algorithm,
            "cost_strategy": first.cost_strategy or "",
            "edge_cost_lambda": first.edge_cost_lambda if first.edge_cost_lambda is not None else "",
            "seed_count": len(selected),
        }
        metric_names = list(METRICS)
        if first.algorithm == "pcst":
            metric_names.extend(PCST_METRICS)
        for metric_name in metric_names:
            values = [record.metrics[metric_name] for record in selected]
            row[f"{metric_name}_mean"] = statistics.fmean(values)
            row[f"{metric_name}_std"] = statistics.stdev(values)
        rows.append(row)
    return rows


def write_raw_csv(path: Path, records: list[RunRecord]) -> None:
    metric_names = [*METRICS, *PCST_METRICS]
    fields = [
        "experiment_id",
        "configuration_id",
        "seed",
        "wandb_run_id",
        "wandb_run_name",
        "wandb_run_url",
        "wandb_group",
        "evidence_lineage_name",
        "state",
        *metric_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "experiment_id": record.expected.experiment_id,
                    "configuration_id": record.expected.configuration_id,
                    "seed": record.expected.seed,
                    "wandb_run_id": record.wandb_run_id,
                    "wandb_run_name": record.wandb_run_name,
                    "wandb_run_url": record.wandb_run_url,
                    "wandb_group": record.wandb_group,
                    "evidence_lineage_name": record.evidence_lineage_name,
                    "state": record.state,
                    **record.metrics,
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "configuration_id",
        "configuration_label",
        "algorithm",
        "cost_strategy",
        "edge_cost_lambda",
        "seed_count",
    ]
    for metric_name in [*METRICS, *PCST_METRICS]:
        fields.extend((f"{metric_name}_mean", f"{metric_name}_std"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _latex_metric(row: dict[str, Any], metric_name: str, decimals: int = 3) -> str:
    mean = float(row[f"{metric_name}_mean"])
    std = float(row[f"{metric_name}_std"])
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Постапка & AverageTriples & AverageDistinctNodes & ContextCandidateCoverage & "
            r"CandidateReduction [\%] & ContextGoldCoverage & ContextFullGoldCoverage \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["configuration_label"]),
                    _latex_metric(row, "average_subgraph_triples", 2),
                    _latex_metric(row, "average_distinct_nodes", 2),
                    _latex_metric(row, "context_candidate_coverage"),
                    _latex_metric(row, "candidate_reduction_percentage", 2),
                    _latex_metric(row, "context_gold_coverage"),
                    _latex_metric(row, "context_full_gold_coverage"),
                ]
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Figure generation requires Matplotlib; run with `uv run --with matplotlib`."
        ) from error

    pcst_rows = [row for row in rows if row["algorithm"] == "pcst"]
    lambda_values = sorted({float(row["edge_cost_lambda"]) for row in pcst_rows})
    x = np.arange(len(lambda_values), dtype=float)
    shortest = next(row for row in rows if row["algorithm"] == "shortest_path")
    plot_specs = (
        ("average_subgraph_triples", "AverageTriples"),
        ("candidate_reduction_percentage", "CandidateReduction [%]"),
        ("context_gold_coverage", "ContextGoldCoverage"),
        ("context_full_gold_coverage", "ContextFullGoldCoverage"),
    )
    colors = {"constant": "#4472C4", "semantic": "#ED7D31"}
    labels = {"constant": "PCST, константна цена", "semantic": "PCST, семантичка цена"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for axis, (metric_name, ylabel) in zip(axes.flat, plot_specs, strict=True):
        for strategy in ("constant", "semantic"):
            strategy_rows = {
                float(row["edge_cost_lambda"]): row
                for row in pcst_rows
                if row["cost_strategy"] == strategy
            }
            means = [
                float(strategy_rows[value][f"{metric_name}_mean"])
                for value in lambda_values
            ]
            errors = [
                float(strategy_rows[value][f"{metric_name}_std"])
                for value in lambda_values
            ]
            axis.errorbar(
                x,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=colors[strategy],
                label=labels[strategy],
            )
        baseline = float(shortest[f"{metric_name}_mean"])
        axis.axhline(
            baseline,
            color="#666666",
            linestyle="--",
            linewidth=1.4,
            label="Унија на најкратки патеки",
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_xticks(x, [f"{value:g}" for value in lambda_values])
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Вредност на $\lambda$")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "evidence_pcst_lambda_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "evidence_pcst_lambda_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_provenance(
    path: Path,
    *,
    entity: str,
    project: str,
    experiment_path: Path,
    base_group: str,
    records: list[RunRecord],
) -> None:
    payload = {
        "wandb_project": f"{entity}/{project}",
        "experiment_file": str(experiment_path.resolve()),
        "authoritative_group_union": [
            base_group,
            f"{base_group}_seed_winners",
            f"{base_group}_winners",
        ],
        "selection_rule": (
            "Exact experiment-defined evidence lineage with matching algorithm, "
            "cost strategy, and lambda; one complete run per declared run."
        ),
        "metrics": {**METRICS, **PCST_METRICS},
        "excluded_metrics": [
            "empty_subgraph_count",
            "empty_subgraph_rate",
            "average_construction_time_ms",
            "average_objective",
        ],
        "runs": [
            {
                "experiment_id": record.expected.experiment_id,
                "configuration_id": record.expected.configuration_id,
                "seed": record.expected.seed,
                "wandb_run_id": record.wandb_run_id,
                "wandb_run_name": record.wandb_run_name,
                "wandb_run_url": record.wandb_run_url,
                "wandb_group": record.wandb_group,
                "evidence_lineage_name": record.evidence_lineage_name,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="itonkdong-org")
    parser.add_argument("--project", default="graphragx")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument(
        "--results-metadata-dir",
        type=Path,
        default=DEFAULT_RESULTS_METADATA,
    )
    parser.add_argument("--wandb-group-prefix", default="experiment1")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    expected = load_expected_runs(arguments.experiment)
    records = fetch_records(
        entity=arguments.entity,
        project=arguments.project,
        expected_runs=expected,
        base_group=arguments.wandb_group_prefix,
    )
    figures_dir = arguments.figures_dir.resolve()
    tables_dir = arguments.tables_dir.resolve()
    results_metadata_dir = arguments.results_metadata_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_metadata_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(records)
    write_raw_csv(results_metadata_dir / "evidence_runs.csv", records)
    write_summary_csv(results_metadata_dir / "evidence_summary.csv", rows)
    write_latex_table(tables_dir / "evidence_primary_table.tex", rows)
    write_provenance(
        results_metadata_dir / "provenance.json",
        entity=arguments.entity,
        project=arguments.project,
        experiment_path=arguments.experiment,
        base_group=arguments.wandb_group_prefix,
        records=records,
    )
    if not arguments.no_figures:
        write_figure(figures_dir, rows)
    print(
        "Wrote evidence-subgraph results to "
        f"figures={figures_dir}, tables={tables_dir}, "
        f"metadata={results_metadata_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
