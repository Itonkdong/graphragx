#!/usr/bin/env python3
"""Build thesis outputs for the corrected end-to-end LLM experiments.

The script reads W&B only. Runs are resolved by the inference lineage names in
``experiments/experiment_2_end_to_end.toml``. The superseded ``experiment2``
group is deliberately excluded; the corrected literal ``experiment2*`` group
and its seed/winner companion groups are authoritative.
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
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments/experiment_2_end_to_end.toml"
DEFAULT_FIGURES = PROJECT_ROOT / "metadata/figures/end_to_end_llm"
DEFAULT_TABLES = PROJECT_ROOT / "metadata/tables/end_to_end_llm"
DEFAULT_RESULTS_METADATA = PROJECT_ROOT / "metadata/results_metadata/end_to_end_llm"

MODEL_ORDER = ("deepseek-v4-flash", "gpt-5.6-luna")
MODEL_LABELS = {
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna",
}
CONFIGURATION_ORDER = (
    "shortest_path",
    "pcst_constant_0.01",
    "pcst_constant_1",
    "pcst_semantic_0.01",
    "pcst_semantic_1",
)
CONFIGURATION_LABELS = {
    "shortest_path": "Најкратки патеки",
    "pcst_constant_0.01": r"PCST, конст., $\lambda=0.01$",
    "pcst_constant_1": r"PCST, конст., $\lambda=1$",
    "pcst_semantic_0.01": r"PCST, сем., $\lambda=0.01$",
    "pcst_semantic_1": r"PCST, сем., $\lambda=1$",
}
PLOT_CONFIGURATION_LABELS = {
    "shortest_path": "Најкратки\nпатеки",
    "pcst_constant_0.01": "PCST конст.\nλ=0.01",
    "pcst_constant_1": "PCST конст.\nλ=1",
    "pcst_semantic_0.01": "PCST сем.\nλ=0.01",
    "pcst_semantic_1": "PCST сем.\nλ=1",
}

SUMMARY_METRICS = {
    "answer_hit_rate": "Summary_Plots/answer_hit_rate",
    "answer_precision": "Summary_Plots/answer_precision",
    "answer_recall": "Summary_Plots/answer_recall",
    "answer_f1": "Summary_Plots/answer_f1",
    "answer_exact_match": "Summary_Plots/answer_accuracy",
    "llm_exact_match_given_full_context": (
        "Summary_Plots/llm_exact_match_given_full_context"
    ),
    "llm_omission_given_full_context": (
        "Summary_Plots/llm_omission_given_full_context_rate"
    ),
    "llm_retrieved_gold_utilization": "Summary_Plots/llm_retrieved_gold_utilization",
    "full_context_complete_answer": (
        "Summary_Plots/full_context_complete_answer_rate"
    ),
    "full_context_llm_omission": (
        "Summary_Plots/full_context_llm_omission_rate"
    ),
    "partial_context_fully_utilized": (
        "Summary_Plots/partial_context_fully_utilized_rate"
    ),
    "partial_context_underutilized": (
        "Summary_Plots/partial_context_underutilized_rate"
    ),
    "context_gold_coverage": "Summary_Plots/reasoning_context_gold_coverage",
    "context_full_gold_coverage": (
        "Summary_Plots/reasoning_context_full_gold_coverage_rate"
    ),
}


@dataclass(frozen=True)
class ExpectedRun:
    experiment_id: str
    inference_name: str
    seed: int
    provider: str
    model: str
    algorithm: str
    cost_strategy: str | None
    edge_cost_lambda: float | None

    @property
    def configuration_id(self) -> str:
        if self.algorithm == "shortest_path":
            return "shortest_path"
        return f"pcst_{self.cost_strategy}_{self.edge_cost_lambda:g}"


@dataclass(frozen=True)
class RunRecord:
    expected: ExpectedRun
    wandb_run_id: str
    wandb_run_name: str
    wandb_run_url: str
    wandb_group: str
    inference_lineage_name: str
    state: str
    metrics: dict[str, float]


def _argument_value(arguments: list[str], option: str) -> str:
    try:
        index = arguments.index(option)
        return arguments[index + 1]
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
    default_arguments = [str(value) for value in payload.get("defaults", {}).get("args", [])]
    expected: list[ExpectedRun] = []
    for run in payload.get("runs", []):
        arguments = [str(value) for value in run.get("args", [])]
        effective_arguments = [*default_arguments, *arguments]
        algorithm = _argument_value(effective_arguments, "--subgraph-algorithm")
        strategy = _optional_argument_value(
            effective_arguments, "--pcst-edge-cost-strategy"
        )
        lambda_value = _optional_argument_value(
            effective_arguments, "--pcst-edge-cost"
        )
        expected.append(
            ExpectedRun(
                experiment_id=str(run["id"]),
                inference_name=str(
                    run.get("wandb_lineage_name")
                    or _argument_value(
                        effective_arguments, "--inference-run-name"
                    )
                ),
                seed=int(_argument_value(effective_arguments, "--seed")),
                provider=_argument_value(effective_arguments, "--llm-provider"),
                model=_argument_value(
                    effective_arguments, "--main-llm-model"
                ),
                algorithm=algorithm,
                cost_strategy=strategy,
                edge_cost_lambda=float(lambda_value) if lambda_value is not None else None,
            )
        )
    if not expected:
        raise ValueError(f"No runs were found in {path}.")
    return expected


def _lineage_name(config: dict[str, Any]) -> str:
    runs = config.get("runs")
    inference = runs.get("inference") if isinstance(runs, dict) else None
    name = inference.get("name") if isinstance(inference, dict) else None
    return str(name or "")


def _matches_lineage(actual: str, expected: str) -> bool:
    return actual == expected or actual.endswith(f"_{expected}")


def _numeric(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_evidence(expected: ExpectedRun, config: dict[str, Any]) -> str | None:
    evidence = (config.get("configs") or {}).get("evidence") or {}
    algorithm = str(evidence.get("algorithm") or "")
    if algorithm != expected.algorithm:
        return f"algorithm={algorithm}"
    if algorithm == "shortest_path":
        return None
    pcst = evidence.get("pcst") or {}
    strategy = str(pcst.get("edge_cost_strategy") or "")
    lambda_value = pcst.get("edge_cost_lambda")
    if strategy != expected.cost_strategy:
        return f"strategy={strategy}"
    try:
        matches_lambda = math.isclose(
            float(lambda_value),
            float(expected.edge_cost_lambda),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        matches_lambda = False
    return None if matches_lambda else f"lambda={lambda_value}"


def fetch_records(
    *,
    entity: str,
    project: str,
    expected_runs: list[ExpectedRun],
    groups: set[str],
) -> list[RunRecord]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("The W&B package is required to fetch results.") from error

    project_runs = list(wandb.Api().runs(f"{entity}/{project}", per_page=300))
    indexed: list[tuple[Any, str, dict[str, Any]]] = []
    for run in project_runs:
        if str(run.group or "") not in groups:
            continue
        config = dict(run.config or {})
        indexed.append((run, _lineage_name(config), config))

    records: list[RunRecord] = []
    for expected in expected_runs:
        matches: list[RunRecord] = []
        rejected: list[str] = []
        for run, lineage_name, config in indexed:
            if not _matches_lineage(lineage_name, expected.inference_name):
                continue
            inference = (config.get("configs") or {}).get("inference") or {}
            model = str(inference.get("model_id") or "")
            provider = str(inference.get("llm_provider") or "")
            model_training = (config.get("configs") or {}).get("model") or {}
            training = model_training.get("training") or {}
            saved_seed = training.get("random_seed")
            evidence_error = _validate_evidence(expected, config)
            if (
                model != expected.model
                or provider != expected.provider
                or int(saved_seed) != expected.seed
                or evidence_error is not None
            ):
                rejected.append(
                    f"{run.id} (model/provider/seed/evidence="
                    f"{model}/{provider}/{saved_seed}/{evidence_error})"
                )
                continue

            summary = dict(run.summary or {})
            values: dict[str, float] = {}
            missing: list[str] = []
            for metric_name, key in SUMMARY_METRICS.items():
                value = _numeric(summary, key)
                if value is None:
                    missing.append(key)
                else:
                    values[metric_name] = value
            for metric_name, key in {
                "prompt_tokens": "total_prompt_tokens",
                "completion_tokens": "total_completion_tokens",
                "total_tokens": "total_tokens",
                "total_requests": "total_requests",
            }.items():
                value = _numeric(inference, key)
                if value is None:
                    missing.append(f"configs.inference.{key}")
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
                    inference_lineage_name=lineage_name,
                    state=str(run.state or ""),
                    metrics=values,
                )
            )
        if len(matches) != 1:
            details = "; ".join(rejected) if rejected else "none"
            raise RuntimeError(
                f"Expected one complete run for {expected.inference_name}, found "
                f"{len(matches)}. Rejected matches: {details}"
            )
        records.append(matches[0])
    return records


def aggregate(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for configuration_id in CONFIGURATION_ORDER:
            selected = [
                record
                for record in records
                if record.expected.model == model
                and record.expected.configuration_id == configuration_id
            ]
            if len(selected) != 3:
                raise RuntimeError(
                    f"{model}/{configuration_id} has {len(selected)} seeds instead of 3."
                )
            row: dict[str, Any] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "configuration_id": configuration_id,
                "configuration_label": CONFIGURATION_LABELS[configuration_id],
                "seed_count": 3,
            }
            for metric_name in selected[0].metrics:
                values = [record.metrics[metric_name] for record in selected]
                row[f"{metric_name}_mean"] = statistics.fmean(values)
                row[f"{metric_name}_std"] = statistics.stdev(values)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _math_value(mean: float, std: float, *, digits: int = 3, bold: bool = False) -> str:
    body = f"${mean:.{digits}f} \\pm {std:.{digits}f}$"
    return rf"{{\boldmath {body}}}" if bold else body


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    quality_metrics = (
        "answer_hit_rate",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "answer_exact_match",
    )
    maxima = {
        metric: max(float(row[f"{metric}_mean"]) for row in rows)
        for metric in quality_metrics
    }
    minimum_tokens = min(float(row["total_tokens_mean"]) for row in rows)
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        (
            r"Јазичен модел & Доказен подграф & Hit & Precision & Recall & "
            r"F1 & ExactMatch & TotalTokens [милиони] \\"
        ),
        r"\midrule",
    ]
    for model_index, model in enumerate(MODEL_ORDER):
        model_rows = [row for row in rows if row["model"] == model]
        for row_index, row in enumerate(model_rows):
            model_cell = MODEL_LABELS[model] if row_index == 0 else ""
            cells = [model_cell, str(row["configuration_label"])]
            for metric in quality_metrics:
                mean = float(row[f"{metric}_mean"])
                cells.append(
                    _math_value(
                        mean,
                        float(row[f"{metric}_std"]),
                        bold=math.isclose(mean, maxima[metric], abs_tol=1e-12),
                    )
                )
            token_mean = float(row["total_tokens_mean"]) / 1_000_000
            token_std = float(row["total_tokens_std"]) / 1_000_000
            cells.append(
                _math_value(
                    token_mean,
                    token_std,
                    digits=2,
                    bold=math.isclose(
                        float(row["total_tokens_mean"]), minimum_tokens, abs_tol=1e-9
                    ),
                )
            )
            lines.append(" & ".join(cells) + r" \\")
        if model_index + 1 < len(MODEL_ORDER):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_quality_tokens(path_pdf: Path, path_png: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [PLOT_CONFIGURATION_LABELS[item] for item in CONFIGURATION_ORDER]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = {"deepseek-v4-flash": "#4472C4", "gpt-5.6-luna": "#ED7D31"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for model_index, model in enumerate(MODEL_ORDER):
        selected = [row for row in rows if row["model"] == model]
        positions = x + (model_index - 0.5) * width
        for axis, metric, title in zip(
            axes[:2],
            ("answer_hit_rate", "answer_f1"),
            ("Hit", "F1"),
            strict=True,
        ):
            axis.bar(
                positions,
                [float(row[f"{metric}_mean"]) for row in selected],
                width,
                yerr=[float(row[f"{metric}_std"]) for row in selected],
                capsize=3,
                color=colors[model],
                label=MODEL_LABELS[model],
            )
            axis.set_title(title)
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
        prompt = np.array([row["prompt_tokens_mean"] for row in selected]) / 1_000_000
        completion = np.array([row["completion_tokens_mean"] for row in selected]) / 1_000_000
        axes[2].bar(
            positions,
            prompt,
            width,
            color=colors[model],
            alpha=0.82,
            label=MODEL_LABELS[model],
        )
        axes[2].bar(
            positions,
            completion,
            width,
            bottom=prompt,
            color=colors[model],
            alpha=0.42,
            hatch="//",
        )
    axes[2].set_title("TotalTokens")
    axes[2].set_ylabel("Средна вредност [милиони]")
    axes[2].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Средна вредност")
    axes[1].set_ylabel("Средна вредност")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_context_outcomes(path_pdf: Path, path_png: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [PLOT_CONFIGURATION_LABELS[item] for item in CONFIGURATION_ORDER]
    outcomes = (
        ("full_context_complete_answer", "FullContextCompleteAnswer"),
        ("full_context_llm_omission", "FullContextLLMOmission"),
        ("llm_omission_given_full_context", "LLMOmissionGivenFullContext"),
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = {"deepseek-v4-flash": "#4472C4", "gpt-5.6-luna": "#ED7D31"}
    fig = plt.figure(figsize=(13.5, 8.5))
    grid = fig.add_gridspec(2, 2, hspace=0.48)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    )
    for axis, (metric, title) in zip(axes, outcomes, strict=True):
        for model_index, model in enumerate(MODEL_ORDER):
            selected = [row for row in rows if row["model"] == model]
            positions = x + (model_index - 0.5) * width
            axis.bar(
                positions,
                [float(row[f"{metric}_mean"]) for row in selected],
                width,
                yerr=[float(row[f"{metric}_std"]) for row in selected],
                capsize=3,
                color=colors[model],
                label=MODEL_LABELS[model],
            )
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Средна вредност")
    axes[2].set_ylabel("Средна вредност")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.tick_params(axis="x", labelsize=8.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.11,
        top=0.88,
        wspace=0.18,
        hspace=0.48,
    )
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_provenance(
    path: Path,
    *,
    records: list[RunRecord],
    experiment_path: Path,
    groups: set[str],
) -> None:
    payload = {
        "authoritative_groups": sorted(groups),
        "explicitly_excluded_superseded_group": "experiment2",
        "experiment_file": str(experiment_path),
        "token_source": "configs.inference API-reported usage totals",
        "runs": [
            {
                "experiment_id": record.expected.experiment_id,
                "seed": record.expected.seed,
                "model": record.expected.model,
                "configuration_id": record.expected.configuration_id,
                "inference_lineage_name": record.inference_lineage_name,
                "wandb_group": record.wandb_group,
                "wandb_run_id": record.wandb_run_id,
                "wandb_run_name": record.wandb_run_name,
                "wandb_run_url": record.wandb_run_url,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    expected_runs = load_expected_runs(arguments.experiment)
    groups = {"experiment2*", "experiment2_seed_winners", "experiment2_winners"}
    records = fetch_records(
        entity=arguments.entity,
        project=arguments.project,
        expected_runs=expected_runs,
        groups=groups,
    )
    if len(records) != 30:
        raise RuntimeError(f"Expected 30 corrected end-to-end runs, found {len(records)}.")
    rows = aggregate(records)
    figures_dir = arguments.figures_dir.resolve()
    tables_dir = arguments.tables_dir.resolve()
    results_metadata_dir = arguments.results_metadata_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_metadata_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    for record in records:
        run_rows.append(
            {
                "experiment_id": record.expected.experiment_id,
                "model": record.expected.model,
                "configuration_id": record.expected.configuration_id,
                "seed": record.expected.seed,
                "wandb_group": record.wandb_group,
                "wandb_run_id": record.wandb_run_id,
                "wandb_run_name": record.wandb_run_name,
                "inference_lineage_name": record.inference_lineage_name,
                **record.metrics,
            }
        )
    _write_csv(results_metadata_dir / "end_to_end_llm_runs.csv", run_rows)
    _write_csv(results_metadata_dir / "end_to_end_llm_summary.csv", rows)
    write_table(tables_dir / "end_to_end_llm_results.tex", rows)
    _plot_quality_tokens(
        figures_dir / "end_to_end_llm_quality_tokens.pdf",
        figures_dir / "end_to_end_llm_quality_tokens.png",
        rows,
    )
    _plot_context_outcomes(
        figures_dir / "end_to_end_context_outcomes.pdf",
        figures_dir / "end_to_end_context_outcomes.png",
        rows,
    )
    write_provenance(
        results_metadata_dir / "provenance.json",
        records=records,
        experiment_path=arguments.experiment,
        groups=groups,
    )
    print(
        "Wrote end-to-end LLM results to "
        f"figures={figures_dir}, tables={tables_dir}, "
        f"metadata={results_metadata_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Could not build end-to-end LLM results: {error}", file=sys.stderr)
        raise SystemExit(1) from error
