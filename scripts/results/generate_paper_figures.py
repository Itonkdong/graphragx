#!/usr/bin/env python3
"""Generate the English figures for the paper from existing results.

This script is intentionally separate from the thesis and presentation figure
generators. It reads the persisted result summaries and writes only to
``metadata/paper/figures``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas

from scripts.results import generate_architecture_figures as architecture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "metadata/paper/figures"
EVIDENCE_SUMMARY = (
    PROJECT_ROOT / "metadata/results_metadata/evidence_subgraphs/evidence_summary.csv"
)
END_TO_END_SUMMARY = (
    PROJECT_ROOT
    / "metadata/results_metadata/end_to_end_llm/end_to_end_llm_summary.csv"
)

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
PLOT_CONFIGURATION_LABELS = {
    "shortest_path": "Shortest\npaths",
    "pcst_constant_0.01": "PCST const.\n$\\lambda=0.01$",
    "pcst_constant_1": "PCST const.\n$\\lambda=1$",
    "pcst_semantic_0.01": "PCST sem.\n$\\lambda=0.01$",
    "pcst_semantic_1": "PCST sem.\n$\\lambda=1$",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a persisted experiment summary without modifying it."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_system_overview(output: Path) -> None:
    """Draw the English paper version of the system overview."""
    width, height = 1200, 450
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("System overview")
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    groups = {
        "input": (14, 36, 202, 378),
        "retrieval": (230, 36, 360, 378),
        "evidence": (604, 36, 272, 378),
        "generation": (890, 36, 296, 378),
    }
    group_styles = {
        "input": (architecture.BLUE_LIGHT, architecture.BLUE),
        "retrieval": (architecture.RETRIEVER_LIGHT, architecture.RETRIEVER),
        "evidence": (architecture.GREEN_LIGHT, architecture.GREEN),
        "generation": (architecture.PURPLE_LIGHT, architecture.PURPLE),
    }
    for name, (x, y, w, h) in groups.items():
        fill, stroke = group_styles[name]
        architecture.stage_wrapper(c, x, y, w, h, fill=fill, stroke=stroke)

    for name, label in {
        "input": "Input",
        "retrieval": "Retrieval",
        "evidence": "Evidence construction",
        "generation": "Generation",
    }.items():
        x, y, w, h = groups[name]
        architecture.stage_heading(c, label, x, y + h - 25, w)

    panels = {
        "input": (28, 58, 174, 310),
        "retriever": (246, 69, 164, 288),
        "candidates": (426, 92, 148, 242),
        "evidence": (619, 58, 242, 310),
        "llm": (905, 58, 170, 310),
        "answer": (1091, 116, 80, 194),
    }
    for name, (x, y, w, h) in panels.items():
        if name == "input":
            fill, stroke = architecture.BLUE_LIGHT, architecture.BLUE
        elif name == "retriever":
            fill, stroke = architecture.RETRIEVER_LIGHT, architecture.RETRIEVER
        elif name == "candidates":
            fill, stroke = architecture.ORANGE_LIGHT, architecture.ORANGE
        elif name == "evidence":
            fill, stroke = architecture.GREEN_LIGHT, architecture.GREEN
        elif name == "llm":
            fill, stroke = architecture.PURPLE_LIGHT, architecture.PURPLE
        else:
            fill, stroke = architecture.TEAL_LIGHT, architecture.TEAL
        architecture.rounded_box(
            c, x, y, w, h, fill=fill, stroke=stroke, radius=12, width=1.35
        )

    x, y, w, h = panels["input"]
    architecture.question_symbol(c, x + 63, y + h - 77, w=48, h=42, show_label=False)
    architecture.centered_lines(
        c, ("Question",), x + w / 2, y + h - 105, size=10.5,
        font="DiagramBold", color=architecture.INK,
    )
    architecture.draw_consistent_graph(
        c, x + 7, y + 67, scale=0.91, show_candidates=False
    )
    architecture.centered_lines(
        c, ("Local graph",), x + w / 2, y + 39, size=10.5,
        color=architecture.MUTED,
    )

    x, y, w, h = panels["retriever"]
    architecture.draw_consistent_graph(
        c, x + 4, y + 153, scale=0.89, show_candidates=False
    )
    architecture.arrow(
        c, x + w / 2, y + 144, x + w / 2, y + 116,
        color=architecture.RETRIEVER, width=1.8, head=7,
    )
    architecture.rounded_box(
        c, x + 35, y + 61, 96, 48, fill=white,
        stroke=architecture.RETRIEVER, radius=9,
    )
    architecture.centered_lines(
        c, ("GNN",), x + w / 2, y + 85, size=16,
        font="DiagramBold", color=architecture.RETRIEVER,
    )
    architecture.centered_lines(
        c, ("Node scoring",), x + w / 2, y + 33, size=9.5,
        color=architecture.MUTED,
    )

    x, y, w, h = panels["candidates"]
    architecture.centered_lines(
        c, ("Candidate", "entities"), x + w / 2, y + h - 28,
        size=10.5, leading=11, font="DiagramBold",
    )
    for rank, bar_width, is_gold in ((1, 84, False), (2, 68, True), (3, 51, False)):
        cy = y + h - 83 - (rank - 1) * 54
        node_fill = architecture.GREEN_LIGHT if is_gold else architecture.ORANGE_LIGHT
        node_stroke = architecture.GREEN if is_gold else architecture.ORANGE
        architecture.node(c, x + 27, cy, radius=9, fill=node_fill, stroke=node_stroke)
        if is_gold:
            architecture.dotted_gold_ring(c, x + 27, cy, radius=14)
        c.setFont("DiagramBold", 9)
        c.setFillColor(node_stroke)
        c.drawCentredString(x + 27, cy - 3, str(rank))
        c.roundRect(x + 47, cy - 6, bar_width, 12, 6, fill=1, stroke=0)

    x, y, w, h = panels["evidence"]
    architecture.rounded_box(
        c, x + 21, y + h - 91, 92, 38,
        fill=architecture.BLUE_LIGHT, stroke=architecture.BLUE, radius=8, width=1.1,
    )
    architecture.rounded_box(
        c, x + 129, y + h - 91, 92, 38,
        fill=architecture.GREEN_LIGHT, stroke=architecture.GREEN, radius=8, width=1.1,
    )
    architecture.centered_lines(
        c, ("Shortest", "paths"), x + 67, y + h - 72,
        size=9.4, leading=10.2, font="DiagramBold",
    )
    architecture.centered_lines(
        c, ("PCST",), x + 175, y + h - 72, size=10.5, font="DiagramBold"
    )
    architecture.arrow(
        c, x + 67, y + h - 98, x + 105, y + 192,
        color=architecture.LINE, width=1.3, head=6,
    )
    architecture.arrow(
        c, x + 175, y + h - 98, x + 133, y + 192,
        color=architecture.LINE, width=1.3, head=6,
    )
    architecture.draw_consistent_graph(
        c, x + 29, y + 61, scale=1.04, show_candidates=True, fade_unselected=True
    )
    architecture.centered_lines(
        c, ("Evidence subgraph",), x + w / 2, y + 35,
        size=9.5, color=architecture.MUTED,
    )

    x, y, w, h = panels["llm"]
    architecture.question_symbol(
        c, x + 14, y + h - 73, w=42, h=38, show_label=False
    )
    c.setFillColor(architecture.MUTED)
    c.setFont("DiagramBold", 16)
    c.drawCentredString(x + 76, y + h - 61, "+")
    architecture.draw_consistent_graph(
        c, x + 78, y + h - 96, scale=0.48,
        show_candidates=True, selected_only=True,
    )
    architecture.arrow(
        c, x + w / 2, y + h - 93, x + w / 2, y + 187,
        color=architecture.PURPLE, width=1.5, head=7,
    )
    architecture.document_symbol(c, x + 64, y + 127, w=48, h=55)
    architecture.centered_lines(
        c, ("Textualization",), x + w / 2, y + 111,
        size=9.5, color=architecture.MUTED,
    )
    architecture.arrow(
        c, x + w / 2, y + 99, x + w / 2, y + 78,
        color=architecture.PURPLE, width=1.5, head=7,
    )
    architecture.rounded_box(
        c, x + 42, y + 31, 92, 43, fill=white,
        stroke=architecture.PURPLE, radius=9,
    )
    architecture.centered_lines(
        c, ("LLM",), x + w / 2, y + 53, size=16,
        font="DiagramBold", color=architecture.PURPLE,
    )
    architecture.centered_lines(
        c, ("Answer generation",), x + w / 2, y + 15,
        size=9.5, color=architecture.MUTED,
    )

    x, y, w, h = panels["answer"]
    architecture.centered_lines(
        c, ("Output",), x + w / 2, y + h - 23, size=11.5, font="DiagramBold"
    )
    answer_x, answer_y = x + w / 2, y + h / 2 - 4
    architecture.node(
        c, answer_x, answer_y, radius=18,
        fill=architecture.GREEN_LIGHT, stroke=architecture.GREEN, width=1.7,
    )
    architecture.dotted_gold_ring(c, answer_x, answer_y, radius=27)

    for source, target in (
        ("input", "retriever"),
        ("retriever", "candidates"),
        ("candidates", "evidence"),
        ("evidence", "llm"),
        ("llm", "answer"),
    ):
        sx, sy, sw, sh = panels[source]
        tx, ty, _, th = panels[target]
        architecture.arrow(
            c, sx + sw + 6, sy + sh / 2, tx - 7, ty + th / 2,
            color=architecture.INK, width=2.0, head=8,
        )

    c.showPage()
    c.save()


def draw_information_flow(output: Path) -> None:
    """Draw the English paper version of the stage-wise information flow."""
    width, height = 1200, 430
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("Information flow through the system")
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    groups = (
        (15, architecture.BLUE_LIGHT, architecture.BLUE, "Local graph"),
        (312, architecture.ORANGE_LIGHT, architecture.ORANGE, "Candidate entities"),
        (609, architecture.GREEN_LIGHT, architecture.GREEN, "Evidence subgraph"),
        (906, architecture.PURPLE_LIGHT, architecture.PURPLE, "Final answer"),
    )
    group_y, group_w, group_h = 75, 279, 285
    inner_y, inner_h = 78, 268
    for x, fill, stroke, heading in groups:
        architecture.rounded_box(
            c, x, group_y, group_w, group_h,
            fill=fill, stroke=stroke, radius=14, width=1.4,
        )
        architecture.stage_heading(c, heading, x, group_y + group_h - 25, group_w)

    local_x = groups[0][0] + 34
    freebase_nodes = ((30, 279), (55, 139), (250, 279), (250, 151))
    freebase_links = (
        (freebase_nodes[0], (local_x + 43 * 1.14, inner_y + 73 + 88 * 1.14)),
        (freebase_nodes[1], (local_x + 50 * 1.14, inner_y + 73 + 18 * 1.14)),
        (freebase_nodes[2], (local_x + 125 * 1.14, inner_y + 73 + 96 * 1.14)),
        (freebase_nodes[3], (local_x + 164 * 1.14, inner_y + 73 + 60 * 1.14)),
    )
    c.setStrokeColor(HexColor("#DCE3E8"))
    c.setLineWidth(1.0)
    for (sx, sy), (tx, ty) in freebase_links:
        c.line(groups[0][0] + sx, sy, tx, ty)
    for px, py in freebase_nodes:
        architecture.node(
            c, groups[0][0] + px, py, radius=7.2,
            fill=HexColor("#F6F8F9"), stroke=HexColor("#DCE3E8"), width=1.1,
        )
    architecture.draw_consistent_graph(
        c, local_x, inner_y + 73, scale=1.14, show_candidates=False
    )
    architecture.centered_lines(
        c,
        ("The local subgraph is extracted from Freebase", "and may not contain the gold answer"),
        groups[0][0] + group_w / 2,
        group_y + 25,
        size=8.6,
        leading=10.5,
        color=architecture.MUTED,
    )

    candidate_x = groups[1][0]
    for rank, bar_width, is_gold in ((1, 151, False), (2, 124, True), (3, 94, False)):
        cy = inner_y + inner_h - 63 - (rank - 1) * 68
        fill = architecture.GREEN_LIGHT if is_gold else architecture.ORANGE_LIGHT
        stroke = architecture.GREEN if is_gold else architecture.ORANGE
        architecture.node(c, candidate_x + 54, cy, radius=11, fill=fill, stroke=stroke, width=1.4)
        if is_gold:
            architecture.dotted_gold_ring(c, candidate_x + 54, cy, radius=17)
        c.setFillColor(stroke)
        c.setFont("DiagramBold", 10)
        c.drawCentredString(candidate_x + 54, cy - 3.5, str(rank))
        c.roundRect(candidate_x + 82, cy - 7, bar_width, 14, 7, fill=1, stroke=0)

    evidence_x = groups[2][0]
    points = [(8, 54), (43, 88), (84, 65), (125, 96), (164, 60), (112, 24), (50, 18)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (1, 6), (6, 5), (0, 6)]
    selected_edges = {(1, 2), (2, 3), (3, 4)}
    selected_nodes = {1, 2, 3, 4}
    origin_x, origin_y, scale = evidence_x + 34, inner_y + 73, 1.14
    for source, target in edges:
        sx, sy = points[source]
        tx, ty = points[target]
        selected = (source, target) in selected_edges
        c.setStrokeColor(architecture.INK if selected else HexColor("#DCE3E8"))
        c.setLineWidth(1.8 if selected else 1.1)
        c.line(origin_x + sx * scale, origin_y + sy * scale, origin_x + tx * scale, origin_y + ty * scale)
    for index, (px, py) in enumerate(points):
        x = origin_x + px * scale
        y = origin_y + py * scale
        if index == 1:
            fill, stroke = architecture.BLUE_LIGHT, architecture.BLUE
        elif index == 4:
            fill, stroke = architecture.GREEN_LIGHT, architecture.GREEN
        elif index == 3:
            fill, stroke = architecture.ORANGE_LIGHT, architecture.ORANGE
        elif index == 5:
            fill, stroke = HexColor("#FFF9EF"), HexColor("#EBCB94")
        elif index in selected_nodes:
            fill, stroke = white, architecture.MUTED
        else:
            fill, stroke = HexColor("#F6F8F9"), HexColor("#DCE3E8")
        architecture.node(c, x, y, radius=8.2, fill=fill, stroke=stroke, width=1.35)
        if index == 4:
            architecture.dotted_gold_ring(c, x, y, radius=13.2)

    answer_x = groups[3][0]
    architecture.node(
        c, answer_x + group_w / 2, group_y + group_h / 2 - 5,
        radius=22, fill=architecture.GREEN_LIGHT, stroke=architecture.GREEN, width=1.7,
    )
    architecture.dotted_gold_ring(
        c, answer_x + group_w / 2, group_y + group_h / 2 - 5, radius=32
    )

    for index in range(3):
        architecture.arrow(
            c, groups[index][0] + group_w + 2, 220,
            groups[index + 1][0] - 4, 220,
            color=architecture.INK, width=1.8, head=6,
        )

    legend_items = (
        (architecture.BLUE_LIGHT, architecture.BLUE, "Seed entity", False),
        (architecture.ORANGE_LIGHT, architecture.ORANGE, "Candidate entity", False),
        (white, architecture.MUTED, "Intermediate entity", False),
        (architecture.GREEN_LIGHT, architecture.GREEN, "Gold answer", True),
    )
    for start, (fill, stroke, label, ring) in zip((265, 465, 680, 900), legend_items, strict=True):
        architecture.node(c, start, 25, radius=6.5, fill=fill, stroke=stroke, width=1.1)
        if ring:
            architecture.dotted_gold_ring(c, start, 25, radius=10)
        c.setFillColor(architecture.INK)
        c.setFont("Diagram", 9.5)
        c.drawString(start + 13, 22, label)

    c.showPage()
    c.save()


def plot_evidence_sensitivity(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot the English PCST sensitivity figure from the saved summary."""
    import matplotlib.pyplot as plt
    import numpy as np

    pcst_rows = [row for row in rows if row["algorithm"] == "pcst"]
    lambda_values = sorted({float(row["edge_cost_lambda"]) for row in pcst_rows})
    shortest = next(row for row in rows if row["algorithm"] == "shortest_path")
    x = np.arange(len(lambda_values), dtype=float)
    plot_specs = (
        ("average_subgraph_triples", "Average triples"),
        ("candidate_reduction_percentage", "Candidate reduction [%]"),
        ("context_gold_coverage", "Context gold coverage"),
        ("context_full_gold_coverage", "Full context gold coverage"),
    )
    colors = {"constant": "#4472C4", "semantic": "#ED7D31"}
    labels = {"constant": "PCST, constant cost", "semantic": "PCST, semantic cost"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for axis, (metric, ylabel) in zip(axes.flat, plot_specs, strict=True):
        for strategy in ("constant", "semantic"):
            selected = {
                float(row["edge_cost_lambda"]): row
                for row in pcst_rows
                if row["cost_strategy"] == strategy
            }
            axis.errorbar(
                x,
                [float(selected[value][f"{metric}_mean"]) for value in lambda_values],
                yerr=[float(selected[value][f"{metric}_std"]) for value in lambda_values],
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=colors[strategy],
                label=labels[strategy],
            )
        axis.axhline(
            float(shortest[f"{metric}_mean"]),
            color="#666666",
            linestyle="--",
            linewidth=1.4,
            label="Shortest-path union",
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_xticks(x, [f"{value:g}" for value in lambda_values])
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Value of $\lambda$")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "evidence_pcst_lambda_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "evidence_pcst_lambda_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ordered_end_to_end_rows(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    """Return one model's persisted rows in the shared plot order."""
    by_configuration = {
        row["configuration_id"]: row for row in rows if row["model"] == model
    }
    missing = set(CONFIGURATION_ORDER) - set(by_configuration)
    if missing:
        raise ValueError(f"Missing configurations for {model}: {sorted(missing)}")
    return [by_configuration[configuration] for configuration in CONFIGURATION_ORDER]


def plot_quality_tokens(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot English answer quality and token usage."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [PLOT_CONFIGURATION_LABELS[item] for item in CONFIGURATION_ORDER]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = {"deepseek-v4-flash": "#4472C4", "gpt-5.6-luna": "#ED7D31"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for model_index, model in enumerate(MODEL_ORDER):
        selected = ordered_end_to_end_rows(rows, model)
        positions = x + (model_index - 0.5) * width
        for axis, metric, title in zip(
            axes[:2], ("answer_hit_rate", "answer_f1"), ("Hit", "F1"), strict=True
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
        prompt = np.array([float(row["prompt_tokens_mean"]) for row in selected]) / 1_000_000
        completion = np.array([float(row["completion_tokens_mean"]) for row in selected]) / 1_000_000
        axes[2].bar(positions, prompt, width, color=colors[model], alpha=0.82, label=MODEL_LABELS[model])
        axes[2].bar(
            positions, completion, width, bottom=prompt,
            color=colors[model], alpha=0.42, hatch="//",
        )
    axes[2].set_title("Total tokens")
    axes[2].set_ylabel("Mean [millions]")
    axes[2].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Mean")
    axes[1].set_ylabel("Mean")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_dir / "end_to_end_llm_quality_tokens.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "end_to_end_llm_quality_tokens.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_context_outcomes(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot English full-context outcome and conditional omission metrics."""
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
            selected = ordered_end_to_end_rows(rows, model)
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
    axes[0].set_ylabel("Mean")
    axes[2].set_ylabel("Mean")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.tick_params(axis="x", labelsize=8.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.88, wspace=0.18, hspace=0.48)
    fig.savefig(output_dir / "end_to_end_context_outcomes.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "end_to_end_context_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all current and optional English paper figures."""
    architecture.register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    architecture_pdfs = (
        OUTPUT_DIR / "system_overview.pdf",
        OUTPUT_DIR / "information_flow.pdf",
    )
    draw_system_overview(architecture_pdfs[0])
    draw_information_flow(architecture_pdfs[1])
    for path in architecture_pdfs:
        architecture.render_png(path)

    plot_evidence_sensitivity(read_rows(EVIDENCE_SUMMARY), OUTPUT_DIR)
    end_to_end_rows = read_rows(END_TO_END_SUMMARY)
    plot_quality_tokens(end_to_end_rows, OUTPUT_DIR)
    plot_context_outcomes(end_to_end_rows, OUTPUT_DIR)

    for path in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"Wrote {path}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
