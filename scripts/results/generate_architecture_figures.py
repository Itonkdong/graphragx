#!/usr/bin/env python3
"""Generate the two vector architecture figures used in the thesis."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor, white


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "metadata/figures/system_architecture"
PNG_DPI = 300

INK = HexColor("#263746")
MUTED = HexColor("#617080")
LINE = HexColor("#A9B4BE")
LIGHT = HexColor("#F7F9FB")
BLUE = HexColor("#4F86C6")
BLUE_LIGHT = HexColor("#EAF2FB")
ORANGE = HexColor("#E5A23C")
ORANGE_LIGHT = HexColor("#FFF4DF")
RETRIEVER = HexColor("#D97942")
RETRIEVER_LIGHT = HexColor("#FBEDE5")
GREEN = HexColor("#4E9B69")
GREEN_LIGHT = HexColor("#EAF6EE")
PURPLE = HexColor("#8064A2")
PURPLE_LIGHT = HexColor("#F1ECF7")
TEAL = HexColor("#398D84")
TEAL_LIGHT = HexColor("#E8F5F3")
GRAY_NODE = HexColor("#DDE3E8")


def register_fonts() -> None:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Diagram", str(regular)))
            pdfmetrics.registerFont(TTFont("DiagramBold", str(bold)))
            return
    raise RuntimeError("A Unicode Arial or DejaVu Sans font could not be found.")


def rounded_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: Color,
    stroke: Color,
    radius: float = 14,
    width: float = 1.4,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def centered_lines(
    c: canvas.Canvas,
    lines: list[str] | tuple[str, ...],
    x: float,
    center_y: float,
    *,
    size: float = 13,
    leading: float | None = None,
    font: str = "Diagram",
    color: Color = INK,
) -> None:
    leading = leading or size * 1.2
    first_y = center_y + (len(lines) - 1) * leading / 2 - size * 0.34
    c.setFillColor(color)
    c.setFont(font, size)
    for index, line in enumerate(lines):
        c.drawCentredString(x, first_y - index * leading, line)


def arrow(
    c: canvas.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: Color = INK,
    width: float = 2.0,
    head: float = 8,
) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    left = (
        x2 - head * math.cos(angle) + head * 0.55 * math.sin(angle),
        y2 - head * math.sin(angle) - head * 0.55 * math.cos(angle),
    )
    right = (
        x2 - head * math.cos(angle) - head * 0.55 * math.sin(angle),
        y2 - head * math.sin(angle) + head * 0.55 * math.cos(angle),
    )
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(*left)
    path.lineTo(*right)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def node(
    c: canvas.Canvas,
    x: float,
    y: float,
    *,
    radius: float = 8,
    fill: Color = white,
    stroke: Color = INK,
    width: float = 1.3,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    c.circle(x, y, radius, fill=1, stroke=1)


def graph(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    points: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    *,
    fills: dict[int, Color] | None = None,
    radius: float = 7,
    scale: float = 1.0,
) -> None:
    fills = fills or {}
    c.setStrokeColor(MUTED)
    c.setLineWidth(1.2)
    for source, target in edges:
        sx, sy = points[source]
        tx, ty = points[target]
        c.line(origin_x + sx * scale, origin_y + sy * scale, origin_x + tx * scale, origin_y + ty * scale)
    for index, (px, py) in enumerate(points):
        node(
            c,
            origin_x + px * scale,
            origin_y + py * scale,
            radius=radius,
            fill=fills.get(index, white),
        )


def stage_heading(c: canvas.Canvas, text: str, x: float, y: float, w: float) -> None:
    c.setFont("DiagramBold", 14)
    c.setFillColor(INK)
    c.drawCentredString(x + w / 2, y, text)


def stage_wrapper(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: Color,
    stroke: Color,
) -> None:
    c.saveState()
    c.setFillColor(fill)
    c.setFillAlpha(0.42)
    c.setStrokeColor(stroke)
    c.setStrokeAlpha(0.62)
    c.setLineWidth(1.5)
    c.setDash(4, 4)
    c.roundRect(x, y, w, h, 16, fill=1, stroke=1)
    c.restoreState()


def question_symbol(
    c: canvas.Canvas,
    x: float,
    y: float,
    *,
    w: float = 48,
    h: float = 42,
    show_label: bool = True,
) -> None:
    c.setFillColor(white)
    c.setStrokeColor(BLUE)
    c.setLineWidth(1.4)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    c.setFillColor(BLUE)
    c.setFont("DiagramBold", 22)
    c.drawCentredString(x + w / 2, y + (h - 22) / 2 + 3, "?")
    if show_label:
        c.setFillColor(INK)
        c.setFont("DiagramBold", 10.5)
        c.drawCentredString(x + w / 2, y - 18, "Прашање")


def document_symbol(c: canvas.Canvas, x: float, y: float, *, w: float = 48, h: float = 55) -> None:
    c.setFillColor(white)
    c.setStrokeColor(PURPLE)
    c.setLineWidth(1.4)
    path = c.beginPath()
    path.moveTo(x, y)
    path.lineTo(x, y + h)
    path.lineTo(x + w - 13, y + h)
    path.lineTo(x + w, y + h - 13)
    path.lineTo(x + w, y)
    path.close()
    c.drawPath(path, fill=1, stroke=1)
    c.line(x + w - 13, y + h, x + w - 13, y + h - 13)
    c.line(x + w - 13, y + h - 13, x + w, y + h - 13)
    c.setStrokeColor(MUTED)
    c.setLineWidth(1.1)
    for offset, length in ((35, 26), (26, 33), (17, 28), (8, 20)):
        c.line(x + 8, y + offset, x + 8 + length, y + offset)


def dotted_gold_ring(c: canvas.Canvas, x: float, y: float, *, radius: float = 12) -> None:
    c.saveState()
    c.setStrokeColor(GREEN)
    c.setLineWidth(1.7)
    c.setDash(2.2, 2.2)
    c.circle(x, y, radius, fill=0, stroke=1)
    c.restoreState()


def draw_consistent_graph(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    *,
    scale: float,
    show_candidates: bool,
    fade_unselected: bool = False,
    selected_only: bool = False,
) -> None:
    points = [(8, 54), (43, 88), (84, 65), (125, 96), (164, 60), (112, 24), (50, 18)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (1, 6), (6, 5), (0, 6)]
    selected_edges = {(1, 2), (2, 3), (3, 4), (2, 5)}
    selected_nodes = {1, 2, 3, 4, 5}
    seed_index = 1
    gold_index = 4
    candidate_indices = {3, 4, 5}

    for source, target in edges:
        if selected_only and (source, target) not in selected_edges:
            continue
        sx, sy = points[source]
        tx, ty = points[target]
        selected = (source, target) in selected_edges
        c.setStrokeColor(INK if not fade_unselected or selected else HexColor("#DCE3E8"))
        c.setLineWidth(1.7 if fade_unselected and selected else 1.15)
        c.line(
            origin_x + sx * scale,
            origin_y + sy * scale,
            origin_x + tx * scale,
            origin_y + ty * scale,
        )

    for index, (px, py) in enumerate(points):
        if selected_only and index not in selected_nodes:
            continue
        x = origin_x + px * scale
        y = origin_y + py * scale
        is_selected = index in selected_nodes
        if fade_unselected and not is_selected:
            fill = HexColor("#F6F8F9")
            stroke = HexColor("#DCE3E8")
        elif index == seed_index:
            fill, stroke = BLUE_LIGHT, BLUE
        elif index == gold_index:
            fill, stroke = GREEN_LIGHT, GREEN
        elif show_candidates and index in candidate_indices:
            fill, stroke = ORANGE_LIGHT, ORANGE
        else:
            fill, stroke = white, MUTED
        node(c, x, y, radius=7.2 * scale, fill=fill, stroke=stroke, width=1.3)
        if index == gold_index:
            dotted_gold_ring(c, x, y, radius=11.5 * scale)


def draw_system_overview(output: Path) -> None:
    width, height = 1200, 450
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("Преглед на целосниот систем")
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    groups = {
        "input": (14, 36, 202, 378),
        "retrieval": (230, 36, 360, 378),
        "evidence": (604, 36, 272, 378),
        "generation": (890, 36, 296, 378),
    }
    group_styles = {
        "input": (BLUE_LIGHT, BLUE),
        "retrieval": (RETRIEVER_LIGHT, RETRIEVER),
        "evidence": (GREEN_LIGHT, GREEN),
        "generation": (PURPLE_LIGHT, PURPLE),
    }
    for name, (x, y, w, h) in groups.items():
        fill, stroke = group_styles[name]
        stage_wrapper(c, x, y, w, h, fill=fill, stroke=stroke)

    for name, text in {
        "input": "Влез",
        "retrieval": "Пребарување",
        "evidence": "Конструкција на докази",
        "generation": "Генерирање",
    }.items():
        x, y, w, h = groups[name]
        stage_heading(c, text, x, y + h - 25, w)

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
            fill, stroke = BLUE_LIGHT, BLUE
        elif name == "retriever":
            fill, stroke = RETRIEVER_LIGHT, RETRIEVER
        elif name == "candidates":
            fill, stroke = ORANGE_LIGHT, ORANGE
        elif name == "evidence":
            fill, stroke = GREEN_LIGHT, GREEN
        elif name == "llm":
            fill, stroke = PURPLE_LIGHT, PURPLE
        else:
            fill, stroke = TEAL_LIGHT, TEAL
        rounded_box(c, x, y, w, h, fill=fill, stroke=stroke, radius=12, width=1.35)

    x, y, w, h = panels["input"]
    question_symbol(c, x + 63, y + h - 77, w=48, h=42)
    draw_consistent_graph(c, x + 7, y + 67, scale=0.91, show_candidates=False)
    centered_lines(c, ("Локален граф",), x + w / 2, y + 39, size=10.5, color=MUTED)

    x, y, w, h = panels["retriever"]
    draw_consistent_graph(c, x + 4, y + 153, scale=0.89, show_candidates=False)
    arrow(c, x + w / 2, y + 144, x + w / 2, y + 116, color=RETRIEVER, width=1.8, head=7)
    rounded_box(c, x + 35, y + 61, 96, 48, fill=white, stroke=RETRIEVER, radius=9)
    centered_lines(c, ("GNN",), x + w / 2, y + 85, size=16, font="DiagramBold", color=RETRIEVER)
    centered_lines(c, ("Оценување на јазлите",), x + w / 2, y + 33, size=9.5, color=MUTED)

    x, y, w, h = panels["candidates"]
    centered_lines(c, ("Ентитети-кандидати",), x + w / 2, y + h - 25, size=11.5, font="DiagramBold")
    candidate_data = ((1, 84, False), (2, 68, True), (3, 51, False))
    for rank, bar_width, is_gold in candidate_data:
        cy = y + h - 76 - (rank - 1) * 57
        node_fill = GREEN_LIGHT if is_gold else ORANGE_LIGHT
        node_stroke = GREEN if is_gold else ORANGE
        node(c, x + 27, cy, radius=9, fill=node_fill, stroke=node_stroke)
        if is_gold:
            dotted_gold_ring(c, x + 27, cy, radius=14)
        c.setFont("DiagramBold", 9)
        c.setFillColor(node_stroke)
        c.drawCentredString(x + 27, cy - 3, str(rank))
        c.setFillColor(node_stroke)
        c.roundRect(x + 47, cy - 6, bar_width, 12, 6, fill=1, stroke=0)

    x, y, w, h = panels["evidence"]
    rounded_box(c, x + 21, y + h - 91, 92, 38, fill=BLUE_LIGHT, stroke=BLUE, radius=8, width=1.1)
    rounded_box(c, x + 129, y + h - 91, 92, 38, fill=GREEN_LIGHT, stroke=GREEN, radius=8, width=1.1)
    centered_lines(c, ("Најкратки", "патеки"), x + 67, y + h - 72, size=9.4, leading=10.2, font="DiagramBold")
    centered_lines(c, ("PCST",), x + 175, y + h - 72, size=10.5, font="DiagramBold")
    arrow(c, x + 67, y + h - 98, x + 105, y + 192, color=LINE, width=1.3, head=6)
    arrow(c, x + 175, y + h - 98, x + 133, y + 192, color=LINE, width=1.3, head=6)
    draw_consistent_graph(c, x + 29, y + 61, scale=1.04, show_candidates=True, fade_unselected=True)
    centered_lines(c, ("Доказен подграф",), x + w / 2, y + 35, size=9.5, color=MUTED)

    x, y, w, h = panels["llm"]
    question_symbol(c, x + 14, y + h - 73, w=42, h=38, show_label=False)
    c.setFillColor(MUTED)
    c.setFont("DiagramBold", 16)
    c.drawCentredString(x + 76, y + h - 61, "+")
    draw_consistent_graph(c, x + 78, y + h - 96, scale=0.48, show_candidates=True, selected_only=True)
    arrow(c, x + w / 2, y + h - 93, x + w / 2, y + 187, color=PURPLE, width=1.5, head=7)
    document_symbol(c, x + 64, y + 127, w=48, h=55)
    centered_lines(c, ("Текстуализација",), x + w / 2, y + 111, size=9.5, color=MUTED)
    arrow(c, x + w / 2, y + 99, x + w / 2, y + 78, color=PURPLE, width=1.5, head=7)
    rounded_box(c, x + 42, y + 31, 92, 43, fill=white, stroke=PURPLE, radius=9)
    centered_lines(c, ("LLM",), x + w / 2, y + 53, size=16, font="DiagramBold", color=PURPLE)
    centered_lines(c, ("Генерирање на одговор",), x + w / 2, y + 15, size=9.5, color=MUTED)

    x, y, w, h = panels["answer"]
    centered_lines(c, ("Излез",), x + w / 2, y + h - 23, size=11.5, font="DiagramBold")
    answer_x, answer_y = x + w / 2, y + h / 2 - 4
    node(c, answer_x, answer_y, radius=18, fill=GREEN_LIGHT, stroke=GREEN, width=1.7)
    dotted_gold_ring(c, answer_x, answer_y, radius=27)

    for source, target in (
        ("input", "retriever"),
        ("retriever", "candidates"),
        ("candidates", "evidence"),
        ("evidence", "llm"),
        ("llm", "answer"),
    ):
        sx, sy, sw, sh = panels[source]
        tx, ty, tw, th = panels[target]
        arrow(c, sx + sw + 6, sy + sh / 2, tx - 7, ty + th / 2, color=INK, width=2.0, head=8)

    c.showPage()
    c.save()


def draw_information_flow(output: Path) -> None:
    width, height = 1200, 430
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("Тек на информациите низ системот")
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    groups = (
        (15, BLUE_LIGHT, BLUE, "Локален граф"),
        (312, ORANGE_LIGHT, ORANGE, "Ентитети-кандидати"),
        (609, GREEN_LIGHT, GREEN, "Доказен подграф"),
        (906, PURPLE_LIGHT, PURPLE, "Конечен одговор"),
    )
    group_y, group_w, group_h = 75, 279, 285
    inner_y, inner_w, inner_h = 78, 251, 268
    for x, fill, stroke, heading in groups:
        rounded_box(c, x, group_y, group_w, group_h, fill=fill, stroke=stroke, radius=14, width=1.4)
        stage_heading(c, heading, x, group_y + group_h - 25, group_w)

    # The same topology and node roles are used in both thesis figures.
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
        node(c, groups[0][0] + px, py, radius=7.2, fill=HexColor("#F6F8F9"), stroke=HexColor("#DCE3E8"), width=1.1)
    draw_consistent_graph(c, local_x, inner_y + 73, scale=1.14, show_candidates=False)
    centered_lines(
        c,
        ("Локалниот подграф е извадок од Freebase", "и може да не го содржи точниот одговор"),
        groups[0][0] + group_w / 2,
        group_y + 25,
        size=8.6,
        leading=10.5,
        color=MUTED,
    )

    candidate_x = groups[1][0]
    candidate_data = ((1, 151, False), (2, 124, True), (3, 94, False))
    for rank, bar_width, is_gold in candidate_data:
        cy = inner_y + inner_h - 63 - (rank - 1) * 68
        fill = GREEN_LIGHT if is_gold else ORANGE_LIGHT
        stroke = GREEN if is_gold else ORANGE
        node(c, candidate_x + 54, cy, radius=11, fill=fill, stroke=stroke, width=1.4)
        if is_gold:
            dotted_gold_ring(c, candidate_x + 54, cy, radius=17)
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
        c.setStrokeColor(INK if selected else HexColor("#DCE3E8"))
        c.setLineWidth(1.8 if selected else 1.1)
        c.line(origin_x + sx * scale, origin_y + sy * scale, origin_x + tx * scale, origin_y + ty * scale)
    for index, (px, py) in enumerate(points):
        x = origin_x + px * scale
        y = origin_y + py * scale
        if index == 1:
            fill, stroke = BLUE_LIGHT, BLUE
        elif index == 4:
            fill, stroke = GREEN_LIGHT, GREEN
        elif index == 3:
            fill, stroke = ORANGE_LIGHT, ORANGE
        elif index == 5:
            fill, stroke = HexColor("#FFF9EF"), HexColor("#EBCB94")
        elif index in selected_nodes:
            fill, stroke = white, MUTED
        else:
            fill, stroke = HexColor("#F6F8F9"), HexColor("#DCE3E8")
        node(c, x, y, radius=8.2, fill=fill, stroke=stroke, width=1.35)
        if index == 4:
            dotted_gold_ring(c, x, y, radius=13.2)

    answer_x = groups[3][0]
    node(c, answer_x + group_w / 2, group_y + group_h / 2 - 5, radius=22, fill=GREEN_LIGHT, stroke=GREEN, width=1.7)
    dotted_gold_ring(c, answer_x + group_w / 2, group_y + group_h / 2 - 5, radius=32)

    for index in range(3):
        x1 = groups[index][0] + group_w + 2
        x2 = groups[index + 1][0] - 4
        arrow(c, x1, 220, x2, 220, color=INK, width=1.8, head=6)

    legend_y = 25
    legend_items = (
        (BLUE_LIGHT, BLUE, "Почетен ентитет", False),
        (ORANGE_LIGHT, ORANGE, "Ентитет-кандидат", False),
        (white, MUTED, "Посреден ентитет", False),
        (GREEN_LIGHT, GREEN, "Точен одговор", True),
    )
    starts = (245, 445, 660, 865)
    for start, (fill, stroke, label, ring) in zip(starts, legend_items, strict=True):
        node(c, start, legend_y, radius=6.5, fill=fill, stroke=stroke, width=1.1)
        if ring:
            dotted_gold_ring(c, start, legend_y, radius=10)
        c.setFillColor(INK)
        c.setFont("Diagram", 9.5)
        c.drawString(start + 13, legend_y - 3, label)

    c.showPage()
    c.save()


def render_png(pdf_path: Path, *, dpi: int = PNG_DPI) -> Path:
    """Rasterize a single-page PDF beside its source at publication quality."""
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError(
            "pdftoppm is required to generate PNG architecture figures. "
            "Install Poppler and rerun the script."
        )
    output_path = pdf_path.with_suffix(".png")
    subprocess.run(
        [
            renderer,
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(pdf_path),
            str(output_path.with_suffix("")),
        ],
        check=True,
    )
    if not output_path.is_file():
        raise RuntimeError(f"PNG renderer did not create {output_path}.")
    return output_path


def main() -> None:
    register_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_paths = (
        FIGURES / "system_overview.pdf",
        FIGURES / "information_flow.pdf",
    )
    draw_system_overview(pdf_paths[0])
    draw_information_flow(pdf_paths[1])
    output_paths = [*pdf_paths, *(render_png(path) for path in pdf_paths)]
    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
