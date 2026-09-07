#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "../..");
const RUNTIME_NODE_MODULES = process.env.RUNTIME_NODE_MODULES;
const SKILL_DIR = process.env.SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
const LIBREOFFICE_BIN = process.env.LIBREOFFICE_BIN;

for (const [name, value] of Object.entries({
  RUNTIME_NODE_MODULES,
  SKILL_DIR,
  RUNTIME_PYTHON,
  LIBREOFFICE_BIN,
})) {
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must be set to an absolute path.`);
  }
}

const artifactToolPath = path.join(
  RUNTIME_NODE_MODULES,
  "@oai/artifact-tool/dist/artifact_tool.mjs",
);
const { Presentation, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);
const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);
const require = createRequire(import.meta.url);
const { PDFDocument } = require(path.join(RUNTIME_NODE_MODULES, "pdf-lib"));

const OUTPUT_ROOT = path.join(ROOT, "metadata/figures/presentation");
const BUILD_ROOT = path.join(ROOT, ".codex-build/presentation-visuals");
const CANDIDATE_PPTX = path.join(BUILD_ROOT, "candidate.pptx");
const FINAL_PPTX = path.join(OUTPUT_ROOT, "presentation_visual_assets.pptx");
const COMBINED_PDF = path.join(BUILD_ROOT, "presentation_visual_assets.pdf");

const W = 1280;
const H = 720;
const FONT = "Arial";

const C = Object.freeze({
  ink: "#263746",
  muted: "#617080",
  line: "#A9B4BE",
  inactive: "#DDE3E8",
  light: "#F7F9FB",
  blue: "#4F86C6",
  blueLight: "#EAF2FB",
  candidate: "#E5A23C",
  candidateLight: "#FFF4DF",
  retrieval: "#D97942",
  retrievalLight: "#FBEDE5",
  gold: "#4E9B69",
  goldLight: "#EAF6EE",
  purple: "#8064A2",
  purpleLight: "#F1ECF7",
  teal: "#398D84",
  tealLight: "#E8F5F3",
  white: "#FFFFFF",
});

const OUTPUTS = [
  ["problem", "two_hop_example"],
  ["architectures", "architecture_overview"],
  ["architectures", "graphsage_comparison"],
  ["architectures", "rgcn_hgt_comparison"],
  ["architectures", "rearev_reasoning_cycle"],
  ["architectures", "nbfnet_path_propagation"],
  ["evidence", "evidence_strategy_comparison"],
  ["evidence", "information_loss_pipeline"],
];

function shape(slide, geometry, left, top, width, height, fill, stroke, lineWidth = 2) {
  return slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: stroke, width: lineWidth },
  });
}

function text(slide, value, left, top, width, height, options = {}) {
  const item = shape(slide, "textbox", left, top, width, height, "none", "none", 0);
  item.text = value;
  item.text.style = {
    typeface: FONT,
    fontSize: options.size ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.align ?? "center",
    verticalAlignment: options.vAlign ?? "middle",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { top: 2, right: 4, bottom: 2, left: 4 },
  };
  return item;
}

function box(slide, value, left, top, width, height, options = {}) {
  const item = shape(
    slide,
    options.geometry ?? "roundRect",
    left,
    top,
    width,
    height,
    options.fill ?? C.white,
    options.stroke ?? C.line,
    options.lineWidth ?? 2,
  );
  item.borderRadius = options.radius ?? 14;
  if (value) {
    item.text = value;
    item.text.style = {
      typeface: FONT,
      fontSize: options.size ?? 22,
      bold: options.bold ?? false,
      color: options.color ?? C.ink,
      alignment: options.align ?? "center",
      verticalAlignment: "middle",
      autoFit: "shrinkText",
      insets: { top: 5, right: 8, bottom: 5, left: 8 },
    };
  }
  return item;
}

function node(slide, x, y, options = {}) {
  const radius = options.radius ?? 19;
  const item = shape(
    slide,
    "ellipse",
    x - radius,
    y - radius,
    radius * 2,
    radius * 2,
    options.fill ?? C.white,
    options.stroke ?? C.muted,
    options.lineWidth ?? 2.5,
  );
  if (options.gold) {
    const ring = shape(
      slide,
      "ellipse",
      x - radius - 9,
      y - radius - 9,
      (radius + 9) * 2,
      (radius + 9) * 2,
      "none",
      C.gold,
      2.5,
    );
    ring.line = { style: "dotted", fill: C.gold, width: 2.5 };
  }
  if (options.label) {
    text(
      slide,
      options.label,
      x - (options.labelWidth ?? 100) / 2,
      y + radius + (options.labelOffset ?? 7),
      options.labelWidth ?? 100,
      options.labelHeight ?? 36,
      { size: options.labelSize ?? 18, color: options.labelColor ?? C.ink },
    );
  }
  return item;
}

function link(slide, from, to, options = {}) {
  return slide.shapes.connect(from, to, {
    kind: options.kind ?? "straight",
    fromSide: options.fromSide,
    toSide: options.toSide,
    line: {
      style: options.style ?? "solid",
      fill: options.color ?? C.ink,
      width: options.width ?? 2.5,
    },
    tail: options.arrow === false
      ? { type: "none" }
      : { type: options.head ?? "triangle", width: "sm", length: "sm" },
  });
}

function relationLabel(slide, value, left, top, width, options = {}) {
  return text(slide, value, left, top, width, options.height ?? 28, {
    size: options.size ?? 16,
    color: options.color ?? C.muted,
    bold: options.bold ?? false,
  });
}

function sectionLabel(slide, value, left, top, width, color = C.ink) {
  text(slide, value, left, top, width, 42, { size: 27, bold: true, color });
}

function arrowBetween(slide, from, to, options = {}) {
  return link(slide, from, to, {
    fromSide: options.fromSide ?? "right",
    toSide: options.toSide ?? "left",
    kind: options.kind ?? "straight",
    color: options.color ?? C.ink,
    width: options.width ?? 3,
  });
}

function addFadedDistractorGraph(slide) {
  const d1 = node(slide, 105, 385, { radius: 13, fill: C.light, stroke: C.inactive });
  const d2 = node(slide, 155, 500, { radius: 13, fill: C.light, stroke: C.inactive });
  const d3 = node(slide, 335, 545, { radius: 13, fill: C.light, stroke: C.inactive });
  const d4 = node(slide, 1080, 470, { radius: 13, fill: C.light, stroke: C.inactive });
  const d5 = node(slide, 1165, 380, { radius: 13, fill: C.light, stroke: C.inactive });
  link(slide, d1, d2, { color: C.inactive, width: 1.5, arrow: false });
  link(slide, d2, d3, { color: C.inactive, width: 1.5, arrow: false });
  link(slide, d4, d5, { color: C.inactive, width: 1.5, arrow: false });
  return { d1, d2, d3, d4, d5 };
}

function twoHopExample(slide) {
  addFadedDistractorGraph(slide);

  const q = box(
    slide,
    "Who did Viggo Mortensen play in The Lord of the Rings?",
    150,
    42,
    980,
    82,
    { fill: C.blueLight, stroke: C.blue, size: 25, bold: true, radius: 18 },
  );
  text(slide, "Прашање од WebQSP", 450, 8, 380, 30, {
    size: 16,
    bold: true,
    color: C.blue,
  });

  const seed = node(slide, 250, 340, {
    radius: 27,
    fill: C.blueLight,
    stroke: C.blue,
    label: "Viggo Mortensen",
    labelWidth: 210,
    labelSize: 21,
  });
  const middle = node(slide, 620, 340, {
    radius: 24,
    fill: C.white,
    stroke: C.muted,
    label: "настап / улога",
    labelWidth: 180,
    labelSize: 20,
  });
  const answer = node(slide, 1000, 340, {
    radius: 27,
    fill: C.goldLight,
    stroke: C.gold,
    gold: true,
    label: "Aragorn",
    labelWidth: 140,
    labelSize: 22,
    labelColor: C.gold,
  });
  const film = node(slide, 620, 570, {
    radius: 22,
    fill: C.light,
    stroke: C.muted,
    label: "The Lord of the Rings",
    labelWidth: 250,
    labelSize: 20,
  });

  link(slide, seed, middle, { color: C.ink, width: 3 });
  link(slide, middle, answer, { color: C.ink, width: 3 });
  link(slide, middle, film, {
    color: C.teal,
    width: 2.5,
    fromSide: "bottom",
    toSide: "top",
  });
  relationLabel(slide, "film.actor.film", 350, 274, 180, { color: C.ink, size: 18 });
  relationLabel(slide, "film.performance.character", 716, 274, 265, { color: C.ink, size: 18 });
  relationLabel(slide, "film.performance.film", 645, 455, 230, { color: C.teal, size: 17 });

  text(slide, "почетен ентитет", 155, 415, 190, 30, { size: 16, color: C.blue });
  text(slide, "точен одговор", 910, 415, 180, 30, { size: 16, color: C.gold });
  text(slide, "две последователни релации", 445, 655, 350, 34, {
    size: 21,
    bold: true,
    color: C.ink,
  });
  q.bringToFront();
}

function miniNeighborhood(slide, cx, cy, color) {
  const center = node(slide, cx, cy, { radius: 17, fill: C.white, stroke: color });
  const neighbors = [
    node(slide, cx - 66, cy - 54, { radius: 12, fill: C.light, stroke: C.muted }),
    node(slide, cx - 80, cy + 45, { radius: 12, fill: C.light, stroke: C.muted }),
    node(slide, cx + 74, cy + 6, { radius: 12, fill: C.light, stroke: C.muted }),
  ];
  for (const n of neighbors) link(slide, n, center, { color, width: 2, arrow: false });
}

function miniRelational(slide, cx, cy) {
  const center = node(slide, cx, cy, { radius: 17, fill: C.white, stroke: C.retrieval });
  const specs = [
    [cx - 78, cy - 42, C.blue],
    [cx - 78, cy + 48, C.candidate],
    [cx + 75, cy - 10, C.teal],
  ];
  for (const [x, y, color] of specs) {
    const n = node(slide, x, y, { radius: 12, fill: C.light, stroke: color });
    link(slide, n, center, { color, width: 4, arrow: false });
  }
}

function miniAdaptive(slide, cx, cy) {
  const inst = box(slide, "i₁  i₂", cx - 55, cy - 82, 110, 42, {
    fill: C.purpleLight,
    stroke: C.purple,
    size: 21,
    bold: true,
  });
  const graph = box(slide, "GNN", cx - 55, cy + 3, 110, 55, {
    fill: C.retrievalLight,
    stroke: C.retrieval,
    size: 22,
    bold: true,
    color: C.retrieval,
  });
  link(slide, inst, graph, { fromSide: "bottom", toSide: "top", color: C.purple, width: 2.5 });
  link(slide, graph, inst, { fromSide: "right", toSide: "right", kind: "curved", color: C.teal, width: 2.5 });
}

function miniPath(slide, cx, cy) {
  const n1 = node(slide, cx - 92, cy, { radius: 13, fill: C.blueLight, stroke: C.blue });
  const n2 = node(slide, cx - 28, cy - 38, { radius: 12, fill: C.white, stroke: C.muted });
  const n3 = node(slide, cx + 36, cy + 5, { radius: 12, fill: C.white, stroke: C.muted });
  const n4 = node(slide, cx + 98, cy - 38, { radius: 13, fill: C.goldLight, stroke: C.gold, gold: true });
  link(slide, n1, n2, { color: C.teal, width: 4 });
  link(slide, n2, n3, { color: C.teal, width: 4 });
  link(slide, n3, n4, { color: C.teal, width: 4 });
}

function architectureOverview(slide) {
  const centers = [165, 475, 790, 1100];
  const anchors = centers.map((x) => shape(slide, "ellipse", x - 7, 341, 14, 14, C.line, C.line, 0));
  for (let i = 0; i < anchors.length - 1; i += 1) {
    arrowBetween(slide, anchors[i], anchors[i + 1], { color: C.line, width: 3 });
  }
  miniNeighborhood(slide, centers[0], 270, C.blue);
  miniRelational(slide, centers[1], 270);
  miniAdaptive(slide, centers[2], 270);
  miniPath(slide, centers[3], 300);

  const labels = [
    ["Агрегација на\nсоседството", "GraphSAGE", C.blue],
    ["Релациско\nмоделирање", "Advance GraphSAGE\nR-GCN · HGT", C.retrieval],
    ["Приспособливо\nрасудување", "ReaRev", C.purple],
    ["Расудување по\nрелациски патеки", "NBFNet", C.teal],
  ];
  labels.forEach(([heading, models, color], index) => {
    text(slide, heading, centers[index] - 130, 420, 260, 72, {
      size: 24,
      bold: true,
      color,
    });
    text(slide, models, centers[index] - 130, 500, 260, 72, {
      size: 20,
      color: C.ink,
    });
  });
}

function aggregationGraph(slide, cx, cy, advanced = false) {
  const target = node(slide, cx, cy, { radius: 23, fill: C.white, stroke: advanced ? C.retrieval : C.blue });
  const positions = [
    [cx - 160, cy - 105, C.blue, "r₁"],
    [cx - 185, cy + 40, C.candidate, "r₂"],
    [cx + 15, cy + 135, C.teal, "r₃"],
  ];
  positions.forEach(([x, y, color, label], index) => {
    const n = node(slide, x, y, { radius: 18, fill: C.light, stroke: color });
    link(slide, n, target, {
      color,
      width: advanced ? 3 + index : 2 + index,
      arrow: true,
    });
    if (advanced) {
      link(slide, target, n, {
        color: C.inactive,
        width: 1.5,
        style: "dashed",
        arrow: true,
      });
    }
    relationLabel(slide, label, (x + cx) / 2 - 20, (y + cy) / 2 - 28, 40, {
      size: 17,
      bold: true,
      color,
    });
  });
  const score = box(slide, "оценка", cx + 90, cy - 26, 105, 52, {
    fill: advanced ? C.retrievalLight : C.blueLight,
    stroke: advanced ? C.retrieval : C.blue,
    size: 18,
    bold: true,
  });
  link(slide, target, score, { fromSide: "right", toSide: "left", color: advanced ? C.retrieval : C.blue, width: 3 });
  if (advanced) {
    const residual = box(slide, "residual", cx - 35, cy + 190, 120, 38, {
      fill: C.light,
      stroke: C.line,
      size: 16,
      color: C.muted,
    });
    link(slide, residual, score, { fromSide: "right", toSide: "bottom", kind: "elbow", color: C.line, width: 2 });
  }
}

function graphsageComparison(slide) {
  sectionLabel(slide, "GraphSAGE", 140, 45, 420, C.blue);
  sectionLabel(slide, "Advance GraphSAGE", 720, 45, 420, C.retrieval);
  shape(slide, "line", 638, 55, 2, 585, "none", C.inactive, 2);

  const q1 = box(slide, "прашање  q", 205, 105, 180, 52, {
    fill: C.blueLight,
    stroke: C.blue,
    size: 20,
    bold: true,
  });
  const cos = box(slide, "cos(q, r)", 405, 105, 150, 52, {
    fill: C.light,
    stroke: C.line,
    size: 20,
    bold: true,
  });
  arrowBetween(slide, q1, cos, { color: C.blue, width: 2.5 });
  aggregationGraph(slide, 340, 350, false);
  text(slide, "Фиксно семантичко пондерирање", 145, 600, 400, 38, {
    size: 22,
    bold: true,
    color: C.blue,
  });

  const q2 = box(slide, "прашање  q", 760, 105, 180, 52, {
    fill: C.blueLight,
    stroke: C.blue,
    size: 20,
    bold: true,
  });
  const gate = box(slide, "научена порта  α(q, r)", 955, 105, 230, 52, {
    fill: C.retrievalLight,
    stroke: C.retrieval,
    size: 19,
    bold: true,
    color: C.retrieval,
  });
  arrowBetween(slide, q2, gate, { color: C.retrieval, width: 2.5 });
  aggregationGraph(slide, 915, 350, true);
  text(slide, "Пондерирање научено за конкретното прашање", 705, 600, 450, 38, {
    size: 22,
    bold: true,
    color: C.retrieval,
  });
}

function rgcnHalf(slide, cx, cy) {
  const target = node(slide, cx + 135, cy, { radius: 24, fill: C.white, stroke: C.retrieval });
  const specs = [
    [cx - 150, cy - 110, C.blue, "r₁", "Wᵣ₁"],
    [cx - 170, cy + 25, C.candidate, "r₂", "Wᵣ₂"],
    [cx - 80, cy + 150, C.teal, "r₃", "Wᵣ₃"],
  ];
  specs.forEach(([x, y, color, rel, transform], idx) => {
    const n = node(slide, x, y, { radius: 18, fill: C.light, stroke: color });
    const tx = cx - 10;
    const ty = cy - 86 + idx * 82;
    const op = box(slide, transform, tx, ty, 74, 42, {
      fill: C.white,
      stroke: color,
      size: 18,
      bold: true,
      color,
    });
    link(slide, n, op, { color, width: 2.5 });
    link(slide, op, target, { color, width: 3 });
    relationLabel(slide, rel, x + 35, y - 22, 44, { size: 17, bold: true, color });
  });
}

function hgtHalf(slide, cx, cy) {
  const target = node(slide, cx + 105, cy, { radius: 24, fill: C.white, stroke: C.purple });
  const specs = [
    [cx - 150, cy - 115, C.blue, "r₁", 5.5, "α₁"],
    [cx - 170, cy + 30, C.candidate, "r₂", 2, "α₂"],
    [cx - 75, cy + 150, C.teal, "r₃", 4, "α₃"],
  ];
  specs.forEach(([x, y, color, rel, width, alpha]) => {
    const n = node(slide, x, y, { radius: 18, fill: C.light, stroke: color });
    link(slide, n, target, { color, width });
    relationLabel(slide, rel, x + 35, y - 22, 44, { size: 17, bold: true, color });
    relationLabel(slide, alpha, (x + cx + 105) / 2 - 18, (y + cy) / 2 - 35, 46, {
      size: 18,
      bold: true,
      color,
    });
  });
  box(slide, "multi-head attention", cx - 40, cy + 205, 250, 45, {
    fill: C.purpleLight,
    stroke: C.purple,
    size: 18,
    bold: true,
    color: C.purple,
  });
}

function rgcnHgtComparison(slide) {
  sectionLabel(slide, "R-GCN", 145, 45, 420, C.retrieval);
  sectionLabel(slide, "HGT", 730, 45, 420, C.purple);
  shape(slide, "line", 638, 55, 2, 585, "none", C.inactive, 2);
  rgcnHalf(slide, 350, 335);
  hgtHalf(slide, 930, 335);
  text(slide, "Трансформација според типот на релацијата", 110, 610, 460, 40, {
    size: 22,
    bold: true,
    color: C.retrieval,
  });
  text(slide, "Релациски зависна тежина на вниманието", 705, 610, 465, 40, {
    size: 22,
    bold: true,
    color: C.purple,
  });
}

function rearevCycle(slide) {
  const question = box(slide, "Прашање", 70, 275, 165, 70, {
    fill: C.blueLight,
    stroke: C.blue,
    size: 23,
    bold: true,
    color: C.blue,
  });
  const instructions = box(slide, "Инструкции\ni₁   i₂   i₃", 320, 100, 220, 105, {
    fill: C.purpleLight,
    stroke: C.purple,
    size: 23,
    bold: true,
    color: C.purple,
  });
  const reasoning = box(slide, "Расудување\nнад графот", 535, 285, 220, 105, {
    fill: C.retrievalLight,
    stroke: C.retrieval,
    size: 23,
    bold: true,
    color: C.retrieval,
  });
  const collected = box(slide, "Собрана\nинформација", 820, 465, 210, 100, {
    fill: C.tealLight,
    stroke: C.teal,
    size: 22,
    bold: true,
    color: C.teal,
  });
  const revision = box(slide, "Ревизија на\nинструкциите", 990, 120, 220, 105, {
    fill: C.purpleLight,
    stroke: C.purple,
    size: 22,
    bold: true,
    color: C.purple,
  });

  arrowBetween(slide, question, instructions, { fromSide: "right", toSide: "left", kind: "elbow", color: C.blue });
  arrowBetween(slide, instructions, reasoning, { fromSide: "bottom", toSide: "top", kind: "elbow", color: C.purple });
  arrowBetween(slide, reasoning, collected, { fromSide: "right", toSide: "left", kind: "elbow", color: C.retrieval });
  arrowBetween(slide, collected, revision, { fromSide: "top", toSide: "bottom", kind: "elbow", color: C.teal });
  arrowBetween(slide, revision, instructions, { fromSide: "left", toSide: "right", kind: "elbow", color: C.purple });

  const seed = node(slide, 585, 515, { radius: 15, fill: C.blueLight, stroke: C.blue });
  const mid = node(slide, 665, 545, { radius: 13, fill: C.white, stroke: C.muted });
  const gold = node(slide, 745, 500, { radius: 15, fill: C.goldLight, stroke: C.gold, gold: true });
  link(slide, seed, mid, { color: C.ink, width: 2.5 });
  link(slide, mid, gold, { color: C.ink, width: 2.5 });
  text(slide, "итерација  t", 570, 425, 160, 32, { size: 18, bold: true, color: C.retrieval });
  text(slide, "следна итерација", 550, 52, 185, 34, { size: 18, bold: true, color: C.purple });
  text(slide, "Инструкциите се менуваат според информацијата откриена во графот", 225, 635, 830, 38, {
    size: 24,
    bold: true,
    color: C.ink,
  });
}

function nbfnetPropagation(slide) {
  const xs = [150, 430, 710, 990];
  ["l = 0", "l = 1", "l = 2", "Оценки"].forEach((label, i) => {
    text(slide, label, xs[i] - 85, 55, 170, 36, { size: 22, bold: true, color: i === 3 ? C.candidate : C.teal });
  });

  const seed = node(slide, xs[0], 340, {
    radius: 25,
    fill: C.blueLight,
    stroke: C.blue,
    label: "почетен ентитет",
    labelWidth: 190,
    labelSize: 19,
  });
  box(slide, "q", 95, 170, 110, 60, { fill: C.blueLight, stroke: C.blue, size: 26, bold: true, color: C.blue });
  text(slide, "иницијализација", 75, 235, 150, 32, { size: 17, color: C.blue });

  const a1 = node(slide, xs[1], 235, { radius: 18, fill: C.white, stroke: C.teal });
  const b1 = node(slide, xs[1], 445, { radius: 18, fill: C.light, stroke: C.inactive });
  const a2 = node(slide, xs[2], 285, { radius: 18, fill: C.white, stroke: C.teal });
  const b2 = node(slide, xs[2], 470, { radius: 18, fill: C.light, stroke: C.inactive });
  const gold = node(slide, xs[3], 260, {
    radius: 25,
    fill: C.goldLight,
    stroke: C.gold,
    gold: true,
    label: "точен одговор",
    labelWidth: 170,
    labelSize: 19,
    labelColor: C.gold,
  });
  const candidate = node(slide, xs[3], 470, {
    radius: 23,
    fill: C.candidateLight,
    stroke: C.candidate,
    label: "друг кандидат",
    labelWidth: 170,
    labelSize: 19,
    labelColor: C.candidate,
  });

  link(slide, seed, a1, { color: C.teal, width: 5 });
  link(slide, a1, a2, { color: C.teal, width: 5 });
  link(slide, a2, gold, { color: C.teal, width: 5 });
  link(slide, seed, b1, { color: C.inactive, width: 2 });
  link(slide, b1, b2, { color: C.inactive, width: 2 });
  link(slide, b2, candidate, { color: C.inactive, width: 2 });
  link(slide, a1, b2, { color: C.line, width: 1.5, style: "dashed" });

  relationLabel(slide, "wᵣ₁(q)", 255, 225, 120, { size: 19, color: C.teal, bold: true });
  relationLabel(slide, "wᵣ₂(q)", 535, 205, 120, { size: 19, color: C.teal, bold: true });
  relationLabel(slide, "wᵣ₃(q)", 815, 190, 120, { size: 19, color: C.teal, bold: true });

  box(slide, "висока", 1070, 230, 130, 50, { fill: C.goldLight, stroke: C.gold, size: 19, bold: true, color: C.gold });
  box(slide, "ниска", 1070, 440, 130, 50, { fill: C.light, stroke: C.line, size: 19, bold: true, color: C.muted });
  text(slide, "Патеките и релациите се условени од прашањето", 260, 625, 760, 40, {
    size: 25,
    bold: true,
    color: C.ink,
  });
}

function evidenceGraph(slide, offsetX, selectedEdges, options = {}) {
  const pts = [
    [offsetX + 60, 355],
    [offsetX + 175, 235],
    [offsetX + 290, 330],
    [offsetX + 410, 205],
    [offsetX + 520, 320],
    [offsetX + 385, 485],
    [offsetX + 175, 500],
  ];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 0], [1, 6]];
  const nodes = pts.map(([x, y], i) => {
    let fill = C.white;
    let stroke = C.muted;
    let gold = false;
    if (i === 0) [fill, stroke] = [C.blueLight, C.blue];
    if ([3, 4, 5].includes(i)) [fill, stroke] = [C.candidateLight, C.candidate];
    if (i === 4) {
      fill = C.goldLight;
      stroke = C.gold;
      gold = true;
    }
    if (options.fadeNodes?.includes(i)) {
      fill = C.light;
      stroke = C.inactive;
      gold = false;
    }
    return node(slide, x, y, { radius: 17, fill, stroke, gold });
  });
  edges.forEach(([a, b], idx) => {
    const active = selectedEdges.includes(idx);
    link(slide, nodes[a], nodes[b], {
      color: active ? C.ink : C.inactive,
      width: active ? 3.5 : 1.5,
      arrow: false,
    });
  });
  return nodes;
}

function evidenceComparison(slide) {
  sectionLabel(slide, "Унија на најкратки патеки", 75, 48, 500, C.blue);
  sectionLabel(slide, "PCST", 760, 48, 420, C.teal);
  shape(slide, "line", 638, 55, 2, 585, "none", C.inactive, 2);

  evidenceGraph(slide, 20, [0, 1, 2, 3, 4, 5, 6, 7]);
  evidenceGraph(slide, 665, [0, 1, 2, 3], { fadeNodes: [5, 6] });

  text(slide, "Се задржува патека до секој достижен кандидат", 80, 575, 500, 55, {
    size: 22,
    bold: true,
    color: C.blue,
  });
  text(slide, "Наградите на јазлите се споредуваат со трошоците на рабовите", 700, 565, 520, 70, {
    size: 21,
    bold: true,
    color: C.teal,
  });
  box(slide, "повеќе тројки", 205, 645, 230, 42, {
    fill: C.blueLight,
    stroke: C.blue,
    size: 18,
    bold: true,
    color: C.blue,
  });
  box(slide, "компактен подграф", 840, 645, 265, 42, {
    fill: C.tealLight,
    stroke: C.teal,
    size: 18,
    bold: true,
    color: C.teal,
  });
}

function smallLocalGraph(slide, cx, cy) {
  const pts = [[cx - 80, cy], [cx - 15, cy - 55], [cx + 55, cy - 10], [cx + 100, cy + 55], [cx, cy + 75]];
  const nodes = pts.map(([x, y], i) => node(slide, x, y, {
    radius: 12,
    fill: i === 0 ? C.blueLight : i === 3 ? C.goldLight : C.white,
    stroke: i === 0 ? C.blue : i === 3 ? C.gold : C.muted,
    gold: i === 3,
  }));
  [[0, 1], [1, 2], [2, 3], [2, 4], [4, 0]].forEach(([a, b]) => link(slide, nodes[a], nodes[b], { color: C.ink, width: 2, arrow: false }));
}

function candidateList(slide, left, top) {
  const ys = [top, top + 68, top + 136];
  ys.forEach((y, i) => {
    node(slide, left + 22, y + 18, {
      radius: 13,
      fill: i === 1 ? C.goldLight : C.candidateLight,
      stroke: i === 1 ? C.gold : C.candidate,
      gold: i === 1,
    });
    shape(slide, "roundRect", left + 55, y + 7, 125 - i * 18, 22, i === 1 ? C.goldLight : C.candidateLight, "none", 0);
  });
}

function smallEvidence(slide, cx, cy) {
  const seed = node(slide, cx - 90, cy + 40, { radius: 13, fill: C.blueLight, stroke: C.blue });
  const mid = node(slide, cx, cy - 25, { radius: 12, fill: C.white, stroke: C.muted });
  const gold = node(slide, cx + 90, cy + 25, { radius: 13, fill: C.goldLight, stroke: C.gold, gold: true });
  link(slide, seed, mid, { color: C.ink, width: 2.8 });
  link(slide, mid, gold, { color: C.inactive, width: 2, style: "dashed" });
  return { seed, mid, gold };
}

function informationLoss(slide) {
  const stageXs = [165, 470, 785, 1090];
  const headings = [
    ["Локален граф", C.blue],
    ["Пребарани\nентитети-кандидати", C.retrieval],
    ["Доказен контекст", C.teal],
    ["Одговор од LLM", C.purple],
  ];
  headings.forEach(([label, color], i) => {
    text(slide, label, stageXs[i] - 125, 60, 250, 70, { size: 24, bold: true, color });
  });

  const anchors = stageXs.map((x) => shape(slide, "ellipse", x - 5, 168, 10, 10, C.line, C.line, 0));
  for (let i = 0; i < anchors.length - 1; i += 1) arrowBetween(slide, anchors[i], anchors[i + 1], { color: C.line, width: 3 });

  smallLocalGraph(slide, stageXs[0], 330);
  candidateList(slide, stageXs[1] - 85, 240);
  smallEvidence(slide, stageXs[2], 330);
  node(slide, stageXs[3], 325, {
    radius: 24,
    fill: C.candidateLight,
    stroke: C.candidate,
    label: "вратен одговор",
    labelWidth: 190,
    labelSize: 19,
  });

  text(slide, "точниот ентитет е присутен", stageXs[2] - 145, 500, 290, 42, {
    size: 21,
    bold: true,
    color: C.gold,
  });
  text(slide, "но не е вратен", stageXs[3] - 110, 500, 220, 42, {
    size: 21,
    bold: true,
    color: C.purple,
  });
  text(slide, "Загубата може да настане и по пребарувањето", 290, 620, 700, 42, {
    size: 26,
    bold: true,
    color: C.ink,
  });
}

const builders = [
  twoHopExample,
  architectureOverview,
  graphsageComparison,
  rgcnHgtComparison,
  rearevCycle,
  nbfnetPropagation,
  evidenceComparison,
  informationLoss,
];

await fs.mkdir(BUILD_ROOT, { recursive: true });
for (const [dir] of OUTPUTS) {
  await fs.mkdir(path.join(OUTPUT_ROOT, dir), { recursive: true });
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });
const slides = [];
for (const build of builders) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  build(slide);
  slides.push(slide);
}

await (await PresentationFile.exportPptx(presentation)).save(CANDIDATE_PPTX);
await fs.rm(FINAL_PPTX, { force: true });
await fs.rm(
  path.join(BUILD_ROOT, "presentation_visual_assets.validation.json"),
  { force: true },
);
await finalizePresentation({
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: ROOT,
  candidatePath: CANDIDATE_PPTX,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000"],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD_ROOT, "presentation_visual_assets.validation.json"),
});

for (let index = 0; index < slides.length; index += 1) {
  const [directory, filename] = OUTPUTS[index];
  const preview = await presentation.export({ slide: slides[index], format: "png", scale: 3.125 });
  await fs.writeFile(
    path.join(OUTPUT_ROOT, directory, `${filename}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  const layout = await presentation.export({ slide: slides[index], format: "layout" });
  await fs.writeFile(path.join(BUILD_ROOT, `${filename}.layout.json`), await layout.text());
}

const conversion = spawnSync(
  LIBREOFFICE_BIN,
  ["--headless", "--convert-to", "pdf", "--outdir", BUILD_ROOT, FINAL_PPTX],
  { encoding: "utf8" },
);
if (conversion.status !== 0) {
  throw new Error(`LibreOffice PDF conversion failed:\n${conversion.stdout}\n${conversion.stderr}`);
}
const producedPdf = path.join(BUILD_ROOT, "presentation_visual_assets.pdf");
if (producedPdf !== COMBINED_PDF) {
  throw new Error(`Unexpected combined PDF path: ${producedPdf}`);
}

const sourcePdf = await PDFDocument.load(await fs.readFile(COMBINED_PDF));
if (sourcePdf.getPageCount() !== OUTPUTS.length) {
  throw new Error(`Expected ${OUTPUTS.length} PDF pages, found ${sourcePdf.getPageCount()}.`);
}
for (let index = 0; index < OUTPUTS.length; index += 1) {
  const [directory, filename] = OUTPUTS[index];
  const target = await PDFDocument.create();
  const [page] = await target.copyPages(sourcePdf, [index]);
  target.addPage(page);
  await fs.writeFile(
    path.join(OUTPUT_ROOT, directory, `${filename}.pdf`),
    await target.save(),
  );
}

console.log(`Generated ${OUTPUTS.length} presentation visuals in ${OUTPUT_ROOT}`);
console.log(`Editable asset library: ${FINAL_PPTX}`);
