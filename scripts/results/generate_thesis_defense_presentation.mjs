#!/usr/bin/env node

import { spawnSync } from "node:child_process";
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

const OUTPUT_DIR = path.join(ROOT, "metadata/presentation");
const BUILD_DIR = path.join(ROOT, ".codex-build/thesis-presentation");
const PREVIEW_DIR = path.join(BUILD_DIR, "previews");
const CANDIDATE_PPTX = path.join(BUILD_DIR, "candidate.pptx");
const FINAL_PPTX = path.join(OUTPUT_DIR, "thesis_defense_presentation.pptx");
const FINAL_PDF = path.join(OUTPUT_DIR, "thesis_defense_presentation.pdf");
const RECEIPT = path.join(BUILD_DIR, "thesis_defense_presentation.validation.json");

const W = 1280;
const H = 720;
const FONT = "Arial";

const C = Object.freeze({
  ink: "#263746",
  muted: "#617080",
  line: "#A9B4BE",
  inactive: "#DDE3E8",
  light: "#F7F9FB",
  white: "#FFFFFF",
  blue: "#4F86C6",
  blueLight: "#EAF2FB",
  orange: "#D97942",
  orangeLight: "#FBEDE5",
  candidate: "#E5A23C",
  candidateLight: "#FFF4DF",
  teal: "#398D84",
  tealLight: "#E8F5F3",
  purple: "#8064A2",
  purpleLight: "#F1ECF7",
  green: "#4E9B69",
  greenLight: "#EAF6EE",
});

const FIGURES = Object.freeze({
  example: "metadata/figures/presentation/problem/two_hop_example.png",
  system: "metadata/figures/system_architecture/system_overview.png",
  architectures: "metadata/figures/presentation/architectures/architecture_overview.png",
  graphsage: "metadata/figures/presentation/architectures/graphsage_comparison.png",
  rgcnHgt: "metadata/figures/presentation/architectures/rgcn_hgt_comparison.png",
  rearev: "metadata/figures/presentation/architectures/rearev_reasoning_cycle.png",
  nbfnet: "metadata/figures/presentation/architectures/nbfnet_path_propagation.png",
  architectureResults:
    "metadata/figures/architecture_retrieval/architecture_primary_metrics_hits1_all_metrics.png",
  evidence: "metadata/figures/presentation/evidence/evidence_strategy_comparison.png",
  evidenceResults:
    "metadata/figures/evidence_subgraphs/evidence_pcst_lambda_sensitivity.png",
  llmResults: "metadata/figures/end_to_end_llm/end_to_end_llm_quality_tokens.png",
  informationFlow: "metadata/figures/system_architecture/information_flow.png",
});

const imageBytes = new Map();
for (const [key, relativePath] of Object.entries(FIGURES)) {
  imageBytes.set(key, await fs.readFile(path.join(ROOT, relativePath)));
}

function addShape(slide, geometry, left, top, width, height, options = {}) {
  const item = slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill: options.fill ?? "none",
    line: {
      style: options.lineStyle ?? "solid",
      fill: options.stroke ?? "none",
      width: options.lineWidth ?? 0,
    },
  });
  if (geometry === "roundRect") item.borderRadius = options.radius ?? 12;
  return item;
}

function addText(slide, value, left, top, width, height, options = {}) {
  const item = addShape(slide, "textbox", left, top, width, height, {
    fill: options.fill ?? "none",
    stroke: options.stroke ?? "none",
    lineWidth: options.lineWidth ?? 0,
    radius: options.radius,
  });
  item.text = value;
  item.text.style = {
    typeface: options.typeface ?? FONT,
    fontSize: options.size ?? 20,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    color: options.color ?? C.ink,
    alignment: options.align ?? "left",
    verticalAlignment: options.vAlign ?? "middle",
    autoFit: options.autoFit ?? "shrinkText",
    wrap: "square",
    insets: options.insets ?? { top: 2, right: 4, bottom: 2, left: 4 },
  };
  return item;
}

function addCard(slide, left, top, width, height, options = {}) {
  return addShape(slide, "roundRect", left, top, width, height, {
    fill: options.fill ?? C.white,
    stroke: options.stroke ?? C.inactive,
    lineWidth: options.lineWidth ?? 1.5,
    radius: options.radius ?? 14,
  });
}

function addImage(slide, key, left, top, width, height, options = {}) {
  return slide.images.add({
    blob: imageBytes.get(key),
    contentType: "image/png",
    alt: options.alt ?? key,
    fit: options.fit ?? "contain",
    position: { left, top, width, height },
  });
}

function addHeader(slide, title, number, color, citation = "") {
  addText(slide, title, 58, 22, 1130, 55, {
    size: 31,
    bold: true,
    color: C.ink,
  });
  addShape(slide, "rect", 58, 82, 1164, 4, { fill: color });
  if (citation) {
    addText(slide, citation, 58, 681, 980, 24, {
      size: 11,
      color: C.muted,
      vAlign: "bottom",
    });
  }
  addText(slide, String(number), 1188, 681, 34, 24, {
    size: 12,
    color: C.muted,
    align: "right",
    vAlign: "bottom",
  });
}

function newSlide(title, number, color, citation = "") {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, title, number, color, citation);
  return slide;
}

function addBullet(slide, text, left, top, width, options = {}) {
  addShape(slide, "ellipse", left, top + 11, 7, 7, {
    fill: options.color ?? C.ink,
  });
  return addText(slide, text, left + 18, top, width - 18, options.height ?? 50, {
    size: options.size ?? 19,
    color: options.textColor ?? C.ink,
    vAlign: "top",
  });
}

function addLabel(slide, textValue, left, top, width, color, fill, options = {}) {
  addCard(slide, left, top, width, options.height ?? 38, {
    fill,
    stroke: color,
    lineWidth: options.lineWidth ?? 1.5,
    radius: options.radius ?? 12,
  });
  addText(slide, textValue, left + 8, top + 2, width - 16, (options.height ?? 38) - 4, {
    size: options.size ?? 16,
    bold: options.bold ?? true,
    color,
    align: "center",
  });
}

function addMetric(slide, value, label, left, top, width, color, options = {}) {
  addText(slide, value, left, top, width, 54, {
    size: options.valueSize ?? 31,
    bold: true,
    color,
    align: options.align ?? "center",
  });
  addText(slide, label, left, top + 48, width, options.labelHeight ?? 42, {
    size: options.labelSize ?? 15,
    color: C.muted,
    align: options.align ?? "center",
    vAlign: "top",
  });
}

function addSectionCaption(slide, textValue, left, top, width, color) {
  addText(slide, textValue, left, top, width, 30, {
    size: 15,
    bold: true,
    color,
    vAlign: "bottom",
  });
}

function addEquation(slide, equation, left, top, width, height, options = {}) {
  addCard(slide, left, top, width, height, {
    fill: options.fill ?? C.light,
    stroke: options.stroke ?? C.inactive,
    lineWidth: 1.2,
    radius: 10,
  });
  addText(slide, equation, left + 12, top + 7, width - 24, height - 14, {
    size: options.size ?? 20,
    color: options.color ?? C.ink,
    align: "center",
    typeface: options.typeface ?? FONT,
  });
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

// 1 — Title
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, "Универзитет „Св. Кирил и Методиј“ во Скопје", 80, 48, 1120, 32, {
    size: 16,
    color: C.muted,
    align: "center",
  });
  addText(
    slide,
    "Факултет за информатички науки и компјутерско инженерство",
    80,
    79,
    1120,
    34,
    { size: 17, bold: true, color: C.ink, align: "center" },
  );
  addShape(slide, "rect", 330, 132, 620, 5, { fill: C.blue });
  addText(
    slide,
    "Напредни архитектури на граф невронски мрежи за пребарување и расудување над графови при одговарање на прашања над графови на знаење",
    115,
    184,
    1050,
    212,
    { size: 35, bold: true, color: C.ink, align: "center" },
  );
  addLabel(slide, "ДИПЛОМСКА РАБОТА", 490, 414, 300, C.blue, C.blueLight, {
    height: 42,
    size: 16,
  });
  addText(slide, "Виктор Костадиноски · 226009", 110, 520, 500, 38, {
    size: 21,
    bold: true,
  });
  addText(slide, "Ментор: проф. д-р Соња Гиевска", 670, 520, 500, 38, {
    size: 21,
    bold: true,
    align: "right",
  });
  addShape(slide, "rect", 110, 590, 1060, 1.5, { fill: C.line });
  addText(slide, "Септември 2026", 110, 610, 1060, 30, {
    size: 16,
    color: C.muted,
    align: "center",
  });
}

// 2 — Example
{
  const slide = newSlide("Од прашање до одговор", 2, C.blue);
  addImage(slide, "example", 80, 90, 1120, 590, {
    alt: "Пример со патека од Колосеум до Европа",
  });
}

// 3 — Related work
{
  const slide = newSlide(
    "Сродни пристапи и истражувачка празнина",
    3,
    C.blue,
    "Mavromatis and Karypis, 2025 · He et al., 2024",
  );
  addSectionCaption(slide, "GNN-RAG", 78, 112, 500, C.blue);
  addText(slide, "GNN-пребарување  →  најкратки патеки  →  LLM", 78, 145, 510, 42, {
    size: 21,
    bold: true,
    color: C.blue,
  });
  addBullet(slide, "GNN ги рангира ентитетите-кандидати.", 82, 210, 490);
  addBullet(slide, "Најкратките патеки го формираат доказниот контекст.", 82, 270, 490);

  addShape(slide, "rect", 635, 118, 2, 225, { fill: C.inactive });
  addSectionCaption(slide, "G-Retriever", 685, 112, 500, C.teal);
  addText(slide, "кодер на графови  →  PCST  →  LLM", 685, 145, 510, 42, {
    size: 21,
    bold: true,
    color: C.teal,
  });
  addBullet(slide, "Кодерот на графови ги оценува јазлите.", 689, 210, 485);
  addBullet(slide, "PCST избира компактен доказен подграф.", 689, 270, 485);

  addCard(slide, 78, 385, 1120, 205, {
    fill: C.light,
    stroke: C.blue,
    lineWidth: 2,
  });
  addText(slide, "Истражувачка празнина", 110, 410, 420, 38, {
    size: 24,
    bold: true,
    color: C.blue,
  });
  addText(
    slide,
    "Недостига контролирана споредба на различни GNN-архитектури, стратегии за доказен подграф и јазични модели во една заедничка рамка.",
    110,
    463,
    1035,
    82,
    { size: 24, bold: true, color: C.ink },
  );
}

// 4 — System
{
  const slide = newSlide("Архитектура на системот", 4, C.blue);
  addImage(slide, "system", 45, 105, 1190, 446, {
    alt: "Целосен системски тек",
  });
  addCard(slide, 145, 577, 990, 72, {
    fill: C.blueLight,
    stroke: C.blue,
    lineWidth: 1.6,
  });
  addText(
    slide,
    "Резултатите од секоја фаза се следат одделно за да се утврди каде се губи потребната информација.",
    175,
    589,
    930,
    48,
    { size: 20, bold: true, color: C.ink, align: "center" },
  );
}

// 5 — Research questions
{
  const slide = newSlide("Истражувачки прашања", 5, C.blue);
  const cards = [
    {
      x: 58,
      n: "1",
      title: "Архитектура за пребарување",
      body: "Како изборот на GNN-архитектура влијае врз рангирањето и опфатеноста на точните одговори?",
      color: C.orange,
      fill: C.orangeLight,
    },
    {
      x: 449,
      n: "2",
      title: "Доказен подграф",
      body: "Како конструкцијата на доказниот подграф влијае врз компактноста и зачувувањето на релевантните информации?",
      color: C.teal,
      fill: C.tealLight,
    },
    {
      x: 840,
      n: "3",
      title: "Јазичен модел",
      body: "Како изборот на LLM влијае врз конечната точност и искористувањето на достапниот доказен контекст?",
      color: C.purple,
      fill: C.purpleLight,
    },
  ];
  for (const card of cards) {
    addCard(slide, card.x, 145, 345, 430, {
      fill: card.fill,
      stroke: card.color,
      lineWidth: 2,
      radius: 18,
    });
    addShape(slide, "ellipse", card.x + 127, 180, 90, 90, {
      fill: C.white,
      stroke: card.color,
      lineWidth: 3,
    });
    addText(slide, card.n, card.x + 127, 183, 90, 84, {
      size: 34,
      bold: true,
      color: card.color,
      align: "center",
    });
    addText(slide, card.title, card.x + 30, 300, 285, 70, {
      size: 23,
      bold: true,
      color: card.color,
      align: "center",
    });
    addText(slide, card.body, card.x + 32, 382, 281, 135, {
      size: 19,
      color: C.ink,
      align: "center",
      vAlign: "top",
    });
  }
}

// 6 — Dataset and setup
{
  const slide = newSlide(
    "Податочно множество и експериментална поставеност",
    6,
    C.blue,
    "WebQSP: Yih et al., 2016",
  );
  addCard(slide, 58, 116, 445, 490, {
    fill: C.blueLight,
    stroke: C.blue,
    lineWidth: 1.8,
  });
  addText(slide, "WebQuestionsSP", 88, 140, 385, 48, {
    size: 27,
    bold: true,
    color: C.blue,
  });
  addBullet(slide, "Прашања на природен јазик поврзани со Freebase", 92, 215, 365, {
    height: 64,
  });
  addBullet(slide, "Локален граф за секое прашање", 92, 300, 365);
  addBullet(slide, "Познати почетни ентитети и точни одговори", 92, 365, 365, {
    height: 64,
  });
  addText(slide, "Заедничка евалуација", 88, 465, 385, 36, {
    size: 20,
    bold: true,
    color: C.ink,
  });
  addText(slide, "seeds: 42 · 1337 · 2026\nпраг: 0,7 · 10–15 кандидати", 88, 505, 385, 62, {
    size: 18,
    color: C.muted,
  });

  addText(slide, "Поделба на податоците", 560, 126, 640, 38, {
    size: 23,
    bold: true,
  });
  const metrics = [
    ["2.848", "тренирање", C.orange],
    ["250", "валидација", C.teal],
    ["1.639", "тестирање", C.purple],
  ];
  metrics.forEach(([value, label, color], i) => {
    const x = 560 + i * 210;
    addCard(slide, x, 190, 182, 135, {
      fill: C.white,
      stroke: color,
      lineWidth: 1.8,
    });
    addMetric(slide, value, label, x + 12, 213, 158, color, {
      valueSize: 32,
      labelSize: 16,
    });
  });
  addCard(slide, 560, 370, 602, 170, {
    fill: C.light,
    stroke: C.line,
    lineWidth: 1.4,
  });
  addMetric(slide, "1.889", "обединети примероци за евалуација пред филтрирањето", 625, 396, 472, C.blue, {
    valueSize: 40,
    labelSize: 18,
    labelHeight: 55,
  });
}

// 7 — Architecture overview
{
  const slide = newSlide("Архитектури за пребарување", 7, C.orange);
  addImage(slide, "architectures", 80, 90, 1120, 590, {
    alt: "Преглед на архитектурите за пребарување",
  });
}

// 8 — GraphSAGE
{
  const slide = newSlide(
    "GraphSAGE и Advance GraphSAGE",
    8,
    C.orange,
    "Hamilton et al., 2017",
  );
  addImage(slide, "graphsage", 28, 108, 815, 458, {
    alt: "Споредба на GraphSAGE и Advance GraphSAGE",
  });
  addText(slide, "Клучна разлика", 875, 116, 330, 34, {
    size: 21,
    bold: true,
    color: C.orange,
  });
  addEquation(slide, "α(q,r) = cos(q,r)", 865, 165, 350, 72, {
    fill: C.blueLight,
    stroke: C.blue,
    color: C.blue,
    size: 23,
  });
  addEquation(slide, "α(q,r) = sigmoid(MLP[q̃ ∥ r̃ ∥ q̃⊙r̃])", 865, 258, 350, 102, {
    fill: C.orangeLight,
    stroke: C.orange,
    color: C.orange,
    size: 19,
  });
  addBullet(slide, "Обратни рабови и резидуални врски", 875, 398, 325, {
    size: 17,
    height: 48,
    color: C.orange,
  });
  addBullet(slide, "Нормализација и оценка условена од прашањето", 875, 458, 325, {
    size: 17,
    height: 66,
    color: C.orange,
  });
  addText(slide, "Од фиксно семантичко пондерирање кон научен gate за конкретното прашање.", 88, 596, 1095, 55, {
    size: 21,
    bold: true,
    color: C.ink,
    align: "center",
  });
}

// 9 — R-GCN and HGT
{
  const slide = newSlide(
    "R-GCN и HGT",
    9,
    C.orange,
    "Schlichtkrull et al., 2018 · Hu et al., 2020",
  );
  addImage(slide, "rgcnHgt", 25, 112, 790, 445, {
    alt: "Споредба на R-GCN и HGT",
  });
  addText(slide, "R-GCN", 850, 116, 355, 34, {
    size: 22,
    bold: true,
    color: C.orange,
  });
  addEquation(
    slide,
    "hᵥ⁽ˡ⁺¹⁾ = σ(W₀⁽ˡ⁾hᵥ⁽ˡ⁾ + Σᵣ Σᵤ 1/cᵥ,ᵣ · Wᵣ⁽ˡ⁾hᵤ⁽ˡ⁾)",
    840,
    158,
    380,
    100,
    { fill: C.orangeLight, stroke: C.orange, color: C.orange, size: 18 },
  );
  addBullet(slide, "Различна трансформација за секој тип релација", 850, 281, 355, {
    size: 17,
    height: 60,
    color: C.orange,
  });
  addText(slide, "HGT", 850, 370, 355, 34, {
    size: 22,
    bold: true,
    color: C.purple,
  });
  addEquation(slide, "αᵣ,ʲᵤᵥ = softmax(eᵣ,ʲᵤᵥ)", 840, 412, 380, 76, {
    fill: C.purpleLight,
    stroke: C.purple,
    color: C.purple,
    size: 22,
  });
  addBullet(slide, "Релациски зависен attention за соседите", 850, 510, 355, {
    size: 17,
    height: 58,
    color: C.purple,
  });
  addText(slide, "И двете архитектури ја задржуваат релациската структура, но ја моделираат на различен начин.", 85, 603, 1110, 48, {
    size: 20,
    bold: true,
    align: "center",
  });
}

// 10 — ReaRev
{
  const slide = newSlide(
    "ReaRev",
    10,
    C.orange,
    "Mavromatis and Karypis, 2022",
  );
  addImage(slide, "rearev", 25, 108, 815, 458, {
    alt: "Циклус на расудување во ReaRev",
  });
  addText(slide, "Приспособливо расудување", 870, 116, 340, 42, {
    size: 22,
    bold: true,
    color: C.purple,
  });
  addBullet(slide, "Повеќе инструкции се извлекуваат од прашањето.", 875, 178, 325, {
    size: 17,
    height: 62,
    color: C.purple,
  });
  addBullet(slide, "Инструкциите го насочуваат преносот на пораки.", 875, 255, 325, {
    size: 17,
    height: 62,
    color: C.purple,
  });
  addBullet(slide, "Собраната информација ги ревидира инструкциите.", 875, 332, 325, {
    size: 17,
    height: 62,
    color: C.purple,
  });
  addEquation(slide, "mᵏ,ᵗᵤ,ᵣ = pᵗᵤ ReLU(Wᵗᵣr ⊙ iᵏ)", 850, 440, 370, 86, {
    fill: C.purpleLight,
    stroke: C.purple,
    color: C.purple,
    size: 19,
  });
  addText(slide, "ReaRev го менува толкувањето на прашањето според информацијата откриена во графот.", 90, 601, 1100, 50, {
    size: 20,
    bold: true,
    align: "center",
  });
}

// 11 — NBFNet
{
  const slide = newSlide("NBFNet", 11, C.orange, "Zhu et al., 2021");
  addImage(slide, "nbfnet", 25, 108, 815, 458, {
    alt: "Пренос по релациски патеки во NBFNet",
  });
  addText(slide, "Научена Bellman–Ford постапка", 860, 116, 360, 45, {
    size: 21,
    bold: true,
    color: C.teal,
  });
  addEquation(slide, "hᵥ⁽⁰⁾ = q  ако v∈S;  0 во спротивно", 850, 176, 370, 78, {
    fill: C.tealLight,
    stroke: C.teal,
    color: C.teal,
    size: 19,
  });
  addEquation(slide, "m⁽ˡ⁾ᵤ,ᵣ,ᵥ = hᵤ⁽ˡ⁾ ⊙ wᵣ⁽ˡ⁾(q)", 850, 273, 370, 78, {
    fill: C.tealLight,
    stroke: C.teal,
    color: C.teal,
    size: 20,
  });
  addBullet(slide, "Патеките и релациите се условени од прашањето.", 860, 386, 350, {
    size: 17,
    height: 62,
    color: C.teal,
  });
  addBullet(slide, "Поддржани се повеќе почетни ентитети.", 860, 463, 350, {
    size: 17,
    height: 52,
    color: C.teal,
  });
  addText(slide, "Репрезентацијата на кандидатот опишува како тој се достигнува преку патеки релевантни за прашањето.", 78, 595, 1124, 62, {
    size: 20,
    bold: true,
    align: "center",
  });
}

// 12 — Retrieval metrics
{
  const slide = newSlide(
    "Метрики за пребарување",
    12,
    C.orange,
    "nDCG: Järvelin and Kekäläinen, 2002",
  );
  addText(slide, "Рангирање", 70, 116, 510, 42, {
    size: 24,
    bold: true,
    color: C.orange,
  });
  addCard(slide, 70, 170, 510, 190, {
    fill: C.orangeLight,
    stroke: C.orange,
    lineWidth: 1.5,
  });
  addBullet(slide, "Hits@1 / Hits@10 — барем еден точен одговор во првите 1 или 10 позиции", 92, 195, 465, {
    size: 18,
    height: 66,
    color: C.orange,
  });
  addBullet(slide, "nDCG@10 — позицијата и бројот на точни одговори во првите 10", 92, 278, 465, {
    size: 18,
    height: 62,
    color: C.orange,
  });

  addText(slide, "Покриеност на точните одговори", 650, 116, 560, 42, {
    size: 24,
    bold: true,
    color: C.blue,
  });
  addEquation(slide, "RetrievalGoldCoverage = (1/|Q|) Σq |Gq ∩ Rq| / |Gq|", 650, 170, 560, 80, {
    fill: C.blueLight,
    stroke: C.blue,
    color: C.blue,
    size: 19,
  });
  addEquation(slide, "RetrievalFullGoldCoverage = (1/|Q|) Σq 𝕀(Gq ⊆ Rq)", 650, 270, 560, 80, {
    fill: C.blueLight,
    stroke: C.blue,
    color: C.blue,
    size: 19,
  });
  addCard(slide, 180, 418, 920, 150, {
    fill: C.light,
    stroke: C.line,
    lineWidth: 1.3,
  });
  addText(slide, "Пример", 215, 442, 145, 34, {
    size: 20,
    bold: true,
    color: C.ink,
  });
  addMetric(slide, "2 / 3", "пронајдени точни одговори", 365, 435, 270, C.blue, {
    valueSize: 31,
    labelSize: 16,
  });
  addMetric(slide, "0", "целосна покриеност", 700, 435, 250, C.orange, {
    valueSize: 31,
    labelSize: 16,
  });
  addText(slide, "Gq — точни одговори · Rq — пребарани кандидати", 250, 585, 780, 34, {
    size: 16,
    color: C.muted,
    align: "center",
  });
}

// 13 — Retrieval results
{
  const slide = newSlide("Резултати од пребарувањето", 13, C.orange);
  addImage(slide, "architectureResults", 42, 112, 870, 400, {
    alt: "Резултати од пребарувањето по архитектура",
  });
  addCard(slide, 940, 122, 280, 382, {
    fill: C.orangeLight,
    stroke: C.orange,
    lineWidth: 1.7,
  });
  addText(slide, "NBFNet", 970, 145, 220, 44, {
    size: 29,
    bold: true,
    color: C.orange,
    align: "center",
  });
  addMetric(slide, "0,692", "Hits@1", 968, 205, 224, C.orange, {
    valueSize: 28,
  });
  addMetric(slide, "0,921", "Hits@10", 968, 292, 224, C.orange, {
    valueSize: 28,
  });
  addMetric(slide, "0,796", "nDCG@10", 968, 379, 224, C.orange, {
    valueSize: 28,
  });
  addText(slide, "средна вредност од 3 извршувања", 960, 464, 240, 24, {
    size: 12,
    color: C.muted,
    align: "center",
  });
  addCard(slide, 75, 553, 1130, 92, {
    fill: C.light,
    stroke: C.line,
    lineWidth: 1.2,
  });
  addText(
    slide,
    "Условувањето со прашањето и експлицитното моделирање на релациските патеки даваат најдобри резултати. Advance GraphSAGE е силен едноставен компромис.",
    105,
    568,
    1070,
    62,
    { size: 19, bold: true, align: "center" },
  );
}

// 14 — Evidence construction
{
  const slide = newSlide(
    "Конструкција на доказниот подграф",
    14,
    C.teal,
    "Goemans and Williamson, 1995",
  );
  addImage(slide, "evidence", 32, 110, 735, 413, {
    alt: "Споредба на најкратки патеки и PCST",
  });
  addText(slide, "PCST: награда на јазлите наспроти трошок на рабовите", 800, 115, 420, 58, {
    size: 21,
    bold: true,
    color: C.teal,
    align: "center",
  });
  addEquation(slide, "T* = arg maxT [Σv∈VT p(v) − Σe∈ET c(e)]", 805, 188, 410, 88, {
    fill: C.tealLight,
    stroke: C.teal,
    color: C.teal,
    size: 19,
  });
  addEquation(slide, "cconst(e) = λ", 805, 296, 410, 65, {
    fill: C.light,
    stroke: C.line,
    size: 20,
  });
  addEquation(slide, "csem(e) = max(ε, λ[1 − cos(q,re)])", 805, 381, 410, 77, {
    fill: C.light,
    stroke: C.line,
    size: 19,
  });
  addCard(slide, 115, 558, 1050, 82, {
    fill: C.tealLight,
    stroke: C.teal,
    lineWidth: 1.5,
  });
  addText(slide, "Најкратките патеки гарантираат поврзаност до секој кандидат; PCST може да отфрли кандидат кога поврзувањето е прескапо.", 145, 570, 990, 57, {
    size: 19,
    bold: true,
    align: "center",
  });
}

// 15 — Evidence metrics
{
  const slide = newSlide("Метрики за доказниот контекст", 15, C.teal);
  addText(slide, "Компактност", 70, 116, 510, 40, {
    size: 24,
    bold: true,
    color: C.teal,
  });
  addCard(slide, 70, 170, 510, 332, {
    fill: C.tealLight,
    stroke: C.teal,
    lineWidth: 1.5,
  });
  const compact = [
    ["AverageTriples", "просечен број тројки"],
    ["ContextCandidateCoverage", "удел на задржани кандидати"],
    ["CandidateReduction", "процент отфрлени кандидати"],
  ];
  compact.forEach(([title, body], i) => {
    const y = 198 + i * 91;
    addText(slide, title, 98, y, 450, 30, {
      size: 19,
      bold: true,
      color: C.teal,
    });
    addText(slide, body, 98, y + 32, 450, 30, {
      size: 17,
      color: C.ink,
    });
  });

  addText(slide, "Зачувана информација", 650, 116, 560, 40, {
    size: 24,
    bold: true,
    color: C.green,
  });
  addEquation(slide, "ContextGoldCoverage = (1/|Q|) Σq |Gq ∩ Cq| / |Gq|", 650, 170, 560, 94, {
    fill: C.greenLight,
    stroke: C.green,
    color: C.green,
    size: 19,
  });
  addEquation(slide, "ContextFullGoldCoverage = (1/|Q|) Σq 𝕀(Gq ⊆ Cq)", 650, 287, 560, 94, {
    fill: C.greenLight,
    stroke: C.green,
    color: C.green,
    size: 19,
  });
  addText(slide, "Cq — ентитети присутни во тројките од доказниот контекст", 670, 405, 520, 62, {
    size: 17,
    color: C.muted,
    align: "center",
  });
  addCard(slide, 180, 548, 920, 82, {
    fill: C.light,
    stroke: C.line,
    lineWidth: 1.2,
  });
  addText(slide, "Компактноста се оценува заедно со информацијата што останува достапна за јазичниот модел.", 210, 561, 860, 56, {
    size: 20,
    bold: true,
    align: "center",
  });
}

// 16 — Evidence results
{
  const slide = newSlide("Компактност и зачувување на точните одговори", 16, C.teal);
  addImage(slide, "evidenceResults", 36, 112, 760, 525, {
    alt: "Чувствителност на PCST на параметарот ламбда",
  });
  addCard(slide, 825, 120, 385, 170, {
    fill: C.tealLight,
    stroke: C.teal,
    lineWidth: 1.8,
  });
  addMetric(slide, "≈ 85%", "помал доказен контекст со PCST", 860, 145, 315, C.teal, {
    valueSize: 42,
    labelSize: 18,
    labelHeight: 56,
  });
  addText(slide, "Просечен број тројки", 840, 330, 355, 32, {
    size: 20,
    bold: true,
    color: C.ink,
    align: "center",
  });
  addMetric(slide, "94,47", "најкратки патеки", 835, 373, 170, C.blue, {
    valueSize: 27,
    labelSize: 14,
  });
  addMetric(slide, "13,50–14,45", "PCST", 1015, 373, 190, C.teal, {
    valueSize: 25,
    labelSize: 14,
  });
  addShape(slide, "rect", 845, 468, 345, 1.5, { fill: C.inactive });
  addText(slide, "Покриеноста се намалува минимално:", 850, 490, 335, 30, {
    size: 17,
    bold: true,
    align: "center",
  });
  addText(slide, "ContextGoldCoverage  0,888 → ≈0,881\nContextFullGoldCoverage  0,838 → ≈0,830", 850, 527, 335, 68, {
    size: 17,
    color: C.ink,
    align: "center",
  });
  addText(slide, "Стратегијата на трошок има поголемо влијание од малите промени на λ.", 830, 612, 375, 43, {
    size: 16,
    bold: true,
    color: C.teal,
    align: "center",
  });
}

// 17 — LLM metrics
{
  const slide = newSlide("Метрики за конечниот одговор", 17, C.purple);
  addText(slide, "Квалитет на одговорот", 70, 116, 510, 40, {
    size: 24,
    bold: true,
    color: C.purple,
  });
  addCard(slide, 70, 170, 510, 255, {
    fill: C.purpleLight,
    stroke: C.purple,
    lineWidth: 1.5,
  });
  addBullet(slide, "Hit — вратен е барем еден точен одговор", 95, 202, 460, {
    size: 18,
    height: 52,
    color: C.purple,
  });
  addBullet(slide, "ExactMatch — целосно совпаѓање на множествата", 95, 272, 460, {
    size: 18,
    height: 52,
    color: C.purple,
  });
  addBullet(slide, "Precision · Recall · F1 — квалитет на генерираното множество", 95, 342, 460, {
    size: 18,
    height: 60,
    color: C.purple,
  });

  addText(slide, "Искористување на контекстот", 650, 116, 560, 40, {
    size: 24,
    bold: true,
    color: C.green,
  });
  addCard(slide, 650, 170, 560, 255, {
    fill: C.greenLight,
    stroke: C.green,
    lineWidth: 1.5,
  });
  addText(slide, "FullContextCompleteAnswer", 675, 201, 510, 30, {
    size: 19,
    bold: true,
    color: C.green,
  });
  addText(slide, "Контекстот ги содржи сите точни одговори и моделот ги враќа сите.", 675, 234, 510, 55, {
    size: 17,
  });
  addText(slide, "FullContextLLMOmission", 675, 313, 510, 30, {
    size: 19,
    bold: true,
    color: C.green,
  });
  addText(slide, "Контекстот е целосен, но моделот испушта најмалку еден точен одговор.", 675, 346, 510, 55, {
    size: 17,
  });
  addEquation(
    slide,
    "LLMOmissionGivenFullContext = #{q: Gq⊆Cq ∧ Gq⊄Pq} / #{q: Gq⊆Cq}",
    190,
    478,
    900,
    86,
    { fill: C.light, stroke: C.line, size: 20 },
  );
  addText(slide, "Метриката го изолира пропустот на јазичниот модел од загубата во претходните фази.", 170, 588, 940, 45, {
    size: 20,
    bold: true,
    align: "center",
  });
}

// 18 — LLM quality and tokens
{
  const slide = newSlide("Квалитет и потрошувачка на токени", 18, C.purple);
  addImage(slide, "llmResults", 45, 107, 1190, 370, {
    alt: "Квалитет и број токени за DeepSeek и GPT",
  });
  addCard(slide, 70, 503, 545, 140, {
    fill: C.blueLight,
    stroke: C.blue,
    lineWidth: 1.5,
  });
  addText(slide, "Унија на најкратки патеки", 95, 518, 495, 30, {
    size: 20,
    bold: true,
    color: C.blue,
    align: "center",
  });
  addMetric(slide, "0,802 / 0,804", "Hit · DeepSeek / GPT", 83, 557, 166, C.blue, {
    valueSize: 20,
    labelSize: 12,
  });
  addMetric(slide, "0,627 / 0,613", "F1 · DeepSeek / GPT", 259, 557, 166, C.blue, {
    valueSize: 20,
    labelSize: 12,
  });
  addMetric(slide, "3,42 / 3,07 M", "токени · DeepSeek / GPT", 435, 557, 166, C.blue, {
    valueSize: 20,
    labelSize: 12,
  });
  addCard(slide, 665, 503, 545, 140, {
    fill: C.tealLight,
    stroke: C.teal,
    lineWidth: 1.5,
  });
  addText(slide, "Семантички PCST, λ=1", 690, 518, 495, 30, {
    size: 20,
    bold: true,
    color: C.teal,
    align: "center",
  });
  addMetric(slide, "0,689 / 0,728", "Hit · DeepSeek / GPT", 678, 557, 166, C.teal, {
    valueSize: 20,
    labelSize: 12,
  });
  addMetric(slide, "0,492 / 0,477", "F1 · DeepSeek / GPT", 854, 557, 166, C.teal, {
    valueSize: 20,
    labelSize: 12,
  });
  addMetric(slide, "0,69 / 0,66 M", "токени · 78–80% помалку", 1030, 557, 166, C.teal, {
    valueSize: 20,
    labelSize: 12,
  });
}

// 19 — Information loss
{
  const slide = newSlide("Каде се губи информацијата?", 19, C.purple);
  addImage(slide, "informationFlow", 35, 105, 1210, 435, {
    alt: "Последователно стеснување на информацијата низ системот",
  });
  const metrics = [
    ["0,877", "RetrievalGoldCoverage", C.orange],
    ["0,888", "ContextGoldCoverage\nнајкратки патеки", C.teal],
    ["0,211 / 0,237", "LLM omission · DeepSeek / GPT\nнајкратки патеки", C.purple],
    ["≈ 0,477 / 0,477", "LLM omission · DeepSeek / GPT\nсемантички PCST", C.green],
  ];
  metrics.forEach(([value, label, color], i) => {
    addMetric(slide, value, label, 55 + i * 305, 545, 280, color, {
      valueSize: i === 2 ? 25 : 28,
      labelSize: 14,
      labelHeight: 48,
    });
  });
  addText(slide, "Присуството на точниот ентитет не е доволно — потребни се јасни релациски тројки што го поврзуваат со прашањето.", 105, 644, 1070, 36, {
    size: 18,
    bold: true,
    color: C.ink,
    align: "center",
  });
}

// 20 — Conclusions
{
  const slide = newSlide("Заклучоци", 20, C.green);
  const rows = [
    {
      y: 120,
      n: "1",
      color: C.orange,
      fill: C.orangeLight,
      title: "NBFNet е најуспешен и најстабилен пребарувач",
      body: "Hits@10 = 0,921 ± 0,005 · nDCG@10 = 0,796 ± 0,007",
    },
    {
      y: 290,
      n: "2",
      color: C.teal,
      fill: C.tealLight,
      title: "Најкратките патеки го даваат највисокиот конечен квалитет",
      body: "Семантичкиот PCST создава ≈85% помал контекст и е подобар кога компактноста е приоритет.",
    },
    {
      y: 460,
      n: "3",
      color: C.purple,
      fill: C.purpleLight,
      title: "Изборот на LLM го менува балансот меѓу Hit и F1",
      body: "Главното тесно грло е искористувањето на достапниот релациски контекст.",
    },
  ];
  for (const row of rows) {
    addCard(slide, 80, row.y, 1120, 135, {
      fill: row.fill,
      stroke: row.color,
      lineWidth: 1.6,
    });
    addShape(slide, "ellipse", 108, row.y + 28, 78, 78, {
      fill: C.white,
      stroke: row.color,
      lineWidth: 2.5,
    });
    addText(slide, row.n, 108, row.y + 30, 78, 74, {
      size: 28,
      bold: true,
      color: row.color,
      align: "center",
    });
    addText(slide, row.title, 220, row.y + 18, 920, 42, {
      size: 22,
      bold: true,
      color: row.color,
    });
    addText(slide, row.body, 220, row.y + 65, 920, 52, {
      size: 18,
      color: C.ink,
    });
  }
}

// 21 — Future work
{
  const slide = newSlide("Идни подобрувања", 21, C.green);
  const cards = [
    {
      x: 58,
      title: "Хибриден доказен подграф",
      body: "Компактност на PCST со гарантирано зачувување на целосни и разбирливи патеки до најважните кандидати.",
      color: C.teal,
      fill: C.tealLight,
    },
    {
      x: 449,
      title: "Заедничка оптимизација",
      body: "Пребарувачот и конструкцијата на подграфот заеднички да учат кои патеки се најкорисни за јазичниот модел.",
      color: C.orange,
      fill: C.orangeLight,
    },
    {
      x: 840,
      title: "Поширока евалуација",
      body: "Дополнителни податочни множества, подолги патеки на расудување, повеќе извршувања и други јазични модели.",
      color: C.purple,
      fill: C.purpleLight,
    },
  ];
  for (const card of cards) {
    addCard(slide, card.x, 145, 345, 385, {
      fill: card.fill,
      stroke: card.color,
      lineWidth: 1.8,
      radius: 18,
    });
    addShape(slide, "rect", card.x + 35, 178, 275, 5, { fill: card.color });
    addText(slide, card.title, card.x + 30, 215, 285, 80, {
      size: 23,
      bold: true,
      color: card.color,
      align: "center",
    });
    addText(slide, card.body, card.x + 35, 325, 275, 140, {
      size: 18,
      color: C.ink,
      align: "center",
      vAlign: "top",
    });
  }
  addText(slide, "Целта е доказниот контекст истовремено да биде компактен и доволно јасен за јазичниот модел.", 135, 578, 1010, 60, {
    size: 22,
    bold: true,
    color: C.green,
    align: "center",
  });
}

// 22 — Closing
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addShape(slide, "rect", 0, 0, 1280, 12, { fill: C.green });
  addText(slide, "Ви благодарам", 120, 170, 1040, 90, {
    size: 47,
    bold: true,
    color: C.ink,
    align: "center",
  });
  addText(slide, "Прашања?", 120, 282, 1040, 82, {
    size: 38,
    bold: true,
    color: C.green,
    align: "center",
  });
  addShape(slide, "rect", 420, 396, 440, 2, { fill: C.inactive });
  addText(slide, "Виктор Костадиноски · 226009", 120, 430, 1040, 38, {
    size: 20,
    color: C.ink,
    align: "center",
  });
  addText(slide, "github.com/Itonkdong/graphragx", 120, 478, 1040, 35, {
    size: 17,
    color: C.blue,
    align: "center",
  });
}

await fs.mkdir(OUTPUT_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });
await fs.rm(RECEIPT, { force: true });
await (await PresentationFile.exportPptx(presentation)).save(CANDIDATE_PPTX);
await fs.rm(FINAL_PPTX, { force: true });
await fs.rm(FINAL_PDF, { force: true });

await finalizePresentation({
  explicitTotalSlideCount: 22,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: ROOT,
  candidatePath: CANDIDATE_PPTX,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000"],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: RECEIPT,
});

for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.items[index];
  const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(
    path.join(PREVIEW_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  const layout = await presentation.export({ slide, format: "layout" });
  await fs.writeFile(
    path.join(BUILD_DIR, `slide-${String(index + 1).padStart(2, "0")}.layout.json`),
    await layout.text(),
  );
}

const conversion = spawnSync(
  LIBREOFFICE_BIN,
  ["--headless", "--convert-to", "pdf", "--outdir", BUILD_DIR, FINAL_PPTX],
  { encoding: "utf8" },
);
if (conversion.status !== 0) {
  throw new Error(`LibreOffice PDF conversion failed:\n${conversion.stdout}\n${conversion.stderr}`);
}
const producedPdf = path.join(BUILD_DIR, "thesis_defense_presentation.pdf");
await fs.copyFile(producedPdf, FINAL_PDF);

console.log(`Presentation: ${FINAL_PPTX}`);
console.log(`PDF: ${FINAL_PDF}`);
console.log(`Previews: ${PREVIEW_DIR}`);
