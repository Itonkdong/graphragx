#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "../..");
const RUNTIME_NODE_MODULES = process.env.RUNTIME_NODE_MODULES;
const SKILL_DIR = process.env.SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;

for (const [name, value] of Object.entries({
  RUNTIME_NODE_MODULES,
  SKILL_DIR,
  RUNTIME_PYTHON,
})) {
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must be set to an absolute path.`);
  }
}

const require = createRequire(import.meta.url);
const JSZip = require(path.join(RUNTIME_NODE_MODULES, "jszip"));
const artifactToolPath = path.join(
  RUNTIME_NODE_MODULES,
  "@oai/artifact-tool/dist/artifact_tool.mjs",
);
const { FileBlob, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);
const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

const PLAIN_SOURCE = path.join(
  ROOT,
  "metadata/presentation/thesis_defense_presentation.pptx",
);
const EQUATION_SOURCE = path.join(
  ROOT,
  "metadata/presentation/thesis_defense_presentation_native_equations.pptx",
);
const BUILD_DIR = path.join(ROOT, ".codex-build/powerpoint-equation-repair");
const CANDIDATE = path.join(BUILD_DIR, "candidate-powerpoint-safe.pptx");
const FINAL = path.join(
  ROOT,
  "metadata/presentation/thesis_defense_presentation_powerpoint_safe.pptx",
);
const RECEIPT = path.join(
  BUILD_DIR,
  "thesis_defense_presentation_powerpoint_safe.validation.json",
);

const affectedSlides = [8, 9, 10, 11, 12, 14, 15, 17];
const MC_NAMESPACE =
  "http://schemas.openxmlformats.org/markup-compatibility/2006";
const A14_NAMESPACE = "http://schemas.microsoft.com/office/drawing/2010/main";
const A_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main";
const R_NAMESPACE =
  "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

function shapeElements(xml) {
  return xml.match(/<p:sp>[\s\S]*?<\/p:sp>/g) ?? [];
}

function shapeId(shapeXml) {
  return shapeXml.match(/<p:cNvPr\s+id="(\d+)"/)?.[1] ?? null;
}

function countMatches(value, pattern) {
  return [...value.matchAll(pattern)].length;
}

async function sha256(filePath) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.rm(RECEIPT, { force: true });
try {
  await fs.access(FINAL);
  throw new Error(`Refusing to overwrite existing output: ${FINAL}`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const plainHashBefore = await sha256(PLAIN_SOURCE);
const equationHashBefore = await sha256(EQUATION_SOURCE);
const plainZip = await JSZip.loadAsync(await fs.readFile(PLAIN_SOURCE));
const equationZip = await JSZip.loadAsync(await fs.readFile(EQUATION_SOURCE));

let repairedEquationCount = 0;

for (const slideNumber of affectedSlides) {
  const entry = `ppt/slides/slide${slideNumber}.xml`;
  const plainFile = plainZip.file(entry);
  const equationFile = equationZip.file(entry);
  if (!plainFile || !equationFile) {
    throw new Error(`Missing ${entry} in one of the source presentations.`);
  }

  const plainXml = await plainFile.async("string");
  const equationXml = await equationFile.async("string");
  const fallbackById = new Map(
    shapeElements(plainXml)
      .map((shape) => [shapeId(shape), shape])
      .filter(([id]) => id !== null),
  );

  const repairedXml = equationXml.replace(
    /<p:sp>[\s\S]*?<\/p:sp>/g,
    (shape) => {
      if (!shape.includes("<a14:m")) return shape;

      const id = shapeId(shape);
      const fallback = id ? fallbackById.get(id) : null;
      if (!fallback) {
        throw new Error(
          `No plain-text fallback found for equation shape ${id ?? "unknown"} on slide ${slideNumber}.`,
        );
      }

      repairedEquationCount += 1;
      return [
        `<mc:AlternateContent xmlns:mc="${MC_NAMESPACE}" xmlns:a="${A_NAMESPACE}" xmlns:r="${R_NAMESPACE}">`,
        `<mc:Choice xmlns:a14="${A14_NAMESPACE}" Requires="a14">`,
        shape,
        "</mc:Choice>",
        "<mc:Fallback>",
        fallback,
        "</mc:Fallback>",
        "</mc:AlternateContent>",
      ].join("");
    },
  );

  const mathCount = countMatches(repairedXml, /<a14:m(?:\s|>)/g);
  const wrapperCount = countMatches(repairedXml, /<mc:AlternateContent(?:\s|>)/g);
  if (mathCount !== wrapperCount) {
    throw new Error(
      `Slide ${slideNumber} has ${mathCount} equations but ${wrapperCount} compatibility wrappers.`,
    );
  }

  equationZip.file(entry, repairedXml);
}

if (repairedEquationCount !== 15) {
  throw new Error(`Expected to repair 15 equations, repaired ${repairedEquationCount}.`);
}

const candidateBytes = await equationZip.generateAsync({
  type: "nodebuffer",
  compression: "DEFLATE",
  compressionOptions: { level: 6 },
});
await fs.writeFile(CANDIDATE, candidateBytes);

// Confirm the repaired package remains readable by the presentation runtime.
const imported = await PresentationFile.importPptx(await FileBlob.load(CANDIDATE));
if (imported.slides.items.length !== 22) {
  throw new Error(`Expected 22 slides after repair, found ${imported.slides.items.length}.`);
}

await finalizePresentation({
  explicitTotalSlideCount: 22,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: ROOT,
  candidatePath: CANDIDATE,
  finalPath: FINAL,
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
  verifyArtifactToolImport: true,
  receiptPath: RECEIPT,
});

const plainHashAfter = await sha256(PLAIN_SOURCE);
const equationHashAfter = await sha256(EQUATION_SOURCE);
if (plainHashBefore !== plainHashAfter || equationHashBefore !== equationHashAfter) {
  throw new Error("A source presentation changed during the repair.");
}

console.log(`Compatibility-wrapped equations: ${repairedEquationCount}`);
console.log(`Plain source SHA-256: ${plainHashAfter}`);
console.log(`Equation source SHA-256: ${equationHashAfter}`);
console.log(`Presentation: ${FINAL}`);
console.log(`Validation: ${RECEIPT}`);
