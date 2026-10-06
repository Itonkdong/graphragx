#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
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

const SOURCE = path.join(
  ROOT,
  "metadata/presentation/thesis_defense_presentation.pptx",
);
const BUILD_DIR = path.join(ROOT, ".codex-build/equation-fix");
const CANDIDATE = path.join(BUILD_DIR, "candidate-native-equations.pptx");
const FINAL = path.join(
  ROOT,
  "metadata/presentation/thesis_defense_presentation_native_equations.pptx",
);
const RECEIPT = path.join(
  BUILD_DIR,
  "thesis_defense_presentation_native_equations.validation.json",
);

const equations = [
  {
    slide: 8,
    id: "sh/je9g3ahk",
    latex: String.raw`\alpha(q,r)=\cos(q,r)`,
    fontSize: 21,
  },
  {
    slide: 8,
    id: "sh/o7ydoret",
    latex: String.raw`\alpha(q,r)=\mathrm{sigmoid}(\mathrm{MLP}[q̃∥r̃∥q̃⊙r̃])`,
    fontSize: 17,
  },
  {
    slide: 9,
    id: "sh/bih4na5w",
    latex: String.raw`h_v^{(l+1)}=σ(W_0^{(l)}h_v^{(l)}+\sum_{r∈ℛ}\sum_{u∈N_v^r}\frac{1}{c_{v,r}}W_r^{(l)}h_u^{(l)})`,
    fontSize: 13,
  },
  {
    slide: 9,
    id: "sh/3y107axw",
    latex: String.raw`\alpha_{uv}^{r,j}=\mathrm{softmax}(e_{uv}^{r,j})`,
    fontSize: 18,
  },
  {
    slide: 10,
    id: "sh/0bit8b2d",
    latex: String.raw`m_{u,r}^{(k,t)}=p_u^{(t)}\mathrm{ReLU}(W_r^{(t)}r⊙i^{(k)})`,
    fontSize: 17,
  },
  {
    slide: 11,
    id: "sh/72t03qp0",
    latex: String.raw`h_v^{(0)}=q\;(v∈S);\;0\;(v∉S)`,
    fontSize: 17,
  },
  {
    slide: 11,
    id: "sh/ofqtgnyt",
    latex: String.raw`m_{u,r,v}^{(l)}=h_u^{(l)}⊙w_r^{(l)}(q)`,
    fontSize: 18,
  },
  {
    slide: 12,
    id: "sh/bulo7u58",
    latex: String.raw`\mathrm{RetrievalGoldCoverage}=\frac{1}{|Q|}\sum_{q∈Q}\frac{|G_q∩R_q|}{|G_q|}`,
    fontSize: 15,
  },
  {
    slide: 12,
    id: "sh/9sj65kn2",
    latex: String.raw`\mathrm{RetrievalFullGoldCoverage}=\frac{1}{|Q|}\sum_{q∈Q}\mathrm{I}(G_q⊆R_q)`,
    fontSize: 15,
  },
  {
    slide: 14,
    id: "sh/zi5c3y98",
    latex: String.raw`T^*=\arg\max_T[\sum_{v∈V_T}p(v)-\sum_{e∈E_T}c(e)]`,
    fontSize: 16,
  },
  {
    slide: 14,
    id: "sh/c3e1gjyd",
    latex: String.raw`c_{\mathrm{const}}(e)=\lambda`,
    fontSize: 18,
  },
  {
    slide: 14,
    id: "sh/a1wze9g7",
    latex: String.raw`c_{\mathrm{sem}}(e)=\max(ε,\lambda[1-\cos(q,r_e)])`,
    fontSize: 16,
  },
  {
    slide: 15,
    id: "sh/4vm9ovel",
    latex: String.raw`\mathrm{ContextGoldCoverage}=\frac{1}{|Q|}\sum_{q∈Q}\frac{|G_q∩C_q|}{|G_q|}`,
    fontSize: 15,
  },
  {
    slide: 15,
    id: "sh/it4rm5wv",
    latex: String.raw`\mathrm{ContextFullGoldCoverage}=\frac{1}{|Q|}\sum_{q∈Q}\mathrm{I}(G_q⊆C_q)`,
    fontSize: 15,
  },
  {
    slide: 17,
    id: "sh/14rqpgne",
    latex: String.raw`\mathrm{LLMOmissionGivenFullContext}=\frac{\#\{q:G_q⊆C_q∧G_q⊄P_q\}}{\#\{q:G_q⊆C_q\}}`,
    fontSize: 16,
  },
];

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

const sourceHashBefore = await sha256(SOURCE);
const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE));

for (const equation of equations) {
  const shape = presentation.resolve(equation.id);
  if (!shape) throw new Error(`Could not resolve ${equation.id} on slide ${equation.slide}.`);

  const preserved = {
    color: shape.text.color?.hex ?? "#263746",
    alignment: shape.text.alignment ?? "center",
    verticalAlignment: shape.text.verticalAlignment ?? "middle",
    autoFit: shape.text.autoFit ?? "shrinkText",
    wrap: shape.text.wrap ?? "square",
    insets: shape.text.insets ?? { top: 2, right: 4, bottom: 2, left: 4 },
  };

  shape.text = [{ latex: equation.latex, displayMode: "block" }];
  shape.text.fontSize = equation.fontSize;
  shape.text.typeface = "Cambria Math";
  shape.text.color = preserved.color;
  shape.text.alignment = preserved.alignment;
  shape.text.verticalAlignment = preserved.verticalAlignment;
  shape.text.autoFit = preserved.autoFit;
  shape.text.wrap = preserved.wrap;
  shape.text.insets = preserved.insets;
}

await (await PresentationFile.exportPptx(presentation)).save(CANDIDATE);

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

const sourceHashAfter = await sha256(SOURCE);
if (sourceHashBefore !== sourceHashAfter) {
  throw new Error("The source presentation changed during equation replacement.");
}

console.log(`Replaced equations: ${equations.length}`);
console.log(`Source SHA-256: ${sourceHashAfter}`);
console.log(`Presentation: ${FINAL}`);
console.log(`Validation: ${RECEIPT}`);
