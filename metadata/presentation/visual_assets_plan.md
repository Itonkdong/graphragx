# Presentation Visual Assets Plan

## Objective

Before constructing the presentation, prepare the reusable technical figures that are not already available in the thesis assets. The figures should follow the visual system defined in `visual_system.md` and reuse the graphical language established by the existing system-architecture figures.

Only visuals that materially improve the explanation of the problem, architectures, evidence construction, or information flow should be created. Slides that are clearer with text, formulas, tables, or existing result charts should not receive an additional illustration.

## Shared requirements

All newly generated figures must:

- use the same semantic colors as the thesis figures;
- use the same visual treatment for the seed entity, entities-candidates, correct answer, and intermediate entities;
- mark the correct answer with the larger dotted green ring;
- fade inactive or unselected graph elements with the established light-gray color;
- use Arial-compatible Macedonian Cyrillic labels;
- use consistent node sizes, edge widths, arrowheads, and label styles;
- avoid gradients, shadows, three-dimensional effects, decorative imagery, and unnecessary legends;
- remain understandable when placed on a 16:9 slide;
- be generated as vector PDF and as a 300 DPI PNG preview;
- be reproducible from a project script rather than manually edited output.

The illustrations should contain only the labels necessary to understand the mechanism. Formulas, detailed explanations, experimental values, and citations should normally remain native presentation elements.

## Output organization

```text
metadata/figures/presentation/
├── problem/
│   ├── two_hop_example.pdf
│   └── two_hop_example.png
├── architectures/
│   ├── architecture_overview.pdf
│   ├── architecture_overview.png
│   ├── graphsage_comparison.pdf
│   ├── graphsage_comparison.png
│   ├── rgcn_hgt_comparison.pdf
│   ├── rgcn_hgt_comparison.png
│   ├── rearev_reasoning_cycle.pdf
│   ├── rearev_reasoning_cycle.png
│   ├── nbfnet_path_propagation.pdf
│   └── nbfnet_path_propagation.png
└── evidence/
    ├── evidence_strategy_comparison.pdf
    ├── evidence_strategy_comparison.png
    ├── information_loss_pipeline.pdf
    └── information_loss_pipeline.png
```

The corresponding generation code should be placed under `scripts/results/` and should derive all output paths from the repository root.

## Figures to create

### 1. Multi-hop question-answering example

**Presentation slide:** 2, „Од прашање до одговор“

**Files:** `problem/two_hop_example.pdf` and `problem/two_hop_example.png`

**Purpose:** Introduce the task through a simple illustrative question that can be understood before the system architecture is discussed.

**Content:**

- question: „На кој континент се наоѓа Колосеумот?“;
- seed entity: Колосеум;
- intermediary entities: Рим and Италија;
- correct answer: Европа;
- main path: Колосеум → Рим → Италија → Европа;
- secondary branch: Италија → Италијански јазик, showing that the structure is a graph rather than a chain.

**Main visual message:** The system must find the answer entity and preserve the relational evidence that connects it to the question.

### 2. Architecture overview

**Presentation slide:** 7, „Архитектури за пребарување“

**Files:** `architectures/architecture_overview.pdf` and `architectures/architecture_overview.png`

**Purpose:** Orient the audience before the individual architecture slides.

**Content:**

- one continuous conceptual progression rather than six unrelated panels;
- GraphSAGE associated with neighborhood aggregation;
- Advance GraphSAGE, R-GCN, and HGT associated with increasingly explicit relation-aware processing;
- ReaRev associated with adaptive instruction-guided reasoning;
- NBFNet associated with question-conditioned propagation along relational paths;
- short Macedonian labels for the principal mechanism of each group.

**Main visual message:** The evaluated architectures progress from local neighborhood aggregation toward explicit, question-conditioned reasoning over relational paths.

### 3. GraphSAGE and Advance GraphSAGE

**Presentation slide:** 8, „GraphSAGE и Advance GraphSAGE“

**Files:** `architectures/graphsage_comparison.pdf` and `architectures/graphsage_comparison.png`

**Purpose:** Explain the difference between fixed semantic weighting and learned question-dependent weighting.

**Content:**

- the same small graph shown for both architectures;
- GraphSAGE side:
  - aggregation of incoming neighborhood messages;
  - fixed cosine-based question-relation weights;
- Advance GraphSAGE side:
  - learned question-relation gate;
  - reverse-edge messages;
  - a subtle indication of the residual connection;
  - question-conditioned final node score;
- no formula inside the figure, because the formula remains a native slide element.

**Main visual message:** Advance GraphSAGE learns which relations matter for the current question instead of relying only on fixed similarity.

### 4. R-GCN and HGT

**Presentation slide:** 9, „R-GCN и HGT“

**Files:** `architectures/rgcn_hgt_comparison.pdf` and `architectures/rgcn_hgt_comparison.png`

**Purpose:** Contrast relation-specific transformations with relation-dependent attention.

**Content:**

- the same underlying relation-colored graph for both architectures;
- R-GCN side:
  - visibly different transformations for different relation types;
  - messages combined at the target entity;
- HGT side:
  - the same neighboring entities and relation types;
  - different attention strengths represented by edge or message emphasis;
- no detailed tensor notation inside the figure.

**Main visual message:** R-GCN changes the transformation according to the relation type, while HGT additionally learns how strongly each relation-specific message should influence the target.

### 5. ReaRev reasoning cycle

**Presentation slide:** 10, „ReaRev“

**Files:** `architectures/rearev_reasoning_cycle.pdf` and `architectures/rearev_reasoning_cycle.png`

**Purpose:** Explain iterative reasoning and instruction revision without requiring a detailed derivation.

**Content:**

- question representation;
- multiple learned instructions derived from the question;
- instruction-guided message propagation over the graph;
- information collected from the current reasoning iteration;
- revision of the instructions;
- a clearly indicated second reasoning iteration;
- the same seed and correct-answer colors used throughout the presentation.

**Main visual message:** ReaRev revises how it interprets the question according to information discovered during graph reasoning.

### 6. NBFNet path propagation

**Presentation slide:** 11, „NBFNet“

**Files:** `architectures/nbfnet_path_propagation.pdf` and `architectures/nbfnet_path_propagation.png`

**Purpose:** Explain the path-based architecture that produces the strongest retrieval results.

**Content:**

- seed entity initialized with the question representation;
- layered propagation across multiple hops;
- at least two competing relational paths;
- stronger emphasis on the path whose relation messages align with the question;
- faded irrelevant path or messages;
- independent candidate scores at the final layer;
- the correct answer shown as a highly ranked candidate, retaining its dotted green ring.

**Main visual message:** NBFNet learns how each candidate can be reached from the seed through relational paths relevant to the question.

### 7. Shortest paths and PCST comparison

**Presentation slide:** 14, „Конструкција на доказниот подграф“

**Files:** `evidence/evidence_strategy_comparison.pdf` and `evidence/evidence_strategy_comparison.png`

**Purpose:** Make the structural difference between the two evidence-construction strategies immediately visible.

**Content:**

- one shared local graph with the same seed and candidate entities;
- shortest-path side:
  - every shortest path from the seed to each reachable candidate;
  - the union of all retained triples;
- PCST side:
  - candidate prizes represented without adding excessive labels;
  - edge costs indicated consistently;
  - a smaller rooted connecting structure;
  - at least one candidate or branch faded to show that PCST may omit it;
- the exact PCST objective remains a native slide formula.

**Main visual message:** The shortest-path union prioritizes complete connectivity to all reachable candidates, while PCST trades candidate prizes against edge costs to construct a smaller context.

### 8. Information-loss pipeline — retained but not selected

**Presentation slide:** Not used. Slide 19 reuses the thesis `information_flow` figure instead.

**Files:** `evidence/information_loss_pipeline.pdf` and `evidence/information_loss_pipeline.png`

**Status:** The generated asset remains in the repository for reference, but it is not selected for the final presentation because the existing thesis figure communicates the information flow more clearly.

**Original purpose:** Summarize the central end-to-end finding by showing where information is preserved or lost.

**Content:**

- build on the graphical language of the existing `information_flow` figure;
- stages:
  - local graph;
  - retrieved entities-candidates;
  - entities and relations in the evidence context;
  - answers returned by the language model;
- preserve the same example entities and graph structure where practical;
- visually distinguish entity-level preservation from relation-level loss;
- leave sufficient space around the stages for the key metric values to be added as native slide text;
- emphasize that the correct entity can remain present while the relations needed to use it become insufficient.

**Main visual message:** Preserving the correct entity does not guarantee a correct final answer when the evidence context does not preserve or clearly express the relations that justify it.

## Existing figures to reuse

The following assets already exist and should not be recreated unless their text becomes unreadable in presentation format:

| Slide | Existing asset |
|---:|---|
| 4 | `metadata/figures/system_architecture/system_overview.pdf` |
| 13 | `metadata/figures/architecture_retrieval/architecture_primary_metrics_hits1_all_metrics.pdf` or the selected thesis variation |
| 16 | `metadata/figures/evidence_subgraphs/evidence_pcst_lambda_sensitivity.pdf` |
| 18 | `metadata/figures/end_to_end_llm/end_to_end_llm_quality_tokens.pdf` |
| 17 or 19, if needed | `metadata/figures/end_to_end_llm/end_to_end_context_outcomes.pdf` |
| 19 | `metadata/figures/system_architecture/information_flow.pdf` |

The selected result-figure variants should be confirmed during slide construction according to readability at presentation scale.

## Slides without dedicated generated figures

No separate illustration should be generated for the following slides:

- slide 1: title;
- slide 3: related work and research gap;
- slide 5: research questions;
- slide 6: dataset and experimental setup;
- slide 12: retrieval metrics;
- slide 15: evidence-context metrics;
- slide 17: final-answer metrics;
- slide 20: conclusions;
- slide 21: future improvements;
- slide 22: thanks and questions.

These slides should use native presentation text, formulas, compact tables, or simple structural arrangements. In particular, the related-work comparison and the retrieval-metric explanation must remain formal and restrained rather than receiving dedicated illustrative graphics.

## Recommended generation order

1. Create the common graph primitives and shared diagram helpers.
2. Generate the multi-hop example.
3. Generate the architecture overview.
4. Generate the four architecture-specific figures.
5. Generate the shortest-path and PCST comparison.
6. Generate the information-loss pipeline.
7. Export every figure to PDF and PNG.
8. Inspect all PNG previews together for consistency.
9. Correct labels, spacing, visual hierarchy, and cross-figure differences before building the presentation.

The presentation itself should only be assembled after these assets and the already existing thesis figures have been reviewed at slide scale.
