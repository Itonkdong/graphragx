# Paper Plan

## Purpose

This paper will be a concise English publication derived entirely from the completed thesis. It will not introduce new experiments, rerun additional configurations, or attempt an exact numerical replication of every setting from GNN-RAG. The thesis, its recorded experiments, and its generated artifacts are the complete empirical basis of the paper.

The paper will present an independent implementation, empirical reassessment, and methodological extension of GNN-guided retrieval for knowledge-graph question answering. GNN-RAG is the principal methodological inspiration and establishes the core retrieval-to-generation workflow, but the paper's main subject is the broader design space studied in the thesis: retrieval architecture, evidence-subgraph construction, final answer generation, and information loss across the complete system.

## Venue and format assumptions

The first target format is TMLR, as required by the MLRC submission route.

- Use the official, unmodified TMLR LaTeX template.
- Use `\documentclass[10pt]{article}` and `\usepackage{tmlr}` for the anonymous submission.
- Follow double-blind review requirements.
- Use `natbib`: `\citet{...}` for an inline author reference and `\citep{...}` for a parenthetical citation attached to a claim.
- Keep the abstract to one paragraph.
- Keep manuscript figures as `figures/*.pdf`.
- Keep manuscript tables as `tables/*.tex`.
- Maintain one authoritative `references.bib`.
- Place appendices after the references, following the TMLR template.
- Keep code and artifact links anonymous during review.

These assumptions should be reconsidered only if a different venue is selected. In that case, preserve the paper's scientific frame while adapting the formatting and length to the new template.

## Length budget

The complete PDF must not exceed approximately 15 pages. The main paper should remain close to 10 pages.

| Component | Target length |
|---|---:|
| Abstract | 0.25 page |
| 1. Introduction | 1.0 page |
| 2. Background and Related Work | 0.75 page |
| 3. Methodology | 2.0 pages |
| 4. Experimental Setup | 0.5 page |
| 5. Results | 3.0 pages |
| 6. Discussion | 1.25 pages |
| 7. Limitations | 0.5 page |
| 8. Conclusion | 0.5 page |
| **Main content** | **approximately 9.75–10 pages** |
| References | approximately 1–1.5 pages |
| Appendix | at most 3.5–4 pages, reduced if references are longer |
| **Complete paper** | **maximum approximately 15 pages** |

The page allocation is a writing constraint rather than an official TMLR limit. If the main paper becomes too long, secondary equations, metric definitions, configuration details, and full tables should move to the appendix before central arguments are removed.

## Working title

### Preferred title

**Revisiting GNN-Guided Retrieval for Knowledge Graph Question Answering: Architectures, Evidence Subgraphs, and End-to-End Analysis**

### Alternatives

- **From Candidate Retrieval to Answer Generation: Evaluating GNN Architectures and Evidence Subgraphs for Knowledge Graph Question Answering**
- **An Empirical Study of GNN-Guided Retrieval and Evidence-Subgraph Construction for Knowledge Graph Question Answering**

The final title should reflect a methodological reassessment without implying an exact replication of all GNN-RAG experiments.

## Central framing

### Problem statement

A GNN-guided knowledge-graph question-answering system must make several dependent choices: how to rank candidate answers, how to transform candidates into evidence that can be supplied to a language model, and how to convert that evidence into a final answer. Evaluating only the final answer obscures where relevant information is lost and whether a smaller evidence context remains useful rather than merely retaining the correct entities.

### Relationship to GNN-RAG

The paper should state that GNN-RAG motivates the core workflow in which a GNN retrieves candidate answers, paths are extracted as evidence, and a language model generates the final answer. The present study independently implements this general workflow and broadens the investigation in two directions already covered by the thesis:

1. it evaluates six substantially different retrieval architectures under a shared experimental pipeline;
2. it re-examines shortest-path evidence construction by comparing it with constant-cost and semantic-cost PCST variants.

The paper should use the following precise description:

> an independent implementation and systematic empirical reassessment of GNN-guided retrieval for knowledge-graph question answering, inspired by the GNN-RAG framework.

It should not claim:

- exact numerical replication of GNN-RAG;
- recreation of all original datasets, models, prompts, or hyperparameters;
- direct confirmation or rejection of every claim in the original paper;
- new experiments beyond those already completed for the thesis.

### Overarching question

**How do retrieval architecture and evidence-subgraph construction affect the effectiveness of a GNN-guided knowledge-graph question-answering pipeline?**

### Research questions

1. **RQ1 — Retrieval architecture:** How does the choice of GNN architecture affect the ranking and coverage of correct answers?
2. **RQ2 — Evidence construction:** How does the evidence-subgraph construction strategy affect compactness and the preservation of relevant information?
3. **RQ3 — Final generation:** How does the choice of language model affect final answer quality and the utilization of the available evidence context?

The introduction should present these questions compactly. The discussion and conclusion should answer them smoothly rather than as a rigid question-and-answer list.

## Claimed contributions

The contribution statement should contain four evidence-supported points:

1. **Unified independent implementation.** A modular GNN-guided question-answering pipeline that separates candidate retrieval, evidence-subgraph construction, and final answer generation, enabling stage-wise evaluation.
2. **Retriever comparison.** A controlled comparison of GraphSAGE, Advance GraphSAGE, R-GCN, HGT, ReaRev, and NBFNet under shared data, training, and candidate-selection conditions.
3. **Evidence-construction reassessment.** A comparison of the shortest-path union used by the motivating workflow with rooted PCST using constant and semantic edge costs.
4. **End-to-end diagnosis.** An analysis showing that retaining correct answer entities is not sufficient: evidence structure and the language model's use of the available context materially affect final answer quality.

Do not frame the contribution as a new state-of-the-art system unless a comparison in the existing evidence directly supports that claim.

## Section-by-section plan

## Abstract — 0.25 page

### Purpose

Summarize the complete scientific story in one paragraph.

### Required content

1. The challenge: retrieving and presenting useful evidence from a local knowledge graph for question answering.
2. The methodological context: an independent GNN-guided retrieval and generation pipeline inspired by GNN-RAG.
3. The study: six retrieval architectures, shortest-path and PCST evidence construction, and two language models on WebQSP.
4. Principal findings:
   - NBFNet is the strongest and most stable retriever;
   - PCST reduces evidence size by about 85% and token use by about 78–80% while largely retaining entity coverage;
   - this compression lowers final answer quality, although semantic PCST provides a better trade-off than constant-cost PCST;
   - the main information loss occurs when available evidence is converted into the final answer.
5. Final implication: evidence usefulness depends on relational structure and successful context utilization, not only on the presence of correct entities.

### Exclude

- architectural equations;
- detailed hyperparameters;
- a list of every metric;
- claims of exact replication.

## 1. Introduction — 1 page

### Purpose

Motivate the problem, establish the relationship to GNN-RAG, identify the unresolved design choices, and state the research questions and contributions.

### Paragraph plan

1. Introduce knowledge-graph question answering and the need to retrieve multi-step structured evidence.
2. Explain why language models benefit from retrieved evidence but why a large or weakly structured context can be difficult to use.
3. Introduce GNN-RAG as the main methodological inspiration: GNN candidate retrieval, shortest-path evidence, textualization, and final generation.
4. Identify the paper's gap: the effectiveness of this workflow may depend on the retrieval architecture and the way candidate answers are transformed into evidence.
5. State the overarching question and the three research questions in compact prose.
6. Present the four contributions.

### Key citations

- RAG;
- a general knowledge-graph reference;
- GNN-RAG;
- G-Retriever when introducing alternative graph-evidence selection.

### Exclude

- extended background definitions;
- detailed descriptions of all six architectures;
- numerical results beyond one short headline sentence.

## 2. Background and Related Work — 0.75 page

### Purpose

Position the study without reproducing the thesis's full theoretical chapter.

### Subsections

#### 2.1 GNN reasoning for knowledge-graph question answering

- Briefly distinguish generic neighborhood aggregation, relation-aware message passing, adaptive question-conditioned reasoning, and path-oriented reasoning.
- Introduce GraphSAGE, R-GCN, HGT, ReaRev, and NBFNet only at the level needed to motivate the architecture selection.
- State that some architectures were adapted to the common answer-retrieval task.

#### 2.2 GNN-guided retrieval and evidence construction

- Explain the GNN-RAG workflow and the role of shortest paths.
- Introduce PCST as a compact connected-subgraph objective.
- Relate the PCST extension to G-Retriever without suggesting that the systems are identical.

### Exclude

- individual architecture equations;
- implementation details;
- a long chronological survey.

## 3. Methodology — 2 pages

### Purpose

Define the common pipeline and the two design dimensions studied in the paper.

### 3.1 Problem formulation and system overview — approximately 0.5 page

- Define question, local graph, seed entities, gold-answer set, candidate set, evidence subgraph, and predicted-answer set.
- Give the compact three-stage formulation:
  1. candidate scoring and selection;
  2. evidence-subgraph construction;
  3. answer generation from textualized evidence.
- Emphasize that outputs from every stage are retained for separate evaluation.
- Use the existing system-overview figure as the principal methodology figure.

### 3.2 Retrieval architectures — approximately 0.75 page

- Present the architectures as a progression of inductive biases rather than six disconnected mini-surveys:
  - GraphSAGE: shared neighborhood aggregation;
  - Advance GraphSAGE: learned question–relation interaction, reverse edges, residual connections, normalization, and question-conditioned classification;
  - R-GCN: relation-specific transformations;
  - HGT: relation-dependent attention;
  - ReaRev: iterative instruction-guided reasoning and revision;
  - NBFNet: question-conditioned path propagation.
- Include one compact comparison table or one compressed paragraph. Do not include every thesis equation.
- Retain at most two representative equations in the main paper:
  - the learned question–relation weighting used by Advance GraphSAGE;
  - the question-conditioned path message used by NBFNet, if space permits.
- Move detailed architecture equations and configuration differences to Appendix A.

### 3.3 Candidate selection — approximately 0.15 page

- Describe the common threshold, minimum top-k completion, and maximum candidate limit.
- Explain that the same selection procedure makes architecture outputs comparable.
- Keep the full formal definition in the appendix if needed.

### 3.4 Evidence-subgraph construction — approximately 0.45 page

- Define the union of all shortest paths connecting seed entities to reachable candidates.
- Define rooted PCST through its prize-minus-cost objective.
- Explain rank-based candidate prizes.
- Present the constant and semantic edge-cost variants.
- Keep the PCST objective and semantic-cost equation in the main paper; move implementation details to the appendix.

### 3.5 Answer generation — approximately 0.15 page

- Explain triple textualization and the constrained structured answer format.
- State that the language model should answer only from the supplied evidence.
- Clarify that malformed or failed generations count as incorrect.

## 4. Experimental Setup — 0.5 page

### Purpose

Provide enough detail for interpretation and reproducibility without repeating the methodology.

### Required content

- WebQSP with local Freebase subgraphs.
- Dataset split: 2,848 training examples, 250 validation examples, and 1,639 test examples; validation and test are combined into 1,889 evaluation examples in the recorded study.
- Exclusion of examples whose local graphs do not contain all labeled answers, with the count to be taken from the recorded metadata when writing the final text.
- Shared training setup: 10 epochs, Adam, learning rate `1e-3`, batch size of one graph, seeds 42, 1337, and 2026.
- Candidate selection: probability threshold 0.7, minimum 10 candidates, maximum 15.
- Stage progression:
  1. compare six retrievers;
  2. carry NBFNet into evidence construction;
  3. evaluate shortest paths and selected PCST configurations with DeepSeek-V4-Flash and GPT-5.6 Luna.
- State that results are reported as mean and standard deviation over three runs.

### Metric compression

Define only the metrics essential for reading the main results:

- retrieval: Hits@1, Hits@10, nDCG@10, and gold-answer coverage;
- evidence: average triples, candidate reduction, and context gold coverage;
- final answers: Hit, F1, Exact Match, total tokens, and `LLMOmissionGivenFullContext`.

Place complete formulas and secondary metrics in Appendix B.

## 5. Results — 3 pages

### Purpose

Present the three experimental stages as one connected chain of evidence. This is the largest section.

### 5.1 Retrieval architecture comparison — approximately 0.9 page

#### Main evidence

- Use the compact architecture-results table in the main paper.
- Report exact mean and standard deviation for the principal metrics.

#### Narrative priorities

1. NBFNet is best on all principal retrieval metrics and is stable across seeds:
   - Hits@1: `0.692 ± 0.015`;
   - Hits@10: `0.921 ± 0.005`;
   - nDCG@10: `0.796 ± 0.007`;
   - RetrievalGoldCoverage: `0.877 ± 0.005`;
   - RetrievalFullGoldCoverage: `0.827 ± 0.006`.
2. ReaRev is the second-strongest architecture but has substantially greater variability.
3. Advance GraphSAGE strongly improves over GraphSAGE, showing that a comparatively simple question-aware extension can be effective.
4. R-GCN is unstable, and HGT's additional complexity does not outperform the simpler Advance GraphSAGE configuration.
5. Explain why NBFNet is carried forward without attributing its advantage to an individual component that was not ablated.

#### Avoid

- narrating every table cell;
- treating unmeasured training time as an experimental result;
- claiming that architectural complexity alone caused the differences.

### 5.2 Evidence-subgraph construction — approximately 0.8 page

#### Main evidence

- Use the PCST sensitivity figure in the main paper.
- Move the complete nine-row evidence table to Appendix C.

#### Narrative priorities

1. Shortest paths produce `94.47 ± 22.58` triples on average.
2. PCST produces approximately 13–15 triples, a reduction of about 85%.
3. Most PCST configurations reduce context-gold coverage by less than one percentage point.
4. Constant-cost PCST with `lambda = 1` removes more candidates and slightly lowers full coverage.
5. Semantic-cost PCST is more stable at higher cost and preserves nearly all candidates.
6. The choice between semantic and constant cost is more consequential than fine adjustment of `lambda` within the tested range.

### 5.3 End-to-end answer quality and context use — approximately 1.3 pages

#### Main evidence

- Use the quality-and-token figure in the main paper.
- Use the context-utilization figure or a compact derived panel for the information-loss finding.
- Move the complete end-to-end table to Appendix D.

#### Narrative priorities

1. NBFNet with shortest-path evidence gives the strongest final answers:
   - DeepSeek-V4-Flash: Hit `0.802 ± 0.005`, F1 `0.627 ± 0.008`;
   - GPT-5.6 Luna: Hit `0.804 ± 0.015`, F1 `0.613 ± 0.006`.
2. PCST reduces total token use by approximately 78–80%.
3. Constant-cost PCST has the largest reduction in final answer quality.
4. Semantic PCST provides the better compactness–quality trade-off:
   - with `lambda = 1`, GPT-5.6 Luna uses approximately 4.6 times fewer tokens than shortest paths while reaching Hit `0.728 ± 0.006`.
5. Entity coverage alone does not explain final quality. PCST often retains the correct entities while removing relational evidence needed to interpret them.
6. With shortest paths, `LLMOmissionGivenFullContext` is `0.211 ± 0.002` for DeepSeek-V4-Flash and `0.237 ± 0.010` for GPT-5.6 Luna. With semantic PCST at `lambda = 1`, it is approximately `0.477` for both models.
7. The dominant information loss occurs during final generation and evidence utilization rather than from a large reduction in answer-entity coverage during evidence construction.

### Results-section ending

End with one transition sentence: the compactness of an evidence subgraph and the presence of correct entities are insufficient proxies for its usefulness to the downstream language model.

## 6. Discussion — 1.25 pages

### Purpose

Interpret the existing findings and connect them to the broader GNN-guided retrieval methodology.

### 6.1 What generalizes from the motivating workflow

- The central idea of using a learned graph retriever followed by explicit relational evidence is effective in the independently implemented pipeline.
- The effectiveness is not architecture-independent: task-aligned, question-conditioned path reasoning performs considerably better than generic aggregation in this setup.
- Shortest-path evidence remains a strong quality baseline because it preserves explanatory connectivity, not merely candidate entities.

Phrase these as findings from the present study, not as a complete replication verdict on GNN-RAG.

### 6.2 Compactness versus evidence usefulness

- PCST demonstrates that large structural compression is possible with minimal loss of answer-entity coverage.
- The downstream quality reduction shows that coverage metrics do not capture whether the remaining relations make the answer inferable.
- Semantic edge costs improve the trade-off but do not close the gap with shortest paths.
- A compact context should therefore be optimized for relational sufficiency, not only node coverage or triple count.

### 6.3 Stage-wise evaluation

- Explain why a single end-to-end accuracy value would hide distinct retrieval, construction, and generation failures.
- Emphasize `LLMOmissionGivenFullContext` as evidence that generation remains imperfect even when all labeled answers are present.
- Identify the transition from evidence context to final answer as the principal bottleneck established by the experiments.

### Reproducibility perspective

- State that the study exposes architectural and methodological sensitivity within a known pipeline pattern.
- Emphasize the shared evaluation conditions, repeated seeds, explicit stage outputs, reproducible configurations, and public implementation.
- Avoid presenting the work as a benchmark-wide universal conclusion.

## 7. Limitations — 0.5 page

Include only limitations already supported by the thesis:

- evaluation is limited to WebQSP and its provided local graphs;
- conclusions may not transfer to larger graphs, different relation structures, or longer reasoning paths;
- some architectures were adapted from tasks other than answer retrieval;
- only three random seeds are available, preventing strong statistical-significance claims;
- the architecture and PCST configuration search is limited;
- only two language models are evaluated;
- token counts are not directly comparable financial costs across providers or tokenizers;
- the study is an independent reassessment, not an exact reproduction of all original GNN-RAG settings.

Do not propose new experiments in this section. Reserve future directions for the final paragraph of the conclusion or the appendix only if the target style benefits from them.

## 8. Conclusion — 0.5 page

### Purpose

Provide a smooth synthesis of the research questions without introducing new analysis.

### Required progression

1. Retrieval architecture materially affects ranking and coverage; NBFNet is the strongest and most stable evaluated retriever.
2. PCST greatly compresses evidence with little entity-coverage loss, but the compressed evidence is less useful for final generation.
3. Semantic PCST gives a better trade-off than constant-cost PCST, while shortest paths retain the best final quality.
4. The two language models are close in Hit but differ in precision, recall, F1, and exact matching.
5. The dominant bottleneck is not simply finding or retaining answer entities, but preserving and using sufficient relational evidence.

### Final message

The final sentence should state that future GNN-guided question-answering systems should optimize retrieval, evidence structure, and context utilization jointly rather than treating them as interchangeable or independently sufficient components.

## Main-paper visual and table plan

### Main figures

1. **System overview** — adapt the existing `metadata/figures/system_architecture/system_overview.pdf` into `metadata/paper/figures/system_overview.pdf` when the LaTeX project is created.
2. **PCST sensitivity** — adapt `metadata/figures/evidence_subgraphs/evidence_pcst_lambda_sensitivity.pdf` for the evidence-construction results.
3. **End-to-end quality and tokens** — adapt `metadata/figures/end_to_end_llm/end_to_end_llm_quality_tokens.pdf`.
4. **Context utilization** — adapt `metadata/figures/end_to_end_llm/end_to_end_context_outcomes.pdf` or combine its essential outcome with the end-to-end figure if page pressure requires it.

Use `$academic-figure-creator` for any material redesign, translation, combination, or layout adaptation. Do not recreate figures manually.

### Main tables

1. **Architecture retrieval results** — adapt `metadata/tables/architecture_retrieval/architecture_primary_table.tex` into a compact main-paper table.

### Appendix tables

1. Architecture characteristics and detailed configurations.
2. Complete evidence-subgraph results from `metadata/tables/evidence_subgraphs/evidence_primary_table.tex`.
3. Complete end-to-end results from `metadata/tables/end_to_end_llm/end_to_end_llm_results.tex`.

Avoid placing a figure and table with the same information in the main paper.

## Appendix plan — maximum 3.5–4 pages

## Appendix A — Architecture details

- Compact architecture-characteristics table.
- Essential adaptation details for the shared retrieval task.
- Additional equations omitted from the main methodology.
- Fixed architectural configurations needed for reproducibility.

## Appendix B — Metrics and evaluation protocol

- Formal definitions of all reported retrieval, evidence, and final-answer metrics.
- Set normalization and failure-handling rules.
- Definition of `FullContextCompleteAnswer`, `FullContextLLMOmission`, and `LLMOmissionGivenFullContext`.

## Appendix C — Complete evidence-subgraph results

- Full results for shortest paths and all constant and semantic PCST values of `lambda`.
- Any secondary coverage and distinct-node metrics omitted from the main paper.

## Appendix D — Complete end-to-end results and reproducibility details

- Full answer-quality and token table for both language models.
- Shared training configuration and candidate-selection values.
- Hardware, software, seeds, run selection, and artifact provenance.
- Anonymous code and artifact availability statement for review.

If the appendix exceeds the page budget, retain reproducibility-critical details and move bulky machine-readable results to anonymized supplementary material.

## Source and citation plan

The initial bibliography should be derived from the thesis and normalized into `metadata/paper/references.bib`. At minimum, retain and verify the primary sources for:

- retrieval-augmented generation;
- knowledge graphs;
- GraphSAGE;
- R-GCN;
- HGT;
- ReaRev;
- NBFNet;
- PCST;
- GNN-RAG;
- G-Retriever;
- MiniLM;
- DistMult;
- PNA;
- WebQSP;
- nDCG.

Use the official published GNN-RAG ACL 2025 entry rather than its earlier preprint entry. Add new literature only when it is necessary to position the paper or support a claim not already covered by the thesis.

## Writing order

Write the paper in this order rather than from the first page forward:

1. Results
2. Methodology
3. Experimental Setup
4. Discussion and Limitations
5. Introduction and Related Work
6. Conclusion
7. Abstract
8. Appendix

This order anchors the framing and claims in the evidence already available.

## Completion criteria

The plan is successfully implemented when:

- the main content remains close to 10 pages;
- the complete PDF remains at or below approximately 15 pages;
- every claim is supported by an existing thesis result or a verified citation;
- no new experiment is implied or invented;
- the relationship to GNN-RAG is clear but does not dominate the complete paper;
- the three research questions are answered by the presented evidence;
- every main figure or table contributes directly to the narrative;
- all secondary detail is either in the appendix or anonymized supplementary material;
- the submission compiles with the unmodified TMLR template and satisfies double-blind requirements.
