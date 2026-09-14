# Editorial notes for the first full draft

These are author-facing notes and are not included in the manuscript upload.

## Draft status

The complete narrative, equations, bibliography, tables, appendix, and English paper figures are written. No local LaTeX compilation was performed at the author's request; the 10-page main/15-page total budget remains a target until Overleaf establishes the rendered length. If shortening is necessary, first reduce repeated discussion and move detailed protocol prose out of the appendix into supplementary documentation. Preserve the evaluation denominator, conditional metric definitions, and full reported results.

## Figures

English-label versions of all four included PDF figures have been generated in `figures/`, together with matching PNG previews. `information_flow.pdf/png` is also available as an optional architecture figure. The original figures and the original results/presentation scripts remain unchanged.

## Verified clarifications relative to the plan

- The evaluation pool contains 1,746 questions, consistent across the saved retrieval runs and the 1,746 requests in each end-to-end summary. The thesis starts from 1,889 combined validation/test questions and describes filtering graphs that lack at least one gold answer; the difference is 143 exclusions.
- The GPT-5.6 Luna omission rate with semantic PCST at lambda 1 is **0.474 ± 0.005**, not 0.477. The corresponding DeepSeek rate is **0.477 ± 0.003**. The draft follows the CSV instead of the rounded statement in the plan.
- Final precision, recall, and F1 are **micro** metrics. Retrieval and context coverage are **macro** metrics. Their difference is not a direct stage-loss estimate. The conditional omission result supports the downstream diagnosis without that invalid subtraction.
- PCST is solved approximately. The manuscript gives the optimization objective without claiming an exact optimum or a guaranteed globally minimal evidence graph.
- With several seed entities, removing the temporary PCST root may leave multiple seed-anchored components. The draft states this rather than implying that the emitted original facts always form a single connected tree.
- GraphSAGE here is the weighted adaptation used in the thesis, not a verbatim original GraphSAGE implementation. R-GCN and HGT do not directly use the question in their evaluated message passing or classifier. These distinctions limit claims about the isolated effect of architecture.
- Context evidence may contain gold entities that were not selected candidates. Coverage is therefore not necessarily monotonically decreasing across stages.
- The two full-context outcomes allow extra predicted answers; they are distinct from Exact Match.

## Items for author review before submission

1. Confirm the preferred title and final author order with the coauthors. The draft is anonymous.
2. Preserve the explicit merged validation/test and filtering description. Selecting the downstream retriever using that evaluation pool prevents treating the resulting end-to-end result as a separate held-out benchmark estimate.
3. **Historical HGT depth requires confirmation.** Commit `fac388e` on 2026-09-06 changed the HGT seed-1337 manifest entry from two layers to three. The saved result source is the earlier run `uqlnazak`, named `112_20260829_214238_hgt` (2026-08-29). The other HGT runs are seed 42: `3udqdq19`; seed 2026: `uwcwhkip`. The aggregate export does not include actual historical layer counts, so a later manifest change alone does not prove the old run used three layers. The appendix table explicitly qualifies the current setting. Inspect the original run/model configuration before removing that note or interpreting HGT's standard deviation as purely seed variation. This requires provenance confirmation, not an automatic rerun; no measurement was changed here.
4. The manuscript uses the exact recorded model labels `deepseek-v4-flash` and `gpt-5.6-luna`. Preserve the source inference configs and record provider endpoints/snapshots when assembling a reproducibility supplement; do not substitute a newer model while presenting these numbers.
5. The thesis specifies raw train/validation/test split counts and training-time filtering, but the copied summary CSVs do not contain the retained training count. The draft gives the raw training count and policy without inventing a post-filter training count. Retrieve the original training metadata if that additional detail is needed.
6. Provide an anonymous implementation/artifact link for review. None has been created or uploaded. Check filenames and PDF metadata for identifying information before a real submission.
7. Confirm the final venue's current template, anonymity, supplementary-material, and AI-assistance disclosure rules at submission time. This draft follows the agreed TMLR template; no submission action was taken.

## Source map

- Narrative and original study: `metadata/thesis/thesis.pdf`.
- Agreed frame: `metadata/paper/paper_plan.md`.
- Retrieval numbers: `../thesis/results_metadata/architecture_retrieval/architecture_summary.csv`.
- Evidence numbers: `../thesis/results_metadata/evidence_subgraphs/evidence_summary.csv`.
- Final quality, tokens, and context outcomes: `../thesis/results_metadata/end_to_end_llm/end_to_end_llm_summary.csv`.
- Run counts and lineage: the `*_runs.csv` and `provenance.json` files beside each summary.
- Configurations: `experiments/experiment_0_gnn_architectures.toml`, `experiment_1_evidence_subgraphs.toml`, `experiment_2_end_to_end.toml`.
- Metrics: `docs/metrics/` and the retrieval/evidence/final-result services.
- PCST implementation: `pipeline/evaluation/services/pcst_evidence_subgraph.py`.
- Generation prompt and settings: `pipeline/evaluation/services/llm_answer_generation.py`.

Bibliography keys correspond to the thesis's primary references, with the published ACL 2025 GNN-RAG entry. Publication metadata were checked against the primary publication pages or author preprints where available; the nDCG bibliographic entry retains the thesis's DOI and citation data because the publisher page did not load in this session.
