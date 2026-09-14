# Paper draft

This folder is a self-contained English LaTeX draft based on `paper_plan.md`, the completed thesis, and its existing result exports. It contains the full paper and appendix, not a section outline. No new experiment or figure was generated for this draft.

## Upload to Overleaf

Upload the contents of this folder, or use `paper_overleaf.zip`. Set `main.tex` as the main document and choose **pdfLaTeX**. The bibliography uses **BibTeX**, `references.bib`, and the included `tmlr.bst`; Overleaf normally runs the required bibliography passes automatically.

Required files are `main.tex`, `appendix.tex`, `references.bib`, `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, and the `figures/` and `tables/` folders. The ZIP contains these plus the template license. Planning, provenance, and editorial notes do not need to be uploaded.

Local compilation was intentionally not performed, as requested. Source-level checks cover citations, labels, included files, table values, and balanced environments. The intended budget remains approximately 10 main-content pages and no more than 15 pages including references and appendices. Actual pagination, float placement, and table widths must be checked in Overleaf; they are not verified here.

## Structure

```text
main.tex                   Complete abstract and eight main sections
appendix.tex               Architecture details, metrics, full results, protocol
references.bib             15 primary references
tmlr.sty / tmlr.bst         Official TMLR template files, unmodified
fancyhdr.sty                File distributed with the official template
TMLR-LICENSE               Official template license
figures/                   Four existing PDF figures, copied without modification
tables/                    Five separate English LaTeX table fragments
sources/                   Snapshot of the three numerical summary CSVs
paper_plan.md              Agreed writing plan
editorial_notes.md          Items to review before submission
paper_overleaf.zip          Uploadable LaTeX project
```

The default template produces an anonymous manuscript. Author names, affiliations, acknowledgments, and identifying repository/tracking links have not been inserted. The template's “under review” heading is its standard formatting and does not mean a submission has been made.

The official template files were obtained from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file) on 2026-09-14. They are governed by the bundled license and retain their original notices.

## Figures

All four planned figures exist and are included. No figure path is missing. Their original Macedonian labels remain, and the English captions explain the panel order and terminology. **English-language editions of these figures are still needed for an English submission.** They have not been created because the current request explicitly asks to reuse available figures and report anything still needed.

| Included figure | Existing source |
| --- | --- |
| `figures/system_overview.pdf` | `../figures/system_architecture/system_overview.pdf` |
| `figures/evidence_pcst_lambda_sensitivity.pdf` | `../figures/evidence_subgraphs/evidence_pcst_lambda_sensitivity.pdf` |
| `figures/end_to_end_llm_quality_tokens.pdf` | `../figures/end_to_end_llm/end_to_end_llm_quality_tokens.pdf` |
| `figures/end_to_end_context_outcomes.pdf` | `../figures/end_to_end_llm/end_to_end_context_outcomes.pdf` |

After translating the figures, the temporary caption sentences about original labels should be removed. No additional conceptual diagram is required by the current paper.

## Evidence and tables

Numbers come from the three saved summary CSVs in `sources/`, copied from `metadata/results_metadata/`. Their original per-run exports and provenance files remain in that directory. The thesis is `../thesis/thesis.pdf`. Methodological details were cross-checked against the experiment manifests, metric documentation, and implementation.

Table fragments contain no document preamble. Their captions and labels live at their insertion points in `main.tex` or `appendix.tex`. Short labels (RGC, RFGC, CGC, CFGC) keep the tables readable; the manuscript defines all of them.

The paper preserves the study's empirical scope. It does not claim an exact numerical reproduction of GNN-RAG, additional runs, a conventional unfiltered test-only evaluation, or statistical significance. See `editorial_notes.md` for the specific wording and provenance details worth checking during revision.
