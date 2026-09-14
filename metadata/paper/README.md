# Paper draft

This folder is a self-contained English LaTeX draft based on `paper_plan.md`, the completed thesis, and its existing result exports. It contains the full paper and appendix, not a section outline. The paper figures are generated in English by the separate paper-figure script described below; no new experiment was run.

## Upload to Overleaf

Upload the contents of this folder to Overleaf. Set `main.tex` as the main document and choose **pdfLaTeX**. The bibliography uses **BibTeX**, `references.bib`, and the included `tmlr.bst`; Overleaf normally runs the required bibliography passes automatically.

Required files are `main.tex`, `appendix.tex`, `references.bib`, `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, and the `figures/` and `tables/` folders. Planning, provenance, and editorial notes do not need to be uploaded.

Local compilation was intentionally not performed, as requested. Source-level checks cover citations, labels, included files, table values, and balanced environments. The intended budget remains approximately 10 main-content pages and no more than 15 pages including references and appendices. Actual pagination, float placement, and table widths must be checked in Overleaf; they are not verified here.

## Structure

```text
main.tex                   Complete abstract and eight main sections
appendix.tex               Architecture details, metrics, full results, protocol
references.bib             15 primary references
tmlr.sty / tmlr.bst         Official TMLR template files, unmodified
fancyhdr.sty                File distributed with the official template
TMLR-LICENSE               Official template license
figures/                   English paper figures in PDF and matching PNG form
tables/                    Five separate English LaTeX table fragments
sources/                   Snapshot of the three numerical summary CSVs
paper_plan.md              Agreed writing plan
editorial_notes.md          Items to review before submission
```

The default template produces an anonymous manuscript. Author names, affiliations, acknowledgments, and identifying repository/tracking links have not been inserted. The template's “under review” heading is its standard formatting and does not mean a submission has been made.

The official template files were obtained from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file) on 2026-09-14. They are governed by the bundled license and retain their original notices.

## Figures

All four planned figures exist with English labels in both PDF and matching PNG form. An English `information_flow` figure is also available as an optional architecture figure. The PDFs are the manuscript assets; the PNGs are convenient for inspection or slides.

| Included figure | Existing source |
| --- | --- |
| `figures/system_overview.pdf` | `scripts/results/generate_paper_figures.py` |
| `figures/evidence_pcst_lambda_sensitivity.pdf` | `scripts/results/generate_paper_figures.py` |
| `figures/end_to_end_llm_quality_tokens.pdf` | `scripts/results/generate_paper_figures.py` |
| `figures/end_to_end_context_outcomes.pdf` | `scripts/results/generate_paper_figures.py` |

Run `scripts/results/generate_paper_figures.py` from the repository root to regenerate these figures from the persisted summary CSVs and the existing architecture primitives. The original results and presentation scripts are unchanged. No additional conceptual diagram is required by the current paper.

## Evidence and tables

Numbers come from the three saved summary CSVs in `sources/`, copied from `../thesis/results_metadata/`. Their original per-run exports and provenance files remain in that directory. The thesis is `../thesis/thesis.pdf`. Methodological details were cross-checked against the experiment manifests, metric documentation, and implementation.

Table fragments contain no document preamble. Their captions and labels live at their insertion points in `main.tex` or `appendix.tex`. Short labels (RGC, RFGC, CGC, CFGC) keep the tables readable; the manuscript defines all of them.

The paper preserves the study's empirical scope. It does not claim an exact numerical reproduction of GNN-RAG, additional runs, a conventional unfiltered test-only evaluation, or statistical significance. See `editorial_notes.md` for the specific wording and provenance details worth checking during revision.
