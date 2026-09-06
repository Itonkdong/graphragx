# Development guide

## Repository layout

```text
graphragx/
├── main.py                    # CLI and pipeline composition
├── helpers/                   # shared constants, environment, logging, paths
├── pipeline/
│   ├── preparation/           # dataset, graph, embedding, model, and training stages
│   └── evaluation/            # retrieval, evidence, LLM, metrics, and W&B stages
├── experiments/               # official manifests, probes, and one example
├── scripts/
│   ├── experiments/           # resumable manifest runner
│   ├── results/               # read-only W&B result generators
│   └── one_time/              # explicitly scoped historical migrations
├── tests/                     # unit and pipeline-composition tests
├── docs/                      # maintained software documentation
├── metadata/                  # generated figures, tables, and provenance
└── data/webqsp/               # generated runtime artifacts; not source code
```

Agent-facing project conventions are stored separately in
[`agents-metadata/`](../agents-metadata/).

## Test workflow

Run focused tests while editing, then the complete suite:

```bash
uv run python -m pytest -q tests/evaluation/test_final_results_evaluation.py
uv run python -m pytest -q
git diff --check
```

Changes to metric semantics should include tests for the calculator, persisted
JSON, and both `Summary_Plots/*` and curated `Run_Summary/*` W&B mappings.
Changes to modes or CLI flags should include parser validation and pipeline
composition tests.

## Adding a retriever architecture

Architecture configuration is registry-driven. A new architecture should:

1. define its options and data requirements in
   `pipeline/preparation/helpers/configuration_definitions.py`;
2. expose a lazy model-builder callback and, if needed, a runtime strategy;
3. persist every architecture-specific value in model configuration;
4. restore the saved configuration for continuation and evaluation;
5. add focused architecture, preparation, training, evaluation, and CLI tests.

Do not add architecture-specific branches to the central parser when the
registry can own the behavior.

## Generated and historical files

- Runtime data under `data/webqsp/` and local W&B directories are generated.
- Result scripts may write only under the matching `metadata/figures`,
  `metadata/tables`, and `metadata/results_metadata` directory.
- Result scripts must be read-only with respect to W&B and record exact run
  provenance.
- One-time migrations belong under `scripts/one_time/`; they should require
  explicit run selectors and must not become permanent CLI behavior.

## Documentation maintenance

Keep the root README as a landing page. Put operational details in `docs/` and
research outputs in `metadata/`.

When behavior changes, update the narrowest relevant page:

- CLI/mode changes: `configuration.md` and possibly `pipeline.md`;
- artifact/W&B changes: `artifacts-and-wandb.md`;
- experiment changes: `experiments.md`;
- metric changes: the relevant file under `docs/metrics/`.

Metric documentation must state the exact key, denominator, zero/undefined
behavior, persisted file, W&B name, and available run modes. Verify these facts
against code and tests rather than copying old generated artifacts.
