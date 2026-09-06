# Experiments and result generation

## Manifest format

The experiment runner executes ordinary `graphragx` arguments from a version-1
TOML file:

```toml
version = 1

[defaults]
args = ["--default", "--dataset", "WebQSP"]

[[runs]]
id = "retriever_seed42"
args = [
  "--retriever-only",
  "--seed", "42",
  "--training-run-name", "example_model",
  "--evaluation-run-name", "example_retriever",
]

[[runs]]
id = "inference_seed42"
after = ["retriever_seed42"]
args = [
  "--inference-only",
  "--retriever-run-name", "example_retriever",
  "--inference-run-name", "example_inference",
]
```

`[defaults].args` is prepended to every run. Each `[[runs]]` entry requires a
unique safe `id`, accepts an optional dependency list in `after`, and can be
disabled with `enabled = false`.

`wandb_lineage_name` appears in some official manifests. It is intentionally
ignored by the experiment runner and used only by the read-only result scripts
to resolve historical W&B runs whose persisted lineage predates the current
experiment naming.

## Running manifests

```bash
# Validate and print the resolved order
uv run graphragx-experiments experiments/example.toml --dry-run

# Run all enabled entries sequentially
uv run graphragx-experiments experiments/example.toml

# Run one entry and its dependencies
uv run graphragx-experiments experiments/example.toml --run inference_seed42
```

Useful controls:

- repeat `--run ID` to select several roots;
- `--continue-on-error` lets independent runs continue after a failure and
  marks dependent entries as blocked;
- `--force` reruns an unchanged command already recorded as successful.

Execution is sequential, so only one training command occupies the GPU. Logs
and resumable state are stored in:

```text
.experiment-runs/<manifest-stem>/
├── state.json
└── logs/<run-id>.log
```

A successful run is skipped on restart only when its command hash is unchanged.

## Official experiments

The code uses zero-based experiment numbers. Human-facing result presentations
may label the same sequence as Experiments 1–3.

| Code manifest | Presentation stage | Purpose |
| --- | ---: | --- |
| `experiment_0_gnn_architectures.toml` | 1 | Compare six GNN retrievers over three seeds. |
| `experiment_1_evidence_subgraphs.toml` | 2 | Compare shortest paths with constant and semantic PCST using the winning NBFNet retrievers. No LLM calls. |
| `experiment_2_end_to_end.toml` | 3 | Compare selected evidence settings with DeepSeek-V4-Flash and GPT-5.6 Luna. |

Files named `probe_*.toml` are diagnostics or recovery runs and are not part of
the official experiment count. `example.toml` demonstrates the format.

The official result scripts resolve only the experiment-defined lineage and
expected W&B groups. They do not modify W&B:

```bash
uv run python scripts/results/architecture_retrieval_results.py --help
uv run python scripts/results/evidence_subgraph_results.py --help
uv run python scripts/results/end_to_end_llm_results.py --help
```

Generated material is separated by type:

```text
metadata/
├── figures/<experiment>/
├── tables/<experiment>/
└── results_metadata/<experiment>/
```

Each results-metadata directory contains CSV data and `provenance.json`,
including the exact W&B run ids and names used to generate its figures and
tables. System diagrams are generated separately:

```bash
uv run --with matplotlib python scripts/results/generate_architecture_figures.py
```

One-time scripts under `scripts/one_time/` repair historical artifacts and are
not part of normal experiment execution.
