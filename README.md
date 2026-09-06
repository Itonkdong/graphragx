# graphragX

`graphragX` is an experimental GraphRAG pipeline for question answering over
knowledge graphs. It trains a graph neural network (GNN) to rank answer
entities, constructs a compact evidence subgraph, and asks a language model to
generate the final answer from structured triples.

The current implementation targets **WebQSP** and supports:

- GraphSAGE, Advance GraphSAGE, R-GCN, HGT, ReaRev, and NBFNet retrievers;
- union-of-shortest-paths and Prize-Collecting Steiner Tree (PCST) evidence;
- constant and semantic PCST edge costs;
- OpenAI, DeepSeek, and FINKI Vezilka-compatible LLM inference;
- local, per-stage artifacts and optional Weights & Biases tracking;
- resumable TOML experiment manifests and reproducible result-generation scripts.

The implementation was inspired by [GNN-RAG: Graph Neural Retrieval for Large
Language Model Reasoning](https://arxiv.org/abs/2405.20139).

## Quick start

The shortest setup uses Docker Compose. Copy the environment template, add the
required API keys, and run the small example experiment together with Qdrant:

```bash
cp .env.example .env
docker compose up --build --abort-on-container-exit --exit-code-from graphragx
```

See [Docker Compose](docs/docker.md) for command overrides, official
experiments, scripts, persistent paths, and optional NVIDIA GPU access.

For a native environment, install [uv](https://docs.astral.sh/uv/), then create
the locked Python 3.11 environment:

```bash
uv sync --frozen
cp .env.example .env
```

Add the API keys needed by your selected stages to `.env`. Embedding-based
architectures and semantic PCST also require Qdrant:

```bash
docker compose up -d qdrant
```

Run the complete pipeline with recommended defaults:

```bash
uv run graphragx --default
```

Common stage-specific commands:

```bash
# Train and evaluate a retriever without LLM calls
uv run graphragx --retriever-only --gnn-architecture nbfnet --default

# Compare evidence construction from a saved retriever
uv run graphragx --evidence-only \
  --retriever-run-name RUN_NAME \
  --subgraph-algorithm pcst \
  --pcst-edge-cost-strategy constant \
  --pcst-edge-cost 1.0 \
  --default

# Generate answers from a saved retriever
uv run graphragx --inference-only \
  --retriever-run-name RUN_NAME \
  --subgraph-algorithm shortest_path \
  --default
```

Use `uv run graphragx --help` for the complete CLI reference.

## Experiment manifests

Preview or execute a resumable manifest:

```bash
uv run graphragx-experiments experiments/example.toml --dry-run
uv run graphragx-experiments experiments/example.toml
```

The three official code-indexed experiments are:

- `experiment_0_gnn_architectures.toml` — retriever architecture comparison;
- `experiment_1_evidence_subgraphs.toml` — evidence-subgraph comparison;
- `experiment_2_end_to_end.toml` — end-to-end LLM comparison.

Additional `probe_*.toml` files are diagnostics, not official experiments.

## Documentation

- [Documentation index](docs/index.md)
- [Installation and first runs](docs/getting-started.md)
- [Docker Compose](docs/docker.md)
- [Pipeline and supported architectures](docs/pipeline.md)
- [Configuration reference](docs/configuration.md)
- [Experiment manifests and result scripts](docs/experiments.md)
- [Artifacts and W&B](docs/artifacts-and-wandb.md)
- [Metrics reference](docs/metrics/index.md)
- [Development guide](docs/development.md)

## Outputs and research material

Runtime artifacts are written under `data/webqsp/`. Generated figures, tables,
and provenance records live under `metadata/`.

## Project lineage

This implementation builds on top of the initial `graphragX` project. The
accompanying [final thesis](metadata/thesis/thesis.pdf) documents the extended
system, methodology, and experimental findings.

## Credits

Created by **Viktor Kostadinoski**, supervised by **PhD Sonja Gievska** at FINKI, Ss. Cyril and
Methodius University in Skopje.

## License

Released under the [MIT License](LICENSE).
