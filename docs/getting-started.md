# Getting started

For the containerized setup that runs the application and Qdrant together, see
[Docker Compose](docker.md). The instructions below describe a native Python
installation.

## Requirements

- Python 3.11 (the supported range is `>=3.11,<3.14`);
- [uv](https://docs.astral.sh/uv/);
- Docker when a local Qdrant service is needed;
- a CUDA, MPS, or CPU PyTorch environment appropriate for the selected run.

The lockfile is the reproducible dependency source. From the repository root:

```bash
uv sync --frozen
```

`uv` installs the development group by default, including `pytest`,
`torch-geometric`, and the figure-generation dependencies. Run commands through
the managed environment:

```bash
uv run graphragx --help
uv run python -m pytest -q
```

`requirements.txt` remains available for pip or Conda environments. The
compiled `pcst-fast` dependency should be installed with PEP 517 isolation:

```bash
conda run -n data-science python -m pip install --use-pep517 -r requirements.txt
```

The project intentionally pins NumPy below version 2 because
`pcst-fast==1.0.10` is not reliable with NumPy 2 on Linux AMD64.

## Environment variables

Create a local environment file:

```bash
cp .env.example .env
```

Set only the credentials needed by the stages you run:

| Variable | Used for | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI embeddings and OpenAI LLMs | none |
| `DEEPSEEK_API_KEY` | DeepSeek LLM inference | none |
| `DEEPSEEK_BASE_URL` | DeepSeek-compatible endpoint | `https://api.deepseek.com` |
| `VEZILKA_API_KEY` | FINKI Vezilka inference | none |
| `QDRANT_URL` | embedding vector store | `http://localhost:6333` |
| `QDRANT_API_KEY` | authenticated Qdrant | none |
| `QDRANT_COLLECTION_PREFIX` | embedding collection prefix | `graphragx_embeddings` |
| `WANDB_API_KEY` | online W&B logging | handled by W&B |
| `WANDB_PROJECT` | W&B project | `graphragx` |
| `WANDB_ENTITY` | W&B entity/team | none |
| `WANDB_MODE` | `online`, `offline`, or `disabled` | `online` |
| `GRAPHRAGX_LOG_LEVEL` | console log level | `INFO` |
| `GRAPHRAGX_LOG_COLOR` | console accent color | `cyan` |

You can also authenticate W&B interactively with `wandb login`. Disable it for
an individual run with `--no-wandb`.

## Qdrant

Start the bundled local service before a stage that must fetch uncached OpenAI
embeddings:

```bash
docker compose up -d qdrant
```

GraphSAGE and Advance GraphSAGE use entity, relation, and question embeddings.
R-GCN and HGT use entity embeddings together with categorical relation types,
while NBFNet uses pooled question embeddings. ReaRev uses its pinned local token
encoder and does not request OpenAI embeddings. Semantic PCST always needs
question and relation embeddings; constant PCST and shortest paths do not add
embedding requests.

## First runs

The pipeline is interactive without `--default`. Recommended defaults can be
selected non-interactively:

```bash
uv run graphragx --default
```

Useful smaller runs while validating an environment:

```bash
uv run graphragx --retriever-only \
  --training-max-instances 20 \
  --evaluation-max-instances 20 \
  --training-epochs 1 \
  --default
```

The main stage-specific forms are:

```bash
uv run graphragx --train-only --default
uv run graphragx --retriever-only --default
uv run graphragx --evaluation-only --evaluation-model-run-number 12 --default
uv run graphragx --evidence-only --retriever-run-number 7 --default
uv run graphragx --inference-only --retriever-run-number 7 --default
```

See [Pipeline](pipeline.md) for what each mode executes and
[Configuration](configuration.md) for valid option combinations.
