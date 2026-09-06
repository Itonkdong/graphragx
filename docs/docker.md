# Docker Compose

Docker Compose runs the application and its local Qdrant vector store together.
The default application command is the deliberately small
[`experiments/example.toml`](../experiments/example.toml) manifest. Runtime
data and generated outputs remain on the host, so rebuilding the image does not
discard them.

## First run

Create the local environment file and add the credentials required by the
example manifest. The example trains an embedding-based GraphSAGE retriever and
then performs DeepSeek inference, so it needs both `OPENAI_API_KEY` and
`DEEPSEEK_API_KEY`. Set W&B credentials as well, or use `WANDB_MODE=offline` or
`WANDB_MODE=disabled`.

```bash
cp .env.example .env
docker compose up --build --abort-on-container-exit --exit-code-from graphragx
```

Qdrant remains available until the Compose project is stopped. Remove the
containers without deleting the persistent Qdrant and Hugging Face volumes:

```bash
docker compose down
```

Use `docker compose down --volumes` only when the cached Qdrant collections and
model downloads should also be deleted.

The following host paths are mounted into the application container:

| Host path | Container path | Contents |
| --- | --- | --- |
| `data/` | `/app/data` | prepared WebQSP data and run artifacts |
| `metadata/` | `/app/metadata` | generated figures, tables, and result metadata |
| `.experiment-runs/` | `/app/.experiment-runs` | resumable manifest state |
| `wandb/` | `/app/wandb` | local W&B files and offline runs |

On Linux, set `GRAPHRAGX_UID` and `GRAPHRAGX_GID` in `.env` to `id -u` and
`id -g` if the defaults do not match the current account. This keeps generated
files writable outside the container.

## Override the default command

`docker compose run --rm graphragx ...` replaces only the default example
command. Qdrant is started automatically. Commands use the installed console
scripts directly; `uv run` is not required inside the image.

Run the pipeline directly:

```bash
docker compose run --rm graphragx \
  graphragx --retriever-only --gnn-architecture nbfnet --default
```

Run an official experiment manifest:

```bash
docker compose run --rm graphragx \
  graphragx-experiments experiments/experiment_0_gnn_architectures.toml
```

Preview a manifest without executing it:

```bash
docker compose run --rm graphragx \
  graphragx-experiments experiments/experiment_1_evidence_subgraphs.toml --dry-run
```

Run a Python utility or result-generation script:

```bash
docker compose run --rm graphragx \
  python scripts/results/architecture_retrieval_results.py --help
```

Run tests:

```bash
docker compose run --rm graphragx python -m pytest -q
```

Open an interactive shell:

```bash
docker compose run --rm graphragx bash
```

For a command that genuinely does not use Qdrant, the readiness wait can be
disabled without removing the service:

```bash
docker compose run --rm -e GRAPHRAGX_WAIT_FOR_QDRANT=0 graphragx \
  python scripts/results/architecture_retrieval_results.py --help
```

## NVIDIA GPU execution

The locked Linux PyTorch installation includes CUDA support. On a Linux host
with an NVIDIA driver, NVIDIA Container Toolkit, and GPU-capable Docker Compose,
add the GPU override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  run --rm graphragx graphragx --default
```

Without the override, PyTorch uses CPU. Apple Metal/MPS is not exposed through
Linux Docker containers, so Docker Desktop on macOS also uses CPU.

## Rebuild after code or dependency changes

```bash
docker compose build graphragx
```

The image is built from `uv.lock`, including the development dependency group.
That group contains `torch-geometric`, which is required by the runtime
architectures, as well as the Matplotlib and ReportLab dependencies used by the
result and architecture-figure scripts.
