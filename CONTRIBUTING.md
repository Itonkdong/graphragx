# Contributing

Thank you for considering a contribution to `graphragX`.

This project is an academic GraphRAG pipeline, so contributions should preserve reproducibility, clear saved artifacts, and the existing pipeline conventions.

## Before You Start

Read the main project documentation:

- `README.md`
- `docs/index.md`
- `agents-metadata/guidlines/PROJECT_GUIDELINES.MD`
- `agents-metadata/overview/pipeline_overview.md`

The `docs/` folder contains the maintained software documentation. The
`metadata/` folder contains generated figures, tables, provenance, and other
research material.
`agents-metadata/` contains instructions for AI coding agents and should be used
as context when an agent contributes changes.

## Development Setup

Create the Python 3.11 environment and install the locked runtime and development dependencies:

```bash
uv sync
```

Run commands through the managed environment:

```bash
uv run python main.py
uv run pytest
```

When dependencies change, update `pyproject.toml` and regenerate the lockfile with `uv lock`. Commit both files so local and remote development environments resolve the same package versions. `requirements.txt` remains available only for pip compatibility.

Create a local environment file:

```bash
cp .env.example .env
```

Set the required API keys and optional W&B settings in `.env`.

## Contribution Guidelines

- Keep pipeline changes modular. Prefer adding or updating a focused step, service, or model instead of mixing responsibilities.
- Add or update tests for metric logic, storage behavior, W&B payloads, and pipeline composition when those areas change.
- Do not remove compatibility behavior unless the project explicitly decides to migrate old runs.
- Keep documentation in sync when CLI flags, output files, or metric semantics change.

## Working With Agents

If you use an AI coding agent, give it the relevant files from
`agents-metadata/` first. At minimum, include:

- `agents-metadata/guidlines/PROJECT_GUIDELINES.MD`
- `agents-metadata/guidlines/SERVICE_GUIDELINES.MD`
- `agents-metadata/guidlines/error_handling_guildline.MD`
- `agents-metadata/overview/pipeline_overview.md`

The agent metadata explains the project architecture, conventions, error-handling expectations, and the intended pipeline flow. This helps agents make changes that match the existing codebase.

## Testing

Run focused tests for the area you changed. For example:

```bash
uv run pytest tests/evaluation/test_final_results_evaluation.py -q
```

For a broader sanity check around the current final pipeline behavior:

```bash
uv run pytest tests/test_main.py tests/evaluation/test_final_results_evaluation.py tests/evaluation/test_wandb_final_results.py -q
```

Some tests require optional heavy dependencies such as PyTorch. If a dependency is missing, install the project requirements or note clearly which tests could not be run.

## Pull Request Checklist

- The change is scoped and follows the current pipeline/service conventions.
- Relevant tests were added or updated.
- Existing focused tests pass.
- The relevant file under `docs/` was updated if behavior changed.
- Generated data, model weights, local W&B files, and private environment files are not committed.

## License

By contributing, you agree that your contribution is provided under the MIT License.
