# graphragX documentation

This documentation describes the current implementation. The CLI definitions,
metric calculators, persistence services, and tests are treated as the source
of truth.

## Use the pipeline

- [Getting started](getting-started.md) — installation, services, credentials,
  and first commands.
- [Docker Compose](docker.md) — one-command example execution, command
  overrides, persistent data, and optional NVIDIA GPU access.
- [Pipeline](pipeline.md) — processing stages, run modes, retriever
  architectures, evidence construction, and answer generation.
- [Configuration](configuration.md) — command-line options and validation rules.
- [Experiments](experiments.md) — TOML manifests, resumable execution, official
  experiments, probes, and result generation.

## Interpret outputs

- [Artifacts and W&B](artifacts-and-wandb.md) — local run directories,
  configuration lineage, artifacts, W&B sections, and tags.
- [Metrics overview](metrics/index.md) — shared conventions and availability by
  run mode.
- [Retrieval metrics](metrics/retrieval.md)
- [Evidence-context metrics](metrics/evidence-context.md)
- [Final-answer metrics](metrics/final-answer.md)

## Contribute

- [Development guide](development.md) — repository layout, tests, architecture
  extension points, and documentation maintenance.
- [Contributing](../CONTRIBUTING.md) — contribution checklist and agent context.

Generated figures, tables, provenance, and other research material are kept
under [`metadata/`](../metadata/); they are outputs and research records, not
the software manual.
