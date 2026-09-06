# Artifacts and W&B

## Local runtime artifacts

All runtime outputs are versioned under `data/webqsp/`.

| Directory | Main contents |
| --- | --- |
| `processed/` and `processed_reverse_edges/` | Cached local graphs and shared node/relation/question vocabularies. Reverse-edge data is separated where required. |
| `training_embedding_tensors/` | Incremental local tensor shards for reusable text embeddings. |
| `models/<run>/` | `model_config.json`, `gnn_answer_retriever.pt`, loss history, and an authoritative `relation_vocabulary.json` where required. |
| `evaluations/<run>/` | `evaluation_config.json`, ranked `predictions.jsonl`, and `retrieval_metrics.json`. |
| `evidence/<run>/` | `evidence_config.json`, `evidence_subgraphs.jsonl`, and `evidence_metrics.json` from evidence-only runs. |
| `inference/<run>/` | `inference_config.json`, `reasoning.jsonl`, and `answers.jsonl`. |
| `results/<run>/` | `results_config.json`, copied retrieval metrics, flattened reasoning metrics, and `per_instance_results.jsonl`. |
| `wandb_runs/` | Dataset-wide local identifiers used when allocating new W&B run names. |

Paths embedded in saved JSON are made project-relative when possible. A later
stage resolves its model or retriever through saved configuration lineage; it
does not rewrite the upstream artifact.

## Important row formats

`predictions.jsonl` stores the question, topic and gold entities, ranked answer
candidates, per-gold scores, and hit/missing-gold flags.

`evidence_subgraphs.jsonl` and `reasoning.jsonl` store structured directed
triples plus a `construction` object. That object records the evidence strategy,
candidate counts and selected ranks, PCST prize/cost/objective values when
applicable, seed counts, construction time, and an empty-result reason.

`answers.jsonl` stores an atomic `answers` array, optional explanation, raw
response, token counts, estimated cost, and a per-instance error. The aggregate
prompt, completion, total token counts, and estimated total cost are recorded
under `inference` in `inference_config.json`.

`per_instance_results.jsonl` joins retrieval, context, and generated-answer
facts for one question. The W&B per-instance table also includes the complete
ranked list of retrieved candidate names.

## W&B lifecycle

W&B is enabled by default and can be disabled with `--no-wandb`. A full,
train-only, or retriever-only command shares one logical W&B run across the
stages it reaches. Evaluation-only, evidence-only, and inference-only create a
new run and copy the available upstream configuration, metrics, and artifact
metadata so comparisons do not mutate the source run.

The W&B config is organized as:

```text
dataset_id
gnn_architecture
model_id / llm_provider              # when inference exists
runs.{model,evaluation,evidence,inference,results}
configs.{model,evaluation,evidence,inference}
source_paths.*
```

Evidence configuration is stored in `configs.evidence`, not nested inside the
W&B inference configuration. Local `inference_config.json` still keeps
`inference.evidence_subgraph` because that is the inference artifact's complete
lineage.

Run titles include the available architecture, evidence strategy (`sp` or
`pcst`), and inference model. Tags are accumulated without duplicates and may
include the dataset, architecture, embedding model, evidence strategy,
`pcst-constant` or `pcst-semantic`, LLM id, and stage instance/run identifiers.

## W&B metric sections

| Section | Purpose |
| --- | --- |
| `Training/*` | Live and epoch-level GNN loss. |
| `Summary_Plots/*` | One aggregate history value per metric, suitable for W&B comparison panels. |
| `Run_Summary/*` | A curated subset of comparison metrics. Names may omit the local `_rate` suffix. |
| `Summary_Metrics/aggregate_metrics` | Table form of final aggregate metrics. |
| `Per_Instance_Metrics/per_instance_results` | Final per-question table. |

Aggregate scalars are history-logged once. If a stage replaces an already
logged value in the same run, the coordinator updates the W&B summary instead
of creating a second history point. This preserves bar-style summary panels.

The precise key mapping and mode availability are documented by stage:

- [Retrieval metrics](metrics/retrieval.md)
- [Evidence-context metrics](metrics/evidence-context.md)
- [Final-answer metrics](metrics/final-answer.md)

## W&B artifacts

Depending on mode and flags, the run uploads:

- model configuration and relation vocabulary (`gnn-model`); model weights are
  included only with `--wandb-upload-retriever`;
- evaluation configuration, predictions, and retrieval metrics
  (`retriever-results`);
- evidence-only configuration, rows, and metrics (`evidence-subgraphs`);
- inference configuration, reasoning rows, and answers
  (`inference-predictions`);
- final configuration, retrieval metrics, reasoning metrics, and per-instance
  rows (`final-results`).

W&B failure metadata is persisted in the local stage configuration when
possible. Local pipeline artifacts remain usable even when optional W&B upload
fails.

## Generated research material

`metadata/` is deliberately separate from runtime artifacts:

```text
metadata/
├── figures/
├── tables/
├── results_metadata/
└── initial_project/
```

Result scripts write figures, LaTeX tables, CSV inputs, and provenance into the
matching experiment subdirectories. They read W&B but do not edit it.
