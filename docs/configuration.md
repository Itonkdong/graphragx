# Configuration reference

The executable reference is always available through:

```bash
uv run graphragx --help
```

Without `--default`, unresolved user choices are requested interactively. With
`--default`, the recommended configuration is WebQSP, GraphSAGE, the
`text-embedding-3-small` embedding model where required, shortest-path
evidence, structured triples, OpenAI, and `gpt-4.1-nano`.

## Run selection

The following flags are mutually exclusive:

| Flag | Purpose |
| --- | --- |
| `--full` | Train, evaluate retrieval, build evidence, run the LLM, and compute final results. This is the default mode. |
| `--train-only` | Stop after saving a trained model. |
| `--retriever-only` | Train and evaluate retrieval, then stop. |
| `--evaluation-only` | Load a saved model and evaluate it; continues to inference unless `--no-llm-inference` is set. |
| `--evidence-only` | Load a saved retriever, build and evaluate evidence, and make no LLM calls. |
| `--inference-only` | Load a saved retriever and run evidence construction, LLM inference, and final evaluation. |

## General and generation options

| Option | Meaning |
| --- | --- |
| `--dataset DATASET` | Dataset id. Only `WebQSP` is currently supported. |
| `--seed INTEGER` | Non-negative Python, NumPy, and PyTorch seed; default `42`. |
| `--llm-provider {openai,deepseek,vezilka}` | Inference provider; OpenAI is the default when omitted. |
| `--main-llm-model ID` | LLM model id. Vezilka accepts a free-form hosted id. |
| `--reasoning-effort VALUE` | Optional provider argument; omitted entirely when the flag is absent. |
| `--generate-explanation` | Request explanations. Without it, explanation output and metrics are zero/empty. |
| `--context-strategy structured_triples` | LLM context representation. This is the only supported strategy. |
| `--llm-inference-batch-size N` | Number of completed rows persisted per batch; default `10`. |
| `--llm-inference-parallel-calls N` | Maximum concurrent calls inside a batch; default `1`. |
| `--inference-run-name NAME` | Label for a new inference run. |

DeepSeek model ids require `--llm-provider deepseek`. Unknown/private model ids
require `--llm-provider vezilka`. Evidence-only mode rejects all LLM-specific
options.

## Evidence options

| Option | Meaning |
| --- | --- |
| `--subgraph-algorithm {shortest_path,pcst}` | Evidence strategy; default `shortest_path`. |
| `--pcst-edge-cost-strategy {constant,semantic}` | PCST cost strategy; default `constant`. |
| `--pcst-edge-cost FLOAT` | Finite positive lambda; default `1.0` for PCST. |
| `--pcst-debug-profile` | Save a replayable diagnostic only when PCST returns malformed rooted output. |
| `--evidence-run-name NAME` | Label for a new evidence-only run. |

PCST-specific options are rejected unless `--subgraph-algorithm pcst` is
selected. The semantic strategy resolves an embedding model even when the
retriever architecture itself does not need one.

## Retriever architecture options

Select an architecture with:

```text
--gnn-architecture {graphsage,aa-graphsage,rgcn,hgt,rearev,nbfnet}
```

Only options owned by the selected architecture are accepted.

| Architecture | Options and defaults |
| --- | --- |
| `graphsage` | `--gnn-layers` 2; `--gnn-hidden-dim` 256; `--node-classifier` `mlp`; `--dropout` 0.1 |
| `aa-graphsage` | GraphSAGE options plus enabled edge MLP, reverse edges, question-aware classifier, layer normalization, and `--edge-mlp-hidden-dim` 256 |
| `rgcn` | layers 2; width 256; dropout 0.1; `--num-bases` 30 |
| `hgt` | layers 2; width 256; dropout 0.1; `--attention-heads` 8 |
| `rearev` | width 50; dropout 0.1; `--num-instructions` 2; `--reasoning-steps` 2; `--adaptive-iterations` 3 |
| `nbfnet` | layers 3; width 32 |

Valid shared choices are:

- standard GNN layers: `2`, `3`; NBFNet also supports `4`, `6`;
- standard hidden dimensions: `128`, `256`, `512`; ReaRev also supports `50`,
  and NBFNet supports `32`, `64`, `128`, `256`;
- classifiers: `mlp`, `linear`;
- dropout: `0.0`, `0.1`, `0.2`, `0.3`, `0.5`;
- R-GCN bases: `8`, `16`, `30`, `64`;
- HGT heads: `1`, `2`, `4`, `8`, with hidden width divisible by the head count.

Advance GraphSAGE boolean options have paired negative flags:
`--no-use-edge-mlp`, `--no-use-reverse-edges`,
`--no-question-aware-classifier`, and `--no-add-layer-normalization`.

`--embedding-model` is the unified selector for all OpenAI text embeddings.
Saved architecture and embedding configuration is authoritative when a model or
retriever is reused; explicitly conflicting values are rejected.

## Training options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--training-epochs N` | 3 | Training epochs. |
| `--training-learning-rate FLOAT` | 0.001 | Optimizer learning rate. |
| `--training-weight-decay FLOAT` | 0.0 | Optimizer weight decay. |
| `--training-max-instances N` | all | Limit the training slice. |
| `--training-start-instance N` | 0 | Zero-based start of the training slice. |
| `--training-batch-size N` | 1 | Disconnected graph batch size where supported. |
| `--training-device {auto,cpu,cuda,mps}` | `auto` | Training device. |
| `--training-log-every N` | 10 | Console progress interval; `0` disables it. |
| `--training-profile` | off | Synchronize and report phase timings. |
| `--training-embedding-cache-device {auto,gpu,cpu}` | `auto` | Compact embedding placement. |
| `--training-embedding-cache-dtype {auto,float32,bfloat16}` | `auto` | Compact embedding storage precision. |
| `--training-gpu-cache-reserve-gb FLOAT` | 6.0 | VRAM excluded from automatic embedding placement. |
| `--training-run-name NAME` | generated | Saved run label. |
| `--continue-training-model-run-name NAME` | none | Continue a saved model selected by name/suffix. |
| `--continue-training-model-run-number N` | none | Continue a saved model selected by numeric prefix. |

Continuation selectors are mutually exclusive and are not valid in
evaluation-only, evidence-only, or inference-only modes.

## Evaluation options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--evaluation-model-run-name NAME` | none | Model selected by name/suffix. |
| `--evaluation-model-run-number N` | none | Model selected by numeric prefix. |
| `--answer-threshold FLOAT` | 0.5 | Probability threshold for candidate selection. |
| `--candidate-top-k N` | 10 | Minimum candidate count; must be at least 10. |
| `--candidate-limit N`, `--limit N` | 15 | Maximum retained candidates. |
| `--evaluation-max-instances N` | all | Limit evaluated test instances. |
| `--evaluation-log-every N` | 5 | Console progress interval. |
| `--evaluation-profile` | off | Synchronize and report phase timings. |
| `--evaluation-embedding-cache-device {auto,gpu,cpu}` | `auto` | Evaluation embedding placement. |
| `--evaluation-embedding-cache-dtype {auto,float32,bfloat16}` | `auto` | Evaluation embedding precision. |
| `--evaluation-gpu-cache-reserve-gb FLOAT` | 6.0 | VRAM excluded from automatic placement. |
| `--evaluation-run-name NAME` | generated | Saved evaluation label. |
| `--no-skip-missing-gold-in-graph` | absent | Opt out of the default incomplete-gold filter. |

Evidence-only and inference-only require one of `--retriever-run-name` or
`--retriever-run-number`. They reject model selectors because they consume
persisted retriever predictions, not a model checkpoint.

## W&B and execution helpers

| Option | Meaning |
| --- | --- |
| `--no-wandb` | Keep local outputs but skip W&B. |
| `--wandb-project`, `--wandb-entity`, `--wandb-mode` | Override W&B destination and mode. |
| `--wandb-training-log-every N` | Live loss interval; `0` disables live events. |
| `--wandb-upload-retriever` | Include large model weights in the W&B artifact. Off by default. |
| `--default` | Resolve configurable user selections to recommended values. |
| `--force-default` | Force each pipeline step through its non-interactive execution path; primarily for tests. |
| `--local-graph-profile` | Report graph cache and preprocessing timings. |
| `--no-llm-inference` | Stop full/evaluation-only runs after retrieval. |
