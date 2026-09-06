# Pipeline

## Processing flow

The complete pipeline has four research-facing stages:

1. **Local graph preparation.** WebQSP questions, topic entities, answer
   entities, and question-specific Freebase triples are normalized into local
   graph instances and shared vocabularies.
2. **GNN retrieval.** A retriever scores graph nodes and retains a ranked pool
   of answer candidates.
3. **Evidence construction.** The ranked candidates and question entities are
   converted into a structured evidence subgraph using shortest paths or PCST.
4. **Answer generation and evaluation.** The evidence triples and question are
   sent to an LLM. Generated answers are normalized and evaluated together with
   retrieval and context availability.

Each completed stage persists its own configuration and data so later modes can
reuse an earlier model or retriever without mutating it.

## Run modes

| Mode | Train GNN | Evaluate retriever | Build evidence | Call LLM | Final metrics |
| --- | :---: | :---: | :---: | :---: | :---: |
| `--full` (default) | yes | yes | yes | yes | yes |
| `--train-only` | yes | no | no | no | no |
| `--retriever-only` | yes | yes | no | no | no |
| `--evaluation-only` | no; loads model | yes | yes* | yes* | yes* |
| `--evidence-only` | no; loads retriever | no | yes | no | evidence only |
| `--inference-only` | no; loads retriever | no | yes | yes | yes |

`*` In `--evaluation-only`, `--no-llm-inference` stops after retriever
evaluation. `--inference-only` and `--evidence-only` require exactly one saved
retriever selector. `--evaluation-only` instead selects a saved model.

## Retriever architectures

| CLI id | Summary | Main architecture-specific options |
| --- | --- | --- |
| `graphsage` | GraphSAGE baseline with semantic question–relation edge weighting | layers, width, classifier, dropout |
| `aa-graphsage` | GraphSAGE with trainable edge scoring, reverse edges, residual normalization, and question-aware classification | GraphSAGE options plus feature toggles and edge-MLP width |
| `rgcn` | basis-decomposed relational graph convolution with categorical inverse relation types | layers, width, dropout, basis count |
| `hgt` | relation-aware multi-head heterogeneous graph attention over one entity node type | layers, width, dropout, attention heads |
| `rearev` | token-conditioned reason-and-revise execution with seed distributions and adaptive stages | width, dropout, instructions, reasoning steps, iterations |
| `nbfnet` | question-conditioned Neural Bellman-Ford path reasoning with DistMult messages and PNA aggregation | layers and width |

The architecture registry owns valid options, defaults, runtime behavior, and
input requirements. Saved model configuration remains authoritative during
evaluation and inference reuse.

## Candidate selection

Nodes above `--answer-threshold` are selected first. If that produces fewer
than `--candidate-top-k` candidates, ranked candidates are added until the
minimum is met. The final pool is capped by `--candidate-limit`. The configured
top-k must be at least 10 so Hits@5 and Hits@10 are meaningful.

By default, training and evaluation omit instances where at least one labeled
gold answer is absent from the local graph. `--no-skip-missing-gold-in-graph`
includes those incomplete instances. See [Retrieval metrics](metrics/retrieval.md)
for the affected counters and denominators.

## Evidence construction

Both strategies consume the same question-specific local graph and emit the
same directed Freebase triples.

**Shortest paths.** The baseline takes the union of candidate paths rooted at
the valid question entities. Candidates without a usable path contribute to
the missing-path count.

**PCST.** Ranked candidates receive linear prizes, `p(c_i) = k - i + 1`, and
are treated as an optional candidate pool. Connector nodes receive zero prize.
The rooted solver selects a compact structure that balances collected prize
against edge cost. With multiple valid question entities, a synthetic root
connects all seeds during solving and is removed from the final evidence.

PCST edge costs are either:

```text
constant: c(e) = lambda
semantic: c(e) = max(1e-6, lambda * (1 - cosine(question, relation)))
```

Semantic costs use normalized CPU FP32 question and relation embeddings. Solver
errors fail the run; they do not silently fall back to shortest paths. Missing
seeds, no valid candidates, and root-only solutions produce an empty evidence
list with a recorded reason so inference can continue.

## LLM generation

The current context representation is `structured_triples`. The model must
return an array of complete answer entity names. Explanations are disabled by
default; `--generate-explanation` requests and evaluates one.

Per-instance generation failures are persisted and evaluated as empty
predictions, so they count as incorrect rather than disappearing from the
denominator. Provider responses indicating exhausted credits terminate the run
immediately. Ordinary request failures remain associated with their instance.

OpenAI is the default provider. DeepSeek and Vezilka require explicit provider
selection; Vezilka accepts a free-form hosted model id.
