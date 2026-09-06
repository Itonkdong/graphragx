# Metrics reference

Metrics are grouped by the stage whose information they evaluate:

- [Retrieval](retrieval.md) — whether and where the GNN ranked gold entities.
- [Evidence context](evidence-context.md) — how much of the retrieved pool and
  gold set survives evidence construction.
- [Final answer](final-answer.md) — generated-answer quality, explanation
  grounding, and retrieval/context-conditioned LLM behavior.

## Shared answer normalization

Entity answers are compared after lowercasing, trimming, removing ASCII
punctuation, removing the English articles `a`, `an`, and `the`, and collapsing
whitespace. Duplicates are removed for set metrics. Generated answers must be
stored as an array of strings; commas inside an entity name are not interpreted
as separators and the old single-string format is rejected.

An empty value and normalized `unknown`, `n/a`, `none`, or `null` are treated as
no generated answer. If an LLM request fails, its row is kept and evaluated as
an empty prediction. It therefore contributes false negatives and counts as an
incorrect answer.

## Denominators

`evaluated_instances` is the number of persisted predictions or final answer
rows included in the relevant evaluation.

`conditioned_evaluated_instances` is the subset with at least one non-empty
normalized gold answer. Gold-coverage and retrieval/context-conditioned metrics
use this set unless their name says `given_full_*`.

By default, training and evaluation remove any local graph missing at least one
labeled gold answer. The opt-out flag
`--no-skip-missing-gold-in-graph` restores inclusion of incomplete graphs. The
skipped count is persisted separately and does not enter the evaluated-instance
denominator.

Rates with an ordinary zero denominator are saved as `0.0`. Conditional
`given_full_*` metrics and `llm_retrieved_gold_utilization` are saved as `null`
when their conditioning denominator does not exist; non-numeric values are not
sent as W&B scalars.

## Availability by run mode

| Metric group | Full | Train only | Retriever only | Evaluation only | Evidence only | Inference only |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Training loss | yes | yes | yes | copied when available | copied when available | copied when available |
| Retrieval | yes | no | yes | yes | copied from selected retriever | copied from selected retriever |
| Evidence size/selection | yes | no | no | yes if inference runs | yes | yes |
| Context gold coverage | yes | no | no | yes if inference runs | yes | yes |
| Final answer and conditioned LLM | yes | no | no | yes if inference runs | no | yes |

`--evaluation-only --no-llm-inference` produces retrieval metrics only.

## Persistence layers

- Evaluation-level retrieval metrics:
  `data/webqsp/evaluations/<run>/retrieval_metrics.json`.
- Evidence-only metrics:
  `data/webqsp/evidence/<run>/evidence_metrics.json`.
- Inference evidence aggregates and token usage:
  `data/webqsp/inference/<run>/inference_config.json`.
- Final flattened metrics:
  `data/webqsp/results/<run>/reasoning_metrics.json`.
- Final per-question values:
  `data/webqsp/results/<run>/per_instance_results.jsonl`.

W&B uses exact snake_case implementation keys. The PascalCase names used in
result presentations are aliases and do not change persisted or W&B keys.
