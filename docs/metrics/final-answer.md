# Final-answer metrics

Final metrics join the LLM `answers.jsonl`, evidence `reasoning.jsonl`, and
retriever `predictions.jsonl` by `instance_index`. They are computed in full,
inference-only, and evaluation-only runs that continue through LLM inference.

## Answer quality

Let `G_q` and `P_q` be the normalized gold and predicted answer sets.

| Local key | Definition |
| --- | --- |
| `evaluated_instances` | Number of joined answer rows. |
| `successful_answers` | Rows without a generation error. |
| `failed_answers` | Rows with a generation error. |
| `exact_match_count` | Questions where `G_q == P_q`; failed rows are forced false. |
| `accuracy` | `exact_match_count / evaluated_instances`. |
| `hit_count` | Questions where `G_q` and `P_q` intersect. |
| `hit_rate` | `hit_count / evaluated_instances`. |
| `hits_at_1_count` | Questions whose first normalized generated answer is gold. |
| `hits_at_1` | `hits_at_1_count / evaluated_instances`. |

Exact match is set-based, so answer order does not matter. Hits@1 retains the
LLM array order and checks only the first unique normalized answer.

Micro counts are pooled across questions:

```text
true_positive_count  = sum |G_q intersect P_q|
false_positive_count = sum |P_q - G_q|
false_negative_count = sum |G_q - P_q|

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
```

Zero denominators produce `0.0`. Failed generations behave like `P_q = {}` and
therefore remain in every aggregate denominator.

W&B maps the rates to:

| Local key | `Summary_Plots` key | Curated `Run_Summary` key |
| --- | --- | --- |
| `accuracy` | `answer_accuracy` | — |
| `hit_rate` | `answer_hit_rate` | `answer_hit_rate` |
| `hits_at_1` | `answer_hits_at_1` | — |
| `precision` | `answer_precision` | — |
| `recall` | `answer_recall` | — |
| `f1` | `answer_f1` | `answer_f1` |

The W&B names above are appended to `Summary_Plots/` or `Run_Summary/`.

## Explanation grounding

When explanations are requested, mentioned triples are extracted from arrow
notation and compared with normalized evidence triples. If no arrow-form triple
is found, literal normalized evidence-triple text present in the explanation is
used as a fallback.

| Local key | Meaning |
| --- | --- |
| `mentioned_triple_count` | Total triples identified in explanations. |
| `grounded_mentioned_triple_count` | Identified triples present in the evidence. |
| `grounded_explanation_count` | Questions with at least one grounded mentioned triple. |
| `grounded_explanation_rate` | Count divided by all evaluated questions. |
| `fully_grounded_explanation_count` | Questions with at least one mentioned triple and all mentioned triples grounded. |
| `fully_grounded_explanation_rate` | Count divided by all evaluated questions. |

Without `--generate-explanation`, explanations are empty and every explanation
metric is saved as zero. W&B uses
`Summary_Plots/grounding_grounded_explanation_rate` and
`Summary_Plots/grounding_fully_grounded_explanation_rate`.
`Run_Summary/grounded_explanation_rate` contains the first rate only.

## Retrieval and context availability

The final payload repeats retrieval ranking metrics and computes downstream
availability from the actual evidence triples. See [Retrieval metrics](retrieval.md)
and [Evidence-context metrics](evidence-context.md) for the base definitions.

For each question:

- `retrieved_gold_answers` is `G_q` intersected with retrieved candidates;
- `context_visible_gold_answers` is `G_q` intersected with evidence entities;
- `full_gold_retrieval` and `full_gold_context` require the corresponding set
  to equal the complete gold set;
- `retrieval_gold_coverage` and `reasoning_context_gold_coverage` are the
  respective per-question fractions.

The following aggregates use questions with at least one normalized gold answer:

| Local key | Definition |
| --- | --- |
| `conditioned_evaluated_instances` | Size of the eligible set. |
| `retrieval_gold_coverage` | Macro mean retrieved-gold fraction. |
| `retrieval_full_gold_coverage_count` / `_rate` | Complete-retrieval count and share of eligible questions. |
| `reasoning_context_gold_coverage` | Macro mean context-visible-gold fraction. |
| `reasoning_context_full_gold_coverage_count` / `_rate` | Complete-context count and share of eligible questions. |
| `retrieved_gold_answer_count` | Total distinct retrieved gold entities, pooled by question. |
| `answered_retrieved_gold_count` | Total retrieved gold entities also returned by the LLM. |
| `llm_retrieved_gold_utilization` | `answered_retrieved_gold_count / retrieved_gold_answer_count`. |

`llm_retrieved_gold_utilization` measures use of all retrievable correct
entities, whether retrieval was complete or partial. It is `null` when no gold
entity was retrieved anywhere in the evaluated set.

## Conditional LLM metrics

These metrics isolate the LLM from upstream information loss.

| Local key | Numerator | Denominator |
| --- | --- | --- |
| `llm_exact_match_given_full_retrieval` | Exact set matches | Questions with `full_gold_retrieval` |
| `llm_omission_given_full_retrieval_rate` | Full-retrieval questions where at least one gold answer is absent from `P_q` | Questions with `full_gold_retrieval` |
| `llm_exact_match_given_full_context` | Exact set matches | Questions with `full_gold_context` |
| `llm_omission_given_full_context_rate` | Full-context questions where at least one gold answer is absent from `P_q` | Questions with `full_gold_context` |

Each has a corresponding `_count` key. A missing conditioning population yields
`null`, not zero. Exact match rejects extra predicted answers; omission checks
only whether every gold answer was returned. A model can therefore avoid an
omission while still failing exact match because it added a false positive.

All numeric local keys are logged under `Summary_Plots/<local_key>`.
`Run_Summary` includes the four rates above but drops `_rate` from the two
omission names.

## Retrieval-to-generation outcomes

Every eligible question is assigned exactly one
`retrieval_generation_outcome`:

| Outcome | Condition |
| --- | --- |
| `full_retrieval_complete_answer` | All gold answers were retrieved and all were returned; extras are allowed. |
| `full_retrieval_llm_omission` | All gold answers were retrieved but at least one was not returned. |
| `partial_retrieval_fully_utilized` | Some but not all gold answers were retrieved, and every retrieved gold answer was returned. |
| `partial_retrieval_underutilized` | Some but not all gold answers were retrieved, and at least one retrieved gold answer was omitted. |
| `no_gold_retrieved_no_gold_answered` | No gold answer was retrieved and the LLM returned no gold answer. |
| `correct_without_gold_retrieval` | No gold answer was retrieved, yet the LLM returned at least one gold answer. |

Each outcome has `_count` and `_rate` aggregate keys. Rates use all
`conditioned_evaluated_instances`, so the six rates sum to one. `Run_Summary`
uses the outcome name without `_rate`; `Summary_Plots` retains the full local
rate key.

## Context-to-generation outcomes

The evidence-specific categories use gold entities actually visible in the
context:

| Local key | Condition |
| --- | --- |
| `full_context_complete_answer_count` / `_rate` | Context contains all gold answers and the LLM returns all of them; extras are allowed. |
| `full_context_llm_omission_count` / `_rate` | Context contains all gold answers but the LLM omits at least one. |
| `partial_context_fully_utilized_count` / `_rate` | Context exposes some but not all gold answers, and the LLM returns every exposed gold answer. |
| `partial_context_underutilized_count` / `_rate` | Context exposes some but not all gold answers, and the LLM omits at least one exposed gold answer. |

Rates again use all eligible questions, not only the category's context subset.
Questions with no context-visible gold answer are outside these four categories,
so these rates need not sum to one. This differs from
`llm_omission_given_full_context_rate`, whose denominator contains only
full-context questions.

All count/rate keys are available under `Summary_Plots`. `Run_Summary` contains
the four rates using names without `_rate`.

## Ranking metrics in final results

`ndcg_at_1`, `ndcg_at_5`, `ndcg_at_10`, and
`ndcg_at_candidate_limit` are copied from or recomputed identically to
retrieval nDCG. W&B uses the `Summary_Plots/ranking_*` namespace;
`Run_Summary/ranking_ndcg_at_10` is the curated ranking value.

## Storage and W&B table

Aggregates are flattened into
`data/webqsp/results/<run>/reasoning_metrics.json`. Per-question values are in
`per_instance_results.jsonl`.

W&B also creates:

- `Summary_Metrics/aggregate_metrics`, with group, metric, and value columns;
- `Per_Instance_Metrics/per_instance_results`, containing question and entity
  data, generated answers, explanation values, answer metrics, nDCG, retrieval
  and context coverage, utilization/outcome, error text, and the full ranked
  retrieved-candidate list.

Token usage and estimated cost are inference accounting fields rather than
answer-quality metrics. Per-request values are in `answers.jsonl`; run totals
are in `inference_config.json` and W&B `configs.inference`.
