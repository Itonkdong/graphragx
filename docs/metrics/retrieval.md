# Retrieval metrics

Retrieval metrics are computed from the persisted ranked candidate lists. They
are available as soon as retriever evaluation finishes; no evidence construction
or LLM call is required.

Let `N` be `evaluated_instances`, `G_q` the normalized gold-answer set for
question `q`, and `C_q[:k]` the first `k` ranked candidates.

## Hit metrics

For `k` equal to 1, 5, 10, or the configured candidate limit:

```text
hit_q@k = 1 if G_q intersects C_q[:k], otherwise 0
hits_at_k = sum(hit_q@k) / N
```

| Local key | Count key | W&B `Summary_Plots` key |
| --- | --- | --- |
| `hits_at_1` | `hits_at_1_count` | `retrieval_hits_at_1` / `retrieval_hits_at_1_count` |
| `hits_at_5` | `hits_at_5_count` | `retrieval_hits_at_5` / `retrieval_hits_at_5_count` |
| `hits_at_10` | `hits_at_10_count` | `retrieval_hits_at_10` / `retrieval_hits_at_10_count` |
| `hits_at_candidate_limit` | `hits_at_candidate_limit_count` | `retrieval_hits_at_candidate_limit` / `retrieval_hits_at_candidate_limit_count` |

The full W&B key is `Summary_Plots/<name>`. `Run_Summary` includes
`retrieval_hits_at_1`, `retrieval_hits_at_10`, and
`retrieval_hits_at_candidate_limit`; Hits@5 remains in `Summary_Plots` only.

## Ranking quality

`ndcg_at_1`, `ndcg_at_5`, `ndcg_at_10`, and
`ndcg_at_candidate_limit` use binary relevance from each candidate's
`is_gold_answer` flag:

```text
DCG@k  = sum(relevance_i / log2(i + 1)), for one-based rank i
IDCG@k = DCG of min(number of distinct labeled gold answers, k) leading ones
nDCG@k = DCG@k / IDCG@k
```

An empty candidate list or question without a labeled gold answer receives
`0.0`. The aggregate is the macro mean over all `N` predictions.

W&B keys are `Summary_Plots/ranking_ndcg_at_1`,
`ranking_ndcg_at_5`, `ranking_ndcg_at_10`, and
`ranking_ndcg_at_candidate_limit`. Only `ranking_ndcg_at_10` is duplicated in
`Run_Summary`.

## Gold-answer coverage

For each question with a non-empty normalized gold set:

```text
retrieval_coverage_q = |G_q intersect C_q| / |G_q|
```

| Local key | Meaning | Denominator |
| --- | --- | --- |
| `conditioned_evaluated_instances` | Questions with at least one normalized gold answer. | none; count |
| `retrieval_gold_coverage` | Macro mean of `retrieval_coverage_q`. | conditioned questions |
| `retrieval_full_gold_coverage_count` | Questions where every gold answer is retrieved. | none; count |
| `retrieval_full_gold_coverage_rate` | Share with complete gold retrieval. | conditioned questions |
| `retrieved_gold_answer_count` | Sum of distinct retrieved gold entities across questions. | none; count |

These are not the same as Hits: a question with one of three gold answers has a
hit but only one-third coverage and is not fully covered.

The same local names appear under `Summary_Plots`. `Run_Summary` contains
`retrieval_gold_coverage` and renames the full-coverage rate to
`retrieval_full_gold_coverage` (without `_rate`).

## Volume and graph-completeness metrics

| Local key | Meaning | W&B `Summary_Plots` name |
| --- | --- | --- |
| `evaluated_instances` | Number of evaluated predictions. | `retrieval_evaluated_instances` |
| `candidate_limit` | Configured maximum candidate count. | stored in local/config artifacts, not mapped as a scalar |
| `average_candidate_count` | Total retained candidates divided by `N`. | `retrieval_average_candidate_count` |
| `missing_gold_in_graph_count` | Source instances where at least one labeled gold answer is absent from the local graph. | `retrieval_missing_gold_in_graph_count` |
| `skipped_missing_gold_in_graph_count` | Such incomplete instances removed by the default policy before evaluation. | `retrieval_skipped_missing_gold_in_graph_count` |

`Run_Summary` also contains `retrieval_evaluated_instances`; the other volume
and missing-gold values remain in `Summary_Plots`.

## Storage and modes

The complete payload is saved at
`data/webqsp/evaluations/<run>/retrieval_metrics.json` and copied to a final
`data/webqsp/results/<run>/retrieval_metrics.json` when final results are built.

It is computed directly in full, retriever-only, and evaluation-only modes.
Evidence-only and inference-only copy the selected retriever's metrics into the
new W&B run, preserving stage-to-stage comparability.
