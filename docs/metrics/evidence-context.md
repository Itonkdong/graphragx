# Evidence-context metrics

Evidence metrics describe the transition from ranked retriever candidates to
the triples actually exposed to the LLM. For shortest paths, a candidate is
selected when a usable rooted path contributes it to the evidence. For PCST, a
candidate is selected only when the solver collects it; retrieval alone does
not imply context visibility.

## Per-instance construction data

Every evidence row has a `construction` object with these exact fields:

| Key | Meaning |
| --- | --- |
| `strategy` | `shortest_path` or `pcst`. |
| `edge_cost_strategy` | `constant` or `semantic` for PCST; otherwise `null`. |
| `edge_cost_lambda` | PCST lambda; otherwise `null`. |
| `semantic_embedding_model` | Embedding model used by semantic PCST. |
| `input_candidate_count` | Ranked candidates received from the retriever. |
| `valid_candidate_count` | Candidates present and reachable in the structural graph. |
| `selected_candidate_count` | Candidates represented in the final evidence. |
| `selected_candidate_ranks` | Original one-based ranks selected into evidence. |
| `collected_prize` | Sum of selected PCST candidate prizes. |
| `selected_node_count` | Distinct selected evidence nodes. |
| `selected_triple_count` | Directed triples emitted as evidence. |
| `total_edge_cost` | Sum of selected PCST structural-edge costs. |
| `objective` | `collected_prize - total_edge_cost`. |
| `valid_seed_count` | Question entities present in the graph. |
| `missing_seed_count` | Question entities absent from the graph. |
| `construction_time_ms` | Construction wall time for the instance. |
| `empty_result_reason` | Reason for an empty evidence list, otherwise `null`. |

`candidate_evidence_coverage` is derived as:

```text
selected_candidate_count / input_candidate_count
```

and is `0.0` when the retriever provided no candidates.

The row analytics also retain `found_reasoning_paths` and
`missing_reasoning_paths`. Across a run, their sum is the number of candidate
decisions, not the number of evaluated questions. In PCST, “missing” includes
valid retrieved candidates deliberately not collected by the optimized tree.

## Aggregate size and selection metrics

Let `N` be the number of evidence rows.

| Local key | Definition |
| --- | --- |
| `average_subgraph_triples` | Mean number of emitted triples per question. |
| `average_distinct_nodes` | Mean number of distinct source/target entities in the evidence. |
| `average_candidate_evidence_coverage` | Macro mean of per-question `selected / input` candidate coverage. |
| `candidate_reduction_percentage` | `100 * total_missing_candidates / (total_found_candidates + total_missing_candidates)`. |
| `empty_subgraph_count` | Questions with zero evidence triples. |
| `empty_subgraph_rate` | `empty_subgraph_count / N`. |
| `average_construction_time_ms` | Mean construction time per question. |

The coverage metric is a macro average, while candidate reduction pools the
found/missing candidate counts before division. They can therefore differ
slightly even though both describe candidate survival.

## Context gold coverage

For a question with normalized gold set `G_q`, let `E_q` be the normalized
source and target entities that occur in the evidence triples:

```text
context_gold_coverage_q = |G_q intersect E_q| / |G_q|
```

| Local key | Meaning |
| --- | --- |
| `reasoning_context_gold_coverage` | Macro mean over questions with non-empty gold sets. |
| `reasoning_context_full_gold_coverage_rate` | Share of those questions where `G_q` is a subset of `E_q`. |

These metrics answer a different question from retrieval coverage: a gold
entity may be retrieved by the GNN but absent from PCST evidence, or included
only when the selected connecting triples expose it.

Evidence-only runs calculate these values directly in `evidence_metrics.json`.
Final inference recalculates them from the persisted `reasoning.jsonl` triples
and stores them in final `reasoning_metrics.json`.

## PCST-only aggregates

When at least one row uses PCST, three additional macro means are present:

| Local key | Meaning |
| --- | --- |
| `average_collected_prize` | Mean sum of selected linear-rank prizes. |
| `average_edge_cost` | Mean total selected edge cost. |
| `average_objective` | Mean `collected_prize - total_edge_cost`. |

These values compare PCST settings; they are not answer-quality metrics and are
not defined for shortest-path evidence.

## Storage

Evidence-only runs save aggregates in
`data/webqsp/evidence/<run>/evidence_metrics.json` and per-instance values in
`evidence_subgraphs.jsonl`.

Full, evaluation-only-with-inference, and inference-only runs save evidence
aggregates under `inference.evidence_metrics` in
`data/webqsp/inference/<run>/inference_config.json`. Their per-instance
construction data and triples are in `reasoning.jsonl`.

## W&B keys

For size, selection, timing, and PCST aggregates, the W&B key is:

```text
Summary_Plots/evidence_<local_key>
```

Examples are `Summary_Plots/evidence_average_subgraph_triples` and
`Summary_Plots/evidence_average_objective`.

Evidence-only context keys omit the `evidence_` prefix:

```text
Summary_Plots/reasoning_context_gold_coverage
Summary_Plots/reasoning_context_full_gold_coverage_rate
```

Final-result logging uses those same clean context keys. Its complete mapping
is described in [Final-answer metrics](final-answer.md).

For evidence-only runs, `Run_Summary` contains:

- `evidence_candidate_reduction_percentage`;
- `reasoning_context_gold_coverage`;
- `reasoning_context_full_gold_coverage` (the local `_rate` suffix is omitted);
- `evidence_empty_subgraph_rate`.

For runs that continue through inference, candidate reduction is added during
the inference stage. Context gold coverage is then added by final-results
logging. Other new evidence aggregates remain under `Summary_Plots` and are not
duplicated in `Run_Summary`.
