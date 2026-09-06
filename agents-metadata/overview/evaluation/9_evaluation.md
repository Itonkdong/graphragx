# Evaluation

Evaluation is split into GNN retrieval quality and final LLM answer quality.

## GNN Retrieval Metrics

The GNN is evaluated as an answer candidate retriever over each WebQSP local graph.

Core metrics:
- Hits@1: the top predicted node is in `a_entity`
- Hits@K / answer coverage: at least one top-k candidate is in `a_entity`
- optional precision, recall, and F1 over selected candidate nodes

## Path Extraction Metrics

Path extraction should track whether the system can find supporting paths from `q_entity` to the retrieved candidates.

Useful measurements:
- path found rate
- average path length
- fallback rate to the local graph
- number of verbalized paths per question

## Final LLM Answer Metrics

The final LLM answer is evaluated against `a_entity`.

Core metrics:
- exact match or normalized answer match
- F1 over answer strings if aliases are available
- optional LLM-based judgment for answer equivalence

## Grounding Metrics

The answer should be supported by the reasoning paths given to the LLM.

Useful checks:
- whether the predicted answer appears in the candidate set
- whether the predicted answer appears in a provided path
- optional LLM-based faithfulness evaluation

This separates the two major responsibilities: the GNN should retrieve answer candidates, and the LLM should select the final answer using the extracted paths.
