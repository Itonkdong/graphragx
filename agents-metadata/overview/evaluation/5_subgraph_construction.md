# Reasoning Path Extraction

After GNN candidate retrieval, extract shortest reasoning paths from the provided topic entity or entities to each candidate answer node.

Before evaluation, build a synthetic global KG by taking the union of triples from WebQSP train, validation, and test graph columns. This graph is used only for path extraction, not for GNN training.

For each candidate:
- find the shortest path from any `q_entity` to the candidate
- allow reverse traversal if directed search fails
- fallback to the local WebQSP graph if no synthetic global path exists

The output is a small set of paths that connect the question entities to the candidate answer entities.
