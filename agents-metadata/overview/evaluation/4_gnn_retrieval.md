# GNN Candidate Answer Retrieval

The trained GNN scores every node in the WebQSP local graph as a possible answer entity.

The output is a ranked list of candidate answer nodes. The first selection strategy should be top-k by probability, with `top_k` set to a small value such as `5` or `10`.

The GNN is only the retriever/ranker. It narrows the answer candidates before shortest-path extraction and LLM answering.

Retrieval metrics should be computed at this stage:
- Hits@1
- Hits@K / answer coverage
- optional precision, recall, and F1 over selected candidate nodes
