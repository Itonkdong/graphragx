## Train GNN Answer Retriever

The GNN is trained as an answer-entity retriever over WebQSP local graphs. It does not generate the final natural-language answer.

For each training example, entity-name embeddings become node features, the question is embedded once, and relation texts are embedded after simple preprocessing. Question-relation relevance is used as an edge weight, starting with cosine similarity.

The model predicts one logit per node:

```text
classifier(node_hidden_state) -> answer logit
```

Training uses binary node classification with `BCEWithLogitsLoss` over all nodes in the local graph. Positive nodes are the gold answer entities from `a_entity`, so `pos_weight` should be considered because answer nodes are sparse.

Trainable parts:
- optional projection from text embedding dimension to hidden dimension
- GNN layers
- node classifier
- optional omega MLP later

Frozen parts:
- external text embedding model
- cached entity, relation, and question embeddings

The saved artifacts should include model weights, embedding caches, and configuration values such as hidden dimension, number of layers, `top_k`, and threshold settings.
