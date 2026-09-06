## WebQSP Test Example Preparation

Each test example is converted into the same local graph representation used during training.

This includes local node IDs, directed edges, relation texts, entity-name node features, question embedding, relation embeddings, and question-relation edge weights. The same embedding caches and projection layer from training must be reused.

No entity linking is needed for WebQSP because `q_entity` is already provided.
