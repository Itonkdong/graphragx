## WebQSP Dataset Scope

The preparation phase starts from WebQSP, not from a standalone Freebase subset. Each example already contains the natural language question, topic entities, gold answer entities, and a question-specific graph.

The selected dataset provides both the training supervision and the local graph structure used by the GNN retriever. For this benchmark, `q_entity` is already provided, so entity linking is not part of this step.

The required fields per example are:
- `question`: natural language question
- `q_entity`: topic/question entity or entities
- `a_entity`: gold answer entity or entities
- `graph`: triples `[head_entity, relation, tail_entity]`

The graph column is treated as the local subgraph `G_q`. The pipeline should not build a new subgraph from an external KG for GNN training.
