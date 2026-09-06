## WebQSP Test Dataset Scope

Evaluation uses WebQSP test examples with the same fields used during training: `question`, `q_entity`, `graph`, and `a_entity`.

The `graph` field is already the question-specific local subgraph. `a_entity` is loaded for metrics only and should not influence retrieval or LLM answering during inference.
