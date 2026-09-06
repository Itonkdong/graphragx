# Pipeline Overview

This file connects the step summaries across the training and evaluation flow.

## Core Idea

WebQSP is the central dataset. Each example already contains a question-specific graph, so the pipeline does not train on a separate Freebase KG and does not build a new local subgraph for the benchmark.

The GNN is an answer-entity retriever. The LLM receives shortest reasoning paths for the retrieved candidates and produces the final answer.

## Training Stage

### 1. WebQSP Dataset Scope

Select and load WebQSP examples with `question`, `q_entity`, `a_entity`, and `graph`.

See: [1_knowledge_graph_dataset_scope.md](preparation/1_knowledge_graph_dataset_scope.md)

### 2. WebQSP Local Graph Construction

Convert each example's `graph` triples into a local graph with node IDs, edges, relation texts, and answer-node labels.

See: [2_knowledge_graph_dataset_standardization.md](preparation/2_knowledge_graph_dataset_standardization.md)

### 3. Train GNN Answer Retriever

Train a node classifier that ranks answer entities inside each WebQSP local graph using entity embeddings, relation embeddings, and question-relation edge weights.

See: [3_train_gnn_and_graph_embeddings.md](preparation/3_train_gnn_and_graph_embeddings.md)

## Evaluation Stage

### 1. WebQSP Test Dataset Scope

Load WebQSP test examples; `a_entity` is used only for evaluation metrics.

See: [1_question_answer_dataset_scope.md](evaluation/1_question_answer_dataset_scope.md)

### 2. WebQSP Test Example Preparation

Build the same local graph representation used during training.

See: [2_question_answer_dataset_standardization.md](evaluation/2_question_answer_dataset_standardization.md)

### 3. Question and Relation Signal Preparation

Embed the question, embed relations, and compute question-relation edge weights.

See: [3_query_transformation.md](evaluation/3_query_transformation.md)

### 4. GNN Candidate Answer Retrieval

Use the trained GNN to rank candidate answer nodes and select top-k candidates.

See: [4_gnn_retrieval.md](evaluation/4_gnn_retrieval.md)

### 5. Reasoning Path Extraction

Find shortest paths from `q_entity` to each candidate using the synthetic global KG, with local graph fallback.

See: [5_subgraph_construction.md](evaluation/5_subgraph_construction.md)

### 6. Reasoning Path Verbalization

Turn extracted paths into compact text evidence for the LLM.

See: [6_textualization.md](evaluation/6_textualization.md)

### 7. LLM Query

Ask the LLM to answer using only the original question and the verbalized reasoning paths.

See: [7_llm_query.md](evaluation/7_llm_query.md)

### 8. Answer and Reasoning Output

Return the final answer with the reasoning paths used as evidence.

See: [8_answer_and_explanation.md](evaluation/8_answer_and_explanation.md)

### 9. Evaluation

Measure GNN retrieval, path extraction, final answer quality, and grounding.

See: [9_evaluation.md](evaluation/9_evaluation.md)

## End-to-End Flow

`WebQSP train examples`
-> `local graph construction`
-> `GNN answer-retriever training`
-> `WebQSP test example`
-> `question/relation signals`
-> `GNN candidate retrieval`
-> `shortest reasoning paths`
-> `path verbalization`
-> `LLM final answer`
-> `evaluation`
