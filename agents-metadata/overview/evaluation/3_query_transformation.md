## Question and Relation Signal Preparation

This step prepares the query-side signals needed by the trained GNN retriever.

For each test example, embed the question and preprocess each relation text, for example:

```text
people.person.sibling_s -> people person sibling
```

Then compute the edge weight for each relation using the same omega function used in training. The first version should use cosine similarity between the question embedding and relation embedding.

The output is the local graph with node features and question-aware edge weights ready for GNN scoring.
