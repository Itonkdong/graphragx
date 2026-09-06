# LLM Query for Final Answer

The LLM receives the original question and the verbalized reasoning paths, then selects the final answer.

Recommended prompt shape:

```text
Question:
{question}

Reasoning paths:
{verbalized_paths}

Use only the reasoning paths to answer the question.
```

The LLM is the final answer generator. The GNN only provides candidate answer nodes and supporting graph paths.
