## WebQSP Local Graph Construction

This step converts each WebQSP example into a local graph object for training or inference. It does not standardize a global KG yet.

For each example, collect every unique head and tail entity from the `graph` triples, map those entity strings to local integer node IDs, and convert each triple into a directed edge.

The output for each example should include:
- local node ID mapping
- `edge_index`
- relation text per edge
- binary node labels where answer entities are labeled `1`
- question text and topic entities carried alongside the graph

Node labels are created from `a_entity`: every local node whose entity string is in `a_entity` is positive, and every other local node is negative.
