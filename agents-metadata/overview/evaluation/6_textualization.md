# Reasoning Path Verbalization

This step turns extracted paths into a compact text format for the LLM.

The first version can keep a structured path style:

```text
Justin Bieber -> people.person.sibling_s -> m.0gxnnwp
m.0gxnnwp -> people.sibling_relationship.sibling -> Jaxon Bieber
```

The goal is not to narrate the entire local graph. The LLM should receive only the shortest reasoning paths attached to the GNN-ranked candidate answers.
