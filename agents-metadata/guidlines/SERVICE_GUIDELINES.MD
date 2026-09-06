# Service Guidelines

- Put business logic, orchestration, validation, and domain state transitions in services or pipeline modules, not in thin entrypoints such as CLI handlers, notebooks, scripts, or temporary experiment glue.
- Use services for bounded domain operations and reusable workflow logic.
- Use pipelines for multi-step execution flows such as retrieval, transformation, prompting, reasoning, evaluation, or experiment runs.
- Keep entrypoints thin: parse inputs, construct dependencies, invoke the owning service or pipeline, and shape outputs.
- Validation and data formatting helpers may prepare inputs and outputs, but they should not orchestrate domain workflows that belong in services or pipelines.
- When one service or pipeline needs another, make the dependency explicit through constructor injection, owned fields, or a clear module boundary.
- Service and pipeline methods should be typed. New non-trivial methods should have clear docstrings describing purpose, inputs, outputs, and raised domain exceptions.
- Keep service and pipeline methods explicit about the boundary they own. If logic belongs to a more specific component, move it there instead of duplicating wrappers in a broader module.
- Raise domain-specific custom exceptions for expected failures. Preserve more specific downstream exceptions instead of flattening them into generic framework errors.
- Keep logs in services, pipelines, and integration adapters for business failures, state transitions, integrations, and recoverable/internal error paths.
- Avoid hidden side effects in unrelated layers. If a module owns a workflow, it should be the clear entry point for that workflow.
- Reuse existing service or pipeline methods when an appropriate one already exists. If the correct domain method does not exist, add it in the appropriate owning module instead of bypassing the framework structure.
- Design services and pipelines so they can be composed for experiments, benchmarks, and alternate retrieval/reasoning strategies without rewriting the surrounding flow.
