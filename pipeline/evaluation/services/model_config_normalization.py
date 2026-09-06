"""Canonical model-config views used for persistence compatibility and W&B."""

from __future__ import annotations

from typing import Any

from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    infer_gnn_architecture,
)


_TRAINING_KEYS = (
    "epochs",
    "learning_rate",
    "weight_decay",
    "max_instances",
    "log_every",
    "batch_size",
    "device",
    "profile",
    "random_seed",
    "embedding_cache_device",
    "embedding_cache_dtype",
    "loss_function",
    "loss_history",
    "trained_instances",
)

_MODEL_TRAINING_OUTCOME_KEYS = (
    "final_loss",
)


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return one canonical, de-duplicated model configuration.

    New model files already use this shape. Older files are projected into it
    without being modified, so loading historical runs remains compatible.
    """
    if not isinstance(config, dict):
        return {}

    raw = {key: value for key, value in config.items() if key != "wandb"}
    raw_training = raw.get("training")
    training = raw_training if isinstance(raw_training, dict) else {}

    try:
        architecture = infer_gnn_architecture(raw)
    except (KeyError, ValueError):
        architecture = raw.get("gnn_architecture") or "graphsage"
    architecture = str(architecture)

    persisted_options = raw.get("gnn_architecture_options")
    if not isinstance(persisted_options, dict):
        persisted_options = training.get("gnn_architecture_options", {})
    if not isinstance(persisted_options, dict):
        persisted_options = {}

    legacy_values = {
        "gnn_layer_count": raw.get("gnn_layer_count", training.get("gnn_layer_count")),
        "gnn_hidden_dimension": raw.get(
            "gnn_hidden_dimension",
            training.get("gnn_hidden_dimension", training.get("hidden_dimension")),
        ),
        "node_classifier": raw.get("node_classifier", training.get("node_classifier")),
        "dropout": raw.get("dropout", training.get("dropout")),
        "use_edge_mlp": raw.get("use_edge_mlp", training.get("use_edge_mlp")),
        "use_reverse_edges": raw.get(
            "use_reverse_edges", training.get("use_reverse_edges")
        ),
        "question_aware_classifier": raw.get(
            "question_aware_classifier", training.get("question_aware_classifier")
        ),
        "add_layer_normalization": raw.get(
            "add_layer_normalization", training.get("add_layer_normalization")
        ),
        "edge_mlp_hidden_dim": raw.get(
            "edge_mlp_hidden_dim", training.get("edge_mlp_hidden_dim")
        ),
        "num_bases": raw.get("num_bases", training.get("num_bases")),
        "attention_heads": raw.get(
            "attention_heads", training.get("attention_heads")
        ),
        "num_instructions": raw.get(
            "num_instructions", training.get("num_instructions")
        ),
        "reasoning_steps": raw.get(
            "reasoning_steps", training.get("reasoning_steps")
        ),
        "adaptive_iterations": raw.get(
            "adaptive_iterations", training.get("adaptive_iterations")
        ),
    }
    try:
        defaults = architecture_defaults(architecture)
    except KeyError:
        defaults = {}
    defaults.pop("gnn_architecture", None)
    supported = GNN_ARCHITECTURES.get(architecture)
    supported_ids = set(supported.option_map) if supported is not None else set(defaults)
    architecture_options: dict[str, Any] = {}
    for option_id, default in defaults.items():
        if option_id not in supported_ids:
            continue
        if option_id in persisted_options:
            architecture_options[option_id] = persisted_options[option_id]
            continue
        legacy_value = legacy_values.get(option_id)
        architecture_options[option_id] = (
            default if legacy_value is None else legacy_value
        )
    if architecture_options.get("use_edge_mlp") is False:
        architecture_options["edge_mlp_hidden_dim"] = None
    architecture_options = {
        key: value for key, value in architecture_options.items() if value is not None
    }

    embedding_model = (
        raw.get("embedding_model")
        or raw.get("entity_embedding_model")
        or raw.get("question_embedding_model")
        or raw.get("relation_embedding_model")
    )
    embedding_dimension = (
        raw.get("embedding_dimension")
        or raw.get("entity_embedding_dimension")
        or raw.get("question_embedding_dimension")
        or raw.get("relation_embedding_dimension")
    )

    canonical: dict[str, Any] = {}
    for key in (
        "run_name",
        "run_number",
        "is_fine_tuned_model",
        "continued_from_model_run_name",
        "continued_from_model_run_number",
    ):
        if key in raw and raw[key] is not None:
            canonical[key] = raw[key]
    canonical.update(
        {
            "gnn_architecture": architecture,
            "gnn_architecture_options": architecture_options,
        }
    )
    architecture_context = raw.get("gnn_architecture_context")
    if isinstance(architecture_context, dict) and architecture_context:
        canonical["gnn_architecture_context"] = architecture_context
    if embedding_model is not None:
        canonical["embedding_model"] = embedding_model
    if embedding_dimension is not None:
        canonical["embedding_dimension"] = embedding_dimension
    for key in _MODEL_TRAINING_OUTCOME_KEYS:
        if key in raw:
            canonical[key] = raw[key]
        elif key in training:
            canonical[key] = training[key]
    for key in ("parameter_count", "estimated_training_parameter_bytes"):
        if key in raw:
            canonical[key] = raw[key]

    canonical_training: dict[str, Any] = {}
    for key in _TRAINING_KEYS:
        if key == "loss_history":
            value = training.get("loss_history") or raw.get("loss_history")
            if isinstance(value, list):
                canonical_training[key] = value
        elif key == "trained_instances":
            continue
        elif key in training:
            canonical_training[key] = training[key]
        elif key in raw:
            canonical_training[key] = raw[key]

    has_training_instance_metadata = any(
        key in raw or key in training
        for key in (
            "trained_instances",
            "training_start_instance",
            "training_end_instance",
            "trained_instance_range",
        )
    )
    training_instances = training.get("trained_instances")
    if isinstance(training_instances, dict):
        start = training_instances.get("start", 0)
        end = training_instances.get("end", 0)
        count = training_instances.get("count", 0)
    else:
        legacy_range = raw.get("trained_instance_range")
        if not isinstance(legacy_range, dict):
            legacy_range = training.get("trained_instance_range", {})
        start = raw.get("training_start_instance", legacy_range.get("start", 0))
        end = raw.get("training_end_instance", legacy_range.get("end", 0))
        legacy_count = raw.get("trained_instances")
        if isinstance(legacy_count, int):
            count = legacy_count
        elif isinstance(training_instances, int):
            count = training_instances
        else:
            count = max(int(end or 0) - int(start or 0), 0)
        # Older model configs sometimes stored only the scalar count.  Preserve
        # the canonical half-open range by deriving its end from the start.
        if not end and count:
            end = int(start or 0) + int(count)
    if has_training_instance_metadata:
        canonical_training["trained_instances"] = {
            "start": int(start or 0),
            "end": int(end or 0),
            "count": int(count or 0),
        }
    if canonical_training:
        canonical["training"] = canonical_training
    return canonical
