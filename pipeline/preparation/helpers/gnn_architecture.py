"""Architecture resolution helpers shared by configuration and artifact loaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from pipeline.preparation.helpers.configuration_definitions import (
    AA_GRAPH_SAGE_ARCHITECTURE_ID,
    GNN_ARCHITECTURES,
    GRAPH_SAGE_ARCHITECTURE_ID,
)

ADVANCED_GNN_BOOLEAN_FIELDS = (
    "use_edge_mlp",
    "use_reverse_edges",
    "question_aware_classifier",
    "add_layer_normalization",
)


@dataclass(frozen=True)
class GnnArchitectureOptionValidationError(ValueError):
    """Validation error tied to the option that should be corrected."""

    option_id: str
    message: str

    def __str__(self) -> str:
        return self.message


def infer_gnn_architecture(config: Mapping[str, Any]) -> str:
    """Resolve an explicit architecture or infer one from a legacy config."""
    explicit = config.get("gnn_architecture")
    if explicit in GNN_ARCHITECTURES:
        return str(explicit)

    training = config.get("training")
    training_config = training if isinstance(training, Mapping) else {}
    training_explicit = training_config.get("gnn_architecture")
    if training_explicit in GNN_ARCHITECTURES:
        return str(training_explicit)
    option_sources = [
        config.get("gnn_architecture_options"),
        training_config.get("gnn_architecture_options"),
    ]
    if any(
        bool(config.get(field) or training_config.get(field))
        or any(
            isinstance(options, Mapping) and bool(options.get(field))
            for options in option_sources
        )
        for field in ADVANCED_GNN_BOOLEAN_FIELDS
    ):
        return AA_GRAPH_SAGE_ARCHITECTURE_ID
    return GRAPH_SAGE_ARCHITECTURE_ID


def architecture_defaults(architecture_id: str) -> dict[str, Any]:
    """Return canonical runtime defaults for an architecture."""
    definition = GNN_ARCHITECTURES[architecture_id]
    defaults = {
        option.option_id: option.default
        for option in definition.options
    }
    defaults["gnn_architecture"] = architecture_id
    return defaults


def architecture_option_definitions() -> dict[str, Any]:
    """Return the unique union of CLI options across registered architectures."""
    options: dict[str, Any] = {}
    for architecture in GNN_ARCHITECTURES.values():
        for option in architecture.options:
            existing = options.get(option.option_id)
            if existing is not None:
                if (
                    existing.cli_flag != option.cli_flag
                    or existing.value_type != option.value_type
                ):
                    raise ValueError(
                        f"Conflicting schemas for GNN option {option.option_id}."
                    )
                options[option.option_id] = existing.model_copy(
                    update={
                        "choices": tuple(
                            dict.fromkeys((*existing.choices, *option.choices))
                        )
                    }
                )
            else:
                options[option.option_id] = option
    return options


def import_architecture_callable(path: str):
    """Load a registry callback lazily to keep configuration imports torch-free."""
    module_name, separator, attribute_name = path.partition(":")
    if not separator:
        raise ValueError(f"Invalid architecture callable path: {path}")
    return getattr(import_module(module_name), attribute_name)


def build_architecture_runtime_strategy(architecture_id: str, **kwargs):
    """Construct the architecture-owned runtime strategy lazily."""
    strategy_type = import_architecture_callable(
        GNN_ARCHITECTURES[architecture_id].runtime_strategy_path
    )
    return strategy_type(**kwargs)


def validate_architecture_options(
    architecture_id: str,
    options: Mapping[str, Any],
) -> None:
    """Run an architecture-owned validation hook when one is configured."""
    definition = GNN_ARCHITECTURES[architecture_id]
    if definition.validator_path is None:
        return
    validator = import_architecture_callable(definition.validator_path)
    validator(options)


def validate_aa_graphsage_options(options: Mapping[str, Any]) -> None:
    """Validate relationships between Advance GraphSAGE options."""
    if (
        options.get("node_classifier") == "linear"
        and options.get("question_aware_classifier") is True
    ):
        raise GnnArchitectureOptionValidationError(
            option_id="node_classifier",
            message=(
                "Advance GraphSAGE linear classification requires "
                "--no-question-aware-classifier."
            ),
        )


def validate_hgt_options(options: Mapping[str, Any]) -> None:
    """Validate the HGT head partition against its hidden width."""
    hidden_dimension = options.get("gnn_hidden_dimension")
    attention_heads = options.get("attention_heads")
    if (
        isinstance(hidden_dimension, int)
        and isinstance(attention_heads, int)
        and hidden_dimension % attention_heads != 0
    ):
        raise GnnArchitectureOptionValidationError(
            option_id="attention_heads",
            message=(
                "HGT hidden dimension must be divisible by the number of "
                "attention heads."
            ),
        )
