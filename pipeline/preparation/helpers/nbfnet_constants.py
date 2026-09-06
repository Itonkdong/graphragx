"""Stable persisted NBFNet architecture contract."""

from __future__ import annotations

from typing import Any


NBFNET_PREPROCESSING_VERSION = 1

NBFNET_FIXED_CONTEXT: dict[str, Any] = {
    "nbfnet_preprocessing_version": NBFNET_PREPROCESSING_VERSION,
    "query_source": "pooled_question_embedding",
    "boundary_schema": "multi_source_full_query",
    "dependent_relations": True,
    "message_function": "distmult",
    "aggregate_function": "pna",
    "short_cut": True,
    "layer_normalization": True,
    "activation": "relu",
    "scorer": "two_layer_mlp",
}


def validate_nbfnet_architecture_context(context: dict[str, Any]) -> None:
    """Reject saved NBFNet runs whose fixed preprocessing/model contract changed."""
    for key, expected in NBFNET_FIXED_CONTEXT.items():
        if context.get(key) != expected:
            raise ValueError(
                f"NBFNet architecture context {key}={context.get(key)!r} does not "
                f"match required value {expected!r}."
            )
