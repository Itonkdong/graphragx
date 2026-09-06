"""Preparation service exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "AbstractDatasetLoaderService": "pipeline.preparation.services.dataset_loader",
    "HuggingFaceWebQSPDatasetLoaderService": "pipeline.preparation.services.dataset_loader",
    "LangChainOpenAiTextEmbeddingService": "pipeline.preparation.services.openai_text_embedding",
    "GnnTrainingDataPreparationConfig": "pipeline.preparation.services.gnn_training_data_preparation",
    "GnnTrainingDataPreparationService": "pipeline.preparation.services.gnn_training_data_preparation",
    "GnnEmbeddingTensorCacheService": "pipeline.preparation.services.gnn_embedding_tensor_cache",
    "SelectionService": "pipeline.preparation.services.selection",
    "TextEmbeddingCache": "pipeline.preparation.services.embedding_cache",
    "WebQSPEmbeddingCacheService": "pipeline.preparation.services.embedding_cache",
    "WebQSPEntityNameMappingService": "pipeline.preparation.services.webqsp_entity_name_mapping",
    "WebQSPLocalGraphProcessorService": "pipeline.preparation.services.webqsp_local_graph_processing",
    "WebQSPLocalGraphStorageService": "pipeline.preparation.services.webqsp_local_graph_storage",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
