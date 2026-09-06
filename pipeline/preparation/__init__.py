"""Preparation phase exports for graphragX.

The package intentionally lazy-loads models, steps, and services. This keeps
base imports such as ``pipeline.abstract`` from recursively importing the full
preparation graph while they are still initializing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "BuildGnnAnswerRetrieverContext": "pipeline.preparation.steps.gnn_model_building",
    "BuildGnnAnswerRetrieverStep": "pipeline.preparation.steps.gnn_model_building",
    "BuildPipelineConfigurationStep": "pipeline.preparation.steps.configuration_building",
    "BuildWebQSPLocalGraphsStep": "pipeline.preparation.steps.webqsp_local_graph_preparation",
    "BuiltGnnAnswerRetriever": "pipeline.preparation.steps.gnn_model_building",
    "BuiltPipelineConfiguration": "pipeline.preparation.steps.configuration_building",
    "PrepareGnnTrainingDataContext": "pipeline.preparation.steps.gnn_training_data_preparation",
    "PrepareGnnTrainingDataStep": "pipeline.preparation.steps.gnn_training_data_preparation",
    "PreparedGnnTrainingData": "pipeline.preparation.models.gnn_training_data",
    "PreparedGnnTrainingInstance": "pipeline.preparation.models.gnn_training_data",
    "GnnTrainingDataPreparationConfig": "pipeline.preparation.services.gnn_training_data_preparation",
    "GnnTrainingDataPreparationService": "pipeline.preparation.services.gnn_training_data_preparation",
    "LoadDatasetStep": "pipeline.preparation.steps.dataset_loading",
    "LoadedDataset": "pipeline.preparation.steps.dataset_loading",
    "PipelineConfigurationInput": "pipeline.preparation.steps.configuration_building",
    "SelectDatasetStep": "pipeline.preparation.steps.dataset_selection",
    "SelectedDataset": "pipeline.preparation.steps.dataset_selection",
    "TrainGnnAnswerRetrieverContext": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "TrainGnnAnswerRetrieverStep": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "TrainedGnnAnswerRetriever": "pipeline.preparation.steps.gnn_answer_retriever_training",
    "PreparedWebQSPGraphDataset": "pipeline.preparation.models.webqsp_local_graph",
    "WebQSPEntityMappingSummary": "pipeline.preparation.models.webqsp_local_graph",
    "WebQSPProcessedInstance": "pipeline.preparation.models.webqsp_local_graph",
    "WebQSPVocabularyStore": "pipeline.preparation.models.webqsp_local_graph",
    "AnswerRetrieverModel": "pipeline.preparation.models.interfaces",
    "CONTEXT_CONSTRUCTION_STRATEGIES": "pipeline.preparation.helpers.configuration_definitions",
    "ContextConstructionDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "GNN_HIDDEN_DIMENSION_OPTIONS": "pipeline.preparation.helpers.configuration_definitions",
    "GNN_ARCHITECTURES": "pipeline.preparation.helpers.configuration_definitions",
    "GNN_LAYER_COUNT_OPTIONS": "pipeline.preparation.helpers.configuration_definitions",
    "GnnHiddenDimensionDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "GnnArchitectureDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "GnnArchitectureOptionDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "GnnLayerCountDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "LlmModelDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "NODE_CLASSIFIERS": "pipeline.preparation.helpers.configuration_definitions",
    "NodeClassifierDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "OPENAI_EMBEDDING_MODELS": "pipeline.preparation.helpers.configuration_definitions",
    "OpenAiEmbeddingModelDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_GNN_HIDDEN_DIMENSION": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_GNN_LAYER_COUNT": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_MAIN_LLM_MODEL_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_NODE_CLASSIFIER_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_RELATION_EMBEDDING_MODEL_ID": "pipeline.preparation.helpers.configuration_definitions",
    "RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID": "pipeline.preparation.helpers.configuration_definitions",
    "SHARED_LLM_MODELS": "pipeline.preparation.helpers.configuration_definitions",
    "SUBGRAPH_CONSTRUCTION_ALGORITHMS": "pipeline.preparation.helpers.configuration_definitions",
    "SubgraphConstructionDefinition": "pipeline.preparation.helpers.configuration_definitions",
    "DATASET_CACHE_ROOT": "pipeline.preparation.helpers.dataset_definitions",
    "DATASET_LOADERS": "pipeline.preparation.helpers.dataset_definitions",
    "PIPELINE_DATASETS": "pipeline.preparation.helpers.dataset_definitions",
    "WEBQSP_DATASET_ID": "pipeline.preparation.helpers.dataset_definitions",
    "DatasetDefinition": "pipeline.preparation.helpers.dataset_definitions",
    "DatasetLoaderDefinition": "pipeline.preparation.helpers.dataset_definitions",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
