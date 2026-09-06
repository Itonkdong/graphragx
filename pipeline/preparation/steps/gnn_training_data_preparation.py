"""Pipeline step that prepares compact embeddings for GNN training."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from helpers.constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE,
    DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE,
    DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB,
)
from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.gnn_training_data import PreparedGnnTrainingData
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationConfig,
    GnnTrainingDataPreparationService,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever

logger = get_logger(__name__)


class PrepareGnnTrainingDataContext(StepContext[BuiltGnnAnswerRetriever]):
    """Context required to prepare selected GNN training tensors."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(...)
    pipeline_configuration: BuiltPipelineConfiguration = Field(...)


class PrepareGnnTrainingDataStep(
    AbstractStep[PreparedGnnTrainingData, BuiltGnnAnswerRetriever]
):
    """Prepare compact frozen embeddings and integer-indexed training graphs."""

    def __init__(
        self,
        training_max_instances: int | None = None,
        training_start_instance: int = 0,
        skip_missing_gold_in_graph: bool = True,
        training_device: str = DEFAULT_TRAINING_DEVICE,
        training_embedding_cache_device: Literal[
            "auto", "gpu", "cpu"
        ] = DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE,
        training_embedding_cache_dtype: Literal[
            "auto", "float32", "bfloat16"
        ] = DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE,
        training_gpu_cache_reserve_gb: float = DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB,
        continue_training_model_run_name: str | None = None,
        continue_training_model_run_number: int | None = None,
        preparation_service: GnnTrainingDataPreparationService | None = None,
        force_default: bool = False,
    ) -> None:
        super().__init__(force_default=force_default)
        self.preparation_config = GnnTrainingDataPreparationConfig(
            start_instance=training_start_instance,
            max_instances=training_max_instances,
            skip_missing_gold_in_graph=skip_missing_gold_in_graph,
            training_device=training_device,
            embedding_cache_device=training_embedding_cache_device,
            embedding_cache_dtype=training_embedding_cache_dtype,
            gpu_cache_reserve_gb=training_gpu_cache_reserve_gb,
            continue_from_model_run_name=continue_training_model_run_name,
            continue_from_model_run_number=continue_training_model_run_number,
        )
        self.preparation_service = (
            preparation_service or GnnTrainingDataPreparationService()
        )

    def execute_default(
        self,
        context: PrepareGnnTrainingDataContext,
    ) -> PreparedGnnTrainingData:
        """Prepare the selected training slice for repeated epoch consumption."""
        built_retriever = context.result
        if built_retriever is None:
            raise InvalidInteractiveConfigurationInputException(
                "GNN training-data preparation requires a built retriever."
            )
        logger.info(
            f"Starting GNN training-data preparation: "
            f"start={self.preparation_config.start_instance} "
            f"max_instances={self.preparation_config.max_instances} "
            f"cache_device={self.preparation_config.embedding_cache_device} "
            f"cache_dtype={self.preparation_config.embedding_cache_dtype}"
        )
        return self.preparation_service.prepare(
            built_retriever=built_retriever,
            prepared_dataset=context.prepared_dataset,
            configuration=context.pipeline_configuration,
            preparation_config=self.preparation_config,
        )
