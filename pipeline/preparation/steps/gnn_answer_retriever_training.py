"""Training step for the GNN answer retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from helpers.constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_TRAINING_PROFILE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_WEIGHT_DECAY,
)
from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.models.gnn_training_data import PreparedGnnTrainingData
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingConfig,
    GnnAnswerRetrieverTrainingService,
)

logger = get_logger(__name__)


class TrainedGnnAnswerRetriever(StepResult):
    """Fully trained GNN answer-retriever artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    gnn_architecture: str = Field(default="graphsage")
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    gnn_architecture_context: dict[str, Any] = Field(default_factory=dict)
    relation_vocabulary_path: Path | None = None
    hidden_dimension: int | None = Field(
        default=None,
        description="Shared hidden dimension used by the trained retriever.",
    )
    gnn_layer_count: int | None = Field(default=None, description="Number of trained GNN layers.")
    node_classifier: str | None = Field(default=None, description="Trained node classifier id.")
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None)
    dropout: float = Field(default=0.1)
    training_epochs: int = Field(..., description="Number of training epochs used.")
    training_learning_rate: float = Field(..., description="Training learning rate.")
    training_weight_decay: float = Field(..., description="Training weight decay.")
    training_max_instances: int | None = Field(
        default=None,
        description="Optional maximum number of training instances used.",
    )
    training_start_instance: int = Field(
        default=0,
        description="Inclusive training split index where training started.",
    )
    training_end_instance: int = Field(
        ...,
        description="Exclusive training split index where training stopped.",
    )
    training_log_every: int = Field(..., description="Training progress log interval.")
    training_batch_size: int = Field(
        default=DEFAULT_TRAINING_BATCH_SIZE,
        description="Configured categorical-relation disconnected-graph batch size.",
    )
    training_device: str = Field(..., description="Requested training device.")
    training_profile: bool = Field(
        default=DEFAULT_TRAINING_PROFILE,
        description="Whether detailed synchronized training timings were enabled.",
    )
    training_run_name: str | None = Field(
        default=None,
        description="Optional user-provided training run label.",
    )
    selected_device: str = Field(..., description="Resolved PyTorch training device.")
    embedding_cache_device: str = Field(
        default="qdrant",
        description="Resolved device used by compact frozen embeddings.",
    )
    embedding_cache_dtype: str = Field(
        default="float32",
        description="Storage precision used by compact frozen embeddings.",
    )
    final_loss: float = Field(..., description="Final average epoch loss.")
    loss_history: list[dict[str, float | int]] = Field(
        default_factory=list,
        description="Average loss per training epoch.",
    )
    trained_instances: int = Field(..., description="Number of instances trained on.")
    skipped_missing_gold_in_graph_count: int = Field(default=0)
    is_fine_tuned_model: bool = Field(
        default=False,
        description="Whether this model continued training from a saved run.",
    )
    continued_from_model_run_name: str | None = Field(
        default=None,
        description="Source model run name when continuation was used.",
    )
    continued_from_model_run_number: int | None = Field(
        default=None,
        description="Source model run number when continuation was used.",
    )
    model: AnswerRetrieverModel = Field(..., description="Trained retriever model.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    model_run_directory: Path = Field(..., description="Versioned training run directory.")
    model_run_name: str = Field(..., description="Resolved training run folder name.")
    model_run_number: int = Field(..., description="Incremental training run number.")
    embedding_cache_directory: Path = Field(
        ...,
        description="Directory containing cached OpenAI embeddings.",
    )
    wandb_status: str | None = None
    wandb_run_id: str | None = None
    wandb_run_url: str | None = None
    wandb_error_message: str | None = None


class TrainGnnAnswerRetrieverContext(StepContext[PreparedGnnTrainingData]):
    """Specialized context for training the GNN answer retriever."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(
        ...,
        description="Prepared WebQSP graph dataset used for training.",
    )
    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration used for training-time embeddings.",
    )


class TrainGnnAnswerRetrieverStep(
    AbstractStep[TrainedGnnAnswerRetriever, PreparedGnnTrainingData]
):
    """Train the built GNN answer retriever on prepared WebQSP graphs."""

    def __init__(
        self,
        training_epochs: int = DEFAULT_TRAINING_EPOCHS,
        training_learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE,
        training_weight_decay: float = DEFAULT_TRAINING_WEIGHT_DECAY,
        training_max_instances: int | None = None,
        training_start_instance: int = 0,
        skip_missing_gold_in_graph: bool = True,
        training_log_every: int = DEFAULT_TRAINING_LOG_EVERY,
        training_batch_size: int = DEFAULT_TRAINING_BATCH_SIZE,
        training_device: str = DEFAULT_TRAINING_DEVICE,
        training_profile: bool = DEFAULT_TRAINING_PROFILE,
        random_seed: int = DEFAULT_RANDOM_SEED,
        training_run_name: str | None = None,
        continue_training_model_run_name: str | None = None,
        continue_training_model_run_number: int | None = None,
        training_service: GnnAnswerRetrieverTrainingService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.training_config = GnnAnswerRetrieverTrainingConfig(
            epochs=training_epochs,
            learning_rate=training_learning_rate,
            weight_decay=training_weight_decay,
            max_instances=training_max_instances,
            start_instance=training_start_instance,
            skip_missing_gold_in_graph=skip_missing_gold_in_graph,
            log_every=training_log_every,
            batch_size=training_batch_size,
            device=training_device,
            profile=training_profile,
            random_seed=random_seed,
            run_name=training_run_name,
            continue_from_model_run_name=continue_training_model_run_name,
            continue_from_model_run_number=continue_training_model_run_number,
        )
        self.training_service = training_service or GnnAnswerRetrieverTrainingService()

    def execute_default(
        self,
        context: TrainGnnAnswerRetrieverContext,
    ) -> TrainedGnnAnswerRetriever:
        prepared_training_data = context.result
        if prepared_training_data is None:
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever training requires prepared training data."
            )

        logger.info(
            f"Starting TrainGnnAnswerRetrieverStep: "
            f"dataset={prepared_training_data.built_retriever.dataset_id} "
            f"epochs={self.training_config.epochs} "
            f"start_instance={self.training_config.start_instance} "
            f"max_instances={self.training_config.max_instances} "
            f"batch_size={self.training_config.batch_size} "
            f"profile={self.training_config.profile} "
            f"run_name={self.training_config.run_name} "
            f"continue_from_name={self.training_config.continue_from_model_run_name} "
            f"continue_from_number={self.training_config.continue_from_model_run_number}"
        )
        try:
            outcome = self.training_service.train(
                prepared_training_data=prepared_training_data,
                prepared_dataset=context.prepared_dataset,
                configuration=context.pipeline_configuration,
                training_config=self.training_config,
            )
        finally:
            self.training_service.release_prepared_embeddings(
                prepared_training_data
            )
        logger.info(
            f"Finished TrainGnnAnswerRetrieverStep: "
            f"run={outcome.model_run_name} final_loss={outcome.final_loss:.6f} "
            f"trained_instances={outcome.trained_instances}"
        )
        return TrainedGnnAnswerRetriever(
            dataset_id=outcome.dataset_id,
            gnn_architecture=outcome.gnn_architecture,
            gnn_architecture_options=outcome.gnn_architecture_options,
            gnn_architecture_context=outcome.gnn_architecture_context,
            relation_vocabulary_path=outcome.relation_vocabulary_path,
            hidden_dimension=outcome.hidden_dimension,
            gnn_layer_count=outcome.gnn_layer_count,
            node_classifier=outcome.node_classifier,
            use_edge_mlp=outcome.use_edge_mlp,
            question_aware_classifier=outcome.question_aware_classifier,
            use_reverse_edges=outcome.use_reverse_edges,
            add_layer_normalization=outcome.add_layer_normalization,
            edge_mlp_hidden_dim=outcome.edge_mlp_hidden_dim,
            dropout=outcome.dropout,
            training_epochs=self.training_config.epochs,
            training_learning_rate=self.training_config.learning_rate,
            training_weight_decay=self.training_config.weight_decay,
            training_max_instances=self.training_config.max_instances,
            training_start_instance=outcome.training_start_instance,
            training_end_instance=outcome.training_end_instance,
            training_log_every=self.training_config.log_every,
            training_batch_size=self.training_config.batch_size,
            training_device=self.training_config.device,
            training_profile=self.training_config.profile,
            training_run_name=self.training_config.run_name,
            selected_device=outcome.selected_device,
            embedding_cache_device=outcome.embedding_cache_device,
            embedding_cache_dtype=outcome.embedding_cache_dtype,
            final_loss=outcome.final_loss,
            loss_history=outcome.loss_history,
            trained_instances=outcome.trained_instances,
            skipped_missing_gold_in_graph_count=(
                outcome.skipped_missing_gold_in_graph_count
            ),
            is_fine_tuned_model=outcome.is_fine_tuned_model,
            continued_from_model_run_name=outcome.continued_from_model_run_name,
            continued_from_model_run_number=outcome.continued_from_model_run_number,
            model=outcome.model,
            model_artifact_path=outcome.model_artifact_path,
            model_config_path=outcome.model_config_path,
            model_run_directory=outcome.model_run_directory,
            model_run_name=outcome.model_run_name,
            model_run_number=outcome.model_run_number,
            embedding_cache_directory=outcome.embedding_cache_directory,
        )
