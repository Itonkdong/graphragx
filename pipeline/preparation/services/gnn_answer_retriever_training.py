"""Training service for the GNN answer retriever."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING, Protocol
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from helpers.path_serialization import make_project_paths_relative
from helpers.constants import (
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_TRAINING_PROFILE,
    DEFAULT_TRAINING_WEIGHT_DECAY,
    DEFAULT_WANDB_TRAINING_LOG_EVERY,
    DEFAULT_RANDOM_SEED,
    GNN_ANSWER_RETRIEVER_CONFIG_FILENAME,
    GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME,
)
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    HGT_ARCHITECTURE_ID,
    NBFNET_ARCHITECTURE_ID,
    RGCN_ARCHITECTURE_ID,
)
from pipeline.preparation.helpers.gnn_architecture import (
    build_architecture_runtime_strategy,
)
from pipeline.preparation.models.gnn_training_data import PreparedGnnTrainingData
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.services import AbstractService
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    LoadedGnnAnswerRetrieverRun,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    RGCN_RELATION_VOCABULARY_FILENAME,
    validate_relation_architecture_context,
)

if TYPE_CHECKING:
    from torch import Tensor
    from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class SynchronizableDeviceRuntime(Protocol):
    """Accelerator runtime capable of synchronizing queued operations."""

    def synchronize(self) -> None:
        """Wait for queued accelerator operations to finish."""


class TorchRuntime(Protocol):
    """Subset of the torch runtime needed by training profiling."""

    cuda: SynchronizableDeviceRuntime
    mps: SynchronizableDeviceRuntime


class GnnAnswerRetrieverTrainingConfig(BaseModel):
    """Runtime training settings for the answer retriever."""

    epochs: int = Field(default=DEFAULT_TRAINING_EPOCHS)
    learning_rate: float = Field(default=DEFAULT_TRAINING_LEARNING_RATE)
    weight_decay: float = Field(default=DEFAULT_TRAINING_WEIGHT_DECAY)
    max_instances: int | None = Field(default=None)
    start_instance: int = Field(default=0)
    skip_missing_gold_in_graph: bool = Field(default=True)
    log_every: int = Field(default=DEFAULT_TRAINING_LOG_EVERY)
    batch_size: int = Field(default=DEFAULT_TRAINING_BATCH_SIZE, gt=0)
    device: str = Field(default=DEFAULT_TRAINING_DEVICE)
    profile: bool = Field(default=DEFAULT_TRAINING_PROFILE)
    random_seed: int = Field(default=DEFAULT_RANDOM_SEED, ge=0)
    run_name: str | None = Field(default=None)
    continue_from_model_run_name: str | None = Field(default=None)
    continue_from_model_run_number: int | None = Field(default=None)


@dataclass
class GnnTrainingPhaseTimings:
    """Accumulate synchronized wall-clock timings for training phases."""

    input_preparation_seconds: float = 0.0
    forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    instance_count: int = 0

    @property
    def total_seconds(self) -> float:
        """Return total measured time across all training phases."""
        return (
            self.input_preparation_seconds
            + self.forward_seconds
            + self.loss_seconds
            + self.backward_seconds
            + self.optimizer_seconds
        )


@dataclass
class PreparedTypedTrainingBatch:
    """One disconnected categorical-relation graph batch cached across epochs."""

    instance_count: int
    node_embedding_indices: Tensor
    edge_index: Tensor
    edge_type: Tensor
    edge_norm: Tensor | None
    active_relation_ids: Tensor
    edge_relation_index: Tensor | None
    active_relation_offsets: Tensor
    node_labels: Tensor
    positive_weights: Tensor
    node_loss_weights: Tensor


# Compatibility for tests and callers written before HGT shared typed batching.
PreparedRGCNTrainingBatch = PreparedTypedTrainingBatch


class GnnAnswerRetrieverTrainingOutcome(BaseModel):
    """Result produced by the training service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_loss: float = Field(..., description="Final average epoch loss.")
    loss_history: list[dict[str, float | int]] = Field(
        default_factory=list,
        description="Average loss per training epoch.",
    )
    trained_instances: int = Field(..., description="Number of training instances used.")
    skipped_missing_gold_in_graph_count: int = Field(default=0)
    training_start_instance: int = Field(..., description="Inclusive training slice start.")
    training_end_instance: int = Field(..., description="Exclusive training slice end.")
    model_artifact_path: Path = Field(..., description="Saved model weights path.")
    model_config_path: Path = Field(..., description="Saved model config path.")
    model_run_directory: Path = Field(..., description="Versioned training run directory.")
    model_run_name: str = Field(..., description="Resolved training run folder name.")
    model_run_number: int = Field(..., description="Incremental training run number.")
    embedding_cache_directory: Path = Field(..., description="Embedding cache root.")
    selected_device: str = Field(..., description="Resolved PyTorch training device.")
    embedding_cache_device: str = Field(..., description="Resolved embedding location.")
    embedding_cache_dtype: str = Field(..., description="Frozen embedding precision.")
    dataset_id: str = Field(..., description="Dataset id used by the trained model.")
    gnn_architecture: str = Field(default="graphsage")
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    gnn_architecture_context: dict[str, Any] = Field(default_factory=dict)
    relation_vocabulary_path: Path | None = None
    entity_embedding_model: str | None = Field(default=None, description="Entity embedding model id.")
    question_embedding_model: str | None = Field(default=None, description="Question embedding model id.")
    relation_embedding_model: str | None = Field(default=None, description="Relation embedding model id.")
    entity_embedding_dimension: int | None = Field(default=None, description="Entity embedding dimension.")
    question_embedding_dimension: int | None = Field(default=None, description="Question embedding dimension.")
    relation_embedding_dimension: int | None = Field(default=None, description="Relation embedding dimension.")
    hidden_dimension: int | None = Field(default=None, description="GNN hidden dimension.")
    gnn_layer_count: int | None = Field(default=None, description="GNN layer count.")
    node_classifier: str | None = Field(default=None, description="Node classifier id.")
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None, description="Edge MLP hidden dimension.")
    dropout: float = Field(default=0.1)
    model: object = Field(..., description="Trained model instance.")
    is_fine_tuned_model: bool = Field(..., description="Whether training continued a saved run.")
    continued_from_model_run_name: str | None = Field(
        default=None,
        description="Source model run name when continuation was used.",
    )
    continued_from_model_run_number: int | None = Field(
        default=None,
        description="Source model run number when continuation was used.",
    )


class GnnAnswerRetrieverTrainingService(AbstractService):
    """Train and persist the GNN answer retriever."""

    model_weights_filename = GNN_ANSWER_RETRIEVER_WEIGHTS_FILENAME
    model_config_filename = GNN_ANSWER_RETRIEVER_CONFIG_FILENAME

    def __init__(
        self,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
        epoch_callback: Callable[[dict[str, float | int]], None] | None = None,
        progress_callback_every: int = DEFAULT_WANDB_TRAINING_LOG_EVERY,
    ) -> None:
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()
        self.progress_callback = progress_callback
        self.epoch_callback = epoch_callback
        self.progress_callback_every = progress_callback_every

    def train(
        self,
        prepared_training_data: PreparedGnnTrainingData,
        prepared_dataset: PreparedWebQSPGraphDataset,
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
    ) -> GnnAnswerRetrieverTrainingOutcome:
        """Train the answer retriever from compact indexed embedding matrices."""
        import torch
        import torch.nn.functional as torch_functional

        device = self._resolve_device(torch, training_config.device)
        if device != prepared_training_data.selected_device:
            raise GnnAnswerRetrieverTrainingException(
                f"Prepared training device {prepared_training_data.selected_device} "
                f"does not match resolved training device {device}."
            )
        logger.info(f"Starting GNN answer retriever training on device={device}")

        cache_root = prepared_training_data.cache_root
        continued_run = self._load_continued_run(
            cache_root=cache_root,
            configuration=configuration,
            training_config=training_config,
            device=device,
        )
        effective_retriever = self._build_effective_retriever(
            built_retriever=prepared_training_data.built_retriever,
            continued_run=continued_run,
        )
        embedding_model = (
            continued_run.config.resolved_embedding_model
            if continued_run is not None
            else configuration.embedding_model
        )
        question_embedding_model = embedding_model
        relation_embedding_model = embedding_model
        embedding_mismatch = (
            prepared_training_data.node_embeddings is not None
            and effective_retriever.entity_embedding_model
            != prepared_training_data.entity_embedding_model
        ) or (
            prepared_training_data.question_embeddings is not None
            and question_embedding_model
            != prepared_training_data.question_embedding_model
        ) or (
            prepared_training_data.relation_embeddings is not None
            and relation_embedding_model
            != prepared_training_data.relation_embedding_model
        )
        if embedding_mismatch:
            raise GnnAnswerRetrieverTrainingException(
                "Prepared training embeddings do not match the effective model run."
            )
        if effective_retriever.dataset_id != prepared_dataset.dataset_id:
            raise GnnAnswerRetrieverTrainingException(
                f"Cannot continue training model run from dataset "
                f"{effective_retriever.dataset_id} on dataset {prepared_dataset.dataset_id}."
            )

        logger.info(
            f"Selected GNN training slice: "
            f"start={prepared_training_data.training_start_instance} "
            f"end={prepared_training_data.training_end_instance} "
            f"instances={len(prepared_training_data.instances)} "
            f"embedding_device={prepared_training_data.embedding_cache_device} "
            f"embedding_dtype={prepared_training_data.embedding_cache_dtype}"
        )
        if continued_run is not None:
            logger.info(
                f"Continuing GNN answer retriever from run={continued_run.run_name} "
                f"run_number={continued_run.run_number} "
                f"additional_epochs={training_config.epochs}"
            )

        model = effective_retriever.model
        runtime_strategy = build_architecture_runtime_strategy(
            effective_retriever.gnn_architecture
        )
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        typed_batches = (
            self._build_typed_training_batches(
                prepared_training_data=prepared_training_data,
                batch_size=training_config.batch_size,
                torch=torch,
                device=device,
                architecture_id=effective_retriever.gnn_architecture,
            )
            if effective_retriever.gnn_architecture
            in {RGCN_ARCHITECTURE_ID, HGT_ARCHITECTURE_ID}
            else None
        )
        runtime_batches = (
            runtime_strategy.build_training_batches(
                prepared_data=prepared_training_data,
                batch_size=training_config.batch_size,
                torch=torch,
                device=device,
            )
            if runtime_strategy.handles_training_batches
            else None
        )
        if runtime_batches is not None:
            logger.info(
                f"Prepared {effective_retriever.gnn_architecture} disconnected "
                f"graph batches: batches={len(runtime_batches)} "
                f"batch_size={training_config.batch_size} graph_tensors_device={device}"
            )
        elif typed_batches is not None:
            logger.info(
                f"Prepared {effective_retriever.gnn_architecture} disconnected "
                f"graph batches: batches={len(typed_batches)} "
                f"batch_size={training_config.batch_size} "
                f"graph_tensors_device={device}"
            )
        elif training_config.batch_size != 1:
            logger.info(
                "GraphSAGE training retains single-graph optimizer steps; "
                f"configured batch_size={training_config.batch_size} only applies "
                "to categorical-relation architectures."
            )

        final_loss = 0.0
        loss_history: list[dict[str, float | int]] = []
        for epoch in range(1, training_config.epochs + 1):
            logger.info(
                f"Training epoch {epoch}/{training_config.epochs} "
                f"over {len(prepared_training_data.instances)} WebQSP instances"
            )
            epoch_loss = torch.zeros((), dtype=torch.float, device=device)
            epoch_contributing_instances = 0
            phase_timings = GnnTrainingPhaseTimings()
            training_units = runtime_batches or typed_batches or prepared_training_data.instances
            processed_instances = 0
            for training_unit in training_units:
                previous_processed_instances = processed_instances
                unit_instance_count = (
                    int(training_unit.instance_count)
                    if hasattr(training_unit, "instance_count")
                    else 1
                )
                processed_instances += unit_instance_count
                optimizer.zero_grad(set_to_none=True)
                phase_started_at = self._start_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                )

                edge_norm = None
                active_relation_ids = None
                edge_relation_index = None
                active_relation_offsets = None
                positive_weights = None
                node_loss_weights = None
                runtime_model_inputs = None
                if runtime_batches is not None:
                    runtime_model_inputs = runtime_strategy.model_inputs(training_unit)
                    entity_features = None
                    question_features = None
                    relation_features = None
                    edge_weight = None
                    edge_index = training_unit.edge_index
                    edge_type = getattr(training_unit, "edge_type", None)
                    node_labels = training_unit.node_labels
                elif isinstance(training_unit, PreparedRGCNTrainingBatch):
                    entity_features = self._gather_embedding_features(
                        embedding_matrix=prepared_training_data.node_embeddings,
                        indices=training_unit.node_embedding_indices,
                        torch=torch,
                        device=device,
                    )
                    question_features = None
                    relation_features = None
                    edge_weight = None
                    edge_index = training_unit.edge_index
                    edge_type = training_unit.edge_type
                    edge_norm = training_unit.edge_norm
                    active_relation_ids = training_unit.active_relation_ids
                    edge_relation_index = training_unit.edge_relation_index
                    active_relation_offsets = training_unit.active_relation_offsets
                    node_labels = training_unit.node_labels
                    positive_weights = training_unit.positive_weights
                    node_loss_weights = training_unit.node_loss_weights
                else:
                    instance = training_unit
                    entity_features = self._gather_embedding_features(
                        embedding_matrix=prepared_training_data.node_embeddings,
                        indices=instance.node_embedding_indices,
                        torch=torch,
                        device=device,
                    )
                    question_features = None
                    if prepared_training_data.question_embeddings is not None:
                        if instance.question_embedding_index is None:
                            raise GnnAnswerRetrieverTrainingException(
                                "Prepared question embeddings are missing an instance index."
                            )
                        question_features = prepared_training_data.question_embeddings[
                            instance.question_embedding_index
                        ].to(device=device, non_blocking=True)
                    relation_features = None
                    if prepared_training_data.relation_embeddings is not None:
                        if instance.relation_embedding_indices is None:
                            raise GnnAnswerRetrieverTrainingException(
                                "Prepared relation embeddings are missing instance indices."
                            )
                        relation_features = self._gather_embedding_features(
                            embedding_matrix=prepared_training_data.relation_embeddings,
                            indices=instance.relation_embedding_indices,
                            torch=torch,
                            device=device,
                        )
                    edge_weight = None
                    if relation_features is not None and question_features is not None:
                        edge_weight = (
                            None
                            if effective_retriever.use_edge_mlp
                            else self._build_edge_weight_tensor(
                                relation_features=relation_features,
                                question_features=question_features,
                                torch=torch,
                                torch_functional=torch_functional,
                                device=device,
                            )
                        )
                    edge_index = instance.edge_index.to(
                        device=device, non_blocking=True
                    )
                    edge_type = (
                        instance.edge_type.to(device=device, non_blocking=True)
                        if instance.edge_type is not None
                        else None
                    )
                    edge_norm = (
                        instance.edge_norm.to(device=device, non_blocking=True)
                        if instance.edge_norm is not None
                        else None
                    )
                    active_relation_ids = (
                        instance.active_relation_ids.to(
                            device=device, non_blocking=True
                        )
                        if instance.active_relation_ids is not None
                        else None
                    )
                    edge_relation_index = (
                        instance.edge_relation_index.to(
                            device=device, non_blocking=True
                        )
                        if instance.edge_relation_index is not None
                        else None
                    )
                    active_relation_offsets = (
                        instance.active_relation_offsets
                        if instance.active_relation_offsets is not None
                        else None
                    )
                    node_labels = instance.node_labels.to(
                        device=device, non_blocking=True
                    )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.input_preparation_seconds += elapsed_seconds

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(
                        device.startswith("cuda")
                        and prepared_training_data.uses_bfloat16
                    ),
                ):
                    logits = (
                        model(**runtime_model_inputs)
                        if runtime_model_inputs is not None
                        else model(
                            entity_features=entity_features,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            question_features=question_features,
                            relation_features=relation_features,
                            edge_type=edge_type,
                            edge_norm=edge_norm,
                            active_relation_ids=active_relation_ids,
                            edge_relation_index=edge_relation_index,
                            active_relation_offsets=active_relation_offsets,
                        )
                    )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.forward_seconds += elapsed_seconds
                if runtime_batches is not None:
                    loss = runtime_strategy.compute_loss(logits.float(), training_unit)
                    contributing_instances = runtime_strategy.contributing_instance_count(
                        training_unit
                    )
                elif positive_weights is not None and node_loss_weights is not None:
                    loss = torch_functional.binary_cross_entropy_with_logits(
                        logits.float(),
                        node_labels,
                        weight=node_loss_weights,
                        pos_weight=positive_weights,
                        reduction="sum",
                    )
                    contributing_instances = unit_instance_count
                else:
                    loss = self._compute_loss(
                        logits=logits.float(),
                        node_labels=node_labels,
                        torch=torch,
                        torch_functional=torch_functional,
                    )
                    contributing_instances = unit_instance_count
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.loss_seconds += elapsed_seconds
                if loss is None:
                    logger.warning(
                        f"Skipping {effective_retriever.gnn_architecture} optimizer "
                        "update because the batch contains no usable answer targets."
                    )
                    continue
                loss.backward()
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.backward_seconds += elapsed_seconds
                optimizer.step()
                _, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=training_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.optimizer_seconds += elapsed_seconds
                phase_timings.instance_count += unit_instance_count

                detached_loss = loss.detach()
                epoch_loss += detached_loss * contributing_instances
                epoch_contributing_instances += contributing_instances
                if (
                    self._is_progress_interval_crossed(
                        previous_instance_index=previous_processed_instances,
                        instance_index=processed_instances,
                        total_instances=len(prepared_training_data.instances),
                        interval=training_config.log_every,
                    )
                ):
                    logger.info(
                        f"Epoch {epoch}/{training_config.epochs} progress: "
                        f"{processed_instances}/{len(prepared_training_data.instances)} "
                        f"instances, "
                        f"latest_loss={detached_loss.item():.6f}"
                    )
                    if training_config.profile:
                        self._log_phase_timings(
                            epoch=epoch,
                            processed_instances=processed_instances,
                            timings=phase_timings,
                        )
                if (
                    self.progress_callback is not None
                    and self._is_progress_interval_crossed(
                        previous_instance_index=previous_processed_instances,
                        instance_index=processed_instances,
                        total_instances=len(prepared_training_data.instances),
                        interval=self.progress_callback_every,
                    )
                ):
                    self.progress_callback(
                        {
                            "epoch": epoch,
                            "instance": processed_instances,
                            "gnn_architecture": (
                                prepared_training_data.built_retriever.gnn_architecture
                            ),
                            "global_step": (
                                (epoch - 1) * len(prepared_training_data.instances)
                                + processed_instances
                            ),
                            "loss": float(detached_loss.item()),
                        }
                    )

            if epoch_contributing_instances == 0:
                raise GnnAnswerRetrieverTrainingException(
                    "Training epoch contains no graphs with in-graph answer targets."
                )
            final_loss = (epoch_loss / epoch_contributing_instances).item()
            loss_history.append(
                {
                    "epoch": epoch,
                    "average_loss": final_loss,
                }
            )
            logger.info(
                f"Finished epoch {epoch}/{training_config.epochs} "
                f"with average_loss={final_loss:.6f}"
            )
            if self.epoch_callback is not None:
                self.epoch_callback(
                    {
                        "epoch": epoch,
                        "gnn_architecture": (
                            prepared_training_data.built_retriever.gnn_architecture
                        ),
                        "average_loss": final_loss,
                    }
                )
            if training_config.profile and (
                training_config.log_every <= 0
                or len(prepared_training_data.instances) % training_config.log_every != 0
            ):
                self._log_phase_timings(
                    epoch=epoch,
                    processed_instances=len(prepared_training_data.instances),
                    timings=phase_timings,
                )

        model_artifact_path = self._save_model_artifacts(
            model=model,
            built_retriever=effective_retriever,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            training_config=training_config,
            selected_device=device,
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(prepared_training_data.instances),
            skipped_missing_gold_in_graph_count=(
                prepared_training_data.skipped_missing_gold_in_graph_count
            ),
            training_start_instance=prepared_training_data.training_start_instance,
            training_end_instance=prepared_training_data.training_end_instance,
            continued_run=continued_run,
            model_run_directory=self._create_model_run_directory(
                model_root=cache_root / "models",
                run_name=training_config.run_name,
            ),
            torch=torch,
            embedding_cache_device=prepared_training_data.embedding_cache_device,
            embedding_cache_dtype=prepared_training_data.embedding_cache_dtype,
            runtime_strategy=runtime_strategy,
        )
        logger.info(f"Saved trained GNN answer retriever to {model_artifact_path}")
        model_run_directory = model_artifact_path.parent
        model_run_number = self._extract_run_number(model_run_directory.name)

        return GnnAnswerRetrieverTrainingOutcome(
            final_loss=final_loss,
            loss_history=loss_history,
            trained_instances=len(prepared_training_data.instances),
            skipped_missing_gold_in_graph_count=(
                prepared_training_data.skipped_missing_gold_in_graph_count
            ),
            training_start_instance=prepared_training_data.training_start_instance,
            training_end_instance=prepared_training_data.training_end_instance,
            model_artifact_path=model_artifact_path,
            model_config_path=model_artifact_path.with_name(self.model_config_filename),
            model_run_directory=model_run_directory,
            model_run_name=model_run_directory.name,
            model_run_number=model_run_number,
            embedding_cache_directory=cache_root / "embeddings",
            selected_device=device,
            embedding_cache_device=prepared_training_data.embedding_cache_device,
            embedding_cache_dtype=prepared_training_data.embedding_cache_dtype,
            dataset_id=effective_retriever.dataset_id,
            gnn_architecture=effective_retriever.gnn_architecture,
            gnn_architecture_options=effective_retriever.gnn_architecture_options,
            gnn_architecture_context=effective_retriever.gnn_architecture_context,
            relation_vocabulary_path=(
                model_run_directory / RGCN_RELATION_VOCABULARY_FILENAME
                if effective_retriever.relation_vocabulary is not None
                else None
            ),
            entity_embedding_model=effective_retriever.entity_embedding_model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_dimension=effective_retriever.entity_embedding_dimension,
            question_embedding_dimension=effective_retriever.question_embedding_dimension,
            relation_embedding_dimension=effective_retriever.relation_embedding_dimension,
            hidden_dimension=effective_retriever.hidden_dimension,
            gnn_layer_count=effective_retriever.gnn_layer_count,
            node_classifier=effective_retriever.node_classifier,
            use_edge_mlp=effective_retriever.use_edge_mlp,
            question_aware_classifier=effective_retriever.question_aware_classifier,
            use_reverse_edges=effective_retriever.use_reverse_edges,
            add_layer_normalization=effective_retriever.add_layer_normalization,
            edge_mlp_hidden_dim=effective_retriever.edge_mlp_hidden_dim,
            dropout=effective_retriever.dropout,
            model=model,
            is_fine_tuned_model=continued_run is not None,
            continued_from_model_run_name=(
                continued_run.run_name if continued_run is not None else None
            ),
            continued_from_model_run_number=(
                continued_run.run_number if continued_run is not None else None
            ),
        )

    @staticmethod
    def _is_progress_due(
        *,
        instance_index: int,
        total_instances: int,
        interval: int,
    ) -> bool:
        """Return whether an interval stream should report this instance."""
        return interval > 0 and (
            instance_index % interval == 0 or instance_index == total_instances
        )

    @staticmethod
    def _is_progress_interval_crossed(
        *,
        previous_instance_index: int,
        instance_index: int,
        total_instances: int,
        interval: int,
    ) -> bool:
        """Report batched progress when an interval boundary is crossed."""
        return interval > 0 and (
            instance_index == total_instances
            or instance_index // interval > previous_instance_index // interval
        )

    @staticmethod
    def _build_rgcn_training_batches(
        *,
        prepared_training_data: PreparedGnnTrainingData,
        batch_size: int,
        torch: ModuleType,
        device: str,
    ) -> list[PreparedRGCNTrainingBatch]:
        """Backward-compatible wrapper for R-GCN batch construction."""
        return GnnAnswerRetrieverTrainingService._build_typed_training_batches(
            prepared_training_data=prepared_training_data,
            batch_size=batch_size,
            torch=torch,
            device=device,
            architecture_id=RGCN_ARCHITECTURE_ID,
        )

    @staticmethod
    def _build_typed_training_batches(
        *,
        prepared_training_data: PreparedGnnTrainingData,
        batch_size: int,
        torch: ModuleType,
        device: str,
        architecture_id: str,
    ) -> list[PreparedRGCNTrainingBatch]:
        """Cache disconnected typed graph batches and graph-balanced loss metadata."""
        if batch_size <= 0:
            raise GnnAnswerRetrieverTrainingException(
                "training_batch_size must be greater than zero."
            )
        batches: list[PreparedRGCNTrainingBatch] = []
        instances = prepared_training_data.instances
        embedding_matrix_device = prepared_training_data.node_embeddings.device
        for batch_start in range(0, len(instances), batch_size):
            batch_instances = instances[batch_start : batch_start + batch_size]
            node_indices_parts = []
            edge_index_parts = []
            edge_type_parts = []
            edge_norm_parts = []
            label_parts = []
            positive_weight_parts = []
            node_loss_weight_parts = []
            node_offset = 0
            for instance in batch_instances:
                if instance.edge_type is None:
                    raise GnnAnswerRetrieverTrainingException(
                        "Prepared categorical-relation instances require edge types."
                    )
                if (
                    architecture_id == RGCN_ARCHITECTURE_ID
                    and instance.edge_norm is None
                ):
                    raise GnnAnswerRetrieverTrainingException(
                        "Prepared R-GCN instances require relation normalization."
                    )
                node_count = int(instance.node_embedding_indices.shape[0])
                if node_count <= 0:
                    raise GnnAnswerRetrieverTrainingException(
                        "R-GCN batches cannot contain empty-node graphs."
                    )
                node_indices_parts.append(instance.node_embedding_indices)
                edge_index_parts.append(instance.edge_index + node_offset)
                edge_type_parts.append(instance.edge_type)
                if instance.edge_norm is not None:
                    edge_norm_parts.append(instance.edge_norm)
                labels = instance.node_labels.float()
                label_parts.append(labels)
                positive_count = float(labels.sum().item())
                negative_count = node_count - positive_count
                positive_weight = (
                    negative_count / positive_count if positive_count > 0 else 1.0
                )
                positive_weight_parts.append(
                    torch.full_like(labels, positive_weight)
                )
                node_loss_weight_parts.append(
                    torch.full_like(
                        labels,
                        1.0 / (len(batch_instances) * node_count),
                    )
                )
                node_offset += node_count

            edge_index = torch.cat(edge_index_parts, dim=1)
            edge_type = torch.cat(edge_type_parts)
            edge_norm = torch.cat(edge_norm_parts) if edge_norm_parts else None
            if edge_type.numel() > 0:
                edge_order = torch.argsort(edge_type, stable=True)
                edge_index = edge_index.index_select(1, edge_order)
                edge_type = edge_type.index_select(0, edge_order)
                if edge_norm is not None:
                    edge_norm = edge_norm.index_select(0, edge_order)
                active_relation_ids, edge_relation_index = torch.unique_consecutive(
                    edge_type,
                    return_inverse=True,
                )
                relation_counts = torch.bincount(
                    edge_relation_index,
                    minlength=active_relation_ids.numel(),
                )
                active_relation_offsets = torch.cat(
                    [relation_counts.new_zeros(1), relation_counts.cumsum(dim=0)]
                )
            else:
                active_relation_ids = torch.empty(0, dtype=torch.long)
                edge_relation_index = torch.empty(0, dtype=torch.long)
                active_relation_offsets = torch.zeros(1, dtype=torch.long)

            batches.append(
                PreparedRGCNTrainingBatch(
                    instance_count=len(batch_instances),
                    node_embedding_indices=torch.cat(node_indices_parts).to(
                        device=embedding_matrix_device,
                        non_blocking=True,
                    ),
                    edge_index=edge_index.to(device=device, non_blocking=True),
                    edge_type=edge_type.to(device=device, non_blocking=True),
                    edge_norm=(
                        edge_norm.to(device=device, non_blocking=True)
                        if edge_norm is not None
                        else None
                    ),
                    active_relation_ids=active_relation_ids.to(
                        device=device, non_blocking=True
                    ),
                    edge_relation_index=(
                        edge_relation_index.to(device=device, non_blocking=True)
                        if architecture_id == RGCN_ARCHITECTURE_ID
                        else None
                    ),
                    active_relation_offsets=active_relation_offsets,
                    node_labels=torch.cat(label_parts).to(
                        device=device, non_blocking=True
                    ),
                    positive_weights=torch.cat(positive_weight_parts).to(
                        device=device, non_blocking=True
                    ),
                    node_loss_weights=torch.cat(node_loss_weight_parts).to(
                        device=device, non_blocking=True
                    ),
                )
            )
        return batches

    @staticmethod
    def release_prepared_embeddings(
        prepared_training_data: PreparedGnnTrainingData,
    ) -> None:
        """Release compact embedding matrices after the training step finishes."""
        import torch

        used_cuda = prepared_training_data.embedding_cache_device.startswith("cuda")
        prepared_training_data.node_embeddings = torch.empty(0)
        if prepared_training_data.relation_embeddings is not None:
            prepared_training_data.relation_embeddings = torch.empty(0)
        if prepared_training_data.question_embeddings is not None:
            prepared_training_data.question_embeddings = torch.empty(0)
        if used_cuda:
            torch.cuda.empty_cache()

    def _load_continued_run(
        self,
        cache_root: Path,
        configuration: BuiltPipelineConfiguration,
        training_config: GnnAnswerRetrieverTrainingConfig,
        device: str,
    ) -> LoadedGnnAnswerRetrieverRun | None:
        if (
            training_config.continue_from_model_run_name is None
            and training_config.continue_from_model_run_number is None
        ):
            return None

        return self.model_run_service.load_run(
            model_root=cache_root / "models",
            run_name=training_config.continue_from_model_run_name,
            run_number=training_config.continue_from_model_run_number,
            pipeline_configuration=configuration,
            device=device,
        )

    @staticmethod
    def _build_effective_retriever(
        built_retriever: BuiltGnnAnswerRetriever,
        continued_run: LoadedGnnAnswerRetrieverRun | None,
    ) -> BuiltGnnAnswerRetriever:
        if continued_run is None:
            return built_retriever

        if built_retriever.gnn_architecture != continued_run.config.resolved_gnn_architecture:
            raise GnnAnswerRetrieverTrainingException(
                "Continued training cannot change GNN architecture from "
                f"{continued_run.config.resolved_gnn_architecture} to "
                f"{built_retriever.gnn_architecture}."
            )
        architecture = GNN_ARCHITECTURES[built_retriever.gnn_architecture]
        if (
            architecture.data_requirements.uses_relation_types
            and built_retriever.gnn_architecture_options
            != continued_run.config.resolved_gnn_architecture_options
        ):
            raise GnnAnswerRetrieverTrainingException(
                "Continued categorical-relation GNN training cannot change resolved "
                "architecture options."
            )

        return BuiltGnnAnswerRetriever(
            dataset_id=continued_run.config.dataset_id,
            gnn_architecture=continued_run.config.resolved_gnn_architecture,
            gnn_architecture_options=(
                continued_run.config.resolved_gnn_architecture_options
            ),
            gnn_architecture_context=continued_run.config.gnn_architecture_context,
            relation_vocabulary=continued_run.relation_vocabulary,
            entity_embedding_model=continued_run.config.resolved_embedding_model,
            entity_embedding_dimension=continued_run.config.resolved_embedding_dimension,
            question_embedding_dimension=(
                continued_run.config.question_embedding_dimension
                or built_retriever.question_embedding_dimension
            ),
            relation_embedding_dimension=(
                continued_run.config.relation_embedding_dimension
                or built_retriever.relation_embedding_dimension
            ),
            hidden_dimension=continued_run.config.resolved_hidden_dimension,
            gnn_layer_count=continued_run.config.resolved_gnn_layer_count,
            node_classifier=continued_run.config.node_classifier,
            use_edge_mlp=continued_run.config.use_edge_mlp,
            question_aware_classifier=continued_run.config.question_aware_classifier,
            use_reverse_edges=continued_run.config.resolved_use_reverse_edges,
            add_layer_normalization=continued_run.config.add_layer_normalization,
            edge_mlp_hidden_dim=continued_run.config.resolved_edge_mlp_hidden_dim,
            dropout=continued_run.config.dropout,
            model=continued_run.model,
        )

    @staticmethod
    def _resolve_device(torch, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _gather_embedding_features(
        embedding_matrix: Tensor,
        indices: Tensor,
        torch: ModuleType,
        device: str,
    ) -> Tensor:
        """Gather compact frozen embeddings and move the result to training device."""
        matrix_device = embedding_matrix.device
        resolved_indices = indices.to(device=matrix_device, non_blocking=True)
        features = torch.index_select(embedding_matrix, 0, resolved_indices)
        return features.to(device=device, non_blocking=True)

    def _build_edge_weight_tensor(
        self,
        relation_features,
        question_features,
        torch,
        torch_functional,
        device: str,
    ):
        if relation_features.shape[0] == 0:
            return torch.empty(0, dtype=relation_features.dtype, device=device)
        return torch_functional.cosine_similarity(
            question_features.reshape(1, -1),
            relation_features,
            dim=1,
        )

    @staticmethod
    def _compute_loss(
        logits,
        node_labels,
        torch,
        torch_functional,
    ):
        """Compute class-balanced binary loss without synchronizing the device."""
        positive_count = node_labels.sum()
        negative_count = node_labels.numel() - positive_count
        positive_weight = torch.where(
            positive_count > 0,
            negative_count / positive_count.clamp_min(1),
            torch.ones_like(positive_count),
        )
        return torch_functional.binary_cross_entropy_with_logits(
            logits,
            node_labels,
            pos_weight=positive_weight,
        )

    @staticmethod
    def _start_profiled_phase(
        torch: TorchRuntime,
        device: str,
        enabled: bool,
    ) -> float:
        """Synchronize the selected device and start one measured phase."""
        if not enabled:
            return 0.0
        GnnAnswerRetrieverTrainingService._synchronize_device(torch, device)
        return time.perf_counter()

    @staticmethod
    def _finish_profiled_phase(
        torch: TorchRuntime,
        device: str,
        enabled: bool,
        phase_started_at: float,
    ) -> tuple[float, float]:
        """Finish one measured phase and return its next start and duration."""
        if not enabled:
            return 0.0, 0.0
        GnnAnswerRetrieverTrainingService._synchronize_device(torch, device)
        elapsed_seconds = time.perf_counter() - phase_started_at
        return time.perf_counter(), elapsed_seconds

    @staticmethod
    def _synchronize_device(torch: TorchRuntime, device: str) -> None:
        """Wait for asynchronous accelerator work before collecting timings."""
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    @staticmethod
    def _log_phase_timings(
        epoch: int,
        processed_instances: int,
        timings: GnnTrainingPhaseTimings,
    ) -> None:
        """Log cumulative and per-instance training phase timings."""
        if timings.instance_count == 0:
            return
        milliseconds_per_instance = 1000.0 / timings.instance_count
        logger.info(
            f"Training profile epoch={epoch} instances={processed_instances}: "
            f"input_ms={timings.input_preparation_seconds * milliseconds_per_instance:.2f} "
            f"forward_ms={timings.forward_seconds * milliseconds_per_instance:.2f} "
            f"loss_ms={timings.loss_seconds * milliseconds_per_instance:.2f} "
            f"backward_ms={timings.backward_seconds * milliseconds_per_instance:.2f} "
            f"optimizer_ms={timings.optimizer_seconds * milliseconds_per_instance:.2f} "
            f"total_ms={timings.total_seconds * milliseconds_per_instance:.2f}"
        )

    def _save_model_artifacts(
        self,
        model,
        built_retriever: BuiltGnnAnswerRetriever,
        question_embedding_model: str | None,
        relation_embedding_model: str | None,
        training_config: GnnAnswerRetrieverTrainingConfig,
        selected_device: str,
        final_loss: float,
        loss_history: list[dict[str, float | int]],
        trained_instances: int,
        training_start_instance: int,
        training_end_instance: int,
        continued_run: LoadedGnnAnswerRetrieverRun | None,
        model_run_directory: Path,
        torch,
        skipped_missing_gold_in_graph_count: int = 0,
        embedding_cache_device: str = "qdrant",
        embedding_cache_dtype: str = "float32",
        runtime_strategy=None,
    ) -> Path:
        model_run_directory.mkdir(parents=True, exist_ok=False)
        model_artifact_path = model_run_directory / self.model_weights_filename
        model_config_path = model_run_directory / self.model_config_filename
        model_run_name = model_run_directory.name
        model_run_number = self._extract_run_number(model_run_name)
        training_payload = training_config.model_dump(
            exclude={
                "run_name",
                "continue_from_model_run_name",
                "continue_from_model_run_number",
                "start_instance",
            }
        )
        training_payload["batch_size"] = (
            training_config.batch_size
            if (
                getattr(runtime_strategy, "handles_training_batches", False)
                or built_retriever.gnn_architecture
                in {RGCN_ARCHITECTURE_ID, HGT_ARCHITECTURE_ID}
            )
            else 1
        )
        # Keep model architecture and embedding identity at the model root.
        # Training settings, the selected instance range, and epoch history
        # are kept together under training so each value has one owner.
        training_payload.update(
            {
                "device": selected_device,
                "loss_function": (
                    "KLDivLoss"
                    if getattr(runtime_strategy, "strategy_id", "default") == "rearev"
                    else "BCEWithLogitsLoss"
                ),
                "embedding_cache_device": embedding_cache_device,
                "embedding_cache_dtype": embedding_cache_dtype,
                "trained_instances": {
                    "start": training_start_instance,
                    "end": training_end_instance,
                    "count": trained_instances,
                },
                "skipped_missing_gold_in_graph_count": (
                    skipped_missing_gold_in_graph_count
                ),
                "loss_history": loss_history,
            }
        )
        is_fine_tuned_model = continued_run is not None
        config_payload = {
            "dataset_id": built_retriever.dataset_id,
            "gnn_architecture": built_retriever.gnn_architecture,
            "gnn_architecture_options": {
                key: value
                for key, value in built_retriever.gnn_architecture_options.items()
                if value is not None
            },
            "gnn_architecture_context": built_retriever.gnn_architecture_context,
            "run_name": model_run_name,
            "run_number": model_run_number,
            "is_fine_tuned_model": is_fine_tuned_model,
            "final_loss": final_loss,
            "training": training_payload,
        }
        persisted_embedding_model = (
            built_retriever.entity_embedding_model
            or question_embedding_model
            or relation_embedding_model
        )
        persisted_embedding_dimension = (
            built_retriever.entity_embedding_dimension
            or built_retriever.question_embedding_dimension
            or built_retriever.relation_embedding_dimension
        )
        if persisted_embedding_model is not None:
            config_payload["embedding_model"] = persisted_embedding_model
        if persisted_embedding_dimension is not None:
            config_payload["embedding_dimension"] = persisted_embedding_dimension
        if (
            built_retriever.gnn_architecture
            in {HGT_ARCHITECTURE_ID, NBFNET_ARCHITECTURE_ID}
            or getattr(runtime_strategy, "strategy_id", "default") == "rearev"
        ):
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )
            config_payload["parameter_count"] = parameter_count
            if built_retriever.gnn_architecture in {
                HGT_ARCHITECTURE_ID,
                NBFNET_ARCHITECTURE_ID,
            }:
                config_payload["estimated_training_parameter_bytes"] = (
                    parameter_count * 16
                )
        if continued_run is not None:
            config_payload.update(
                {
                    "continued_from_model_run_name": continued_run.run_name,
                    "continued_from_model_run_number": continued_run.run_number,
                }
            )
        if built_retriever.relation_vocabulary is not None:
            try:
                validate_relation_architecture_context(
                    built_retriever.gnn_architecture_context,
                    built_retriever.relation_vocabulary,
                )
                if built_retriever.gnn_architecture == NBFNET_ARCHITECTURE_ID:
                    from pipeline.preparation.helpers.nbfnet_constants import (
                        validate_nbfnet_architecture_context,
                    )

                    validate_nbfnet_architecture_context(
                        built_retriever.gnn_architecture_context
                    )
            except ValueError as error:
                raise GnnAnswerRetrieverTrainingException(
                    f"Cannot save invalid categorical relation vocabulary: {error}"
                ) from error
            relation_vocabulary_path = (
                model_run_directory / RGCN_RELATION_VOCABULARY_FILENAME
            )
            relation_vocabulary_path.write_text(
                json.dumps(
                    built_retriever.relation_vocabulary,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        checkpoint = (
            runtime_strategy.checkpoint_state_dict(model)
            if runtime_strategy is not None
            else model.state_dict()
        )
        torch.save(checkpoint, model_artifact_path)
        model_config_path.write_text(
            json.dumps(
                make_project_paths_relative(config_payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return model_artifact_path

    def _create_model_run_directory(
        self,
        model_root: Path,
        run_name: str | None,
    ) -> Path:
        model_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(model_root)
        run_label = self._resolve_run_label(run_name)
        return model_root / f"{run_number}_{run_label}"

    @classmethod
    def _next_run_number(cls, model_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in model_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))

    @classmethod
    def _resolve_run_label(cls, run_name: str | None) -> str:
        if run_name is None or not run_name.strip():
            return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        return cls._sanitize_run_name(run_name)

    @staticmethod
    def _sanitize_run_name(run_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
        sanitized = sanitized.strip("._-")
        return sanitized or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
