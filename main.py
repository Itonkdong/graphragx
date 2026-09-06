"""Entry point for running the graphragX framework."""

from __future__ import annotations

import argparse
import math
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from helpers.constants import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_CANDIDATE_TOP_K,
    MIN_CANDIDATE_TOP_K,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE,
    DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB,
    DEFAULT_EVALUATION_LOG_EVERY,
    DEFAULT_EVALUATION_PROFILE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_DEVICE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_LOG_EVERY,
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_TRAINING_PROFILE,
    DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE,
    DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE,
    DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB,
    DEFAULT_TRAINING_WEIGHT_DECAY,
    DEFAULT_WANDB_TRAINING_LOG_EVERY,
)
from helpers.logging_config import get_logger, setup_logger
from helpers.reproducibility import configure_random_seed
from pipeline import (
    BuildGnnAnswerRetrieverStep,
    BuildPipelineConfigurationStep,
    BuildWebQSPLocalGraphsStep,
    InitialStepResult,
    LoadDatasetStep,
    Pipeline,
    PipelineExecutionResult,
    PrepareGnnTrainingDataStep,
    SelectDatasetStep,
    StepContext,
    TrainGnnAnswerRetrieverStep,
    EvaluateGnnAnswerRetrieverStep,
    BuildReasoningSamplesFromGnnEvaluationStep,
    BuildEvidenceSubgraphsBatchStep,
    ComputeFinalResultsStep,
    GenerateAndSaveFinalAnswersBatchesStep,
    LogFinalResultsToWandbStep,
    LogRetrieverToWandbStep,
    LogTrainingToWandbStep,
    LogInferenceToWandbStep,
    LogEvidenceToWandbStep,
    LoadGnnAnswerRetrieverRunStep,
    SaveEvidenceSubgraphsStep,
    GnnRetrieverResultsService,
    PipelineException,
)
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    LLM_PROVIDERS,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_GNN_ARCHITECTURE_ID,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
    SHARED_LLM_MODELS,
    SUBGRAPH_CONSTRUCTION_ALGORITHMS,
)
from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    architecture_option_definitions,
)
from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingService,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    SavedGnnAnswerRetrieverConfig,
)

logger = get_logger(__name__)


def _saved_model_conflicts(
    requested: "PipelineRuntimeConfig",
    saved: SavedGnnAnswerRetrieverConfig,
) -> list[str]:
    """Return explicitly requested fields that disagree with saved lineage."""
    requested_embedding_model = (
        requested.embedding_model
        or requested.entity_embedding_model
        or requested.question_embedding_model
        or requested.relation_embedding_model
    )
    comparisons = {
        "gnn_architecture": (requested.gnn_architecture, saved.resolved_gnn_architecture),
        "embedding_model": (
            requested_embedding_model,
            saved.resolved_embedding_model,
        ),
    }
    requested_options = dict(requested.gnn_options)
    for option_id in architecture_option_definitions():
        if hasattr(requested, option_id):
            value = getattr(requested, option_id)
            if value is not None:
                requested_options[option_id] = value
    saved_options = saved.resolved_gnn_architecture_options
    for option_id, requested_value in requested_options.items():
        comparisons[option_id] = (requested_value, saved_options.get(option_id))
    conflicts = {
        name
        for name, (requested_value, saved_value) in comparisons.items()
        if requested_value is not None
        and requested_value != saved_value
        and not (name == "embedding_model" and saved_value is None)
    }
    return sorted(conflicts)


def _is_deepseek_model(model_id: str | None) -> bool:
    """Return whether a model requires explicit DeepSeek provider selection."""
    if model_id is None:
        return False
    definition = SHARED_LLM_MODELS.get(model_id)
    return model_id.startswith("deepseek-") or (
        definition is not None and definition.provider_id == "deepseek"
    )


def _is_unqualified_private_model(
    model_id: str | None,
    provider_id: str | None,
) -> bool:
    """Return whether an unknown model omitted the required Vezilka provider."""
    return (
        model_id is not None
        and provider_id is None
        and model_id not in SHARED_LLM_MODELS
    )


def _apply_saved_model_config(
    resolved: "PipelineRuntimeConfig",
    saved: SavedGnnAnswerRetrieverConfig,
) -> "PipelineRuntimeConfig":
    options = saved.resolved_gnn_architecture_options
    return resolved.model_copy(
        update={
            "dataset": saved.dataset_id,
            "gnn_architecture": saved.resolved_gnn_architecture,
            "gnn_options": options,
            "gnn_layer_count": options.get("gnn_layer_count"),
            "gnn_hidden_dimension": options.get("gnn_hidden_dimension"),
            "node_classifier": options.get("node_classifier"),
            "use_edge_mlp": options.get("use_edge_mlp"),
            "question_aware_classifier": options.get("question_aware_classifier"),
            # Mandatory reverse edges belong to architecture data requirements,
            # not to the user-configurable option set. The configuration builder
            # resolves the effective value after validating saved options.
            "use_reverse_edges": options.get("use_reverse_edges"),
            "add_layer_normalization": options.get("add_layer_normalization"),
            "edge_mlp_hidden_dim": options.get("edge_mlp_hidden_dim"),
            "dropout": options.get("dropout"),
            "embedding_model": (
                saved.resolved_embedding_model or resolved.embedding_model
            ),
            # Compatibility aliases for existing preparation services.
            "question_embedding_model": (
                saved.resolved_embedding_model or resolved.embedding_model
            ),
            "relation_embedding_model": (
                saved.resolved_embedding_model or resolved.embedding_model
            ),
            "entity_embedding_model": (
                saved.resolved_embedding_model or resolved.embedding_model
            ),
        }
    )


class PipelineRuntimeConfig(BaseModel):
    """Runtime configuration used to initialize the current framework pipeline."""

    run_mode: Literal[
        "full",
        "train-only",
        "retriever-only",
        "evaluation-only",
        "inference-only",
        "evidence-only",
    ] = "full"
    random_seed: int = Field(default=DEFAULT_RANDOM_SEED, ge=0)
    dataset: str | None = None
    local_graph_profile: bool = False
    llm_provider: str | None = None
    reasoning_effort: str | None = None
    generate_explanation: bool = False
    main_llm_model: str | None = None
    subgraph_algorithm: str | None = None
    pcst_edge_cost_strategy: str | None = None
    pcst_edge_cost: float | None = None
    pcst_debug_profile: bool = False
    context_strategy: str | None = None
    gnn_architecture: str | None = None
    gnn_layer_count: int | None = None
    gnn_hidden_dimension: int | None = None
    node_classifier: str | None = None
    use_edge_mlp: bool | None = None
    question_aware_classifier: bool | None = None
    use_reverse_edges: bool | None = None
    add_layer_normalization: bool | None = None
    edge_mlp_hidden_dim: int | None = None
    dropout: float | None = None
    gnn_options: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    # Deprecated compatibility inputs. New CLI/configuration uses embedding_model.
    question_embedding_model: str | None = None
    relation_embedding_model: str | None = None
    entity_embedding_model: str | None = None
    training_epochs: int = DEFAULT_TRAINING_EPOCHS
    training_learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    training_weight_decay: float = DEFAULT_TRAINING_WEIGHT_DECAY
    training_max_instances: int | None = None
    training_start_instance: int = 0
    skip_missing_gold_in_graph: bool = True
    training_log_every: int = DEFAULT_TRAINING_LOG_EVERY
    training_batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    wandb_training_log_every: int = DEFAULT_WANDB_TRAINING_LOG_EVERY
    wandb_upload_retriever: bool = False
    training_device: str = DEFAULT_TRAINING_DEVICE
    training_profile: bool = DEFAULT_TRAINING_PROFILE
    training_embedding_cache_device: str = DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE
    training_embedding_cache_dtype: str = DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE
    training_gpu_cache_reserve_gb: float = DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB
    training_run_name: str | None = None
    continue_training_model_run_name: str | None = None
    continue_training_model_run_number: int | None = None
    evaluation_model_run_name: str | None = None
    evaluation_model_run_number: int | None = None
    answer_threshold: float = DEFAULT_ANSWER_THRESHOLD
    candidate_top_k: int = Field(
        default=DEFAULT_CANDIDATE_TOP_K,
        ge=MIN_CANDIDATE_TOP_K,
    )
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    evaluation_run_name: str | None = None
    retriever_run_name: str | None = None
    retriever_run_number: int | None = None
    evaluation_max_instances: int | None = None
    evaluation_log_every: int = DEFAULT_EVALUATION_LOG_EVERY
    evaluation_profile: bool = DEFAULT_EVALUATION_PROFILE
    evaluation_embedding_cache_device: str = DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE
    evaluation_embedding_cache_dtype: str = DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE
    evaluation_gpu_cache_reserve_gb: float = DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB
    no_llm_inference: bool = False
    inference_run_name: str | None = None
    evidence_run_name: str | None = None
    llm_inference_batch_size: int = 10
    llm_inference_parallel_calls: int = 1
    no_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None
    use_default_config_values: bool = False
    force_all_default: bool = False

    def with_defaulted_user_inputs(self) -> "PipelineRuntimeConfig":
        """Fill all user-provided selections with recommended defaults when requested."""
        if not self.use_default_config_values:
            return self

        architecture_id = self.gnn_architecture or RECOMMENDED_GNN_ARCHITECTURE_ID
        defaults = architecture_defaults(architecture_id)
        defaults.pop("gnn_architecture", None)
        requested_options = dict(self.gnn_options)
        for option_id in architecture_option_definitions():
            if hasattr(self, option_id):
                value = getattr(self, option_id)
                if value is not None:
                    requested_options[option_id] = value
        resolved_options = {**defaults, **requested_options}
        architecture = GNN_ARCHITECTURES[architecture_id]
        needs_openai_embeddings = any(
            (
                architecture.data_requirements.uses_entity_embeddings,
                architecture.data_requirements.uses_question_embeddings,
                architecture.data_requirements.uses_relation_embeddings,
                (self.subgraph_algorithm or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID)
                == "pcst"
                and (self.pcst_edge_cost_strategy or "constant") == "semantic",
            )
        )
        resolved_embedding_model = (
            self.embedding_model
            or self.entity_embedding_model
            or self.question_embedding_model
            or self.relation_embedding_model
            or (RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID if needs_openai_embeddings else None)
        )
        for option in architecture.options:
            if (
                option.enabled_when_option is not None
                and resolved_options.get(option.enabled_when_option)
                != option.enabled_when_value
                and option.option_id not in requested_options
            ):
                resolved_options[option.option_id] = None
        architecture_updates = {
            option_id: value
            for option_id, value in resolved_options.items()
            if hasattr(self, option_id)
        }
        architecture_updates.update(
            gnn_architecture=architecture_id,
            gnn_options=resolved_options,
        )
        # OpenAI is the default provider. DeepSeek is intentionally not
        # inferred from the model name because selecting that backend should
        # be explicit and visible in commands/configuration.
        llm_provider = self.llm_provider or "openai"
        main_llm_model = self.main_llm_model
        if main_llm_model is None and llm_provider != "vezilka":
            provider_models = [
                model_id
                for model_id, definition in SHARED_LLM_MODELS.items()
                if definition.provider_id == llm_provider
            ]
            main_llm_model = (
                RECOMMENDED_MAIN_LLM_MODEL_ID
                if llm_provider == "openai"
                else provider_models[0]
            )
        return self.model_copy(
            update={
                "dataset": self.dataset or WEBQSP_DATASET_ID,
                "llm_provider": llm_provider,
                "main_llm_model": main_llm_model,
                "subgraph_algorithm": self.subgraph_algorithm
                or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
                "pcst_edge_cost_strategy": (
                    self.pcst_edge_cost_strategy or "constant"
                    if (self.subgraph_algorithm or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID)
                    == "pcst"
                    else None
                ),
                "pcst_edge_cost": (
                    self.pcst_edge_cost if self.pcst_edge_cost is not None else 1.0
                    if (self.subgraph_algorithm or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID)
                    == "pcst"
                    else None
                ),
                "context_strategy": self.context_strategy
                or RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
                **architecture_updates,
                "embedding_model": resolved_embedding_model,
                "question_embedding_model": resolved_embedding_model,
                "relation_embedding_model": resolved_embedding_model,
                "entity_embedding_model": resolved_embedding_model,
            }
        )


def build_pipeline(config: PipelineRuntimeConfig) -> Pipeline:
    """Build the current runnable graphragX pipeline."""
    if _is_deepseek_model(config.main_llm_model) and config.llm_provider != "deepseek":
        raise PipelineException(
            f"Model {config.main_llm_model} requires explicit "
            "--llm-provider deepseek."
        )
    if _is_unqualified_private_model(
        config.main_llm_model,
        config.llm_provider,
    ):
        raise PipelineException(
            f"Model {config.main_llm_model} is not a configured OpenAI model. "
            "Privately hosted models require explicit --llm-provider vezilka."
        )
    if config.run_mode == "inference-only" and config.no_llm_inference:
        raise PipelineException(
            "--no-llm-inference is not valid with --inference-only."
        )
    if (
        config.run_mode in {"inference-only", "evidence-only"}
        and config.retriever_run_name is None
        and config.retriever_run_number is None
    ):
        raise PipelineException(
            f"{config.run_mode} mode requires --retriever-run-name or "
            "--retriever-run-number."
        )
    if config.run_mode == "evidence-only" and (
        config.llm_provider is not None
        or config.main_llm_model is not None
        or config.reasoning_effort is not None
        or config.generate_explanation
        or config.no_llm_inference
        or config.inference_run_name is not None
    ):
        raise PipelineException(
            "Evidence-only mode does not accept LLM provider/model, reasoning, "
            "explanation, inference-run naming, or --no-llm-inference options."
        )
    if config.run_mode != "evidence-only" and config.evidence_run_name is not None:
        raise PipelineException(
            "--evidence-run-name is only valid with --evidence-only."
        )
    if config.retriever_run_name is not None and config.retriever_run_number is not None:
        raise PipelineException(
            "Select a retriever run by --retriever-run-name or "
            "--retriever-run-number, not both."
        )
    if config.training_start_instance < 0:
        raise PipelineException(
            "--training-start-instance must be greater than or equal to 0."
        )
    if config.training_log_every < 0:
        raise PipelineException(
            "--training-log-every must be greater than or equal to 0."
        )
    if config.wandb_training_log_every < 0:
        raise PipelineException(
            "--wandb-training-log-every must be greater than or equal to 0."
        )
    if config.llm_inference_batch_size < 1:
        raise PipelineException(
            "--llm-inference-batch-size must be greater than or equal to 1."
        )
    if config.llm_inference_parallel_calls < 1:
        raise PipelineException(
            "--llm-inference-parallel-calls must be greater than or equal to 1."
        )
    if (
        config.use_default_config_values
        and config.llm_provider == "vezilka"
        and not config.main_llm_model
    ):
        raise PipelineException(
            "--llm-provider vezilka requires --main-llm-model when --default is used."
        )
    if config.run_mode in {"evaluation-only", "inference-only", "evidence-only"} and (
        config.continue_training_model_run_name is not None
        or config.continue_training_model_run_number is not None
    ):
        raise PipelineException(
            f"Training continuation flags are not valid in {config.run_mode} mode."
        )
    if config.run_mode in {"inference-only", "evidence-only"} and (
        config.evaluation_model_run_name is not None
        or config.evaluation_model_run_number is not None
    ):
        raise PipelineException(
            f"Evaluation model selectors are not valid in {config.run_mode} mode; "
            "select the persisted retriever run instead."
        )

    resolved_config = config.with_defaulted_user_inputs()
    resolved_subgraph_algorithm = (
        resolved_config.subgraph_algorithm
        or RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID
    )
    if resolved_subgraph_algorithm != "pcst" and (
        config.pcst_edge_cost_strategy is not None
        or config.pcst_edge_cost is not None
        or config.pcst_debug_profile
    ):
        raise PipelineException(
            "--pcst-edge-cost-strategy, --pcst-edge-cost, and "
            "--pcst-debug-profile require "
            "--subgraph-algorithm pcst."
        )
    if resolved_subgraph_algorithm == "pcst":
        edge_cost = (
            resolved_config.pcst_edge_cost
            if resolved_config.pcst_edge_cost is not None
            else 1.0
        )
        if not math.isfinite(edge_cost) or edge_cost <= 0:
            raise PipelineException(
                "--pcst-edge-cost must be a finite value greater than zero."
            )
    dataset_id = resolved_config.dataset or WEBQSP_DATASET_ID
    loader_definition = DATASET_LOADERS.get(dataset_id)
    if loader_definition is None:
        raise PipelineException(f"Unsupported dataset {dataset_id}.")

    saved_model_config: SavedGnnAnswerRetrieverConfig | None = None
    lineage_label: str | None = None
    if resolved_config.run_mode in {"inference-only", "evidence-only"}:
        saved_model_config = GnnRetrieverResultsService().load_model_config(
            evaluation_root=loader_definition.cache_root / "evaluations",
            run_name=resolved_config.retriever_run_name,
            run_number=resolved_config.retriever_run_number,
        )
        lineage_label = "saved retriever run"
    elif resolved_config.run_mode == "evaluation-only":
        saved_model_config = GnnAnswerRetrieverModelRunService().resolve_run(
            model_root=loader_definition.cache_root / "models",
            run_name=resolved_config.evaluation_model_run_name,
            run_number=resolved_config.evaluation_model_run_number,
        ).config
        lineage_label = "saved model run"
    elif (
        resolved_config.continue_training_model_run_name is not None
        or resolved_config.continue_training_model_run_number is not None
    ):
        saved_model_config = GnnAnswerRetrieverModelRunService().resolve_run(
            model_root=loader_definition.cache_root / "models",
            run_name=resolved_config.continue_training_model_run_name,
            run_number=resolved_config.continue_training_model_run_number,
        ).config
        lineage_label = "continued model run"

    if saved_model_config is not None:
        if config.dataset is not None and config.dataset != saved_model_config.dataset_id:
            raise PipelineException(
                f"Dataset {config.dataset} conflicts with {lineage_label} dataset "
                f"{saved_model_config.dataset_id}."
            )
        conflicting_fields = _saved_model_conflicts(config, saved_model_config)
        if conflicting_fields:
            raise PipelineException(
                f"Configuration conflicts with the {lineage_label}: "
                + ", ".join(conflicting_fields)
            )
        resolved_config = _apply_saved_model_config(
            resolved_config,
            saved_model_config,
        )
    wandb_dataset_id = resolved_config.dataset or WEBQSP_DATASET_ID
    wandb_loader_definition = DATASET_LOADERS.get(wandb_dataset_id)
    if wandb_loader_definition is None:
        raise PipelineException(
            f"W&B run identifiers do not support dataset {wandb_dataset_id}."
        )
    wandb_coordinator = WandbExperimentCoordinator(
        project=resolved_config.wandb_project,
        entity=resolved_config.wandb_entity,
        mode=resolved_config.wandb_mode,
        enabled=not resolved_config.no_wandb,
        resume_from_lineage=resolved_config.run_mode
        not in {"evaluation-only", "inference-only", "evidence-only"},
        run_root=wandb_loader_definition.cache_root / "wandb_runs",
        architecture_name=resolved_config.gnn_architecture,
    )
    setup_steps = [
        SelectDatasetStep(
            requested_dataset=resolved_config.dataset,
        ),
        BuildPipelineConfigurationStep(
            llm_provider=resolved_config.llm_provider,
            reasoning_effort=resolved_config.reasoning_effort,
            main_llm_model=resolved_config.main_llm_model,
            subgraph_algorithm=resolved_config.subgraph_algorithm,
            pcst_edge_cost_strategy=resolved_config.pcst_edge_cost_strategy,
            pcst_edge_cost=resolved_config.pcst_edge_cost,
            context_strategy=resolved_config.context_strategy,
            gnn_architecture=resolved_config.gnn_architecture,
            gnn_options=resolved_config.gnn_options,
            gnn_layer_count=resolved_config.gnn_layer_count,
            gnn_hidden_dimension=resolved_config.gnn_hidden_dimension,
            node_classifier=resolved_config.node_classifier,
            use_edge_mlp=resolved_config.use_edge_mlp,
            question_aware_classifier=resolved_config.question_aware_classifier,
            use_reverse_edges=resolved_config.use_reverse_edges,
            add_layer_normalization=resolved_config.add_layer_normalization,
            edge_mlp_hidden_dim=resolved_config.edge_mlp_hidden_dim,
            dropout=resolved_config.dropout,
            embedding_model=resolved_config.embedding_model,
            question_embedding_model=resolved_config.question_embedding_model,
            relation_embedding_model=resolved_config.relation_embedding_model,
            entity_embedding_model=resolved_config.entity_embedding_model,
        ),
        LoadDatasetStep(),
        BuildWebQSPLocalGraphsStep(
            load_train_instances=resolved_config.run_mode
            in {"full", "train-only", "retriever-only"},
            load_test_instances=resolved_config.run_mode != "train-only",
            profile=resolved_config.local_graph_profile,
        ),
    ]
    training_steps = [
        BuildGnnAnswerRetrieverStep(),
        PrepareGnnTrainingDataStep(
            training_max_instances=resolved_config.training_max_instances,
            training_start_instance=resolved_config.training_start_instance,
            skip_missing_gold_in_graph=(
                resolved_config.skip_missing_gold_in_graph
            ),
            training_device=resolved_config.training_device,
            training_embedding_cache_device=(
                resolved_config.training_embedding_cache_device
            ),
            training_embedding_cache_dtype=(
                resolved_config.training_embedding_cache_dtype
            ),
            training_gpu_cache_reserve_gb=(
                resolved_config.training_gpu_cache_reserve_gb
            ),
            continue_training_model_run_name=(
                resolved_config.continue_training_model_run_name
            ),
            continue_training_model_run_number=(
                resolved_config.continue_training_model_run_number
            ),
        ),
        TrainGnnAnswerRetrieverStep(
            training_epochs=resolved_config.training_epochs,
            training_learning_rate=resolved_config.training_learning_rate,
            training_weight_decay=resolved_config.training_weight_decay,
            training_max_instances=resolved_config.training_max_instances,
            training_start_instance=resolved_config.training_start_instance,
            skip_missing_gold_in_graph=(
                resolved_config.skip_missing_gold_in_graph
            ),
            training_log_every=resolved_config.training_log_every,
            training_batch_size=resolved_config.training_batch_size,
            training_device=resolved_config.training_device,
            training_profile=resolved_config.training_profile,
            random_seed=resolved_config.random_seed,
            training_run_name=resolved_config.training_run_name,
            continue_training_model_run_name=(
                resolved_config.continue_training_model_run_name
            ),
            continue_training_model_run_number=(
                resolved_config.continue_training_model_run_number
            ),
            training_service=GnnAnswerRetrieverTrainingService(
                progress_callback=wandb_coordinator.log_training_progress
                if not resolved_config.no_wandb
                else None,
                epoch_callback=wandb_coordinator.log_training_epoch
                if not resolved_config.no_wandb
                else None,
                progress_callback_every=resolved_config.wandb_training_log_every,
            ),
        ),
    ]
    if not resolved_config.no_wandb:
        training_steps.append(
            LogTrainingToWandbStep(
                coordinator=wandb_coordinator,
                upload_retriever=resolved_config.wandb_upload_retriever,
            )
        )
    retriever_steps = [
        EvaluateGnnAnswerRetrieverStep(
            model_run_name=resolved_config.evaluation_model_run_name,
            model_run_number=resolved_config.evaluation_model_run_number,
            answer_threshold=resolved_config.answer_threshold,
            candidate_top_k=resolved_config.candidate_top_k,
            candidate_limit=resolved_config.candidate_limit,
            evaluation_run_name=resolved_config.evaluation_run_name,
            evaluation_max_instances=resolved_config.evaluation_max_instances,
            skip_missing_gold_in_graph=(
                resolved_config.skip_missing_gold_in_graph
            ),
            evaluation_log_every=resolved_config.evaluation_log_every,
            evaluation_profile=resolved_config.evaluation_profile,
            evaluation_embedding_cache_device=(
                resolved_config.evaluation_embedding_cache_device
            ),
            evaluation_embedding_cache_dtype=(
                resolved_config.evaluation_embedding_cache_dtype
            ),
            evaluation_gpu_cache_reserve_gb=(
                resolved_config.evaluation_gpu_cache_reserve_gb
            ),
        ),
    ]
    if not resolved_config.no_wandb:
        retriever_steps.append(
            LogRetrieverToWandbStep(
                coordinator=wandb_coordinator,
                upload_retriever=resolved_config.wandb_upload_retriever,
            )
        )
    inference_steps = [
        BuildReasoningSamplesFromGnnEvaluationStep(),
        BuildEvidenceSubgraphsBatchStep(
            pcst_debug_profile=resolved_config.pcst_debug_profile,
        ),
        GenerateAndSaveFinalAnswersBatchesStep(
            llm_provider=resolved_config.llm_provider,
            reasoning_effort=resolved_config.reasoning_effort,
            generate_explanation=resolved_config.generate_explanation,
            model_id=resolved_config.main_llm_model,
            inference_run_name=resolved_config.inference_run_name,
            inference_batch_size=resolved_config.llm_inference_batch_size,
            inference_parallel_calls=resolved_config.llm_inference_parallel_calls,
        ),
    ]
    if not resolved_config.no_wandb:
        inference_steps.append(
            LogInferenceToWandbStep(coordinator=wandb_coordinator)
        )

    evidence_only_steps = [
        BuildReasoningSamplesFromGnnEvaluationStep(),
        BuildEvidenceSubgraphsBatchStep(
            pcst_debug_profile=resolved_config.pcst_debug_profile,
        ),
        SaveEvidenceSubgraphsStep(run_name=resolved_config.evidence_run_name),
    ]
    if not resolved_config.no_wandb:
        evidence_only_steps.append(
            LogEvidenceToWandbStep(coordinator=wandb_coordinator)
        )
    inference_steps.append(ComputeFinalResultsStep())
    if not resolved_config.no_wandb:
        inference_steps.append(
            LogFinalResultsToWandbStep(
                project=resolved_config.wandb_project,
                entity=resolved_config.wandb_entity,
                mode=resolved_config.wandb_mode,
                coordinator=wandb_coordinator,
            )
        )

    if resolved_config.run_mode == "train-only":
        preparation_steps = [*setup_steps, *training_steps]
        selected_evaluation_steps = []
    elif resolved_config.run_mode == "retriever-only":
        preparation_steps = [*setup_steps, *training_steps]
        selected_evaluation_steps = retriever_steps
    elif resolved_config.run_mode == "evaluation-only":
        preparation_steps = setup_steps
        selected_evaluation_steps = [*retriever_steps]
        if not resolved_config.no_llm_inference:
            selected_evaluation_steps.extend(inference_steps)
    elif resolved_config.run_mode == "inference-only":
        preparation_steps = setup_steps
        selected_evaluation_steps = [
            LoadGnnAnswerRetrieverRunStep(
                run_name=resolved_config.retriever_run_name,
                run_number=resolved_config.retriever_run_number,
            ),
            *(
                [
                    LogRetrieverToWandbStep(
                        coordinator=wandb_coordinator,
                        copy_to_new_experiment=True,
                    )
                ]
                if not resolved_config.no_wandb
                else []
            ),
            *inference_steps,
        ]
    elif resolved_config.run_mode == "evidence-only":
        preparation_steps = setup_steps
        selected_evaluation_steps = [
            LoadGnnAnswerRetrieverRunStep(
                run_name=resolved_config.retriever_run_name,
                run_number=resolved_config.retriever_run_number,
            ),
            *(
                [
                    LogRetrieverToWandbStep(
                        coordinator=wandb_coordinator,
                        copy_to_new_experiment=True,
                    )
                ]
                if not resolved_config.no_wandb
                else []
            ),
            *evidence_only_steps,
        ]
    else:
        preparation_steps = [*setup_steps, *training_steps]
        selected_evaluation_steps = [*retriever_steps]
        if not resolved_config.no_llm_inference:
            selected_evaluation_steps.extend(inference_steps)

    return Pipeline(
        preparation_steps=preparation_steps,
        evaluation_steps=selected_evaluation_steps,
        force_all_default=resolved_config.force_all_default,
        completion_callbacks=[wandb_coordinator.finish]
        if not resolved_config.no_wandb
        else [],
    )


def run_pipeline(config: PipelineRuntimeConfig) -> PipelineExecutionResult:
    """Run the full graphragX pipeline."""
    if (
        config.run_mode == "evaluation-only"
        and config.evaluation_model_run_name is None
        and config.evaluation_model_run_number is None
    ):
        raise PipelineException(
            "Evaluation-only mode requires --evaluation-model-run-name or "
            "--evaluation-model-run-number."
        )
    if (
        config.run_mode in {"inference-only", "evidence-only"}
        and config.retriever_run_name is None
        and config.retriever_run_number is None
    ):
        raise PipelineException(
            f"{config.run_mode} mode requires --retriever-run-name or "
            "--retriever-run-number."
        )

    configure_random_seed(config.random_seed)
    pipeline = build_pipeline(config=config)
    initial_context = StepContext(result=InitialStepResult())
    if config.run_mode == "train-only":
        return pipeline.prepare(initial_context)

    return pipeline.run(initial_context)


def _serialize_value(value: Any) -> Any:
    """Convert nested Pydantic objects into JSON-serializable values."""
    if isinstance(value, BaseModel):
        return {key: _serialize_value(item) for key, item in value.model_dump().items()}

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    return value


def serialize_execution_result(result: PipelineExecutionResult) -> dict[str, Any]:
    """Convert a pipeline execution result into a JSON-serializable dictionary."""
    payload = result.model_dump()
    payload["final_result"] = _serialize_value(result.final_result)
    return payload


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summary_line(label: str, value: Any) -> str:
    return f"  {label}: {_format_summary_value(value)}"


def _build_success_summary_lines(result: PipelineExecutionResult) -> list[str]:
    final_result = result.final_result
    lines = [
        _summary_line("steps", f"{result.steps_executed}/{result.total_steps}"),
        _summary_line("execution_time_ms", result.execution_time_ms),
    ]
    if final_result is None:
        return lines

    summary_fields = [
        "gnn_architecture",
        "subgraph_algorithm",
        "model_run_name",
        "evaluation_run_name",
        "evidence_run_name",
        "inference_run_name",
        "results_run_name",
        "evaluated_instances",
        "hit_rate",
        "f1",
        "ndcg_at_10",
        "grounded_explanation_rate",
        "wandb_status",
        "wandb_run_url",
    ]
    for field_name in summary_fields:
        if hasattr(final_result, field_name):
            lines.append(_summary_line(field_name, getattr(final_result, field_name)))

    if hasattr(final_result, "wandb_error_message"):
        error_message = getattr(final_result, "wandb_error_message")
        if error_message:
            lines.append(_summary_line("wandb_error_message", error_message))

    return lines


def _log_summary_banner(title: str, lines: list[str], color: str) -> None:
    content_width = max(
        [len(title), *(len(line) for line in lines)],
        default=len(title),
    )
    border = "═" * max(content_width + 4, 72)
    body = "\n".join(lines)
    logger.info(
        "\n\n%s%s\n%s\n%s\n%s",
        color,
        border,
        title.center(len(border)),
        body,
        f"{border}\033[0m",
    )


def log_success_summary(result: PipelineExecutionResult) -> None:
    """Log a compact human-readable successful run summary."""
    _log_summary_banner(
        title="RUN SUMMARY",
        lines=_build_success_summary_lines(result),
        color="\033[92m",
    )


def log_error_summary(
    error_message: str,
    exception_type: str,
    result: PipelineExecutionResult | None = None,
) -> None:
    """Log a compact human-readable failure summary."""
    lines = [
        _summary_line("success", False),
        _summary_line("exception_type", exception_type),
        _summary_line("error_message", error_message),
    ]
    if result is not None:
        lines.insert(1, _summary_line("steps", f"{result.steps_executed}/{result.total_steps}"))
        lines.insert(2, _summary_line("execution_time_ms", result.execution_time_ms))

    _log_summary_banner(
        title="PIPELINE FAILED",
        lines=lines,
        color="\033[91m",
    )


def _add_gnn_architecture_option_arguments(parser: argparse.ArgumentParser) -> None:
    """Generate the union of architecture-owned CLI flags from the registry."""
    type_map = {"integer": int, "float": float, "string": str}
    for option in architecture_option_definitions().values():
        kwargs: dict[str, Any] = {
            "dest": option.option_id,
            "default": None,
            "help": option.description,
        }
        if option.value_type == "boolean":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type_map[option.value_type]
            if option.choices:
                kwargs["choices"] = option.choices
        parser.add_argument(option.cli_flag, **kwargs)


def _non_negative_integer(value: str) -> int:
    """Parse a non-negative integer for reproducible random seeds."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the framework entry point."""
    parser = argparse.ArgumentParser(description="Run graphragX.")
    run_mode_group = parser.add_mutually_exclusive_group()
    run_mode_group.add_argument(
        "--full",
        dest="run_mode",
        action="store_const",
        const="full",
        default="full",
        help="Run training and evaluation. This is the default.",
    )
    run_mode_group.add_argument(
        "--train-only",
        dest="run_mode",
        action="store_const",
        const="train-only",
        help="Run setup and GNN training, then stop successfully.",
    )
    run_mode_group.add_argument(
        "--evaluation-only",
        dest="run_mode",
        action="store_const",
        const="evaluation-only",
        help="Run setup and evaluate a saved model run.",
    )
    run_mode_group.add_argument(
        "--retriever-only",
        dest="run_mode",
        action="store_const",
        const="retriever-only",
        help="Train and evaluate the GNN retriever, then stop before LLM inference.",
    )
    run_mode_group.add_argument(
        "--inference-only",
        dest="run_mode",
        action="store_const",
        const="inference-only",
        help="Run LLM inference and final results from a saved retriever run.",
    )
    run_mode_group.add_argument(
        "--evidence-only",
        dest="run_mode",
        action="store_const",
        const="evidence-only",
        help="Build and save evidence subgraphs from a saved retriever run.",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=_non_negative_integer,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Seed Python, NumPy, and PyTorch before model construction. "
            f"Default: {DEFAULT_RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset choice for the current run.",
    )
    parser.add_argument(
        "--local-graph-profile",
        action="store_true",
        default=False,
        help="Report detailed WebQSP graph processing and cache phase timings.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=tuple(LLM_PROVIDERS),
        default=None,
        help=(
            "LLM provider used for inference. OpenAI is used when omitted. "
            "DeepSeek and Vezilka models require an explicit provider; Vezilka "
            "accepts any value passed through --main-llm-model."
        ),
    )
    parser.add_argument(
        "--main-llm-model",
        default=None,
        help="Optional main LLM model id for non-interactive configuration.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Optional reasoning_effort value passed to the selected LLM provider, "
            "for example 'none', 'low', 'medium', or 'high'. When omitted, the "
            "field is not sent."
        ),
    )
    parser.add_argument(
        "--generate-explanation",
        action="store_true",
        default=False,
        help=(
            "Ask the LLM to generate and save an explanation. By default only "
            "the answer is generated and explanation metrics remain zero."
        ),
    )
    parser.add_argument(
        "--subgraph-algorithm",
        choices=tuple(SUBGRAPH_CONSTRUCTION_ALGORITHMS),
        default=None,
        help="Optional subgraph construction algorithm id for non-interactive configuration.",
    )
    parser.add_argument(
        "--pcst-edge-cost-strategy",
        choices=("constant", "semantic"),
        default=None,
        help="PCST edge-cost strategy. Defaults to constant when PCST is selected.",
    )
    parser.add_argument(
        "--pcst-edge-cost",
        type=float,
        default=None,
        help="Positive PCST edge-cost lambda. Defaults to 1.0.",
    )
    parser.add_argument(
        "--pcst-debug-profile",
        action="store_true",
        default=False,
        help=(
            "Save a detailed replayable JSON diagnostic when PCST returns an "
            "invalid rooted solution."
        ),
    )
    parser.add_argument(
        "--context-strategy",
        default=None,
        help="Optional context construction strategy id for non-interactive configuration.",
    )
    parser.add_argument(
        "--gnn-architecture",
        choices=tuple(GNN_ARCHITECTURES),
        default=None,
        help="GNN architecture. Defaults to graphsage.",
    )
    _add_gnn_architecture_option_arguments(parser)
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="OpenAI embedding model used for question, relation, and entity text.",
    )
    # Accept old flags without advertising separate choices. They are mapped to
    # the unified selector in main() for scripts written before this refactor.
    parser.add_argument(
        "--question-embedding-model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--relation-embedding-model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--entity-embedding-model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help="Number of GNN answer-retriever training epochs.",
    )
    parser.add_argument(
        "--training-learning-rate",
        type=float,
        default=DEFAULT_TRAINING_LEARNING_RATE,
        help="Learning rate for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-weight-decay",
        type=float,
        default=DEFAULT_TRAINING_WEIGHT_DECAY,
        help="Weight decay for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-max-instances",
        type=int,
        default=None,
        help="Optional maximum number of WebQSP training instances to use.",
    )
    parser.add_argument(
        "--training-start-instance",
        type=int,
        default=0,
        help="Zero-based train split index where GNN training should start.",
    )
    parser.add_argument(
        "--training-log-every",
        type=int,
        default=DEFAULT_TRAINING_LOG_EVERY,
        help="Write console training progress after this many instances. Use 0 to disable.",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=DEFAULT_TRAINING_BATCH_SIZE,
        help=(
            "Number of disconnected WebQSP graphs per R-GCN, HGT, or ReaRev "
            "optimizer step. GraphSAGE retains single-graph training."
        ),
    )
    parser.add_argument(
        "--wandb-training-log-every",
        type=int,
        default=DEFAULT_WANDB_TRAINING_LOG_EVERY,
        help="Send live training loss to W&B after this many instances. Use 0 to disable.",
    )
    parser.add_argument(
        "--training-device",
        default=DEFAULT_TRAINING_DEVICE,
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device used for GNN answer-retriever training.",
    )
    parser.add_argument(
        "--training-profile",
        action="store_true",
        default=DEFAULT_TRAINING_PROFILE,
        help="Synchronize and report detailed GNN training phase timings.",
    )
    parser.add_argument(
        "--training-embedding-cache-device",
        choices=["auto", "gpu", "cpu"],
        default=DEFAULT_TRAINING_EMBEDDING_CACHE_DEVICE,
        help="Place compact frozen training embeddings on GPU, CPU, or automatically.",
    )
    parser.add_argument(
        "--training-embedding-cache-dtype",
        choices=["auto", "float32", "bfloat16"],
        default=DEFAULT_TRAINING_EMBEDDING_CACHE_DTYPE,
        help="Storage precision for compact frozen training embeddings.",
    )
    parser.add_argument(
        "--training-gpu-cache-reserve-gb",
        type=float,
        default=DEFAULT_TRAINING_GPU_CACHE_RESERVE_GB,
        help="GPU memory reserved for the model, activations, and CUDA overhead.",
    )
    parser.add_argument(
        "--training-run-name",
        default=None,
        help="Optional label for the versioned training run folder.",
    )
    parser.add_argument(
        "--continue-training-model-run-name",
        default=None,
        help="Saved model run folder name or suffix to continue training from.",
    )
    parser.add_argument(
        "--continue-training-model-run-number",
        type=int,
        default=None,
        help="Saved model run numeric prefix to continue training from.",
    )
    parser.add_argument(
        "--evaluation-model-run-name",
        default=None,
        help="Saved model run folder name or suffix to evaluate.",
    )
    parser.add_argument(
        "--evaluation-model-run-number",
        type=int,
        default=None,
        help="Saved model run numeric prefix to evaluate.",
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=DEFAULT_ANSWER_THRESHOLD,
        help="Minimum answer-node probability for threshold candidate selection.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help=(
            "Minimum selected candidate count when threshold selection is too small "
            f"(must be at least {MIN_CANDIDATE_TOP_K})."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        "--limit",
        dest="candidate_limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Maximum selected candidate count after threshold selection.",
    )
    parser.add_argument(
        "--evaluation-run-name",
        default=None,
        help="Optional label for the versioned evaluation run folder.",
    )
    retriever_run_group = parser.add_mutually_exclusive_group()
    retriever_run_group.add_argument(
        "--retriever-run-name",
        default=None,
        help="Saved retriever evaluation run folder name or suffix.",
    )
    retriever_run_group.add_argument(
        "--retriever-run-number",
        type=int,
        default=None,
        help="Saved retriever evaluation run numeric prefix.",
    )
    parser.add_argument(
        "--evaluation-max-instances",
        type=int,
        default=None,
        help="Optional maximum number of WebQSP test instances to evaluate.",
    )
    parser.add_argument(
        "--no-skip-missing-gold-in-graph",
        dest="skip_missing_gold_in_graph",
        action="store_false",
        default=True,
        help=(
            "Include training/evaluation graphs with one or more gold answer "
            "entities absent. By default these incomplete graphs are skipped."
        ),
    )
    parser.add_argument(
        "--evaluation-log-every",
        type=int,
        default=DEFAULT_EVALUATION_LOG_EVERY,
        help="Log GNN evaluation progress after this many evaluated instances.",
    )
    parser.add_argument(
        "--evaluation-profile",
        action="store_true",
        default=DEFAULT_EVALUATION_PROFILE,
        help="Synchronize and report detailed GNN evaluation phase timings.",
    )
    parser.add_argument(
        "--evaluation-embedding-cache-device",
        choices=["auto", "gpu", "cpu"],
        default=DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE,
        help="Place compact frozen evaluation embeddings on GPU, CPU, or automatically.",
    )
    parser.add_argument(
        "--evaluation-embedding-cache-dtype",
        choices=["auto", "float32", "bfloat16"],
        default=DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE,
        help="Storage precision for compact frozen evaluation embeddings.",
    )
    parser.add_argument(
        "--evaluation-gpu-cache-reserve-gb",
        type=float,
        default=DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB,
        help="GPU memory reserved outside the evaluation embedding cache.",
    )
    parser.add_argument(
        "--no-llm-inference",
        dest="no_llm_inference",
        action="store_true",
        help="Stop after GNN candidate retrieval and skip final LLM inference/results.",
    )
    parser.add_argument(
        "--inference-run-name",
        default=None,
        help="Optional label for the versioned LLM inference run folder.",
    )
    parser.add_argument(
        "--evidence-run-name",
        default=None,
        help="Optional label for the versioned evidence-only run folder.",
    )
    parser.add_argument(
        "--llm-inference-batch-size",
        type=int,
        default=10,
        help="Number of samples to generate and save per LLM inference batch.",
    )
    parser.add_argument(
        "--llm-inference-parallel-calls",
        type=int,
        default=1,
        help="Maximum simultaneous LLM API calls. Defaults to sequential inference.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="no_wandb",
        action="store_true",
        help="Disable W&B logging for every pipeline stage.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional WandB project. Defaults to WANDB_PROJECT or graphragx.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional WandB entity/team. Defaults to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional WandB mode. Defaults to WANDB_MODE or online.",
    )
    parser.add_argument(
        "--wandb-upload-retriever",
        dest="wandb_upload_retriever",
        action="store_true",
        help=(
            "Upload the trained GNN retriever weights to W&B. By default, "
            "W&B receives retriever configs and metrics but not the large model file."
        ),
    )
    parser.add_argument(
        "--default",
        dest="use_default_config_values",
        action="store_true",
        help="Use default values for all configurable user selections.",
    )
    parser.add_argument(
        "--force-default",
        dest="force_all_default",
        action="store_true",
        help="Force every pipeline step to use its execute_default path.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for graphragX."""
    setup_logger()
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_config = PipelineRuntimeConfig(
        run_mode=args.run_mode,
        random_seed=args.random_seed,
        dataset=args.dataset,
        local_graph_profile=args.local_graph_profile,
        llm_provider=args.llm_provider,
        reasoning_effort=args.reasoning_effort,
        generate_explanation=args.generate_explanation,
        main_llm_model=args.main_llm_model,
        subgraph_algorithm=args.subgraph_algorithm,
        pcst_edge_cost_strategy=args.pcst_edge_cost_strategy,
        pcst_edge_cost=args.pcst_edge_cost,
        pcst_debug_profile=args.pcst_debug_profile,
        context_strategy=args.context_strategy,
        gnn_architecture=args.gnn_architecture,
        gnn_layer_count=args.gnn_layer_count,
        gnn_hidden_dimension=args.gnn_hidden_dimension,
        node_classifier=args.node_classifier,
        use_edge_mlp=args.use_edge_mlp,
        question_aware_classifier=args.question_aware_classifier,
        use_reverse_edges=args.use_reverse_edges,
        add_layer_normalization=args.add_layer_normalization,
        edge_mlp_hidden_dim=args.edge_mlp_hidden_dim,
        dropout=args.dropout,
        gnn_options={
            option_id: getattr(args, option_id)
            for option_id in architecture_option_definitions()
            if getattr(args, option_id) is not None
        },
        embedding_model=(
            args.embedding_model
            or args.entity_embedding_model
            or args.question_embedding_model
            or args.relation_embedding_model
        ),
        question_embedding_model=args.question_embedding_model,
        relation_embedding_model=args.relation_embedding_model,
        entity_embedding_model=args.entity_embedding_model,
        training_epochs=args.training_epochs,
        training_learning_rate=args.training_learning_rate,
        training_weight_decay=args.training_weight_decay,
        training_max_instances=args.training_max_instances,
        training_start_instance=args.training_start_instance,
        skip_missing_gold_in_graph=args.skip_missing_gold_in_graph,
        training_log_every=args.training_log_every,
        training_batch_size=args.training_batch_size,
        wandb_training_log_every=args.wandb_training_log_every,
        training_device=args.training_device,
        training_profile=args.training_profile,
        training_embedding_cache_device=args.training_embedding_cache_device,
        training_embedding_cache_dtype=args.training_embedding_cache_dtype,
        training_gpu_cache_reserve_gb=args.training_gpu_cache_reserve_gb,
        training_run_name=args.training_run_name,
        continue_training_model_run_name=args.continue_training_model_run_name,
        continue_training_model_run_number=args.continue_training_model_run_number,
        evaluation_model_run_name=args.evaluation_model_run_name,
        evaluation_model_run_number=args.evaluation_model_run_number,
        answer_threshold=args.answer_threshold,
        candidate_top_k=args.candidate_top_k,
        candidate_limit=args.candidate_limit,
        evaluation_run_name=args.evaluation_run_name,
        retriever_run_name=args.retriever_run_name,
        retriever_run_number=args.retriever_run_number,
        evaluation_max_instances=args.evaluation_max_instances,
        evaluation_log_every=args.evaluation_log_every,
        evaluation_profile=args.evaluation_profile,
        evaluation_embedding_cache_device=args.evaluation_embedding_cache_device,
        evaluation_embedding_cache_dtype=args.evaluation_embedding_cache_dtype,
        evaluation_gpu_cache_reserve_gb=args.evaluation_gpu_cache_reserve_gb,
        no_llm_inference=args.no_llm_inference,
        inference_run_name=args.inference_run_name,
        evidence_run_name=args.evidence_run_name,
        llm_inference_batch_size=args.llm_inference_batch_size,
        llm_inference_parallel_calls=args.llm_inference_parallel_calls,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_upload_retriever=args.wandb_upload_retriever,
        use_default_config_values=args.use_default_config_values,
        force_all_default=args.force_all_default,
    )

    try:
        result = run_pipeline(config=runtime_config)
    except Exception as error:
        log_error_summary(
            error_message=str(error),
            exception_type=error.__class__.__name__,
        )
        return 1

    if result.success:
        log_success_summary(result)
        return 0

    log_error_summary(
        error_message=result.error_message or "Unknown pipeline failure.",
        exception_type=result.exception_type or "PipelineExecutionError",
        result=result,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
