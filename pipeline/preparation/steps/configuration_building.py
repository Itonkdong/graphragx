"""Pipeline configuration building step for the preparation pipeline."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, BaseModel

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import (
    InvalidContextConstructionSelectionException,
    InvalidEntityEmbeddingModelSelectionException,
    InvalidGnnHiddenDimensionSelectionException,
    InvalidGnnArchitectureConfigurationException,
    InvalidGnnArchitectureSelectionException,
    InvalidGnnLayerCountSelectionException,
    InvalidInteractiveConfigurationInputException,
    InvalidLlmProviderSelectionException,
    InvalidMainLlmSelectionException,
    InvalidNodeClassifierSelectionException,
    InvalidSubgraphConstructionSelectionException,
)
from pipeline.preparation.services.selection import SelectionService
from pipeline.preparation.helpers.configuration_definitions import (
    CONTEXT_CONSTRUCTION_STRATEGIES,
    GNN_ARCHITECTURES,
    GnnArchitectureOptionDefinition,
    GRAPH_SAGE_ARCHITECTURE_ID,
    LLM_PROVIDERS,
    OPENAI_EMBEDDING_MODELS,
    PCST_EDGE_COST_STRATEGIES,
    RECOMMENDED_PCST_EDGE_COST_STRATEGY_ID,
    DEFAULT_PCST_EDGE_COST,
    RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
    RECOMMENDED_GNN_ARCHITECTURE_ID,
    RECOMMENDED_MAIN_LLM_MODEL_ID,
    RECOMMENDED_LLM_PROVIDER_ID,
    RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
    RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
    SHARED_LLM_MODELS,
    SUBGRAPH_CONSTRUCTION_ALGORITHMS,
)
from pipeline.preparation.helpers.gnn_architecture import (
    GnnArchitectureOptionValidationError,
    validate_architecture_options,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset

logger = get_logger(__name__)


class PipelineConfigurationInput(BaseModel):
    """Optional programmatic input for pipeline configuration building."""

    llm_provider: str | None = Field(default=None)
    reasoning_effort: str | None = Field(default=None)
    main_llm_model: str | None = Field(default=None)
    subgraph_construction_algorithm: str | None = Field(default=None)
    pcst_edge_cost_strategy: str | None = Field(default=None)
    pcst_edge_cost: float | None = Field(default=None)
    context_construction_strategy: str | None = Field(default=None)
    gnn_architecture: str | None = Field(default=None)
    gnn_layer_count: int | None = Field(default=None)
    gnn_hidden_dimension: int | None = Field(default=None)
    node_classifier: str | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    # Legacy per-resource fields remain accepted for programmatic callers and
    # old pipeline integrations. New configuration always resolves one model.
    question_embedding_model: str | None = Field(default=None)
    relation_embedding_model: str | None = Field(default=None)
    entity_embedding_model: str | None = Field(default=None)
    use_edge_mlp: bool | None = Field(default=None)
    question_aware_classifier: bool | None = Field(default=None)
    use_reverse_edges: bool | None = Field(default=None)
    add_layer_normalization: bool | None = Field(default=None)
    edge_mlp_hidden_dim: int | None = Field(default=None)
    dropout: float | None = Field(default=None)
    gnn_options: dict[str, Any] = Field(default_factory=dict)


class BuiltPipelineConfiguration(StepResult):
    """Unified pipeline configuration artifact."""

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    gnn_architecture: str = Field(default="graphsage", description="Selected GNN architecture id.")
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    llm_provider: str = Field(default="openai", description="Selected LLM provider id.")
    reasoning_effort: str | None = Field(
        default=None,
        description="Optional reasoning effort passed to the selected LLM provider.",
    )
    main_llm_model: str = Field(..., description="Selected main LLM model id.")
    subgraph_construction_algorithm: str = Field(
        ..., description="Selected subgraph construction algorithm id."
    )
    pcst_edge_cost_strategy: str | None = Field(default=None)
    pcst_edge_cost: float | None = Field(default=None)
    context_construction_strategy: str = Field(
        ..., description="Selected context construction strategy id."
    )
    gnn_layer_count: int | None = Field(default=None, description="Selected number of GNN layers.")
    gnn_hidden_dimension: int | None = Field(
        default=None,
        description="Selected hidden dimension for projected GNN node states.",
    )
    node_classifier: str | None = Field(default=None, description="Selected node classifier id.")
    embedding_model: str | None = Field(
        default=None,
        description="OpenAI embedding model used for all graph and question text.",
    )
    # Compatibility aliases. They are resolved to embedding_model by the builder.
    question_embedding_model: str | None = Field(
        default=None, description="OpenAI embedding model for question text."
    )
    relation_embedding_model: str | None = Field(
        default=None, description="OpenAI embedding model for relation text."
    )
    entity_embedding_model: str | None = Field(
        default=None, description="OpenAI embedding model for entity text."
    )
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None)
    dropout: float = Field(default=0.1)


class BuildPipelineConfigurationStep(
    AbstractStep[BuiltPipelineConfiguration, SelectedDataset]
):
    """Build the core pipeline configuration after dataset selection."""

    def __init__(
        self,
        llm_provider: str | None = None,
        reasoning_effort: str | None = None,
        main_llm_model: str | None = None,
        subgraph_algorithm: str | None = None,
        pcst_edge_cost_strategy: str | None = None,
        pcst_edge_cost: float | None = None,
        context_strategy: str | None = None,
        gnn_architecture: str | None = None,
        gnn_layer_count: int | None = None,
        gnn_hidden_dimension: int | None = None,
        node_classifier: str | None = None,
        embedding_model: str | None = None,
        question_embedding_model: str | None = None,
        relation_embedding_model: str | None = None,
        entity_embedding_model: str | None = None,
        use_edge_mlp: bool | None = None,
        question_aware_classifier: bool | None = None,
        use_reverse_edges: bool | None = None,
        add_layer_normalization: bool | None = None,
        edge_mlp_hidden_dim: int | None = None,
        dropout: float | None = None,
        gnn_options: dict[str, Any] | None = None,
        input_func=None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.configuration_input = PipelineConfigurationInput(
            llm_provider=llm_provider,
            reasoning_effort=reasoning_effort,
            main_llm_model=main_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            pcst_edge_cost_strategy=pcst_edge_cost_strategy,
            pcst_edge_cost=pcst_edge_cost,
            context_construction_strategy=context_strategy,
            gnn_architecture=gnn_architecture,
            gnn_layer_count=gnn_layer_count,
            gnn_hidden_dimension=gnn_hidden_dimension,
            node_classifier=node_classifier,
            embedding_model=embedding_model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
            use_edge_mlp=use_edge_mlp,
            question_aware_classifier=question_aware_classifier,
            use_reverse_edges=use_reverse_edges,
            add_layer_normalization=add_layer_normalization,
            edge_mlp_hidden_dim=edge_mlp_hidden_dim,
            dropout=dropout,
            gnn_options=gnn_options or {},
        )
        self.selection_service = SelectionService(input_func=input_func)

    def execute_default(
        self,
        context: StepContext[SelectedDataset],
    ) -> BuiltPipelineConfiguration:
        selected_dataset = context.result
        if selected_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "Configuration building requires a selected dataset in the incoming context."
            )

        logger.info(f"Building pipeline configuration for dataset={selected_dataset.dataset_id}")
        provided_options = self._provided_architecture_options()
        implicit_graphsage = (
            GRAPH_SAGE_ARCHITECTURE_ID
            if self.configuration_input.gnn_architecture is None
            and bool(provided_options)
            else None
        )
        gnn_architecture = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.gnn_architecture or implicit_graphsage,
            options=GNN_ARCHITECTURES,
            prompt_title="GNN Architecture",
            prompt_help="Select the GNN architecture before configuring its options.",
            recommended_id=RECOMMENDED_GNN_ARCHITECTURE_ID,
            invalid_exception_type=InvalidGnnArchitectureSelectionException,
            value_getter=lambda item: item.architecture_id,
            label_getter=lambda item: item.display_name,
        )
        architecture = GNN_ARCHITECTURES[gnn_architecture]
        unsupported_options = sorted(set(provided_options) - set(architecture.option_map))
        if unsupported_options:
            raise InvalidGnnArchitectureConfigurationException(
                f"Architecture {gnn_architecture} does not support: "
                + ", ".join(unsupported_options)
            )
        shared_options_complete = all(
            option_id in provided_options
            for option_id in ("gnn_layer_count", "gnn_hidden_dimension", "node_classifier")
        )
        if shared_options_complete and "dropout" not in provided_options:
            provided_options["dropout"] = architecture.option_map["dropout"].default

        resolved_gnn_options: dict[str, Any] = {}
        for option in architecture.options:
            if (
                option.enabled_when_option is not None
                and resolved_gnn_options.get(option.enabled_when_option)
                != option.enabled_when_value
            ):
                if option.option_id in provided_options:
                    raise InvalidGnnArchitectureConfigurationException(
                        f"{option.cli_flag} cannot be used when "
                        f"{option.enabled_when_option} is disabled."
                    )
                resolved_gnn_options[option.option_id] = None
                continue
            resolved_gnn_options[option.option_id] = self._resolve_architecture_option(
                option=option,
                provided_value=provided_options.get(option.option_id),
                architecture_name=architecture.display_name,
            )

        while True:
            try:
                validate_architecture_options(gnn_architecture, resolved_gnn_options)
                break
            except GnnArchitectureOptionValidationError as error:
                if error.option_id in provided_options:
                    raise InvalidGnnArchitectureConfigurationException(str(error)) from error
                self.selection_service.output_func(str(error))
                option = architecture.option_map[error.option_id]
                resolved_gnn_options[error.option_id] = self._resolve_architecture_option(
                    option=option,
                    provided_value=None,
                    architecture_name=architecture.display_name,
                )
        requested_model = self.configuration_input.main_llm_model
        if (
            self.configuration_input.llm_provider is None
            and requested_model is not None
            and (
                requested_model.startswith("deepseek-")
                or (
                    SHARED_LLM_MODELS.get(requested_model) is not None
                    and SHARED_LLM_MODELS[requested_model].provider_id == "deepseek"
                )
            )
        ):
            raise InvalidLlmProviderSelectionException(
                f"Model {requested_model} requires explicit provider selection: "
                "use llm_provider='deepseek'."
            )
        if (
            self.configuration_input.llm_provider is None
            and requested_model is not None
            and requested_model not in SHARED_LLM_MODELS
        ):
            raise InvalidMainLlmSelectionException(
                f"Model {requested_model} is not a configured OpenAI model. "
                "Privately hosted models require explicit provider selection: "
                "use llm_provider='vezilka'."
            )
        inferred_provider = self._infer_llm_provider(requested_model)
        llm_provider = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.llm_provider or inferred_provider,
            options=LLM_PROVIDERS,
            prompt_title="LLM Provider",
            prompt_help="Select the provider used for final question answering.",
            recommended_id=RECOMMENDED_LLM_PROVIDER_ID,
            invalid_exception_type=InvalidLlmProviderSelectionException,
            value_getter=lambda item: item.provider_id,
            label_getter=lambda item: item.display_name,
        )
        if LLM_PROVIDERS[llm_provider].accepts_arbitrary_models:
            main_llm_model = self.selection_service.resolve_text(
                provided_value=self.configuration_input.main_llm_model,
                prompt="Enter the Vezilka model name: ",
                invalid_exception_type=InvalidMainLlmSelectionException,
            )
        else:
            provider_models = {
                model_id: definition
                for model_id, definition in SHARED_LLM_MODELS.items()
                if definition.provider_id == llm_provider
            }
            recommended_model = (
                RECOMMENDED_MAIN_LLM_MODEL_ID
                if llm_provider == RECOMMENDED_LLM_PROVIDER_ID
                else next(iter(provider_models))
            )
            main_llm_model = self.selection_service.resolve_choice(
                provided_value=self.configuration_input.main_llm_model,
                options=provider_models,
                prompt_title="Main LLM Model",
                prompt_help="Select the primary model used for final question answering.",
                recommended_id=recommended_model,
                invalid_exception_type=InvalidMainLlmSelectionException,
                value_getter=lambda item: item.model_id,
                label_getter=lambda item: item.display_name,
            )
        subgraph_algorithm = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.subgraph_construction_algorithm,
            options=SUBGRAPH_CONSTRUCTION_ALGORITHMS,
            prompt_title="Subgraph Construction Algorithm",
            prompt_help="Select the algorithm used to build question-specific subgraphs.",
            recommended_id=RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID,
            invalid_exception_type=InvalidSubgraphConstructionSelectionException,
            value_getter=lambda item: item.algorithm_id,
            label_getter=lambda item: item.display_name,
        )
        pcst_edge_cost_strategy = None
        pcst_edge_cost = None
        if subgraph_algorithm == "pcst":
            pcst_edge_cost_strategy = self.selection_service.resolve_choice(
                provided_value=self.configuration_input.pcst_edge_cost_strategy,
                options=PCST_EDGE_COST_STRATEGIES,
                prompt_title="PCST Edge Cost Strategy",
                prompt_help="Select how PCST assigns costs to graph relations.",
                recommended_id=RECOMMENDED_PCST_EDGE_COST_STRATEGY_ID,
                invalid_exception_type=InvalidSubgraphConstructionSelectionException,
                value_getter=lambda item: item.strategy_id,
                label_getter=lambda item: item.display_name,
            )
            pcst_edge_cost = self._resolve_pcst_edge_cost(
                self.configuration_input.pcst_edge_cost
            )
        context_strategy = self.selection_service.resolve_choice(
            provided_value=self.configuration_input.context_construction_strategy,
            options=CONTEXT_CONSTRUCTION_STRATEGIES,
            prompt_title="Context Construction Strategy",
            prompt_help="Select how the retrieved subgraph will be represented for the LLM.",
            recommended_id=RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID,
            invalid_exception_type=InvalidContextConstructionSelectionException,
            value_getter=lambda item: item.strategy_id,
            label_getter=lambda item: item.display_name,
        )
        embedding_model = None
        if any(
            (
                architecture.data_requirements.uses_entity_embeddings,
                architecture.data_requirements.uses_question_embeddings,
                architecture.data_requirements.uses_relation_embeddings,
                subgraph_algorithm == "pcst"
                and pcst_edge_cost_strategy == "semantic",
            )
        ):
            embedding_model = self.selection_service.resolve_choice(
                provided_value=self._provided_embedding_model(),
                options=OPENAI_EMBEDDING_MODELS,
                prompt_title="Embedding Model",
                prompt_help="Select the OpenAI embedding model used everywhere embeddings are needed.",
                recommended_id=RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID,
                # Keep the historical exception type for callers that catch it;
                # there is now only one embedding selection.
                invalid_exception_type=InvalidEntityEmbeddingModelSelectionException,
                value_getter=lambda item: item.model_id,
                label_getter=lambda item: item.display_name,
            )
        question_embedding_model = embedding_model
        relation_embedding_model = embedding_model
        entity_embedding_model = embedding_model

        logger.info(
            f"Built pipeline configuration: gnn_architecture={gnn_architecture} "
            f"gnn_options={resolved_gnn_options} "
            f"embedding_model={embedding_model}"
        )
        return BuiltPipelineConfiguration(
            dataset_id=selected_dataset.dataset_id,
            llm_provider=llm_provider,
            reasoning_effort=self.configuration_input.reasoning_effort,
            gnn_architecture=gnn_architecture,
            gnn_architecture_options=resolved_gnn_options,
            main_llm_model=main_llm_model,
            subgraph_construction_algorithm=subgraph_algorithm,
            pcst_edge_cost_strategy=pcst_edge_cost_strategy,
            pcst_edge_cost=pcst_edge_cost,
            context_construction_strategy=context_strategy,
            gnn_layer_count=resolved_gnn_options.get("gnn_layer_count"),
            gnn_hidden_dimension=resolved_gnn_options.get("gnn_hidden_dimension"),
            node_classifier=resolved_gnn_options.get("node_classifier"),
            embedding_model=embedding_model,
            question_embedding_model=question_embedding_model,
            relation_embedding_model=relation_embedding_model,
            entity_embedding_model=entity_embedding_model,
            use_edge_mlp=bool(resolved_gnn_options.get("use_edge_mlp", False)),
            question_aware_classifier=bool(
                resolved_gnn_options.get("question_aware_classifier", False)
            ),
            use_reverse_edges=(
                architecture.data_requirements.requires_reverse_edges
                or bool(resolved_gnn_options.get("use_reverse_edges", False))
            ),
            add_layer_normalization=bool(
                resolved_gnn_options.get("add_layer_normalization", False)
            ),
            edge_mlp_hidden_dim=resolved_gnn_options.get("edge_mlp_hidden_dim"),
            dropout=float(resolved_gnn_options.get("dropout", 0.0)),
        )

    def _resolve_pcst_edge_cost(self, provided_value: float | None) -> float:
        """Resolve a positive finite PCST lambda from CLI or interactive input."""
        if provided_value is not None:
            value = float(provided_value)
            if math.isfinite(value) and value > 0:
                return value
            raise InvalidSubgraphConstructionSelectionException(
                "PCST edge cost must be a finite value greater than zero."
            )
        while True:
            try:
                raw_value = self.selection_service.input_func(
                    f"Enter PCST edge cost lambda [{DEFAULT_PCST_EDGE_COST}]: "
                ).strip()
            except (EOFError, KeyboardInterrupt) as error:
                raise InvalidSubgraphConstructionSelectionException(
                    "Unable to read the PCST edge cost."
                ) from error
            try:
                value = DEFAULT_PCST_EDGE_COST if not raw_value else float(raw_value)
            except ValueError:
                self.selection_service.output_func(
                    "Invalid value. Enter a finite number greater than zero."
                )
                continue
            if math.isfinite(value) and value > 0:
                return value
            self.selection_service.output_func(
                "Invalid value. Enter a finite number greater than zero."
            )

    @staticmethod
    def _infer_llm_provider(model_id: str | None) -> str | None:
        if model_id is None:
            return None
        # OpenAI remains the recommended/default provider. Unknown model names
        # are validated against that provider unless Vezilka is selected.
        return "openai"

    def _provided_embedding_model(self) -> str | None:
        """Resolve the unified model while accepting legacy constructor fields."""
        unified = self.configuration_input.embedding_model
        legacy_values = [
            value
            for value in (
                self.configuration_input.entity_embedding_model,
                self.configuration_input.question_embedding_model,
                self.configuration_input.relation_embedding_model,
            )
            if value is not None
        ]
        if unified is not None:
            conflicting = {value for value in legacy_values if value != unified}
            if conflicting:
                raise InvalidInteractiveConfigurationInputException(
                    "embedding_model conflicts with a legacy per-resource embedding model."
                )
            return unified
        # Entity was the historical primary selector, so preserve that choice
        # when old programmatic callers provide only legacy fields.
        if len(set(legacy_values)) > 1 and all(
            value in OPENAI_EMBEDDING_MODELS for value in legacy_values
        ):
            raise InvalidInteractiveConfigurationInputException(
                "Legacy per-resource embedding models disagree; use one embedding_model."
            )
        if self.configuration_input.entity_embedding_model is not None:
            return self.configuration_input.entity_embedding_model
        if self.configuration_input.question_embedding_model is not None:
            return self.configuration_input.question_embedding_model
        return self.configuration_input.relation_embedding_model

    def _provided_architecture_options(self) -> dict[str, Any]:
        provided = dict(self.configuration_input.gnn_options)
        for option_id in (
            "gnn_layer_count",
            "gnn_hidden_dimension",
            "node_classifier",
            "dropout",
            "use_edge_mlp",
            "use_reverse_edges",
            "question_aware_classifier",
            "add_layer_normalization",
            "edge_mlp_hidden_dim",
        ):
            value = getattr(self.configuration_input, option_id)
            if value is not None:
                existing = provided.get(option_id)
                if existing is not None and existing != value:
                    raise InvalidGnnArchitectureConfigurationException(
                        f"Conflicting values supplied for GNN option {option_id}."
                    )
                provided[option_id] = value
        return provided

    def _resolve_architecture_option(
        self,
        *,
        option: GnnArchitectureOptionDefinition,
        provided_value: Any,
        architecture_name: str,
    ) -> Any:
        if provided_value is None and not option.prompt_when_missing:
            return option.default
        if not option.choices and option.value_type != "boolean":
            if provided_value is None:
                return option.default
            return provided_value

        invalid_exception = {
            "gnn_layer_count": InvalidGnnLayerCountSelectionException,
            "gnn_hidden_dimension": InvalidGnnHiddenDimensionSelectionException,
            "node_classifier": InvalidNodeClassifierSelectionException,
        }.get(option.option_id, InvalidGnnArchitectureConfigurationException)
        if option.value_type == "boolean":
            choices = {"yes": True, "no": False}
            selected = self.selection_service.resolve_choice(
                provided_value=(
                    "yes" if provided_value is True else "no"
                    if provided_value is False else None
                ),
                options=choices,
                prompt_title=option.display_name,
                prompt_help=f"{option.description} ({architecture_name})",
                recommended_id="yes" if option.default else "no",
                invalid_exception_type=invalid_exception,
                value_getter=lambda value: "yes" if value else "no",
                label_getter=lambda value: "Yes" if value else "No",
            )
            return selected == "yes"

        choices = {str(value): value for value in option.choices}
        selected = self.selection_service.resolve_choice(
            provided_value=str(provided_value) if provided_value is not None else None,
            options=choices,
            prompt_title=option.display_name,
            prompt_help=f"{option.description} ({architecture_name})",
            recommended_id=str(option.default),
            invalid_exception_type=invalid_exception,
            value_getter=str,
            label_getter=lambda value: str(value),
        )
        return choices[selected]
