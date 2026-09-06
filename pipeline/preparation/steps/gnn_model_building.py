"""GNN answer-retriever construction step."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    HGT_ARCHITECTURE_ID,
    NBFNET_ARCHITECTURE_ID,
    OPENAI_EMBEDDING_MODELS,
    REAREV_ARCHITECTURE_ID,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

logger = get_logger(__name__)


class BuiltGnnAnswerRetriever(StepResult):
    """Constructed PyTorch answer-retriever architecture artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Selected dataset identifier.")
    gnn_architecture: str = Field(default="graphsage")
    gnn_architecture_options: dict[str, Any] = Field(default_factory=dict)
    gnn_architecture_context: dict[str, Any] = Field(default_factory=dict)
    relation_vocabulary: dict[str, int] | None = Field(default=None)
    entity_embedding_model: str | None = Field(
        default=None,
        description="OpenAI embedding model used for entity text.",
    )
    entity_embedding_dimension: int | None = Field(
        default=None,
        description="Entity text embedding dimension before projection.",
    )
    hidden_dimension: int | None = Field(
        default=None,
        description="Shared hidden dimension used by projection, GNN, and classifier.",
    )
    gnn_layer_count: int | None = Field(default=None, description="Number of weighted GNN layers.")
    node_classifier: str | None = Field(default=None, description="Node classifier architecture id.")
    question_embedding_dimension: int | None = Field(
        default=None,
        description="Question text embedding dimension before projection.",
    )
    relation_embedding_dimension: int | None = Field(
        default=None,
        description="Relation text embedding dimension before projection.",
    )
    use_edge_mlp: bool = Field(default=False)
    question_aware_classifier: bool = Field(default=False)
    use_reverse_edges: bool = Field(default=False)
    add_layer_normalization: bool = Field(default=False)
    edge_mlp_hidden_dim: int | None = Field(default=None, description="Hidden dimension for edge MLP.")
    dropout: float = Field(default=0.1)
    model: AnswerRetrieverModel = Field(
        ...,
        description="Constructed PyTorch answer-retriever model.",
    )


class BuildGnnAnswerRetrieverContext(StepContext[PreparedWebQSPGraphDataset]):
    """Specialized context for constructing the GNN answer retriever."""

    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration required to build the retriever.",
    )


class BuildGnnAnswerRetrieverStep(
    AbstractStep[BuiltGnnAnswerRetriever, PreparedWebQSPGraphDataset]
):
    """Build the PyTorch GNN retriever and node classifier."""

    def __init__(
        self,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)

    def execute_default(
        self,
        context: BuildGnnAnswerRetrieverContext,
    ) -> BuiltGnnAnswerRetriever:
        prepared_dataset = context.result
        if prepared_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever building requires a prepared WebQSP dataset."
            )

        configuration = context.pipeline_configuration
        from pipeline.preparation.models.gnn_answer_retriever import build_gnn_answer_retriever

        embedding_model = configuration.embedding_model or configuration.entity_embedding_model
        architecture_options = dict(configuration.gnn_architecture_options)
        architecture = GNN_ARCHITECTURES[configuration.gnn_architecture]
        embedding_definition = (
            OPENAI_EMBEDDING_MODELS[embedding_model]
            if any(
                (
                    architecture.data_requirements.uses_entity_embeddings,
                    architecture.data_requirements.uses_question_embeddings,
                    architecture.data_requirements.uses_relation_embeddings,
                )
            )
            and embedding_model is not None
            else None
        )
        supported_option_ids = set(
            architecture.option_map
        )
        for option_id, value in {
            "gnn_layer_count": configuration.gnn_layer_count,
            "gnn_hidden_dimension": configuration.gnn_hidden_dimension,
            "node_classifier": configuration.node_classifier,
            "dropout": configuration.dropout,
            "use_edge_mlp": configuration.use_edge_mlp,
            "question_aware_classifier": configuration.question_aware_classifier,
            "use_reverse_edges": configuration.use_reverse_edges,
            "add_layer_normalization": configuration.add_layer_normalization,
            "edge_mlp_hidden_dim": configuration.edge_mlp_hidden_dim,
        }.items():
            if option_id in supported_option_ids and value is not None:
                architecture_options.setdefault(option_id, value)

        architecture_context: dict[str, Any] = {}
        relation_vocabulary: dict[str, int] | None = None
        if architecture.data_requirements.uses_relation_types:
            from pipeline.preparation.services.gnn_relation_vocabulary import (
                build_relation_architecture_context,
                validate_relation_vocabulary,
            )

            if not prepared_dataset.use_reverse_edges:
                raise InvalidInteractiveConfigurationInputException(
                    f"Architecture {configuration.gnn_architecture} requires reverse edges."
                )
            relation_vocabulary = dict(prepared_dataset.vocabulary_store.relations)
            validate_relation_vocabulary(relation_vocabulary)
            architecture_context = build_relation_architecture_context(
                relation_vocabulary
            )
        if configuration.gnn_architecture == REAREV_ARCHITECTURE_ID:
            from pipeline.preparation.helpers.rearev_constants import (
                REAREV_ENCODER_MODEL_ID,
                REAREV_ENCODER_REVISION,
                REAREV_ENCODER_WIDTH,
                REAREV_QUESTION_MAX_LENGTH,
                REAREV_RELATION_MAX_LENGTH,
                REAREV_RELATION_TEXT_SCHEMA_VERSION,
            )

            architecture_context.update(
                {
                    "rearev_preprocessing_version": 2,
                    "encoder_model_id": REAREV_ENCODER_MODEL_ID,
                    "encoder_revision": REAREV_ENCODER_REVISION,
                    "encoder_width": REAREV_ENCODER_WIDTH,
                    "question_max_length": REAREV_QUESTION_MAX_LENGTH,
                    "relation_max_length": REAREV_RELATION_MAX_LENGTH,
                    "relation_text_schema_version": REAREV_RELATION_TEXT_SCHEMA_VERSION,
                    "encoder_frozen": True,
                    "seed_feedback_aggregation": "sum",
                    "instruction_revision_schema": "four-way-gru-update-v2",
                }
            )
        elif configuration.gnn_architecture == NBFNET_ARCHITECTURE_ID:
            from pipeline.preparation.helpers.nbfnet_constants import (
                NBFNET_FIXED_CONTEXT,
            )

            architecture_context.update(NBFNET_FIXED_CONTEXT)

        logger.info(
            f"Building GNN answer retriever: dataset={prepared_dataset.dataset_id} "
            f"gnn_architecture={configuration.gnn_architecture} "
            f"embedding_model={embedding_model} "
            f"embedding_dimension={getattr(embedding_definition, 'dimensions', 'n/a')} "
            f"hidden_dimension={configuration.gnn_hidden_dimension} "
            f"gnn_layers={configuration.gnn_layer_count} "
            f"node_classifier={configuration.node_classifier} "
            f"use_edge_mlp={configuration.use_edge_mlp} "
            f"question_aware_classifier={configuration.question_aware_classifier} "
            f"add_layer_normalization={configuration.add_layer_normalization} "
            f"dropout={configuration.dropout}"
        )
        model = build_gnn_answer_retriever(
            gnn_architecture=configuration.gnn_architecture,
            architecture_options=architecture_options,
            architecture_context=architecture_context,
            entity_embedding_dimension=(embedding_definition.dimensions if embedding_definition else None),
            question_embedding_dimension=(embedding_definition.dimensions if embedding_definition else None),
            relation_embedding_dimension=(embedding_definition.dimensions if embedding_definition else None),
        )

        if configuration.gnn_architecture in {
            HGT_ARCHITECTURE_ID,
            NBFNET_ARCHITECTURE_ID,
        }:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            estimated_training_bytes = parameter_count * 16
            logger.info(
                f"Built {configuration.gnn_architecture} parameterization: "
                f"relations={architecture_context.get('relation_type_count')} "
                f"attention_heads={architecture_options.get('attention_heads', 'n/a')} "
                f"parameters={parameter_count} "
                f"estimated_parameters_gradients_adam_gib="
                f"{estimated_training_bytes / 1024**3:.2f}"
            )

        logger.info(f"Built GNN answer retriever architecture")
        return BuiltGnnAnswerRetriever(
            dataset_id=prepared_dataset.dataset_id,
            gnn_architecture=configuration.gnn_architecture,
            gnn_architecture_options=architecture_options,
            gnn_architecture_context=architecture_context,
            relation_vocabulary=relation_vocabulary,
            entity_embedding_model=(
                embedding_model if architecture.data_requirements.uses_entity_embeddings else None
            ),
            entity_embedding_dimension=(
                embedding_definition.dimensions
                if embedding_definition and architecture.data_requirements.uses_entity_embeddings
                else None
            ),
            question_embedding_dimension=(
                embedding_definition.dimensions
                if embedding_definition and architecture.data_requirements.uses_question_embeddings
                else None
            ),
            relation_embedding_dimension=(
                embedding_definition.dimensions
                if embedding_definition and architecture.data_requirements.uses_relation_embeddings
                else None
            ),
            hidden_dimension=getattr(model, "hidden_dimension", None),
            gnn_layer_count=getattr(model, "gnn_layer_count", None),
            node_classifier=getattr(model, "node_classifier", None),
            use_edge_mlp=configuration.use_edge_mlp,
            question_aware_classifier=getattr(model, "question_aware_classifier", False),
            use_reverse_edges=prepared_dataset.use_reverse_edges,
            add_layer_normalization=getattr(model, "add_layer_normalization", False),
            edge_mlp_hidden_dim=getattr(model, "edge_mlp_hidden_dim", None),
            dropout=getattr(model, "dropout_value", 0.0),
            model=model,
        )
