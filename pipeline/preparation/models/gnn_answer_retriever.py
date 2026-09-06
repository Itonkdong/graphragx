"""PyTorch modules for answer-node retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn

from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.helpers.configuration_definitions import (
    AA_GRAPH_SAGE_ARCHITECTURE_ID,
    GNN_ARCHITECTURES,
    GRAPH_SAGE_ARCHITECTURE_ID,
)
from pipeline.preparation.helpers.gnn_architecture import import_architecture_callable


def build_node_classifier(
    node_classifier: str,
    hidden_dimension: int,
    question_aware_classifier: bool,
    dropout: float,
) -> nn.Module:
    """Build the shared answer-node classifier used by registered architectures."""
    if question_aware_classifier:
        return nn.Sequential(
            nn.Linear(hidden_dimension * 3, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    if node_classifier == "linear":
        return nn.Linear(hidden_dimension, 1)

    if node_classifier == "mlp":
        layers: list[nn.Module] = [
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dimension, 1))
        return nn.Sequential(*layers)

    raise ValueError(f"Unsupported node classifier: {node_classifier}")


class WeightedMessagePassingLayer(nn.Module):
    """Message-passing layer that weights source-node messages per edge."""

    def __init__(self, hidden_dimension: int):
        super().__init__()
        self.message_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.update_projection = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        source_nodes = edge_index[0].long()
        target_nodes = edge_index[1].long()
        source_messages = self.message_projection(node_features[source_nodes])
        weighted_messages = (
            source_messages * edge_weight.reshape(-1, 1)
        ).to(dtype=node_features.dtype)

        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add_(0, target_nodes, weighted_messages)

        updated_features = torch.cat([node_features, aggregated_messages], dim=1)
        return self.activation(self.update_projection(updated_features))


class GnnAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Weighted GNN plus node-classifier head for answer retrieval."""

    def __init__(
        self,
        entity_embedding_dimension: int,
        hidden_dimension: int,
        gnn_layer_count: int,
        node_classifier: str,
        question_embedding_dimension: int | None = None,
        relation_embedding_dimension: int | None = None,
        use_edge_mlp: bool = False,
        question_aware_classifier: bool = False,
        add_layer_normalization: bool = False,
        edge_mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.entity_embedding_dimension = entity_embedding_dimension
        self.question_embedding_dimension = question_embedding_dimension
        self.relation_embedding_dimension = relation_embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = gnn_layer_count
        self.node_classifier = node_classifier
        self.use_edge_mlp = use_edge_mlp
        self.question_aware_classifier = question_aware_classifier
        self.add_layer_normalization = add_layer_normalization
        self.edge_mlp_hidden_dim = edge_mlp_hidden_dim or hidden_dimension
        self.dropout_value = dropout

        self.entity_projection = nn.Linear(
            entity_embedding_dimension,
            hidden_dimension,
        )
        self.question_projection = (
            nn.Linear(question_embedding_dimension, hidden_dimension)
            if question_embedding_dimension is not None
            and (use_edge_mlp or question_aware_classifier)
            else None
        )
        self.relation_projection = (
            nn.Linear(relation_embedding_dimension, hidden_dimension)
            if relation_embedding_dimension is not None and use_edge_mlp
            else None
        )
        self.edge_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dimension * 3, self.edge_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.edge_mlp_hidden_dim, 1),
            )
            if use_edge_mlp
            else None
        )
        self.gnn_layers = nn.ModuleList(
            WeightedMessagePassingLayer(hidden_dimension)
            for _ in range(gnn_layer_count)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dimension)
            for _ in range(gnn_layer_count)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.classifier = self._build_classifier(
            node_classifier=node_classifier,
            hidden_dimension=hidden_dimension,
            question_aware_classifier=question_aware_classifier,
            dropout=dropout,
        )

    def forward(
        self,
        entity_features: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        question_features: Tensor | None = None,
        relation_features: Tensor | None = None,
        edge_type: Tensor | None = None,
        edge_norm: Tensor | None = None,
        active_relation_ids: Tensor | None = None,
        edge_relation_index: Tensor | None = None,
        active_relation_offsets: Tensor | None = None,
    ) -> Tensor:
        node_features = self.entity_projection(entity_features)
        projected_question = self._project_question(question_features)
        resolved_edge_weight = self._resolve_edge_weight(
            edge_weight=edge_weight,
            question_features=question_features,
            relation_features=relation_features,
            projected_question=projected_question,
        )
        for layer_index, layer in enumerate(self.gnn_layers):
            updated_features = layer(node_features, edge_index, resolved_edge_weight)
            if self.add_layer_normalization:
                node_features = self.layer_norms[layer_index](
                    node_features + self.dropout(updated_features)
                )
                node_features = self.activation(node_features)
            else:
                node_features = updated_features

        if self.question_aware_classifier:
            if projected_question is None:
                raise ValueError(
                    "question_features are required for question-aware classification."
                )
            question_features_for_nodes = projected_question.reshape(1, -1).expand(
                node_features.shape[0],
                -1,
            )
            node_features = torch.cat(
                [
                    node_features,
                    question_features_for_nodes,
                    node_features * question_features_for_nodes,
                ],
                dim=1,
            )

        return self.classifier(node_features).squeeze(-1)

    @staticmethod
    def _build_classifier(
        node_classifier: str,
        hidden_dimension: int,
        question_aware_classifier: bool,
        dropout: float,
    ) -> nn.Module:
        return build_node_classifier(
            node_classifier=node_classifier,
            hidden_dimension=hidden_dimension,
            question_aware_classifier=question_aware_classifier,
            dropout=dropout,
        )

    def _project_question(self, question_features: Tensor | None) -> Tensor | None:
        if self.question_projection is None:
            return None
        if question_features is None:
            return None
        return self.question_projection(question_features)

    def _resolve_edge_weight(
        self,
        edge_weight: Tensor | None,
        question_features: Tensor | None,
        relation_features: Tensor | None,
        projected_question: Tensor | None,
    ) -> Tensor:
        if self.use_edge_mlp:
            if (
                self.edge_mlp is None
                or self.relation_projection is None
                or projected_question is None
                or relation_features is None
            ):
                raise ValueError(
                    "question_features and relation_features are required when "
                    "use_edge_mlp is enabled."
                )
            relation_projection = self.relation_projection(relation_features)
            question_projection = projected_question.reshape(1, -1).expand(
                relation_projection.shape[0],
                -1,
            )
            edge_input = torch.cat(
                [
                    question_projection,
                    relation_projection,
                    question_projection * relation_projection,
                ],
                dim=1,
            )
            return torch.sigmoid(self.edge_mlp(edge_input)).squeeze(-1)

        if edge_weight is not None:
            return edge_weight

        if question_features is None or relation_features is None:
            raise ValueError(
                "edge_weight or question_features plus relation_features are required."
            )
        return torch_functional.cosine_similarity(
            question_features.reshape(1, -1),
            relation_features,
            dim=1,
        )


class GraphSageAnswerRetriever(GnnAnswerRetriever):
    """Concrete baseline GraphSAGE retriever."""

    def __init__(self, **kwargs):
        kwargs.update(
            use_edge_mlp=False,
            question_aware_classifier=False,
            add_layer_normalization=False,
            edge_mlp_hidden_dim=None,
        )
        super().__init__(**kwargs)
        self.gnn_architecture = GRAPH_SAGE_ARCHITECTURE_ID


class AdvancedGraphSageAnswerRetriever(GnnAnswerRetriever):
    """Concrete advanced answer-aware GraphSAGE retriever."""

    def __init__(self, **kwargs):
        if kwargs.get("node_classifier") == "linear" and kwargs.get(
            "question_aware_classifier", False
        ):
            raise ValueError(
                "Advance GraphSAGE linear classification requires question-aware "
                "classification to be disabled."
            )
        super().__init__(**kwargs)
        self.gnn_architecture = AA_GRAPH_SAGE_ARCHITECTURE_ID


def build_gnn_answer_retriever(
    gnn_architecture: str,
    architecture_options: dict | None = None,
    **kwargs,
) -> AnswerRetrieverModel:
    """Build any registered retriever through its lazy registry callback."""
    definition = GNN_ARCHITECTURES.get(gnn_architecture)
    if definition is None:
        raise ValueError(f"Unsupported GNN architecture: {gnn_architecture}")
    builder = import_architecture_callable(definition.model_builder_path)
    return builder(architecture_options=architecture_options or {}, **kwargs)


def _shared_model_kwargs(architecture_options: dict, kwargs: dict) -> dict:
    resolved = dict(kwargs)
    # Architecture context is consumed by architectures such as R-GCN. Keep
    # the legacy GraphSAGE constructors byte-for-byte compatible with their
    # existing argument contract.
    resolved.pop("architecture_context", None)
    option_to_argument = {
        "gnn_hidden_dimension": "hidden_dimension",
        "gnn_layer_count": "gnn_layer_count",
        "node_classifier": "node_classifier",
        "dropout": "dropout",
    }
    for option_id, argument_name in option_to_argument.items():
        if option_id in architecture_options:
            resolved[argument_name] = architecture_options[option_id]
    return resolved


def build_graphsage_model(
    *, architecture_options: dict, **kwargs
) -> GraphSageAnswerRetriever:
    """Registry callback for baseline GraphSAGE."""
    return GraphSageAnswerRetriever(
        **_shared_model_kwargs(architecture_options, kwargs)
    )


def build_aa_graphsage_model(
    *, architecture_options: dict, **kwargs
) -> AdvancedGraphSageAnswerRetriever:
    """Registry callback for Advance GraphSAGE."""
    resolved = _shared_model_kwargs(architecture_options, kwargs)
    for option_id in (
        "use_edge_mlp",
        "question_aware_classifier",
        "add_layer_normalization",
        "edge_mlp_hidden_dim",
    ):
        if option_id in architecture_options:
            resolved[option_id] = architecture_options[option_id]
    return AdvancedGraphSageAnswerRetriever(**resolved)
