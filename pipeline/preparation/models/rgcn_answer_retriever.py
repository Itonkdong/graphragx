"""Basis-decomposed R-GCN answer retriever."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from pipeline.preparation.helpers.configuration_definitions import RGCN_ARCHITECTURE_ID
from pipeline.preparation.models.gnn_answer_retriever import build_node_classifier
from pipeline.preparation.models.interfaces import AnswerRetrieverModel


class ActiveRelationBasisRGCNLayer(nn.Module):
    """R-GCN layer that evaluates only relations present in the current graph."""

    def __init__(
        self,
        hidden_dimension: int,
        num_relations: int,
        num_bases: int,
        message_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be greater than zero.")
        if num_relations <= 0:
            raise ValueError("num_relations must be greater than zero.")
        if num_bases <= 0:
            raise ValueError("num_bases must be greater than zero.")
        if message_chunk_size <= 0:
            raise ValueError("message_chunk_size must be greater than zero.")

        self.hidden_dimension = hidden_dimension
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.message_chunk_size = message_chunk_size
        self.basis_weights = nn.Parameter(
            torch.empty(num_bases, hidden_dimension, hidden_dimension)
        )
        self.relation_coefficients = nn.Parameter(
            torch.empty(num_relations, num_bases)
        )
        self.root_weight = nn.Parameter(
            torch.empty(hidden_dimension, hidden_dimension)
        )
        self.bias = nn.Parameter(torch.empty(hidden_dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_weights)
        nn.init.xavier_uniform_(self.relation_coefficients)
        nn.init.xavier_uniform_(self.root_weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_norm: Tensor | None = None,
        active_relation_ids: Tensor | None = None,
        edge_relation_index: Tensor | None = None,
    ) -> Tensor:
        self._validate_inputs(node_features, edge_index, edge_type)
        root_output = node_features @ self.root_weight
        output = root_output.new_zeros(
            (node_features.shape[0], self.hidden_dimension)
        )
        source_nodes = edge_index[0].long()
        target_nodes = edge_index[1].long()

        if edge_type.numel() > 0:
            (
                edge_norm,
                active_relation_ids,
                edge_relation_index,
            ) = self._resolve_aggregation_metadata(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_norm=edge_norm,
                active_relation_ids=active_relation_ids,
                edge_relation_index=edge_relation_index,
                node_count=node_features.shape[0],
            )
            active_coefficients = self.relation_coefficients.index_select(
                0, active_relation_ids
            )
            active_weights = torch.einsum(
                "rb,bio->rio",
                active_coefficients,
                self.basis_weights,
            )
            edge_count = edge_type.shape[0]
            for start in range(0, edge_count, self.message_chunk_size):
                end = min(start + self.message_chunk_size, edge_count)
                relation_weights = active_weights.index_select(
                    0, edge_relation_index[start:end]
                )
                source_features = node_features.index_select(
                    0, source_nodes[start:end]
                )
                messages = torch.bmm(
                    source_features.unsqueeze(1),
                    relation_weights,
                ).squeeze(1)
                messages = messages * edge_norm[start:end].to(
                    dtype=messages.dtype
                ).unsqueeze(1)
                output.index_add_(0, target_nodes[start:end], messages)

        return output + root_output + self.bias.to(dtype=output.dtype)

    @staticmethod
    def _resolve_aggregation_metadata(
        *,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_norm: Tensor | None,
        active_relation_ids: Tensor | None,
        edge_relation_index: Tensor | None,
        node_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use prepared metadata or derive a compatible fallback on device."""
        if (
            edge_norm is not None
            and active_relation_ids is not None
            and edge_relation_index is not None
        ):
            if edge_norm.shape != edge_type.shape:
                raise ValueError("edge_norm must contain one value per edge.")
            if edge_relation_index.shape != edge_type.shape:
                raise ValueError(
                    "edge_relation_index must contain one value per edge."
                )
            return edge_norm, active_relation_ids, edge_relation_index

        active_relation_ids, edge_relation_index = torch.unique(
            edge_type,
            sorted=True,
            return_inverse=True,
        )
        relation_target_keys = edge_type * node_count + edge_index[1]
        _, normalization_index, counts = torch.unique(
            relation_target_keys,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        edge_norm = counts.index_select(0, normalization_index).float().reciprocal()
        return edge_norm, active_relation_ids, edge_relation_index

    def _validate_inputs(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> None:
        if node_features.ndim != 2 or node_features.shape[1] != self.hidden_dimension:
            raise ValueError(
                "node_features must have shape [num_nodes, hidden_dimension]."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must contain one relation id per graph edge.")
        if edge_type.dtype != torch.long:
            raise ValueError("edge_type must use torch.long dtype.")
        if edge_type.numel() > 0 and (
            torch.any(edge_type < 0) or torch.any(edge_type >= self.num_relations)
        ):
            raise ValueError("edge_type contains a relation id outside the model vocabulary.")


class RGCNAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Entity encoder with relational message passing and a fixed MLP head."""

    def __init__(
        self,
        entity_embedding_dimension: int,
        hidden_dimension: int,
        gnn_layer_count: int,
        num_relations: int,
        num_bases: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gnn_architecture = RGCN_ARCHITECTURE_ID
        self.entity_embedding_dimension = entity_embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = gnn_layer_count
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout_value = dropout
        self.node_classifier = "mlp"
        self.use_edge_mlp = False
        self.question_aware_classifier = False
        self.add_layer_normalization = False
        self.edge_mlp_hidden_dim = None

        self.entity_projection = nn.Linear(
            entity_embedding_dimension,
            hidden_dimension,
        )
        self.gnn_layers = nn.ModuleList(
            ActiveRelationBasisRGCNLayer(
                hidden_dimension=hidden_dimension,
                num_relations=num_relations,
                num_bases=num_bases,
            )
            for _ in range(gnn_layer_count)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = build_node_classifier(
            node_classifier="mlp",
            hidden_dimension=hidden_dimension,
            question_aware_classifier=False,
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
        if edge_type is None:
            raise ValueError("edge_type is required for R-GCN message passing.")
        node_features = self.entity_projection(entity_features)
        for layer_index, layer in enumerate(self.gnn_layers):
            node_features = self.activation(
                layer(
                    node_features,
                    edge_index,
                    edge_type,
                    edge_norm=edge_norm,
                    active_relation_ids=active_relation_ids,
                    edge_relation_index=edge_relation_index,
                )
            )
            if layer_index + 1 < len(self.gnn_layers):
                node_features = self.dropout(node_features)
        return self.classifier(node_features).squeeze(-1)


def build_rgcn_model(
    *,
    architecture_options: dict[str, Any],
    architecture_context: dict[str, Any] | None = None,
    entity_embedding_dimension: int,
    **_: Any,
) -> RGCNAnswerRetriever:
    """Registry callback for the R-GCN architecture."""
    context = architecture_context or {}
    relation_type_count = context.get("relation_type_count")
    if not isinstance(relation_type_count, int) or relation_type_count <= 0:
        raise ValueError(
            "R-GCN construction requires a positive relation_type_count."
        )
    return RGCNAnswerRetriever(
        entity_embedding_dimension=entity_embedding_dimension,
        hidden_dimension=int(architecture_options["gnn_hidden_dimension"]),
        gnn_layer_count=int(architecture_options["gnn_layer_count"]),
        num_relations=relation_type_count,
        num_bases=int(architecture_options["num_bases"]),
        dropout=float(architecture_options["dropout"]),
    )
