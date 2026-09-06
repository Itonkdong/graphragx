"""Pure-PyTorch Neural Bellman-Ford answer retriever."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from pipeline.preparation.helpers.configuration_definitions import (
    NBFNET_ARCHITECTURE_ID,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel


class NeuralBellmanFordLayer(nn.Module):
    """One query-dependent DistMult/PNA Bellman-Ford iteration."""

    pna_epsilon = 1e-6

    def __init__(self, hidden_dimension: int, num_relations: int) -> None:
        super().__init__()
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be greater than zero.")
        if num_relations <= 0:
            raise ValueError("num_relations must be greater than zero.")
        self.hidden_dimension = hidden_dimension
        self.num_relations = num_relations
        # This is equivalent to Linear(H, R * H), but the relation-major shape
        # lets forward select only active (graph, relation) parameter blocks.
        self.relation_weight = nn.Parameter(
            torch.empty(num_relations, hidden_dimension, hidden_dimension)
        )
        self.relation_bias = nn.Parameter(
            torch.empty(num_relations, hidden_dimension)
        )
        self.output_projection = nn.Linear(hidden_dimension * 13, hidden_dimension)
        self.layer_norm = nn.LayerNorm(hidden_dimension)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.relation_weight.view(
                self.num_relations * self.hidden_dimension,
                self.hidden_dimension,
            )
        )
        nn.init.zeros_(self.relation_bias)
        self.output_projection.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(
        self,
        node_states: Tensor,
        boundary: Tensor,
        query: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_graph_index: Tensor,
        active_pair_graph_index: Tensor | None = None,
        active_pair_relation_ids: Tensor | None = None,
        edge_pair_index: Tensor | None = None,
        node_degree: Tensor | None = None,
        graph_mean_log_degree: Tensor | None = None,
    ) -> Tensor:
        self._validate_inputs(
            node_states=node_states,
            boundary=boundary,
            query=query,
            edge_index=edge_index,
            edge_type=edge_type,
            node_graph_index=node_graph_index,
        )
        (
            active_pair_graph_index,
            active_pair_relation_ids,
            edge_pair_index,
        ) = self._resolve_active_pairs(
            edge_index=edge_index,
            edge_type=edge_type,
            node_graph_index=node_graph_index,
            active_pair_graph_index=active_pair_graph_index,
            active_pair_relation_ids=active_pair_relation_ids,
            edge_pair_index=edge_pair_index,
        )
        node_degree, graph_mean_log_degree = self._resolve_degree_metadata(
            edge_index=edge_index,
            node_graph_index=node_graph_index,
            graph_count=query.shape[0],
            node_degree=node_degree,
            graph_mean_log_degree=graph_mean_log_degree,
        )

        if active_pair_relation_ids.numel() > 0:
            pair_query = query.index_select(0, active_pair_graph_index)
            pair_weight = self.relation_weight.index_select(
                0, active_pair_relation_ids
            )
            pair_relations = torch.bmm(
                pair_weight,
                pair_query.unsqueeze(-1),
            ).squeeze(-1)
            pair_relations = pair_relations + self.relation_bias.index_select(
                0, active_pair_relation_ids
            )
            source_nodes = edge_index[0].long()
            target_nodes = edge_index[1].long()
            messages = node_states.index_select(0, source_nodes) * pair_relations.index_select(
                0, edge_pair_index
            )
        else:
            target_nodes = edge_index.new_empty(0)
            messages = node_states.new_empty((0, self.hidden_dimension))

        pna = self._pna_aggregate(
            messages=messages,
            target_nodes=target_nodes,
            boundary=boundary,
            node_degree=node_degree,
            node_graph_index=node_graph_index,
            graph_mean_log_degree=graph_mean_log_degree,
        )
        combined = torch.cat(
            (node_states, pna.to(dtype=node_states.dtype)), dim=-1
        )
        transformed = self.activation(
            self.layer_norm(self.output_projection(combined))
        )
        return transformed + node_states

    def _pna_aggregate(
        self,
        *,
        messages: Tensor,
        target_nodes: Tensor,
        boundary: Tensor,
        node_degree: Tensor,
        node_graph_index: Tensor,
        graph_mean_log_degree: Tensor,
    ) -> Tensor:
        """Aggregate incoming edge messages plus the fixed boundary in FP32."""
        boundary_fp32 = boundary.float()
        messages_fp32 = messages.float()
        sum_messages = boundary_fp32.clone()
        squared_sum = boundary_fp32.square()
        maximum = boundary_fp32.clone()
        minimum = boundary_fp32.clone()
        if messages_fp32.numel() > 0:
            sum_messages.index_add_(0, target_nodes, messages_fp32)
            squared_sum.index_add_(0, target_nodes, messages_fp32.square())
            expanded_targets = target_nodes[:, None].expand_as(messages_fp32)
            maximum.scatter_reduce_(
                0,
                expanded_targets,
                messages_fp32,
                reduce="amax",
                include_self=True,
            )
            minimum.scatter_reduce_(
                0,
                expanded_targets,
                messages_fp32,
                reduce="amin",
                include_self=True,
            )

        divisor = node_degree.float().clamp_min(1.0).unsqueeze(-1)
        mean = sum_messages / divisor
        squared_mean = squared_sum / divisor
        standard_deviation = (
            squared_mean - mean.square()
        ).clamp_min(self.pna_epsilon).sqrt()
        statistics = torch.cat((mean, maximum, minimum, standard_deviation), dim=-1)

        log_degree = node_degree.float().clamp_min(1.0).log()
        graph_scale_mean = graph_mean_log_degree.index_select(0, node_graph_index)
        normalized_scale = torch.where(
            graph_scale_mean > self.pna_epsilon,
            log_degree / graph_scale_mean.clamp_min(self.pna_epsilon),
            torch.ones_like(log_degree),
        )
        scalers = torch.stack(
            (
                torch.ones_like(normalized_scale),
                normalized_scale,
                normalized_scale.clamp_min(1e-2).reciprocal(),
            ),
            dim=-1,
        )
        return (statistics.unsqueeze(-1) * scalers[:, None, :]).flatten(1)

    def _resolve_active_pairs(
        self,
        *,
        edge_index: Tensor,
        edge_type: Tensor,
        node_graph_index: Tensor,
        active_pair_graph_index: Tensor | None,
        active_pair_relation_ids: Tensor | None,
        edge_pair_index: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        supplied = (
            active_pair_graph_index,
            active_pair_relation_ids,
            edge_pair_index,
        )
        if all(value is not None for value in supplied):
            assert active_pair_graph_index is not None
            assert active_pair_relation_ids is not None
            assert edge_pair_index is not None
            if (
                active_pair_graph_index.dtype != torch.long
                or active_pair_relation_ids.dtype != torch.long
                or edge_pair_index.dtype != torch.long
            ):
                raise ValueError("Active graph-relation metadata must use torch.long.")
            if active_pair_graph_index.shape != active_pair_relation_ids.shape:
                raise ValueError("Active graph and relation IDs must have matching shapes.")
            if edge_pair_index.shape != edge_type.shape:
                raise ValueError("edge_pair_index must contain one ID per edge.")
            if active_pair_relation_ids.numel() > 0 and (
                torch.any(active_pair_relation_ids < 0)
                or torch.any(active_pair_relation_ids >= self.num_relations)
            ):
                raise ValueError(
                    "Active relation metadata contains an ID outside the vocabulary."
                )
            if active_pair_graph_index.numel() > 0 and (
                torch.any(active_pair_graph_index < 0)
                or torch.any(active_pair_graph_index >= node_graph_index.max() + 1)
            ):
                raise ValueError("Active relation metadata contains an invalid graph ID.")
            if edge_pair_index.numel() > 0 and (
                torch.any(edge_pair_index < 0)
                or torch.any(edge_pair_index >= active_pair_relation_ids.numel())
            ):
                raise ValueError("edge_pair_index contains an invalid active-pair ID.")
            return active_pair_graph_index, active_pair_relation_ids, edge_pair_index
        if any(value is not None for value in supplied):
            raise ValueError("All active graph-relation metadata must be supplied together.")
        if edge_type.numel() == 0:
            return (
                edge_type.new_empty(0),
                edge_type.new_empty(0),
                edge_type.new_empty(0),
            )
        source_graph = node_graph_index.index_select(0, edge_index[0].long())
        target_graph = node_graph_index.index_select(0, edge_index[1].long())
        if not torch.equal(source_graph, target_graph):
            raise ValueError("NBFNet edges cannot connect different batch graphs.")
        pair_keys = source_graph * self.num_relations + edge_type
        active_keys, edge_pair_index = torch.unique(
            pair_keys, sorted=True, return_inverse=True
        )
        return (
            torch.div(active_keys, self.num_relations, rounding_mode="floor"),
            active_keys.remainder(self.num_relations),
            edge_pair_index,
        )

    @staticmethod
    def _resolve_degree_metadata(
        *,
        edge_index: Tensor,
        node_graph_index: Tensor,
        graph_count: int,
        node_degree: Tensor | None,
        graph_mean_log_degree: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if node_degree is None:
            node_degree = torch.ones(
                node_graph_index.shape[0],
                dtype=torch.float32,
                device=node_graph_index.device,
            )
            if edge_index.shape[1] > 0:
                node_degree.index_add_(
                    0,
                    edge_index[1].long(),
                    torch.ones(
                        edge_index.shape[1],
                        dtype=torch.float32,
                        device=edge_index.device,
                    ),
                )
        if graph_mean_log_degree is None:
            log_degree = node_degree.float().clamp_min(1.0).log()
            sums = torch.zeros(
                graph_count, dtype=torch.float32, device=node_graph_index.device
            )
            counts = torch.zeros_like(sums)
            sums.index_add_(0, node_graph_index, log_degree)
            counts.index_add_(0, node_graph_index, torch.ones_like(log_degree))
            graph_mean_log_degree = sums / counts.clamp_min(1.0)
        return node_degree, graph_mean_log_degree

    def _validate_inputs(
        self,
        *,
        node_states: Tensor,
        boundary: Tensor,
        query: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_graph_index: Tensor,
    ) -> None:
        expected = (node_states.shape[0], self.hidden_dimension)
        if node_states.shape != expected or boundary.shape != expected:
            raise ValueError("node_states and boundary must have shape [nodes, hidden].")
        if query.ndim != 2 or query.shape[1] != self.hidden_dimension:
            raise ValueError("query must have shape [graphs, hidden].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges].")
        if edge_type.dtype != torch.long or edge_type.shape != (edge_index.shape[1],):
            raise ValueError("edge_type must contain one torch.long ID per edge.")
        if node_graph_index.dtype != torch.long or node_graph_index.shape != (
            node_states.shape[0],
        ):
            raise ValueError("node_graph_index must contain one graph ID per node.")
        if edge_type.numel() > 0 and (
            torch.any(edge_type < 0) or torch.any(edge_type >= self.num_relations)
        ):
            raise ValueError("edge_type contains an ID outside the relation vocabulary.")


class NBFNetAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Question-conditioned multi-source Neural Bellman-Ford retriever."""

    def __init__(
        self,
        *,
        question_embedding_dimension: int,
        hidden_dimension: int,
        gnn_layer_count: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        if question_embedding_dimension <= 0:
            raise ValueError("question_embedding_dimension must be greater than zero.")
        if gnn_layer_count <= 0:
            raise ValueError("gnn_layer_count must be greater than zero.")
        self.gnn_architecture = NBFNET_ARCHITECTURE_ID
        self.question_embedding_dimension = question_embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = gnn_layer_count
        self.num_relations = num_relations
        self.node_classifier = "mlp"
        self.use_edge_mlp = False
        self.question_aware_classifier = False
        self.use_reverse_edges = True
        self.add_layer_normalization = True
        self.edge_mlp_hidden_dim = None
        self.dropout_value = 0.0

        self.question_projection = nn.Linear(
            question_embedding_dimension, hidden_dimension
        )
        self.layers = nn.ModuleList(
            NeuralBellmanFordLayer(hidden_dimension, num_relations)
            for _ in range(gnn_layer_count)
        )
        scoring_dimension = hidden_dimension * 2
        self.scorer = nn.Sequential(
            nn.Linear(scoring_dimension, scoring_dimension),
            nn.ReLU(),
            nn.Linear(scoring_dimension, 1),
        )

    def forward(
        self,
        *,
        question_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_graph_index: Tensor,
        seed_node_index: Tensor,
        active_pair_graph_index: Tensor | None = None,
        active_pair_relation_ids: Tensor | None = None,
        edge_pair_index: Tensor | None = None,
        node_degree: Tensor | None = None,
        graph_mean_log_degree: Tensor | None = None,
        graph_count: int | None = None,
        **_: Any,
    ) -> Tensor:
        if question_features.ndim == 1:
            question_features = question_features.unsqueeze(0)
        query = self.question_projection(question_features)
        resolved_graph_count = query.shape[0] if graph_count is None else graph_count
        if resolved_graph_count != query.shape[0]:
            raise ValueError("graph_count must match the question batch size.")
        if node_graph_index.numel() == 0:
            return query.new_empty(0)
        if torch.any(node_graph_index < 0) or torch.any(
            node_graph_index >= resolved_graph_count
        ):
            raise ValueError("node_graph_index contains an invalid graph ID.")
        if seed_node_index.numel() == 0:
            raise ValueError("NBFNet requires at least one seed node per batch graph.")
        if seed_node_index.dtype != torch.long or torch.any(seed_node_index < 0) or torch.any(
            seed_node_index >= node_graph_index.shape[0]
        ):
            raise ValueError("seed_node_index contains an invalid node ID.")

        boundary = query.new_zeros(
            (node_graph_index.shape[0], self.hidden_dimension)
        )
        seed_graph_index = node_graph_index.index_select(0, seed_node_index.long())
        if torch.unique(seed_graph_index).numel() != resolved_graph_count:
            raise ValueError("NBFNet requires at least one seed node per batch graph.")
        boundary.index_copy_(
            0,
            seed_node_index.long(),
            query.index_select(0, seed_graph_index),
        )
        states = boundary
        for layer in self.layers:
            states = layer(
                node_states=states,
                boundary=boundary,
                query=query,
                edge_index=edge_index,
                edge_type=edge_type,
                node_graph_index=node_graph_index,
                active_pair_graph_index=active_pair_graph_index,
                active_pair_relation_ids=active_pair_relation_ids,
                edge_pair_index=edge_pair_index,
                node_degree=node_degree,
                graph_mean_log_degree=graph_mean_log_degree,
            )
        node_query = query.index_select(0, node_graph_index)
        return self.scorer(torch.cat((states, node_query), dim=-1)).squeeze(-1)


def build_nbfnet_model(
    *,
    architecture_options: dict[str, Any],
    architecture_context: dict[str, Any] | None = None,
    question_embedding_dimension: int | None,
    **_: Any,
) -> NBFNetAnswerRetriever:
    """Registry callback for NBFNet."""
    context = architecture_context or {}
    relation_type_count = context.get("relation_type_count")
    if not isinstance(relation_type_count, int) or relation_type_count <= 0:
        raise ValueError("NBFNet construction requires a positive relation_type_count.")
    if not isinstance(question_embedding_dimension, int) or question_embedding_dimension <= 0:
        raise ValueError("NBFNet construction requires a question embedding dimension.")
    return NBFNetAnswerRetriever(
        question_embedding_dimension=question_embedding_dimension,
        hidden_dimension=int(architecture_options["gnn_hidden_dimension"]),
        gnn_layer_count=int(architecture_options["gnn_layer_count"]),
        num_relations=relation_type_count,
    )
