"""Manual ReaRev answer retriever with a frozen token-level MiniLM encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pipeline.preparation.helpers.configuration_definitions import (
    REAREV_ARCHITECTURE_ID,
)
from pipeline.preparation.helpers.rearev_constants import (
    REAREV_ENCODER_MODEL_ID,
    REAREV_ENCODER_REVISION,
    REAREV_ENCODER_WIDTH,
    REAREV_QUESTION_MAX_LENGTH,
    REAREV_RELATION_MAX_LENGTH,
    REAREV_RELATION_TEXT_SCHEMA_VERSION,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel


def graph_softmax(scores: Tensor, graph_index: Tensor, graph_count: int) -> Tensor:
    """Compute stable FP32 softmax independently for every disconnected graph."""
    scores_float = scores.float()
    if scores_float.ndim != 1 or graph_index.shape != scores_float.shape:
        raise ValueError("scores and graph_index must be aligned one-dimensional tensors.")
    if graph_count <= 0:
        raise ValueError("graph_count must be greater than zero.")
    maxima = torch.full(
        (graph_count,),
        -torch.inf,
        dtype=torch.float32,
        device=scores.device,
    )
    if scores.numel() > 0:
        maxima.scatter_reduce_(0, graph_index, scores_float, reduce="amax", include_self=True)
    shifted = scores_float - maxima.index_select(0, graph_index)
    exponentials = shifted.exp()
    denominators = torch.zeros(
        graph_count,
        dtype=torch.float32,
        device=scores.device,
    )
    denominators.index_add_(0, graph_index, exponentials)
    return exponentials / denominators.index_select(0, graph_index).clamp_min(1e-12)


class ReaRevQueryReform(nn.Module):
    """Paper-aligned four-way query interaction with a GRU-style update gate."""

    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.candidate_projection = nn.Linear(hidden_dimension * 4, hidden_dimension)
        self.gate_input_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.gate_hidden_projection = nn.Linear(
            hidden_dimension, hidden_dimension, bias=False
        )

    def forward(self, instruction: Tensor, seed_state: Tensor) -> Tensor:
        interaction = torch.cat(
            (
                instruction,
                seed_state,
                instruction - seed_state,
                instruction * seed_state,
            ),
            dim=-1,
        )
        candidate = self.candidate_projection(interaction)
        update_gate = torch.sigmoid(
            self.gate_input_projection(seed_state)
            + self.gate_hidden_projection(instruction)
        )
        return (1.0 - update_gate) * instruction + update_gate * candidate


class ReaRevAnswerRetriever(nn.Module, AnswerRetrieverModel):
    """Question-conditioned adaptive reason-and-revise graph executor."""

    def __init__(
        self,
        *,
        hidden_dimension: int,
        num_instructions: int,
        reasoning_steps: int,
        adaptive_iterations: int,
        dropout: float,
        text_encoder: nn.Module,
        encoder_width: int = REAREV_ENCODER_WIDTH,
    ) -> None:
        super().__init__()
        if min(hidden_dimension, num_instructions, reasoning_steps, adaptive_iterations) <= 0:
            raise ValueError("ReaRev dimensions and iteration counts must be positive.")
        self.gnn_architecture = REAREV_ARCHITECTURE_ID
        self.hidden_dimension = hidden_dimension
        self.gnn_layer_count = None
        self.node_classifier = None
        self.num_instructions = num_instructions
        self.reasoning_steps = reasoning_steps
        self.adaptive_iterations = adaptive_iterations
        self.encoder_width = encoder_width
        self.dropout_value = dropout
        self.use_edge_mlp = False
        self.question_aware_classifier = False
        self.add_layer_normalization = False
        self.edge_mlp_hidden_dim = None
        self.entity_embedding_dimension = None
        self.question_embedding_dimension = encoder_width
        self.relation_embedding_dimension = encoder_width

        self.text_encoder = text_encoder
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.text_encoder.eval()

        self.question_projection = nn.Linear(encoder_width, hidden_dimension)
        self.relation_projection = nn.Linear(encoder_width, hidden_dimension)
        self.question_state_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.instruction_state_layers = nn.ModuleList(
            nn.Linear(hidden_dimension, hidden_dimension)
            for _ in range(num_instructions)
        )
        self.instruction_attention_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dimension * 4, hidden_dimension),
                nn.Tanh(),
                nn.Linear(hidden_dimension, 1),
            )
            for _ in range(num_instructions)
        )
        self.relation_attention = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, 1),
        )
        self.relation_message_layers = nn.ModuleList(
            nn.Linear(hidden_dimension, hidden_dimension)
            for _ in range(reasoning_steps)
        )
        self.node_update_layers = nn.ModuleList(
            nn.Linear(hidden_dimension * (num_instructions + 1), hidden_dimension)
            for _ in range(reasoning_steps)
        )
        self.node_score_layers = nn.ModuleList(
            nn.Linear(hidden_dimension, 1) for _ in range(reasoning_steps)
        )
        self.query_reform_layers = nn.ModuleList(
            ReaRevQueryReform(hidden_dimension)
            for _ in range(num_instructions)
        )
        self.dropout = nn.Dropout(dropout)

    def train(self, mode: bool = True):
        """Keep the frozen external encoder deterministic in every model mode."""
        super().train(mode)
        self.text_encoder.eval()
        return self

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Return checkpoint state without the external MiniLM snapshot."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("text_encoder.")
        }

    def load_trainable_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load trainable ReaRev state while retaining the reconstructed encoder."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("text_encoder.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Invalid ReaRev checkpoint; missing={missing}, unexpected={unexpected}."
            )

    def _encode_tokens(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.shape[0] == 0:
            return torch.empty(
                (0, input_ids.shape[1], self.encoder_width),
                dtype=self.question_projection.weight.dtype,
                device=input_ids.device,
            )
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return output.last_hidden_state

    def _decode_instructions(
        self,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> list[Tensor]:
        tokens = self.question_projection(question_tokens)
        state = self.question_state_projection(tokens[:, 0])
        instructions: list[Tensor] = []
        for state_layer, attention_layer in zip(
            self.instruction_state_layers,
            self.instruction_attention_layers,
            strict=True,
        ):
            state = F.relu(state_layer(state))
            expanded = state.unsqueeze(1).expand_as(tokens)
            attention_input = torch.cat(
                (tokens, expanded, tokens * expanded, tokens - expanded),
                dim=-1,
            )
            attention_scores = attention_layer(attention_input).squeeze(-1).float()
            attention_scores = attention_scores.masked_fill(~question_mask.bool(), -torch.inf)
            attention = torch.softmax(attention_scores, dim=1).to(tokens.dtype)
            state = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
            instructions.append(state)
        return instructions

    def _pool_relations(self, relation_tokens: Tensor, relation_mask: Tensor) -> Tensor:
        tokens = self.relation_projection(relation_tokens)
        scores = self.relation_attention(tokens).squeeze(-1).float()
        scores = scores.masked_fill(~relation_mask.bool(), -torch.inf)
        attention = torch.softmax(scores, dim=1).to(tokens.dtype)
        return torch.sum(tokens * attention.unsqueeze(-1), dim=1)

    def _initialize_nodes(
        self,
        relation_features: Tensor,
        initialization_edge_index: Tensor,
        initialization_relation_index: Tensor,
        node_count: int,
    ) -> Tensor:
        node_states = relation_features.new_zeros((node_count, self.hidden_dimension))
        counts = relation_features.new_zeros((node_count, 1))
        if initialization_relation_index.numel() == 0:
            return node_states
        edge_relations = relation_features.index_select(0, initialization_relation_index)
        ones = counts.new_ones((edge_relations.shape[0], 1))
        for endpoints in (initialization_edge_index[0], initialization_edge_index[1]):
            node_states.index_add_(0, endpoints, edge_relations)
            counts.index_add_(0, endpoints, ones)
        return node_states / counts.clamp_min(1.0)

    def _reason_step(
        self,
        *,
        node_states: Tensor,
        current_distribution: Tensor,
        instructions: list[Tensor],
        relation_features: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        node_graph_index: Tensor,
        step_index: int,
        graph_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        source, target = edge_index.long()
        relation_messages = F.relu(
            self.relation_message_layers[step_index](relation_features)
        )
        aggregated_per_instruction: list[Tensor] = []
        edge_graph_index = node_graph_index.index_select(0, source)
        for instruction in instructions:
            edge_instruction = instruction.index_select(0, edge_graph_index)
            edge_relation = relation_messages.index_select(0, edge_relation_index)
            messages = F.relu(edge_relation * edge_instruction)
            messages = messages * current_distribution.index_select(0, source).to(
                messages.dtype
            ).unsqueeze(-1)
            aggregated = node_states.new_zeros(node_states.shape)
            if target.numel() > 0:
                aggregated.index_add_(0, target, messages.to(node_states.dtype))
            aggregated_per_instruction.append(aggregated)
        update_input = torch.cat([node_states, *aggregated_per_instruction], dim=-1)
        node_states = F.relu(self.node_update_layers[step_index](update_input))
        node_states = self.dropout(node_states)
        scores = self.node_score_layers[step_index](node_states).squeeze(-1)
        probabilities = graph_softmax(scores, node_graph_index, graph_count)
        return node_states, scores, probabilities

    def _revise_instructions(
        self,
        instructions: list[Tensor],
        node_states: Tensor,
        seed_mask: Tensor,
        node_graph_index: Tensor,
        graph_count: int,
    ) -> list[Tensor]:
        kg_state = node_states.new_zeros((graph_count, self.hidden_dimension))
        if seed_mask.ndim != 1 or seed_mask.shape[0] != node_states.shape[0]:
            raise ValueError("seed_mask must be one-dimensional and aligned with nodes.")
        kg_state.index_add_(
            0,
            node_graph_index,
            node_states * seed_mask.to(node_states.dtype).unsqueeze(-1),
        )
        return [
            reform(instruction, kg_state)
            for instruction, reform in zip(
                instructions,
                self.query_reform_layers,
                strict=True,
            )
        ]

    def forward(
        self,
        *,
        question_input_ids: Tensor,
        question_attention_mask: Tensor,
        relation_input_ids: Tensor,
        relation_attention_mask: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        initialization_edge_index: Tensor,
        initialization_relation_index: Tensor,
        node_graph_index: Tensor,
        seed_distribution: Tensor,
        seed_mask: Tensor,
        graph_count: int,
        **_: Any,
    ) -> Tensor:
        question_tokens = self._encode_tokens(
            question_input_ids, question_attention_mask
        )
        relation_tokens = self._encode_tokens(
            relation_input_ids, relation_attention_mask
        )
        instructions = self._decode_instructions(
            question_tokens, question_attention_mask
        )
        relation_features = self._pool_relations(
            relation_tokens, relation_attention_mask
        )
        node_states = self._initialize_nodes(
            relation_features,
            initialization_edge_index,
            initialization_relation_index,
            node_graph_index.shape[0],
        )
        scores = node_states.new_zeros(node_states.shape[0])
        for iteration in range(self.adaptive_iterations):
            current_distribution = seed_distribution.float()
            for step_index in range(self.reasoning_steps):
                node_states, scores, current_distribution = self._reason_step(
                    node_states=node_states,
                    current_distribution=current_distribution,
                    instructions=instructions,
                    relation_features=relation_features,
                    edge_index=edge_index,
                    edge_relation_index=edge_relation_index,
                    node_graph_index=node_graph_index,
                    step_index=step_index,
                    graph_count=graph_count,
                )
            if iteration + 1 < self.adaptive_iterations:
                instructions = self._revise_instructions(
                    instructions,
                    node_states,
                    seed_mask,
                    node_graph_index,
                    graph_count,
                )
        return scores


def _load_pinned_encoder() -> nn.Module:
    try:
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            REAREV_ENCODER_MODEL_ID,
            revision=REAREV_ENCODER_REVISION,
        )
    except Exception as error:  # transformers raises several cache/network errors
        raise RuntimeError(
            "ReaRev requires the pinned MiniLM encoder "
            f"{REAREV_ENCODER_MODEL_ID}@{REAREV_ENCODER_REVISION}. The encoder is "
            "not available locally and could not be downloaded."
        ) from error


def build_rearev_model(
    *,
    architecture_options: dict[str, Any],
    architecture_context: dict[str, Any] | None = None,
    text_encoder: nn.Module | None = None,
    **_: Any,
) -> ReaRevAnswerRetriever:
    """Registry callback for ReaRev."""
    context = architecture_context or {}
    expected_context = {
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
    for key, expected_value in expected_context.items():
        if key not in context:
            raise ValueError(f"ReaRev construction requires architecture context key {key}.")
        actual_value = context[key]
        if actual_value != expected_value:
            raise ValueError(
                f"Saved ReaRev {key}={actual_value!r} does not match the supported "
                f"value {expected_value!r}."
            )
    relation_type_count = context.get("relation_type_count")
    if not isinstance(relation_type_count, int) or relation_type_count <= 0:
        raise ValueError("ReaRev construction requires a positive relation_type_count.")
    return ReaRevAnswerRetriever(
        hidden_dimension=int(architecture_options["gnn_hidden_dimension"]),
        num_instructions=int(architecture_options["num_instructions"]),
        reasoning_steps=int(architecture_options["reasoning_steps"]),
        adaptive_iterations=int(architecture_options["adaptive_iterations"]),
        dropout=float(architecture_options["dropout"]),
        text_encoder=text_encoder or _load_pinned_encoder(),
        encoder_width=int(context.get("encoder_width", REAREV_ENCODER_WIDTH)),
    )
