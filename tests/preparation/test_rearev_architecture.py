from __future__ import annotations

from types import SimpleNamespace

import pytest
import main

torch = pytest.importorskip("torch")
from torch import nn

from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import architecture_defaults
from pipeline.abstract import StepContext
from pipeline.preparation.exceptions import InvalidGnnArchitectureConfigurationException
from pipeline.preparation.steps.configuration_building import BuildPipelineConfigurationStep
from pipeline.preparation.steps.dataset_selection import SelectedDataset
from pipeline.preparation.models.rearev_answer_retriever import (
    ReaRevAnswerRetriever,
    ReaRevQueryReform,
    graph_softmax,
)
from pipeline.preparation.services.gnn_architecture_runtime import (
    ReaRevRuntimeStrategy,
)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 384)
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        width = kwargs["max_length"]
        return {
            "input_ids": torch.ones((len(texts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), width), dtype=torch.long),
        }


def _model(**overrides):
    settings = {
        "hidden_dimension": 50,
        "num_instructions": 2,
        "reasoning_steps": 2,
        "adaptive_iterations": 3,
        "dropout": 0.0,
        "text_encoder": FakeEncoder(),
    }
    settings.update(overrides)
    return ReaRevAnswerRetriever(**settings)


def _forward_inputs():
    return {
        "question_input_ids": torch.tensor([[1, 2, 3, 0]]),
        "question_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "relation_input_ids": torch.tensor([[4, 5, 0], [6, 7, 0]]),
        "relation_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "edge_index": torch.tensor([[0, 1], [1, 2]]),
        "edge_relation_index": torch.tensor([0, 1]),
        "initialization_edge_index": torch.tensor([[0], [1]]),
        "initialization_relation_index": torch.tensor([0]),
        "node_graph_index": torch.zeros(3, dtype=torch.long),
        "seed_distribution": torch.tensor([1.0, 0.0, 0.0]),
        "seed_mask": torch.tensor([True, False, False]),
        "graph_count": 1,
    }


def test_rearev_query_reform_uses_four_way_interaction_and_gru_update_gate():
    reform = ReaRevQueryReform(2)
    with torch.no_grad():
        reform.candidate_projection.weight.zero_()
        reform.candidate_projection.bias.zero_()
        # Select the element-wise interaction i * h_e from the fourth block.
        reform.candidate_projection.weight[0, 6] = 1.0
        reform.candidate_projection.weight[1, 7] = 1.0
        reform.gate_input_projection.weight.zero_()
        reform.gate_input_projection.bias.zero_()
        reform.gate_hidden_projection.weight.zero_()
    instruction = torch.tensor([[2.0, 3.0]])
    seed_state = torch.tensor([[4.0, 5.0]])
    expected_candidate = torch.tensor([[8.0, 15.0]])
    expected = 0.5 * instruction + 0.5 * expected_candidate
    assert torch.allclose(reform(instruction, seed_state), expected)


def test_rearev_revision_sums_multiple_seed_states_instead_of_averaging():
    model = _model(hidden_dimension=2, num_instructions=1)
    captured: dict[str, torch.Tensor] = {}

    class CaptureReform(nn.Module):
        def forward(self, instruction, seed_state):
            captured["seed_state"] = seed_state.detach().clone()
            return instruction

    model.query_reform_layers[0] = CaptureReform()
    model._revise_instructions(
        [torch.zeros((1, 2))],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]),
        torch.tensor([True, True, False]),
        torch.zeros(3, dtype=torch.long),
        1,
    )
    assert torch.equal(captured["seed_state"], torch.tensor([[4.0, 6.0]]))


def test_rearev_registry_defaults_and_requirements():
    definition = GNN_ARCHITECTURES["rearev"]
    assert definition.display_name == "ReaRev"
    assert architecture_defaults("rearev") == {
        "gnn_architecture": "rearev",
        "gnn_hidden_dimension": 50,
        "dropout": 0.1,
        "num_instructions": 2,
        "reasoning_steps": 2,
        "adaptive_iterations": 3,
    }
    assert definition.data_requirements.requires_reverse_edges
    assert definition.data_requirements.uses_relation_types
    assert not definition.data_requirements.uses_entity_embeddings
    assert not definition.data_requirements.uses_question_embeddings
    assert not definition.data_requirements.uses_relation_embeddings


def _dataset_context():
    return StepContext(
        result=SelectedDataset(
            dataset_id="WebQSP",
            display_name="WebQSP",
            dataset_family="question_answering",
            task_domain="knowledge_graph_question_answering",
            description="dataset",
            supported=True,
        )
    )


def test_rearev_cli_and_configuration_skip_openai_embedding_selection():
    args = main.build_parser().parse_args(
        [
            "--gnn-architecture",
            "rearev",
            "--num-instructions",
            "3",
            "--reasoning-steps",
            "1",
            "--adaptive-iterations",
            "2",
        ]
    )
    assert args.num_instructions == 3
    assert args.reasoning_steps == 1
    assert args.adaptive_iterations == 2

    options = architecture_defaults("rearev")
    options.pop("gnn_architecture")
    result = BuildPipelineConfigurationStep(
        gnn_architecture="rearev",
        gnn_options=options,
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        input_func=lambda _prompt: pytest.fail("ReaRev configuration should be complete"),
    ).execute(_dataset_context())
    assert result.embedding_model is None
    assert result.entity_embedding_model is None
    assert result.question_embedding_model is None
    assert result.relation_embedding_model is None
    assert result.use_reverse_edges


def test_rearev_interactive_prompts_only_rearev_options_in_registry_order(capsys):
    answers = iter(["5", "1", "2", "2", "2", "3"])
    result = BuildPipelineConfigurationStep(
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        input_func=lambda _prompt: next(answers),
    ).execute(_dataset_context())
    assert result.gnn_architecture == "rearev"
    assert result.gnn_architecture_options == {
        "gnn_hidden_dimension": 50,
        "dropout": 0.1,
        "num_instructions": 2,
        "reasoning_steps": 2,
        "adaptive_iterations": 3,
    }
    prompts = capsys.readouterr().out
    ordered_titles = [
        "GNN Architecture",
        "ReaRev Hidden Dimension",
        "GNN Dropout",
        "ReaRev Instructions",
        "ReaRev Reasoning Steps",
        "ReaRev Adaptive Iterations",
    ]
    positions = [prompts.index(title) for title in ordered_titles]
    assert positions == sorted(positions)
    assert "GNN Layer Count" not in prompts
    assert "Node Classifier" not in prompts
    assert "Embedding Model" not in prompts


@pytest.mark.parametrize(
    ("option_id", "value"),
    [
        ("gnn_layer_count", 2),
        ("node_classifier", "mlp"),
        ("num_bases", 8),
        ("attention_heads", 2),
        ("use_reverse_edges", True),
        ("use_reverse_edges", False),
        ("use_edge_mlp", True),
        ("question_aware_classifier", True),
        ("add_layer_normalization", True),
        ("edge_mlp_hidden_dim", 256),
    ],
)
def test_rearev_rejects_options_owned_by_other_architectures(option_id, value):
    options = architecture_defaults("rearev")
    options.pop("gnn_architecture")
    with pytest.raises(
        InvalidGnnArchitectureConfigurationException,
        match=f"does not support: {option_id}",
    ):
        BuildPipelineConfigurationStep(
            gnn_architecture="rearev",
            gnn_options={**options, option_id: value},
        ).execute(_dataset_context())


def test_rearev_forward_is_graph_normalizable_and_encoder_stays_frozen():
    model = _model()
    model.train()
    assert not model.text_encoder.training
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
    scores = model(**_forward_inputs())
    assert scores.shape == (3,)
    probabilities = graph_softmax(scores, torch.zeros(3, dtype=torch.long), 1)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    scores.sum().backward()
    assert model.question_projection.weight.grad is not None
    assert model.text_encoder.embedding.weight.grad is None
    assert model.text_encoder.calls == 2


def test_rearev_checkpoint_excludes_external_encoder():
    model = _model()
    checkpoint = model.trainable_state_dict()
    assert checkpoint
    assert not any(key.startswith("text_encoder.") for key in checkpoint)
    restored = _model()
    restored.load_trainable_state_dict(checkpoint)
    torch.testing.assert_close(
        restored.question_projection.weight,
        model.question_projection.weight,
    )


def test_rearev_empty_edges_and_isolated_nodes_remain_valid():
    model = _model(reasoning_steps=1, adaptive_iterations=1)
    inputs = _forward_inputs()
    inputs.update(
        relation_input_ids=torch.empty((0, 3), dtype=torch.long),
        relation_attention_mask=torch.empty((0, 3), dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_relation_index=torch.empty(0, dtype=torch.long),
        initialization_edge_index=torch.empty((2, 0), dtype=torch.long),
        initialization_relation_index=torch.empty(0, dtype=torch.long),
    )
    scores = model(**inputs)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_rearev_preparation_skips_zero_node_graphs_and_preserves_source_index(caplog):
    strategy = ReaRevRuntimeStrategy(tokenizer=FakeTokenizer())
    empty = SimpleNamespace(
        nodes=[],
        node2id={},
        q_entity=[],
        a_entity=[],
        question="empty",
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_relations=[],
        node_labels=torch.empty(0),
    )
    valid = SimpleNamespace(
        nodes=["m.entity"],
        node2id={"m.entity": 0},
        q_entity=["m.entity"],
        a_entity=["m.entity"],
        question="valid",
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_relations=[],
        node_labels=torch.tensor([1.0]),
    )

    prepared, _, _ = strategy._prepare_instances(
        instances=[empty, valid],
        relation_vocabulary={},
        source_start=112,
        torch=torch,
        evaluation=False,
    )

    assert len(prepared) == 1
    assert prepared[0].source_instance_index == 113
    assert "Skipping ReaRev graph with no nodes: instance_index=112" in caplog.text


def test_relation_normalization_distinguishes_inverse_direction():
    strategy = ReaRevRuntimeStrategy(tokenizer=object())
    assert strategy.normalize_relation_text("people.person.place_of_birth") == (
        "person place of birth"
    )
    assert strategy.normalize_relation_text(
        "reverse__people.person.place_of_birth"
    ) == "birth of place person"


def test_graph_balanced_kl_skips_graphs_without_answers():
    strategy = ReaRevRuntimeStrategy(tokenizer=object())
    batch = SimpleNamespace(
        valid_target_graphs=torch.tensor([True, False]),
        node_graph_index=torch.tensor([0, 0, 1, 1]),
        graph_count=2,
        node_labels=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    scores = torch.tensor([0.0, 1.0, 3.0, -2.0], requires_grad=True)
    loss = strategy.compute_loss(scores, batch)
    expected = -torch.log_softmax(scores[:2], dim=0)[1]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert scores.grad is not None
    torch.testing.assert_close(scores.grad[2:], torch.zeros(2))
