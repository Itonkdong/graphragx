"""HGT registry, operator, batching, and persistence tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import main

torch = pytest.importorskip("torch")

from pipeline.abstract import StepContext
from pipeline.evaluation.services.model_config_normalization import (
    normalize_model_config,
)
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.exceptions import (
    InvalidGnnArchitectureConfigurationException,
)
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import architecture_defaults
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.models.hgt_answer_retriever import (
    HeterogeneousGraphTransformerLayer,
    HGTAnswerRetriever,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
)
from pipeline.preparation.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingService,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    build_active_relation_groups,
    build_relation_architecture_context,
)
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationConfig,
    GnnTrainingDataPreparationService,
)
from pipeline.preparation.steps.configuration_building import (
    BuildPipelineConfigurationStep,
    BuiltPipelineConfiguration,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever


def _dataset_context() -> StepContext[SelectedDataset]:
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


def _options(**overrides) -> dict:
    return {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 8,
        "dropout": 0.0,
        "attention_heads": 2,
        **overrides,
    }


def _pipeline_config(options: dict) -> BuiltPipelineConfiguration:
    return BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        gnn_architecture_options=options,
        main_llm_model="gpt-5.4",
        subgraph_construction_algorithm="shortest_path",
        context_construction_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        entity_embedding_model="text-embedding-3-small",
        use_reverse_edges=True,
    )


def test_hgt_registry_defaults_and_requirements() -> None:
    definition = GNN_ARCHITECTURES["hgt"]
    assert architecture_defaults("hgt") == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "attention_heads": 8,
        "gnn_architecture": "hgt",
    }
    assert definition.display_name == "HGT"
    assert definition.data_requirements.requires_reverse_edges
    assert definition.data_requirements.uses_relation_types
    assert not definition.data_requirements.uses_question_embeddings
    assert not definition.data_requirements.uses_relation_embeddings
    assert main.PipelineRuntimeConfig().training_batch_size == 1


def test_hgt_cli_and_default_configuration() -> None:
    args = main.build_parser().parse_args(
        ["--gnn-architecture", "hgt", "--attention-heads", "4"]
    )
    assert args.attention_heads == 4

    result = BuildPipelineConfigurationStep(
        gnn_architecture="hgt",
        gnn_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "attention_heads": 8,
        },
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
    ).execute(_dataset_context())
    assert result.gnn_architecture == "hgt"
    assert result.use_reverse_edges
    assert result.node_classifier is None


def test_hgt_interactive_prompts_follow_registry_order(capsys) -> None:
    answers = iter(["4", "1", "2", "2", "4"])

    result = BuildPipelineConfigurationStep(
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        input_func=lambda _prompt: next(answers),
    ).execute(_dataset_context())

    assert result.gnn_architecture == "hgt"
    assert result.gnn_architecture_options == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "attention_heads": 8,
    }
    prompt_text = capsys.readouterr().out
    assert prompt_text.index("GNN Architecture") < prompt_text.index("GNN Layer Count")
    assert prompt_text.index("GNN Layer Count") < prompt_text.index(
        "GNN Hidden Dimension"
    )
    assert prompt_text.index("GNN Hidden Dimension") < prompt_text.index("GNN Dropout")
    assert prompt_text.index("GNN Dropout") < prompt_text.index("Attention Heads")
    assert "Node Classifier" not in prompt_text
    assert "Basis Count" not in prompt_text


@pytest.mark.parametrize(
    ("option_id", "value"),
    [
        ("node_classifier", "mlp"),
        ("num_bases", 8),
        ("use_reverse_edges", True),
        ("use_reverse_edges", False),
        ("use_edge_mlp", False),
        ("question_aware_classifier", False),
        ("add_layer_normalization", True),
        ("edge_mlp_hidden_dim", 256),
    ],
)
def test_hgt_rejects_options_owned_by_other_architectures(
    option_id: str,
    value,
) -> None:
    step = BuildPipelineConfigurationStep(
        gnn_architecture="hgt",
        gnn_options={**_options(), option_id: value},
    )
    with pytest.raises(
        InvalidGnnArchitectureConfigurationException,
        match=f"does not support: {option_id}",
    ):
        step.execute(_dataset_context())


def test_hgt_layer_attention_is_normalized_per_target_and_head() -> None:
    layer = HeterogeneousGraphTransformerLayer(
        hidden_dimension=8,
        num_relations=2,
        attention_heads=2,
        dropout=0.0,
    )
    node_features = torch.randn(4, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 3]])
    edge_type = torch.tensor([0, 0, 1, 1])
    output = layer(node_features, edge_index, edge_type)

    assert output.shape == (4, 8)
    assert layer.last_attention is not None
    for target in (2, 3):
        mask = edge_index[1] == target
        assert torch.allclose(
            layer.last_attention[mask].sum(dim=0),
            torch.ones(2),
            atol=1e-6,
        )
    output.square().mean().backward()
    assert node_features.grad is not None
    assert torch.isfinite(node_features.grad).all()


def test_hgt_layer_batches_relations_with_equal_edge_counts(monkeypatch) -> None:
    layer = HeterogeneousGraphTransformerLayer(
        hidden_dimension=8,
        num_relations=4,
        attention_heads=2,
        dropout=0.0,
    )
    node_features = torch.randn(6, 8, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 2, 2, 3, 3], dtype=torch.long)
    original_einsum = torch.einsum
    einsum_calls = 0

    def counted_einsum(*args, **kwargs):
        nonlocal einsum_calls
        einsum_calls += 1
        return original_einsum(*args, **kwargs)

    monkeypatch.setattr(torch, "einsum", counted_einsum)
    output = layer(node_features, edge_index, edge_type)
    output.sum().backward()

    # Key and value transforms run once for each distinct relation edge count
    # (one and two), rather than once for every active relation.
    assert einsum_calls == 4
    assert layer.relation_attention.grad is not None
    assert layer.relation_message.grad is not None


def test_hgt_layer_supports_bfloat16_autocast() -> None:
    layer = HeterogeneousGraphTransformerLayer(
        hidden_dimension=8,
        num_relations=2,
        attention_heads=2,
        dropout=0.0,
    )
    node_features = torch.randn(4, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 3]])
    edge_type = torch.tensor([0, 0, 1, 1])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = layer(node_features, edge_index, edge_type)
        loss = output.square().mean()
    loss.backward()

    assert output.shape == (4, 8)
    assert node_features.grad is not None
    assert torch.isfinite(node_features.grad).all()


def test_hgt_layer_handles_empty_edges_through_residual_path() -> None:
    layer = HeterogeneousGraphTransformerLayer(8, 2, 2, 0.0)
    node_features = torch.randn(3, 8)
    output = layer(
        node_features,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert output.shape == node_features.shape
    assert torch.isfinite(output).all()


def test_hgt_layer_matches_pyg_hgt_conv_without_layer_norm() -> None:
    geometric_nn = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(7)
    hidden_dimension = 8
    heads = 2
    relation_count = 2
    custom = HeterogeneousGraphTransformerLayer(
        hidden_dimension,
        relation_count,
        heads,
        0.0,
        use_layer_normalization=False,
    ).eval()
    edge_types = [
        ("entity", "r0", "entity"),
        ("entity", "r1", "entity"),
    ]
    reference = geometric_nn.HGTConv(
        hidden_dimension,
        hidden_dimension,
        (["entity"], edge_types),
        heads=heads,
    ).eval()

    with torch.no_grad():
        state = reference.state_dict()
        state["kqv_lin.lins.entity.weight"].copy_(
            torch.cat(
                [
                    custom.key_projection.weight,
                    custom.query_projection.weight,
                    custom.value_projection.weight,
                ]
            )
        )
        state["kqv_lin.lins.entity.bias"].copy_(
            torch.cat(
                [
                    custom.key_projection.bias,
                    custom.query_projection.bias,
                    custom.value_projection.bias,
                ]
            )
        )
        state["out_lin.lins.entity.weight"].copy_(custom.output_projection.weight)
        state["out_lin.lins.entity.bias"].copy_(custom.output_projection.bias)
        for head in range(heads):
            for relation in range(relation_count):
                type_index = head * relation_count + relation
                state["k_rel.weight"][type_index].copy_(
                    custom.relation_attention[relation, head]
                )
                state["v_rel.weight"][type_index].copy_(
                    custom.relation_message[relation, head]
                )
        for relation in range(relation_count):
            state[f"p_rel.entity__r{relation}__entity"].copy_(
                custom.relation_prior[relation].reshape(1, heads)
            )
        state["skip.entity"].copy_(custom.skip)
        reference.load_state_dict(state)

    node_features = torch.randn(5, hidden_dimension)
    relation_zero_edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    relation_one_edges = torch.tensor([[3, 4], [4, 0]])
    edge_index = torch.cat([relation_zero_edges, relation_one_edges], dim=1)
    edge_type = torch.tensor([0, 0, 0, 1, 1])

    custom_output = custom(node_features, edge_index, edge_type)
    reference_output = reference(
        {"entity": node_features},
        {
            edge_types[0]: relation_zero_edges,
            edge_types[1]: relation_one_edges,
        },
    )["entity"]
    assert torch.allclose(custom_output, reference_output, atol=1e-6)


def test_hgt_disconnected_batch_matches_individual_graphs() -> None:
    model = HGTAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=2,
        attention_heads=2,
        dropout=0.0,
    ).eval()
    embeddings = torch.randn(4, 4)

    def instance(index: int) -> PreparedGnnTrainingInstance:
        edge_type = torch.tensor([0, 1])
        active_ids, offsets = build_active_relation_groups(
            edge_type=edge_type,
            torch=torch,
        )
        return PreparedGnnTrainingInstance(
            source_instance_index=index,
            node_embedding_indices=torch.tensor([index * 2, index * 2 + 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_type=edge_type,
            active_relation_ids=active_ids,
            active_relation_offsets=offsets,
            node_labels=torch.tensor([0.0, 1.0]),
        )

    instances = [instance(0), instance(1)]
    prepared = PreparedGnnTrainingData.model_construct(
        instances=instances,
        node_embeddings=embeddings,
    )
    batch = GnnAnswerRetrieverTrainingService._build_typed_training_batches(
        prepared_training_data=prepared,
        batch_size=2,
        torch=torch,
        device="cpu",
        architecture_id="hgt",
    )[0]
    assert batch.active_relation_offsets.device.type == "cpu"

    individual = torch.cat(
        [
            model(
                embeddings.index_select(0, item.node_embedding_indices),
                item.edge_index,
                edge_type=item.edge_type,
                active_relation_ids=item.active_relation_ids,
                active_relation_offsets=item.active_relation_offsets,
            )
            for item in instances
        ]
    )
    combined = model(
        embeddings.index_select(0, batch.node_embedding_indices),
        batch.edge_index,
        edge_type=batch.edge_type,
        active_relation_ids=batch.active_relation_ids,
        active_relation_offsets=batch.active_relation_offsets,
    )
    assert torch.allclose(individual, combined, atol=1e-6)


class _EmbeddingService:
    def __init__(self) -> None:
        self.node_calls = 0
        self.relation_calls = 0
        self.question_calls = 0

    def load_node_cache(self, **_):
        self.node_calls += 1
        return SimpleNamespace(vector_size=4, cache_kind="nodes")

    def load_relation_cache(self, **_):
        self.relation_calls += 1
        raise AssertionError("HGT must not load textual relation embeddings")

    def load_question_cache(self, **_):
        self.question_calls += 1
        raise AssertionError("HGT must not load question embeddings")


class _TensorCacheService:
    def load_matrix(self, *, texts, dtype, device, **_):
        return torch.ones((len(texts), 4), dtype=dtype, device=device)


def test_hgt_training_preparation_uses_only_entities_and_relation_groups(
    tmp_path,
) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    options = _options()
    built = BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        gnn_architecture_options=options,
        gnn_architecture_context=build_relation_architecture_context(vocabulary),
        relation_vocabulary=vocabulary,
        entity_embedding_model="text-embedding-3-small",
        entity_embedding_dimension=4,
        question_embedding_dimension=4,
        relation_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        node_classifier="mlp",
        use_reverse_edges=True,
        add_layer_normalization=True,
        model=HGTAnswerRetriever(
            entity_embedding_dimension=4,
            hidden_dimension=8,
            gnn_layer_count=2,
            num_relations=2,
            attention_heads=2,
            dropout=0.0,
        ),
    )
    instance = WebQSPProcessedInstance(
        question="Who is the parent?",
        q_entity=["A"],
        a_entity=["B"],
        nodes=["A", "B"],
        node2id={"A": 0, "B": 1},
        edge_index=torch.tensor([[1, 0], [0, 1]]),
        edge_relations=["reverse__parent", "parent"],
        node_labels=torch.tensor([0.0, 1.0]),
    )
    dataset = PreparedWebQSPGraphDataset(
        dataset_id="WebQSP",
        processing_version="test",
        use_reverse_edges=True,
        train_instances=[instance],
        test_instances=[],
        vocabulary_store=WebQSPVocabularyStore(
            nodes={"A": 0, "B": 1},
            relations=vocabulary,
            questions={instance.question: 0},
        ),
        cache_directory=tmp_path / "processed_reverse_edges",
    )
    embedding_service = _EmbeddingService()
    prepared = GnnTrainingDataPreparationService(
        embedding_cache_service=embedding_service,
        embedding_tensor_cache_service=_TensorCacheService(),
    ).prepare(
        built_retriever=built,
        prepared_dataset=dataset,
        configuration=_pipeline_config(options),
        preparation_config=GnnTrainingDataPreparationConfig(
            training_device="cpu",
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
    )

    assert embedding_service.node_calls == 1
    assert embedding_service.relation_calls == 0
    assert embedding_service.question_calls == 0
    assert prepared.question_embeddings is None
    assert prepared.relation_embeddings is None
    assert prepared.instances[0].edge_type.tolist() == [0, 1]
    assert prepared.instances[0].edge_norm is None
    assert prepared.instances[0].active_relation_ids.tolist() == [0, 1]
    assert prepared.instances[0].active_relation_offsets.tolist() == [0, 1, 2]


def test_hgt_saved_model_round_trip_uses_relation_context(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = build_relation_architecture_context(vocabulary)
    options = _options(gnn_hidden_dimension=128)
    model = HGTAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=128,
        gnn_layer_count=2,
        num_relations=2,
        attention_heads=2,
        dropout=0.0,
    )
    run_directory = tmp_path / "models" / "1_hgt"
    run_directory.mkdir(parents=True)
    torch.save(model.state_dict(), run_directory / "gnn_answer_retriever.pt")
    (run_directory / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "hgt",
                "gnn_architecture_options": options,
                "gnn_architecture_context": context,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 4,
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "relation_vocabulary.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )
    configuration = BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="hgt",
        gnn_architecture_options=options,
        main_llm_model="gpt-5.4",
        subgraph_construction_algorithm="shortest_path",
        context_construction_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        entity_embedding_model="text-embedding-3-small",
    )
    loaded = GnnAnswerRetrieverModelRunService().load_run(
        model_root=tmp_path / "models",
        run_name="1_hgt",
        run_number=None,
        pipeline_configuration=configuration,
        device="cpu",
    )
    assert loaded.config.resolved_gnn_architecture == "hgt"
    assert loaded.config.resolved_gnn_architecture_options == options
    assert loaded.relation_vocabulary == vocabulary
    assert isinstance(loaded.model, HGTAnswerRetriever)


def test_hgt_wandb_model_config_and_tag_are_canonical() -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    normalized = normalize_model_config(
        {
            "dataset_id": "WebQSP",
            "gnn_architecture": "hgt",
            "gnn_architecture_options": {
                "gnn_layer_count": 2,
                "gnn_hidden_dimension": 256,
                "dropout": 0.1,
                "attention_heads": 8,
            },
            "gnn_architecture_context": build_relation_architecture_context(
                vocabulary
            ),
            "embedding_model": "text-embedding-3-small",
            "parameter_count": 1234,
            "estimated_training_parameter_bytes": 19744,
            "training": {"epochs": 2, "loss_history": []},
        }
    )

    assert normalized["gnn_architecture"] == "hgt"
    assert normalized["gnn_architecture_options"]["attention_heads"] == 8
    assert normalized["gnn_architecture_context"]["relation_type_count"] == 2
    assert normalized["parameter_count"] == 1234
    assert normalized["estimated_training_parameter_bytes"] == 19744
    assert "gnn_architecture" not in normalized["training"]
    tags = WandbExperimentCoordinator._build_tags_from_config(
        {"configs": {"model": normalized}}
    )
    assert "hgt" in tags
