"""R-GCN registry, layer, vocabulary, and preparation tests."""

from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import main

torch = pytest.importorskip("torch")

from pipeline.abstract import StepContext
from pipeline.preparation.exceptions import (
    InvalidGnnArchitectureConfigurationException,
)
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import architecture_defaults
from pipeline.preparation.models.rgcn_answer_retriever import (
    ActiveRelationBasisRGCNLayer,
    RGCNAnswerRetriever,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    build_relation_aggregation_metadata,
    build_relation_architecture_context,
    build_sorted_typed_edges,
    validate_relation_architecture_context,
)
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationConfig,
    GnnTrainingDataPreparationService,
)
from pipeline.preparation.services.gnn_evaluation_data_preparation import (
    GnnEvaluationDataPreparationService,
)
from pipeline.evaluation.models import GnnAnswerRetrieverEvaluationConfig
from pipeline.evaluation.services.model_config_normalization import (
    normalize_model_config,
)
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverModelRunException
from pipeline.preparation.steps.configuration_building import (
    BuildPipelineConfigurationStep,
    BuiltPipelineConfiguration,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingConfig,
    GnnAnswerRetrieverTrainingService,
)


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


def _pipeline_config(options: dict) -> BuiltPipelineConfiguration:
    return BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
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


def test_rgcn_registry_defaults_and_requirements() -> None:
    defaults = architecture_defaults("rgcn")
    definition = GNN_ARCHITECTURES["rgcn"]

    assert defaults == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "num_bases": 30,
        "gnn_architecture": "rgcn",
    }
    assert definition.display_name == "R-GCN"
    assert definition.data_requirements.requires_reverse_edges
    assert definition.data_requirements.uses_relation_types
    assert not definition.data_requirements.uses_question_embeddings
    assert not definition.data_requirements.uses_relation_embeddings


def test_rgcn_cli_exposes_only_its_registered_numeric_option() -> None:
    args = main.build_parser().parse_args(
        [
            "--gnn-architecture",
            "rgcn",
            "--num-bases",
            "16",
            "--training-batch-size",
            "32",
        ]
    )

    assert args.gnn_architecture == "rgcn"
    assert args.num_bases == 16
    assert args.training_batch_size == 32


def test_rgcn_interactive_prompts_follow_registry_order(capsys) -> None:
    answers = iter(["3", "1", "2", "2", "3"])
    prompts: list[str] = []

    def choose(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    result = BuildPipelineConfigurationStep(
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        input_func=choose,
    ).execute(_dataset_context())

    assert result.gnn_architecture == "rgcn"
    assert result.gnn_architecture_options == {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 256,
        "dropout": 0.1,
        "num_bases": 30,
    }
    prompt_text = capsys.readouterr().out
    assert prompt_text.index("GNN Architecture") < prompt_text.index("GNN Layer Count")
    assert prompt_text.index("GNN Layer Count") < prompt_text.index(
        "GNN Hidden Dimension"
    )
    assert prompt_text.index("GNN Hidden Dimension") < prompt_text.index("GNN Dropout")
    assert prompt_text.index("GNN Dropout") < prompt_text.index("R-GCN Basis Count")
    assert "Node Classifier" not in prompt_text


@pytest.mark.parametrize(
    ("option_id", "value"),
    [
        ("node_classifier", "mlp"),
        ("use_reverse_edges", True),
        ("use_reverse_edges", False),
        ("use_edge_mlp", False),
        ("question_aware_classifier", False),
        ("add_layer_normalization", False),
        ("edge_mlp_hidden_dim", 256),
    ],
)
def test_rgcn_rejects_graphsage_specific_options(option_id: str, value) -> None:
    step = BuildPipelineConfigurationStep(
        gnn_architecture="rgcn",
        gnn_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "num_bases": 30,
            option_id: value,
        },
    )

    with pytest.raises(
        InvalidGnnArchitectureConfigurationException,
        match=f"does not support: {option_id}",
    ):
        step.execute(_dataset_context())


def test_rgcn_configuration_makes_reverse_edges_mandatory() -> None:
    result = BuildPipelineConfigurationStep(
        gnn_architecture="rgcn",
        gnn_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 256,
            "dropout": 0.1,
            "num_bases": 30,
        },
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
    ).execute(_dataset_context())

    assert result.use_reverse_edges
    assert result.node_classifier is None


def test_typed_edges_are_sorted_and_keep_inverse_relations_distinct() -> None:
    vocabulary = {"reverse__parent": 0, "parent": 1}
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    sorted_edges, edge_type = build_sorted_typed_edges(
        edge_index=edge_index,
        edge_relations=["parent", "reverse__parent", "parent"],
        vocabulary=vocabulary,
        torch=torch,
    )

    assert edge_type.tolist() == [0, 1, 1]
    assert sorted_edges.tolist() == [[1, 0, 2], [2, 1, 0]]
    context = build_relation_architecture_context(vocabulary)
    validate_relation_architecture_context(context, vocabulary)
    with pytest.raises(ValueError, match="relation_vocabulary_sha256"):
        validate_relation_architecture_context(
            context,
            {"reverse__parent": 1, "parent": 0},
        )


def test_relation_mean_normalization_is_precomputed_once() -> None:
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 2]])
    edge_type = torch.tensor([0, 0, 1])

    edge_norm, active_relation_ids, edge_relation_index = (
        build_relation_aggregation_metadata(
            edge_index=edge_index,
            edge_type=edge_type,
            node_count=3,
            torch=torch,
        )
    )

    assert edge_norm.tolist() == [0.5, 0.5, 1.0]
    assert active_relation_ids.tolist() == [0, 1]
    assert edge_relation_index.tolist() == [0, 0, 1]


def test_rgcn_layer_uses_per_relation_target_mean_and_root_transform() -> None:
    layer = ActiveRelationBasisRGCNLayer(
        hidden_dimension=1,
        num_relations=2,
        num_bases=2,
    )
    with torch.no_grad():
        layer.basis_weights.copy_(torch.tensor([[[2.0]], [[5.0]]]))
        layer.relation_coefficients.copy_(torch.eye(2))
        layer.root_weight.fill_(3.0)
        layer.bias.fill_(1.0)
    node_features = torch.tensor([[1.0], [3.0], [10.0]])
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 2]])
    edge_type = torch.tensor([0, 0, 1])

    output = layer(node_features, edge_index, edge_type)

    # Relation 0 contributes mean(1*2, 3*2)=4, relation 1 contributes
    # 10*5=50, and the target's root/bias contributes 10*3+1.
    assert output[2].item() == pytest.approx(85.0)
    assert output[0].item() == pytest.approx(4.0)


def test_rgcn_layer_supports_empty_edges_gradients_and_bfloat16_autocast() -> None:
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=3,
        num_bases=2,
        dropout=0.1,
    )
    entity_features = torch.randn(4, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.empty(0, dtype=torch.long)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(entity_features, edge_index, edge_type=edge_type)
        loss = logits.float().sum()
    loss.backward()

    assert logits.shape == (4,)
    assert model.entity_projection.weight.grad is not None
    assert model.gnn_layers[0].root_weight.grad is not None


def test_rgcn_layer_uses_autocast_compute_dtype_for_index_add_accumulators() -> None:
    layer = ActiveRelationBasisRGCNLayer(8, num_relations=3, num_bases=2)
    node_features = torch.randn(4, 8, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first_output = layer(node_features, edge_index, edge_type)
        second_output = layer(first_output, edge_index, edge_type)

    assert first_output.dtype == torch.bfloat16
    assert second_output.dtype == torch.bfloat16


def test_vectorized_rgcn_metadata_matches_fallback_and_chunks_backward() -> None:
    torch.manual_seed(9)
    layer = ActiveRelationBasisRGCNLayer(
        8,
        num_relations=3,
        num_bases=2,
        message_chunk_size=7,
    )
    node_features = torch.randn(20, 8, requires_grad=True)
    edge_count = 41
    edge_type = torch.arange(edge_count, dtype=torch.long) % 3
    edge_type, order = torch.sort(edge_type)
    edge_index = torch.stack(
        (
            torch.arange(edge_count) % 20,
            (torch.arange(edge_count) * 3 + 1) % 20,
        )
    ).index_select(1, order)
    edge_norm, active_relation_ids, edge_relation_index = (
        build_relation_aggregation_metadata(
            edge_index=edge_index,
            edge_type=edge_type,
            node_count=20,
            torch=torch,
        )
    )

    fallback = layer(node_features, edge_index, edge_type)
    prepared = layer(
        node_features,
        edge_index,
        edge_type,
        edge_norm=edge_norm,
        active_relation_ids=active_relation_ids,
        edge_relation_index=edge_relation_index,
    )
    assert torch.allclose(prepared, fallback, atol=1e-6)
    prepared.sum().backward()
    assert layer.basis_weights.grad is not None


def test_disconnected_rgcn_batch_matches_individual_graphs_and_losses(tmp_path) -> None:
    torch.manual_seed(12)
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=2,
        dropout=0.0,
    ).eval()
    embedding_matrix = torch.randn(5, 4)

    def prepared_instance(
        node_indices: list[int],
        edge_index,
        edge_type,
        labels: list[float],
        source_index: int,
    ) -> PreparedGnnTrainingInstance:
        edge_norm, active_ids, edge_relation_index = (
            build_relation_aggregation_metadata(
                edge_index=edge_index,
                edge_type=edge_type,
                node_count=len(node_indices),
                torch=torch,
            )
        )
        return PreparedGnnTrainingInstance(
            source_instance_index=source_index,
            node_embedding_indices=torch.tensor(node_indices),
            edge_index=edge_index,
            edge_type=edge_type,
            edge_norm=edge_norm,
            active_relation_ids=active_ids,
            edge_relation_index=edge_relation_index,
            node_labels=torch.tensor(labels),
        )

    first = prepared_instance(
        [0, 1],
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0, 1]),
        [0.0, 1.0],
        0,
    )
    second = prepared_instance(
        [2, 3, 4],
        torch.tensor([[0, 1, 2], [1, 2, 0]]),
        torch.tensor([0, 0, 1]),
        [1.0, 0.0, 0.0],
        1,
    )
    built = BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
        gnn_architecture_options={},
        gnn_architecture_context={},
        entity_embedding_model="text-embedding-3-small",
        entity_embedding_dimension=4,
        question_embedding_dimension=4,
        relation_embedding_dimension=4,
        model=model,
    )
    prepared_data = PreparedGnnTrainingData(
        built_retriever=built,
        instances=[first, second],
        node_embeddings=embedding_matrix,
        training_start_instance=0,
        training_end_instance=2,
        selected_device="cpu",
        embedding_cache_device="cpu",
        embedding_cache_dtype="float32",
        entity_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        cache_root=tmp_path,
    )
    batch = GnnAnswerRetrieverTrainingService._build_rgcn_training_batches(
        prepared_training_data=prepared_data,
        batch_size=2,
        torch=torch,
        device="cpu",
    )[0]

    individual_logits = []
    for instance in (first, second):
        individual_logits.append(
            model(
                embedding_matrix.index_select(0, instance.node_embedding_indices),
                instance.edge_index,
                edge_type=instance.edge_type,
                edge_norm=instance.edge_norm,
                active_relation_ids=instance.active_relation_ids,
                edge_relation_index=instance.edge_relation_index,
            )
        )
    batch_logits = model(
        embedding_matrix.index_select(0, batch.node_embedding_indices),
        batch.edge_index,
        edge_type=batch.edge_type,
        edge_norm=batch.edge_norm,
        active_relation_ids=batch.active_relation_ids,
        edge_relation_index=batch.edge_relation_index,
    )
    assert torch.allclose(batch_logits, torch.cat(individual_logits), atol=1e-6)

    import torch.nn.functional as functional

    batch_loss = functional.binary_cross_entropy_with_logits(
        batch_logits,
        batch.node_labels,
        weight=batch.node_loss_weights,
        pos_weight=batch.positive_weights,
        reduction="sum",
    )
    individual_loss = sum(
        GnnAnswerRetrieverTrainingService._compute_loss(
            logits=logits,
            node_labels=instance.node_labels,
            torch=torch,
            torch_functional=functional,
        )
        for logits, instance in zip(individual_logits, (first, second), strict=True)
    ) / 2
    assert batch_loss.item() == pytest.approx(individual_loss.item(), abs=1e-6)


def test_batched_rgcn_training_smoke_saves_vocabulary_and_model(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    options = {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 128,
        "dropout": 0.1,
        "num_bases": 8,
    }
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=128,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=8,
        dropout=0.1,
    )
    built = BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
        gnn_architecture_options=options,
        gnn_architecture_context=build_relation_architecture_context(vocabulary),
        relation_vocabulary=vocabulary,
        entity_embedding_model="text-embedding-3-small",
        entity_embedding_dimension=4,
        question_embedding_dimension=4,
        relation_embedding_dimension=4,
        hidden_dimension=128,
        gnn_layer_count=2,
        node_classifier="mlp",
        use_reverse_edges=True,
        dropout=0.1,
        model=model,
    )

    def instance(index: int) -> PreparedGnnTrainingInstance:
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_type = torch.tensor([0, 1])
        edge_norm, active_ids, edge_relation_index = (
            build_relation_aggregation_metadata(
                edge_index=edge_index,
                edge_type=edge_type,
                node_count=2,
                torch=torch,
            )
        )
        return PreparedGnnTrainingInstance(
            source_instance_index=index,
            node_embedding_indices=torch.tensor([index * 2, index * 2 + 1]),
            edge_index=edge_index,
            edge_type=edge_type,
            edge_norm=edge_norm,
            active_relation_ids=active_ids,
            edge_relation_index=edge_relation_index,
            node_labels=torch.tensor([0.0, 1.0]),
        )

    prepared_data = PreparedGnnTrainingData(
        built_retriever=built,
        instances=[instance(0), instance(1)],
        node_embeddings=torch.randn(4, 4),
        training_start_instance=0,
        training_end_instance=2,
        selected_device="cpu",
        embedding_cache_device="cpu",
        embedding_cache_dtype="float32",
        entity_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        cache_root=tmp_path,
    )
    dataset = PreparedWebQSPGraphDataset(
        dataset_id="WebQSP",
        processing_version="test",
        use_reverse_edges=True,
        train_instances=[],
        test_instances=[],
        vocabulary_store=WebQSPVocabularyStore(relations=vocabulary),
        cache_directory=tmp_path / "processed_reverse_edges",
    )

    epoch_events: list[dict[str, float | int]] = []
    outcome = GnnAnswerRetrieverTrainingService(
        epoch_callback=epoch_events.append,
    ).train(
        prepared_training_data=prepared_data,
        prepared_dataset=dataset,
        configuration=_pipeline_config(options),
        training_config=GnnAnswerRetrieverTrainingConfig(
            epochs=1,
            batch_size=2,
            log_every=0,
            device="cpu",
        ),
    )

    assert outcome.trained_instances == 2
    assert outcome.model_artifact_path.exists()
    assert outcome.relation_vocabulary_path is not None
    assert outcome.relation_vocabulary_path.exists()
    assert epoch_events == [
        {
            "epoch": 1,
            "gnn_architecture": "rgcn",
            "average_loss": outcome.final_loss,
        }
    ]


def test_custom_layer_matches_pyg_rgcn_conv() -> None:
    geometric_nn = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(4)
    custom = ActiveRelationBasisRGCNLayer(4, num_relations=3, num_bases=2)
    reference = geometric_nn.RGCNConv(
        4,
        4,
        num_relations=3,
        num_bases=2,
        aggr="mean",
        root_weight=True,
        bias=True,
    )
    with torch.no_grad():
        reference.weight.copy_(custom.basis_weights)
        reference.comp.copy_(custom.relation_coefficients)
        reference.root.copy_(custom.root_weight)
        reference.bias.copy_(custom.bias)
    node_features = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 2]])
    edge_type = torch.tensor([0, 0, 2, 1, 2])

    assert torch.allclose(
        custom(node_features, edge_index, edge_type),
        reference(node_features, edge_index, edge_type),
        atol=1e-6,
    )


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
        raise AssertionError("R-GCN must not load relation embeddings")

    def load_question_cache(self, **_):
        self.question_calls += 1
        raise AssertionError("R-GCN must not load question embeddings")


class _TensorCacheService:
    def load_matrix(self, *, texts, dtype, device, **_):
        return torch.ones((len(texts), 4), dtype=dtype, device=device)


def test_rgcn_training_preparation_loads_only_entity_embeddings(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = build_relation_architecture_context(vocabulary)
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=2,
        dropout=0.1,
    )
    built = BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="rgcn",
        gnn_architecture_options={
            "gnn_layer_count": 2,
            "gnn_hidden_dimension": 8,
            "dropout": 0.1,
            "num_bases": 2,
        },
        gnn_architecture_context=context,
        relation_vocabulary=vocabulary,
        entity_embedding_model="text-embedding-3-small",
        entity_embedding_dimension=4,
        question_embedding_dimension=4,
        relation_embedding_dimension=4,
        hidden_dimension=8,
        gnn_layer_count=2,
        node_classifier="mlp",
        use_reverse_edges=True,
        model=model,
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
    service = GnnTrainingDataPreparationService(
        embedding_cache_service=embedding_service,
        embedding_tensor_cache_service=_TensorCacheService(),
    )
    configuration = _pipeline_config(built.gnn_architecture_options)

    prepared = service.prepare(
        built_retriever=built,
        prepared_dataset=dataset,
        configuration=configuration,
        preparation_config=GnnTrainingDataPreparationConfig(
            training_device="cpu",
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
    )

    assert embedding_service.node_calls == 1
    assert embedding_service.relation_calls == 0
    assert embedding_service.question_calls == 0
    assert prepared.relation_embeddings is None
    assert prepared.question_embeddings is None
    assert prepared.instances[0].edge_type.tolist() == [0, 1]
    assert prepared.instances[0].edge_index.tolist() == [[0, 1], [1, 0]]


def test_rgcn_evaluation_preparation_loads_only_entity_embeddings(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
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
    embedding_service = _EmbeddingService()
    service = GnnEvaluationDataPreparationService(
        embedding_cache_service=embedding_service,
        embedding_tensor_cache_service=_TensorCacheService(),
    )

    prepared = service.prepare(
        torch=torch,
        test_instances=[instance],
        cache_root=tmp_path,
        dataset_id="WebQSP",
        entity_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        selected_device="cpu",
        evaluation_config=GnnAnswerRetrieverEvaluationConfig(
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
        gnn_architecture="rgcn",
        relation_vocabulary=vocabulary,
    )

    assert embedding_service.node_calls == 1
    assert embedding_service.relation_calls == 0
    assert embedding_service.question_calls == 0
    assert prepared.relation_embeddings is None
    assert prepared.question_embeddings is None
    assert prepared.instances[0].edge_type.tolist() == [0, 1]


def test_rgcn_evaluation_preparation_filters_empty_graph_before_edge_metadata(
    tmp_path,
) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    valid = WebQSPProcessedInstance(
        question="Who is the parent?",
        q_entity=["A"],
        a_entity=["B"],
        nodes=["A", "B"],
        node2id={"A": 0, "B": 1},
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_relations=["parent", "reverse__parent"],
        node_labels=torch.tensor([0.0, 1.0]),
    )
    empty = WebQSPProcessedInstance(
        question="Empty graph",
        q_entity=["missing-seed"],
        a_entity=["missing-answer"],
        nodes=[],
        node2id={},
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_relations=[],
        node_labels=torch.empty(0),
    )
    service = GnnEvaluationDataPreparationService(
        embedding_cache_service=_EmbeddingService(),
        embedding_tensor_cache_service=_TensorCacheService(),
    )

    prepared = service.prepare(
        torch=torch,
        test_instances=[valid, empty],
        cache_root=tmp_path,
        dataset_id="WebQSP",
        entity_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        selected_device="cpu",
        evaluation_config=GnnAnswerRetrieverEvaluationConfig(
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
        gnn_architecture="rgcn",
        relation_vocabulary=vocabulary,
    )

    assert [item.source_instance_index for item in prepared.instances] == [0]
    assert prepared.skipped_missing_gold_in_graph_count == 1


def test_rgcn_evaluation_keeps_empty_graph_as_explicit_miss_when_filter_disabled(
    tmp_path,
) -> None:
    empty = WebQSPProcessedInstance(
        question="Empty graph",
        q_entity=["missing-seed"],
        a_entity=["missing-answer"],
        nodes=[],
        node2id={},
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_relations=[],
        node_labels=torch.empty(0),
    )
    service = GnnEvaluationDataPreparationService(
        embedding_cache_service=_EmbeddingService(),
        embedding_tensor_cache_service=_TensorCacheService(),
    )

    prepared = service.prepare(
        torch=torch,
        test_instances=[empty],
        cache_root=tmp_path,
        dataset_id="WebQSP",
        entity_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        selected_device="cpu",
        evaluation_config=GnnAnswerRetrieverEvaluationConfig(
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
            skip_missing_gold_in_graph=False,
        ),
        gnn_architecture="rgcn",
        relation_vocabulary={"parent": 0, "reverse__parent": 1},
    )

    assert len(prepared.instances) == 1
    assert prepared.instances[0].skip_reason == "empty_graph"
    assert prepared.instances[0].edge_type is None


def test_rgcn_model_artifact_round_trip_uses_saved_relation_context(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = build_relation_architecture_context(vocabulary)
    options = {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 128,
        "dropout": 0.1,
        "num_bases": 8,
    }
    model = RGCNAnswerRetriever(
        entity_embedding_dimension=4,
        hidden_dimension=128,
        gnn_layer_count=2,
        num_relations=2,
        num_bases=8,
        dropout=0.1,
    )
    run_directory = tmp_path / "models" / "1_rgcn"
    run_directory.mkdir(parents=True)
    torch.save(model.state_dict(), run_directory / "gnn_answer_retriever.pt")
    (run_directory / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "rgcn",
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

    loaded = GnnAnswerRetrieverModelRunService().load_run(
        model_root=tmp_path / "models",
        run_name=None,
        run_number=1,
        pipeline_configuration=_pipeline_config(options),
        device="cpu",
    )

    assert loaded.relation_vocabulary == vocabulary
    assert loaded.config.resolved_use_reverse_edges
    assert loaded.model.num_relations == 2


def test_rgcn_model_artifact_rejects_tampered_relation_vocabulary(tmp_path) -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    run_directory = tmp_path / "models" / "1_rgcn"
    run_directory.mkdir(parents=True)
    (run_directory / "gnn_answer_retriever.pt").touch()
    (run_directory / "model_config.json").write_text(
        json.dumps(
            {
                "dataset_id": "WebQSP",
                "gnn_architecture": "rgcn",
                "gnn_architecture_options": {
                    "gnn_layer_count": 2,
                    "gnn_hidden_dimension": 128,
                    "dropout": 0.1,
                    "num_bases": 8,
                },
                "gnn_architecture_context": build_relation_architecture_context(
                    vocabulary
                ),
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 4,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "relation_vocabulary.json").write_text(
        json.dumps({"parent": 1, "reverse__parent": 0}),
        encoding="utf-8",
    )

    with pytest.raises(
        GnnAnswerRetrieverModelRunException,
        match="relation vocabulary is invalid",
    ):
        GnnAnswerRetrieverModelRunService().resolve_run(
            model_root=tmp_path / "models",
            run_name=None,
            run_number=1,
        )


def test_rgcn_wandb_model_config_and_tag_are_canonical() -> None:
    vocabulary = {"parent": 0, "reverse__parent": 1}
    normalized = normalize_model_config(
        {
            "dataset_id": "WebQSP",
            "gnn_architecture": "rgcn",
            "gnn_architecture_options": {
                "gnn_layer_count": 2,
                "gnn_hidden_dimension": 256,
                "dropout": 0.1,
                "num_bases": 30,
            },
            "gnn_architecture_context": build_relation_architecture_context(
                vocabulary
            ),
            "embedding_model": "text-embedding-3-small",
            "training": {"epochs": 2, "loss_history": []},
        }
    )

    assert normalized["gnn_architecture"] == "rgcn"
    assert normalized["gnn_architecture_options"]["num_bases"] == 30
    assert normalized["gnn_architecture_context"]["relation_type_count"] == 2
    assert "gnn_architecture" not in normalized["training"]
    tags = WandbExperimentCoordinator._build_tags_from_config(
        {"configs": {"model": normalized}}
    )
    assert "rgcn" in tags
