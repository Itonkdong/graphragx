"""NBFNet registry, operator, batching, preparation, and persistence tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import main

torch = pytest.importorskip("torch")

from pipeline.abstract import StepContext
from pipeline.evaluation.models import PreparedGnnEvaluationData, PreparedGnnEvaluationInstance
from pipeline.evaluation.services.model_config_normalization import normalize_model_config
from pipeline.evaluation.services.gnn_retriever_results import GnnRetrieverResultsService
from pipeline.evaluation.services.wandb_experiment import WandbExperimentCoordinator
from pipeline.preparation.exceptions import InvalidGnnArchitectureConfigurationException
from pipeline.preparation.helpers.configuration_definitions import GNN_ARCHITECTURES
from pipeline.preparation.helpers.gnn_architecture import architecture_defaults
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.models.nbfnet_answer_retriever import (
    NBFNetAnswerRetriever,
    NeuralBellmanFordLayer,
)
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
)
from pipeline.preparation.services.gnn_answer_retriever_evaluation import (
    GnnAnswerRetrieverEvaluationService,
)
from pipeline.preparation.services.gnn_architecture_runtime import NBFNetRuntimeStrategy
from pipeline.preparation.services.gnn_evaluation_data_preparation import (
    GnnEvaluationDataPreparationService,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
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


def _options(**overrides):
    return {
        "gnn_layer_count": 2,
        "gnn_hidden_dimension": 4,
        **overrides,
    }


def _pipeline_config(options=None):
    return BuiltPipelineConfiguration(
        dataset_id="WebQSP",
        gnn_architecture="nbfnet",
        gnn_architecture_options=options or _options(),
        main_llm_model="gpt-5.4",
        subgraph_construction_algorithm="shortest_path",
        context_construction_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        entity_embedding_model="text-embedding-3-small",
        use_reverse_edges=True,
    )


def _instance(*, question="question", seed=True, answer=True):
    return WebQSPProcessedInstance(
        question=question,
        q_entity=["A"] if seed else ["missing"],
        a_entity=["B"],
        nodes=["A", "B", "C"],
        node2id={"A": 0, "B": 1, "C": 2},
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_relations=["parent", "reverse__parent"],
        node_labels=torch.tensor([0.0, 1.0 if answer else 0.0, 0.0]),
    )


def test_nbfnet_registry_defaults_and_cli_contract():
    definition = GNN_ARCHITECTURES["nbfnet"]
    assert definition.display_name == "NBFNet"
    assert architecture_defaults("nbfnet") == {
        "gnn_architecture": "nbfnet",
        "gnn_layer_count": 3,
        "gnn_hidden_dimension": 32,
    }
    assert definition.data_requirements.requires_reverse_edges
    assert definition.data_requirements.uses_relation_types
    assert definition.data_requirements.uses_question_embeddings
    assert definition.data_requirements.uses_seed_distributions
    assert not definition.data_requirements.uses_entity_embeddings
    assert not definition.data_requirements.uses_relation_embeddings

    args = main.build_parser().parse_args(
        ["--gnn-architecture", "nbfnet", "--gnn-layers", "6", "--gnn-hidden-dim", "64"]
    )
    assert args.gnn_layer_count == 6
    assert args.gnn_hidden_dimension == 64


def test_nbfnet_default_configuration_uses_embedding_and_reverse_edges():
    options = architecture_defaults("nbfnet")
    options.pop("gnn_architecture")
    result = BuildPipelineConfigurationStep(
        gnn_architecture="nbfnet",
        gnn_options=options,
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        embedding_model="text-embedding-3-small",
    ).execute(_dataset_context())
    assert result.gnn_architecture_options == options
    assert result.embedding_model == "text-embedding-3-small"
    assert result.use_reverse_edges
    assert result.node_classifier is None
    assert result.dropout == 0.0


@pytest.mark.parametrize(
    ("option_id", "value"),
    [
        ("dropout", 0.1),
        ("node_classifier", "mlp"),
        ("num_bases", 8),
        ("attention_heads", 2),
        ("num_instructions", 2),
        ("reasoning_steps", 2),
        ("adaptive_iterations", 2),
        ("use_reverse_edges", True),
        ("use_reverse_edges", False),
        ("use_edge_mlp", True),
        ("question_aware_classifier", True),
        ("add_layer_normalization", True),
        ("edge_mlp_hidden_dim", 128),
    ],
)
def test_nbfnet_rejects_other_architecture_options(option_id, value):
    options = architecture_defaults("nbfnet")
    options.pop("gnn_architecture")
    with pytest.raises(
        InvalidGnnArchitectureConfigurationException,
        match=f"does not support: {option_id}",
    ):
        BuildPipelineConfigurationStep(
            gnn_architecture="nbfnet",
            gnn_options={**options, option_id: value},
        ).execute(_dataset_context())


def test_nbfnet_layer_matches_literal_pna_reference():
    layer = NeuralBellmanFordLayer(hidden_dimension=2, num_relations=2)
    with torch.no_grad():
        layer.relation_weight.zero_()
        layer.relation_weight[0].copy_(torch.eye(2))
        layer.relation_weight[1].copy_(2 * torch.eye(2))
        layer.relation_bias.zero_()
    states = torch.tensor([[1.0, 2.0], [0.5, 1.0], [0.0, 0.0]])
    boundary = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    query = torch.tensor([[1.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_type = torch.tensor([0, 1])
    node_graph = torch.zeros(3, dtype=torch.long)
    relation_vectors = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    messages = states[edge_index[0]] * relation_vectors[edge_type]
    degree = torch.tensor([1.0, 1.0, 3.0])
    mean_log = degree.log().mean().reshape(1)
    expected_pna = layer._pna_aggregate(
        messages=messages,
        target_nodes=edge_index[1],
        boundary=boundary,
        node_degree=degree,
        node_graph_index=node_graph,
        graph_mean_log_degree=mean_log,
    )
    captured = {}
    hook = layer.output_projection.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("combined", inputs[0].detach())
    )
    layer(
        states,
        boundary,
        query,
        edge_index,
        edge_type,
        node_graph,
    )
    hook.remove()
    torch.testing.assert_close(captured["combined"][:, 2:], expected_pna)


def test_nbfnet_active_pair_forward_and_gradients():
    model = NBFNetAnswerRetriever(
        question_embedding_dimension=4,
        hidden_dimension=4,
        gnn_layer_count=2,
        num_relations=3,
    )
    inputs = {
        "question_features": torch.randn(1, 4),
        "edge_index": torch.tensor([[0, 1], [1, 2]]),
        "edge_type": torch.tensor([0, 2]),
        "node_graph_index": torch.zeros(3, dtype=torch.long),
        "seed_node_index": torch.tensor([0]),
        "graph_count": 1,
    }
    logits = model(**inputs)
    assert logits.shape == (3,)
    logits.sum().backward()
    assert model.question_projection.weight.grad is not None
    assert model.layers[0].relation_weight.grad is not None
    assert torch.isfinite(model.layers[0].relation_weight.grad).all()


def test_nbfnet_disconnected_batch_matches_individual_execution(tmp_path):
    model = NBFNetAnswerRetriever(
        question_embedding_dimension=4,
        hidden_dimension=4,
        gnn_layer_count=2,
        num_relations=2,
    ).eval()
    instances = []
    for index in range(2):
        instance = _instance(question=f"q{index}")
        instances.append(
            PreparedGnnTrainingInstance(
                source_instance_index=index,
                question_embedding_index=index,
                edge_index=instance.edge_index,
                edge_type=torch.tensor([0, 1]),
                node_labels=instance.node_labels,
                seed_node_indices=torch.tensor([0]),
            )
        )
    built = _built({"parent": 0, "reverse__parent": 1})
    prepared = PreparedGnnTrainingData(
        built_retriever=built,
        instances=instances,
        question_embeddings=torch.randn(2, 4),
        training_start_instance=0,
        training_end_instance=2,
        selected_device="cpu",
        embedding_cache_device="cpu",
        embedding_cache_dtype="float32",
        cache_root=tmp_path,
    )
    strategy = NBFNetRuntimeStrategy()
    batched = strategy.build_training_batches(
        prepared_data=prepared, batch_size=2, torch=torch, device="cpu"
    )[0]
    batched_logits = model(**strategy.model_inputs(batched))
    individual_logits = []
    individual_losses = []
    for item in instances:
        single_data = prepared.model_copy(update={"instances": [item]})
        batch = strategy.build_training_batches(
            prepared_data=single_data, batch_size=1, torch=torch, device="cpu"
        )[0]
        logits = model(**strategy.model_inputs(batch))
        individual_logits.append(logits)
        individual_losses.append(strategy.compute_loss(logits, batch))
    torch.testing.assert_close(batched_logits, torch.cat(individual_logits), atol=1e-5, rtol=1e-5)
    loss = strategy.compute_loss(batched_logits, batched)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.stack(individual_losses).mean())


class _EmbeddingService:
    def __init__(self):
        self.node_calls = 0
        self.relation_calls = 0
        self.question_calls = 0

    def load_node_cache(self, **_):
        self.node_calls += 1
        raise AssertionError("NBFNet must not load entity embeddings")

    def load_relation_cache(self, **_):
        self.relation_calls += 1
        raise AssertionError("NBFNet must not load relation embeddings")

    def load_question_cache(self, **_):
        self.question_calls += 1
        return SimpleNamespace(vector_size=4, cache_kind="questions")


class _TensorCacheService:
    def load_matrix(self, *, texts, dtype, device, **_):
        return torch.arange(len(texts) * 4, dtype=dtype, device=device).reshape(-1, 4)


def _built(vocabulary):
    return BuiltGnnAnswerRetriever(
        dataset_id="WebQSP",
        gnn_architecture="nbfnet",
        gnn_architecture_options=_options(),
        gnn_architecture_context=build_relation_architecture_context(vocabulary),
        relation_vocabulary=vocabulary,
        question_embedding_dimension=4,
        hidden_dimension=4,
        gnn_layer_count=2,
        use_reverse_edges=True,
        model=NBFNetAnswerRetriever(
            question_embedding_dimension=4,
            hidden_dimension=4,
            gnn_layer_count=2,
            num_relations=len(vocabulary),
        ),
    )


def test_nbfnet_preparation_loads_only_questions_and_skips_seedless_training(tmp_path):
    vocabulary = {"parent": 0, "reverse__parent": 1}
    valid = _instance(question="valid")
    seedless = _instance(question="seedless", seed=False)
    dataset = PreparedWebQSPGraphDataset(
        dataset_id="WebQSP",
        processing_version="test",
        use_reverse_edges=True,
        train_instances=[valid, seedless],
        test_instances=[valid, seedless],
        vocabulary_store=WebQSPVocabularyStore(
            nodes={"A": 0, "B": 1, "C": 2},
            relations=vocabulary,
            questions={"valid": 0, "seedless": 1},
        ),
        cache_directory=tmp_path / "processed_reverse_edges",
    )
    embeddings = _EmbeddingService()
    service = GnnTrainingDataPreparationService(
        embedding_cache_service=embeddings,
        embedding_tensor_cache_service=_TensorCacheService(),
    )
    prepared = service.prepare(
        built_retriever=_built(vocabulary),
        prepared_dataset=dataset,
        configuration=_pipeline_config(),
        preparation_config=GnnTrainingDataPreparationConfig(
            training_device="cpu",
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
        ),
    )
    assert embeddings.question_calls == 1
    assert embeddings.node_calls == 0
    assert embeddings.relation_calls == 0
    assert prepared.node_embeddings is None
    assert prepared.relation_embeddings is None
    assert prepared.question_embeddings.shape == (2, 4)
    assert len(prepared.instances) == 1
    assert prepared.instances[0].seed_node_indices.tolist() == [0]

    evaluation = GnnEvaluationDataPreparationService(
        embedding_cache_service=embeddings,
        embedding_tensor_cache_service=_TensorCacheService(),
    ).prepare(
        torch=torch,
        test_instances=dataset.test_instances,
        cache_root=tmp_path,
        dataset_id="WebQSP",
        entity_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        question_embedding_model="text-embedding-3-small",
        selected_device="cpu",
        evaluation_config=SimpleNamespace(
            gpu_cache_reserve_gb=0.0,
            embedding_cache_dtype="float32",
            embedding_cache_device="cpu",
        ),
        gnn_architecture="nbfnet",
        relation_vocabulary=vocabulary,
    )
    assert evaluation.instances[0].skip_reason is None
    assert evaluation.instances[1].skip_reason == "missing_question_entity"
    evaluation_batch = NBFNetRuntimeStrategy().build_evaluation_batch(
        prepared_data=evaluation,
        prepared_instance=evaluation.instances[0],
        torch=torch,
        device="cpu",
    )
    assert evaluation_batch.node_labels.tolist() == valid.node_labels.tolist()
    assert evaluation_batch.seed_node_index.tolist() == [0]


def test_nbfnet_skipped_evaluation_graph_remains_in_metric_denominator():
    prediction = GnnAnswerRetrieverEvaluationService._build_skipped_prediction(
        instance_index=7,
        instance=_instance(seed=False),
        global_node_vocabulary={"A": 0, "B": 1, "C": 2},
    )
    metrics = GnnRetrieverResultsService().build_metrics(
        dataset_id="WebQSP",
        model_run_name="1_model",
        model_run_number=1,
        predictions=[prediction],
        candidate_limit=10,
    )
    assert prediction.answer_candidates == []
    assert not prediction.hit_at_1
    assert metrics.evaluated_instances == 1
    assert metrics.hits_at_1 == 0.0


def test_nbfnet_saved_model_round_trip_and_wandb_config(tmp_path):
    vocabulary = {"parent": 0, "reverse__parent": 1}
    context = {
        **build_relation_architecture_context(vocabulary),
        "nbfnet_preprocessing_version": 1,
        "query_source": "pooled_question_embedding",
        "boundary_schema": "multi_source_full_query",
        "dependent_relations": True,
        "message_function": "distmult",
        "aggregate_function": "pna",
        "short_cut": True,
        "layer_normalization": True,
        "activation": "relu",
        "scorer": "two_layer_mlp",
    }
    model = NBFNetAnswerRetriever(
        question_embedding_dimension=4,
        hidden_dimension=32,
        gnn_layer_count=3,
        num_relations=2,
    )
    run_directory = tmp_path / "models" / "1_nbfnet"
    run_directory.mkdir(parents=True)
    torch.save(model.state_dict(), run_directory / "gnn_answer_retriever.pt")
    payload = {
        "dataset_id": "WebQSP",
        "gnn_architecture": "nbfnet",
        "gnn_architecture_options": {
            "gnn_layer_count": 3,
            "gnn_hidden_dimension": 32,
        },
        "gnn_architecture_context": context,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 4,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "estimated_training_parameter_bytes": sum(p.numel() for p in model.parameters()) * 16,
        "training": {"loss_function": "BCEWithLogitsLoss"},
    }
    (run_directory / "model_config.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_directory / "relation_vocabulary.json").write_text(
        json.dumps(vocabulary), encoding="utf-8"
    )
    loaded = GnnAnswerRetrieverModelRunService().load_run(
        model_root=tmp_path / "models",
        run_name="1_nbfnet",
        run_number=None,
        pipeline_configuration=_pipeline_config(
            {"gnn_layer_count": 3, "gnn_hidden_dimension": 32}
        ),
        device="cpu",
    )
    assert isinstance(loaded.model, NBFNetAnswerRetriever)
    assert loaded.relation_vocabulary == vocabulary
    normalized = normalize_model_config(payload)
    assert normalized["gnn_architecture_options"] == payload["gnn_architecture_options"]
    tags = WandbExperimentCoordinator._build_tags_from_config(
        {"configs": {"model": normalized}}
    )
    assert "nbfnet" in tags
    assert "text-embedding-3-small" in tags
