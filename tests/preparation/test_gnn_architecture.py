"""Tests for architecture registry defaults and legacy inference."""

from pipeline.preparation.helpers.gnn_architecture import (
    architecture_defaults,
    infer_gnn_architecture,
)
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    GnnArchitectureDefinition,
    GnnArchitectureOptionDefinition,
)
from pipeline.preparation.steps.configuration_building import (
    BuildPipelineConfigurationStep,
)
from pipeline.preparation.steps.dataset_selection import SelectedDataset
from pipeline.abstract import StepContext
import main


def test_architecture_defaults_are_stable() -> None:
    baseline = architecture_defaults("graphsage")
    advanced = architecture_defaults("aa-graphsage")

    assert baseline["gnn_layer_count"] == 2
    assert baseline["gnn_hidden_dimension"] == 256
    assert baseline["node_classifier"] == "mlp"
    assert baseline["dropout"] == 0.1
    assert "use_edge_mlp" not in baseline
    assert advanced["use_edge_mlp"]
    assert advanced["use_reverse_edges"]
    assert advanced["question_aware_classifier"]
    assert advanced["add_layer_normalization"]
    assert advanced["edge_mlp_hidden_dim"] == 256


def test_legacy_architecture_inference_ignores_dropout_and_unused_edge_width() -> None:
    assert infer_gnn_architecture(
        {"dropout": 0.1, "edge_mlp_hidden_dim": 512}
    ) == "graphsage"


def test_legacy_architecture_inference_detects_any_advanced_boolean() -> None:
    assert infer_gnn_architecture(
        {"training": {"use_reverse_edges": True}}
    ) == "aa-graphsage"


def test_explicit_architecture_takes_precedence_over_legacy_fields() -> None:
    assert infer_gnn_architecture(
        {"gnn_architecture": "graphsage", "use_edge_mlp": True}
    ) == "graphsage"


def test_new_architecture_options_drive_cli_and_configuration_without_core_changes(
    monkeypatch,
) -> None:
    architecture = GnnArchitectureDefinition(
        architecture_id="attention-gnn",
        display_name="Attention GNN",
        description="Test architecture with a completely different option.",
        options=(
            GnnArchitectureOptionDefinition(
                option_id="attention_heads",
                display_name="Attention Heads",
                description="Number of attention heads.",
                value_type="integer",
                choices=(4, 8, 16),
                default=8,
                cli_flag="--attention-heads",
            ),
        ),
        model_builder_path="builtins:dict",
    )
    monkeypatch.setitem(GNN_ARCHITECTURES, architecture.architecture_id, architecture)

    args = main.build_parser().parse_args(
        ["--gnn-architecture", "attention-gnn", "--attention-heads", "16"]
    )
    assert args.attention_heads == 16
    defaulted = main.PipelineRuntimeConfig(
        gnn_architecture="attention-gnn",
        use_default_config_values=True,
    ).with_defaulted_user_inputs()
    assert defaulted.gnn_options == {"attention_heads": 8}

    result = BuildPipelineConfigurationStep(
        gnn_architecture="attention-gnn",
        gnn_options={"attention_heads": args.attention_heads},
        main_llm_model="gpt-5.4",
        subgraph_algorithm="shortest_path",
        context_strategy="structured_triples",
        question_embedding_model="text-embedding-3-small",
        relation_embedding_model="text-embedding-3-small",
        entity_embedding_model="text-embedding-3-small",
    ).execute(
        StepContext(
            result=SelectedDataset(
                dataset_id="WebQSP",
                display_name="WebQSP",
                dataset_family="question_answering",
                task_domain="knowledge_graph_question_answering",
                description="dataset",
                supported=True,
            )
        )
    )

    assert result.gnn_architecture == "attention-gnn"
    assert result.gnn_architecture_options == {"attention_heads": 16}
    assert result.gnn_layer_count is None
    assert result.node_classifier is None
