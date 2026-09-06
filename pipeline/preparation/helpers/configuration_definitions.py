"""Typed built-in configuration definitions for pipeline construction."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field


class LlmModelDefinition(BaseModel):
    """Typed definition of an available LLM model option."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Stable model identifier.")
    display_name: str = Field(..., description="Human-readable model name.")
    description: str = Field(..., description="Short description of the model.")
    provider_id: str = Field(default="openai", description="Owning LLM provider id.")


class LlmProviderDefinition(BaseModel):
    """Typed definition of an LLM inference provider."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    display_name: str
    description: str
    accepts_arbitrary_models: bool = False


class SubgraphConstructionDefinition(BaseModel):
    """Typed definition of an available subgraph construction algorithm."""

    model_config = ConfigDict(frozen=True)

    algorithm_id: str = Field(..., description="Stable algorithm identifier.")
    display_name: str = Field(..., description="Human-readable algorithm name.")
    description: str = Field(..., description="Short description of the algorithm.")


class PcstEdgeCostStrategyDefinition(BaseModel):
    """Typed definition of a PCST edge-cost strategy."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    display_name: str
    description: str


class ContextConstructionDefinition(BaseModel):
    """Typed definition of an available context construction strategy."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str = Field(..., description="Stable context strategy identifier.")
    display_name: str = Field(..., description="Human-readable context strategy name.")
    description: str = Field(..., description="Short description of the context strategy.")


class GnnLayerCountDefinition(BaseModel):
    """Typed definition of an available GNN layer count option."""

    model_config = ConfigDict(frozen=True)

    layer_count: int = Field(..., description="Number of GNN message-passing layers.")
    display_name: str = Field(..., description="Human-readable layer count label.")
    description: str = Field(..., description="Short description of the layer setting.")


class GnnHiddenDimensionDefinition(BaseModel):
    """Typed definition of an available GNN hidden dimension option."""

    model_config = ConfigDict(frozen=True)

    hidden_dimension: int = Field(
        ...,
        description="Width of projected node states inside the GNN.",
    )
    display_name: str = Field(..., description="Human-readable hidden dimension label.")
    description: str = Field(
        ...,
        description="Short description of the hidden dimension setting.",
    )


class NodeClassifierDefinition(BaseModel):
    """Typed definition of an available node classifier option."""

    model_config = ConfigDict(frozen=True)

    classifier_id: str = Field(..., description="Stable node classifier identifier.")
    display_name: str = Field(..., description="Human-readable classifier name.")
    description: str = Field(..., description="Short description of the classifier.")


class GnnArchitectureOptionDefinition(BaseModel):
    """One declarative CLI/interactive option owned by a GNN architecture."""

    model_config = ConfigDict(frozen=True)

    option_id: str
    display_name: str
    description: str
    value_type: Literal["boolean", "integer", "float", "string"]
    choices: tuple[Any, ...] = ()
    default: Any = None
    cli_flag: str
    prompt_when_missing: bool = True
    enabled_when_option: str | None = None
    enabled_when_value: Any = True


class GnnArchitectureDefinition(BaseModel):
    """Self-contained schema and lazy model hook for one GNN architecture."""

    model_config = ConfigDict(frozen=True)

    architecture_id: str
    display_name: str
    description: str
    options: tuple[GnnArchitectureOptionDefinition, ...]
    model_builder_path: str
    validator_path: str | None = None
    runtime_strategy_path: str = (
        "pipeline.preparation.services.gnn_architecture_runtime:DefaultGnnRuntimeStrategy"
    )
    data_requirements: "GnnArchitectureDataRequirements" = Field(
        default_factory=lambda: GnnArchitectureDataRequirements()
    )

    @property
    def option_map(self) -> dict[str, GnnArchitectureOptionDefinition]:
        return {option.option_id: option for option in self.options}


class GnnArchitectureDataRequirements(BaseModel):
    """Architecture-owned graph and embedding input requirements."""

    model_config = ConfigDict(frozen=True)

    requires_reverse_edges: bool = False
    uses_entity_embeddings: bool = True
    uses_question_embeddings: bool = True
    uses_relation_embeddings: bool = True
    uses_relation_types: bool = False
    uses_raw_question_tokens: bool = False
    uses_relation_text_tokens: bool = False
    uses_seed_distributions: bool = False


class OpenAiEmbeddingModelDefinition(BaseModel):
    """Typed definition of an available OpenAI embedding model option."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Stable OpenAI embedding model identifier.")
    display_name: str = Field(..., description="Human-readable embedding model name.")
    dimensions: int = Field(..., description="Default embedding vector dimension.")
    description: str = Field(..., description="Short description of the embedding model.")


SHARED_LLM_MODELS: Final[dict[str, LlmModelDefinition]] = {
    "gpt-5.4": LlmModelDefinition(
        model_id="gpt-5.4",
        display_name="GPT-5.4",
        description="Highest-capability shared model option for primary reasoning.",
    ),
    "gpt-5.6-luna": LlmModelDefinition(
        model_id="gpt-5.6-luna",
        display_name="GPT-5.6 Luna",
        description="OpenAI GPT-5.6 Luna model for primary reasoning and generation.",
    ),
    "gpt-5.4-mini": LlmModelDefinition(
        model_id="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        description="Balanced shared model option for supporting reasoning tasks.",
    ),
    "gpt-5.4-nano": LlmModelDefinition(
        model_id="gpt-5.4-nano",
        display_name="GPT-5.4 Nano",
        description="Fast lightweight shared model option.",
    ),
    "gpt-5-mini": LlmModelDefinition(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        description="Faster, cost-efficient GPT-5 model for well-defined tasks.",
    ),
    "gpt-5-nano": LlmModelDefinition(
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        description="Fastest and cheapest GPT-5 model option.",
    ),
    "deepseek-v4-flash": LlmModelDefinition(
        model_id="deepseek-v4-flash",
        display_name="DeepSeek-V4-Flash",
        description="Fast DeepSeek V4 model served through the DeepSeek API.",
        provider_id="deepseek",
    ),
    "deepseek-v4-pro": LlmModelDefinition(
        model_id="deepseek-v4-pro",
        display_name="DeepSeek-V4-Pro",
        description="Higher-capability DeepSeek V4 model served through the DeepSeek API.",
        provider_id="deepseek",
    ),
    "gpt-4.1": LlmModelDefinition(
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        description="Strong shared model option for reasoning and generation.",
    ),
    "gpt-4.1-mini": LlmModelDefinition(
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        description="Compact shared model option for support tasks.",
    ),
    "gpt-4.1-nano": LlmModelDefinition(
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        description="Smallest shared model option for lightweight tasks.",
    ),
}

LLM_PROVIDERS: Final[dict[str, LlmProviderDefinition]] = {
    "openai": LlmProviderDefinition(
        provider_id="openai",
        display_name="OpenAI",
        description="Use an OpenAI-hosted chat model.",
    ),
    "deepseek": LlmProviderDefinition(
        provider_id="deepseek",
        display_name="DeepSeek",
        description="Use a DeepSeek-hosted chat model.",
    ),
    "vezilka": LlmProviderDefinition(
        provider_id="vezilka",
        display_name="Vezilka",
        description="Use any model exposed by the FINKI vLLM chat-completions endpoint.",
        accepts_arbitrary_models=True,
    ),
}

RECOMMENDED_LLM_PROVIDER_ID: Final[str] = "openai"

SUBGRAPH_CONSTRUCTION_ALGORITHMS: Final[dict[str, SubgraphConstructionDefinition]] = {
    "shortest_path": SubgraphConstructionDefinition(
        algorithm_id="shortest_path",
        display_name="Shortest Path",
        description="Build a question-specific subgraph using shortest paths between relevant nodes.",
    ),
    "pcst": SubgraphConstructionDefinition(
        algorithm_id="pcst",
        display_name="Prize-Collecting Steiner Tree (PCST)",
        description="Optimize a compact rooted subgraph that balances candidate prizes and edge costs.",
    ),
}

PCST_EDGE_COST_STRATEGIES: Final[dict[str, PcstEdgeCostStrategyDefinition]] = {
    "constant": PcstEdgeCostStrategyDefinition(
        strategy_id="constant",
        display_name="Constant edge cost",
        description="Charge the configured lambda for every selected edge.",
    ),
    "semantic": PcstEdgeCostStrategyDefinition(
        strategy_id="semantic",
        display_name="Semantic cosine edge cost",
        description="Make question-relevant relations cheaper using cosine distance.",
    ),
}

RECOMMENDED_PCST_EDGE_COST_STRATEGY_ID: Final[str] = "constant"
DEFAULT_PCST_EDGE_COST: Final[float] = 1.0

CONTEXT_CONSTRUCTION_STRATEGIES: Final[dict[str, ContextConstructionDefinition]] = {

    "structured_triples": ContextConstructionDefinition(
        strategy_id="structured_triples",
        display_name="Structured Triples",
        description="Represent the subgraph as <Head, Relation, Tail> triples.",
    ),
    # Not Supported Yet
    # "textualized": ContextConstructionDefinition(
    #     strategy_id="textualized",
    #     display_name="Textualized",
    #     description="Represent the subgraph as natural language descriptions for the LLM.",
    # ),
}

GNN_LAYER_COUNT_OPTIONS: Final[dict[str, GnnLayerCountDefinition]] = {
    "2": GnnLayerCountDefinition(
        layer_count=2,
        display_name="2 GNN layers",
        description="A compact message-passing depth for the first retriever baseline.",
    ),
    "3": GnnLayerCountDefinition(
        layer_count=3,
        display_name="3 GNN layers",
        description="A deeper message-passing depth for slightly broader local context.",
    ),
}

GNN_HIDDEN_DIMENSION_OPTIONS: Final[dict[str, GnnHiddenDimensionDefinition]] = {
    "128": GnnHiddenDimensionDefinition(
        hidden_dimension=128,
        display_name="128 hidden dimensions",
        description="A lightweight hidden size for faster early experiments.",
    ),
    "256": GnnHiddenDimensionDefinition(
        hidden_dimension=256,
        display_name="256 hidden dimensions",
        description="A balanced hidden size for the first retriever baseline.",
    ),
    "512": GnnHiddenDimensionDefinition(
        hidden_dimension=512,
        display_name="512 hidden dimensions",
        description="A wider hidden size with more capacity and higher compute cost.",
    ),
}

NODE_CLASSIFIERS: Final[dict[str, NodeClassifierDefinition]] = {
    "mlp": NodeClassifierDefinition(
        classifier_id="mlp",
        display_name="MLP node classifier",
        description="A two-layer classifier over final node hidden states.",
    ),
    "linear": NodeClassifierDefinition(
        classifier_id="linear",
        display_name="Linear node classifier",
        description="A single linear classifier over final node hidden states.",
    ),
}

GRAPH_SAGE_ARCHITECTURE_ID: Final[str] = "graphsage"
AA_GRAPH_SAGE_ARCHITECTURE_ID: Final[str] = "aa-graphsage"
RGCN_ARCHITECTURE_ID: Final[str] = "rgcn"
HGT_ARCHITECTURE_ID: Final[str] = "hgt"
REAREV_ARCHITECTURE_ID: Final[str] = "rearev"
NBFNET_ARCHITECTURE_ID: Final[str] = "nbfnet"

GNN_LAYER_COUNT_OPTION: Final[GnnArchitectureOptionDefinition] = (
    GnnArchitectureOptionDefinition(
        option_id="gnn_layer_count",
        display_name="GNN Layer Count",
        description="Number of GNN message-passing layers.",
        value_type="integer",
        choices=(2, 3),
        default=2,
        cli_flag="--gnn-layers",
    )
)
GNN_HIDDEN_DIMENSION_OPTION: Final[GnnArchitectureOptionDefinition] = (
    GnnArchitectureOptionDefinition(
        option_id="gnn_hidden_dimension",
        display_name="GNN Hidden Dimension",
        description="Width of projected node states inside the GNN.",
        value_type="integer",
        choices=(128, 256, 512),
        default=256,
        cli_flag="--gnn-hidden-dim",
    )
)
NODE_CLASSIFIER_OPTION: Final[GnnArchitectureOptionDefinition] = (
    GnnArchitectureOptionDefinition(
        option_id="node_classifier",
        display_name="Node Classifier",
        description="Classifier used after the final GNN layer.",
        value_type="string",
        choices=("mlp", "linear"),
        default="mlp",
        cli_flag="--node-classifier",
    )
)
GNN_DROPOUT_OPTION: Final[GnnArchitectureOptionDefinition] = (
    GnnArchitectureOptionDefinition(
        option_id="dropout",
        display_name="GNN Dropout",
        description="Dropout used by the architecture.",
        value_type="float",
        choices=(0.0, 0.1, 0.2, 0.3, 0.5),
        default=0.1,
        cli_flag="--dropout",
    )
)

GNN_SHARED_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GNN_LAYER_COUNT_OPTION,
    GNN_HIDDEN_DIMENSION_OPTION,
    NODE_CLASSIFIER_OPTION,
    GNN_DROPOUT_OPTION,
)

AA_GRAPH_SAGE_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GnnArchitectureOptionDefinition(
        option_id="use_edge_mlp", display_name="Use Edge MLP",
        description="Use a trainable question-relation edge scorer.",
        value_type="boolean", default=True, cli_flag="--use-edge-mlp",
    ),
    GnnArchitectureOptionDefinition(
        option_id="use_reverse_edges", display_name="Use Reverse Edges",
        description="Materialize reverse graph edges.",
        value_type="boolean", default=True, cli_flag="--use-reverse-edges",
    ),
    GnnArchitectureOptionDefinition(
        option_id="question_aware_classifier", display_name="Question-Aware Classifier",
        description="Condition node classification on the question embedding.",
        value_type="boolean", default=True, cli_flag="--question-aware-classifier",
    ),
    GnnArchitectureOptionDefinition(
        option_id="add_layer_normalization", display_name="Add Layer Normalization",
        description="Use residual LayerNorm message-passing blocks.",
        value_type="boolean", default=True, cli_flag="--add-layer-normalization",
    ),
    GnnArchitectureOptionDefinition(
        option_id="edge_mlp_hidden_dim", display_name="Edge MLP Hidden Dimension",
        description="Hidden width of the relation-aware edge MLP.",
        value_type="integer", choices=(128, 256, 512), default=256,
        cli_flag="--edge-mlp-hidden-dim", enabled_when_option="use_edge_mlp",
    ),
)

RGCN_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GNN_LAYER_COUNT_OPTION,
    GNN_HIDDEN_DIMENSION_OPTION,
    GNN_DROPOUT_OPTION,
    GnnArchitectureOptionDefinition(
        option_id="num_bases",
        display_name="R-GCN Basis Count",
        description="Number of shared basis matrices used for relation transforms.",
        value_type="integer",
        choices=(8, 16, 30, 64),
        default=30,
        cli_flag="--num-bases",
    ),
)

HGT_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GNN_LAYER_COUNT_OPTION,
    GNN_HIDDEN_DIMENSION_OPTION,
    GNN_DROPOUT_OPTION,
    GnnArchitectureOptionDefinition(
        option_id="attention_heads",
        display_name="HGT Attention Heads",
        description="Number of heterogeneous attention heads.",
        value_type="integer",
        choices=(1, 2, 4, 8),
        default=8,
        cli_flag="--attention-heads",
    ),
)

REAREV_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GnnArchitectureOptionDefinition(
        option_id="gnn_hidden_dimension",
        display_name="ReaRev Hidden Dimension",
        description="Width of instructions, relation states, and node reasoning states.",
        value_type="integer",
        choices=(50, 128, 256, 512),
        default=50,
        cli_flag="--gnn-hidden-dim",
    ),
    GNN_DROPOUT_OPTION,
    GnnArchitectureOptionDefinition(
        option_id="num_instructions",
        display_name="ReaRev Instructions",
        description="Number of token-attended reasoning instructions.",
        value_type="integer",
        choices=(1, 2, 3),
        default=2,
        cli_flag="--num-instructions",
    ),
    GnnArchitectureOptionDefinition(
        option_id="reasoning_steps",
        display_name="ReaRev Reasoning Steps",
        description="BFS-style reasoning steps executed in each adaptive stage.",
        value_type="integer",
        choices=(1, 2, 3),
        default=2,
        cli_flag="--reasoning-steps",
    ),
    GnnArchitectureOptionDefinition(
        option_id="adaptive_iterations",
        display_name="ReaRev Adaptive Iterations",
        description="Reason-and-revise stages used for adaptive reasoning.",
        value_type="integer",
        choices=(1, 2, 3),
        default=3,
        cli_flag="--adaptive-iterations",
    ),
)

NBFNET_OPTIONS: Final[tuple[GnnArchitectureOptionDefinition, ...]] = (
    GnnArchitectureOptionDefinition(
        option_id="gnn_layer_count",
        display_name="NBFNet Layer Count",
        description="Number of Neural Bellman-Ford propagation layers.",
        value_type="integer",
        choices=(2, 3, 4, 6),
        default=3,
        cli_flag="--gnn-layers",
    ),
    GnnArchitectureOptionDefinition(
        option_id="gnn_hidden_dimension",
        display_name="NBFNet Hidden Dimension",
        description="Width of query-conditioned path representations.",
        value_type="integer",
        choices=(32, 64, 128, 256),
        default=32,
        cli_flag="--gnn-hidden-dim",
    ),
)

GNN_ARCHITECTURES: Final[dict[str, GnnArchitectureDefinition]] = {
    GRAPH_SAGE_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=GRAPH_SAGE_ARCHITECTURE_ID,
        display_name="GraphSAGE",
        description="Baseline GraphSAGE with configurable depth, width, classifier, and dropout.",
        options=GNN_SHARED_OPTIONS,
        model_builder_path=(
            "pipeline.preparation.models.gnn_answer_retriever:build_graphsage_model"
        ),
    ),
    AA_GRAPH_SAGE_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=AA_GRAPH_SAGE_ARCHITECTURE_ID,
        display_name="Advance GraphSAGE",
        description="Advanced answer-aware GraphSAGE with relational and question-aware components.",
        options=(*GNN_SHARED_OPTIONS, *AA_GRAPH_SAGE_OPTIONS),
        model_builder_path=(
            "pipeline.preparation.models.gnn_answer_retriever:build_aa_graphsage_model"
        ),
        validator_path=(
            "pipeline.preparation.helpers.gnn_architecture:validate_aa_graphsage_options"
        ),
    ),
    RGCN_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=RGCN_ARCHITECTURE_ID,
        display_name="R-GCN",
        description=(
            "Basis-decomposed relational graph convolution with mandatory inverse relations."
        ),
        options=RGCN_OPTIONS,
        model_builder_path=(
            "pipeline.preparation.models.rgcn_answer_retriever:build_rgcn_model"
        ),
        data_requirements=GnnArchitectureDataRequirements(
            requires_reverse_edges=True,
            uses_question_embeddings=False,
            uses_relation_embeddings=False,
            uses_relation_types=True,
        ),
    ),
    HGT_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=HGT_ARCHITECTURE_ID,
        display_name="HGT",
        description=(
            "Relation-aware heterogeneous multi-head attention with one entity node type."
        ),
        options=HGT_OPTIONS,
        model_builder_path=(
            "pipeline.preparation.models.hgt_answer_retriever:build_hgt_model"
        ),
        validator_path=(
            "pipeline.preparation.helpers.gnn_architecture:validate_hgt_options"
        ),
        data_requirements=GnnArchitectureDataRequirements(
            requires_reverse_edges=True,
            uses_question_embeddings=False,
            uses_relation_embeddings=False,
            uses_relation_types=True,
        ),
    ),
    REAREV_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=REAREV_ARCHITECTURE_ID,
        display_name="ReaRev",
        description=(
            "Question-conditioned adaptive reason-and-revise execution over "
            "relation-aware knowledge graphs."
        ),
        options=REAREV_OPTIONS,
        model_builder_path=(
            "pipeline.preparation.models.rearev_answer_retriever:build_rearev_model"
        ),
        runtime_strategy_path=(
            "pipeline.preparation.services.gnn_architecture_runtime:ReaRevRuntimeStrategy"
        ),
        data_requirements=GnnArchitectureDataRequirements(
            requires_reverse_edges=True,
            uses_entity_embeddings=False,
            uses_question_embeddings=False,
            uses_relation_embeddings=False,
            uses_relation_types=True,
            uses_raw_question_tokens=True,
            uses_relation_text_tokens=True,
            uses_seed_distributions=True,
        ),
    ),
    NBFNET_ARCHITECTURE_ID: GnnArchitectureDefinition(
        architecture_id=NBFNET_ARCHITECTURE_ID,
        display_name="NBFNet",
        description=(
            "Question-conditioned Neural Bellman-Ford path reasoning with "
            "DistMult messages and PNA aggregation."
        ),
        options=NBFNET_OPTIONS,
        model_builder_path=(
            "pipeline.preparation.models.nbfnet_answer_retriever:build_nbfnet_model"
        ),
        runtime_strategy_path=(
            "pipeline.preparation.services.gnn_architecture_runtime:NBFNetRuntimeStrategy"
        ),
        data_requirements=GnnArchitectureDataRequirements(
            requires_reverse_edges=True,
            uses_entity_embeddings=False,
            uses_question_embeddings=True,
            uses_relation_embeddings=False,
            uses_relation_types=True,
            uses_seed_distributions=True,
        ),
    ),
}

OPENAI_EMBEDDING_MODELS: Final[dict[str, OpenAiEmbeddingModelDefinition]] = {
    "text-embedding-3-small": OpenAiEmbeddingModelDefinition(
        model_id="text-embedding-3-small",
        display_name="text-embedding-3-small",
        dimensions=1536,
        description="Small OpenAI embedding model for cost-efficient text embeddings.",
    ),
    "text-embedding-3-large": OpenAiEmbeddingModelDefinition(
        model_id="text-embedding-3-large",
        display_name="text-embedding-3-large",
        dimensions=3072,
        description="Most capable OpenAI embedding model for text embeddings.",
    )
}

RECOMMENDED_MAIN_LLM_MODEL_ID: Final[str] = "gpt-4.1-nano"
RECOMMENDED_SUBGRAPH_CONSTRUCTION_ALGORITHM_ID: Final[str] = "shortest_path"
RECOMMENDED_CONTEXT_CONSTRUCTION_STRATEGY_ID: Final[str] = "structured_triples"
RECOMMENDED_GNN_LAYER_COUNT: Final[int] = 2
RECOMMENDED_GNN_HIDDEN_DIMENSION: Final[int] = 256
RECOMMENDED_NODE_CLASSIFIER_ID: Final[str] = "mlp"
RECOMMENDED_GNN_ARCHITECTURE_ID: Final[str] = GRAPH_SAGE_ARCHITECTURE_ID
RECOMMENDED_QUESTION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_RELATION_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
RECOMMENDED_ENTITY_EMBEDDING_MODEL_ID: Final[str] = "text-embedding-3-small"
