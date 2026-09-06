"""Tests for preparation-time GNN answer-retriever construction."""

import builtins
import sys
import unittest
from unittest.mock import patch

from pipeline import (
    AbstractStep,
    BuildGnnAnswerRetrieverStep,
    BuiltGnnAnswerRetriever,
    BuiltPipelineConfiguration,
    LoadedDataset,
    PreparedWebQSPGraphDataset,
    Pipeline,
    PipelineException,
    StepContext,
    StepContextBuilder,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPVocabularyStore
from pathlib import Path


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


class FakeDatasetDict(dict):
    pass


class FakeConfigurationStep(AbstractStep[BuiltPipelineConfiguration, LoadedDataset]):
    def execute_default(
        self,
        context: StepContext[LoadedDataset],
    ) -> BuiltPipelineConfiguration:
        return BuildGnnAnswerRetrieverStepTests.make_configuration()


class FakeLoadedDatasetStep(
    AbstractStep[PreparedWebQSPGraphDataset, BuiltPipelineConfiguration]
):
    def execute_default(
        self,
        context: StepContext[BuiltPipelineConfiguration],
    ) -> PreparedWebQSPGraphDataset:
        return PreparedWebQSPGraphDataset(
            dataset_id="WebQSP",
            processing_version="test",
            train_instances=[],
            test_instances=[],
            vocabulary_store=WebQSPVocabularyStore(),
            cache_directory=Path("data/webqsp/processed"),
        )


class BuildGnnAnswerRetrieverStepTests(unittest.TestCase):
    @staticmethod
    def make_configuration(
        node_classifier: str = "mlp",
        gnn_architecture: str = "graphsage",
    ) -> BuiltPipelineConfiguration:
        return BuiltPipelineConfiguration(
            dataset_id="WebQSP",
            gnn_architecture=gnn_architecture,
            main_llm_model="gpt-5.4",
            subgraph_construction_algorithm="shortest_path",
            context_construction_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier=node_classifier,
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

    @staticmethod
    def make_loaded_dataset_context(
        builder: StepContextBuilder,
    ) -> StepContext[PreparedWebQSPGraphDataset]:
        prepared_dataset = PreparedWebQSPGraphDataset(
            dataset_id="WebQSP",
            processing_version="test",
            train_instances=[],
            test_instances=[],
            vocabulary_store=WebQSPVocabularyStore(),
            cache_directory=Path("data/webqsp/processed"),
        )
        return builder.create_context(
            result=prepared_dataset,
            next_step=BuildGnnAnswerRetrieverStep(),
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_builds_gnn_answer_retriever_from_loaded_dataset_and_stored_configuration(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration())
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result, BuiltGnnAnswerRetriever)
        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_dimension, 1536)
        self.assertEqual(result.hidden_dimension, 256)
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(len(result.model.gnn_layers), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_linear_classifier_builds_linear_head(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration(node_classifier="linear"))
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result.model.classifier, nn.Linear)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_mlp_classifier_builds_sequential_head(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration(node_classifier="mlp"))
        step = BuildGnnAnswerRetrieverStep()

        result = step.execute(self.make_loaded_dataset_context(builder))

        self.assertIsInstance(result.model.classifier, nn.Sequential)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_factory_builds_distinct_architecture_types(self) -> None:
        from pipeline.preparation.models.gnn_answer_retriever import (
            AdvancedGraphSageAnswerRetriever,
            GraphSageAnswerRetriever,
            build_gnn_answer_retriever,
        )

        shared = dict(
            entity_embedding_dimension=8,
            question_embedding_dimension=6,
            relation_embedding_dimension=6,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="mlp",
            dropout=0.1,
        )
        baseline = build_gnn_answer_retriever("graphsage", **shared)
        advanced = build_gnn_answer_retriever(
            "aa-graphsage",
            **shared,
            use_edge_mlp=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=128,
        )

        self.assertIsInstance(baseline, GraphSageAnswerRetriever)
        self.assertIsInstance(advanced, AdvancedGraphSageAnswerRetriever)
        self.assertFalse(baseline.use_edge_mlp)
        self.assertTrue(advanced.use_edge_mlp)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_model_forward_returns_one_logit_per_node(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(self.make_configuration())
        step = BuildGnnAnswerRetrieverStep()
        result = step.execute(self.make_loaded_dataset_context(builder))

        entity_features = torch.randn(4, result.entity_embedding_dimension)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_weight = torch.tensor([0.9, 0.5, 0.2])

        logits = result.model(entity_features, edge_index, edge_weight)

        self.assertEqual(tuple(logits.shape), (4,))

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_edge_mlp_question_aware_and_layer_norm_forward(self) -> None:
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        model = GnnAnswerRetriever(
            entity_embedding_dimension=8,
            question_embedding_dimension=6,
            relation_embedding_dimension=6,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="mlp",
            use_edge_mlp=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=3,
            dropout=0.2,
        )
        entity_features = torch.randn(4, 8)
        question_features = torch.randn(6)
        relation_features = torch.randn(3, 6)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        logits = model(
            entity_features=entity_features,
            edge_index=edge_index,
            question_features=question_features,
            relation_features=relation_features,
        )
        logits.sum().backward()

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(len(model.layer_norms), 2)
        self.assertEqual(model.edge_mlp_hidden_dim, 3)
        self.assertIsNotNone(model.edge_mlp[0].weight.grad)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_combined_options_support_bfloat16_autocast(self) -> None:
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        model = GnnAnswerRetriever(
            entity_embedding_dimension=8,
            question_embedding_dimension=6,
            relation_embedding_dimension=6,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="mlp",
            use_edge_mlp=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=256,
            dropout=0.1,
        )
        entity_features = torch.randn(4, 8)
        question_features = torch.randn(6)
        relation_features = torch.randn(6, 6)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2],
            ]
        )

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(
                entity_features=entity_features,
                edge_index=edge_index,
                question_features=question_features,
                relation_features=relation_features,
            )
            loss = logits.float().sum()
        loss.backward()

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertIsNotNone(model.edge_mlp[0].weight.grad)

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_question_aware_classifier_uses_wide_input(self) -> None:
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        model = GnnAnswerRetriever(
            entity_embedding_dimension=8,
            question_embedding_dimension=6,
            relation_embedding_dimension=6,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="linear",
            question_aware_classifier=True,
            dropout=0.1,
        )

        self.assertEqual(model.classifier[0].in_features, 12)

    def test_missing_configuration_in_result_bank_raises(self) -> None:
        builder = StepContextBuilder()
        step = BuildGnnAnswerRetrieverStep()

        with self.assertRaises(PipelineException):
            step.execute(self.make_loaded_dataset_context(builder))

    @unittest.skipIf(torch is None, "PyTorch is not installed.")
    def test_pipeline_stores_built_gnn_answer_retriever(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                FakeConfigurationStep(),
                FakeLoadedDatasetStep(),
                BuildGnnAnswerRetrieverStep(),
            ],
        )

        result = pipeline.prepare(StepContext(result=None))

        self.assertTrue(result.success)
        self.assertTrue(
            pipeline.context_builder.has_stored_result(BuiltGnnAnswerRetriever)
        )


if __name__ == "__main__":
    unittest.main()
