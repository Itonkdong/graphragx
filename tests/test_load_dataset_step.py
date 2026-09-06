"""Tests for preparation-time dataset loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
from pipeline import (
    BuildGnnAnswerRetrieverStep,
    BuildWebQSPLocalGraphsStep,
    BuiltGnnAnswerRetriever,
    BuiltPipelineConfiguration,
    DatasetLoadingException,
    EvaluateGnnAnswerRetrieverStep,
    GnnAnswerRetrieverEvaluationResult,
    LoadDatasetStep,
    LoadedDataset,
    MalformedDatasetException,
    MissingHuggingFaceDatasetsDependencyException,
    Pipeline,
    PrepareGnnTrainingDataStep,
    PreparedGnnTrainingData,
    PreparedWebQSPGraphDataset,
    StepContext,
    TrainGnnAnswerRetrieverStep,
    TrainedGnnAnswerRetriever,
    UnsupportedDatasetLoaderException,
    WebQSPVocabularyStore,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.services import AbstractDatasetLoaderService


class FakeSplit:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeDatasetDict(dict[str, FakeSplit]):
    pass


class FakeLoaderService(AbstractDatasetLoaderService):
    def __init__(self, dataset: FakeDatasetDict | None = None):
        if dataset is None:
            dataset = FakeDatasetDict(
                {
                    "train": FakeSplit(3),
                    "validation": FakeSplit(2),
                    "test": FakeSplit(1),
                }
            )
        self.dataset = dataset

    def load_dataset(self, dataset_id: str) -> FakeDatasetDict:
        return self.dataset


class FakeAnswerRetrieverModel(AnswerRetrieverModel):
    pass


class FakeGnnAnswerRetrieverStep(BuildGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return BuiltGnnAnswerRetriever(
            dataset_id=context.result.dataset_id,
            entity_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            question_embedding_dimension=1536,
            relation_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            use_edge_mlp=False,
            question_aware_classifier=False,
            use_reverse_edges=False,
            add_layer_normalization=False,
            edge_mlp_hidden_dim=256,
            dropout=0.1,
            model=FakeAnswerRetrieverModel(),
        )


class FakeWebQSPLocalGraphsStep(BuildWebQSPLocalGraphsStep):
    def execute_default(self, context):
        return PreparedWebQSPGraphDataset(
            dataset_id=context.result.dataset_id,
            processing_version="test",
            train_instances=[],
            test_instances=[],
            vocabulary_store=WebQSPVocabularyStore(),
            cache_directory="/tmp/graphragx-test",
        )


class FakeTrainGnnAnswerRetrieverStep(TrainGnnAnswerRetrieverStep):
    def execute_default(self, context):
        built_retriever = context.result.built_retriever
        return TrainedGnnAnswerRetriever(
            dataset_id=built_retriever.dataset_id,
            hidden_dimension=built_retriever.hidden_dimension,
            gnn_layer_count=built_retriever.gnn_layer_count,
            node_classifier=built_retriever.node_classifier,
            training_epochs=1,
            training_learning_rate=1e-3,
            training_weight_decay=0.0,
            training_max_instances=None,
            training_start_instance=0,
            training_end_instance=0,
            training_log_every=25,
            training_device="cpu",
            training_run_name=None,
            selected_device="cpu",
            final_loss=0.0,
            trained_instances=0,
            is_fine_tuned_model=False,
            model=built_retriever.model,
            model_artifact_path="/tmp/graphragx-test/gnn_answer_retriever.pt",
            model_config_path="/tmp/graphragx-test/model_config.json",
            model_run_directory="/tmp/graphragx-test/1_test",
            model_run_name="1_test",
            model_run_number=1,
            embedding_cache_directory="/tmp/graphragx-test/embeddings",
        )


class FakePrepareGnnTrainingDataStep(PrepareGnnTrainingDataStep):
    def execute_default(self, context):
        return PreparedGnnTrainingData(
            built_retriever=context.result,
            instances=[],
            node_embeddings=[],
            relation_embeddings=[],
            question_embeddings=[],
            training_start_instance=0,
            training_end_instance=0,
            selected_device="cpu",
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
            entity_embedding_model="text-embedding-3-small",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            cache_root="/tmp/graphragx-test",
        )


class FakeEvaluateGnnAnswerRetrieverStep(EvaluateGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return GnnAnswerRetrieverEvaluationResult(
            dataset_id=context.result.dataset_id,
            model_run_directory=context.result.model_run_directory,
            model_run_name=context.result.model_run_name,
            model_run_number=context.result.model_run_number,
            evaluation_run_directory="/tmp/graphragx-test/evaluations/1_test",
            evaluation_run_name="1_test",
            evaluation_run_number=1,
            evaluated_instances=0,
            hits_at_1=0.0,
            hits_at_1_count=0,
            hits_at_5=0.0,
            hits_at_5_count=0,
            hits_at_10=0.0,
            hits_at_10_count=0,
            average_candidate_count=0.0,
            missing_gold_in_graph_count=0,
            predictions_path="/tmp/graphragx-test/evaluations/1_test/predictions.jsonl",
            evaluation_config_path="/tmp/graphragx-test/evaluations/1_test/evaluation_config.json",
        )


class LoadDatasetStepTests(unittest.TestCase):
    @staticmethod
    def make_configuration_context(
        dataset_id: str = "WebQSP",
    ) -> StepContext[BuiltPipelineConfiguration]:
        return StepContext(
            result=BuiltPipelineConfiguration(
                dataset_id=dataset_id,
                main_llm_model="gpt-5.4",
                subgraph_construction_algorithm="shortest_path",
                context_construction_strategy="structured_triples",
                gnn_layer_count=2,
                gnn_hidden_dimension=256,
                node_classifier="mlp",
                question_embedding_model="text-embedding-3-small",
                relation_embedding_model="text-embedding-3-small",
                entity_embedding_model="text-embedding-3-small",
            )
        )

    def test_successful_load_returns_expected_artifact(self) -> None:
        step = LoadDatasetStep(loader_service=FakeLoaderService())

        result = step.execute(self.make_configuration_context())

        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.dataset_family, "question_answering")
        self.assertEqual(result.hugging_face_dataset_name, "ml1996/webqsp")
        self.assertEqual(result.split_names, ["train", "validation", "test"])
        self.assertEqual(result.split_sizes, {"train": 3, "validation": 2, "test": 1})
        self.assertIsNotNone(result.hugging_face_dataset)

    def test_step_consumes_configuration_result(self) -> None:
        step = LoadDatasetStep(loader_service=FakeLoaderService())

        result = step.execute(self.make_configuration_context())

        self.assertEqual(result.dataset_id, "WebQSP")

    def test_unsupported_dataset_raises(self) -> None:
        step = LoadDatasetStep(loader_service=FakeLoaderService())

        with self.assertRaises(UnsupportedDatasetLoaderException):
            step.execute(self.make_configuration_context(dataset_id="WN18RR"))

    def test_empty_loaded_dataset_raises(self) -> None:
        step = LoadDatasetStep(loader_service=FakeLoaderService(dataset=FakeDatasetDict()))

        with self.assertRaises(MalformedDatasetException):
            step.execute(self.make_configuration_context())

    def test_loading_result_is_stored_in_result_bank(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                LoadDatasetStep(loader_service=FakeLoaderService()),
            ]
        )

        result = pipeline.prepare(self.make_configuration_context())

        self.assertTrue(result.success)
        self.assertTrue(pipeline.context_builder.has_stored_result(LoadedDataset))

    def test_full_preparation_pipeline_runs_dataset_selection_configuration_and_loading(self) -> None:
        fake_loader = FakeLoaderService()
        with patch(
            "main.LoadDatasetStep",
            return_value=LoadDatasetStep(loader_service=fake_loader),
        ), patch(
            "main.BuildWebQSPLocalGraphsStep",
            return_value=FakeWebQSPLocalGraphsStep(),
        ), patch(
            "main.BuildGnnAnswerRetrieverStep",
            return_value=FakeGnnAnswerRetrieverStep(),
        ), patch(
            "main.PrepareGnnTrainingDataStep",
            return_value=FakePrepareGnnTrainingDataStep(),
        ), patch(
            "main.TrainGnnAnswerRetrieverStep",
            return_value=FakeTrainGnnAnswerRetrieverStep(),
        ), patch(
            "main.EvaluateGnnAnswerRetrieverStep",
            return_value=FakeEvaluateGnnAnswerRetrieverStep(),
        ):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    dataset="WebQSP",
                    main_llm_model="gpt-5.4",
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                    no_llm_inference=True,
                    no_wandb=True,
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 8)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")


class HuggingFaceWebQSPDatasetLoaderServiceTests(unittest.TestCase):
    def test_missing_hugging_face_datasets_raises_dedicated_exception(self) -> None:
        from pipeline.preparation.services.dataset_loader import HuggingFaceWebQSPDatasetLoaderService

        service = HuggingFaceWebQSPDatasetLoaderService()

        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                raise ModuleNotFoundError("No module named 'datasets'")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(MissingHuggingFaceDatasetsDependencyException):
                service.load_dataset("WebQSP")

    def test_loader_failure_raises_dataset_loading_exception(self) -> None:
        from pipeline.preparation.services.dataset_loader import HuggingFaceWebQSPDatasetLoaderService

        service = HuggingFaceWebQSPDatasetLoaderService()

        def broken_load_dataset(dataset_name: str, cache_dir: str):
            raise RuntimeError("broken")

        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                return SimpleNamespace(load_dataset=broken_load_dataset)
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(DatasetLoadingException):
                service.load_dataset("WebQSP")


if __name__ == "__main__":
    unittest.main()
