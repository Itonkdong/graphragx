"""Tests for the graphragX entry point."""

import unittest
from contextlib import contextmanager
from io import StringIO
from unittest.mock import patch

import main
from pipeline import (
    BuildReasoningSamplesFromGnnEvaluationStep,
    BuildGnnAnswerRetrieverStep,
    BuildPipelineConfigurationStep,
    BuildWebQSPLocalGraphsStep,
    BuiltGnnAnswerRetriever,
    ComputeFinalResultsStep,
    EvaluateGnnAnswerRetrieverStep,
    ExtractShortestPathsBatchStep,
    GenerateAndSaveFinalAnswersBatchesStep,
    GnnAnswerRetrieverEvaluationResult,
    LogFinalResultsToWandbStep,
    LogEvidenceToWandbStep,
    LogRetrieverToWandbStep,
    LogTrainingToWandbStep,
    LogInferenceToWandbStep,
    LoadGnnAnswerRetrieverRunStep,
    LoadDatasetStep,
    PrepareGnnTrainingDataStep,
    PreparedGnnTrainingData,
    PreparedWebQSPGraphDataset,
    SaveEvidenceSubgraphsStep,
    SelectedDataset,
    StepContext,
    TrainGnnAnswerRetrieverStep,
    TrainedGnnAnswerRetriever,
    WebQSPVocabularyStore,
)
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.services import AbstractDatasetLoaderService
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    SavedGnnAnswerRetrieverConfig,
    SavedGnnAnswerRetrieverTrainingConfig,
)


class FakeSplit:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeLoaderService(AbstractDatasetLoaderService):
    def load_dataset(self, dataset_id: str) -> dict[str, FakeSplit]:
        return {
            "train": FakeSplit(3),
            "validation": FakeSplit(2),
            "test": FakeSplit(1),
        }


class FakeAnswerRetrieverModel(AnswerRetrieverModel):
    pass


class FakeGnnAnswerRetrieverStep(BuildGnnAnswerRetrieverStep):
    def execute_default(self, context):
        return BuiltGnnAnswerRetriever(
            dataset_id=context.result.dataset_id,
            gnn_architecture="graphsage",
            entity_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            question_embedding_dimension=1536,
            relation_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
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
            gnn_architecture=built_retriever.gnn_architecture,
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


class MainEntrypointTests(unittest.TestCase):
    def test_seed_defaults_and_accepts_non_negative_override(self) -> None:
        parser = main.build_parser()

        self.assertEqual(parser.parse_args([]).random_seed, 42)
        self.assertEqual(parser.parse_args(["--seed", "7"]).random_seed, 7)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--seed", "-1"])

    def test_candidate_top_k_must_be_at_least_ten(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than or equal to 10"):
            main.PipelineRuntimeConfig(candidate_top_k=9)

    @staticmethod
    def _patch_dataset_loading_step():
        return patch(
            "main.LoadDatasetStep",
            return_value=LoadDatasetStep(
                loader_service=FakeLoaderService(),
            ),
        )

    @staticmethod
    def _patch_gnn_builder_step():
        return patch(
            "main.BuildGnnAnswerRetrieverStep",
            return_value=FakeGnnAnswerRetrieverStep(),
        )

    @staticmethod
    def _patch_webqsp_local_graph_step():
        return patch(
            "main.BuildWebQSPLocalGraphsStep",
            return_value=FakeWebQSPLocalGraphsStep(),
        )

    @staticmethod
    @contextmanager
    def _patch_training_step():
        with patch(
            "main.PrepareGnnTrainingDataStep",
            return_value=FakePrepareGnnTrainingDataStep(),
        ), patch(
            "main.TrainGnnAnswerRetrieverStep",
            return_value=FakeTrainGnnAnswerRetrieverStep(),
        ):
            yield

    @staticmethod
    def _patch_evaluation_step():
        return patch(
            "main.EvaluateGnnAnswerRetrieverStep",
            return_value=FakeEvaluateGnnAnswerRetrieverStep(),
        )

    def test_run_pipeline_returns_success_for_webqsp(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step():
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
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.model_run_number, 1)

    def test_main_logs_success_summary_without_printing_full_payload(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(
                [
                    "--dataset",
                    "WebQSP",
                    "--main-llm-model",
                    "gpt-5.4",
                    "--subgraph-algorithm",
                    "shortest_path",
                    "--context-strategy",
                    "structured_triples",
                    "--gnn-layers",
                    "2",
                    "--gnn-hidden-dim",
                    "256",
                    "--node-classifier",
                    "mlp",
                    "--question-embedding-model",
                    "text-embedding-3-small",
                    "--relation-embedding-model",
                    "text-embedding-3-small",
                    "--entity-embedding-model",
                    "text-embedding-3-small",
                    "--no-llm-inference",
                    "--no-wandb",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_returns_error_for_unsupported_dataset(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(
                [
                    "--dataset",
                    "WN18RR",
                    "--main-llm-model",
                    "gpt-5.4",
                    "--subgraph-algorithm",
                    "shortest_path",
                    "--context-strategy",
                    "structured_triples",
                    "--gnn-layers",
                    "2",
                    "--gnn-hidden-dim",
                    "256",
                    "--node-classifier",
                    "mlp",
                    "--question-embedding-model",
                    "text-embedding-3-small",
                    "--relation-embedding-model",
                    "text-embedding-3-small",
                    "--entity-embedding-model",
                    "text-embedding-3-small",
                    "--no-llm-inference",
                    "--no-wandb",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_interactively_prompts_missing_configuration_flags(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=["1", "1", "2", "1", "2", "1", "1", "1", "1", "1", "1"],
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--dataset", "WebQSP", "--no-llm-inference", "--no-wandb"])

        self.assertEqual(exit_code, 0)
        self.assertNotIn('"success"', stdout.getvalue())

    def test_main_default_flag_runs_without_prompting(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            exit_code = main.main(["--default", "--no-llm-inference", "--no-wandb"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")

    def test_main_no_arguments_does_not_force_default_configuration_values(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertFalse(captured_configs[0].use_default_config_values)
        self.assertFalse(captured_configs[0].no_llm_inference)
        self.assertFalse(captured_configs[0].no_wandb)
        self.assertIsNone(captured_configs[0].training_max_instances)
        self.assertEqual(captured_configs[0].training_start_instance, 0)
        self.assertIsNone(captured_configs[0].evaluation_max_instances)
        self.assertEqual(
            captured_configs[0].evaluation_log_every,
            main.DEFAULT_EVALUATION_LOG_EVERY,
        )

    def test_training_continuation_flags_are_parsed(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main(
                [
                    "--training-start-instance",
                    "101",
                    "--training-max-instances",
                    "100",
                    "--continue-training-model-run-name",
                    "12_old",
                    "--continue-training-model-run-number",
                    "12",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertEqual(captured_configs[0].training_start_instance, 101)
        self.assertEqual(captured_configs[0].training_max_instances, 100)
        self.assertEqual(captured_configs[0].continue_training_model_run_name, "12_old")
        self.assertEqual(captured_configs[0].continue_training_model_run_number, 12)

    def test_gnn_architecture_upgrade_flags_are_parsed(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main(
                [
                    "--gnn-architecture",
                    "aa-graphsage",
                    "--use-edge-mlp",
                    "--question-aware-classifier",
                    "--use-reverse-edges",
                    "--add-layer-normalization",
                    "--edge-mlp-hidden-dim",
                    "128",
                    "--dropout",
                    "0.2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertEqual(captured_configs[0].gnn_architecture, "aa-graphsage")
        self.assertTrue(captured_configs[0].use_edge_mlp)
        self.assertTrue(captured_configs[0].question_aware_classifier)
        self.assertTrue(captured_configs[0].use_reverse_edges)
        self.assertTrue(captured_configs[0].add_layer_normalization)
        self.assertEqual(captured_configs[0].edge_mlp_hidden_dim, 128)
        self.assertEqual(captured_configs[0].dropout, 0.2)

    def test_aa_architecture_defaults_enable_all_advanced_features(self) -> None:
        resolved = main.PipelineRuntimeConfig(
            gnn_architecture="aa-graphsage",
            use_default_config_values=True,
        ).with_defaulted_user_inputs()

        self.assertEqual(resolved.gnn_architecture, "aa-graphsage")
        self.assertTrue(resolved.use_edge_mlp)
        self.assertTrue(resolved.use_reverse_edges)
        self.assertTrue(resolved.question_aware_classifier)
        self.assertTrue(resolved.add_layer_normalization)
        self.assertEqual(resolved.edge_mlp_hidden_dim, 256)
        self.assertEqual(resolved.dropout, 0.1)

    def test_aa_boolean_negative_forms_are_parsed(self) -> None:
        args = main.build_parser().parse_args(
            [
                "--gnn-architecture", "aa-graphsage",
                "--no-use-edge-mlp",
                "--no-use-reverse-edges",
                "--no-question-aware-classifier",
                "--no-add-layer-normalization",
            ]
        )

        self.assertFalse(args.use_edge_mlp)
        self.assertFalse(args.use_reverse_edges)
        self.assertFalse(args.question_aware_classifier)
        self.assertFalse(args.add_layer_normalization)

    def test_vezilka_provider_and_free_form_model_are_parsed(self) -> None:
        args = main.build_parser().parse_args(
            [
                "--llm-provider",
                "vezilka",
                "--main-llm-model",
                "qwen3-4b-new",
                "--reasoning-effort",
                "none",
            ]
        )

        self.assertEqual(args.llm_provider, "vezilka")
        self.assertEqual(args.main_llm_model, "qwen3-4b-new")
        self.assertEqual(args.reasoning_effort, "none")

    def test_explanation_generation_is_opt_in(self) -> None:
        parser = main.build_parser()

        self.assertFalse(parser.parse_args([]).generate_explanation)
        self.assertTrue(
            parser.parse_args(["--generate-explanation"]).generate_explanation
        )

        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(generate_explanation=True),
        )
        generation_step = next(
            step
            for step in pipeline.evaluation_steps
            if isinstance(step, GenerateAndSaveFinalAnswersBatchesStep)
        )
        self.assertTrue(generation_step.generate_explanation)

    def test_deepseek_model_requires_explicit_provider(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "llm-provider deepseek"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    main_llm_model="deepseek-v4-pro",
                )
            )

    def test_private_model_requires_explicit_vezilka_provider(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "llm-provider vezilka"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    main_llm_model="qwen3.8-27b",
                )
            )

    def test_openai_model_uses_implicit_openai_provider(self) -> None:
        resolved = main.PipelineRuntimeConfig(
            main_llm_model="gpt-5.6-luna",
            use_default_config_values=True,
        ).with_defaulted_user_inputs()

        self.assertEqual(resolved.llm_provider, "openai")
        self.assertEqual(resolved.main_llm_model, "gpt-5.6-luna")

    def test_llm_inference_parallel_calls_flag_is_parsed(self) -> None:
        args = main.build_parser().parse_args(
            ["--llm-inference-parallel-calls", "6"]
        )

        self.assertEqual(args.llm_inference_parallel_calls, 6)

    def test_llm_inference_parallel_calls_defaults_to_one(self) -> None:
        args = main.build_parser().parse_args([])

        self.assertEqual(args.llm_inference_parallel_calls, 1)

    def test_evaluation_log_every_flag_is_parsed(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main(["--evaluation-log-every", "25"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertEqual(captured_configs[0].evaluation_log_every, 25)

    def test_console_and_wandb_training_log_flags_are_parsed_separately(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main(
                [
                    "--training-log-every",
                    "20",
                    "--wandb-training-log-every",
                    "5",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured_configs[0].training_log_every, 20)
        self.assertEqual(captured_configs[0].wandb_training_log_every, 5)

    def test_wandb_retriever_upload_flag_defaults_off_and_parses(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            self.assertEqual(main.main([]), 0)
            self.assertFalse(captured_configs[-1].wandb_upload_retriever)
            self.assertEqual(main.main(["--wandb-upload-retriever"]), 0)

        self.assertTrue(captured_configs[-1].wandb_upload_retriever)

    def test_local_graph_profile_flag_defaults_off_and_is_wired(self) -> None:
        parser = main.build_parser()
        self.assertFalse(parser.parse_args([]).local_graph_profile)
        self.assertTrue(
            parser.parse_args(
                ["--local-graph-profile"]
            ).local_graph_profile
        )

        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                local_graph_profile=True,
                no_wandb=True,
            )
        )
        graph_step = next(
            step
            for step in pipeline.preparation_steps
            if isinstance(step, BuildWebQSPLocalGraphsStep)
        )
        self.assertTrue(graph_step.profile)

    def test_evaluation_embedding_and_profile_flags_are_parsed(self) -> None:
        captured_configs: list[main.PipelineRuntimeConfig] = []

        def fake_run_pipeline(config: main.PipelineRuntimeConfig):
            captured_configs.append(config)
            return main.PipelineExecutionResult.success_result(
                final_result=main.InitialStepResult(),
                execution_time_ms=0.0,
                steps_executed=0,
                total_steps=0,
            )

        with patch("main.run_pipeline", side_effect=fake_run_pipeline), patch(
            "sys.stdout",
            new_callable=StringIO,
        ):
            exit_code = main.main(
                [
                    "--evaluation-profile",
                    "--evaluation-embedding-cache-device",
                    "gpu",
                    "--evaluation-embedding-cache-dtype",
                    "bfloat16",
                    "--evaluation-gpu-cache-reserve-gb",
                    "4.5",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_configs), 1)
        self.assertTrue(captured_configs[0].evaluation_profile)
        self.assertEqual(
            captured_configs[0].evaluation_embedding_cache_device,
            "gpu",
        )
        self.assertEqual(
            captured_configs[0].evaluation_embedding_cache_dtype,
            "bfloat16",
        )
        self.assertEqual(captured_configs[0].evaluation_gpu_cache_reserve_gb, 4.5)

    def test_run_pipeline_default_config_succeeds_from_neutral_initial_result(self) -> None:
        with self._patch_dataset_loading_step(), self._patch_webqsp_local_graph_step(), self._patch_gnn_builder_step(), self._patch_training_step(), self._patch_evaluation_step(), patch(
            "builtins.input",
            side_effect=AssertionError("input should not be called"),
        ):
            result = main.run_pipeline(
                config=main.PipelineRuntimeConfig(
                    use_default_config_values=True,
                    no_llm_inference=True,
                    no_wandb=True,
                ),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.final_result.dataset_id, "WebQSP")
        self.assertEqual(result.final_result.model_run_number, 1)

    def test_force_default_flag_sets_step_execution_mode_only(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(force_all_default=True),
        )

        self.assertTrue(pipeline.force_all_default)
        self.assertTrue(all(step.force_default for step in pipeline.preparation_steps))
        self.assertIsNone(
            pipeline.preparation_steps[0].requested_dataset,
        )

    def test_default_pipeline_appends_final_results_and_wandb_steps(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(),
        )

        self.assertIsInstance(pipeline.evaluation_steps[0], EvaluateGnnAnswerRetrieverStep)
        self.assertIsInstance(pipeline.evaluation_steps[1], LogRetrieverToWandbStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[2],
            BuildReasoningSamplesFromGnnEvaluationStep,
        )
        self.assertIsInstance(pipeline.evaluation_steps[3], ExtractShortestPathsBatchStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[4],
            GenerateAndSaveFinalAnswersBatchesStep,
        )
        self.assertIsInstance(pipeline.evaluation_steps[5], LogInferenceToWandbStep)
        self.assertIsInstance(pipeline.evaluation_steps[6], ComputeFinalResultsStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[7],
            LogFinalResultsToWandbStep,
        )
        self.assertEqual(len(pipeline.evaluation_steps), 8)
        self.assertIsInstance(pipeline.preparation_steps[-1], LogTrainingToWandbStep)

    def test_pipeline_modes_load_only_required_processed_graph_splits(self) -> None:
        expectations = {
            "full": (True, True),
            "train-only": (True, False),
            "retriever-only": (True, True),
            "evaluation-only": (False, True),
            "inference-only": (False, True),
            "evidence-only": (False, True),
        }
        saved_config = SavedGnnAnswerRetrieverConfig(dataset_id="WebQSP")
        with patch.object(
            main.GnnAnswerRetrieverModelRunService,
            "resolve_run",
            return_value=type("Run", (), {"config": saved_config})(),
        ), patch.object(
            main.GnnRetrieverResultsService,
            "load_model_config",
            return_value=saved_config,
        ):
            for run_mode, expected in expectations.items():
                with self.subTest(run_mode=run_mode):
                    pipeline = main.build_pipeline(
                        config=main.PipelineRuntimeConfig(
                            run_mode=run_mode,
                            no_wandb=True,
                            evaluation_model_run_number=(
                                1 if run_mode == "evaluation-only" else None
                            ),
                            retriever_run_number=(
                                1
                                if run_mode in {"inference-only", "evidence-only"}
                                else None
                            ),
                        ),
                    )
                    graph_step = next(
                        step
                        for step in pipeline.preparation_steps
                        if isinstance(step, BuildWebQSPLocalGraphsStep)
                    )
                    self.assertEqual(
                        (
                            graph_step.load_train_instances,
                            graph_step.load_test_instances,
                        ),
                        expected,
                    )

    def test_evaluation_log_every_is_wired_into_evaluation_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(evaluation_log_every=25),
        )

        evaluation_step = pipeline.evaluation_steps[0]
        self.assertIsInstance(evaluation_step, EvaluateGnnAnswerRetrieverStep)
        self.assertEqual(evaluation_step.evaluation_config.log_every, 25)

    def test_evaluation_embedding_and_profile_config_is_wired_into_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                evaluation_profile=True,
                evaluation_embedding_cache_device="gpu",
                evaluation_embedding_cache_dtype="bfloat16",
                evaluation_gpu_cache_reserve_gb=4.5,
            ),
        )

        evaluation_step = pipeline.evaluation_steps[0]
        self.assertIsInstance(evaluation_step, EvaluateGnnAnswerRetrieverStep)
        self.assertTrue(evaluation_step.evaluation_config.profile)
        self.assertEqual(evaluation_step.evaluation_config.embedding_cache_device, "gpu")
        self.assertEqual(
            evaluation_step.evaluation_config.embedding_cache_dtype,
            "bfloat16",
        )
        self.assertEqual(evaluation_step.evaluation_config.gpu_cache_reserve_gb, 4.5)

    def test_training_continuation_is_wired_into_training_step(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP", entity_embedding_model="text-embedding-3-small",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536, question_embedding_dimension=1536,
            relation_embedding_dimension=1536, hidden_dimension=256,
            gnn_layer_count=2, node_classifier="mlp",
            training=SavedGnnAnswerRetrieverTrainingConfig(
                epochs=1, learning_rate=0.001, weight_decay=0.0,
                log_every=3, device="cpu",
            ), final_loss=0.5, trained_instances=3,
        )
        with patch.object(
            main.GnnAnswerRetrieverModelRunService,
            "resolve_run",
            return_value=type("Run", (), {"config": saved_config})(),
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    training_start_instance=101,
                    training_max_instances=100,
                    continue_training_model_run_name="12_old",
                    continue_training_model_run_number=12,
                ),
            )

        training_step = next(
            step for step in pipeline.preparation_steps
            if isinstance(step, TrainGnnAnswerRetrieverStep)
        )
        self.assertIsInstance(training_step, TrainGnnAnswerRetrieverStep)
        self.assertEqual(training_step.training_config.start_instance, 101)
        self.assertEqual(training_step.training_config.max_instances, 100)
        self.assertEqual(
            training_step.training_config.continue_from_model_run_name,
            "12_old",
        )
        self.assertEqual(
            training_step.training_config.continue_from_model_run_number,
            12,
        )

    def test_training_profile_is_wired_into_training_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(training_profile=True),
        )

        training_step = next(
            step for step in pipeline.preparation_steps
            if isinstance(step, TrainGnnAnswerRetrieverStep)
        )
        self.assertIsInstance(training_step, TrainGnnAnswerRetrieverStep)
        self.assertTrue(training_step.training_config.profile)

    def test_console_and_wandb_training_intervals_are_wired_separately(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                training_log_every=20,
                wandb_training_log_every=5,
            ),
        )

        training_step = next(
            step for step in pipeline.preparation_steps
            if isinstance(step, TrainGnnAnswerRetrieverStep)
        )
        self.assertEqual(training_step.training_config.log_every, 20)
        self.assertEqual(training_step.training_service.progress_callback_every, 5)

    def test_embedding_cache_settings_are_wired_into_preparation_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                training_embedding_cache_device="gpu",
                training_embedding_cache_dtype="bfloat16",
                training_gpu_cache_reserve_gb=5.0,
            ),
        )

        preparation_step = next(
            step for step in pipeline.preparation_steps
            if isinstance(step, PrepareGnnTrainingDataStep)
        )
        self.assertIsInstance(preparation_step, PrepareGnnTrainingDataStep)
        self.assertEqual(
            preparation_step.preparation_config.embedding_cache_device,
            "gpu",
        )
        self.assertEqual(
            preparation_step.preparation_config.embedding_cache_dtype,
            "bfloat16",
        )
        self.assertEqual(
            preparation_step.preparation_config.gpu_cache_reserve_gb,
            5.0,
        )

    def test_gnn_architecture_flags_are_wired_into_configuration_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                gnn_architecture="aa-graphsage",
                use_edge_mlp=True,
                question_aware_classifier=True,
                use_reverse_edges=True,
                add_layer_normalization=True,
                edge_mlp_hidden_dim=128,
                dropout=0.2,
            ),
        )

        configuration_step = pipeline.preparation_steps[1]
        self.assertTrue(configuration_step.configuration_input.use_edge_mlp)
        self.assertTrue(configuration_step.configuration_input.question_aware_classifier)
        self.assertTrue(configuration_step.configuration_input.use_reverse_edges)
        self.assertTrue(configuration_step.configuration_input.add_layer_normalization)
        self.assertEqual(configuration_step.configuration_input.gnn_architecture, "aa-graphsage")
        self.assertEqual(configuration_step.configuration_input.edge_mlp_hidden_dim, 128)
        self.assertEqual(configuration_step.configuration_input.dropout, 0.2)

    def test_saved_rearev_mandatory_reverse_edges_are_not_restored_as_an_option(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP",
            gnn_architecture="rearev",
            gnn_architecture_options={
                "gnn_hidden_dimension": 50,
                "dropout": 0.1,
                "num_instructions": 2,
                "reasoning_steps": 2,
                "adaptive_iterations": 3,
            },
        )

        restored = main._apply_saved_model_config(
            main.PipelineRuntimeConfig(
                run_mode="inference-only",
                llm_provider="vezilka",
                main_llm_model="qwen3.8-27b",
            ),
            saved_config,
        )

        self.assertIsNone(restored.use_reverse_edges)
        configuration_step = BuildPipelineConfigurationStep(
            llm_provider=restored.llm_provider,
            main_llm_model=restored.main_llm_model,
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_architecture=restored.gnn_architecture,
            gnn_options=restored.gnn_options,
            use_reverse_edges=restored.use_reverse_edges,
        )
        configuration = configuration_step.execute(
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
        self.assertTrue(configuration.use_reverse_edges)

    def test_negative_training_start_instance_fails_early(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "training-start-instance",
        ):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(training_start_instance=-1),
            )

    def test_negative_training_log_intervals_fail_early(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "training-log-every"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(training_log_every=-1),
            )
        with self.assertRaisesRegex(main.PipelineException, "wandb-training-log-every"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(wandb_training_log_every=-1),
            )

    def test_non_positive_llm_inference_parallel_calls_fail_early(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "llm-inference-parallel-calls",
        ):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    llm_inference_parallel_calls=0,
                ),
            )

    def test_evaluation_only_rejects_training_continuation_flags(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "continuation flags",
        ):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="evaluation-only",
                    continue_training_model_run_number=1,
                ),
            )

    def test_evaluation_only_uses_a_new_wandb_run(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP",
            entity_embedding_model="text-embedding-3-small",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            question_embedding_dimension=1536,
            relation_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
            node_classifier="mlp",
            training=SavedGnnAnswerRetrieverTrainingConfig(
                epochs=1, learning_rate=0.001, weight_decay=0.0,
                log_every=3, device="cpu",
            ),
            final_loss=0.5,
            trained_instances=3,
        )
        with patch.object(
            main.GnnAnswerRetrieverModelRunService,
            "resolve_run",
            return_value=type("Run", (), {"config": saved_config})(),
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="evaluation-only",
                    evaluation_model_run_number=7,
                )
            )

        retriever_wandb_step = pipeline.evaluation_steps[1]
        self.assertIsInstance(retriever_wandb_step, LogRetrieverToWandbStep)
        self.assertFalse(retriever_wandb_step.coordinator.resume_from_lineage)

    def test_no_wandb_keeps_final_results_without_wandb_step(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                no_wandb=True,
                wandb_project="project",
                wandb_mode="disabled",
            ),
        )

        self.assertIsInstance(pipeline.evaluation_steps[4], ComputeFinalResultsStep)
        self.assertEqual(len(pipeline.evaluation_steps), 5)

    def test_no_llm_inference_skips_post_retrieval_steps_when_wandb_is_also_disabled(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(
                no_llm_inference=True,
                no_wandb=True,
            ),
        )

        self.assertIsInstance(pipeline.evaluation_steps[0], EvaluateGnnAnswerRetrieverStep)
        self.assertEqual(len(pipeline.evaluation_steps), 1)

    def test_wandb_logs_retriever_when_llm_inference_is_disabled(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(no_llm_inference=True),
        )

        self.assertIsInstance(pipeline.evaluation_steps[0], EvaluateGnnAnswerRetrieverStep)
        self.assertIsInstance(pipeline.evaluation_steps[1], LogRetrieverToWandbStep)
        self.assertEqual(len(pipeline.evaluation_steps), 2)

    def test_retriever_only_composes_training_and_retriever_stages(self) -> None:
        pipeline = main.build_pipeline(
            config=main.PipelineRuntimeConfig(run_mode="retriever-only"),
        )

        self.assertTrue(
            any(isinstance(step, TrainGnnAnswerRetrieverStep) for step in pipeline.preparation_steps)
        )
        self.assertEqual(
            [type(step) for step in pipeline.evaluation_steps],
            [EvaluateGnnAnswerRetrieverStep, LogRetrieverToWandbStep],
        )

    def test_inference_only_requires_retriever_selector(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "requires --retriever-run"):
            main.run_pipeline(
                config=main.PipelineRuntimeConfig(run_mode="inference-only")
            )

    def test_inference_only_rejects_no_llm_inference(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "not valid"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="inference-only",
                    retriever_run_number=1,
                    no_llm_inference=True,
                )
            )

    def test_inference_only_composes_loader_and_inference_without_training(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP",
            entity_embedding_model="text-embedding-3-small",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            question_embedding_dimension=1536,
            relation_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
            node_classifier="mlp",
            training=SavedGnnAnswerRetrieverTrainingConfig(
                epochs=1,
                learning_rate=0.001,
                weight_decay=0.0,
                log_every=3,
                device="cpu",
            ),
            final_loss=0.5,
            trained_instances=3,
        )
        with patch.object(
            main.GnnRetrieverResultsService,
            "load_model_config",
            return_value=saved_config,
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="inference-only",
                    retriever_run_number=7,
                    use_default_config_values=True,
                    no_wandb=True,
                )
            )

        self.assertFalse(
            any(isinstance(step, TrainGnnAnswerRetrieverStep) for step in pipeline.preparation_steps)
        )
        self.assertIsInstance(pipeline.evaluation_steps[0], LoadGnnAnswerRetrieverRunStep)
        self.assertIsInstance(
            pipeline.evaluation_steps[1], BuildReasoningSamplesFromGnnEvaluationStep
        )
        self.assertIsInstance(pipeline.evaluation_steps[-1], ComputeFinalResultsStep)

    def test_inference_only_copies_retriever_into_new_wandb_run(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP",
            entity_embedding_model="text-embedding-3-small",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_dimension=1536,
            question_embedding_dimension=1536,
            relation_embedding_dimension=1536,
            hidden_dimension=256,
            gnn_layer_count=2,
            node_classifier="mlp",
            training=SavedGnnAnswerRetrieverTrainingConfig(
                epochs=1,
                learning_rate=0.001,
                weight_decay=0.0,
                log_every=3,
                device="cpu",
            ),
            final_loss=0.5,
            trained_instances=3,
        )
        with patch.object(
            main.GnnRetrieverResultsService,
            "load_model_config",
            return_value=saved_config,
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="inference-only",
                    retriever_run_number=7,
                    use_default_config_values=True,
                )
            )

        retriever_wandb_step = pipeline.evaluation_steps[1]
        self.assertIsInstance(retriever_wandb_step, LogRetrieverToWandbStep)
        self.assertTrue(retriever_wandb_step.copy_to_new_experiment)

    def test_evidence_only_requires_retriever_selector(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "requires --retriever-run"):
            main.run_pipeline(
                config=main.PipelineRuntimeConfig(run_mode="evidence-only")
            )

    def test_evidence_only_composes_without_llm_or_final_results(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(dataset_id="WebQSP")
        with patch.object(
            main.GnnRetrieverResultsService,
            "load_model_config",
            return_value=saved_config,
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="evidence-only",
                    retriever_run_number=7,
                    no_wandb=True,
                    use_default_config_values=True,
                )
            )

        self.assertEqual(
            [type(step) for step in pipeline.evaluation_steps],
            [
                LoadGnnAnswerRetrieverRunStep,
                BuildReasoningSamplesFromGnnEvaluationStep,
                ExtractShortestPathsBatchStep,
                SaveEvidenceSubgraphsStep,
            ],
        )
        self.assertFalse(
            any(
                isinstance(step, GenerateAndSaveFinalAnswersBatchesStep)
                for step in pipeline.evaluation_steps
            )
        )
        self.assertFalse(
            any(
                isinstance(step, ComputeFinalResultsStep)
                for step in pipeline.evaluation_steps
            )
        )

    def test_evidence_only_creates_new_wandb_run_and_logs_evidence(self) -> None:
        saved_config = SavedGnnAnswerRetrieverConfig(dataset_id="WebQSP")
        with patch.object(
            main.GnnRetrieverResultsService,
            "load_model_config",
            return_value=saved_config,
        ):
            pipeline = main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    run_mode="evidence-only",
                    retriever_run_number=7,
                    use_default_config_values=True,
                )
            )

        self.assertIsInstance(pipeline.evaluation_steps[1], LogRetrieverToWandbStep)
        self.assertTrue(pipeline.evaluation_steps[1].copy_to_new_experiment)
        self.assertIsInstance(pipeline.evaluation_steps[-1], LogEvidenceToWandbStep)

    def test_evidence_only_rejects_llm_and_inference_run_options(self) -> None:
        for updates in (
            {"main_llm_model": "gpt-5.6-luna"},
            {"llm_provider": "openai"},
            {"reasoning_effort": "low"},
            {"generate_explanation": True},
            {"inference_run_name": "not-evidence"},
        ):
            with self.subTest(updates=updates), self.assertRaisesRegex(
                main.PipelineException,
                "Evidence-only mode does not accept",
            ):
                main.build_pipeline(
                    config=main.PipelineRuntimeConfig(
                        run_mode="evidence-only",
                        retriever_run_number=7,
                        **updates,
                    )
                )

    def test_evidence_run_name_is_restricted_to_evidence_only(self) -> None:
        with self.assertRaisesRegex(main.PipelineException, "only valid"):
            main.build_pipeline(
                config=main.PipelineRuntimeConfig(
                    evidence_run_name="evidence-label"
                )
            )

    def test_evidence_only_cli_flags_parse(self) -> None:
        args = main.build_parser().parse_args(
            [
                "--evidence-only",
                "--retriever-run-number",
                "7",
                "--evidence-run-name",
                "pcst-comparison",
            ]
        )

        self.assertEqual(args.run_mode, "evidence-only")
        self.assertEqual(args.retriever_run_number, 7)
        self.assertEqual(args.evidence_run_name, "pcst-comparison")


class PcstCliTests(unittest.TestCase):
    def test_pcst_cli_options_parse(self) -> None:
        args = main.build_parser().parse_args(
            [
                "--subgraph-algorithm",
                "pcst",
                "--pcst-edge-cost-strategy",
                "semantic",
                "--pcst-edge-cost",
                "0.5",
                "--pcst-debug-profile",
            ]
        )
        self.assertEqual(args.subgraph_algorithm, "pcst")
        self.assertEqual(args.pcst_edge_cost_strategy, "semantic")
        self.assertEqual(args.pcst_edge_cost, 0.5)
        self.assertTrue(args.pcst_debug_profile)

        pipeline = main.build_pipeline(
            main.PipelineRuntimeConfig(
                subgraph_algorithm="pcst",
                pcst_debug_profile=True,
                no_wandb=True,
                use_default_config_values=True,
            )
        )
        evidence_step = next(
            step
            for step in pipeline.evaluation_steps
            if isinstance(step, ExtractShortestPathsBatchStep)
        )
        self.assertTrue(evidence_step.pcst_debug_profile)

    def test_default_pcst_configuration_uses_constant_cost(self) -> None:
        resolved = main.PipelineRuntimeConfig(
            subgraph_algorithm="pcst",
            use_default_config_values=True,
        ).with_defaulted_user_inputs()
        self.assertEqual(resolved.pcst_edge_cost_strategy, "constant")
        self.assertEqual(resolved.pcst_edge_cost, 1.0)

    def test_pcst_flags_are_rejected_for_shortest_path(self) -> None:
        with self.assertRaisesRegex(
            main.PipelineException,
            "require --subgraph-algorithm pcst",
        ):
            main.build_pipeline(
                main.PipelineRuntimeConfig(
                    subgraph_algorithm="shortest_path",
                    pcst_edge_cost=1.0,
                    use_default_config_values=True,
                )
            )

    def test_semantic_pcst_resolves_embedding_for_rearev(self) -> None:
        resolved = main.PipelineRuntimeConfig(
            gnn_architecture="rearev",
            subgraph_algorithm="pcst",
            pcst_edge_cost_strategy="semantic",
            use_default_config_values=True,
        ).with_defaulted_user_inputs()
        self.assertEqual(resolved.embedding_model, "text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
