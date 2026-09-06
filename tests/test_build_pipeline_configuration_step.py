"""Tests for the preparation configuration-building step."""

import unittest

from pipeline import (
    BuildPipelineConfigurationStep,
    InvalidContextConstructionSelectionException,
    InvalidEntityEmbeddingModelSelectionException,
    InvalidGnnLayerCountSelectionException,
    InvalidGnnArchitectureConfigurationException,
    InvalidMainLlmSelectionException,
    InvalidNodeClassifierSelectionException,
    InvalidSubgraphConstructionSelectionException,
    Pipeline,
    SelectedDataset,
    StepContext,
)


class BuildPipelineConfigurationStepTests(unittest.TestCase):
    def make_dataset_context(self) -> StepContext[SelectedDataset]:
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

    def test_constructor_only_path_succeeds(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(result.llm_provider, "openai")
        self.assertEqual(result.gnn_architecture, "graphsage")
        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(result.question_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.relation_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")

    def test_additional_gpt_5_models_are_supported_main_llm_models(self) -> None:
        for model_id in ("gpt-5.6-luna", "gpt-5-mini", "gpt-5-nano"):
            with self.subTest(model_id=model_id):
                step = BuildPipelineConfigurationStep(
                    main_llm_model=model_id,
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                )

                result = step.execute(self.make_dataset_context())

                self.assertEqual(result.main_llm_model, model_id)

    def test_deepseek_v4_models_are_supported_main_llm_models(self) -> None:
        for model_id in ("deepseek-v4-flash", "deepseek-v4-pro"):
            with self.subTest(model_id=model_id):
                step = BuildPipelineConfigurationStep(
                    llm_provider="deepseek",
                    main_llm_model=model_id,
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                )

                result = step.execute(self.make_dataset_context())

                self.assertEqual(result.llm_provider, "deepseek")
                self.assertEqual(result.main_llm_model, model_id)

    def test_private_model_without_vezilka_provider_is_rejected(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="qwen3.8-27b",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            embedding_model="text-embedding-3-small",
        )

        with self.assertRaisesRegex(Exception, "llm_provider='vezilka'"):
            step.execute(self.make_dataset_context())

    def test_vezilka_accepts_an_arbitrary_model_name(self) -> None:
        step = BuildPipelineConfigurationStep(
            llm_provider="vezilka",
            main_llm_model="qwen3-4b-custom",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            embedding_model="text-embedding-3-small",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.llm_provider, "vezilka")
        self.assertEqual(result.main_llm_model, "qwen3-4b-custom")

    def test_reasoning_effort_is_provider_independent_configuration(self) -> None:
        step = BuildPipelineConfigurationStep(
            llm_provider="openai",
            main_llm_model="gpt-5-mini",
            reasoning_effort="low",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            embedding_model="text-embedding-3-small",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.reasoning_effort, "low")

    def test_vezilka_prompts_for_a_free_form_model_name(self) -> None:
        step = BuildPipelineConfigurationStep(
            llm_provider="vezilka",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            embedding_model="text-embedding-3-small",
            input_func=lambda _: "new-private-model",
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "new-private-model")

    def test_mixed_path_prompts_only_missing_fields(self) -> None:
        answers = iter(["1"])
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
            input_func=fake_input,
        )

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(len(prompts), 1)

    def test_constructor_gnn_architecture_flags_are_preserved(self) -> None:
        step = BuildPipelineConfigurationStep(
            gnn_architecture="aa-graphsage",
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
            use_edge_mlp=True,
            question_aware_classifier=True,
            use_reverse_edges=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=128,
            dropout=0.2,
        )

        result = step.execute(self.make_dataset_context())

        self.assertTrue(result.use_edge_mlp)
        self.assertTrue(result.question_aware_classifier)
        self.assertTrue(result.use_reverse_edges)
        self.assertTrue(result.add_layer_normalization)
        self.assertEqual(result.gnn_architecture, "aa-graphsage")
        self.assertEqual(result.edge_mlp_hidden_dim, 128)
        self.assertEqual(result.dropout, 0.2)

    def test_fully_interactive_path_works(self) -> None:
        answers = iter(["1", "1", "2", "1", "2", "1", "1", "1", "1", "1", "1"])

        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))
        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)
        self.assertEqual(result.gnn_hidden_dimension, 256)
        self.assertEqual(result.node_classifier, "mlp")
        self.assertEqual(result.main_llm_model, "gpt-5.4")
        self.assertEqual(result.subgraph_construction_algorithm, "shortest_path")
        self.assertEqual(result.context_construction_strategy, "structured_triples")
        self.assertEqual(result.question_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.relation_embedding_model, "text-embedding-3-small")
        self.assertEqual(result.entity_embedding_model, "text-embedding-3-small")

    def test_graphsage_rejects_explicit_aa_negative_option(self) -> None:
        step = BuildPipelineConfigurationStep(
            gnn_architecture="graphsage",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            use_edge_mlp=False,
        )

        with self.assertRaisesRegex(
            InvalidGnnArchitectureConfigurationException,
            "does not support: use_edge_mlp",
        ):
            step.execute(self.make_dataset_context())

    def test_aa_linear_requires_question_aware_classifier_disabled(self) -> None:
        step = BuildPipelineConfigurationStep(
            gnn_architecture="aa-graphsage",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="linear",
            dropout=0.1,
            use_edge_mlp=False,
            use_reverse_edges=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
        )

        with self.assertRaisesRegex(
            InvalidGnnArchitectureConfigurationException,
            "linear classification requires",
        ):
            step.execute(self.make_dataset_context())

    def test_edge_width_is_rejected_when_edge_mlp_is_disabled(self) -> None:
        step = BuildPipelineConfigurationStep(
            gnn_architecture="aa-graphsage",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            dropout=0.1,
            use_edge_mlp=False,
            use_reverse_edges=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=256,
        )

        with self.assertRaisesRegex(
            InvalidGnnArchitectureConfigurationException,
            "cannot be used when use_edge_mlp is disabled",
        ):
            step.execute(self.make_dataset_context())

    def test_invalid_main_llm_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="invalid-model",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidMainLlmSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_subgraph_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="invalid",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidSubgraphConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_context_strategy_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="invalid",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidContextConstructionSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_gnn_layer_count_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=9,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidGnnLayerCountSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_node_classifier_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="invalid",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="text-embedding-3-small",
        )

        with self.assertRaises(InvalidNodeClassifierSelectionException):
            step.execute(self.make_dataset_context())

    def test_invalid_embedding_model_constructor_value_raises(self) -> None:
        step = BuildPipelineConfigurationStep(
            main_llm_model="gpt-5.4",
            subgraph_algorithm="shortest_path",
            context_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=256,
            node_classifier="mlp",
            question_embedding_model="text-embedding-3-small",
            relation_embedding_model="text-embedding-3-small",
            entity_embedding_model="invalid",
        )

        with self.assertRaises(InvalidEntityEmbeddingModelSelectionException):
            step.execute(self.make_dataset_context())

    def test_interactive_invalid_numeric_input_reprompts(self) -> None:
        answers = iter(["abc", "1", "1", "2", "1", "2", "1", "1", "1", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)

    def test_interactive_out_of_range_input_reprompts(self) -> None:
        answers = iter(["99", "1", "1", "2", "1", "2", "1", "1", "1", "1", "1", "1"])
        step = BuildPipelineConfigurationStep(input_func=lambda _: next(answers))

        result = step.execute(self.make_dataset_context())

        self.assertEqual(result.gnn_layer_count, 2)

    def test_configuration_result_is_stored_in_result_bank(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[
                BuildPipelineConfigurationStep(
                    main_llm_model="gpt-5.4",
                    subgraph_algorithm="shortest_path",
                    context_strategy="structured_triples",
                    gnn_layer_count=2,
                    gnn_hidden_dimension=256,
                    node_classifier="mlp",
                    question_embedding_model="text-embedding-3-small",
                    relation_embedding_model="text-embedding-3-small",
                    entity_embedding_model="text-embedding-3-small",
                )
            ]
        )

        result = pipeline.prepare(self.make_dataset_context())

        self.assertTrue(result.success)
        self.assertTrue(
            pipeline.context_builder.has_stored_result(type(result.final_result))
        )


if __name__ == "__main__":
    unittest.main()
