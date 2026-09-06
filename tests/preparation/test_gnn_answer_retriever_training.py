"""Tests for GNN answer-retriever training helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.models.interfaces import AnswerRetrieverModel
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    LoadedGnnAnswerRetrieverRun,
    SavedGnnAnswerRetrieverConfig,
)
from pipeline.preparation.services.gnn_answer_retriever_training import (
    GnnAnswerRetrieverTrainingConfig,
    GnnAnswerRetrieverTrainingService,
)
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationService,
)
from pipeline.preparation.steps.gnn_model_building import BuiltGnnAnswerRetriever


class FakeAnswerRetrieverModel(AnswerRetrieverModel):
    """Small model stand-in that avoids importing torch in unit tests."""

    def state_dict(self) -> dict[str, float]:
        return {"weight": 1.0}


class FakeTorch:
    """Tiny torch.save replacement for config-writing tests."""

    @staticmethod
    def save(state_dict, path: Path) -> None:
        path.write_text(json.dumps(state_dict), encoding="utf-8")


class GnnAnswerRetrieverTrainingServiceTests(unittest.TestCase):
    def test_console_and_wandb_intervals_can_be_evaluated_independently(self) -> None:
        service = GnnAnswerRetrieverTrainingService()

        self.assertFalse(
            service._is_progress_due(
                instance_index=5,
                total_instances=20,
                interval=10,
            )
        )
        self.assertTrue(
            service._is_progress_due(
                instance_index=5,
                total_instances=20,
                interval=5,
            )
        )
        self.assertTrue(
            service._is_progress_due(
                instance_index=20,
                total_instances=20,
                interval=7,
            )
        )
        self.assertFalse(
            service._is_progress_due(
                instance_index=20,
                total_instances=20,
                interval=0,
            )
        )

    def test_select_train_instances_defaults_to_full_split(self) -> None:
        dataset = SimpleNamespace(train_instances=list(range(3)))

        selected, start, end = GnnTrainingDataPreparationService._select_instances(
            prepared_dataset=dataset,
            start_instance=0,
            max_instances=None,
        )

        self.assertEqual(selected, [0, 1, 2])
        self.assertEqual(start, 0)
        self.assertEqual(end, 3)

    def test_select_train_instances_uses_start_plus_max_semantics(self) -> None:
        dataset = SimpleNamespace(train_instances=list(range(300)))

        selected, start, end = GnnTrainingDataPreparationService._select_instances(
            prepared_dataset=dataset,
            start_instance=101,
            max_instances=100,
        )

        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[0], 101)
        self.assertEqual(selected[-1], 200)
        self.assertEqual(start, 101)
        self.assertEqual(end, 201)

    def test_select_train_instances_rejects_negative_start(self) -> None:
        dataset = SimpleNamespace(train_instances=list(range(3)))

        with self.assertRaisesRegex(
            GnnAnswerRetrieverTrainingException,
            "greater than or equal to 0",
        ):
            GnnTrainingDataPreparationService._select_instances(
                prepared_dataset=dataset,
                start_instance=-1,
                max_instances=None,
            )

    def test_select_train_instances_rejects_empty_slice(self) -> None:
        dataset = SimpleNamespace(train_instances=list(range(3)))

        with self.assertRaisesRegex(
            GnnAnswerRetrieverTrainingException,
            "selected no instances",
        ):
            GnnTrainingDataPreparationService._select_instances(
                prepared_dataset=dataset,
                start_instance=3,
                max_instances=None,
            )

    def test_save_model_config_marks_fresh_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "1_test"
            service = GnnAnswerRetrieverTrainingService()
            built_retriever = BuiltGnnAnswerRetriever(
                dataset_id="WebQSP",
                entity_embedding_model="text-embedding-3-small",
                entity_embedding_dimension=1536,
                question_embedding_dimension=1536,
                relation_embedding_dimension=1536,
                hidden_dimension=256,
                gnn_layer_count=2,
                node_classifier="mlp",
                use_edge_mlp=False,
                question_aware_classifier=False,
                use_reverse_edges=True,
                add_layer_normalization=True,
                edge_mlp_hidden_dim=64,
                dropout=0.25,
                model=FakeAnswerRetrieverModel(),
            )

            service._save_model_artifacts(
                model=built_retriever.model,
                built_retriever=built_retriever,
                question_embedding_model="text-embedding-3-small",
                relation_embedding_model="text-embedding-3-small",
                training_config=GnnAnswerRetrieverTrainingConfig(
                    epochs=1,
                    max_instances=10,
                    start_instance=5,
                ),
                selected_device="cpu",
                final_loss=0.5,
                loss_history=[{"epoch": 1, "average_loss": 0.5}],
                trained_instances=10,
                training_start_instance=5,
                training_end_instance=15,
                continued_run=None,
                model_run_directory=run_directory,
                torch=FakeTorch,
                embedding_cache_device="cuda",
                embedding_cache_dtype="bfloat16",
            )

            config = json.loads((run_directory / "model_config.json").read_text())

        self.assertFalse(config["is_fine_tuned_model"])
        self.assertNotIn("continued_from_model_run_name", config)
        self.assertNotIn("training_start_instance", config)
        self.assertNotIn("training_end_instance", config)
        self.assertNotIn("trained_instance_range", config)
        self.assertNotIn("trained_instances", config)
        self.assertEqual(
            config["training"]["trained_instances"],
            {"start": 5, "end": 15, "count": 10},
        )
        self.assertEqual(
            config["training"]["loss_history"],
            [{"epoch": 1, "average_loss": 0.5}],
        )
        self.assertEqual(config["training"]["device"], "cpu")
        self.assertEqual(config["training"]["random_seed"], 42)
        self.assertEqual(config["embedding_model"], "text-embedding-3-small")
        self.assertNotIn("entity_embedding_model", config)
        self.assertNotIn("question_embedding_model", config)
        self.assertNotIn("relation_embedding_model", config)
        self.assertNotIn("use_edge_mlp", config)
        self.assertNotIn("use_reverse_edges", config)
        self.assertNotIn("use_reverse_edges", config["training"])
        self.assertNotIn("add_layer_normalization", config["training"])
        self.assertEqual(config["training"]["embedding_cache_device"], "cuda")
        self.assertEqual(config["training"]["embedding_cache_dtype"], "bfloat16")

    def test_save_model_config_marks_continued_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "2_new"
            source_config_path = root / "1_old" / "model_config.json"
            source_weights_path = root / "1_old" / "gnn_answer_retriever.pt"
            source_config_path.parent.mkdir()
            source_config_path.write_text("{}", encoding="utf-8")
            source_weights_path.write_text("{}", encoding="utf-8")
            source_config = SavedGnnAnswerRetrieverConfig(
                dataset_id="WebQSP",
                entity_embedding_model="text-embedding-3-large",
                question_embedding_model="text-embedding-3-large",
                relation_embedding_model="text-embedding-3-large",
                entity_embedding_dimension=3072,
                hidden_dimension=512,
                gnn_layer_count=3,
                node_classifier="linear",
                training={
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "log_every": 10,
                    "device": "cpu",
                },
                run_name="1_old",
                run_number=1,
                final_loss=0.8,
                trained_instances=100,
            )
            continued_run = LoadedGnnAnswerRetrieverRun(
                run_directory=source_config_path.parent,
                run_name="1_old",
                run_number=1,
                weights_path=source_weights_path,
                config_path=source_config_path,
                config=source_config,
                model=FakeAnswerRetrieverModel(),
                question_embedding_model="text-embedding-3-large",
                relation_embedding_model="text-embedding-3-large",
            )
            built_retriever = GnnAnswerRetrieverTrainingService._build_effective_retriever(
                built_retriever=BuiltGnnAnswerRetriever(
                    dataset_id="WebQSP",
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
                ),
                continued_run=continued_run,
            )

            GnnAnswerRetrieverTrainingService()._save_model_artifacts(
                model=built_retriever.model,
                built_retriever=built_retriever,
                question_embedding_model=continued_run.question_embedding_model,
                relation_embedding_model=continued_run.relation_embedding_model,
                training_config=GnnAnswerRetrieverTrainingConfig(
                    epochs=2,
                    start_instance=101,
                    continue_from_model_run_number=1,
                ),
                selected_device="cpu",
                final_loss=0.4,
                loss_history=[{"epoch": 1, "average_loss": 0.6}],
                trained_instances=50,
                training_start_instance=101,
                training_end_instance=151,
                continued_run=continued_run,
                model_run_directory=run_directory,
                torch=FakeTorch,
            )

            config = json.loads((run_directory / "model_config.json").read_text())

        self.assertTrue(config["is_fine_tuned_model"])
        self.assertEqual(config["continued_from_model_run_name"], "1_old")
        self.assertEqual(config["continued_from_model_run_number"], 1)
        self.assertNotIn("continued_from_model_config_path", config)
        self.assertNotIn("continued_from_weights_path", config)
        self.assertEqual(config["embedding_model"], "text-embedding-3-large")
        self.assertNotIn("entity_embedding_model", config)
        self.assertNotIn("question_embedding_model", config)
        self.assertNotIn("relation_embedding_model", config)
        self.assertNotIn("hidden_dimension", config)
        self.assertNotIn("gnn_layer_count", config)
        self.assertNotIn("node_classifier", config)
        self.assertNotIn("hidden_dimension", config["training"])
        self.assertNotIn("gnn_layer_count", config["training"])
        self.assertNotIn("training_start_instance", config)
        self.assertNotIn("training_end_instance", config)
        self.assertEqual(
            config["training"]["trained_instances"],
            {"start": 101, "end": 151, "count": 50},
        )


if __name__ == "__main__":
    unittest.main()
