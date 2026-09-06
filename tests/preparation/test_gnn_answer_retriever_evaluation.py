"""Integration tests for compact cached GNN evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.evaluation.models import GnnAnswerRetrieverEvaluationConfig
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.embedding_cache import TextEmbeddingCache
from pipeline.preparation.services.gnn_answer_retriever_evaluation import (
    GnnAnswerRetrieverEvaluationService,
)
from pipeline.preparation.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStorageService,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    LoadedGnnAnswerRetrieverRun,
    SavedGnnAnswerRetrieverConfig,
    SavedGnnAnswerRetrieverTrainingConfig,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration

try:
    import torch
except ModuleNotFoundError:
    torch = None


class FakeEvaluationEmbeddingCacheService:
    """Embedding cache double that records every simulated Qdrant operation."""

    batch_size = 1024

    def __init__(self) -> None:
        self.available_checks = 0
        self.ensure_requests: list[tuple[str, list[str]]] = []
        self.retrieval_requests: list[tuple[str, list[str]]] = []
        self.cache_load_collection_flags: list[bool] = []

    def load_node_cache(self, **kwargs) -> TextEmbeddingCache:
        return self._load_cache("nodes", **kwargs)

    def load_relation_cache(self, **kwargs) -> TextEmbeddingCache:
        return self._load_cache("relations", **kwargs)

    def load_question_cache(self, **kwargs) -> TextEmbeddingCache:
        return self._load_cache("questions", **kwargs)

    def _load_cache(
        self,
        cache_kind: str,
        cache_root,
        model_id: str,
        vocabulary: dict[str, int],
        dataset_id: str,
        ensure_collection: bool,
    ) -> TextEmbeddingCache:
        self.cache_load_collection_flags.append(ensure_collection)
        return TextEmbeddingCache(
            dataset_id=dataset_id,
            model_id=model_id,
            cache_kind=cache_kind,
            vocabulary=vocabulary,
            collection_name=f"test_{cache_kind}",
            vector_size=2,
        )

    @staticmethod
    def point_id(cache: TextEmbeddingCache, text: str) -> str:
        return f"{cache.dataset_id}|{cache.model_id}|{cache.cache_kind}|{text}"

    def ensure_cache_available(self, cache: TextEmbeddingCache) -> None:
        self.available_checks += 1

    def ensure_embeddings(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
        preprocess: bool = False,
    ) -> None:
        self.ensure_requests.append((cache.cache_kind, list(texts)))

    def embeddings_for_texts(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> list[list[float]]:
        self.retrieval_requests.append((cache.cache_kind, list(texts)))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = float(sum(ord(character) for character in text) % 31) / 31.0
        return [value, 1.0 - value]


class FakeModelRunService:
    """Return one already constructed model run without disk model loading."""

    def __init__(self, loaded_run: LoadedGnnAnswerRetrieverRun) -> None:
        self.loaded_run = loaded_run

    def load_run(self, **kwargs) -> LoadedGnnAnswerRetrieverRun:
        return self.loaded_run


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class GnnAnswerRetrieverEvaluationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temporary_directory.name)
        self.dataset = self._build_dataset()
        self.pipeline_configuration = self._build_pipeline_configuration()
        self.loaded_run = self._build_loaded_run()
        self.evaluation_config = GnnAnswerRetrieverEvaluationConfig(
            model_run_number=1,
            max_instances=2,
            candidate_top_k=10,
            candidate_limit=10,
            embedding_cache_device="cpu",
            embedding_cache_dtype="float32",
            profile=True,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_second_evaluation_uses_local_tensors_without_qdrant(self) -> None:
        cold_remote_cache = FakeEvaluationEmbeddingCacheService()
        cold_service = self._build_service(cold_remote_cache)

        first_outcome = cold_service.evaluate(
            prepared_dataset=self.dataset,
            pipeline_configuration=self.pipeline_configuration,
            evaluation_config=self.evaluation_config,
        )

        self.assertEqual(first_outcome.evaluated_instances, 2)
        self.assertEqual(cold_remote_cache.available_checks, 3)
        self.assertEqual(len(cold_remote_cache.ensure_requests), 3)
        self.assertEqual(len(cold_remote_cache.retrieval_requests), 3)
        self.assertEqual(cold_remote_cache.cache_load_collection_flags, [False] * 3)
        persisted_config = json.loads(
            first_outcome.storage_result.evaluation_config_path.read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(persisted_config["embedding_cache_device"], "cpu")
        self.assertEqual(persisted_config["embedding_cache_dtype"], "float32")

        warm_remote_cache = FakeEvaluationEmbeddingCacheService()
        warm_service = self._build_service(warm_remote_cache)
        second_outcome = warm_service.evaluate(
            prepared_dataset=self.dataset,
            pipeline_configuration=self.pipeline_configuration,
            evaluation_config=self.evaluation_config,
        )

        self.assertEqual(second_outcome.evaluated_instances, 2)
        self.assertEqual(warm_remote_cache.available_checks, 0)
        self.assertEqual(warm_remote_cache.ensure_requests, [])
        self.assertEqual(warm_remote_cache.retrieval_requests, [])
        self.assertEqual(warm_remote_cache.cache_load_collection_flags, [False] * 3)

    def test_larger_evaluation_fetches_only_new_test_embeddings(self) -> None:
        remote_cache = FakeEvaluationEmbeddingCacheService()
        service = self._build_service(remote_cache)
        one_instance_config = self.evaluation_config.model_copy(
            update={"max_instances": 1}
        )
        service.evaluate(
            prepared_dataset=self.dataset,
            pipeline_configuration=self.pipeline_configuration,
            evaluation_config=one_instance_config,
        )
        retrieval_count_after_first_run = len(remote_cache.retrieval_requests)

        service.evaluate(
            prepared_dataset=self.dataset,
            pipeline_configuration=self.pipeline_configuration,
            evaluation_config=self.evaluation_config,
        )

        incremental_requests = remote_cache.retrieval_requests[
            retrieval_count_after_first_run:
        ]
        self.assertEqual(
            incremental_requests,
            [
                ("nodes", ["C"]),
                ("questions", ["question two"]),
            ],
        )

    def _build_service(
        self,
        embedding_cache_service: FakeEvaluationEmbeddingCacheService,
    ) -> GnnAnswerRetrieverEvaluationService:
        return GnnAnswerRetrieverEvaluationService(
            model_run_service=FakeModelRunService(self.loaded_run),
            embedding_cache_service=embedding_cache_service,
            storage_service=GnnAnswerRetrieverEvaluationStorageService(),
        )

    def _build_dataset(self) -> PreparedWebQSPGraphDataset:
        first_instance = WebQSPProcessedInstance(
            question="question one",
            q_entity=["A"],
            a_entity=["B"],
            nodes=["A", "B"],
            node2id={"A": 0, "B": 1},
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_relations=["relation", "reverse relation"],
            node_labels=torch.tensor([0.0, 1.0]),
        )
        second_instance = WebQSPProcessedInstance(
            question="question two",
            q_entity=["B"],
            a_entity=["C"],
            nodes=["B", "C"],
            node2id={"B": 0, "C": 1},
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_relations=["relation", "reverse relation"],
            node_labels=torch.tensor([0.0, 1.0]),
        )
        return PreparedWebQSPGraphDataset(
            dataset_id="WebQSP",
            processing_version="test",
            use_reverse_edges=True,
            train_instances=[],
            test_instances=[first_instance, second_instance],
            vocabulary_store=WebQSPVocabularyStore(
                nodes={"A": 0, "B": 1, "C": 2},
                relations={"relation": 0, "reverse relation": 1},
                questions={"question one": 0, "question two": 1},
            ),
            cache_directory=self.cache_root / "processed_reverse_edges",
        )

    @staticmethod
    def _build_pipeline_configuration() -> BuiltPipelineConfiguration:
        return BuiltPipelineConfiguration(
            dataset_id="WebQSP",
            main_llm_model="gpt-5.4",
            subgraph_construction_algorithm="shortest_path",
            context_construction_strategy="structured_triples",
            gnn_layer_count=2,
            gnn_hidden_dimension=4,
            node_classifier="mlp",
            question_embedding_model="test-model",
            relation_embedding_model="test-model",
            entity_embedding_model="test-model",
            use_edge_mlp=True,
            question_aware_classifier=True,
            use_reverse_edges=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=4,
            dropout=0.1,
        )

    def _build_loaded_run(self) -> LoadedGnnAnswerRetrieverRun:
        from pipeline.preparation.models.gnn_answer_retriever import GnnAnswerRetriever

        training_config = SavedGnnAnswerRetrieverTrainingConfig(
            epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            max_instances=2,
            start_instance=0,
            log_every=1,
            device="cpu",
            gnn_layer_count=2,
            hidden_dimension=4,
            use_edge_mlp=True,
            question_aware_classifier=True,
            use_reverse_edges=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=4,
            dropout=0.1,
        )
        saved_config = SavedGnnAnswerRetrieverConfig(
            dataset_id="WebQSP",
            entity_embedding_model="test-model",
            question_embedding_model="test-model",
            relation_embedding_model="test-model",
            entity_embedding_dimension=2,
            question_embedding_dimension=2,
            relation_embedding_dimension=2,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="mlp",
            use_edge_mlp=True,
            question_aware_classifier=True,
            use_reverse_edges=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=4,
            dropout=0.1,
            training=training_config,
            final_loss=0.5,
            trained_instances=2,
        )
        model = GnnAnswerRetriever(
            entity_embedding_dimension=2,
            question_embedding_dimension=2,
            relation_embedding_dimension=2,
            hidden_dimension=4,
            gnn_layer_count=2,
            node_classifier="mlp",
            use_edge_mlp=True,
            question_aware_classifier=True,
            add_layer_normalization=True,
            edge_mlp_hidden_dim=4,
            dropout=0.1,
        )
        model.eval()
        run_directory = self.cache_root / "models" / "1_test"
        return LoadedGnnAnswerRetrieverRun(
            run_directory=run_directory,
            run_name="1_test",
            run_number=1,
            weights_path=run_directory / "gnn_answer_retriever.pt",
            config_path=run_directory / "model_config.json",
            config=saved_config,
            model=model,
            question_embedding_model="test-model",
            relation_embedding_model="test-model",
        )


if __name__ == "__main__":
    unittest.main()
