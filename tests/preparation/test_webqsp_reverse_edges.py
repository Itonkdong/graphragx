"""Tests for optional WebQSP reverse-edge graph processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPEntityMappingSummary,
    WebQSPVocabularyStore,
)
from pipeline.preparation.services.webqsp_local_graph_processing import (
    WebQSPLocalGraphProcessorService,
)
from pipeline.preparation.services.webqsp_local_graph_storage import (
    WebQSPLocalGraphStorageService,
)


class FakeEntityNameMappingService:
    def resolve_entity(self, entity: str) -> str:
        return entity

    def reset_summary(self) -> None:
        pass

    def build_summary(self) -> WebQSPEntityMappingSummary:
        return WebQSPEntityMappingSummary()


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class WebQSPReverseEdgeProcessingTests(unittest.TestCase):
    def make_processor(self) -> WebQSPLocalGraphProcessorService:
        return WebQSPLocalGraphProcessorService(
            entity_name_mapping_service=FakeEntityNameMappingService()
        )

    def test_without_reverse_edges_keeps_original_edges(self) -> None:
        vocabulary = WebQSPVocabularyStore()

        instance = self.make_processor().process_row(
            row={
                "question": "q",
                "q_entity": ["A"],
                "a_entity": ["B"],
                "graph": [["A", "people.person.parents", "B"]],
            },
            vocabulary_store=vocabulary,
            use_reverse_edges=False,
        )

        self.assertEqual(instance.edge_relations, ["people.person.parents"])
        self.assertEqual(instance.edge_index.tolist(), [[0], [1]])
        self.assertNotIn("reverse__people.person.parents", vocabulary.relations)

    def test_reverse_edges_double_edges_and_add_reverse_relation(self) -> None:
        vocabulary = WebQSPVocabularyStore()

        instance = self.make_processor().process_row(
            row={
                "question": "q",
                "q_entity": ["A"],
                "a_entity": ["B"],
                "graph": [["A", "people.person.parents", "B"]],
            },
            vocabulary_store=vocabulary,
            use_reverse_edges=True,
        )

        self.assertEqual(
            instance.edge_relations,
            ["people.person.parents", "reverse__people.person.parents"],
        )
        self.assertEqual(instance.edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertIn("people.person.parents", vocabulary.relations)
        self.assertIn("reverse__people.person.parents", vocabulary.relations)

    def test_reverse_edge_cache_variant_does_not_match_normal_cache(self) -> None:
        storage = WebQSPLocalGraphStorageService()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=True,
                train_instances=[],
                test_instances=[],
                vocabulary_store=WebQSPVocabularyStore(),
                cache_directory=cache_directory,
            )
            storage.save(dataset)
            metadata = storage._build_metadata(dataset)

        self.assertTrue(
            storage._is_valid_metadata(
                metadata=metadata,
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=True,
            )
        )
        self.assertFalse(
            storage._is_valid_metadata(
                metadata=metadata,
                dataset_id="WebQSP",
                processing_version="test",
                use_reverse_edges=False,
            )
        )

    def test_packed_cache_round_trip_selective_loading_and_provenance(self) -> None:
        class TemporaryStorage(WebQSPLocalGraphStorageService):
            def __init__(self, root: Path):
                self.root = root

            def get_cache_directory(self, dataset_id, use_reverse_edges=False):
                return self.root

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            storage = TemporaryStorage(cache_directory)
            vocabulary = WebQSPVocabularyStore()
            train_instance = self.make_processor().process_row(
                row={
                    "question": "train question",
                    "q_entity": ["A"],
                    "a_entity": ["B"],
                    "graph": [["A", "parent", "B"]],
                },
                vocabulary_store=vocabulary,
                use_reverse_edges=True,
            )
            test_instance = self.make_processor().process_row(
                row={
                    "question": "test question",
                    "q_entity": ["C"],
                    "a_entity": ["D"],
                    "graph": [["C", "child", "D"]],
                },
                vocabulary_store=vocabulary,
                use_reverse_edges=True,
            )
            dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="6",
                use_reverse_edges=True,
                train_instances=[train_instance],
                test_instances=[test_instance],
                vocabulary_store=vocabulary,
                source_fingerprints={"train": "train-fingerprint"},
                entity_mapping_sha256="mapping-hash",
                cache_directory=cache_directory,
            )
            storage.save(dataset)

            payload = torch.load(
                cache_directory / storage.train_instances_filename,
                weights_only=False,
            )
            self.assertEqual(payload["storage_format"], storage.storage_format)
            self.assertEqual(payload["edge_index"].shape, (2, 2))

            loaded = storage.load_if_available(
                dataset_id="WebQSP",
                processing_version="6",
                use_reverse_edges=True,
                load_train_instances=False,
                load_test_instances=True,
                source_fingerprints={"train": "train-fingerprint"},
                entity_mapping_sha256="mapping-hash",
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.train_instances, [])
            self.assertEqual(len(loaded.test_instances), 1)
            self.assertEqual(loaded.test_instances[0].node2id, {"C": 0, "D": 1})
            self.assertEqual(
                loaded.test_instances[0].edge_index.tolist(),
                [[0, 1], [1, 0]],
            )

            stale = storage.load_if_available(
                dataset_id="WebQSP",
                processing_version="6",
                use_reverse_edges=True,
                source_fingerprints={"train": "changed"},
                entity_mapping_sha256="mapping-hash",
            )
            self.assertIsNone(stale)


if __name__ == "__main__":
    unittest.main()
