"""Tests for the incremental local GNN embedding tensor cache."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.preparation.services.embedding_cache import TextEmbeddingCache
from pipeline.preparation.services.gnn_embedding_tensor_cache import (
    GnnEmbeddingTensorCacheService,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class FakeRemoteEmbeddingCacheService:
    """Deterministic remote cache that records every external operation."""

    batch_size = 2

    def __init__(self) -> None:
        self.available_checks = 0
        self.ensure_requests: list[list[str]] = []
        self.retrieval_requests: list[list[str]] = []

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
        self.ensure_requests.append(list(texts))

    def embeddings_for_texts(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> list[list[float]]:
        self.retrieval_requests.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = float(ord(text[0]))
        return [value, value + 0.5]


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class GnnEmbeddingTensorCacheServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temporary_directory.name)
        self.cache = TextEmbeddingCache(
            dataset_id="WebQSP",
            model_id="test-model",
            cache_kind="nodes",
            vocabulary={},
            collection_name="test_collection",
            vector_size=2,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_partial_hit_fetches_and_appends_only_missing_vectors(self) -> None:
        remote_cache = FakeRemoteEmbeddingCacheService()
        service = GnnEmbeddingTensorCacheService(remote_cache)

        with self.assertLogs(
            "pipeline.preparation.services.gnn_embedding_tensor_cache",
            level="INFO",
        ) as captured_logs:
            first_matrix = service.load_matrix(
                torch=torch,
                cache_root=self.cache_root,
                cache=self.cache,
                texts=["A", "B"],
                dtype=torch.float32,
                dtype_name="float32",
                device="cpu",
            )
        second_matrix = service.load_matrix(
            torch=torch,
            cache_root=self.cache_root,
            cache=self.cache,
            texts=["A", "B", "C", "D"],
            dtype=torch.float32,
            dtype_name="float32",
            device="cpu",
        )

        self.assertTrue(
            torch.equal(
                first_matrix,
                torch.tensor([[65.0, 65.5], [66.0, 66.5]]),
            )
        )
        self.assertTrue(
            torch.equal(
                second_matrix,
                torch.tensor(
                    [
                        [65.0, 65.5],
                        [66.0, 66.5],
                        [67.0, 67.5],
                        [68.0, 68.5],
                    ]
                ),
            )
        )
        self.assertEqual(remote_cache.available_checks, 2)
        self.assertEqual(remote_cache.ensure_requests, [["A", "B"], ["C", "D"]])
        self.assertEqual(remote_cache.retrieval_requests, [["A", "B"], ["C", "D"]])
        self.assertTrue(
            any(
                "Fetching Qdrant nodes/test-model batch 1/1" in message
                for message in captured_logs.output
            )
        )

        cache_directory = service._cache_directory(
            cache_root=self.cache_root,
            cache=self.cache,
            dtype_name="float32",
        )
        self.assertEqual(len(list(cache_directory.glob("shard_*.pt"))), 2)

    def test_full_local_hit_does_not_contact_remote_cache(self) -> None:
        warm_remote_cache = FakeRemoteEmbeddingCacheService()
        GnnEmbeddingTensorCacheService(warm_remote_cache).load_matrix(
            torch=torch,
            cache_root=self.cache_root,
            cache=self.cache,
            texts=["A", "B"],
            dtype=torch.float32,
            dtype_name="float32",
            device="cpu",
        )
        hit_remote_cache = FakeRemoteEmbeddingCacheService()

        matrix = GnnEmbeddingTensorCacheService(hit_remote_cache).load_matrix(
            torch=torch,
            cache_root=self.cache_root,
            cache=self.cache,
            texts=["B", "A"],
            dtype=torch.float32,
            dtype_name="float32",
            device="cpu",
        )

        self.assertTrue(
            torch.equal(
                matrix,
                torch.tensor([[66.0, 66.5], [65.0, 65.5]]),
            )
        )
        self.assertEqual(hit_remote_cache.available_checks, 0)
        self.assertEqual(hit_remote_cache.ensure_requests, [])
        self.assertEqual(hit_remote_cache.retrieval_requests, [])

    def test_corrupt_shard_is_replaced_from_remote_cache(self) -> None:
        remote_cache = FakeRemoteEmbeddingCacheService()
        service = GnnEmbeddingTensorCacheService(remote_cache)
        service.load_matrix(
            torch=torch,
            cache_root=self.cache_root,
            cache=self.cache,
            texts=["A", "B"],
            dtype=torch.float32,
            dtype_name="float32",
            device="cpu",
        )
        cache_directory = service._cache_directory(
            cache_root=self.cache_root,
            cache=self.cache,
            dtype_name="float32",
        )
        first_shard = next(cache_directory.glob("shard_*.pt"))
        first_shard.write_bytes(b"not a tensor")

        recovered_matrix = service.load_matrix(
            torch=torch,
            cache_root=self.cache_root,
            cache=self.cache,
            texts=["A", "B"],
            dtype=torch.float32,
            dtype_name="float32",
            device="cpu",
        )

        self.assertTrue(
            torch.equal(
                recovered_matrix,
                torch.tensor([[65.0, 65.5], [66.0, 66.5]]),
            )
        )
        self.assertEqual(remote_cache.ensure_requests, [["A", "B"], ["A", "B"]])
        self.assertEqual(remote_cache.retrieval_requests, [["A", "B"], ["A", "B"]])

    def test_storage_dtype_uses_a_separate_cache_identity(self) -> None:
        service = GnnEmbeddingTensorCacheService(
            FakeRemoteEmbeddingCacheService()
        )

        float32_directory = service._cache_directory(
            cache_root=self.cache_root,
            cache=self.cache,
            dtype_name="float32",
        )
        bfloat16_directory = service._cache_directory(
            cache_root=self.cache_root,
            cache=self.cache,
            dtype_name="bfloat16",
        )

        self.assertNotEqual(float32_directory, bfloat16_directory)

    def test_cache_lock_does_not_wrap_cuda_out_of_memory(self) -> None:
        service = GnnEmbeddingTensorCacheService(
            FakeRemoteEmbeddingCacheService()
        )
        cache_directory = service._cache_directory(
            cache_root=self.cache_root,
            cache=self.cache,
            dtype_name="float32",
        )

        with self.assertRaises(torch.OutOfMemoryError):
            with service._exclusive_lock(cache_directory):
                raise torch.OutOfMemoryError("synthetic allocation failure")


if __name__ == "__main__":
    unittest.main()
