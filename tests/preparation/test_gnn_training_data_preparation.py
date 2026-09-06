"""Tests for compact GNN training-data preparation helpers."""

from __future__ import annotations

import unittest

from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationService,
)


class FakeCudaRuntime:
    """CUDA memory probe used by embedding-placement tests."""

    def __init__(self, free_gib: float):
        self.free_bytes = int(free_gib * 1024**3)

    def mem_get_info(self) -> tuple[int, int]:
        return self.free_bytes, self.free_bytes


class FakeTorchRuntime:
    """Torch runtime subset used by embedding-placement tests."""

    def __init__(self, free_gib: float):
        self.cuda = FakeCudaRuntime(free_gib=free_gib)


class GnnTrainingDataPreparationServiceTests(unittest.TestCase):
    def test_compact_text_indices_deduplicates_across_instances(self) -> None:
        texts, indices = GnnTrainingDataPreparationService._compact_text_indices(
            [["A", "B", "A"], ["B", "C"]]
        )

        self.assertEqual(texts, ["A", "B", "C"])
        self.assertEqual(indices, [[0, 1, 0], [1, 2]])

    def test_embedding_storage_bytes_includes_each_matrix(self) -> None:
        storage_bytes = GnnTrainingDataPreparationService._embedding_storage_bytes(
            text_counts=(10, 3, 2),
            vector_sizes=(4, 4, 4),
            element_size=2,
        )

        self.assertEqual(storage_bytes, 120)

    def test_auto_embedding_device_uses_cuda_with_safe_reserve(self) -> None:
        service = GnnTrainingDataPreparationService()

        device = service._resolve_embedding_device(
            torch=FakeTorchRuntime(free_gib=16),
            requested_device="auto",
            selected_device="cuda",
            required_bytes=4 * 1024**3,
            reserve_gb=6,
        )

        self.assertEqual(device, "cuda")

    def test_auto_embedding_device_falls_back_when_budget_is_unsafe(self) -> None:
        service = GnnTrainingDataPreparationService()

        device = service._resolve_embedding_device(
            torch=FakeTorchRuntime(free_gib=8),
            requested_device="auto",
            selected_device="cuda",
            required_bytes=4 * 1024**3,
            reserve_gb=6,
        )

        self.assertEqual(device, "cpu")

    def test_forced_gpu_embedding_device_rejects_unsafe_budget(self) -> None:
        service = GnnTrainingDataPreparationService()

        with self.assertRaisesRegex(
            GnnAnswerRetrieverTrainingException,
            "only 8.00 GiB is free",
        ):
            service._resolve_embedding_device(
                torch=FakeTorchRuntime(free_gib=8),
                requested_device="gpu",
                selected_device="cuda",
                required_bytes=4 * 1024**3,
                reserve_gb=6,
            )


if __name__ == "__main__":
    unittest.main()
