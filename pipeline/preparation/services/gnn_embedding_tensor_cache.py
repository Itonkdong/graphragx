"""Incremental local tensor cache for GNN training embeddings."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Iterator

from helpers.constants import GNN_TRAINING_EMBEDDING_TENSOR_CACHE_DIRECTORY
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import GnnAnswerRetrieverTrainingException
from pipeline.preparation.models.gnn_embedding_tensor_cache import (
    GnnEmbeddingTensorCacheManifest,
    GnnEmbeddingTensorShardReference,
)
from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from pipeline.services import AbstractService

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch import Tensor, dtype as TorchDtype


class GnnEmbeddingTensorCacheService(AbstractService):
    """Persist embedding vectors as append-only shards and reuse local hits."""

    schema_version = 1
    manifest_filename = "manifest.json"
    lock_filename = ".cache.lock"

    def __init__(
        self,
        embedding_cache_service: WebQSPEmbeddingCacheService,
    ) -> None:
        self.embedding_cache_service = embedding_cache_service

    def load_matrix(
        self,
        torch: ModuleType,
        cache_root: Path,
        cache: TextEmbeddingCache,
        texts: list[str],
        dtype: TorchDtype,
        dtype_name: str,
        device: str,
        preprocess: bool = False,
    ) -> Tensor:
        """Load requested vectors locally and append only Qdrant misses."""
        cache_directory = self._cache_directory(
            cache_root=cache_root,
            cache=cache,
            dtype_name=dtype_name,
        )
        started_at = time.perf_counter()
        with self._exclusive_lock(cache_directory):
            manifest = self._load_manifest(
                cache_directory=cache_directory,
                cache=cache,
                dtype_name=dtype_name,
            )
            matrix = torch.empty(
                (len(texts), cache.vector_size),
                dtype=dtype,
                device=device,
            )
            point_positions, text_by_point_id = self._index_requested_texts(
                cache=cache,
                texts=texts,
            )
            missing_point_ids = self._load_local_hits(
                torch=torch,
                cache_directory=cache_directory,
                manifest=manifest,
                matrix=matrix,
                point_positions=point_positions,
                dtype=dtype,
                device=device,
            )
            missing_texts = [
                text_by_point_id[point_id] for point_id in missing_point_ids
            ]
            if missing_texts:
                self.embedding_cache_service.ensure_cache_available(cache)
                self.embedding_cache_service.ensure_embeddings(
                    cache=cache,
                    texts=missing_texts,
                    preprocess=preprocess,
                )
                self._fetch_and_append_missing(
                    torch=torch,
                    cache_directory=cache_directory,
                    cache=cache,
                    manifest=manifest,
                    matrix=matrix,
                    missing_point_ids=missing_point_ids,
                    text_by_point_id=text_by_point_id,
                    point_positions=point_positions,
                    dtype=dtype,
                    device=device,
                )
                self._write_manifest_atomic(
                    cache_directory=cache_directory,
                    manifest=manifest,
                )

        local_hit_count = len(point_positions) - len(missing_point_ids)
        logger.info(
            f"Incremental local embedding cache {cache.cache_kind}/{cache.model_id}: "
            f"requested={len(point_positions)} local_hits={local_hit_count} "
            f"qdrant_misses={len(missing_point_ids)} "
            f"elapsed_seconds={time.perf_counter() - started_at:.2f}"
        )
        return matrix

    def _load_local_hits(
        self,
        torch: ModuleType,
        cache_directory: Path,
        manifest: GnnEmbeddingTensorCacheManifest,
        matrix: Tensor,
        point_positions: dict[str, list[int]],
        dtype: TorchDtype,
        device: str,
    ) -> list[str]:
        """Copy valid local shard rows and return point IDs requiring Qdrant."""
        references_by_shard: dict[
            str, list[tuple[str, GnnEmbeddingTensorShardReference]]
        ] = {}
        missing_point_ids: list[str] = []
        for point_id in point_positions:
            reference = manifest.entries.get(point_id)
            if reference is None:
                missing_point_ids.append(point_id)
                continue
            references_by_shard.setdefault(reference.shard_filename, []).append(
                (point_id, reference)
            )

        for shard_filename, references in references_by_shard.items():
            shard_path = cache_directory / shard_filename
            try:
                shard = torch.load(
                    shard_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
                self._validate_shard(
                    shard=shard,
                    expected_vector_size=manifest.vector_size,
                    references=references,
                )
            except Exception as error:
                logger.warning(
                    f"Ignoring invalid local embedding shard {shard_path}: {error}"
                )
                invalid_point_ids = {
                    point_id
                    for point_id, reference in manifest.entries.items()
                    if reference.shard_filename == shard_filename
                }
                for point_id in invalid_point_ids:
                    manifest.entries.pop(point_id, None)
                missing_point_ids.extend(
                    point_id for point_id, _ in references
                )
                continue

            source_rows: list[int] = []
            destination_rows: list[int] = []
            for point_id, reference in references:
                for destination_row in point_positions[point_id]:
                    source_rows.append(reference.row_index)
                    destination_rows.append(destination_row)
            selected_vectors = shard.index_select(
                0,
                torch.tensor(source_rows, dtype=torch.long),
            ).to(dtype=dtype)
            self._copy_rows_to_matrix(
                torch=torch,
                matrix=matrix,
                vectors=selected_vectors,
                destination_rows=destination_rows,
                device=device,
            )

        return list(dict.fromkeys(missing_point_ids))

    def _fetch_and_append_missing(
        self,
        torch: ModuleType,
        cache_directory: Path,
        cache: TextEmbeddingCache,
        manifest: GnnEmbeddingTensorCacheManifest,
        matrix: Tensor,
        missing_point_ids: list[str],
        text_by_point_id: dict[str, str],
        point_positions: dict[str, list[int]],
        dtype: TorchDtype,
        device: str,
    ) -> None:
        """Retrieve missing vectors in bounded batches and append tensor shards."""
        batch_size = self.embedding_cache_service.batch_size
        total_batches = (len(missing_point_ids) + batch_size - 1) // batch_size
        for batch_index, start_index in enumerate(
            range(0, len(missing_point_ids), batch_size),
            start=1,
        ):
            batch_point_ids = missing_point_ids[start_index : start_index + batch_size]
            batch_texts = [text_by_point_id[point_id] for point_id in batch_point_ids]
            end_index = min(
                start_index + len(batch_texts),
                len(missing_point_ids),
            )
            logger.info(
                f"Fetching Qdrant {cache.cache_kind}/{cache.model_id} batch "
                f"{batch_index}/{total_batches}: vectors={len(batch_texts)} "
                f"range={start_index + 1}-{end_index}/{len(missing_point_ids)}"
            )
            vectors = self.embedding_cache_service.embeddings_for_texts(
                cache=cache,
                texts=batch_texts,
            )
            shard = torch.tensor(vectors, dtype=torch.float32).to(dtype=dtype)
            if tuple(shard.shape) != (len(batch_texts), cache.vector_size):
                raise GnnAnswerRetrieverTrainingException(
                    f"Unexpected embedding tensor shape for {cache.cache_kind}/"
                    f"{cache.model_id}: expected={(len(batch_texts), cache.vector_size)} "
                    f"actual={tuple(shard.shape)}."
                )

            shard_filename = f"shard_{manifest.next_shard_index:08d}.pt"
            self._write_shard_atomic(
                torch=torch,
                shard_path=cache_directory / shard_filename,
                shard=shard,
            )
            manifest.next_shard_index += 1
            destination_rows: list[int] = []
            expanded_source_rows: list[int] = []
            for row_index, point_id in enumerate(batch_point_ids):
                manifest.entries[point_id] = GnnEmbeddingTensorShardReference(
                    shard_filename=shard_filename,
                    row_index=row_index,
                )
                for destination_row in point_positions[point_id]:
                    expanded_source_rows.append(row_index)
                    destination_rows.append(destination_row)
            expanded_shard = shard.index_select(
                0,
                torch.tensor(expanded_source_rows, dtype=torch.long),
            )
            self._copy_rows_to_matrix(
                torch=torch,
                matrix=matrix,
                vectors=expanded_shard,
                destination_rows=destination_rows,
                device=device,
            )
            logger.info(
                f"Cached Qdrant {cache.cache_kind}/{cache.model_id} batch "
                f"{batch_index}/{total_batches}: "
                f"progress={end_index}/{len(missing_point_ids)} "
                f"shard={shard_filename}"
            )

    @staticmethod
    def _copy_rows_to_matrix(
        torch: ModuleType,
        matrix: Tensor,
        vectors: Tensor,
        destination_rows: list[int],
        device: str,
    ) -> None:
        """Copy CPU vectors into arbitrary rows of the destination matrix."""
        destination = torch.tensor(
            destination_rows,
            dtype=torch.long,
            device=device,
        )
        matrix.index_copy_(
            0,
            destination,
            vectors.to(device=device, non_blocking=True),
        )

    @staticmethod
    def _validate_shard(
        shard: Tensor,
        expected_vector_size: int,
        references: list[tuple[str, GnnEmbeddingTensorShardReference]],
    ) -> None:
        """Validate a loaded shard before copying any of its vectors."""
        if not hasattr(shard, "shape") or len(shard.shape) != 2:
            raise ValueError("Tensor shard must contain a rank-two tensor.")
        if shard.shape[1] != expected_vector_size:
            raise ValueError(
                f"Tensor shard vector size is {shard.shape[1]}, "
                f"expected {expected_vector_size}."
            )
        if any(reference.row_index >= shard.shape[0] for _, reference in references):
            raise ValueError("Tensor shard manifest row is out of bounds.")

    def _load_manifest(
        self,
        cache_directory: Path,
        cache: TextEmbeddingCache,
        dtype_name: str,
    ) -> GnnEmbeddingTensorCacheManifest:
        """Load a valid manifest or preserve and replace an invalid one."""
        manifest_path = cache_directory / self.manifest_filename
        if manifest_path.exists():
            try:
                manifest = GnnEmbeddingTensorCacheManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                self._validate_manifest_identity(
                    manifest=manifest,
                    cache=cache,
                    dtype_name=dtype_name,
                )
                return manifest
            except Exception as error:
                corrupt_path = manifest_path.with_name(
                    f"manifest.corrupt.{uuid.uuid4().hex}.json"
                )
                manifest_path.replace(corrupt_path)
                logger.warning(
                    f"Preserved invalid embedding tensor manifest at {corrupt_path}: "
                    f"{error}"
                )
        return self._new_manifest(cache=cache, dtype_name=dtype_name)

    def _new_manifest(
        self,
        cache: TextEmbeddingCache,
        dtype_name: str,
    ) -> GnnEmbeddingTensorCacheManifest:
        """Build an empty manifest for one embedding identity and dtype."""
        return GnnEmbeddingTensorCacheManifest(
            schema_version=self.schema_version,
            dataset_id=cache.dataset_id,
            model_id=cache.model_id,
            cache_kind=cache.cache_kind,
            vector_size=cache.vector_size,
            dtype_name=dtype_name,
        )

    def _validate_manifest_identity(
        self,
        manifest: GnnEmbeddingTensorCacheManifest,
        cache: TextEmbeddingCache,
        dtype_name: str,
    ) -> None:
        """Reject metadata that does not represent the requested cache."""
        actual_identity = (
            manifest.schema_version,
            manifest.dataset_id,
            manifest.model_id,
            manifest.cache_kind,
            manifest.vector_size,
            manifest.dtype_name,
        )
        expected_identity = (
            self.schema_version,
            cache.dataset_id,
            cache.model_id,
            cache.cache_kind,
            cache.vector_size,
            dtype_name,
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"Embedding tensor manifest identity mismatch: "
                f"expected={expected_identity} actual={actual_identity}."
            )

    def _cache_directory(
        self,
        cache_root: Path,
        cache: TextEmbeddingCache,
        dtype_name: str,
    ) -> Path:
        """Return a stable local directory for one embedding cache identity."""
        identity = json.dumps(
            {
                "schema_version": self.schema_version,
                "dataset_id": cache.dataset_id,
                "model_id": cache.model_id,
                "cache_kind": cache.cache_kind,
                "vector_size": cache.vector_size,
                "dtype_name": dtype_name,
            },
            sort_keys=True,
        )
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        safe_model_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cache.model_id).strip("_")
        return (
            cache_root
            / GNN_TRAINING_EMBEDDING_TENSOR_CACHE_DIRECTORY
            / cache.cache_kind
            / f"{safe_model_id}_{dtype_name}_{identity_hash}"
        )

    def _index_requested_texts(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> tuple[dict[str, list[int]], dict[str, str]]:
        """Map deterministic embedding IDs to all requested output rows."""
        point_positions: dict[str, list[int]] = {}
        text_by_point_id: dict[str, str] = {}
        for position, requested_text in enumerate(texts):
            point_id = self.embedding_cache_service.point_id(
                cache=cache,
                text=requested_text,
            )
            point_positions.setdefault(point_id, []).append(position)
            text_by_point_id[point_id] = requested_text
        return point_positions, text_by_point_id

    def _write_shard_atomic(
        self,
        torch: ModuleType,
        shard_path: Path,
        shard: Tensor,
    ) -> None:
        """Publish a tensor shard only after serialization succeeds."""
        temporary_path = shard_path.with_name(
            f".{shard_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(shard.contiguous(), temporary_path)
            temporary_path.replace(shard_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise GnnAnswerRetrieverTrainingException(
                f"Could not persist local embedding tensor shard {shard_path}: {error}"
            ) from error

    def _write_manifest_atomic(
        self,
        cache_directory: Path,
        manifest: GnnEmbeddingTensorCacheManifest,
    ) -> None:
        """Atomically publish the local shard index."""
        manifest_path = cache_directory / self.manifest_filename
        temporary_path = manifest_path.with_name(
            f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(manifest_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise GnnAnswerRetrieverTrainingException(
                f"Could not persist local embedding tensor manifest "
                f"{manifest_path}: {error}"
            ) from error

    @contextmanager
    def _exclusive_lock(self, cache_directory: Path) -> Iterator[None]:
        """Serialize readers and appenders for one local tensor cache."""
        try:
            cache_directory.mkdir(parents=True, exist_ok=True)
            lock_file = (cache_directory / self.lock_filename).open("a+b")
        except Exception as error:
            raise GnnAnswerRetrieverTrainingException(
                f"Could not access incremental embedding tensor cache "
                f"{cache_directory}: {error}"
            ) from error

        with lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception as error:
                raise GnnAnswerRetrieverTrainingException(
                    f"Could not lock incremental embedding tensor cache "
                    f"{cache_directory}: {error}"
                ) from error
            try:
                yield
            finally:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError as error:
                    logger.warning(
                        f"Could not unlock incremental embedding tensor cache "
                        f"{cache_directory}: {error}"
                    )
