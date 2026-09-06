"""Storage service for cached WebQSP processed graph artifacts."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import (
    ProcessedDatasetStorageException,
    UnsupportedDatasetProcessorException,
)
from pipeline.preparation.helpers.dataset_definitions import (
    DATASET_LOADERS,
    WEBQSP_DATASET_ID,
)
from helpers.constants import WEBQSP_QUESTION_VOCABULARY_FILENAME
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPEntityMappingSummary,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.services import AbstractService

logger = get_logger(__name__)


class WebQSPLocalGraphStorageService(AbstractService):
    """Persist and load processed WebQSP graph artifacts."""

    metadata_filename = "metadata.json"
    train_instances_filename = "train_instances.pt"
    test_instances_filename = "test_instances.pt"
    nodes_filename = "nodes.json"
    relations_filename = "relations.json"
    questions_filename = WEBQSP_QUESTION_VOCABULARY_FILENAME
    storage_format = "packed-webqsp-v1"

    def get_cache_directory(
        self,
        dataset_id: str,
        use_reverse_edges: bool = False,
    ) -> Path:
        """Return the processed cache directory for a supported dataset."""
        if dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetProcessorException(
                f"Unsupported processed dataset cache for dataset: {dataset_id}"
            )

        directory_name = "processed_reverse_edges" if use_reverse_edges else "processed"
        return DATASET_LOADERS[dataset_id].cache_root / directory_name

    def load_if_available(
        self,
        dataset_id: str,
        processing_version: str,
        use_reverse_edges: bool = False,
        load_train_instances: bool = True,
        load_test_instances: bool = True,
        source_fingerprints: dict[str, str] | None = None,
        entity_mapping_sha256: str | None = None,
        profile: bool = False,
    ) -> PreparedWebQSPGraphDataset | None:
        """Load cached processed artifacts when cache metadata is valid."""
        cache_directory = self.get_cache_directory(
            dataset_id,
            use_reverse_edges=use_reverse_edges,
        )
        metadata_path = cache_directory / self.metadata_filename
        train_instances_path = cache_directory / self.train_instances_filename
        test_instances_path = cache_directory / self.test_instances_filename
        nodes_path = cache_directory / self.nodes_filename
        relations_path = cache_directory / self.relations_filename
        questions_path = cache_directory / self.questions_filename

        required_paths = [
            metadata_path,
            train_instances_path,
            test_instances_path,
            nodes_path,
            relations_path,
            questions_path,
        ]
        missing_paths = [path.name for path in required_paths if not path.exists()]
        if missing_paths:
            logger.info(
                "WebQSP graph cache is incomplete: "
                f"missing_files={missing_paths} directory={cache_directory}"
            )
            return None

        load_started_at = time.perf_counter()
        try:
            phase_started_at = time.perf_counter()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_seconds = time.perf_counter() - phase_started_at
            if not self._is_valid_metadata(
                metadata=metadata,
                dataset_id=dataset_id,
                processing_version=processing_version,
                use_reverse_edges=use_reverse_edges,
                source_fingerprints=source_fingerprints,
                entity_mapping_sha256=entity_mapping_sha256,
            ):
                logger.info(
                    "WebQSP graph cache metadata does not match current inputs: "
                    f"cached_processing_version={metadata.get('processing_version')} "
                    f"required_processing_version={processing_version} "
                    f"cached_storage_format={metadata.get('storage_format')} "
                    f"required_storage_format={self.storage_format}"
                )
                return None

            phase_started_at = time.perf_counter()
            train_instances = (
                self.load_instances(train_instances_path)
                if load_train_instances
                else []
            )
            train_seconds = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            test_instances = (
                self.load_instances(test_instances_path)
                if load_test_instances
                else []
            )
            test_seconds = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            vocabulary_store = WebQSPVocabularyStore(
                nodes=json.loads(nodes_path.read_text(encoding="utf-8")),
                relations=json.loads(relations_path.read_text(encoding="utf-8")),
                questions=json.loads(questions_path.read_text(encoding="utf-8")),
            )
            vocabulary_seconds = time.perf_counter() - phase_started_at

            if load_train_instances and metadata["train_size"] != len(train_instances):
                return None

            if load_test_instances and metadata["test_size"] != len(test_instances):
                return None
            if metadata["node_count"] != len(vocabulary_store.nodes):
                return None
            if metadata["relation_count"] != len(vocabulary_store.relations):
                return None
            if metadata["question_count"] != len(vocabulary_store.questions):
                return None

            dataset = PreparedWebQSPGraphDataset(
                dataset_id=dataset_id,
                processing_version=processing_version,
                use_reverse_edges=use_reverse_edges,
                train_instances=train_instances,
                test_instances=test_instances,
                vocabulary_store=vocabulary_store,
                entity_mapping_summary=self._load_entity_mapping_summary(metadata),
                source_fingerprints=dict(metadata.get("source_fingerprints", {})),
                entity_mapping_sha256=metadata.get("entity_mapping_sha256"),
                cache_directory=cache_directory,
            )
            if profile:
                logger.info(
                    "WebQSP graph cache load profile: "
                    f"metadata_ms={metadata_seconds * 1000:.2f} "
                    f"train_ms={train_seconds * 1000:.2f} "
                    f"test_ms={test_seconds * 1000:.2f} "
                    f"vocabularies_ms={vocabulary_seconds * 1000:.2f} "
                    f"total_ms={(time.perf_counter() - load_started_at) * 1000:.2f} "
                    f"train_mib={train_instances_path.stat().st_size / 1024**2:.2f} "
                    f"test_mib={test_instances_path.stat().st_size / 1024**2:.2f} "
                    f"loaded_train={load_train_instances} "
                    f"loaded_test={load_test_instances}"
                )
            return dataset
        except Exception as error:
            raise ProcessedDatasetStorageException(
                f"Failed to load processed WebQSP dataset cache: {error}"
            ) from error

    def save(
        self,
        dataset: PreparedWebQSPGraphDataset,
        profile: bool = False,
    ) -> None:
        """Save processed instances, vocabularies, and metadata."""
        cache_directory = dataset.cache_directory
        cache_directory.mkdir(parents=True, exist_ok=True)

        try:
            save_started_at = time.perf_counter()
            phase_started_at = time.perf_counter()
            self._save_instances(
                path=cache_directory / self.train_instances_filename,
                instances=dataset.train_instances,
            )
            train_seconds = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            self._save_instances(
                path=cache_directory / self.test_instances_filename,
                instances=dataset.test_instances,
            )
            test_seconds = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            self._write_text_atomic(
                cache_directory / self.nodes_filename,
                json.dumps(dataset.vocabulary_store.nodes, indent=2, sort_keys=True),
            )
            self._write_text_atomic(
                cache_directory / self.relations_filename,
                json.dumps(dataset.vocabulary_store.relations, indent=2, sort_keys=True),
            )
            self._write_text_atomic(
                cache_directory / self.questions_filename,
                json.dumps(dataset.vocabulary_store.questions, indent=2, sort_keys=True),
            )
            vocabulary_seconds = time.perf_counter() - phase_started_at
            self._write_text_atomic(
                cache_directory / self.metadata_filename,
                json.dumps(self._build_metadata(dataset), indent=2, sort_keys=True),
            )
            if profile:
                logger.info(
                    "WebQSP graph cache save profile: "
                    f"train_ms={train_seconds * 1000:.2f} "
                    f"test_ms={test_seconds * 1000:.2f} "
                    f"vocabularies_ms={vocabulary_seconds * 1000:.2f} "
                    f"total_ms={(time.perf_counter() - save_started_at) * 1000:.2f}"
                )
        except Exception as error:
            raise ProcessedDatasetStorageException(
                f"Failed to save processed WebQSP dataset cache: {error}"
            ) from error

    @staticmethod
    def _is_valid_metadata(
        metadata: dict,
        dataset_id: str,
        processing_version: str,
        use_reverse_edges: bool,
        source_fingerprints: dict[str, str] | None = None,
        entity_mapping_sha256: str | None = None,
    ) -> bool:
        """Return whether metadata matches the requested processed dataset."""
        return (
            metadata.get("dataset_id") == dataset_id
            and metadata.get("processing_version") == processing_version
            and metadata.get("storage_format")
            == WebQSPLocalGraphStorageService.storage_format
            and bool(metadata.get("use_reverse_edges", False)) == use_reverse_edges
            and isinstance(metadata.get("train_size"), int)
            and isinstance(metadata.get("test_size"), int)
            and isinstance(metadata.get("node_count"), int)
            and isinstance(metadata.get("relation_count"), int)
            and isinstance(metadata.get("question_count"), int)
            and (
                not source_fingerprints
                or metadata.get("source_fingerprints") == source_fingerprints
            )
            and (
                entity_mapping_sha256 is None
                or metadata.get("entity_mapping_sha256")
                == entity_mapping_sha256
            )
        )

    @staticmethod
    def _build_metadata(dataset: PreparedWebQSPGraphDataset) -> dict:
        """Build persisted metadata for a processed dataset."""
        return {
            "dataset_id": dataset.dataset_id,
            "processing_version": dataset.processing_version,
            "storage_format": WebQSPLocalGraphStorageService.storage_format,
            "use_reverse_edges": dataset.use_reverse_edges,
            "train_size": dataset.train_size,
            "test_size": dataset.test_size,
            "node_count": len(dataset.vocabulary_store.nodes),
            "relation_count": len(dataset.vocabulary_store.relations),
            "question_count": len(dataset.vocabulary_store.questions),
            "source_fingerprints": dataset.source_fingerprints,
            "entity_mapping_sha256": dataset.entity_mapping_sha256,
            "entity_mapping": dataset.entity_mapping_summary.model_dump(),
        }

    @staticmethod
    def _load_entity_mapping_summary(metadata: dict) -> WebQSPEntityMappingSummary:
        """Load entity mapping summary from persisted metadata when available."""
        entity_mapping = metadata.get("entity_mapping")
        if not isinstance(entity_mapping, dict):
            return WebQSPEntityMappingSummary()

        return WebQSPEntityMappingSummary(**entity_mapping)

    def load_instances(self, path: Path) -> list[WebQSPProcessedInstance]:
        """Load packed graph instances with tensor storages mapped from disk."""
        import torch

        try:
            payload = torch.load(path, weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            try:
                payload = torch.load(path, weights_only=False)
            except TypeError:
                payload = torch.load(path)
        if isinstance(payload, list):
            return [
                item
                if isinstance(item, WebQSPProcessedInstance)
                else WebQSPProcessedInstance.model_validate(item)
                for item in payload
            ]
        if not isinstance(payload, dict) or payload.get("storage_format") != self.storage_format:
            raise ValueError(f"Unsupported WebQSP graph cache payload in {path}.")

        required_lists = (
            "questions",
            "q_entities",
            "a_entities",
            "nodes",
            "edge_relations",
        )
        instance_count = len(payload["questions"])
        if any(len(payload[key]) != instance_count for key in required_lists):
            raise ValueError(f"Misaligned WebQSP graph cache lists in {path}.")
        edge_offsets = payload["edge_offsets"]
        node_offsets = payload["node_offsets"]
        if edge_offsets.shape != (instance_count + 1,) or node_offsets.shape != (
            instance_count + 1,
        ):
            raise ValueError(f"Invalid WebQSP graph cache offsets in {path}.")
        if int(edge_offsets[-1]) != payload["edge_index"].shape[1]:
            raise ValueError(f"WebQSP edge offsets do not cover the payload in {path}.")
        if int(node_offsets[-1]) != payload["node_labels"].shape[0]:
            raise ValueError(f"WebQSP node offsets do not cover the payload in {path}.")

        instances: list[WebQSPProcessedInstance] = []
        for index in range(instance_count):
            edge_start = int(edge_offsets[index])
            edge_end = int(edge_offsets[index + 1])
            node_start = int(node_offsets[index])
            node_end = int(node_offsets[index + 1])
            nodes = payload["nodes"][index]
            edge_relations = payload["edge_relations"][index]
            if node_end - node_start != len(nodes):
                raise ValueError(f"Invalid node offsets for instance {index} in {path}.")
            if edge_end - edge_start != len(edge_relations):
                raise ValueError(f"Invalid edge offsets for instance {index} in {path}.")
            instances.append(
                WebQSPProcessedInstance.model_construct(
                    question=payload["questions"][index],
                    q_entity=payload["q_entities"][index],
                    a_entity=payload["a_entities"][index],
                    nodes=nodes,
                    node2id={node: node_id for node_id, node in enumerate(nodes)},
                    edge_index=payload["edge_index"][:, edge_start:edge_end],
                    edge_relations=edge_relations,
                    node_labels=payload["node_labels"][node_start:node_end],
                )
            )
        return instances

    def _save_instances(
        self,
        path: Path,
        instances: list[WebQSPProcessedInstance],
    ) -> None:
        """Pack instance tensors into contiguous storages and save atomically."""
        import torch

        edge_parts = []
        label_parts = []
        edge_offsets = [0]
        node_offsets = [0]
        for instance in instances:
            edge_index = instance.edge_index.detach().cpu().long().contiguous()
            node_labels = instance.node_labels.detach().cpu().float().contiguous()
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("WebQSP edge_index must have shape [2, edges].")
            if edge_index.shape[1] != len(instance.edge_relations):
                raise ValueError("WebQSP edge relations must align with edge_index.")
            if node_labels.shape != (len(instance.nodes),):
                raise ValueError("WebQSP node labels must align with nodes.")
            edge_parts.append(edge_index)
            label_parts.append(node_labels)
            edge_offsets.append(edge_offsets[-1] + edge_index.shape[1])
            node_offsets.append(node_offsets[-1] + node_labels.shape[0])

        payload = {
            "storage_format": self.storage_format,
            "questions": [instance.question for instance in instances],
            "q_entities": [instance.q_entity for instance in instances],
            "a_entities": [instance.a_entity for instance in instances],
            "nodes": [instance.nodes for instance in instances],
            "edge_relations": [instance.edge_relations for instance in instances],
            "edge_offsets": torch.tensor(edge_offsets, dtype=torch.long),
            "node_offsets": torch.tensor(node_offsets, dtype=torch.long),
            "edge_index": (
                torch.cat(edge_parts, dim=1)
                if edge_parts
                else torch.empty((2, 0), dtype=torch.long)
            ),
            "node_labels": (
                torch.cat(label_parts)
                if label_parts
                else torch.empty(0, dtype=torch.float32)
            ),
        }
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _write_text_atomic(path: Path, value: str) -> None:
        """Write a UTF-8 text artifact atomically."""
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(value, encoding="utf-8")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
