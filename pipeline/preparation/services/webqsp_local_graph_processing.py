"""Services for converting WebQSP rows into processed graph tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import (
    MalformedWebQSPExampleException,
    UnsupportedDatasetProcessorException,
)
from pipeline.preparation.helpers.dataset_definitions import WEBQSP_DATASET_ID
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
    WebQSPVocabularyStore,
)
from pipeline.services import AbstractService
from pipeline.preparation.services.webqsp_entity_name_mapping import WebQSPEntityNameMappingService

if TYPE_CHECKING:
    from pipeline.preparation.steps.dataset_loading import LoadedDataset

logger = get_logger(__name__)


class WebQSPLocalGraphProcessorService(AbstractService):
    """Convert loaded WebQSP rows into trainable processed instances."""

    def __init__(
        self,
        entity_name_mapping_service: WebQSPEntityNameMappingService | None = None,
    ):
        self.entity_name_mapping_service = (
            entity_name_mapping_service or WebQSPEntityNameMappingService()
        )

    def process_loaded_dataset(
        self,
        loaded_dataset: LoadedDataset,
        processing_version: str,
        use_reverse_edges: bool,
        cache_directory: Path,
        source_fingerprints: dict[str, str] | None = None,
        entity_mapping_sha256: str | None = None,
        profile: bool = False,
    ) -> PreparedWebQSPGraphDataset:
        """Process WebQSP train, validation, and test splits."""
        if loaded_dataset.dataset_id != WEBQSP_DATASET_ID:
            raise UnsupportedDatasetProcessorException(
                f"Unsupported dataset processor for dataset: {loaded_dataset.dataset_id}"
            )

        vocabulary_store = WebQSPVocabularyStore()
        self.entity_name_mapping_service.reset_summary()
        logger.info(f"Processing WebQSP train split into local graph instances")
        phase_started_at = time.perf_counter()
        train_instances = self._process_split(
            rows=loaded_dataset.hugging_face_dataset["train"],
            vocabulary_store=vocabulary_store,
            use_reverse_edges=use_reverse_edges,
        )
        train_seconds = time.perf_counter() - phase_started_at
        test_rows = self._combined_test_rows(loaded_dataset.hugging_face_dataset)
        logger.info(f"Processing WebQSP validation+test split into local graph instances")
        phase_started_at = time.perf_counter()
        test_instances = self._process_split(
            rows=test_rows,
            vocabulary_store=vocabulary_store,
            use_reverse_edges=use_reverse_edges,
        )
        test_seconds = time.perf_counter() - phase_started_at
        entity_mapping_summary = self.entity_name_mapping_service.build_summary()
        logger.info(
            f"Finished WebQSP entity mapping: "
            f"total_references={entity_mapping_summary.total_entity_references} "
            f"mapped_references={entity_mapping_summary.mapped_entity_references} "
            f"disambiguated_references={entity_mapping_summary.disambiguated_entity_references} "
            f"unmapped_mid_references={entity_mapping_summary.unmapped_mid_entity_references} "
            f"unique_mapped_mids={entity_mapping_summary.unique_mapped_mid_count} "
            f"unique_disambiguated_mids={entity_mapping_summary.unique_disambiguated_mid_count} "
            f"unique_unmapped_mids={entity_mapping_summary.unique_unmapped_mid_count}"
        )
        if entity_mapping_summary.unique_unmapped_mid_count > 0:
            logger.warning(
                f"Unmapped WebQSP MID-like entities remain after processing: "
                f"samples={entity_mapping_summary.unmapped_mid_samples}"
            )
        if profile:
            logger.info(
                "WebQSP graph processing profile: "
                f"train_ms={train_seconds * 1000:.2f} "
                f"test_ms={test_seconds * 1000:.2f} "
                f"train_instances={len(train_instances)} "
                f"test_instances={len(test_instances)}"
            )

        return PreparedWebQSPGraphDataset(
            dataset_id=loaded_dataset.dataset_id,
            processing_version=processing_version,
            use_reverse_edges=use_reverse_edges,
            train_instances=train_instances,
            test_instances=test_instances,
            vocabulary_store=vocabulary_store,
            entity_mapping_summary=entity_mapping_summary,
            source_fingerprints=source_fingerprints or {},
            entity_mapping_sha256=entity_mapping_sha256,
            cache_directory=cache_directory,
        )

    def _process_split(
        self,
        rows: Iterable[Mapping[str, Any]],
        vocabulary_store: WebQSPVocabularyStore,
        use_reverse_edges: bool,
    ) -> list[WebQSPProcessedInstance]:
        """Process every row from one logical split."""
        return [
            self.process_row(
                row=row,
                vocabulary_store=vocabulary_store,
                use_reverse_edges=use_reverse_edges,
            )
            for row in rows
        ]

    def process_row(
        self,
        row: Mapping[str, Any],
        vocabulary_store: WebQSPVocabularyStore,
        use_reverse_edges: bool = False,
    ) -> WebQSPProcessedInstance:
        """Convert one WebQSP row into a processed graph instance."""
        self._validate_row(row)

        question = str(row["question"])
        self._get_or_add_vocabulary_item(question, vocabulary_store.questions)
        q_entity = self._resolve_entities(
            self._normalize_string_list(row["q_entity"], "q_entity")
        )
        a_entity = self._resolve_entities(
            self._normalize_string_list(row["a_entity"], "a_entity")
        )
        triples = self._resolve_entity_triples(self._normalize_triples(row["graph"]))

        nodes: list[str] = []
        node2id: dict[str, int] = {}
        edge_relations: list[str] = []
        edge_sources: list[int] = []
        edge_targets: list[int] = []

        for head, relation, tail in triples:
            head_id = self._get_or_add_local_node(head, nodes, node2id)
            tail_id = self._get_or_add_local_node(tail, nodes, node2id)
            self._get_or_add_vocabulary_item(head, vocabulary_store.nodes)
            self._get_or_add_vocabulary_item(tail, vocabulary_store.nodes)
            self._get_or_add_vocabulary_item(relation, vocabulary_store.relations)

            edge_sources.append(head_id)
            edge_targets.append(tail_id)
            edge_relations.append(relation)
            if use_reverse_edges:
                reverse_relation = self._reverse_relation(relation)
                self._get_or_add_vocabulary_item(
                    reverse_relation,
                    vocabulary_store.relations,
                )
                edge_sources.append(tail_id)
                edge_targets.append(head_id)
                edge_relations.append(reverse_relation)

        import torch

        edge_index = torch.tensor(
            [edge_sources, edge_targets],
            dtype=torch.long,
        )
        answer_entities = set(a_entity)
        node_labels = torch.tensor(
            [1.0 if node in answer_entities else 0.0 for node in nodes],
            dtype=torch.float,
        )

        return WebQSPProcessedInstance(
            question=question,
            q_entity=q_entity,
            a_entity=a_entity,
            nodes=nodes,
            node2id=node2id,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_labels=node_labels,
        )

    @staticmethod
    def _reverse_relation(relation: str) -> str:
        """Return the stable readable reverse relation identifier."""
        return f"reverse__{relation}"

    def _resolve_entities(self, entities: list[str]) -> list[str]:
        """Resolve entity IDs to readable names while preserving list order."""
        return [
            self.entity_name_mapping_service.resolve_entity(entity)
            for entity in entities
        ]

    def _resolve_entity_triples(
        self,
        triples: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Resolve only triple endpoints, keeping relation text unchanged."""
        return [
            (
                self.entity_name_mapping_service.resolve_entity(head),
                relation,
                self.entity_name_mapping_service.resolve_entity(tail),
            )
            for head, relation, tail in triples
        ]

    @staticmethod
    def _combined_test_rows(dataset: Mapping[str, Iterable[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
        """Return validation and test rows as one evaluation split."""
        rows: list[Mapping[str, Any]] = []
        rows.extend(dataset["validation"])
        rows.extend(dataset["test"])
        return rows

    @staticmethod
    def _validate_row(row: Mapping[str, Any]) -> None:
        """Validate the required WebQSP row fields."""
        required_fields = ["question", "q_entity", "a_entity", "graph"]
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise MalformedWebQSPExampleException(
                f"WebQSP row is missing fields: {', '.join(missing_fields)}"
            )

    @staticmethod
    def _normalize_string_list(value: Any, field_name: str) -> list[str]:
        """Normalize WebQSP entity fields into string lists."""
        if isinstance(value, str):
            return [value]

        if not isinstance(value, Sequence):
            raise MalformedWebQSPExampleException(
                f"WebQSP field {field_name} must be a string or sequence of strings."
            )

        return [str(item) for item in value]

    @staticmethod
    def _normalize_triples(value: Any) -> list[tuple[str, str, str]]:
        """Normalize WebQSP graph triples into typed string tuples."""
        if not isinstance(value, Sequence):
            raise MalformedWebQSPExampleException(
                "WebQSP field graph must be a sequence of triples."
            )

        triples: list[tuple[str, str, str]] = []
        for triple in value:
            if not isinstance(triple, Sequence) or len(triple) != 3:
                raise MalformedWebQSPExampleException(
                    "Every WebQSP graph item must be a triple."
                )

            triples.append((str(triple[0]), str(triple[1]), str(triple[2])))

        return triples

    @staticmethod
    def _get_or_add_local_node(
        node: str,
        nodes: list[str],
        node2id: dict[str, int],
    ) -> int:
        """Return a local node id, adding the node when needed."""
        if node not in node2id:
            node2id[node] = len(nodes)
            nodes.append(node)

        return node2id[node]

    @staticmethod
    def _get_or_add_vocabulary_item(item: str, vocabulary: dict[str, int]) -> int:
        """Return a vocabulary id, adding the item when needed."""
        if item not in vocabulary:
            vocabulary[item] = len(vocabulary)

        return vocabulary[item]
