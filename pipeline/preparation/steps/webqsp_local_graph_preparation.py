"""WebQSP processed graph dataset preparation step."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from helpers.logging_config import get_logger
from pydantic import Field

from pipeline.abstract import AbstractStep, StepContext
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.dataset_loading import LoadedDataset
from pipeline.preparation.services.webqsp_local_graph_processing import (
    WebQSPLocalGraphProcessorService,
)
from pipeline.preparation.services.webqsp_local_graph_storage import (
    WebQSPLocalGraphStorageService,
)

WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION = "6"
logger = get_logger(__name__)


class BuildWebQSPLocalGraphsStep(
    AbstractStep[PreparedWebQSPGraphDataset, LoadedDataset]
):
    """Build or load cached WebQSP processed graph instances."""

    def __init__(
        self,
        processor_service: WebQSPLocalGraphProcessorService | None = None,
        storage_service: WebQSPLocalGraphStorageService | None = None,
        load_train_instances: bool = True,
        load_test_instances: bool = True,
        profile: bool = False,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.processor_service = processor_service or WebQSPLocalGraphProcessorService()
        self.storage_service = storage_service or WebQSPLocalGraphStorageService()
        self.load_train_instances = load_train_instances
        self.load_test_instances = load_test_instances
        self.profile = profile

    def execute_default(
        self,
        context: StepContext[LoadedDataset],
    ) -> PreparedWebQSPGraphDataset:
        loaded_dataset = context.result
        if loaded_dataset is None:
            raise InvalidInteractiveConfigurationInputException(
                "WebQSP local graph preparation requires a loaded dataset."
            )
        pipeline_configuration = getattr(context, "pipeline_configuration", None)
        use_reverse_edges = bool(
            pipeline_configuration.use_reverse_edges
            if pipeline_configuration is not None
            else False
        )
        source_fingerprints = self._source_fingerprints(
            loaded_dataset.hugging_face_dataset
        )
        entity_mapping_sha256 = self._entity_mapping_sha256()

        logger.info(
            f"Preparing WebQSP graph instances for dataset={loaded_dataset.dataset_id} "
            f"processing_version={WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION} "
            f"use_reverse_edges={use_reverse_edges}"
        )
        cached_dataset = self.storage_service.load_if_available(
            dataset_id=loaded_dataset.dataset_id,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
            use_reverse_edges=use_reverse_edges,
            load_train_instances=self.load_train_instances,
            load_test_instances=self.load_test_instances,
            source_fingerprints=source_fingerprints,
            entity_mapping_sha256=entity_mapping_sha256,
            profile=self.profile,
        )
        if cached_dataset is not None:
            logger.info(
                f"Loaded prepared WebQSP graph dataset from cache: "
                f"train_size={cached_dataset.train_size} test_size={cached_dataset.test_size}"
            )
            return cached_dataset

        logger.info(f"Prepared WebQSP graph cache miss; processing loaded dataset")
        prepared_dataset = self.processor_service.process_loaded_dataset(
            loaded_dataset=loaded_dataset,
            processing_version=WEBQSP_LOCAL_GRAPH_PROCESSING_VERSION,
            use_reverse_edges=use_reverse_edges,
            source_fingerprints=source_fingerprints,
            entity_mapping_sha256=entity_mapping_sha256,
            profile=self.profile,
            cache_directory=self.storage_service.get_cache_directory(
                loaded_dataset.dataset_id,
                use_reverse_edges=use_reverse_edges,
            ),
        )
        self.storage_service.save(prepared_dataset, profile=self.profile)
        cached_train_size = prepared_dataset.train_size
        cached_test_size = prepared_dataset.test_size
        prepared_dataset = self._select_required_splits(prepared_dataset)
        logger.info(
            f"Saved prepared WebQSP graph dataset: "
            f"cached_train_size={cached_train_size} cached_test_size={cached_test_size} "
            f"loaded_train_size={prepared_dataset.train_size} "
            f"loaded_test_size={prepared_dataset.test_size}"
        )
        return prepared_dataset

    def _select_required_splits(
        self,
        dataset: PreparedWebQSPGraphDataset,
    ) -> PreparedWebQSPGraphDataset:
        """Discard instance splits that the selected pipeline mode will not consume."""
        return dataset.model_copy(
            update={
                "train_instances": (
                    dataset.train_instances if self.load_train_instances else []
                ),
                "test_instances": (
                    dataset.test_instances if self.load_test_instances else []
                ),
            }
        )

    @staticmethod
    def _source_fingerprints(dataset) -> dict[str, str]:
        """Return stable Hugging Face split fingerprints when available."""
        return {
            str(split_name): str(fingerprint)
            for split_name, split in dataset.items()
            if (fingerprint := getattr(split, "_fingerprint", None)) is not None
        }

    def _entity_mapping_sha256(self) -> str | None:
        """Hash the mapping input so stale processed caches are rejected."""
        mapping_service = getattr(
            self.processor_service,
            "entity_name_mapping_service",
            None,
        )
        mapping_path = getattr(mapping_service, "mapping_path", None)
        if mapping_path is None:
            return None
        path = Path(mapping_path)
        if not path.is_file():
            return None
        started_at = time.perf_counter()
        digest = hashlib.sha256()
        with path.open("rb") as mapping_file:
            for chunk in iter(lambda: mapping_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if self.profile:
            logger.info(
                "Fingerprinting WebQSP entity mapping finished: "
                f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.2f}"
            )
        return digest.hexdigest()


class BuildWebQSPLocalGraphsContext(StepContext[LoadedDataset]):
    """Context for WebQSP graph preparation with pipeline configuration."""

    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration controlling graph preparation.",
    )
