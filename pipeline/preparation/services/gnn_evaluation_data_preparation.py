"""Prepare compact cached embedding matrices for GNN evaluation."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from helpers.logging_config import get_logger
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationConfig,
    PreparedGnnEvaluationData,
    PreparedGnnEvaluationInstance,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.preparation.helpers.configuration_definitions import (
    GNN_ARCHITECTURES,
    RGCN_ARCHITECTURE_ID,
)
from pipeline.preparation.helpers.gnn_architecture import (
    build_architecture_runtime_strategy,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance
from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)
from pipeline.preparation.services.gnn_embedding_tensor_cache import (
    GnnEmbeddingTensorCacheService,
)
from pipeline.preparation.services.gnn_relation_vocabulary import (
    build_active_relation_groups,
    build_relation_aggregation_metadata,
    build_sorted_typed_edges,
)
from pipeline.services import AbstractService

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch import Tensor, dtype as TorchDtype


class GnnEvaluationDataPreparationService(AbstractService):
    """Compact selected test graphs and load their embeddings once."""

    bytes_per_gibibyte = 1024**3

    def __init__(
        self,
        embedding_cache_service: WebQSPEmbeddingCacheService,
        embedding_tensor_cache_service: GnnEmbeddingTensorCacheService | None = None,
    ) -> None:
        self.embedding_cache_service = embedding_cache_service
        self.embedding_tensor_cache_service = (
            embedding_tensor_cache_service
            or GnnEmbeddingTensorCacheService(embedding_cache_service)
        )

    def prepare(
        self,
        torch: ModuleType,
        test_instances: list[WebQSPProcessedInstance],
        cache_root: Path,
        dataset_id: str,
        entity_embedding_model: str,
        relation_embedding_model: str,
        question_embedding_model: str,
        selected_device: str,
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
        gnn_architecture: str = "graphsage",
        relation_vocabulary: dict[str, int] | None = None,
    ) -> PreparedGnnEvaluationData:
        """Prepare compact evaluation tensors and recover safely from CUDA OOM."""
        if evaluation_config.gpu_cache_reserve_gb < 0:
            raise GnnAnswerRetrieverEvaluationException(
                "evaluation_gpu_cache_reserve_gb must be greater than or equal to zero."
            )

        runtime_strategy = build_architecture_runtime_strategy(gnn_architecture)
        if runtime_strategy.handles_data_preparation:
            autocast_dtype = self._resolve_embedding_dtype(
                torch=torch,
                requested_dtype=evaluation_config.embedding_cache_dtype,
                selected_device=selected_device,
            )
            prepared_data = runtime_strategy.prepare_evaluation_data(
                instances=test_instances,
                relation_vocabulary=relation_vocabulary,
                selected_device=selected_device,
                torch=torch,
                autocast_dtype=autocast_dtype,
            )
            return self._filter_missing_gold_evaluation_instances(
                prepared_data=prepared_data,
                enabled=getattr(
                    evaluation_config,
                    "skip_missing_gold_in_graph",
                    True,
                ),
            )

        requirements = GNN_ARCHITECTURES[gnn_architecture].data_requirements
        node_texts: list[str] = []
        node_index_tensors: list[Tensor | None] = [None] * len(test_instances)
        if requirements.uses_entity_embeddings:
            node_texts, node_index_tensors = self._compact_text_index_tensors(
                instance_texts=[instance.nodes for instance in test_instances],
                torch=torch,
            )
        relation_texts: list[str] = []
        relation_index_tensors: list[Tensor | None] = [None] * len(test_instances)
        if requirements.uses_relation_embeddings:
            relation_texts, relation_index_tensors = self._compact_text_index_tensors(
                instance_texts=[instance.edge_relations for instance in test_instances],
                torch=torch,
            )
        question_texts: list[str] = []
        question_indices: list[list[int]] = [[] for _ in test_instances]
        if requirements.uses_question_embeddings:
            question_texts, question_indices = self._compact_text_indices(
                [[instance.question] for instance in test_instances]
            )
        node_cache = (
            self.embedding_cache_service.load_node_cache(
                cache_root=cache_root,
                model_id=entity_embedding_model,
                vocabulary={text: index for index, text in enumerate(node_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            )
            if requirements.uses_entity_embeddings
            else None
        )
        relation_cache = (
            self.embedding_cache_service.load_relation_cache(
                cache_root=cache_root,
                model_id=relation_embedding_model,
                vocabulary={text: index for index, text in enumerate(relation_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            )
            if requirements.uses_relation_embeddings
            else None
        )
        question_cache = (
            self.embedding_cache_service.load_question_cache(
                cache_root=cache_root,
                model_id=question_embedding_model,
                vocabulary={text: index for index, text in enumerate(question_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            )
            if requirements.uses_question_embeddings
            else None
        )
        embedding_dtype = self._resolve_embedding_dtype(
            torch=torch,
            requested_dtype=evaluation_config.embedding_cache_dtype,
            selected_device=selected_device,
        )
        cache_specs: list[tuple[str, TextEmbeddingCache, list[str]]] = []
        if node_cache is not None:
            cache_specs.append(("entity", node_cache, node_texts))
        if relation_cache is not None:
            cache_specs.append(("relation", relation_cache, relation_texts))
        if question_cache is not None:
            cache_specs.append(("question", question_cache, question_texts))
        total_embedding_bytes = sum(
            len(texts) * cache.vector_size * (2 if embedding_dtype == "bfloat16" else 4)
            for _, cache, texts in cache_specs
        )
        embedding_device = self._resolve_embedding_device(
            torch=torch,
            requested_device=evaluation_config.embedding_cache_device,
            selected_device=selected_device,
            required_bytes=total_embedding_bytes,
            reserve_gb=evaluation_config.gpu_cache_reserve_gb,
        )
        logger.info(
            f"Preparing compact GNN evaluation embeddings: "
            f"instances={len(test_instances)} nodes={len(node_texts)} "
            f"relations={len(relation_texts)} questions={len(question_texts)} "
            f"storage_gib={total_embedding_bytes / self.bytes_per_gibibyte:.2f} "
            f"device={embedding_device} dtype={embedding_dtype}"
        )

        torch_dtype = torch.bfloat16 if embedding_dtype == "bfloat16" else torch.float32
        cuda_allocation_failed = False
        try:
            matrices = self._load_embedding_matrices(
                torch=torch,
                cache_root=cache_root,
                caches=tuple(cache for _, cache, _ in cache_specs),
                text_groups=tuple(texts for _, _, texts in cache_specs),
                dtype=torch_dtype,
                dtype_name=embedding_dtype,
                device=embedding_device,
            )
        except torch.OutOfMemoryError as error:
            if not embedding_device.startswith("cuda"):
                raise GnnAnswerRetrieverEvaluationException(
                    "Host memory was exhausted while loading evaluation embeddings."
                ) from error
            if evaluation_config.embedding_cache_device == "gpu":
                raise GnnAnswerRetrieverEvaluationException(
                    "CUDA ran out of memory while loading the forced GPU evaluation "
                    "embedding cache."
                ) from error
            logger.warning(
                "CUDA allocation failed while preparing evaluation embeddings; "
                "retrying with CPU storage."
            )
            embedding_device = "cpu"
            matrices = self._load_embedding_matrices(
                torch=torch,
                cache_root=cache_root,
                caches=tuple(cache for _, cache, _ in cache_specs),
                text_groups=tuple(texts for _, _, texts in cache_specs),
                dtype=torch_dtype,
                dtype_name=embedding_dtype,
                device=embedding_device,
            )
            cuda_allocation_failed = True
        if cuda_allocation_failed:
            torch.cuda.empty_cache()

        matrix_by_kind = {
            kind: matrix for (kind, _, _), matrix in zip(cache_specs, matrices, strict=True)
        }
        node_embeddings = matrix_by_kind.get("entity")
        relation_embeddings = matrix_by_kind.get("relation")
        question_embeddings = matrix_by_kind.get("question")
        prepared_instances: list[PreparedGnnEvaluationInstance] = []
        for instance_index, instance in enumerate(test_instances):
            seed_node_indices = None
            skip_reason = "empty_graph" if not instance.nodes else None
            if requirements.uses_seed_distributions and skip_reason is None:
                seed_ids = sorted(
                    {
                        instance.node2id[entity]
                        for entity in instance.q_entity
                        if entity in instance.node2id
                    }
                )
                if not seed_ids:
                    skip_reason = "missing_question_entity"
                else:
                    seed_node_indices = torch.tensor(seed_ids, dtype=torch.long)
            if skip_reason is not None:
                logger.warning(
                    f"{GNN_ARCHITECTURES[gnn_architecture].display_name} evaluation "
                    "graph cannot be scored: "
                    f"instance_index={instance_index} reason={skip_reason}"
                )
            edge_index = instance.edge_index
            edge_type = None
            edge_norm = None
            active_relation_ids = None
            edge_relation_index = None
            active_relation_offsets = None
            if requirements.uses_relation_types and skip_reason is None:
                if relation_vocabulary is None:
                    raise GnnAnswerRetrieverEvaluationException(
                        "Categorical-relation evaluation requires the saved relation "
                        "vocabulary."
                    )
                try:
                    edge_index, edge_type = build_sorted_typed_edges(
                        edge_index=edge_index,
                        edge_relations=instance.edge_relations,
                        vocabulary=relation_vocabulary,
                        torch=torch,
                    )
                    active_relation_ids, active_relation_offsets = (
                        build_active_relation_groups(
                            edge_type=edge_type,
                            torch=torch,
                        )
                    )
                    if gnn_architecture == RGCN_ARCHITECTURE_ID:
                        (
                            edge_norm,
                            active_relation_ids,
                            edge_relation_index,
                        ) = build_relation_aggregation_metadata(
                            edge_index=edge_index,
                            edge_type=edge_type,
                            node_count=len(instance.nodes),
                            torch=torch,
                        )
                except ValueError as error:
                    raise GnnAnswerRetrieverEvaluationException(
                        f"Could not prepare categorical edge types for evaluation instance "
                        f"{instance_index}: {error}"
                    ) from error
            prepared_instances.append(
                PreparedGnnEvaluationInstance(
                    source_instance_index=instance_index,
                    instance=instance,
                    node_embedding_indices=node_index_tensors[instance_index],
                    relation_embedding_indices=relation_index_tensors[instance_index],
                    question_embedding_index=(
                        question_indices[instance_index][0]
                        if question_indices[instance_index]
                        else None
                    ),
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_norm=edge_norm,
                    active_relation_ids=active_relation_ids,
                    edge_relation_index=edge_relation_index,
                    active_relation_offsets=active_relation_offsets,
                    seed_node_indices=seed_node_indices,
                    skip_reason=skip_reason,
                )
            )
        prepared_data = PreparedGnnEvaluationData(
            instances=prepared_instances,
            node_embeddings=node_embeddings,
            relation_embeddings=relation_embeddings,
            question_embeddings=question_embeddings,
            selected_device=selected_device,
            embedding_cache_device=embedding_device,
            embedding_cache_dtype=embedding_dtype,
        )
        return self._filter_missing_gold_evaluation_instances(
            prepared_data=prepared_data,
            enabled=getattr(
                evaluation_config,
                "skip_missing_gold_in_graph",
                True,
            ),
        )

    @staticmethod
    def _filter_missing_gold_evaluation_instances(
        *,
        prepared_data: PreparedGnnEvaluationData,
        enabled: bool,
    ) -> PreparedGnnEvaluationData:
        if not enabled:
            return prepared_data
        kept_instances = [
            prepared_instance
            for prepared_instance in prepared_data.instances
            if bool(set(prepared_instance.instance.a_entity))
            and set(prepared_instance.instance.a_entity).issubset(
                prepared_instance.instance.node2id
            )
        ]
        skipped_count = len(prepared_data.instances) - len(kept_instances)
        if skipped_count:
            logger.warning(
                f"Skipped {skipped_count} evaluation graphs because at least one "
                "gold answer entity is absent from the graph."
            )
        return prepared_data.model_copy(
            update={
                "instances": kept_instances,
                "skipped_missing_gold_in_graph_count": skipped_count,
            }
        )

    def _build_cache_handles(
        self,
        cache_root: Path,
        dataset_id: str,
        entity_embedding_model: str,
        relation_embedding_model: str,
        question_embedding_model: str,
        node_texts: list[str],
        relation_texts: list[str],
        question_texts: list[str],
    ) -> tuple[TextEmbeddingCache, TextEmbeddingCache, TextEmbeddingCache]:
        """Build Qdrant identities without contacting Qdrant on local hits."""
        return (
            self.embedding_cache_service.load_node_cache(
                cache_root=cache_root,
                model_id=entity_embedding_model,
                vocabulary={text: index for index, text in enumerate(node_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            ),
            self.embedding_cache_service.load_relation_cache(
                cache_root=cache_root,
                model_id=relation_embedding_model,
                vocabulary={text: index for index, text in enumerate(relation_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            ),
            self.embedding_cache_service.load_question_cache(
                cache_root=cache_root,
                model_id=question_embedding_model,
                vocabulary={text: index for index, text in enumerate(question_texts)},
                dataset_id=dataset_id,
                ensure_collection=False,
            ),
        )

    def _load_embedding_matrices(
        self,
        torch: ModuleType,
        cache_root: Path,
        caches: tuple[TextEmbeddingCache, ...],
        text_groups: tuple[list[str], ...],
        dtype: TorchDtype,
        dtype_name: str,
        device: str,
    ) -> tuple[Tensor, ...]:
        """Load all compact matrices through the shared incremental cache."""
        return tuple(
            self.embedding_tensor_cache_service.load_matrix(
                torch=torch,
                cache_root=cache_root,
                cache=cache,
                texts=texts,
                dtype=dtype,
                dtype_name=dtype_name,
                device=device,
                preprocess=cache.cache_kind == "relations",
            )
            for cache, texts in zip(caches, text_groups, strict=True)
        )

    @staticmethod
    def _compact_text_indices(
        instance_texts: list[list[str]],
    ) -> tuple[list[str], list[list[int]]]:
        """Create one compact vocabulary and per-instance integer indices."""
        text_to_index: dict[str, int] = {}
        compact_texts: list[str] = []
        instance_indices: list[list[int]] = []
        for texts in instance_texts:
            indices: list[int] = []
            for text in texts:
                compact_index = text_to_index.get(text)
                if compact_index is None:
                    compact_index = len(compact_texts)
                    text_to_index[text] = compact_index
                    compact_texts.append(text)
                indices.append(compact_index)
            instance_indices.append(indices)
        return compact_texts, instance_indices

    @classmethod
    def _compact_text_index_tensors(
        cls,
        instance_texts: list[list[str]],
        torch: ModuleType,
    ) -> tuple[list[str], list[Tensor]]:
        """Create compact IDs as CPU tensors for later indexed gathers."""
        compact_texts, instance_indices = cls._compact_text_indices(instance_texts)
        return compact_texts, [
            torch.tensor(indices, dtype=torch.long) for indices in instance_indices
        ]

    @staticmethod
    def _resolve_embedding_dtype(
        torch: ModuleType,
        requested_dtype: str,
        selected_device: str,
    ) -> str:
        """Resolve evaluation embedding precision for the selected accelerator."""
        if not selected_device.startswith("cuda"):
            if requested_dtype == "bfloat16":
                raise GnnAnswerRetrieverEvaluationException(
                    "bfloat16 evaluation embeddings require a CUDA device."
                )
            return "float32"
        if requested_dtype == "float32":
            return "float32"
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        if requested_dtype == "bfloat16":
            raise GnnAnswerRetrieverEvaluationException(
                "The selected CUDA device does not support bfloat16 embeddings."
            )
        return "float32"

    def _resolve_embedding_device(
        self,
        torch: ModuleType,
        requested_device: str,
        selected_device: str,
        required_bytes: int,
        reserve_gb: float,
    ) -> str:
        """Place compact embeddings on CUDA when they fit the safe budget."""
        if requested_device == "cpu":
            return "cpu"
        if not selected_device.startswith("cuda"):
            if requested_device == "gpu":
                raise GnnAnswerRetrieverEvaluationException(
                    "GPU evaluation embedding cache requested without CUDA."
                )
            return "cpu"
        free_bytes, _ = torch.cuda.mem_get_info()
        reserve_bytes = int(reserve_gb * self.bytes_per_gibibyte)
        if required_bytes <= max(0, free_bytes - reserve_bytes):
            return selected_device
        if requested_device == "gpu":
            raise GnnAnswerRetrieverEvaluationException(
                f"GPU evaluation embedding cache requires "
                f"{required_bytes / self.bytes_per_gibibyte:.2f} GiB with a "
                f"{reserve_gb:.2f} GiB reserve, but only "
                f"{free_bytes / self.bytes_per_gibibyte:.2f} GiB is free."
            )
        logger.warning(
            f"Compact evaluation embeddings do not fit the safe CUDA budget; "
            f"using CPU storage: required_gib="
            f"{required_bytes / self.bytes_per_gibibyte:.2f} "
            f"free_gib={free_bytes / self.bytes_per_gibibyte:.2f} "
            f"reserve_gib={reserve_gb:.2f}"
        )
        return "cpu"

    @staticmethod
    def _embedding_storage_bytes(
        text_counts: tuple[int, int, int],
        vector_sizes: tuple[int, int, int],
        element_size: int,
    ) -> int:
        """Calculate bytes required by all compact embedding matrices."""
        return sum(
            text_count * vector_size * element_size
            for text_count, vector_size in zip(text_counts, vector_sizes, strict=True)
        )
