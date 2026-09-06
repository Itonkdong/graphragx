"""Evaluation service for saved GNN answer-retriever runs."""

from __future__ import annotations

import time
from types import ModuleType
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from helpers.logging_config import get_logger
from helpers.constants import MIN_CANDIDATE_TOP_K
from pipeline.evaluation.models import (
    AnswerCandidateScore,
    EvaluatedAnswerRetrievalInstance,
    GnnAnswerRetrieverEvaluationConfig,
    GoldAnswerScore,
    PreparedGnnEvaluationData,
)
from pipeline.evaluation.services.gnn_retriever_results import (
    GnnRetrieverResultsService,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.preparation.models.webqsp_local_graph import (
    PreparedWebQSPGraphDataset,
    WebQSPProcessedInstance,
)
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.helpers.gnn_architecture import (
    build_architecture_runtime_strategy,
)
from pipeline.services import AbstractService
from pipeline.preparation.services.embedding_cache import WebQSPEmbeddingCacheService
from pipeline.preparation.services.gnn_evaluation_data_preparation import (
    GnnEvaluationDataPreparationService,
)
from pipeline.preparation.services.gnn_answer_retriever_evaluation_storage import (
    GnnAnswerRetrieverEvaluationStoragePayload,
    GnnAnswerRetrieverEvaluationStorageResult,
    GnnAnswerRetrieverEvaluationStorageService,
)
from pipeline.preparation.services.gnn_answer_retriever_model_runs import (
    GnnAnswerRetrieverModelRunService,
    LoadedGnnAnswerRetrieverRun,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch import Tensor


class GnnEvaluationPhaseTimings(BaseModel):
    """Accumulated synchronized timings for GNN evaluation phases."""

    model_load_seconds: float = 0.0
    embedding_preparation_seconds: float = 0.0
    input_preparation_seconds: float = 0.0
    forward_seconds: float = 0.0
    prediction_seconds: float = 0.0
    storage_seconds: float = 0.0
    instance_count: int = 0


class GnnAnswerRetrieverEvaluationOutcome(BaseModel):
    """Evaluation output and persisted run metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loaded_model_run: LoadedGnnAnswerRetrieverRun = Field(
        ...,
        description="Loaded model run used for evaluation.",
    )
    storage_result: GnnAnswerRetrieverEvaluationStorageResult = Field(
        ...,
        description="Saved evaluation run paths.",
    )
    evaluated_instances: int = Field(..., description="Number of evaluated instances.")
    hits_at_1: float = Field(..., description="Hits@1 rate.")
    hits_at_1_count: int = Field(..., description="Hits@1 count.")
    hits_at_5: float = Field(..., description="Hits@5 rate.")
    hits_at_5_count: int = Field(..., description="Hits@5 count.")
    hits_at_10: float = Field(..., description="Hits@10 rate.")
    hits_at_10_count: int = Field(..., description="Hits@10 count.")
    hits_at_candidate_limit: float = Field(
        ...,
        description="Hits at configured candidate limit rate.",
    )
    hits_at_candidate_limit_count: int = Field(
        ...,
        description="Hits at configured candidate limit count.",
    )
    ndcg_at_1: float = Field(default=0.0, description="Mean retrieval nDCG@1.")
    ndcg_at_5: float = Field(default=0.0, description="Mean retrieval nDCG@5.")
    ndcg_at_10: float = Field(default=0.0, description="Mean retrieval nDCG@10.")
    ndcg_at_candidate_limit: float = Field(
        default=0.0,
        description="Mean retrieval nDCG at the configured candidate limit.",
    )
    conditioned_evaluated_instances: int = Field(default=0)
    retrieval_gold_coverage: float = Field(default=0.0)
    retrieval_full_gold_coverage_count: int = Field(default=0)
    retrieval_full_gold_coverage_rate: float = Field(default=0.0)
    retrieved_gold_answer_count: int = Field(default=0)
    average_candidate_count: float = Field(
        ...,
        description="Average number of selected candidate nodes.",
    )
    missing_gold_in_graph_count: int = Field(
        ...,
        description="Instances where no gold answer appears in the local graph.",
    )
    skipped_missing_gold_in_graph_count: int = Field(default=0)


class GnnAnswerRetrieverEvaluationService(AbstractService):
    """Evaluate a saved GNN answer-retriever model over prepared WebQSP test graphs."""

    def __init__(
        self,
        model_run_service: GnnAnswerRetrieverModelRunService | None = None,
        embedding_cache_service: WebQSPEmbeddingCacheService | None = None,
        data_preparation_service: GnnEvaluationDataPreparationService | None = None,
        storage_service: GnnAnswerRetrieverEvaluationStorageService | None = None,
        results_service: GnnRetrieverResultsService | None = None,
    ):
        self.model_run_service = model_run_service or GnnAnswerRetrieverModelRunService()
        self.embedding_cache_service = (
            embedding_cache_service or WebQSPEmbeddingCacheService()
        )
        self.data_preparation_service = (
            data_preparation_service
            or GnnEvaluationDataPreparationService(self.embedding_cache_service)
        )
        self.storage_service = (
            storage_service or GnnAnswerRetrieverEvaluationStorageService()
        )
        self.results_service = results_service or GnnRetrieverResultsService()

    def evaluate(
        self,
        prepared_dataset: PreparedWebQSPGraphDataset,
        pipeline_configuration: BuiltPipelineConfiguration,
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
    ) -> GnnAnswerRetrieverEvaluationOutcome:
        """Run saved-model retrieval evaluation and persist all outputs."""
        import torch
        import torch.nn.functional as torch_functional

        test_instances = self._select_test_instances(
            prepared_dataset=prepared_dataset,
            max_instances=evaluation_config.max_instances,
        )
        self._validate_candidate_selection_config(evaluation_config)
        if not test_instances:
            raise GnnAnswerRetrieverEvaluationException(
                "GNN answer-retriever evaluation requires at least one test instance."
            )

        device = self._resolve_device(torch)
        cache_root = prepared_dataset.cache_directory.parent
        phase_timings = GnnEvaluationPhaseTimings()
        phase_started_at = self._start_profiled_phase(
            torch=torch,
            device=device,
            enabled=evaluation_config.profile,
        )
        loaded_model_run = self.model_run_service.load_run(
            model_root=cache_root / "models",
            run_name=evaluation_config.model_run_name,
            run_number=evaluation_config.model_run_number,
            pipeline_configuration=pipeline_configuration,
            device=device,
        )
        phase_started_at, elapsed_seconds = self._finish_profiled_phase(
            torch=torch,
            device=device,
            enabled=evaluation_config.profile,
            phase_started_at=phase_started_at,
        )
        phase_timings.model_load_seconds = elapsed_seconds
        runtime_strategy = build_architecture_runtime_strategy(
            loaded_model_run.config.resolved_gnn_architecture
        )
        if loaded_model_run.config.dataset_id != prepared_dataset.dataset_id:
            raise GnnAnswerRetrieverEvaluationException(
                f"Model run dataset {loaded_model_run.config.dataset_id} does not "
                f"match prepared dataset {prepared_dataset.dataset_id}."
            )

        logger.info(
            f"Starting GNN answer-retriever evaluation: "
            f"model_run={loaded_model_run.run_name} "
            f"instances={len(test_instances)} device={device} "
            f"threshold={evaluation_config.answer_threshold} "
            f"candidate_top_k={evaluation_config.candidate_top_k} "
            f"candidate_limit={evaluation_config.candidate_limit} "
            f"cache_device={evaluation_config.embedding_cache_device} "
            f"cache_dtype={evaluation_config.embedding_cache_dtype}"
        )
        prepared_evaluation_data = self.data_preparation_service.prepare(
            torch=torch,
            test_instances=test_instances,
            cache_root=cache_root,
            dataset_id=prepared_dataset.dataset_id,
            entity_embedding_model=loaded_model_run.config.resolved_embedding_model,
            relation_embedding_model=loaded_model_run.relation_embedding_model,
            question_embedding_model=loaded_model_run.question_embedding_model,
            selected_device=device,
            evaluation_config=evaluation_config,
            gnn_architecture=loaded_model_run.config.resolved_gnn_architecture,
            relation_vocabulary=loaded_model_run.relation_vocabulary,
        )
        phase_started_at, elapsed_seconds = self._finish_profiled_phase(
            torch=torch,
            device=device,
            enabled=evaluation_config.profile,
            phase_started_at=phase_started_at,
        )
        phase_timings.embedding_preparation_seconds = elapsed_seconds
        logger.info(
            f"Beginning GNN answer-retriever instance evaluation: "
            f"model_run={loaded_model_run.run_name} "
            f"instances={len(prepared_evaluation_data.instances)} "
            f"embedding_device={prepared_evaluation_data.embedding_cache_device} "
            f"embedding_dtype={prepared_evaluation_data.embedding_cache_dtype} "
            f"log_every={evaluation_config.log_every}"
        )
        prepared_instance_count = len(prepared_evaluation_data.instances)
        if prepared_instance_count == 0:
            raise GnnAnswerRetrieverEvaluationException(
                "GNN evaluation has no usable graphs after skipping instances "
                "without their complete in-graph gold-answer set. Use "
                "--no-skip-missing-gold-in-graph to restore the previous behavior."
            )

        predictions: list[EvaluatedAnswerRetrievalInstance] = []
        hits_at_1_count = 0
        hits_at_5_count = 0
        hits_at_10_count = 0
        hits_at_candidate_limit_count = 0
        missing_gold_in_graph_count = (
            prepared_evaluation_data.skipped_missing_gold_in_graph_count
        )
        total_candidate_count = 0

        # Keep the evaluation boundary robust for injected/custom model-run
        # services as well as the standard loader. Inputs below are placed on
        # the selected device, so the model must follow them explicitly.
        model = loaded_model_run.model.to(device)
        model.eval()
        with torch.inference_mode():
            for prepared_instance in prepared_evaluation_data.instances:
                instance = prepared_instance.instance
                if prepared_instance.skip_reason is not None:
                    prediction = self._build_skipped_prediction(
                        instance_index=prepared_instance.source_instance_index,
                        instance=instance,
                        global_node_vocabulary=prepared_dataset.vocabulary_store.nodes,
                    )
                    predictions.append(prediction)
                    missing_gold_in_graph_count += int(prediction.missing_gold_in_graph)
                    phase_timings.instance_count += 1
                    logger.warning(
                        "Recorded GNN retrieval miss without model execution: "
                        f"instance_index={prepared_instance.source_instance_index} "
                        f"reason={prepared_instance.skip_reason}"
                    )
                    continue
                phase_started_at = self._start_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=evaluation_config.profile,
                )
                runtime_batch = None
                if runtime_strategy.handles_training_batches:
                    runtime_batch = runtime_strategy.build_evaluation_batch(
                        prepared_data=prepared_evaluation_data,
                        prepared_instance=prepared_instance,
                        torch=torch,
                        device=device,
                    )
                    entity_features = None
                else:
                    if (
                        prepared_evaluation_data.node_embeddings is None
                        or prepared_instance.node_embedding_indices is None
                    ):
                        raise GnnAnswerRetrieverEvaluationException(
                            "Prepared node embeddings are missing for this architecture."
                        )
                    entity_features = self._gather_embedding_features(
                        embedding_matrix=prepared_evaluation_data.node_embeddings,
                        indices=prepared_instance.node_embedding_indices,
                        torch=torch,
                        device=device,
                    )
                relation_features = None
                if prepared_evaluation_data.relation_embeddings is not None:
                    if prepared_instance.relation_embedding_indices is None:
                        raise GnnAnswerRetrieverEvaluationException(
                            "Prepared relation embeddings are missing instance indices."
                        )
                    relation_features = self._gather_embedding_features(
                        embedding_matrix=prepared_evaluation_data.relation_embeddings,
                        indices=prepared_instance.relation_embedding_indices,
                        torch=torch,
                        device=device,
                    )
                question_features = None
                if prepared_evaluation_data.question_embeddings is not None:
                    if prepared_instance.question_embedding_index is None:
                        raise GnnAnswerRetrieverEvaluationException(
                            "Prepared question embeddings are missing an instance index."
                        )
                    question_features = prepared_evaluation_data.question_embeddings[
                        prepared_instance.question_embedding_index
                    ].to(device=device, non_blocking=True)
                edge_weight = None
                if relation_features is not None and question_features is not None:
                    edge_weight = (
                        None
                        if loaded_model_run.config.use_edge_mlp
                        else self._build_edge_weight_tensor(
                            relation_features=relation_features,
                            question_features=question_features,
                            torch=torch,
                            torch_functional=torch_functional,
                            device=device,
                        )
                    )
                edge_index = prepared_instance.edge_index.to(
                    device=device, non_blocking=True
                )
                edge_type = (
                    prepared_instance.edge_type.to(device=device, non_blocking=True)
                    if prepared_instance.edge_type is not None
                    else None
                )
                edge_norm = (
                    prepared_instance.edge_norm.to(device=device, non_blocking=True)
                    if prepared_instance.edge_norm is not None
                    else None
                )
                active_relation_ids = (
                    prepared_instance.active_relation_ids.to(
                        device=device, non_blocking=True
                    )
                    if prepared_instance.active_relation_ids is not None
                    else None
                )
                edge_relation_index = (
                    prepared_instance.edge_relation_index.to(
                        device=device, non_blocking=True
                    )
                    if prepared_instance.edge_relation_index is not None
                    else None
                )
                active_relation_offsets = (
                    prepared_instance.active_relation_offsets
                    if prepared_instance.active_relation_offsets is not None
                    else None
                )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=evaluation_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.input_preparation_seconds += elapsed_seconds

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(
                        device.startswith("cuda")
                        and prepared_evaluation_data.uses_bfloat16
                    ),
                ):
                    logits = (
                        model(**runtime_strategy.model_inputs(runtime_batch))
                        if runtime_batch is not None
                        else model(
                            entity_features=entity_features,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            question_features=question_features,
                            relation_features=relation_features,
                            edge_type=edge_type,
                            edge_norm=edge_norm,
                            active_relation_ids=active_relation_ids,
                            edge_relation_index=edge_relation_index,
                            active_relation_offsets=active_relation_offsets,
                        )
                    )
                probabilities = runtime_strategy.probabilities(
                    logits,
                    graph_index=(
                        runtime_batch.node_graph_index if runtime_batch is not None else None
                    ),
                    graph_count=(runtime_batch.graph_count if runtime_batch is not None else None),
                )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=evaluation_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.forward_seconds += elapsed_seconds

                scores = torch.stack((logits, probabilities), dim=1).float().cpu()
                prediction = self._build_prediction(
                    instance_index=prepared_instance.source_instance_index,
                    instance=instance,
                    logits=scores[:, 0],
                    probabilities=scores[:, 1],
                    answer_threshold=evaluation_config.answer_threshold,
                    candidate_top_k=evaluation_config.candidate_top_k,
                    candidate_limit=evaluation_config.candidate_limit,
                    global_node_vocabulary=prepared_dataset.vocabulary_store.nodes,
                    torch=torch,
                )
                phase_started_at, elapsed_seconds = self._finish_profiled_phase(
                    torch=torch,
                    device=device,
                    enabled=evaluation_config.profile,
                    phase_started_at=phase_started_at,
                )
                phase_timings.prediction_seconds += elapsed_seconds
                phase_timings.instance_count += 1
                predictions.append(prediction)
                hits_at_1_count += int(prediction.hit_at_1)
                hits_at_5_count += int(prediction.hit_at_5)
                hits_at_10_count += int(prediction.hit_at_10)
                hits_at_candidate_limit_count += int(
                    prediction.hit_at_candidate_limit
                )
                missing_gold_in_graph_count += int(prediction.missing_gold_in_graph)
                total_candidate_count += len(prediction.answer_candidates)
                processed_count = phase_timings.instance_count
                if (
                    evaluation_config.log_every > 0
                    and (
                        processed_count % evaluation_config.log_every == 0
                        or processed_count == prepared_instance_count
                    )
                ):
                    logger.info(
                        f"GNN answer-retriever evaluation progress: "
                            f"{processed_count}/{prepared_instance_count} instances "
                        f"hits_at_1={hits_at_1_count / processed_count:.4f} "
                        f"hits_at_5={hits_at_5_count / processed_count:.4f} "
                        f"hits_at_10={hits_at_10_count / processed_count:.4f} "
                        f"hits_at_candidate_limit="
                        f"{hits_at_candidate_limit_count / processed_count:.4f} "
                        f"average_candidate_count="
                        f"{total_candidate_count / processed_count:.2f}"
                    )
                    if evaluation_config.profile:
                        self._log_phase_timings(
                            processed_instances=processed_count,
                            timings=phase_timings,
                        )

        evaluated_instances = len(predictions)
        hits_at_1 = hits_at_1_count / evaluated_instances
        hits_at_5 = hits_at_5_count / evaluated_instances
        hits_at_10 = hits_at_10_count / evaluated_instances
        hits_at_candidate_limit = hits_at_candidate_limit_count / evaluated_instances
        average_candidate_count = total_candidate_count / evaluated_instances
        phase_started_at = self._start_profiled_phase(
            torch=torch,
            device=device,
            enabled=evaluation_config.profile,
        )
        retrieval_metrics = self.results_service.build_metrics(
            dataset_id=prepared_dataset.dataset_id,
            model_run_name=loaded_model_run.run_name,
            model_run_number=loaded_model_run.run_number,
            predictions=predictions,
            candidate_limit=evaluation_config.candidate_limit,
            missing_gold_in_graph_count=missing_gold_in_graph_count,
            skipped_missing_gold_in_graph_count=(
                prepared_evaluation_data.skipped_missing_gold_in_graph_count
            ),
        )
        storage_result = self.storage_service.save_evaluation_run(
            evaluation_root=cache_root / "evaluations",
            run_name=evaluation_config.run_name,
            payload=GnnAnswerRetrieverEvaluationStoragePayload(
                evaluation_config=self._build_evaluation_config_payload(
                    evaluation_config=evaluation_config,
                    loaded_model_run=loaded_model_run,
                    pipeline_configuration=pipeline_configuration,
                    device=device,
                    prepared_evaluation_data=prepared_evaluation_data,
                ),
                predictions=predictions,
                metrics=retrieval_metrics,
            ),
        )
        _, elapsed_seconds = self._finish_profiled_phase(
            torch=torch,
            device=device,
            enabled=evaluation_config.profile,
            phase_started_at=phase_started_at,
        )
        phase_timings.storage_seconds = elapsed_seconds
        if evaluation_config.profile:
            self._log_total_timings(phase_timings)
        logger.info(
            f"Finished GNN answer-retriever evaluation: "
            f"run={storage_result.evaluation_run_name} "
            f"hits_at_1={hits_at_1:.4f} hits_at_5={hits_at_5:.4f} "
            f"hits_at_10={hits_at_10:.4f} "
            f"hits_at_candidate_limit={hits_at_candidate_limit:.4f} "
            f"retrieval_gold_coverage="
            f"{retrieval_metrics.retrieval_gold_coverage:.4f} "
            f"retrieval_full_gold_coverage="
            f"{retrieval_metrics.retrieval_full_gold_coverage_rate:.4f} "
            f"average_candidate_count={average_candidate_count:.2f} "
            f"skipped_missing_gold="
            f"{prepared_evaluation_data.skipped_missing_gold_in_graph_count}"
        )
        return GnnAnswerRetrieverEvaluationOutcome(
            loaded_model_run=loaded_model_run,
            storage_result=storage_result,
            evaluated_instances=evaluated_instances,
            hits_at_1=hits_at_1,
            hits_at_1_count=hits_at_1_count,
            hits_at_5=hits_at_5,
            hits_at_5_count=hits_at_5_count,
            hits_at_10=hits_at_10,
            hits_at_10_count=hits_at_10_count,
            hits_at_candidate_limit=hits_at_candidate_limit,
            hits_at_candidate_limit_count=hits_at_candidate_limit_count,
            ndcg_at_1=retrieval_metrics.ndcg_at_1,
            ndcg_at_5=retrieval_metrics.ndcg_at_5,
            ndcg_at_10=retrieval_metrics.ndcg_at_10,
            ndcg_at_candidate_limit=retrieval_metrics.ndcg_at_candidate_limit,
            conditioned_evaluated_instances=(
                retrieval_metrics.conditioned_evaluated_instances
            ),
            retrieval_gold_coverage=retrieval_metrics.retrieval_gold_coverage,
            retrieval_full_gold_coverage_count=(
                retrieval_metrics.retrieval_full_gold_coverage_count
            ),
            retrieval_full_gold_coverage_rate=(
                retrieval_metrics.retrieval_full_gold_coverage_rate
            ),
            retrieved_gold_answer_count=(
                retrieval_metrics.retrieved_gold_answer_count
            ),
            average_candidate_count=average_candidate_count,
            missing_gold_in_graph_count=missing_gold_in_graph_count,
            skipped_missing_gold_in_graph_count=(
                prepared_evaluation_data.skipped_missing_gold_in_graph_count
            ),
        )

    @staticmethod
    def _select_test_instances(
        prepared_dataset: PreparedWebQSPGraphDataset,
        max_instances: int | None,
    ) -> list[WebQSPProcessedInstance]:
        if max_instances is None:
            return prepared_dataset.test_instances

        return prepared_dataset.test_instances[:max_instances]

    @staticmethod
    def _validate_candidate_selection_config(
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
    ) -> None:
        if evaluation_config.candidate_top_k < MIN_CANDIDATE_TOP_K:
            raise GnnAnswerRetrieverEvaluationException(
                f"candidate_top_k must be at least {MIN_CANDIDATE_TOP_K}."
            )

        if evaluation_config.candidate_limit <= 0:
            raise GnnAnswerRetrieverEvaluationException(
                "candidate_limit must be greater than zero."
            )

        if evaluation_config.candidate_limit < evaluation_config.candidate_top_k:
            raise GnnAnswerRetrieverEvaluationException(
                "candidate_limit must be greater than or equal to candidate_top_k."
            )

    @staticmethod
    def _resolve_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _gather_embedding_features(
        embedding_matrix: Tensor,
        indices: Tensor,
        torch: ModuleType,
        device: str,
    ) -> Tensor:
        """Gather frozen embeddings without contacting Qdrant."""
        matrix_device = embedding_matrix.device
        resolved_indices = indices.to(device=matrix_device, non_blocking=True)
        features = torch.index_select(embedding_matrix, 0, resolved_indices)
        return features.to(device=device, non_blocking=True)

    def _build_edge_weight_tensor(
        self,
        relation_features,
        question_features,
        torch,
        torch_functional,
        device: str,
    ):
        if relation_features.shape[0] == 0:
            return torch.empty(0, dtype=relation_features.dtype, device=device)
        return torch_functional.cosine_similarity(
            question_features.reshape(1, -1),
            relation_features,
            dim=1,
        )

    @staticmethod
    def _start_profiled_phase(
        torch: ModuleType,
        device: str,
        enabled: bool,
    ) -> float:
        """Synchronize the selected device and begin one measured phase."""
        if not enabled:
            return 0.0
        GnnAnswerRetrieverEvaluationService._synchronize_device(torch, device)
        return time.perf_counter()

    @staticmethod
    def _finish_profiled_phase(
        torch: ModuleType,
        device: str,
        enabled: bool,
        phase_started_at: float,
    ) -> tuple[float, float]:
        """Finish one synchronized phase and return the next start time."""
        if not enabled:
            return 0.0, 0.0
        GnnAnswerRetrieverEvaluationService._synchronize_device(torch, device)
        elapsed_seconds = time.perf_counter() - phase_started_at
        return time.perf_counter(), elapsed_seconds

    @staticmethod
    def _synchronize_device(torch: ModuleType, device: str) -> None:
        """Wait for asynchronous accelerator work before recording timings."""
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    @staticmethod
    def _log_phase_timings(
        processed_instances: int,
        timings: GnnEvaluationPhaseTimings,
    ) -> None:
        """Log per-instance evaluation timings accumulated so far."""
        if timings.instance_count == 0:
            return
        milliseconds_per_instance = 1000.0 / timings.instance_count
        logger.info(
            f"Evaluation profile instances={processed_instances}: "
            f"input_ms="
            f"{timings.input_preparation_seconds * milliseconds_per_instance:.2f} "
            f"forward_ms={timings.forward_seconds * milliseconds_per_instance:.2f} "
            f"prediction_ms={timings.prediction_seconds * milliseconds_per_instance:.2f}"
        )

    @staticmethod
    def _log_total_timings(timings: GnnEvaluationPhaseTimings) -> None:
        """Log startup, per-instance, and persistence evaluation timings."""
        logger.info(
            f"Evaluation profile totals: "
            f"model_load_ms={timings.model_load_seconds * 1000.0:.2f} "
            f"embedding_preparation_ms="
            f"{timings.embedding_preparation_seconds * 1000.0:.2f} "
            f"storage_ms={timings.storage_seconds * 1000.0:.2f}"
        )

    def _build_prediction(
        self,
        instance_index: int,
        instance: WebQSPProcessedInstance,
        logits,
        probabilities,
        answer_threshold: float,
        candidate_top_k: int,
        candidate_limit: int,
        global_node_vocabulary: dict[str, int],
        torch,
    ) -> EvaluatedAnswerRetrievalInstance:
        gold_answers = set(instance.a_entity)
        selected_ids = [
            node_id
            for node_id, probability in enumerate(probabilities.tolist())
            if probability >= answer_threshold
        ]
        selection_reason = "threshold"
        if len(selected_ids) < candidate_top_k:
            top_k_ids = torch.topk(
                probabilities,
                k=min(candidate_top_k, len(instance.nodes)),
            ).indices.tolist()
            selected_ids = list(dict.fromkeys([*selected_ids, *top_k_ids]))
            selection_reason = "fallback_top_k"

        selected_ids = sorted(
            selected_ids,
            key=lambda node_id: float(probabilities[node_id].item()),
            reverse=True,
        )[:candidate_limit]
        answer_candidates = [
            AnswerCandidateScore(
                node=instance.nodes[node_id],
                local_node_id=node_id,
                global_node_id=global_node_vocabulary[instance.nodes[node_id]],
                logit=float(logits[node_id].item()),
                probability=float(probabilities[node_id].item()),
                is_gold_answer=instance.nodes[node_id] in gold_answers,
                selection_reason=selection_reason,
            )
            for node_id in selected_ids
        ]
        gold_answer_scores = [
            self._build_gold_answer_score(
                gold_answer=gold_answer,
                instance=instance,
                logits=logits,
                probabilities=probabilities,
                global_node_vocabulary=global_node_vocabulary,
            )
            for gold_answer in instance.a_entity
        ]

        top_node_id = int(torch.argmax(probabilities).item())
        hit_at_1 = instance.nodes[top_node_id] in gold_answers
        hit_at_5 = any(candidate.is_gold_answer for candidate in answer_candidates[:5])
        hit_at_10 = any(candidate.is_gold_answer for candidate in answer_candidates[:10])
        hit_at_candidate_limit = any(
            candidate.is_gold_answer for candidate in answer_candidates
        )
        missing_gold_in_graph = any(
            not score.present_in_graph for score in gold_answer_scores
        )

        return EvaluatedAnswerRetrievalInstance(
            instance_index=instance_index,
            question=instance.question,
            q_entity=instance.q_entity,
            a_entity=instance.a_entity,
            answer_candidates=answer_candidates,
            gold_answer_scores=gold_answer_scores,
            hit_at_1=hit_at_1,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            hit_at_candidate_limit=hit_at_candidate_limit,
            missing_gold_in_graph=missing_gold_in_graph,
        )

    @staticmethod
    def _build_skipped_prediction(
        *,
        instance_index: int,
        instance: WebQSPProcessedInstance,
        global_node_vocabulary: dict[str, int],
    ) -> EvaluatedAnswerRetrievalInstance:
        """Persist an explicit miss when an architecture cannot score the graph."""
        gold_answer_scores = [
            GoldAnswerScore(
                node=gold_answer,
                local_node_id=instance.node2id.get(gold_answer),
                global_node_id=global_node_vocabulary.get(gold_answer),
                logit=None,
                probability=None,
                present_in_graph=gold_answer in instance.node2id,
            )
            for gold_answer in instance.a_entity
        ]
        return EvaluatedAnswerRetrievalInstance(
            instance_index=instance_index,
            question=instance.question,
            q_entity=instance.q_entity,
            a_entity=instance.a_entity,
            answer_candidates=[],
            gold_answer_scores=gold_answer_scores,
            hit_at_1=False,
            hit_at_5=False,
            hit_at_10=False,
            hit_at_candidate_limit=False,
            missing_gold_in_graph=any(
                not score.present_in_graph for score in gold_answer_scores
            ),
        )

    @staticmethod
    def _build_gold_answer_score(
        gold_answer: str,
        instance: WebQSPProcessedInstance,
        logits,
        probabilities,
        global_node_vocabulary: dict[str, int],
    ) -> GoldAnswerScore:
        if gold_answer not in instance.node2id:
            return GoldAnswerScore(
                node=gold_answer,
                local_node_id=None,
                global_node_id=global_node_vocabulary.get(gold_answer),
                logit=None,
                probability=None,
                present_in_graph=False,
            )

        node_id = instance.node2id[gold_answer]
        return GoldAnswerScore(
            node=gold_answer,
            local_node_id=node_id,
            global_node_id=global_node_vocabulary[gold_answer],
            logit=float(logits[node_id].item()),
            probability=float(probabilities[node_id].item()),
            present_in_graph=True,
        )

    @staticmethod
    def _build_evaluation_config_payload(
        evaluation_config: GnnAnswerRetrieverEvaluationConfig,
        loaded_model_run: LoadedGnnAnswerRetrieverRun,
        pipeline_configuration: BuiltPipelineConfiguration,
        device: str,
        prepared_evaluation_data: PreparedGnnEvaluationData,
    ) -> dict:
        evaluation_payload = evaluation_config.model_dump(mode="json")
        requested_run_name = evaluation_payload.pop("run_name", None)
        return {
            "dataset_id": pipeline_configuration.dataset_id,
            "gnn_architecture": loaded_model_run.config.resolved_gnn_architecture,
            "run_name": requested_run_name,
            "selected_device": device,
            "embedding_cache_device": (
                prepared_evaluation_data.embedding_cache_device
            ),
            "embedding_cache_dtype": prepared_evaluation_data.embedding_cache_dtype,
            "model_config": {
                "model_run_name": loaded_model_run.run_name,
                "model_run_number": loaded_model_run.run_number,
                "full_config_path": str(loaded_model_run.config_path),
                "weights_path": str(loaded_model_run.weights_path),
                **(
                    {
                        "relation_vocabulary_path": str(
                            loaded_model_run.relation_vocabulary_path
                        )
                    }
                    if loaded_model_run.relation_vocabulary_path is not None
                    else {}
                ),
            },
            "evaluation": evaluation_payload,
        }
