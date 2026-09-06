"""Evaluation step for saved GNN answer-retriever runs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from helpers.constants import (
    DEFAULT_ANSWER_THRESHOLD,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE,
    DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE,
    DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB,
    DEFAULT_EVALUATION_LOG_EVERY,
    DEFAULT_EVALUATION_PROFILE,
)
from helpers.logging_config import get_logger
from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationConfig,
    GnnAnswerRetrieverEvaluationResult,
)
from pipeline.preparation.exceptions import InvalidInteractiveConfigurationInputException
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainedGnnAnswerRetriever,
)
from pipeline.preparation.services.gnn_answer_retriever_evaluation import (
    GnnAnswerRetrieverEvaluationService,
)
from pipeline.preparation.services.embedding_cache import WebQSPEmbeddingCacheService

logger = get_logger(__name__)


class EvaluateGnnAnswerRetrieverContext(StepContext[StepResult]):
    """Specialized context for evaluating a saved GNN answer-retriever run."""

    prepared_dataset: PreparedWebQSPGraphDataset = Field(
        ...,
        description="Prepared WebQSP graph dataset used for evaluation.",
    )
    pipeline_configuration: BuiltPipelineConfiguration = Field(
        ...,
        description="Pipeline configuration used for embedding cache choices.",
    )


class EvaluateGnnAnswerRetrieverStep(AbstractStep[GnnAnswerRetrieverEvaluationResult, StepResult]):
    """Evaluate a saved trained GNN answer-retriever run over WebQSP test graphs."""

    def __init__(
        self,
        model_run_name: str | None = None,
        model_run_number: int | None = None,
        answer_threshold: float = DEFAULT_ANSWER_THRESHOLD,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        evaluation_run_name: str | None = None,
        evaluation_max_instances: int | None = None,
        skip_missing_gold_in_graph: bool = True,
        evaluation_log_every: int = DEFAULT_EVALUATION_LOG_EVERY,
        evaluation_profile: bool = DEFAULT_EVALUATION_PROFILE,
        evaluation_embedding_cache_device: Literal[
            "auto", "gpu", "cpu"
        ] = DEFAULT_EVALUATION_EMBEDDING_CACHE_DEVICE,
        evaluation_embedding_cache_dtype: Literal[
            "auto", "float32", "bfloat16"
        ] = DEFAULT_EVALUATION_EMBEDDING_CACHE_DTYPE,
        evaluation_gpu_cache_reserve_gb: float = (
            DEFAULT_EVALUATION_GPU_CACHE_RESERVE_GB
        ),
        evaluation_service: GnnAnswerRetrieverEvaluationService | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.evaluation_config = GnnAnswerRetrieverEvaluationConfig(
            model_run_name=model_run_name,
            model_run_number=model_run_number,
            answer_threshold=answer_threshold,
            candidate_top_k=candidate_top_k,
            candidate_limit=candidate_limit,
            run_name=evaluation_run_name,
            max_instances=evaluation_max_instances,
            skip_missing_gold_in_graph=skip_missing_gold_in_graph,
            log_every=evaluation_log_every,
            profile=evaluation_profile,
            embedding_cache_device=evaluation_embedding_cache_device,
            embedding_cache_dtype=evaluation_embedding_cache_dtype,
            gpu_cache_reserve_gb=evaluation_gpu_cache_reserve_gb,
        )
        self.evaluation_service = (
            evaluation_service
            or GnnAnswerRetrieverEvaluationService(
                embedding_cache_service=WebQSPEmbeddingCacheService()
            )
        )

    def execute_default(
        self,
        context: EvaluateGnnAnswerRetrieverContext,
    ) -> GnnAnswerRetrieverEvaluationResult:
        evaluation_config = self.evaluation_config
        if isinstance(context.result, TrainedGnnAnswerRetriever):
            evaluation_config = evaluation_config.model_copy(
                update={
                    "model_run_name": evaluation_config.model_run_name
                    or context.result.model_run_name,
                    "model_run_number": evaluation_config.model_run_number
                    or context.result.model_run_number,
                }
            )

        if (
            evaluation_config.model_run_name is None
            and evaluation_config.model_run_number is None
        ):
            raise InvalidInteractiveConfigurationInputException(
                "GNN answer-retriever evaluation requires a saved model run name "
                "or number."
            )

        logger.info(
            f"Starting EvaluateGnnAnswerRetrieverStep: "
            f"requested_model_run_name={evaluation_config.model_run_name} "
            f"requested_model_run_number={evaluation_config.model_run_number} "
            f"threshold={evaluation_config.answer_threshold} "
            f"candidate_top_k={evaluation_config.candidate_top_k} "
            f"candidate_limit={evaluation_config.candidate_limit} "
            f"log_every={evaluation_config.log_every} "
            f"profile={evaluation_config.profile} "
            f"cache_device={evaluation_config.embedding_cache_device} "
            f"cache_dtype={evaluation_config.embedding_cache_dtype}"
        )
        outcome = self.evaluation_service.evaluate(
            prepared_dataset=context.prepared_dataset,
            pipeline_configuration=context.pipeline_configuration,
            evaluation_config=evaluation_config,
        )
        logger.info(
            f"Finished EvaluateGnnAnswerRetrieverStep: "
            f"evaluation_run={outcome.storage_result.evaluation_run_name} "
            f"evaluated_instances={outcome.evaluated_instances}"
        )
        return GnnAnswerRetrieverEvaluationResult(
            dataset_id=context.prepared_dataset.dataset_id,
            gnn_architecture=outcome.loaded_model_run.config.resolved_gnn_architecture,
            model_run_directory=outcome.loaded_model_run.run_directory,
            model_run_name=outcome.loaded_model_run.run_name,
            model_run_number=outcome.loaded_model_run.run_number,
            evaluation_run_directory=outcome.storage_result.evaluation_run_directory,
            evaluation_run_name=outcome.storage_result.evaluation_run_name,
            evaluation_run_number=outcome.storage_result.evaluation_run_number,
            evaluated_instances=outcome.evaluated_instances,
            hits_at_1=outcome.hits_at_1,
            hits_at_1_count=outcome.hits_at_1_count,
            hits_at_5=outcome.hits_at_5,
            hits_at_5_count=outcome.hits_at_5_count,
            hits_at_10=outcome.hits_at_10,
            hits_at_10_count=outcome.hits_at_10_count,
            hits_at_candidate_limit=outcome.hits_at_candidate_limit,
            hits_at_candidate_limit_count=outcome.hits_at_candidate_limit_count,
            ndcg_at_1=outcome.ndcg_at_1,
            ndcg_at_5=outcome.ndcg_at_5,
            ndcg_at_10=outcome.ndcg_at_10,
            ndcg_at_candidate_limit=outcome.ndcg_at_candidate_limit,
            conditioned_evaluated_instances=(
                outcome.conditioned_evaluated_instances
            ),
            retrieval_gold_coverage=outcome.retrieval_gold_coverage,
            retrieval_full_gold_coverage_count=(
                outcome.retrieval_full_gold_coverage_count
            ),
            retrieval_full_gold_coverage_rate=(
                outcome.retrieval_full_gold_coverage_rate
            ),
            retrieved_gold_answer_count=outcome.retrieved_gold_answer_count,
            average_candidate_count=outcome.average_candidate_count,
            missing_gold_in_graph_count=outcome.missing_gold_in_graph_count,
            skipped_missing_gold_in_graph_count=(
                outcome.skipped_missing_gold_in_graph_count
            ),
            predictions_path=outcome.storage_result.predictions_path,
            evaluation_config_path=outcome.storage_result.evaluation_config_path,
            retrieval_metrics_path=outcome.storage_result.retrieval_metrics_path,
        )
