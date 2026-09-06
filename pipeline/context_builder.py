"""Centralized utilities for building step contexts between pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.exceptions import PipelineException
from pipeline.evaluation.steps.gnn_answer_retriever_evaluation import (
    EvaluateGnnAnswerRetrieverContext,
    EvaluateGnnAnswerRetrieverStep,
)
from pipeline.evaluation.steps.final_results_evaluation import (
    ComputeFinalResultsContext,
    ComputeFinalResultsStep,
)
from pipeline.evaluation.steps.llm_inference import (
    BuildEvidenceSubgraphsBatchStep,
    BuildEvidenceSubgraphsContext,
    BuildReasoningSamplesFromGnnEvaluationContext,
    BuildReasoningSamplesFromGnnEvaluationStep,
    GenerateAndSaveFinalAnswersBatchesContext,
    GenerateAndSaveFinalAnswersBatchesStep,
    SaveEvidenceSubgraphsContext,
    SaveEvidenceSubgraphsStep,
)
from pipeline.models import PipelineResultBank
from pipeline.evaluation.models import GnnAnswerRetrieverEvaluationResult
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.preparation.steps.gnn_model_building import (
    BuildGnnAnswerRetrieverContext,
    BuildGnnAnswerRetrieverStep,
)
from pipeline.preparation.models.webqsp_local_graph import PreparedWebQSPGraphDataset
from pipeline.preparation.steps.webqsp_local_graph_preparation import (
    BuildWebQSPLocalGraphsContext,
    BuildWebQSPLocalGraphsStep,
)
from pipeline.preparation.steps.gnn_answer_retriever_training import (
    TrainGnnAnswerRetrieverContext,
    TrainGnnAnswerRetrieverStep,
)
from pipeline.preparation.steps.gnn_training_data_preparation import (
    PrepareGnnTrainingDataContext,
    PrepareGnnTrainingDataStep,
)


class StepContextBuilder:
    """Creates step contexts from prior step outputs."""

    def __init__(self) -> None:
        self.result_bank = PipelineResultBank()
        self.context_factories: dict[
            type[AbstractStep],
            Callable[[StepResult | None, bool, PipelineException], StepContext],
        ] = {
            BuildWebQSPLocalGraphsStep: self._create_build_webqsp_local_graphs_context,
            BuildGnnAnswerRetrieverStep: self._create_gnn_answer_retriever_context,
            PrepareGnnTrainingDataStep: self._create_prepare_gnn_training_data_context,
            TrainGnnAnswerRetrieverStep: self._create_train_gnn_answer_retriever_context,
            EvaluateGnnAnswerRetrieverStep: self._create_evaluate_gnn_answer_retriever_context,
            BuildReasoningSamplesFromGnnEvaluationStep: (
                self._create_build_reasoning_samples_from_gnn_context
            ),
            BuildEvidenceSubgraphsBatchStep: self._create_evidence_subgraphs_context,
            SaveEvidenceSubgraphsStep: self._create_save_evidence_subgraphs_context,
            GenerateAndSaveFinalAnswersBatchesStep: (
                self._create_generate_and_save_final_answers_batches_context
            ),
            ComputeFinalResultsStep: self._create_compute_final_results_context,
        }

    def create_context(
        self,
        result: Optional[StepResult],
        outcome: bool = True,
        exception=None,
        next_step: AbstractStep | None = None,
    ) -> StepContext:
        """Wrap a previous result into the context required by the next step."""
        if result is not None:
            self.store_result(result)

        if next_step is not None:
            for step_type, context_factory in self.context_factories.items():
                if isinstance(next_step, step_type):
                    return context_factory(result, outcome, exception)

        return StepContext(
            result=result,
            outcome=outcome,
            exception=exception,
        )

    def _create_build_webqsp_local_graphs_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> BuildWebQSPLocalGraphsContext:
        """Create the context required by WebQSP graph preparation."""
        return BuildWebQSPLocalGraphsContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_gnn_answer_retriever_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> BuildGnnAnswerRetrieverContext:
        """Create the specialized context required by the GNN builder step."""
        return BuildGnnAnswerRetrieverContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_train_gnn_answer_retriever_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> TrainGnnAnswerRetrieverContext:
        """Create the specialized context required by the GNN training step."""
        return TrainGnnAnswerRetrieverContext(
            result=result,
            outcome=outcome,
            exception=exception,
            prepared_dataset=self.get_required_result(PreparedWebQSPGraphDataset),
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_prepare_gnn_training_data_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> PrepareGnnTrainingDataContext:
        """Create the context required by GNN training-data preparation."""
        return PrepareGnnTrainingDataContext(
            result=result,
            outcome=outcome,
            exception=exception,
            prepared_dataset=self.get_required_result(PreparedWebQSPGraphDataset),
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_evaluate_gnn_answer_retriever_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> EvaluateGnnAnswerRetrieverContext:
        """Create the specialized context required by the GNN evaluation step."""
        return EvaluateGnnAnswerRetrieverContext(
            result=result,
            outcome=outcome,
            exception=exception,
            prepared_dataset=self.get_required_result(PreparedWebQSPGraphDataset),
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_build_reasoning_samples_from_gnn_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> BuildReasoningSamplesFromGnnEvaluationContext:
        """Create the context required by the post-retrieval reasoning adapter."""
        return BuildReasoningSamplesFromGnnEvaluationContext(
            result=result,
            outcome=outcome,
            exception=exception,
            prepared_dataset=self.get_required_result(PreparedWebQSPGraphDataset),
        )

    def _create_generate_and_save_final_answers_batches_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> GenerateAndSaveFinalAnswersBatchesContext:
        """Create the context required by batched LLM inference."""
        return GenerateAndSaveFinalAnswersBatchesContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_save_evidence_subgraphs_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> SaveEvidenceSubgraphsContext:
        return SaveEvidenceSubgraphsContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
            gnn_evaluation_result=self.get_required_result(
                GnnAnswerRetrieverEvaluationResult
            ),
        )

    def _create_evidence_subgraphs_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> BuildEvidenceSubgraphsContext:
        """Create the context required by strategy-aware evidence construction."""
        return BuildEvidenceSubgraphsContext(
            result=result,
            outcome=outcome,
            exception=exception,
            pipeline_configuration=self.get_required_result(
                BuiltPipelineConfiguration
            ),
        )

    def _create_compute_final_results_context(
        self,
        result: StepResult | None,
        outcome: bool,
        exception: PipelineException,
    ) -> ComputeFinalResultsContext:
        """Create the context required by final results evaluation."""
        return ComputeFinalResultsContext(
            result=result,
            outcome=outcome,
            exception=exception,
            gnn_evaluation_result=self.get_required_result(
                GnnAnswerRetrieverEvaluationResult
            ),
        )

    def store_result(self, result: StepResult) -> None:
        """Store a result in the shared in-memory bank."""
        self.result_bank.store(result)

    def get_stored_result(self, result_type: type[StepResult]) -> StepResult | None:
        """Return the latest stored result for a given result type."""
        return self.result_bank.get(result_type)

    def get_required_result(self, result_type: type[StepResult]) -> StepResult:
        """Return the latest stored result or raise when it is missing."""
        return self.result_bank.get_required(result_type)

    def has_stored_result(self, result_type: type[StepResult]) -> bool:
        """Check whether the given result type exists in the bank."""
        return self.result_bank.has(result_type)

    def clear_result_bank(self) -> None:
        """Clear the shared result bank."""
        self.result_bank.clear()
