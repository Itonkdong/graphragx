"""Missing-gold graph filtering policy tests."""

from __future__ import annotations

import pytest

import main

torch = pytest.importorskip("torch")

from pipeline.evaluation.models import (
    GnnAnswerRetrieverEvaluationConfig,
    PreparedGnnEvaluationData,
    PreparedGnnEvaluationInstance,
)
from pipeline.preparation.models.gnn_training_data import (
    PreparedGnnTrainingData,
    PreparedGnnTrainingInstance,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance
from pipeline.preparation.services.gnn_evaluation_data_preparation import (
    GnnEvaluationDataPreparationService,
)
from pipeline.preparation.services.gnn_answer_retriever_evaluation import (
    GnnAnswerRetrieverEvaluationService,
)
from pipeline.preparation.services.gnn_training_data_preparation import (
    GnnTrainingDataPreparationConfig,
    GnnTrainingDataPreparationService,
)


def _graph(*, answer_present: bool) -> WebQSPProcessedInstance:
    return WebQSPProcessedInstance(
        question="question",
        q_entity=["seed"],
        a_entity=["answer"],
        nodes=["seed", "answer" if answer_present else "distractor"],
        node2id={"seed": 0, "answer" if answer_present else "distractor": 1},
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_relations=["relation"],
        node_labels=torch.tensor([0.0, 1.0 if answer_present else 0.0]),
    )


def test_missing_gold_skip_policy_is_default_on_with_only_opt_out_flag() -> None:
    parser = main.build_parser()

    assert parser.parse_args([]).skip_missing_gold_in_graph is True
    assert (
        parser.parse_args(
            ["--no-skip-missing-gold-in-graph"]
        ).skip_missing_gold_in_graph
        is False
    )
    assert main.PipelineRuntimeConfig().skip_missing_gold_in_graph is True
    assert GnnTrainingDataPreparationConfig().skip_missing_gold_in_graph is True
    assert GnnAnswerRetrieverEvaluationConfig().skip_missing_gold_in_graph is True


def test_training_filter_skips_graph_without_any_gold_node() -> None:
    valid = PreparedGnnTrainingInstance.model_construct(
        source_instance_index=4,
        node_labels=torch.tensor([0.0, 1.0]),
        gold_answer_count=1,
    )
    missing = PreparedGnnTrainingInstance.model_construct(
        source_instance_index=5,
        node_labels=torch.tensor([0.0, 0.0]),
        gold_answer_count=1,
    )
    prepared = PreparedGnnTrainingData.model_construct(instances=[valid, missing])

    filtered = GnnTrainingDataPreparationService._filter_missing_gold_training_instances(
        prepared_data=prepared,
        enabled=True,
        architecture_name="HGT",
    )

    assert [item.source_instance_index for item in filtered.instances] == [4]
    assert filtered.skipped_missing_gold_in_graph_count == 1
    assert (
        GnnTrainingDataPreparationService._filter_missing_gold_training_instances(
            prepared_data=prepared,
            enabled=False,
            architecture_name="HGT",
        )
        is prepared
    )


def test_training_filter_skips_graph_with_only_part_of_gold_set_present() -> None:
    complete = PreparedGnnTrainingInstance.model_construct(
        source_instance_index=1,
        node_labels=torch.tensor([0.0, 1.0, 1.0]),
        gold_answer_count=2,
    )
    partial = PreparedGnnTrainingInstance.model_construct(
        source_instance_index=2,
        node_labels=torch.tensor([0.0, 1.0, 0.0]),
        gold_answer_count=2,
    )
    prepared = PreparedGnnTrainingData.model_construct(
        instances=[complete, partial]
    )

    filtered = GnnTrainingDataPreparationService._filter_missing_gold_training_instances(
        prepared_data=prepared,
        enabled=True,
        architecture_name="ReaRev",
    )

    assert [item.source_instance_index for item in filtered.instances] == [1]
    assert filtered.skipped_missing_gold_in_graph_count == 1


def test_evaluation_filter_preserves_original_instance_indices() -> None:
    valid = PreparedGnnEvaluationInstance.model_construct(
        source_instance_index=8,
        instance=_graph(answer_present=True),
    )
    missing = PreparedGnnEvaluationInstance.model_construct(
        source_instance_index=9,
        instance=_graph(answer_present=False),
    )
    prepared = PreparedGnnEvaluationData.model_construct(instances=[valid, missing])

    filtered = (
        GnnEvaluationDataPreparationService._filter_missing_gold_evaluation_instances(
            prepared_data=prepared,
            enabled=True,
        )
    )

    assert [item.source_instance_index for item in filtered.instances] == [8]
    assert filtered.skipped_missing_gold_in_graph_count == 1
    assert (
        GnnEvaluationDataPreparationService._filter_missing_gold_evaluation_instances(
            prepared_data=prepared,
            enabled=False,
        )
        is prepared
    )


def test_evaluation_filter_skips_graph_with_only_part_of_gold_set_present() -> None:
    partial_graph = _graph(answer_present=True).model_copy(
        update={"a_entity": ["answer", "missing-answer"]}
    )
    partial = PreparedGnnEvaluationInstance.model_construct(
        source_instance_index=10,
        instance=partial_graph,
    )
    prepared = PreparedGnnEvaluationData.model_construct(instances=[partial])

    filtered = GnnEvaluationDataPreparationService._filter_missing_gold_evaluation_instances(
        prepared_data=prepared,
        enabled=True,
    )

    assert filtered.instances == []
    assert filtered.skipped_missing_gold_in_graph_count == 1


def test_prediction_marks_partial_gold_set_as_missing_in_graph() -> None:
    partial_graph = _graph(answer_present=True).model_copy(
        update={"a_entity": ["answer", "missing-answer"]}
    )

    prediction = GnnAnswerRetrieverEvaluationService._build_skipped_prediction(
        instance_index=10,
        instance=partial_graph,
        global_node_vocabulary={"seed": 0, "answer": 1},
    )

    assert prediction.missing_gold_in_graph is True
