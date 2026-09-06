"""Load saved GNN predictions as candidate scores for path extraction."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.abstract import AbstractStep, StepContext, StepResult
from pipeline.evaluation.models import (
    CandidateNodeScore,
    CandidateNodeScores,
    EvaluatedAnswerRetrievalInstance,
    EvaluationSample,
    GraphTriple,
)
from pipeline.preparation.exceptions import GnnAnswerRetrieverEvaluationException
from pipeline.preparation.helpers.dataset_definitions import WEBQSP_DATASET_ID
from pipeline.preparation.services.webqsp_local_graph_storage import WebQSPLocalGraphStorageService


class GnnPredictionCandidateScoringStep(
    AbstractStep[CandidateNodeScores, StepResult]
):
    """Create path-extraction candidate scores from a saved predictions file."""

    def __init__(
        self,
        predictions_path: str | Path,
        prediction_index: int = 0,
        top_k: int | None = None,
        processed_instances_path: str | Path | None = None,
        force_default: bool = False,
    ):
        super().__init__(force_default=force_default)
        self.predictions_path = Path(predictions_path)
        self.prediction_index = prediction_index
        self.top_k = top_k
        self.processed_instances_path = (
            Path(processed_instances_path)
            if processed_instances_path is not None
            else WebQSPLocalGraphStorageService().get_cache_directory(WEBQSP_DATASET_ID)
            / WebQSPLocalGraphStorageService.test_instances_filename
        )

    def execute_default(self, context: StepContext[StepResult]) -> CandidateNodeScores:
        prediction = self._load_prediction(self.prediction_index)
        processed_instance = self._load_processed_instance(prediction.instance_index)
        triples = self._build_graph_triples(processed_instance)
        sample = EvaluationSample(
            sample_id=str(prediction.instance_index),
            question=prediction.question,
            q_entities=prediction.q_entity,
            a_entities=prediction.a_entity,
            graph_triples=triples,
        )
        candidate_scores = [
            CandidateNodeScore(
                node_id=candidate.node,
                score=candidate.probability,
                local_node_id=candidate.local_node_id,
                global_node_id=candidate.global_node_id,
                logit=candidate.logit,
                probability=candidate.probability,
                is_gold_answer=candidate.is_gold_answer,
                selection_reason=candidate.selection_reason,
            )
            for candidate in prediction.answer_candidates
        ]
        if self.top_k is not None:
            candidate_scores = candidate_scores[: self.top_k]

        return CandidateNodeScores(
            sample=sample,
            candidates=candidate_scores,
            top_k=len(candidate_scores),
        )

    def _load_prediction(self, prediction_index: int) -> EvaluatedAnswerRetrievalInstance:
        predictions = self._load_predictions()
        if prediction_index < 0 or prediction_index >= len(predictions):
            raise GnnAnswerRetrieverEvaluationException(
                f"Prediction index {prediction_index} is out of range for "
                f"{self.predictions_path}; found {len(predictions)} predictions."
            )

        return predictions[prediction_index]

    def _load_predictions(self) -> list[EvaluatedAnswerRetrievalInstance]:
        try:
            raw_text = self.predictions_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Could not read predictions file {self.predictions_path}: {error}"
            ) from error

        if not raw_text:
            return []

        try:
            parsed = json.loads(raw_text)
            raw_predictions = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            raw_predictions = self._parse_json_objects(raw_text)

        return [
            EvaluatedAnswerRetrievalInstance.model_validate(raw_prediction)
            for raw_prediction in raw_predictions
        ]

    @staticmethod
    def _parse_json_objects(raw_text: str) -> list[dict[str, object]]:
        decoder = json.JSONDecoder()
        position = 0
        objects: list[dict[str, object]] = []
        text_length = len(raw_text)

        while position < text_length:
            while position < text_length and raw_text[position].isspace():
                position += 1
            if position >= text_length:
                break

            value, position = decoder.raw_decode(raw_text, position)
            if not isinstance(value, dict):
                raise GnnAnswerRetrieverEvaluationException(
                    "Predictions file must contain JSON objects."
                )
            objects.append(value)

        return objects

    def _load_processed_instance(self, instance_index: int):
        try:
            instances = WebQSPLocalGraphStorageService().load_instances(
                self.processed_instances_path
            )
        except Exception as error:
            raise GnnAnswerRetrieverEvaluationException(
                f"Could not load processed test instances from "
                f"{self.processed_instances_path}: {error}"
            ) from error

        if instance_index < 0 or instance_index >= len(instances):
            raise GnnAnswerRetrieverEvaluationException(
                f"Processed instance index {instance_index} is out of range for "
                f"{self.processed_instances_path}; found {len(instances)} instances."
            )

        return instances[instance_index]

    @staticmethod
    def _build_graph_triples(processed_instance) -> list[GraphTriple]:
        edge_index = processed_instance.edge_index.tolist()
        edge_sources = edge_index[0]
        edge_targets = edge_index[1]

        return [
            GraphTriple(
                source=processed_instance.nodes[source_id],
                relation=relation,
                target=processed_instance.nodes[target_id],
            )
            for source_id, target_id, relation in zip(
                edge_sources,
                edge_targets,
                processed_instance.edge_relations,
                strict=True,
            )
        ]
