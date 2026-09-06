"""Tests for evaluation candidate scoring and shortest path extraction."""

import json
import tempfile
import threading
import unittest
from pathlib import Path

from pipeline import (
    BuildReasoningSamplesFromGnnEvaluationContext,
    BuildReasoningSamplesFromGnnEvaluationStep,
    CandidateNodeScore,
    EvaluationSample,
    EvaluatedAnswerRetrievalInstance,
    ExtractedReasoningPathsBatch,
    ExtractShortestPathsBatchStep,
    ExtractShortestPathsStep,
    GenerateAndSaveFinalAnswersBatchesStep,
    GenerateFinalAnswersBatchStep,
    GenerateFinalAnswerStep,
    GnnAnswerRetrieverEvaluationResult,
    GnnPredictionCandidateScoringStep,
    InvalidEvaluationSampleException,
    MockCandidateNodeScoringStep,
    PreparedWebQSPGraphDataset,
    Pipeline,
    SaveInferenceRunStep,
    ShortestPathExtractionService,
    StepContext,
    ReasoningPathsForPrediction,
    WebQSPVocabularyStore,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance
from pipeline.evaluation.models import (
    GeneratedAnswerForPrediction,
    GeneratedFinalAnswersBatch,
)
from pipeline.evaluation.services.llm_inference_storage import (
    LlmInferenceStorageService,
)
from pipeline.evaluation.exceptions import InsufficientLlmCreditsException


def make_sample() -> EvaluationSample:
    return EvaluationSample.from_webqsp_row(
        {
            "id": "sample-1",
            "question": "what is the name of justin bieber brother",
            "q_entity": ["Justin Bieber"],
            "a_entity": ["Jaxon Bieber"],
            "graph": [
                ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                ["m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"],
                ["Justin Bieber", "people.person.place_of_birth", "London"],
                ["Scooter Braun", "music.artist.manager", "Justin Bieber"],
            ],
        }
    )


class EvaluationSampleParsingTests(unittest.TestCase):
    def test_valid_webqsp_row_becomes_evaluation_sample(self) -> None:
        print("\n[test_valid_webqsp_row_becomes_evaluation_sample] Starting.")
        sample = make_sample()

        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(sample.question, "what is the name of justin bieber brother")
        self.assertEqual(sample.q_entities, ["Justin Bieber"])
        self.assertEqual(sample.a_entities, ["Jaxon Bieber"])
        self.assertEqual(len(sample.graph_triples), 4)
        self.assertEqual(sample.graph_triples[0].relation, "people.person.sibling_s")
        print("[test_valid_webqsp_row_becomes_evaluation_sample] Passed.")

    def test_missing_q_entity_fails(self) -> None:
        print("\n[test_missing_q_entity_fails] Starting.")
        with self.assertRaises(InvalidEvaluationSampleException):
            EvaluationSample.from_webqsp_row(
                {
                    "question": "who?",
                    "a_entity": ["answer"],
                    "graph": [["a", "r", "b"]],
                }
            )
        print("[test_missing_q_entity_fails] Passed.")

    def test_malformed_graph_triple_fails(self) -> None:
        print("\n[test_malformed_graph_triple_fails] Starting.")
        with self.assertRaises(InvalidEvaluationSampleException):
            EvaluationSample.from_webqsp_row(
                {
                    "question": "who?",
                    "q_entity": ["topic"],
                    "a_entity": ["answer"],
                    "graph": [["a", "r"]],
                }
            )
        print("[test_malformed_graph_triple_fails] Passed.")


class MockCandidateNodeScoringStepTests(unittest.TestCase):
    def test_mock_candidates_include_gold_then_distractors(self) -> None:
        print("\n[test_mock_candidates_include_gold_then_distractors] Starting.")
        step = MockCandidateNodeScoringStep(top_k=3)
        result = step.execute(StepContext(result=make_sample()))

        self.assertEqual(result.top_k, 3)
        self.assertEqual(result.candidates[0].node_id, "Jaxon Bieber")
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(len({candidate.node_id for candidate in result.candidates}), 3)
        print("[test_mock_candidates_include_gold_then_distractors] Passed.")


class GnnPredictionCandidateScoringStepTests(unittest.TestCase):
    def test_predictions_file_becomes_candidate_scores_with_graph(self) -> None:
        print("\n[test_predictions_file_becomes_candidate_scores_with_graph] Starting.")
        import torch

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            predictions_path = directory / "predictions.jsonl"
            instances_path = directory / "test_instances.pt"
            predictions_path.write_text(
                """
{
  "instance_index": 0,
  "question": "who is connected",
  "q_entity": ["Topic"],
  "a_entity": ["Answer"],
  "answer_candidates": [
    {
      "node": "Answer",
      "local_node_id": 1,
      "global_node_id": 10,
      "logit": 2.0,
      "probability": 0.88,
      "is_gold_answer": true,
      "selection_reason": "threshold"
    }
  ],
  "gold_answer_scores": [],
  "hit_at_1": true,
  "hit_at_5": true,
  "hit_at_10": true,
  "missing_gold_in_graph": false
}
""".strip(),
                encoding="utf-8",
            )
            torch.save(
                [
                    WebQSPProcessedInstance(
                        question="who is connected",
                        q_entity=["Topic"],
                        a_entity=["Answer"],
                        nodes=["Topic", "Answer"],
                        node2id={"Topic": 0, "Answer": 1},
                        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                        edge_relations=["related.to"],
                        node_labels=torch.tensor([0.0, 1.0]),
                    )
                ],
                instances_path,
            )

            result = GnnPredictionCandidateScoringStep(
                predictions_path=predictions_path,
                processed_instances_path=instances_path,
            ).execute(StepContext())

        self.assertEqual(result.sample.question, "who is connected")
        self.assertEqual(result.candidates[0].node_id, "Answer")
        self.assertEqual(result.candidates[0].score, 0.88)
        self.assertEqual(result.candidates[0].local_node_id, 1)
        self.assertEqual(result.candidates[0].global_node_id, 10)
        self.assertEqual(result.candidates[0].logit, 2.0)
        self.assertEqual(result.candidates[0].probability, 0.88)
        self.assertTrue(result.candidates[0].is_gold_answer)
        self.assertEqual(result.candidates[0].selection_reason, "threshold")
        self.assertEqual(result.sample.graph_triples[0].source, "Topic")
        self.assertEqual(result.sample.graph_triples[0].relation, "related.to")
        self.assertEqual(result.sample.graph_triples[0].target, "Answer")
        print("[test_predictions_file_becomes_candidate_scores_with_graph] Passed.")


class ShortestPathExtractionServiceTests(unittest.TestCase):
    @staticmethod
    def _make_processed_instance_with_reverse_edges():
        import torch

        return WebQSPProcessedInstance(
            question="who is connected",
            q_entity=["Topic"],
            a_entity=["Answer"],
            nodes=["Topic", "Middle", "Answer"],
            node2id={"Topic": 0, "Middle": 1, "Answer": 2},
            edge_index=torch.tensor(
                [[0, 1, 1, 2], [1, 2, 0, 1]],
                dtype=torch.long,
            ),
            edge_relations=["r1", "r2", "reverse__r1", "reverse__r2"],
            node_labels=torch.tensor([0.0, 0.0, 1.0]),
        )

    def test_multi_hop_path_preserves_relation_labels(self) -> None:
        print("\n[test_multi_hop_path_preserves_relation_labels] Starting.")
        service = ShortestPathExtractionService()
        sample = make_sample()
        result = service.extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)
            ],
        )

        self.assertEqual(result.found_paths, 1)
        self.assertEqual(result.missing_paths, 0)
        self.assertTrue(result.paths[0].path_found)
        self.assertEqual(
            [triple.relation for triple in result.paths[0].triples],
            ["people.person.sibling_s", "people.sibling_relationship.sibling"],
        )
        self.assertIn(
            "Justin Bieber -> people.person.sibling_s -> m.0gxnnwp",
            result.reasoning_paths_text,
        )
        self.assertEqual(
            [triple.relation for triple in result.reasoning_subgraph_triples],
            ["people.person.sibling_s", "people.sibling_relationship.sibling"],
        )
        print("[test_multi_hop_path_preserves_relation_labels] Passed.")

    def test_reasoning_subgraph_deduplicates_shortest_path_triples(self) -> None:
        print("\n[test_reasoning_subgraph_deduplicates_shortest_path_triples] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who is connected",
                "q_entity": ["Topic"],
                "a_entity": ["Answer A", "Answer B"],
                "graph": [
                    ["Topic", "r1", "Shared"],
                    ["Shared", "r2", "Answer A"],
                    ["Shared", "r3", "Answer B"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Answer A", score=1.0),
                CandidateNodeScore(node_id="Answer B", score=0.9),
            ],
        )

        self.assertEqual(
            [(triple.source, triple.relation, triple.target) for triple in result.reasoning_subgraph_triples],
            [
                ("Topic", "r1", "Shared"),
                ("Shared", "r2", "Answer A"),
                ("Shared", "r3", "Answer B"),
            ],
        )
        self.assertEqual(result.reasoning_paths_text.count("Topic -> r1 -> Shared"), 1)
        print("[test_reasoning_subgraph_deduplicates_shortest_path_triples] Passed.")

    def test_all_equal_length_shortest_paths_contribute_to_subgraph(self) -> None:
        print("\n[test_all_equal_length_shortest_paths_contribute_to_subgraph] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "what is the name of justin bieber brother",
                "q_entity": ["Justin Bieber"],
                "a_entity": ["Jaxon Bieber"],
                "graph": [
                    ["Justin Bieber", "people.person.nationality", "Canada"],
                    ["Jaxon Bieber", "people.person.nationality", "Canada"],
                    ["Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                    ["Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"],
                    ["m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"],
                    ["m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)
            ],
        )

        self.assertEqual(len(result.paths[0].shortest_paths), 5)
        self.assertEqual(
            {
                tuple((triple.source, triple.relation, triple.target) for triple in path)
                for path in result.paths[0].shortest_paths
            },
            {
                (
                    ("Justin Bieber", "people.person.nationality", "Canada"),
                    ("Jaxon Bieber", "people.person.nationality", "Canada"),
                ),
                (
                    ("Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                    ("Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                ),
                (
                    ("Justin Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"),
                ),
                (
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"),
                    ("Jaxon Bieber", "people.person.sibling_s", "m.0gxnnwp"),
                ),
                (
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Justin Bieber"),
                    ("m.0gxnnwp", "people.sibling_relationship.sibling", "Jaxon Bieber"),
                ),
            },
        )
        self.assertIn(
            "m.0gxnnwp -> people.sibling_relationship.sibling -> Jaxon Bieber",
            result.reasoning_paths_text,
        )
        print("[test_all_equal_length_shortest_paths_contribute_to_subgraph] Passed.")

    def test_direct_reverse_traversal_path_is_found(self) -> None:
        print("\n[test_direct_reverse_traversal_path_is_found] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who manages justin bieber",
                "q_entity": ["Justin Bieber"],
                "a_entity": ["Scooter Braun"],
                "graph": [["Scooter Braun", "music.artist.manager", "Justin Bieber"]],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[
                CandidateNodeScore(node_id="Scooter Braun", score=1.0)
            ],
        )

        self.assertTrue(result.paths[0].path_found)
        self.assertEqual(result.paths[0].triples[0].source, "Scooter Braun")
        self.assertEqual(result.paths[0].triples[0].target, "Justin Bieber")
        print("[test_direct_reverse_traversal_path_is_found] Passed.")

    def test_multiple_q_entities_choose_shortest_available_path(self) -> None:
        print("\n[test_multiple_q_entities_choose_shortest_available_path] Starting.")
        sample = EvaluationSample.from_webqsp_row(
            {
                "question": "who is connected",
                "q_entity": ["Far Topic", "Near Topic"],
                "a_entity": ["Answer"],
                "graph": [
                    ["Far Topic", "r1", "Middle"],
                    ["Middle", "r2", "Answer"],
                    ["Near Topic", "r3", "Answer"],
                ],
            }
        )
        result = ShortestPathExtractionService().extract_paths(
            sample=sample,
            candidates=[CandidateNodeScore(node_id="Answer", score=1.0)],
        )

        self.assertEqual(len(result.paths[0].triples), 1)
        self.assertEqual(result.paths[0].triples[0].relation, "r3")
        print("[test_multiple_q_entities_choose_shortest_available_path] Passed.")

    def test_unreachable_candidate_is_recorded_without_abort(self) -> None:
        print("\n[test_unreachable_candidate_is_recorded_without_abort] Starting.")
        result = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[
                CandidateNodeScore(node_id="Unreachable", score=0.3)
            ],
        )

        self.assertEqual(result.found_paths, 0)
        self.assertEqual(result.missing_paths, 1)
        self.assertFalse(result.paths[0].path_found)
        self.assertIn("No reasoning subgraph found.", result.reasoning_paths_text)
        print("[test_unreachable_candidate_is_recorded_without_abort] Passed.")

    def test_processed_graph_path_uses_entity_names_and_ignores_reverse_duplicates(
        self,
    ) -> None:
        service = ShortestPathExtractionService()
        result = service.extract_paths_from_processed_graph(
            instance=self._make_processed_instance_with_reverse_edges(),
            sample=EvaluationSample(
                sample_id="0",
                question="who is connected",
                q_entities=["Topic"],
                a_entities=["Answer"],
                graph_triples=[],
            ),
            candidates=[
                CandidateNodeScore(
                    node_id="Answer",
                    score=0.9,
                    local_node_id=2,
                )
            ],
        )

        self.assertEqual(len(result.paths[0].shortest_paths), 1)
        self.assertEqual(
            [
                (triple.source, triple.relation, triple.target)
                for triple in result.paths[0].triples
            ],
            [("Topic", "r1", "Middle"), ("Middle", "r2", "Answer")],
        )
        self.assertNotIn("reverse__", result.reasoning_paths_text)

    def test_processed_graph_extracts_multiple_candidates_with_one_bfs(self) -> None:
        import torch
        from unittest.mock import patch

        instance = WebQSPProcessedInstance(
            question="who is connected",
            q_entity=["Topic"],
            a_entity=["Answer A", "Answer B"],
            nodes=["Topic", "Shared", "Answer A", "Answer B"],
            node2id={"Topic": 0, "Shared": 1, "Answer A": 2, "Answer B": 3},
            edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long),
            edge_relations=["r1", "r2", "r3"],
            node_labels=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )
        sample = EvaluationSample(
            sample_id="0",
            question=instance.question,
            q_entities=instance.q_entity,
            a_entities=instance.a_entity,
            graph_triples=[],
        )
        service = ShortestPathExtractionService()

        with patch.object(
            service,
            "_multi_target_bfs",
            wraps=service._multi_target_bfs,
        ) as bfs:
            result = service.extract_paths_from_processed_graph(
                instance=instance,
                sample=sample,
                candidates=[
                    CandidateNodeScore(node_id="Answer A", score=0.9),
                    CandidateNodeScore(node_id="Answer B", score=0.8),
                ],
            )

        self.assertEqual(bfs.call_count, 1)
        self.assertEqual(result.found_paths, 2)
        self.assertEqual(
            [triple.relation for triple in result.reasoning_subgraph_triples],
            ["r1", "r2", "r3"],
        )

    def test_processed_graph_candidate_seed_has_empty_shortest_path(self) -> None:
        instance = self._make_processed_instance_with_reverse_edges()
        result = ShortestPathExtractionService().extract_paths_from_processed_graph(
            instance=instance,
            sample=EvaluationSample(
                sample_id="0",
                question=instance.question,
                q_entities=instance.q_entity,
                a_entities=instance.a_entity,
                graph_triples=[],
            ),
            candidates=[CandidateNodeScore(node_id="Topic", score=1.0)],
        )

        self.assertTrue(result.paths[0].path_found)
        self.assertEqual(result.paths[0].shortest_paths, [[]])


class PathExtractionPipelineTests(unittest.TestCase):
    def test_mock_scoring_and_path_extraction_run_as_pipeline_steps(self) -> None:
        print("\n[test_mock_scoring_and_path_extraction_run_as_pipeline_steps] Starting.")
        pipeline = Pipeline(
            evaluation_steps=[
                MockCandidateNodeScoringStep(top_k=3),
                ExtractShortestPathsStep(),
            ],
        )

        result = pipeline.evaluate(StepContext(result=make_sample()))

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 2)
        self.assertEqual(result.final_result.sample.sample_id, "sample-1")
        self.assertGreaterEqual(result.final_result.found_paths, 1)
        self.assertIn("Reasoning subgraph:", result.final_result.reasoning_paths_text)
        print("[test_mock_scoring_and_path_extraction_run_as_pipeline_steps] Passed.")


class FakeAnswerGenerationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def generate_answer(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
    ) -> tuple[list[str], str]:
        self.calls.append((question, reasoning_paths_text, model_id))
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Reasoning paths:\n"
            f"{reasoning_paths_text}\n\n"
            "Answer the question using only the reasoning paths."
        )
        return ["Jaxon Bieber"], prompt

    def generate_answer_with_explanation(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
        provider_id: str = "openai",
        reasoning_effort: str | None = None,
        generate_explanation: bool = True,
    ) -> dict[str, str]:
        self.calls.append((question, reasoning_paths_text, model_id))
        return {
            "answers": ["Jaxon Bieber"],
            "explanation": (
                "Used Justin Bieber -> people.person.sibling_s -> m.0gxnnwp "
                "and m.0gxnnwp -> people.sibling_relationship.sibling -> Jaxon Bieber."
            ),
            "raw_response": json.dumps(
                {
                    "answers": ["Jaxon Bieber"],
                    "explanation": (
                        "Used Justin Bieber -> people.person.sibling_s -> m.0gxnnwp "
                        "and m.0gxnnwp -> people.sibling_relationship.sibling -> Jaxon Bieber."
                    ),
                }
            ),
            "prompt": "unused in batch storage",
        }


class ConcurrentFakeAnswerGenerationService:
    def __init__(self, expected_parallel_calls: int) -> None:
        self.expected_parallel_calls = expected_parallel_calls
        self.active_calls = 0
        self.maximum_active_calls = 0
        self._lock = threading.Lock()
        self._all_workers_started = threading.Event()

    def generate_answer_with_explanation(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
        provider_id: str = "openai",
        reasoning_effort: str | None = None,
        generate_explanation: bool = True,
    ) -> dict[str, str | int | float]:
        with self._lock:
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls,
                self.active_calls,
            )
            if self.active_calls == self.expected_parallel_calls:
                self._all_workers_started.set()

        workers_started = self._all_workers_started.wait(timeout=2.0)
        with self._lock:
            self.active_calls -= 1
        if not workers_started:
            raise AssertionError("Expected concurrent LLM calls did not start.")

        return {
            "answers": ["Answer"],
            "explanation": "Explanation",
            "raw_response": '{"answers":["Answer"],"explanation":"Explanation"}',
            "prompt": question,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "estimated_cost_usd": 0.0,
        }


class InternallyBrokenAnswerGenerationService:
    def __init__(self) -> None:
        self.calls = 0

    def generate_answer_with_explanation(
        self,
        question: str,
        reasoning_paths_text: str,
        model_id: str,
        provider_id: str = "openai",
        reasoning_effort: str | None = None,
        generate_explanation: bool = True,
    ) -> dict[str, str]:
        self.calls += 1
        raise TypeError("internal reasoning_effort handling failed")


class LlmAnswerGenerationStepTests(unittest.TestCase):
    def test_generates_final_answer_from_extracted_paths(self) -> None:
        print("\n[test_generates_final_answer_from_extracted_paths] Starting.")
        extracted_paths = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)],
        )
        fake_service = FakeAnswerGenerationService()
        step = GenerateFinalAnswerStep(
            model_id="test-model",
            answer_generation_service=fake_service,
        )

        result = step.execute(StepContext(result=extracted_paths))

        self.assertEqual(result.answers, ["Jaxon Bieber"])
        self.assertEqual(result.model_id, "test-model")
        self.assertEqual(result.extracted_paths, extracted_paths)
        self.assertIn("Question:", result.prompt)
        self.assertIn("Reasoning paths:", result.prompt)
        self.assertEqual(fake_service.calls[0][0], make_sample().question)
        self.assertEqual(fake_service.calls[0][1], extracted_paths.reasoning_paths_text)
        self.assertEqual(fake_service.calls[0][2], "test-model")
        print("[test_generates_final_answer_from_extracted_paths] Passed.")


class LlmInferenceBatchStepTests(unittest.TestCase):
    def test_insufficient_credits_are_not_recorded_as_an_ordinary_failed_answer(
        self,
    ) -> None:
        class InsufficientCreditsService:
            def generate_answer_with_explanation(self, **kwargs):
                raise InsufficientLlmCreditsException("insufficient credits")

        extracted_paths = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)],
        )
        prediction = EvaluatedAnswerRetrievalInstance(
            instance_index=0,
            question=make_sample().question,
            q_entity=make_sample().q_entities,
            a_entity=make_sample().a_entities,
            answer_candidates=[],
            gold_answer_scores=[],
            hit_at_1=False,
            missing_gold_in_graph=False,
        )
        item = ReasoningPathsForPrediction(
            instance_index=0,
            prediction=prediction,
            extracted_paths=extracted_paths,
        )

        with self.assertRaises(InsufficientLlmCreditsException):
            GenerateFinalAnswersBatchStep(
                model_id="deepseek-v4-flash",
                llm_provider="deepseek",
                answer_generation_service=InsufficientCreditsService(),
            )._generate_answer(item)

    def test_internal_type_error_is_recorded_without_retrying_request(self) -> None:
        extracted_paths = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)],
        )
        prediction = EvaluatedAnswerRetrievalInstance(
            instance_index=0,
            question=make_sample().question,
            q_entity=make_sample().q_entities,
            a_entity=make_sample().a_entities,
            answer_candidates=[],
            gold_answer_scores=[],
            hit_at_1=False,
            missing_gold_in_graph=False,
        )
        item = ReasoningPathsForPrediction(
            instance_index=0,
            prediction=prediction,
            extracted_paths=extracted_paths,
        )
        fake_service = InternallyBrokenAnswerGenerationService()

        generated = GenerateFinalAnswersBatchStep(
            model_id="test-model",
            reasoning_effort="low",
            answer_generation_service=fake_service,
        )._generate_answer(item)

        self.assertEqual(fake_service.calls, 1)
        self.assertEqual(generated.answers, [])
        self.assertEqual(
            generated.error_message,
            "internal reasoning_effort handling failed",
        )

    def test_candidate_reduction_percentage_uses_global_candidate_counts(self) -> None:
        answers = GeneratedFinalAnswersBatch(
            dataset_id="WebQSP",
            evaluation_run_name="1_test",
            model_id="test-model",
            items=[
                GeneratedAnswerForPrediction(
                    instance_index=0,
                    question="Question one?",
                    model_id="test-model",
                    found_reasoning_paths=3,
                    missing_reasoning_paths=1,
                ),
                GeneratedAnswerForPrediction(
                    instance_index=1,
                    question="Question two?",
                    model_id="test-model",
                    found_reasoning_paths=2,
                    missing_reasoning_paths=2,
                ),
            ],
        )

        metrics = LlmInferenceStorageService._build_evidence_metrics(answers)

        self.assertEqual(metrics["candidate_reduction_percentage"], 37.5)

    @staticmethod
    def _make_processed_instance():
        import torch

        return WebQSPProcessedInstance(
            question="what is the name of justin bieber brother",
            q_entity=["Justin Bieber"],
            a_entity=["Jaxon Bieber"],
            nodes=["Justin Bieber", "m.0gxnnwp", "Jaxon Bieber"],
            node2id={"Justin Bieber": 0, "m.0gxnnwp": 1, "Jaxon Bieber": 2},
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_relations=[
                "people.person.sibling_s",
                "people.sibling_relationship.sibling",
            ],
            node_labels=torch.tensor([0.0, 0.0, 1.0]),
        )

    @staticmethod
    def _write_prediction_file(directory: Path) -> Path:
        predictions_path = directory / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "instance_index": 0,
                    "question": "what is the name of justin bieber brother",
                    "q_entity": ["Justin Bieber"],
                    "a_entity": ["Jaxon Bieber"],
                    "answer_candidates": [
                        {
                            "node": "Jaxon Bieber",
                            "local_node_id": 2,
                            "global_node_id": 102,
                            "logit": 5.0,
                            "probability": 0.99,
                            "is_gold_answer": True,
                            "selection_reason": "threshold",
                        }
                    ],
                    "gold_answer_scores": [],
                    "hit_at_1": True,
                    "hit_at_5": True,
                    "hit_at_10": True,
                    "missing_gold_in_graph": False,
                }
            ),
            encoding="utf-8",
        )
        return predictions_path

    def test_gnn_predictions_become_batch_reasoning_samples(self) -> None:
        print("\n[test_gnn_predictions_become_batch_reasoning_samples] Starting.")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            predictions_path = self._write_prediction_file(directory)
            prepared_dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="test",
                train_instances=[],
                test_instances=[self._make_processed_instance()],
                vocabulary_store=WebQSPVocabularyStore(),
                cache_directory=directory,
            )
            evaluation_result = GnnAnswerRetrieverEvaluationResult(
                dataset_id="WebQSP",
                model_run_directory=directory / "models" / "1_test",
                model_run_name="1_test",
                model_run_number=1,
                evaluation_run_directory=directory / "evaluations" / "1_test",
                evaluation_run_name="1_test",
                evaluation_run_number=1,
                evaluated_instances=1,
                hits_at_1=1.0,
                hits_at_1_count=1,
                hits_at_5=1.0,
                hits_at_5_count=1,
                hits_at_10=1.0,
                hits_at_10_count=1,
                average_candidate_count=1.0,
                missing_gold_in_graph_count=0,
                predictions_path=predictions_path,
                evaluation_config_path=directory / "evaluation_config.json",
            )

            result = BuildReasoningSamplesFromGnnEvaluationStep().execute(
                BuildReasoningSamplesFromGnnEvaluationContext(
                    result=evaluation_result,
                    prepared_dataset=prepared_dataset,
                )
            )

        self.assertEqual(result.dataset_id, "WebQSP")
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples[0].candidate_scores.candidates[0].node_id, "Jaxon Bieber")
        self.assertEqual(result.samples[0].candidate_scores.sample.graph_triples, [])
        self.assertIsNotNone(result.samples[0].graph_instance)
        self.assertEqual(
            result.samples[0].graph_instance.edge_relations[0],
            "people.person.sibling_s",
        )
        self.assertNotIn("graph_instance", result.samples[0].model_dump())
        print("[test_gnn_predictions_become_batch_reasoning_samples] Passed.")

    def test_batch_inference_saves_expected_files(self) -> None:
        print("\n[test_batch_inference_saves_expected_files] Starting.")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            predictions_path = self._write_prediction_file(directory)
            prepared_dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="test",
                train_instances=[],
                test_instances=[self._make_processed_instance()],
                vocabulary_store=WebQSPVocabularyStore(),
                cache_directory=directory,
            )
            evaluation_result = GnnAnswerRetrieverEvaluationResult(
                dataset_id="WebQSP",
                model_run_directory=directory / "models" / "1_test",
                model_run_name="1_test",
                model_run_number=1,
                evaluation_run_directory=directory / "evaluations" / "1_test",
                evaluation_run_name="1_test",
                evaluation_run_number=1,
                evaluated_instances=1,
                hits_at_1=1.0,
                hits_at_1_count=1,
                hits_at_5=1.0,
                hits_at_5_count=1,
                hits_at_10=1.0,
                hits_at_10_count=1,
                average_candidate_count=1.0,
                missing_gold_in_graph_count=0,
                predictions_path=predictions_path,
                evaluation_config_path=directory / "evaluation_config.json",
            )
            built_samples = BuildReasoningSamplesFromGnnEvaluationStep().execute(
                BuildReasoningSamplesFromGnnEvaluationContext(
                    result=evaluation_result,
                    prepared_dataset=prepared_dataset,
                )
            )
            paths_batch = ExtractShortestPathsBatchStep().execute(
                StepContext(result=built_samples)
            )
            answers_batch = GenerateFinalAnswersBatchStep(
                model_id="test-model",
                generate_explanation=True,
                answer_generation_service=FakeAnswerGenerationService(),
            ).execute(StepContext(result=paths_batch))

            saved_run = SaveInferenceRunStep(
                inference_root=directory / "inference",
                inference_run_name="test",
            ).execute(StepContext(result=answers_batch))

            self.assertTrue(saved_run.reasoning_path.exists())
            self.assertTrue(saved_run.answers_path.exists())
            self.assertTrue(saved_run.inference_config_path.exists())
            self.assertEqual(saved_run.total_instances, 1)
            self.assertEqual(saved_run.successful_answers, 1)
            self.assertFalse((saved_run.inference_run_directory / "reasoning_paths.jsonl").exists())
            self.assertFalse((saved_run.inference_run_directory / "prompts.jsonl").exists())
            self.assertFalse((saved_run.inference_run_directory / "reasoning_subgraphs.jsonl").exists())
            self.assertFalse((saved_run.inference_run_directory / "summary.json").exists())
            self.assertEqual(saved_run.reasoning_path.name, "reasoning.jsonl")
            self.assertEqual(saved_run.inference_config_path.name, "inference_config.json")
            reasoning_rows = [
                json.loads(line)
                for line in saved_run.reasoning_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(reasoning_rows), 1)
            reasoning_row = reasoning_rows[0]
            answers_text = saved_run.answers_path.read_text(encoding="utf-8")
            self.assertEqual(reasoning_row["q_entity"], ["Justin Bieber"])
            self.assertIn("subgraph", reasoning_row)
            self.assertNotIn("triples", reasoning_row)
            self.assertEqual(reasoning_row["analytics"]["total_subgraph_triples"], 2)
            self.assertEqual(reasoning_row["analytics"]["total_relations"], 2)
            self.assertEqual(reasoning_row["analytics"]["total_distinct_nodes"], 3)
            self.assertEqual(reasoning_row["analytics"]["max_length"], 2)
            self.assertEqual(reasoning_row["analytics"]["min_length"], 2)
            self.assertIn("Jaxon Bieber", answers_text)
            self.assertIn("explanation", answers_text)
            self.assertIn("Used Justin Bieber", answers_text)
            inference_config = json.loads(
                saved_run.inference_config_path.read_text(encoding="utf-8")
            )
            self.assertTrue(inference_config["inference"]["generate_explanation"])
        print("[test_batch_inference_saves_expected_files] Passed.")

    def test_batched_inference_saves_each_batch(self) -> None:
        print("\n[test_batched_inference_saves_each_batch] Starting.")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            predictions_path = self._write_prediction_file(directory)
            prepared_dataset = PreparedWebQSPGraphDataset(
                dataset_id="WebQSP",
                processing_version="test",
                train_instances=[],
                test_instances=[self._make_processed_instance()],
                vocabulary_store=WebQSPVocabularyStore(),
                cache_directory=directory,
            )
            evaluation_result = GnnAnswerRetrieverEvaluationResult(
                dataset_id="WebQSP",
                model_run_directory=directory / "models" / "1_test",
                model_run_name="1_test",
                model_run_number=1,
                evaluation_run_directory=directory / "evaluations" / "1_test",
                evaluation_run_name="1_test",
                evaluation_run_number=1,
                evaluated_instances=1,
                hits_at_1=1.0,
                hits_at_1_count=1,
                hits_at_5=1.0,
                hits_at_5_count=1,
                hits_at_10=1.0,
                hits_at_10_count=1,
                average_candidate_count=1.0,
                missing_gold_in_graph_count=0,
                predictions_path=predictions_path,
                evaluation_config_path=directory / "evaluation_config.json",
            )
            built_samples = BuildReasoningSamplesFromGnnEvaluationStep().execute(
                BuildReasoningSamplesFromGnnEvaluationContext(
                    result=evaluation_result,
                    prepared_dataset=prepared_dataset,
                )
            )
            paths_batch = ExtractShortestPathsBatchStep().execute(
                StepContext(result=built_samples)
            )
            paths_batch = paths_batch.model_copy(
                update={"items": [paths_batch.items[0], paths_batch.items[0]]}
            )
            fake_service = FakeAnswerGenerationService()

            saved_run = GenerateAndSaveFinalAnswersBatchesStep(
                model_id="test-model",
                inference_root=directory / "inference",
                inference_run_name="test",
                inference_batch_size=1,
                answer_generation_service=fake_service,
            ).execute(StepContext(result=paths_batch))

            answer_lines = saved_run.answers_path.read_text(encoding="utf-8").splitlines()
            reasoning_rows = saved_run.reasoning_path.read_text(encoding="utf-8").splitlines()
            summary = json.loads(saved_run.inference_config_path.read_text(encoding="utf-8"))
            self.assertEqual(len(answer_lines), 2)
            self.assertEqual(json.loads(answer_lines[0])["answers"], ["Jaxon Bieber"])
            self.assertEqual(len(reasoning_rows), 2)
            self.assertEqual(len(fake_service.calls), 2)
            self.assertEqual(saved_run.total_instances, 2)
            self.assertEqual(saved_run.successful_answers, 2)
            self.assertEqual(summary["inference"]["total_requests"], 2)
            self.assertEqual(summary["inference"]["total_tokens"], 0)
            self.assertEqual(summary["inference"]["total_cost_usd"], 0.0)
            self.assertFalse(summary["inference"]["generate_explanation"])
            self.assertEqual(
                summary["inference"]["evidence_metrics"][
                    "candidate_reduction_percentage"
                ],
                0.0,
            )
            self.assertTrue(
                all(
                    json.loads(line)["explanation"] == ""
                    for line in answer_lines
                )
            )
            self.assertEqual(summary["successful_answers"], 2)
        print("[test_batched_inference_saves_each_batch] Passed.")

    def test_parallel_inference_is_bounded_ordered_and_persisted(self) -> None:
        extracted_paths = ShortestPathExtractionService().extract_paths(
            sample=make_sample(),
            candidates=[CandidateNodeScore(node_id="Jaxon Bieber", score=1.0)],
        )
        prediction = EvaluatedAnswerRetrievalInstance(
            instance_index=0,
            question=make_sample().question,
            q_entity=make_sample().q_entities,
            a_entity=make_sample().a_entities,
            answer_candidates=[],
            gold_answer_scores=[],
            hit_at_1=False,
            missing_gold_in_graph=False,
        )
        items = [
            ReasoningPathsForPrediction(
                instance_index=instance_index,
                prediction=prediction.model_copy(
                    update={"instance_index": instance_index}
                ),
                extracted_paths=extracted_paths,
            )
            for instance_index in range(3)
        ]
        paths_batch = ExtractedReasoningPathsBatch(
            dataset_id="WebQSP",
            evaluation_run_name="1_test",
            items=items,
        )
        fake_service = ConcurrentFakeAnswerGenerationService(
            expected_parallel_calls=3
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            saved_run = GenerateAndSaveFinalAnswersBatchesStep(
                model_id="test-model",
                inference_root=Path(temporary_directory) / "inference",
                inference_batch_size=3,
                inference_parallel_calls=3,
                answer_generation_service=fake_service,
            ).execute(StepContext(result=paths_batch))

            answer_rows = [
                json.loads(line)
                for line in saved_run.answers_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            inference_config = json.loads(
                saved_run.inference_config_path.read_text(encoding="utf-8")
            )

        self.assertEqual(fake_service.maximum_active_calls, 3)
        self.assertEqual(
            [row["instance_index"] for row in answer_rows],
            [0, 1, 2],
        )
        self.assertEqual(saved_run.inference_parallel_calls, 3)
        self.assertEqual(inference_config["inference"]["parallel_calls"], 3)
        self.assertEqual(inference_config["inference"]["batch_size"], 3)


if __name__ == "__main__":
    unittest.main()
