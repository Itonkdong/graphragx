"""Unit tests for the graphragX pipeline foundation."""

import unittest

from pydantic import Field

from pipeline import (
    AbstractStep,
    InitialStepResult,
    Pipeline,
    StepContext,
    StepContextBuilder,
    StepResult,
)


class ExampleInitialResult(InitialStepResult):
    """Initial artifact for first-step tests."""

    query: str = "who?"


class QueryNodesResult(StepResult):
    """Example first-step output."""

    query_text: str
    nodes: list[str] = Field(default_factory=list)


class CandidateNodesResult(StepResult):
    """Example retrieval output."""

    nodes: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class FinalAnswerResult(StepResult):
    """Example terminal output."""

    answer: str


class InitialToQueryStep(AbstractStep[QueryNodesResult, ExampleInitialResult]):
    def execute_default(self, context: StepContext[ExampleInitialResult]) -> QueryNodesResult:
        return QueryNodesResult(query_text=context.result.query, nodes=["n1", "n2"])


class QueryToCandidatesStep(AbstractStep[CandidateNodesResult, QueryNodesResult]):
    def execute_default(self, context: StepContext[QueryNodesResult]) -> CandidateNodesResult:
        return CandidateNodesResult(nodes=context.result.nodes + ["n3"])


class CandidateToAnswerStep(AbstractStep[FinalAnswerResult, CandidateNodesResult]):
    def execute_default(self, context: StepContext[CandidateNodesResult]) -> FinalAnswerResult:
        return FinalAnswerResult(answer="done")


class BankAwareAnswerStep(AbstractStep[FinalAnswerResult, CandidateNodesResult]):
    def __init__(self, builder: StepContextBuilder):
        super().__init__()
        self.builder = builder

    def execute_default(self, context: StepContext[CandidateNodesResult]) -> FinalAnswerResult:
        query_result = self.builder.get_required_result(QueryNodesResult)
        return FinalAnswerResult(answer=f"answer-for-{query_result.query_text}")


class FailingStep(AbstractStep[CandidateNodesResult, QueryNodesResult]):
    def execute_default(self, context: StepContext[QueryNodesResult]) -> CandidateNodesResult:
        raise RuntimeError("boom")


class ContextBuilderTests(unittest.TestCase):
    def test_context_builder_wraps_previous_result(self) -> None:
        builder = StepContextBuilder()
        result = QueryNodesResult(query_text="q", nodes=["n1"])

        context = builder.create_context(result=result, outcome=True)

        self.assertIs(context.result, result)
        self.assertTrue(context.outcome)
        self.assertIsNone(context.exception)

    def test_builder_stores_and_retrieves_result_by_type(self) -> None:
        builder = StepContextBuilder()
        result = QueryNodesResult(query_text="q", nodes=["n1"])

        builder.store_result(result)

        self.assertTrue(builder.has_stored_result(QueryNodesResult))
        self.assertIs(builder.get_stored_result(QueryNodesResult), result)

    def test_builder_keeps_different_result_types_independently(self) -> None:
        builder = StepContextBuilder()
        query_result = QueryNodesResult(query_text="q", nodes=["n1"])
        candidate_result = CandidateNodesResult(nodes=["n2"])

        builder.store_result(query_result)
        builder.store_result(candidate_result)

        self.assertIs(builder.get_stored_result(QueryNodesResult), query_result)
        self.assertIs(builder.get_stored_result(CandidateNodesResult), candidate_result)

    def test_builder_replaces_older_result_of_same_type(self) -> None:
        builder = StepContextBuilder()
        old_result = QueryNodesResult(query_text="old", nodes=["n1"])
        new_result = QueryNodesResult(query_text="new", nodes=["n2"])

        builder.store_result(old_result)
        builder.store_result(new_result)

        self.assertIs(builder.get_stored_result(QueryNodesResult), new_result)

    def test_builder_clear_removes_stored_values(self) -> None:
        builder = StepContextBuilder()
        builder.store_result(QueryNodesResult(query_text="q", nodes=["n1"]))

        builder.clear_result_bank()

        self.assertFalse(builder.has_stored_result(QueryNodesResult))
        self.assertIsNone(builder.get_stored_result(QueryNodesResult))

    def test_builder_required_lookup_raises_when_missing(self) -> None:
        builder = StepContextBuilder()

        with self.assertRaisesRegex(Exception, "No stored result found"):
            builder.get_required_result(QueryNodesResult)


class PipelineFoundationTests(unittest.TestCase):
    def make_initial_context(self) -> StepContext[ExampleInitialResult]:
        return StepContext(result=ExampleInitialResult())

    def test_prepare_executes_preparation_steps_successfully(self) -> None:
        pipeline = Pipeline(
            preparation_steps=[InitialToQueryStep(), QueryToCandidatesStep()],
        )

        result = pipeline.prepare(self.make_initial_context())

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 2)
        self.assertEqual(result.total_steps, 2)
        self.assertEqual(result.final_result.nodes, ["n1", "n2", "n3"])
        self.assertIsNone(result.error_message)
        self.assertTrue(pipeline.context_builder.has_stored_result(QueryNodesResult))
        self.assertTrue(pipeline.context_builder.has_stored_result(CandidateNodesResult))

    def test_evaluate_executes_evaluation_steps_successfully(self) -> None:
        pipeline = Pipeline(
            evaluation_steps=[CandidateToAnswerStep()],
        )
        initial_context = StepContext(result=CandidateNodesResult(nodes=["n1", "n2"]))

        result = pipeline.evaluate(initial_context)

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(result.total_steps, 1)
        self.assertEqual(result.final_result.answer, "done")
        self.assertFalse(pipeline.context_builder.has_stored_result(QueryNodesResult))
        self.assertTrue(pipeline.context_builder.has_stored_result(FinalAnswerResult))

    def test_run_executes_full_pipeline(self) -> None:
        builder = StepContextBuilder()
        pipeline = Pipeline(
            preparation_steps=[InitialToQueryStep(), QueryToCandidatesStep()],
            evaluation_steps=[BankAwareAnswerStep(builder)],
            context_builder=builder,
        )

        result = pipeline.run(self.make_initial_context())

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 3)
        self.assertEqual(result.total_steps, 3)
        self.assertEqual(result.final_result.answer, "answer-for-who?")
        self.assertTrue(builder.has_stored_result(QueryNodesResult))
        self.assertTrue(builder.has_stored_result(CandidateNodesResult))
        self.assertTrue(builder.has_stored_result(FinalAnswerResult))

    def test_step_failure_stops_phase_execution(self) -> None:
        completions: list[str] = []
        pipeline = Pipeline(
            preparation_steps=[InitialToQueryStep(), FailingStep(), CandidateToAnswerStep()],
            completion_callbacks=[lambda: completions.append("finished")],
        )

        result = pipeline.prepare(self.make_initial_context())

        self.assertFalse(result.success)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(result.exception_type, "PipelineException")
        self.assertEqual(completions, ["finished"])

    def test_first_step_can_run_from_empty_initial_context(self) -> None:
        class EmptyAwareInitialStep(AbstractStep[QueryNodesResult, InitialStepResult]):
            def execute_default(self, context: StepContext[InitialStepResult]) -> QueryNodesResult:
                return QueryNodesResult(query_text="boot", nodes=["root"])

        pipeline = Pipeline(preparation_steps=[EmptyAwareInitialStep()])
        initial_context = StepContext(result=None)

        result = pipeline.prepare(initial_context)

        self.assertTrue(result.success)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(result.final_result.nodes, ["root"])

    def test_execution_metadata_is_populated_on_failure(self) -> None:
        pipeline = Pipeline(preparation_steps=[InitialToQueryStep(), FailingStep()])

        result = pipeline.prepare(self.make_initial_context())

        self.assertFalse(result.success)
        self.assertIsNotNone(result.execution_time_ms)
        self.assertGreaterEqual(result.execution_time_ms, 0)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(result.total_steps, 2)
        self.assertIsNotNone(result.error_message)
        self.assertIsNotNone(result.exception_type)

    def test_prepare_starts_with_clean_bank(self) -> None:
        pipeline = Pipeline(preparation_steps=[InitialToQueryStep()])
        pipeline.context_builder.store_result(CandidateNodesResult(nodes=["stale"]))

        result = pipeline.prepare(self.make_initial_context())

        self.assertTrue(result.success)
        self.assertFalse(pipeline.context_builder.has_stored_result(CandidateNodesResult))
        self.assertTrue(pipeline.context_builder.has_stored_result(QueryNodesResult))

    def test_evaluate_starts_with_clean_bank(self) -> None:
        pipeline = Pipeline(evaluation_steps=[CandidateToAnswerStep()])
        pipeline.context_builder.store_result(QueryNodesResult(query_text="stale", nodes=["n1"]))

        result = pipeline.evaluate(StepContext(result=CandidateNodesResult(nodes=["n2"])))

        self.assertTrue(result.success)
        self.assertFalse(pipeline.context_builder.has_stored_result(QueryNodesResult))
        self.assertTrue(pipeline.context_builder.has_stored_result(FinalAnswerResult))


if __name__ == "__main__":
    unittest.main()
