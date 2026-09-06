"""Tests for visible LLM answer-generation rate-limit retry logs."""

import unittest
from unittest.mock import patch

from pipeline.evaluation.exceptions import LlmAnswerGenerationException
from pipeline.evaluation.exceptions import InsufficientLlmCreditsException
from pipeline.evaluation.services.llm_answer_generation import (
    LangChainOpenAiAnswerGenerationService,
)


class FakeResponse:
    status_code = 429
    headers = {"retry-after": "0"}


class FakeRateLimitError(Exception):
    response = FakeResponse()


class FakeInsufficientCreditResponse:
    status_code = 402
    headers = {}


class FakeInsufficientCreditError(Exception):
    response = FakeInsufficientCreditResponse()


class FakeInsufficientCreditChatModel:
    def invoke(self, messages):
        raise FakeInsufficientCreditError("Insufficient Balance")


class FakeChatModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise FakeRateLimitError("chat rate limited")
        return "ok"


class CapturingChatOpenAI:
    captured_kwargs = None
    instance_count = 0
    invoke_count = 0
    captured_messages = None

    def __init__(self, **kwargs):
        self.__class__.captured_kwargs = kwargs
        self.__class__.instance_count += 1

    def invoke(self, messages):
        self.__class__.invoke_count += 1
        self.__class__.captured_messages = messages
        return type(
            "LangChainResponse",
            (),
            {
                "content": '{"answers":["A"],"explanation":"B"}',
                "usage_metadata": {
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )()


class AnswerOnlyCapturingChatOpenAI(CapturingChatOpenAI):
    def invoke(self, messages):
        self.__class__.invoke_count += 1
        self.__class__.captured_messages = messages
        return type(
            "LangChainResponse",
            (),
            {
                "content": '{"answers":["A"]}',
                "usage_metadata": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
            },
        )()


class InternallyBrokenChatOpenAI:
    calls = 0

    def __init__(self, **kwargs):
        self.__class__.calls += 1
        raise TypeError("internal constructor bug")


class LlmAnswerGenerationRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        CapturingChatOpenAI.captured_kwargs = None
        CapturingChatOpenAI.instance_count = 0
        CapturingChatOpenAI.invoke_count = 0
        CapturingChatOpenAI.captured_messages = None
        InternallyBrokenChatOpenAI.calls = 0

    def test_rate_limit_is_logged_and_retried(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        chat_model = FakeChatModel()

        with patch("time.sleep") as sleep, self.assertLogs(
            "pipeline.evaluation.services.llm_answer_generation",
            level="WARNING",
        ) as logs:
            result = service._invoke_with_visible_rate_limit_retries(
                chat_model=chat_model,
                messages=[],
                model_id="gpt-4.1-nano",
                prompt="question",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(chat_model.calls, 2)
        sleep.assert_called_once_with(0.0)
        self.assertTrue(
            any(
                "OpenAI rate limit hit" in message
                and "operation=llm_answer_generation" in message
                and "\033[91m" in message
                for message in logs.output
            )
        )

    def test_insufficient_credit_error_fails_without_rate_limit_retry(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        chat_model = FakeInsufficientCreditChatModel()

        with patch("time.sleep") as sleep, self.assertRaisesRegex(
            InsufficientLlmCreditsException,
            "insufficient credits",
        ):
            service._invoke_with_visible_rate_limit_retries(
                chat_model=chat_model,
                messages=[],
                model_id="deepseek-v4-flash",
                prompt="question",
            )

        sleep.assert_not_called()

    def test_insufficient_credit_marker_is_recognized_inside_wrapped_error(self) -> None:
        cause = RuntimeError(
            "Error code: 429 - {'error': {'code': 'insufficient_quota'}}"
        )
        wrapped = LlmAnswerGenerationException("request failed")
        wrapped.__cause__ = cause

        self.assertTrue(
            LangChainOpenAiAnswerGenerationService.is_insufficient_credit_error(
                wrapped
            )
        )

    def test_deepseek_chat_model_uses_deepseek_key_and_base_url(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        model = service._create_chat_model(
            chat_openai_type=CapturingChatOpenAI,
            model_id="deepseek-v4-flash",
            api_key="deepseek-key",
            base_url="https://api.deepseek.com",
        )

        self.assertIsInstance(model, CapturingChatOpenAI)
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["api_key"], "deepseek-key")
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["base_url"],
            "https://api.deepseek.com",
        )
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["timeout"], 45.0)
        self.assertNotIn("model_kwargs", CapturingChatOpenAI.captured_kwargs)

    def test_openai_chat_model_uses_timeout_and_json_response_format(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        model = service._create_chat_model(
            chat_openai_type=CapturingChatOpenAI,
            model_id="gpt-4.1-nano",
            api_key="openai-key",
        )

        self.assertIsInstance(model, CapturingChatOpenAI)
        self.assertEqual(CapturingChatOpenAI.captured_kwargs["timeout"], 45.0)
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["model_kwargs"],
            {"response_format": {"type": "json_object"}},
        )

    def test_reasoning_effort_is_passed_to_openai_chat_model(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        service._create_chat_model(
            chat_openai_type=CapturingChatOpenAI,
            model_id="gpt-5-mini",
            api_key="openai-key",
            reasoning_effort="low",
        )

        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["reasoning_effort"],
            "low",
        )

    def test_chat_model_constructor_type_error_is_not_hidden_by_fallbacks(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        with self.assertRaisesRegex(TypeError, "internal constructor bug"):
            service._create_chat_model(
                chat_openai_type=InternallyBrokenChatOpenAI,
                model_id="gpt-5-mini",
                api_key="openai-key",
            )

        self.assertEqual(InternallyBrokenChatOpenAI.calls, 1)

    def test_deepseek_missing_api_key_mentions_deepseek_env_name(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()

        with patch(
            "pipeline.evaluation.services.llm_answer_generation.DEEPSEEK_API_KEY",
            None,
        ), self.assertRaisesRegex(Exception, "DEEPSEEK_API_KEY"):
            service.generate_answer_with_explanation(
                question="q",
                reasoning_paths_text="paths",
                model_id="deepseek-v4-pro",
            )

    def test_deepseek_cost_estimation_uses_deepseek_prices(self) -> None:
        cost = LangChainOpenAiAnswerGenerationService.estimate_cost_usd(
            model_id="deepseek-v4-pro",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )

        self.assertEqual(cost, 1.305)

    def test_vezilka_uses_chat_openai_with_custom_endpoint(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        with patch(
            "pipeline.evaluation.services.llm_answer_generation.VEZILKA_API_KEY",
            "vezilka-key",
        ), patch("langchain_openai.ChatOpenAI", CapturingChatOpenAI):
            result = service.generate_answer_with_explanation(
                question="Question?",
                reasoning_paths_text="<A, relation, B>",
                model_id="qwen3-4b-custom",
                provider_id="vezilka",
                reasoning_effort="none",
            )

        self.assertEqual(result["answers"], ["A"])
        self.assertEqual(result["total_tokens"], 20)
        self.assertEqual(result["estimated_cost_usd"], 0.0)
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["base_url"],
            "https://vllm.finki.ukim.mk/v1",
        )
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["model"],
            "qwen3-4b-custom",
        )
        self.assertEqual(
            CapturingChatOpenAI.captured_kwargs["reasoning_effort"],
            "none",
        )
        self.assertFalse(CapturingChatOpenAI.captured_kwargs["streaming"])
        self.assertNotIn("max_completion_tokens", CapturingChatOpenAI.captured_kwargs)
        self.assertFalse(CapturingChatOpenAI.captured_kwargs["use_responses_api"])
        self.assertEqual(
            CapturingChatOpenAI.captured_messages[0].type,
            "system",
        )

    def test_vezilka_reuses_chat_model_and_omits_optional_reasoning(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        with patch(
            "pipeline.evaluation.services.llm_answer_generation.VEZILKA_API_KEY",
            "vezilka-key",
        ), patch("langchain_openai.ChatOpenAI", CapturingChatOpenAI):
            service.generate_answer_with_explanation(
                question="Question one?",
                reasoning_paths_text="paths",
                model_id="qwen3.8-27b",
                provider_id="vezilka",
            )
            service.generate_answer_with_explanation(
                question="Question two?",
                reasoning_paths_text="paths",
                model_id="qwen3.8-27b",
                provider_id="vezilka",
            )

        self.assertEqual(CapturingChatOpenAI.instance_count, 1)
        self.assertEqual(CapturingChatOpenAI.invoke_count, 2)
        self.assertNotIn("reasoning_effort", CapturingChatOpenAI.captured_kwargs)
        self.assertFalse(CapturingChatOpenAI.captured_kwargs["streaming"])

    def test_answer_only_generation_omits_explanation_from_prompt_and_result(self) -> None:
        service = LangChainOpenAiAnswerGenerationService()
        with patch(
            "pipeline.evaluation.services.llm_answer_generation.OPENAI_API_KEY",
            "openai-key",
        ), patch("langchain_openai.ChatOpenAI", AnswerOnlyCapturingChatOpenAI):
            result = service.generate_answer_with_explanation(
                question="Question?",
                reasoning_paths_text="<A, relation, B>",
                model_id="gpt-4.1-nano",
                generate_explanation=False,
            )

        self.assertEqual(result["answers"], ["A"])
        self.assertEqual(result["explanation"], "")
        self.assertNotIn('"explanation"', result["prompt"])
        self.assertIn('"answers": ["complete entity name"]', result["prompt"])
        self.assertIn(
            "Do not generate an explanation",
            AnswerOnlyCapturingChatOpenAI.captured_messages[0].content,
        )

    def test_atomic_answer_array_preserves_commas_inside_entity_names(self) -> None:
        result = LangChainOpenAiAnswerGenerationService.parse_json_response(
            '{"answers":["Washington, D.C.","Paris, Texas"],'
            '"explanation":"supported"}'
        )

        self.assertEqual(result["answers"], ["Washington, D.C.", "Paris, Texas"])

    def test_singular_answer_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(LlmAnswerGenerationException, "answers"):
            LangChainOpenAiAnswerGenerationService.parse_json_response(
                '{"answer":"Alpha, Beta","explanation":"supported"}'
            )


if __name__ == "__main__":
    unittest.main()
