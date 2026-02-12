"""Unit tests for model factory functions."""

from unittest.mock import MagicMock, patch

from strands_sglang import SGLangClient, SGLangModel

from strands_env.core.models import (
    DEFAULT_SAMPLING_PARAMS,
    bedrock_model_factory,
    openai_model_factory,
    sglang_model_factory,
)

# ---------------------------------------------------------------------------
# sglang_model_factory
# ---------------------------------------------------------------------------


class TestSGLangModelFactory:
    def test_returns_callable(self):
        factory = sglang_model_factory(
            tokenizer=MagicMock(),
            client=MagicMock(spec=SGLangClient),
        )
        assert callable(factory)

    def test_creates_sglang_model(self):
        tokenizer = MagicMock()
        client = MagicMock(spec=SGLangClient)
        factory = sglang_model_factory(
            tokenizer=tokenizer,
            client=client,
            sampling_params={"max_new_tokens": 1024},
            enable_thinking=True,
        )
        model = factory()
        assert isinstance(model, SGLangModel)
        assert model.tokenizer is tokenizer
        assert model.client is client

    def test_each_call_creates_new_instance(self):
        factory = sglang_model_factory(
            tokenizer=MagicMock(),
            client=MagicMock(spec=SGLangClient),
        )
        model1 = factory()
        model2 = factory()
        assert model1 is not model2

    def test_return_routed_experts_passed_through(self):
        factory = sglang_model_factory(
            tokenizer=MagicMock(),
            client=MagicMock(spec=SGLangClient),
            return_routed_experts=True,
        )
        model = factory()
        assert isinstance(model, SGLangModel)
        assert model.get_config()["return_routed_experts"] is True

    def test_return_routed_experts_default_false(self):
        factory = sglang_model_factory(
            tokenizer=MagicMock(),
            client=MagicMock(spec=SGLangClient),
        )
        model = factory()
        assert model.get_config().get("return_routed_experts", False) is False


# ---------------------------------------------------------------------------
# bedrock_model_factory
# ---------------------------------------------------------------------------


class TestBedrockModelFactory:
    @patch("strands_env.core.models.BedrockModel")
    def test_returns_callable(self, mock_bedrock_cls):
        import boto3

        factory = bedrock_model_factory(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            boto_session=MagicMock(spec=boto3.Session),
        )
        assert callable(factory)

    @patch("strands_env.core.models.BedrockModel")
    def test_remaps_max_new_tokens(self, mock_bedrock_cls):
        import boto3

        factory = bedrock_model_factory(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            boto_session=MagicMock(spec=boto3.Session),
            sampling_params={"max_new_tokens": 2048, "temperature": 0.7},
        )
        factory()

        call_kwargs = mock_bedrock_cls.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_new_tokens" not in call_kwargs
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.7

    @patch("strands_env.core.models.BedrockModel")
    def test_does_not_mutate_default_params(self, mock_bedrock_cls):
        import boto3

        original = dict(DEFAULT_SAMPLING_PARAMS)
        bedrock_model_factory(
            model_id="test",
            boto_session=MagicMock(spec=boto3.Session),
        )
        assert DEFAULT_SAMPLING_PARAMS == original


# ---------------------------------------------------------------------------
# openai_model_factory
# ---------------------------------------------------------------------------


class TestOpenAIModelFactory:
    @patch("strands_env.core.models.OpenAIModel")
    def test_returns_callable(self, mock_openai_cls):
        factory = openai_model_factory(model_id="gpt-4o")
        assert callable(factory)

    @patch("strands_env.core.models.OpenAIModel")
    def test_remaps_max_new_tokens(self, mock_openai_cls):
        factory = openai_model_factory(
            model_id="gpt-4o",
            sampling_params={"max_new_tokens": 4096, "temperature": 0.5},
        )
        factory()

        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["params"]["max_tokens"] == 4096
        assert "max_new_tokens" not in call_kwargs["params"]

    @patch("strands_env.core.models.OpenAIModel")
    def test_does_not_mutate_default_params(self, mock_openai_cls):
        original = dict(DEFAULT_SAMPLING_PARAMS)
        openai_model_factory(model_id="gpt-4o")
        assert DEFAULT_SAMPLING_PARAMS == original
