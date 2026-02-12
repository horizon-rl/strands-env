# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model factory functions for supported backends.

Each function returns a `ModelFactory` (zero-arg callable that creates a fresh
`Model` instance) for use with `Environment`::

    from strands_sglang import SGLangClient
    from strands_env.core.models import sglang_model_factory

    client = SGLangClient("http://localhost:30000")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    env = Environment(
        model_factory=sglang_model_factory(tokenizer=tokenizer, client=client),
    )

Users can easily create their own model factories by implementing the `ModelFactory` type.

Example:
    >>> from strands_env.core.models import ModelFactory
    >>> def my_model_factory() -> ModelFactory:
    >>>     return lambda: MyModel()
    >>>
    >>> env = Environment(model_factory=my_model_factory())
    >>>
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import boto3
import botocore.config
import httpx
from strands.models import Model
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel
from strands_sglang import SGLangClient, SGLangModel
from strands_sglang.tool_parsers import HermesToolParser, ToolParser
from transformers import PreTrainedTokenizerBase

#: Factory that produces a fresh `Model` per step (for concurrent step isolation).
ModelFactory = Callable[[], Model]

DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 16384, "temperature": 1.0, "top_p": 1.0}

# ---------------------------------------------------------------------------
# SGLang Model
# ---------------------------------------------------------------------------


def sglang_model_factory(
    *,
    client: SGLangClient,
    tokenizer: PreTrainedTokenizerBase,
    tool_parser: ToolParser | None = None,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    return_logprob: bool = True,
    enable_thinking: bool | None = None,
    return_routed_experts: bool = False,
) -> ModelFactory:
    """Return a factory that creates `SGLangModel` instances.

    Args:
        client: `SGLangClient` for HTTP communication with the SGLang server.
        tokenizer: HuggingFace tokenizer for chat template and tokenization.
        tool_parser: Tool parser for extracting tool calls from model output. Defaults to `HermesToolParser`.
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
        return_logprob: Whether to return logprobs for each token.
        enable_thinking: Enable thinking mode for Qwen3 hybrid models.
        return_routed_experts: Record MoE routing decisions for routing replay.
    """
    if tool_parser is None:
        tool_parser = HermesToolParser()

    return lambda: SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        sampling_params=sampling_params,
        return_logprob=return_logprob,
        enable_thinking=enable_thinking,
        return_routed_experts=return_routed_experts,
    )


# ---------------------------------------------------------------------------
# Bedrock Model
# ---------------------------------------------------------------------------


DEFAULT_BOTO_CLIENT_CONFIG = botocore.config.Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    max_pool_connections=100,
    connect_timeout=5.0,
    read_timeout=600.0,
)


def bedrock_model_factory(
    *,
    model_id: str,
    boto_session: boto3.Session,
    boto_client_config: botocore.config.Config = DEFAULT_BOTO_CLIENT_CONFIG,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
) -> ModelFactory:
    """Return a factory that creates `BedrockModel` instances.

    Args:
        model_id: Bedrock model ID (e.g. "us.anthropic.claude-sonnet-4-20250514-v1:0").
        boto_session: Boto3 session for AWS credentials.
        boto_client_config: Botocore client configuration.
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
    """
    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    return lambda: BedrockModel(
        model_id=model_id,
        boto_session=boto_session,
        boto_client_config=boto_client_config,
        streaming=False,
        **sampling_params,
    )


# ---------------------------------------------------------------------------
# OpenAI Model
# ---------------------------------------------------------------------------

# OpenAI client arguments for SGLang server
DEFAULT_OPENAI_CLIENT_ARGS = {
    "api_key": "EMPTY",
    "base_url": "http://localhost:30000/v1",
    "timeout": httpx.Timeout(timeout=600.0, connect=5.0),
    "max_retries": 5,
}


def openai_model_factory(
    *,
    model_id: str,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    client_args: dict[str, Any] = DEFAULT_OPENAI_CLIENT_ARGS,
) -> ModelFactory:
    """Return a factory that creates `OpenAIModel` instances.

    Args:
        model_id: OpenAI model ID (e.g. "gpt-4o").
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
        client_args: Arguments for the OpenAI client (e.g. `{"api_key": "...", "base_url": "..."}`).
    """
    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    return lambda: OpenAIModel(
        model_id=model_id,
        params=sampling_params,
        client_args=client_args,
    )
