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

"""CLI utility functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import click

from strands_env.core.models import ModelFactory, bedrock_model_factory, sglang_model_factory

from .config import EnvConfig, ModelConfig

if TYPE_CHECKING:
    from strands_env.eval import AsyncEnvFactory

#: Type for the create_env_factory function exported by hook files.
EnvFactoryCreator = Callable[[ModelFactory, EnvConfig], "AsyncEnvFactory"]


def build_model_factory(config: ModelConfig, max_concurrency: int) -> ModelFactory:
    """Build a ModelFactory from ModelConfig.

    Args:
        config: Model configuration.
        max_concurrency: Max concurrent connections (for SGLang client pooling).

    Returns:
        ModelFactory callable.
    """
    sampling = config.sampling.to_dict()

    if config.backend == "sglang":
        from strands_env.utils.sglang import (
            check_server_health,
            get_cached_client,
            get_cached_tokenizer,
            get_model_id,
        )

        # Check server health before proceeding
        try:
            check_server_health(config.base_url)
        except ConnectionError as e:
            raise click.ClickException(str(e))

        client = get_cached_client(config.base_url, max_concurrency)
        model_id = config.model_id or get_model_id(config.base_url)
        tokenizer_path = config.tokenizer_path or model_id
        tokenizer = get_cached_tokenizer(tokenizer_path)
        return sglang_model_factory(client=client, model_id=model_id, tokenizer=tokenizer, sampling_params=sampling)

    elif config.backend == "bedrock":
        from strands_env.utils.aws import get_assumed_role_session, get_boto3_session

        if not config.model_id:
            raise click.ClickException("--model-id is required for Bedrock backend")
        if config.role_arn:
            session = get_assumed_role_session(config.role_arn, config.region)
        else:
            session = get_boto3_session(config.region, config.profile_name)
        return bedrock_model_factory(session, config.model_id, sampling)

    else:
        raise click.ClickException(f"Unknown backend: {config.backend}")


def load_env_hook(env_path: Path) -> EnvFactoryCreator:
    """Load environment hook file and return create_env_factory function.

    The hook file must export a `create_env_factory(model_factory, env_config)` function.

    Args:
        env_path: Path to the Python hook file.

    Returns:
        The create_env_factory function from the hook file.

    Raises:
        click.ClickException: If the file cannot be loaded or doesn't export the function.
    """
    spec = importlib.util.spec_from_file_location("env_hook", env_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load hook file: {env_path}")

    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    if not hasattr(hook, "create_env_factory"):
        raise click.ClickException(
            "Hook file must export 'create_env_factory(model_factory, env_config)' function.\n"
            "Example:\n"
            "  def create_env_factory(model_factory, env_config):\n"
            "      async def env_factory(action):\n"
            "          return MyEnv(\n"
            "              model_factory=model_factory,\n"
            "              system_prompt=env_config.system_prompt,\n"
            "              max_tool_iterations=env_config.max_tool_iterations,\n"
            "          )\n"
            "      return env_factory"
        )

    return hook.create_env_factory
