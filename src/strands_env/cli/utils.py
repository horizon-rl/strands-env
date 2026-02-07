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
    from strands_env.eval import AsyncEnvFactory, Evaluator

#: Type for the create_env_factory function exported by hook files.
EnvFactoryCreator = Callable[[ModelFactory, EnvConfig], "AsyncEnvFactory"]

#: Type for evaluator class.
EvaluatorClass = type["Evaluator"]


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

        # Resolve and backfill model_id/tokenizer_path for reproducibility
        if not config.model_id:
            config.model_id = get_model_id(config.base_url)
        if not config.tokenizer_path:
            config.tokenizer_path = config.model_id

        tokenizer = get_cached_tokenizer(config.tokenizer_path)
        return sglang_model_factory(
            client=client, model_id=config.model_id, tokenizer=tokenizer, sampling_params=sampling
        )

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


def load_evaluator_hook(evaluator_path: Path) -> EvaluatorClass:
    """Load evaluator hook file and return the Evaluator class.

    The hook file must export an `EvaluatorClass` that extends `Evaluator`.

    Args:
        evaluator_path: Path to the Python hook file.

    Returns:
        The Evaluator subclass from the hook file.

    Raises:
        click.ClickException: If the file cannot be loaded or doesn't export the class.
    """
    from strands_env.eval import Evaluator

    spec = importlib.util.spec_from_file_location("evaluator_hook", evaluator_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load evaluator hook file: {evaluator_path}")

    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    if not hasattr(hook, "EvaluatorClass"):
        raise click.ClickException(
            "Evaluator hook file must export 'EvaluatorClass' (an Evaluator subclass).\n"
            "Example:\n"
            "  from strands_env.eval import Evaluator\n"
            "\n"
            "  class MyEvaluator(Evaluator):\n"
            "      benchmark_name = 'my-benchmark'\n"
            "\n"
            "      def load_dataset(self):\n"
            "          ...\n"
            "\n"
            "  EvaluatorClass = MyEvaluator"
        )

    evaluator_cls = hook.EvaluatorClass
    if not isinstance(evaluator_cls, type) or not issubclass(evaluator_cls, Evaluator):
        raise click.ClickException("EvaluatorClass must be a subclass of Evaluator")

    return evaluator_cls
