"""Environment hook file for the synthetic benchmark.

Usage:
    strands-env eval run synthetic --env src/strands_env/environments/synthetic_env/env_hook.py --data-dir /path/to/AWM-1K
"""

from __future__ import annotations

from pathlib import Path

from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.synthetic_env import SyntheticEnv, SyntheticEnvConfig


def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create an async env_factory for SyntheticEnv.

    The returned factory extracts ``scenario`` and ``task_idx`` from
    the action's TaskContext extra fields (set by SyntheticEvaluator).
    """

    async def env_factory(action):
        ctx = action.task_context
        config = SyntheticEnvConfig(
            scenario=ctx.scenario,
            task_idx=ctx.task_idx,
            data_dir=Path(ctx.data_dir),
        )
        return SyntheticEnv(
            model_factory=model_factory,
            config=config,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
        )

    return env_factory
