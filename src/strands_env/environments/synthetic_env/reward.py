"""Reward function for synthetic environments using AWM verifier code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

if TYPE_CHECKING:
    from .env import SyntheticEnv

logger = logging.getLogger(__name__)


class SyntheticEnvRewardFunction(RewardFunction):
    """Compute reward by running the AWM verifier code against initial and final DB states.

    The verifier compares the SQLite database before and after the agent's actions
    to determine whether the task was completed successfully.
    """

    def __init__(self, env: SyntheticEnv) -> None:
        self._env = env

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        try:
            verifier_code = self._env.data_loader.get_verifier(self._env.config.scenario, self._env.config.task_idx)

            namespace: dict = {}
            exec(verifier_code, namespace)  # noqa: S102
            verify_fn = namespace.get("verify_task_completion")

            if verify_fn is None:
                logger.error("verify_task_completion function not found in verifier code")
                return RewardResult(reward=0.0, info={"error": "verify_task_completion not found"})

            result = verify_fn(
                initial_db_path=str(self._env.initial_db_path),
                final_db_path=str(self._env.db_path),
                final_answer=step_result.observation.final_response,
            )

            reward = 1.0 if result.get("result") == "complete" else 0.0
            return RewardResult(reward=reward, info=result)
        except Exception as e:
            logger.exception("Verification failed: %s", e)
            return RewardResult(reward=0.0, info={"error": str(e)})
