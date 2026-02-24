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

"""Reward function for Agent World Model environments using verifier code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

if TYPE_CHECKING:
    from .env import AgentWorldModelEnv

logger = logging.getLogger(__name__)


class AgentWorldModelRewardFunction(RewardFunction):
    """Compute reward by running the Agent World Model verifier code against initial and final DB states.

    The verifier compares the SQLite database before and after the agent's actions
    to determine whether the task was completed successfully.
    """

    def __init__(self, env: AgentWorldModelEnv) -> None:
        self._env = env

    @override
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
