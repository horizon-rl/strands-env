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

"""Reward functions for strands-env environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .action import Action
    from .environment import StepResult


class RewardResult(BaseModel):
    """Result of reward computation."""

    reward: float = Field(..., description="The computed reward value.")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional reward computation details.")


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Compute reward for the current step.

        Args:
            action: The action with message and task_context (contains ground_truth).
            step_result: Contains observation and termination_reason.
        """
        pass
