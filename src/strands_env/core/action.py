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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from strands.types.content import Message, Messages


class TaskContext(BaseModel):
    """Task context with ground truth, conversation history, and optional extra fields.

    Use extra kwargs for task-specific data that reward functions need.

    Example:
        >>> # Single-turn
        >>> TaskContext(ground_truth=["4", "four"])
        >>> # Multi-turn with history
        >>> TaskContext(
        ...     ground_truth="6",
        ...     conversation_history=[
        ...         {"role": "user", "content": [{"text": "What is 2+2?"}]},
        ...         {"role": "assistant", "content": [{"text": "4"}]},
        ...     ]
        ... )
    """

    model_config = ConfigDict(extra="allow")

    ground_truth: Any = None
    conversation_history: Messages = Field(default_factory=list)


class Action(BaseModel):
    """Action represents a task: message to send + task context for reward."""

    message: str | Message = Field(..., description="The message/prompt to send to the agent.")
    task_context: TaskContext = Field(default_factory=TaskContext)
