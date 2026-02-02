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

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from strands.types.content import Messages

from .utils import extract_text_content

if TYPE_CHECKING:
    from strands_sglang import TokenManager


class TokenObservation(BaseModel):
    """Token-level observation for Token-in/Token-out training.

    Stores token IDs, log probabilities, and loss mask for a single step.
    """

    token_ids: list[int] = Field(default_factory=list, description="All token IDs for this step.")
    prompt_length: int = Field(default=0, description="Length of initial prompt (segment[0]).")
    loss_mask: list[int] = Field(
        default_factory=list,
        description="Loss mask for rollout tokens only (after prompt).",
    )
    logprobs: list[float | None] = Field(
        default_factory=list,
        description="Log probabilities for rollout tokens only.",
    )

    @property
    def rollout_token_ids(self) -> list[int]:
        """Tokens added during rollout (everything after initial prompt)."""
        return self.token_ids[self.prompt_length :]

    @property
    def rollout_logprobs(self) -> list[float | None]:
        return self.logprobs[self.prompt_length :]

    @property
    def rollout_loss_mask(self) -> list[int]:
        return self.loss_mask[self.prompt_length :]

    @property
    def initial_prompt_token_ids(self) -> list[int]:
        """Initial prompt tokens:
        - system + tools + first user message in single-turn
        - full conversation history in multi-turn
        """
        return self.token_ids[: self.prompt_length]

    @classmethod
    def from_token_manager(cls, token_manager: TokenManager) -> TokenObservation | None:
        """Create from a TokenManager instance.

        Extracts tokens in slime-compatible format:
        - token_ids: tm.token_ids
        - prompt_length: len(tm.initial_prompt)
        - loss_mask: tm.loss_mask
        - logprobs: tm.logprobs
        """
        if len(token_manager) == 0:
            return None
        return cls(
            token_ids=token_manager.token_ids,
            prompt_length=len(token_manager.initial_prompt),
            loss_mask=token_manager.loss_mask,
            logprobs=token_manager.logprobs,
        )


class Observation(BaseModel):
    """Observation after a step: messages, tokens, and metrics."""

    messages: Messages = Field(
        default_factory=list,
        description="Messages generated during this step (strands format).",
    )
    tokens: TokenObservation | None = Field(
        default=None,
        description="Token-level data for TITO training.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Customizable metrics computed by the environment.",
    )

    def get_final_response(self) -> str | None:
        """Get text content from the last assistant message.

        Handles both strands formats (SGLang/OpenAI and Bedrock).

        Returns:
            Text content from the last assistant message, or None if not found.
        """
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return None
        return extract_text_content(self.messages[-1])
