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

"""Base Environment class for strands-env.

This module defines the Environment base class that implements the template method
pattern for agent-environment interaction.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel
from strands import Agent
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import Messages
from strands.types.exceptions import EventLoopException, MaxTokensReachedException
from strands_sglang import MaxToolIterationsReachedError, TokenManager, ToolIterationLimiter

from .action import Action
from .model import ModelConfig
from .observation import Observation, TokenObservation
from .reward import RewardResult
from .utils import render_prompt

if TYPE_CHECKING:
    from .reward import RewardFunction

logger = logging.getLogger(__name__)


class TerminationReason(str, Enum):
    """Reasons why an episode might end."""

    NOT_TERMINATED = "not_terminated"
    TASK_COMPLETE = "task_complete"
    MAX_TOOL_ITERATIONS_REACHED = "max_tool_iterations_reached"
    MAX_TOKENS_REACHED = "max_tokens_reached"
    AGENT_ERROR = "agent_error"

    @property
    def is_terminated(self) -> bool:
        """Episode ended naturally (task complete or agent error)."""
        return self in {TerminationReason.TASK_COMPLETE, TerminationReason.AGENT_ERROR}

    @property
    def is_truncated(self) -> bool:
        """Episode ended due to limits (tools/tokens)."""
        return self in {TerminationReason.MAX_TOOL_ITERATIONS_REACHED, TerminationReason.MAX_TOKENS_REACHED}

    @property
    def is_done(self) -> bool:
        """Episode has ended (either terminal or truncation)."""
        return self != TerminationReason.NOT_TERMINATED


class StepResult(BaseModel):
    """Result of a single environment step."""

    observation: Observation
    reward: RewardResult | None = None
    termination_reason: TerminationReason = TerminationReason.NOT_TERMINATED


class Environment:
    """Base class for strands-env environments with streamlined agent loop.

    A minimal tool-using environment implementation should usually override:
    - get_tools(): Provide environment-specific tools

    Important methods to override when needed:
    - reset(): Reset the environment state for a new episode
    - cleanup(): Clean up resources (e.g., sandbox containers)
    """

    default_system_prompt_path: ClassVar[Path | None] = None

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iterations: int = 10,
        verbose: bool = False,
    ):
        self.model_config = model_config
        self.reward_fn = reward_fn
        self.max_tool_iterations = max_tool_iterations
        self.verbose = verbose

        # Load system prompt from argument or default path
        path = self.default_system_prompt_path
        self.system_prompt = system_prompt or (render_prompt(path) if path and path.exists() else None)

    # -------------------------------------------------------------------------
    # Core API: reset, step, cleanup
    # -------------------------------------------------------------------------

    async def reset(self) -> None:
        """Reset the environment for a new episode.

        Override in subclasses to add environment-specific initialization.
        """
        pass

    async def step(self, action: Action) -> StepResult:
        """Execute one step in the environment.

        Args:
            action: The action containing message and task_context.
                   task_context.conversation_history is used for multi-turn.

        Returns:
            StepResult with observation, reward, and termination status.
        """
        # Create ephemeral agent and invoke
        conversation_history = action.task_context.conversation_history
        tool_limiter = ToolIterationLimiter(self.max_tool_iterations)
        agent = self.create_agent(conversation_history, tool_limiter)
        error = None
        try:
            await agent.invoke_async(action.message)
        except Exception as e:
            error = e
        termination_reason = self.check_termination(error)

        # Build observation, metrics, and step result
        step_messages = list(agent.messages)[len(conversation_history) :]
        token_obs = TokenObservation.from_token_manager(agent.model.token_manager)
        tool_parse_errors = getattr(agent.model, "tool_parse_errors", None)
        metrics = {
            "message_count": len(step_messages),
            "tool_iters": tool_limiter.iteration_count,
            **self.compute_metrics(agent.event_loop_metrics, tool_parse_errors=tool_parse_errors),
        }
        observation = Observation(messages=step_messages, tokens=token_obs, metrics=metrics)
        step_result = StepResult(observation=observation, termination_reason=termination_reason)
        step_result.reward = (
            (await self.reward_fn.compute(action=action, step_result=step_result)) if self.reward_fn else None
        )

        return step_result

    async def cleanup(self) -> None:
        """Clean up resources. Override in subclasses."""
        pass

    # -------------------------------------------------------------------------
    # Overridable: Tools, hooks, termination, metrics
    # -------------------------------------------------------------------------

    def get_tools(self) -> list:
        """Return tools for the agent. Override in subclasses."""
        return []

    def get_hooks(self) -> list:
        """Return hooks for the agent. Override and call super() to add more."""
        return []

    def create_agent(self, conversation_history: Messages, tool_limiter: ToolIterationLimiter) -> Agent:
        """Create an ephemeral strands Agent with fresh token manager."""
        model = self.model_config.create_model()
        model.token_manager = TokenManager()
        return Agent(
            model=model,
            messages=list(conversation_history),
            tools=self.get_tools(),
            system_prompt=self.system_prompt,
            hooks=self.get_hooks() + [tool_limiter],
            callback_handler=PrintingCallbackHandler() if self.verbose else None,
        )

    def check_termination(self, error: Exception | None = None) -> TerminationReason:
        """Check for termination conditions. Override for custom termination logic."""

        # Return task complete if there's no error
        if error is None:
            return TerminationReason.TASK_COMPLETE

        # Unwrap EventLoopException to get the underlying cause
        cause = error.__cause__ if isinstance(error, EventLoopException) else error

        match cause:
            case MaxToolIterationsReachedError():
                reason = TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            case MaxTokensReachedException():
                reason = TerminationReason.MAX_TOKENS_REACHED
            case _:
                reason = TerminationReason.AGENT_ERROR

        logger.warning(f"Step terminated: {reason.value} - {cause}")
        return reason

    def compute_metrics(
        self,
        event_loop_metrics: EventLoopMetrics,
        tool_parse_errors: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Compute metrics for the current step. Override to add custom metrics.

        Args:
            event_loop_metrics: Metrics from the strands event loop.
            tool_parse_errors: Parse errors per tool name from `SGLangModel` (training only).
        """
        usage = event_loop_metrics.accumulated_usage
        metrics_data = event_loop_metrics.accumulated_metrics

        # Per-tool metrics (merge parse errors if available)
        per_tool_metrics = {
            name: {
                "calls": tm.call_count,
                "successes": tm.success_count,
                "errors": tm.error_count,
                "parse_errors": (tool_parse_errors or {}).get(name, 0),  # only valid for SGLangModel
                "latency_s": round(tm.total_time, 4),
            }
            for name, tm in event_loop_metrics.tool_metrics.items()
        } or None

        return {
            "model_calls": event_loop_metrics.cycle_count or None,
            "model_latency_s": round(metrics_data["latencyMs"] / 1000.0, 4) if metrics_data.get("latencyMs") else None,
            "input_tokens": usage.get("inputTokens") or None,
            "output_tokens": usage.get("outputTokens") or None,
            "total_tokens": usage.get("totalTokens") or None,
            "per_tool_metrics": per_tool_metrics,
        }
