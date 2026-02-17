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

"""Integration tests for Environment with a real SGLang model and mock tools.

Requires a running SGLang server (default: http://localhost:30000).
"""

from strands_tools import calculator

from strands_env.core.environment import Environment
from strands_env.core.types import Action, RewardResult, StepResult, TaskContext, TerminationReason

# ---------------------------------------------------------------------------
# Math Environment
# ---------------------------------------------------------------------------


class MathEnvironment(Environment):
    """Simple math environment that provides a calculator tool."""

    def get_tools(self) -> list:
        return [calculator]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMathEnvironment:
    async def test_step_completes(self, model_factory):
        """Agent can solve a simple math problem and terminate normally."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool to solve problems. Be concise.",
        )
        action = Action(message="What is 17 * 23?")
        result = await env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.messages
        assert result.observation.metrics["message_count"] > 0

    async def test_step_produces_token_observation(self, model_factory):
        """SGLang model produces token-level observations for TITO training."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool. Be concise.",
        )
        action = Action(message="What is 5 + 3?")
        result = await env.step(action)

        assert result.observation.tokens is not None
        assert len(result.observation.tokens.token_ids) > 0
        assert result.observation.tokens.prompt_length > 0
        assert len(result.observation.tokens.rollout_token_ids) > 0

    async def test_step_with_conversation_history(self, model_factory):
        """Agent uses conversation history for multi-turn interaction."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool. Be concise.",
        )
        # First turn
        action1 = Action(message="What is 10 + 5?")
        result1 = await env.step(action1)
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        # Second turn with history from first
        all_messages = result1.observation.messages
        action2 = Action(
            message="Now multiply that result by 3.",
            task_context=TaskContext(conversation_history=all_messages),
        )
        result2 = await env.step(action2)
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_step_metrics(self, model_factory):
        """Step produces expected metric keys."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool. Be concise.",
        )
        action = Action(message="What is 100 / 4?")
        result = await env.step(action)

        metrics = result.observation.metrics
        assert "message_count" in metrics
        assert "tool_iters" in metrics
        assert "model_calls" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics

    async def test_tool_iteration_limit(self, model_factory):
        """Environment respects max_tool_iters."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt=(
                "You are a math assistant. Use the calculator tool for every single step. "
                "Break every problem into many small steps, each requiring a separate calculator call."
            ),
            max_tool_iters=1,
        )
        action = Action(message="Compute 1+1, then 2+2, then 3+3, then 4+4, then 5+5 one at a time.")
        result = await env.step(action)

        assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
        assert result.observation.metrics["tool_iters"] <= 1

    async def test_final_response(self, model_factory):
        """Observation provides final assistant response text."""
        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool. Be concise.",
        )
        action = Action(message="What is 7 * 8?")
        result = await env.step(action)

        response = result.observation.final_response
        # The agent should produce some text response (may or may not contain "56")
        assert response is not None
        assert len(response) > 0

    async def test_reward_fn_called(self, model_factory):
        """Reward function is invoked and result attached to StepResult."""

        class ExactMatchReward:
            async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
                response = step_result.observation.final_response or ""
                expected = str(action.task_context.ground_truth)
                match = expected in response
                return RewardResult(reward=1.0 if match else 0.0, info={"exact_match": match})

        env = MathEnvironment(
            model_factory=model_factory,
            system_prompt="You are a math assistant. Use the calculator tool. Be concise. Give the final number.",
            reward_fn=ExactMatchReward(),
        )
        action = Action(
            message="What is 6 * 7?",
            task_context=TaskContext(ground_truth=42),
        )
        result = await env.step(action)

        assert result.reward is not None
        assert isinstance(result.reward.reward, float)
        assert "exact_match" in result.reward.info
