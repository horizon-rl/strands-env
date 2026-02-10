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

"""Code sandbox environment using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.tools import CodeInterpreterToolkit
from strands_env.utils.aws import get_session

if TYPE_CHECKING:
    import boto3

    from strands_env.core.types import ModelFactory, RewardFunction


class CodeMode(str, Enum):
    """Tool modes for CodeSandboxEnv."""

    CODE = "code"
    """Only `execute_code` tool (Python execution)."""

    TERMINAL = "terminal"
    """Only `execute_command` tool (shell commands)."""

    CODE_AND_TERMINAL = "code_and_terminal"
    """Both `execute_code` and `execute_command` tools."""


class CodeSandboxEnv(Environment):
    """Code sandbox environment using AWS Bedrock AgentCore Code Interpreter.

    Provides `execute_code` (Python) and/or `execute_command` (shell) tools
    depending on the configured `CodeMode`.

    Example:
        from strands_env.environments.code_sandbox import CodeSandboxEnv, CodeMode
        from strands_env.utils.aws import get_session

        session = get_session(region="us-east-1")
        env = CodeSandboxEnv(
            boto3_session=session,
            model_factory=model_factory,
            mode=CodeMode.CODE,  # Only Python execution
        )

        result = await env.step(action)
        await env.cleanup()  # Clean up code interpreter session
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iterations: int = 10,
        verbose: bool = False,
        boto3_session: boto3.Session | None = None,
        mode: CodeMode = CodeMode.CODE,
    ):
        """Initialize the code sandbox environment.

        Args:
            boto3_session: boto3 session for AWS credentials.
            model_factory: Factory function that creates a fresh Model per step.
            system_prompt: Optional system prompt override.
            reward_fn: Optional reward function to compute rewards.
            max_tool_iterations: Maximum tool iterations per step.
            verbose: Whether to print verbose output.
            mode: Tool mode - CODE, TERMINAL, or CODE_AND_TERMINAL.
        """
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=system_prompt,
            max_tool_iterations=max_tool_iterations,
            verbose=verbose,
        )
        self.mode = mode
        self._toolkit = CodeInterpreterToolkit(
            boto3_session=boto3_session or get_session(), session_name="strands-env-code-sandbox"
        )

    @override
    def get_tools(self):
        """Return tools based on configured mode."""
        tool_map = {
            CodeMode.CODE: [self._toolkit.execute_code],
            CodeMode.TERMINAL: [self._toolkit.execute_command],
            CodeMode.CODE_AND_TERMINAL: [self._toolkit.execute_code, self._toolkit.execute_command],
        }
        return tool_map[self.mode]

    @override
    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        self._toolkit.cleanup()
