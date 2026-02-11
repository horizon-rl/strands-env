"""Terminal-Bench environment using Harbor's DockerEnvironment for container management and test execution."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from strands import tool

from strands_env.core import Environment, ModelFactory
from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchConfig:
    """Configuration for task-dependent arguments in TerminalBenchEnv.

    Attributes:
        task_id: Unique identifier for the task.
        task_dir: Path to task directory containing Dockerfile, tests/, environment/.
        trial_path: Path to trial output directory for storing results.
        env_config: Harbor EnvironmentConfig for Docker setup.
        test_timeout_sec: Timeout in seconds for test execution.
    """

    task_id: str
    task_dir: Path
    trial_path: Path
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    test_timeout_sec: int = 1200

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "task_id": self.task_id,
            "task_dir": str(self.task_dir),
            "trial_path": str(self.trial_path),
            "env_config": self.env_config.model_dump() if self.env_config else None,
            "test_timeout_sec": self.test_timeout_sec,
        }


class TerminalBenchEnv(Environment):
    """Docker-based environment for terminal-bench tasks."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        config: TerminalBenchConfig,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iterations: int = 25,
        verbose: bool = False,
    ):
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=None,
            max_tool_iterations=max_tool_iterations,
            verbose=verbose,
        )

        self.config = config
        self.task_paths = TaskPaths(config.task_dir)
        self.trial_paths = TrialPaths(trial_dir=config.trial_path)
        self._env: BaseEnvironment | None = None
        self._execute_command = None

        # Use provided reward_fn or create default TBRewardFunction
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            env = self

            class TBRewardFunction(RewardFunction):
                """Execute test scripts and compute binary reward (0 or 1)."""

                async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
                    try:
                        reward = await env._run_verification()
                        return RewardResult(reward=reward)
                    except Exception as e:
                        logger.exception("Verification failed")
                        return RewardResult(reward=0.0, info={"error": str(e)})

            self.reward_fn = TBRewardFunction()

    async def _run_verification(self) -> float:
        """Execute test script and parse reward."""

        await self._env.upload_dir(
            source_dir=self.task_paths.tests_dir,
            target_dir="/tests",
        )

        test_cmd = f"bash /tests/test.sh | tee {EnvironmentPaths.verifier_dir}/test-stdout.txt 2>&1"
        await self._executor(test_cmd, self.config.test_timeout_sec)

        if not self._env.is_mounted:
            await self._env.download_dir(
                source_dir=str(EnvironmentPaths.verifier_dir),
                target_dir=self.trial_paths.verifier_dir,
            )

        return self._parse_reward()

    def _parse_reward(self) -> float:
        """Parse reward from reward.txt, returning 0.0 or 1.0."""
        reward_path = self.trial_paths.reward_text_path
        if reward_path.exists() and reward_path.stat().st_size > 0:
            value = float(reward_path.read_text().strip())
            return 1.0 if value >= 1.0 else 0.0
        logger.warning(f"No reward file at {reward_path}")
        return 0.0

    async def _executor(self, cmd: str, timeout_sec: int = 900) -> tuple[int, str, str]:
        """Execute a command in the container."""
        result = await self._env.exec(cmd, timeout_sec=timeout_sec)
        return result.return_code, result.stdout or "", result.stderr or ""

    def _create_execute_command_tool(self):
        """Create the execute_command tool with access to self._executor."""
        executor = self._executor

        @tool
        async def execute_command(command: str) -> str:
            """Execute a shell command in the environment.

            Args:
                command: The shell command to execute (e.g., "ls -la", "cat file.txt")

            Returns:
                Command output (stdout + stderr combined).
            """
            returncode, stdout, stderr = await executor(command)

            output = stdout or ""
            if stderr:
                output += f"\n[stderr]: {stderr}"
            if returncode != 0:
                output += f"\n[exit code]: {returncode}"

            return output if output.strip() else "(no output)"

        return execute_command

    @override
    async def reset(self) -> None:
        """Build and start the Docker environment."""
        self.trial_paths.mkdir()
        # Generate unique session_id to avoid conflicts when multiple rollouts
        session_id = f"{self.config.task_id}-{uuid.uuid4().hex[:8]}"
        self._env = EnvironmentFactory.create_environment(
            type=EnvironmentType.DOCKER,
            environment_dir=self.task_paths.environment_dir,
            environment_name=session_id,
            session_id=session_id,
            trial_paths=self.trial_paths,
            task_env_config=self.config.env_config,
        )
        await self._env.start(force_build=True)
        self._execute_command = self._create_execute_command_tool()

    @override
    def get_tools(self) -> list:
        return [self._execute_command]

    @override
    async def cleanup(self) -> None:
        if self._env:
            await self._env.stop(delete=True)
            self._env = None
        self._execute_command = None
