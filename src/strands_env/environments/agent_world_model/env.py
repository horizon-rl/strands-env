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

"""Agent World Model environment.

Creates SQLite-backed tool-use environments from JSONL data folders, with
dynamically generated API tools and verifier-based reward computation.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override

from strands_env.core import Environment, ModelFactory
from strands_env.core.types import RewardFunction

from .data_loader import AgentWorldModelDataLoader
from .db import copy_database, create_database
from .reward import AgentWorldModelRewardFunction
from .tools import create_api_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentWorldModelEnvConfig:
    """Configuration for an Agent World Model environment episode.

    Attributes:
        scenario: Scenario name (e.g., ``"e_commerce_33"``).
        task_idx: Task index within the scenario (0-9 for AgentWorldModel-1K).
        data_dir: Path to the AgentWorldModel-format data folder.
        db_dir: Directory for SQLite database files. If ``None``, a temp directory is used.
    """

    scenario: str
    task_idx: int
    data_dir: Path
    db_dir: Path | None = None


class AgentWorldModelEnv(Environment):
    """RL environment backed by SQLite and dynamically generated API tools.

    Loads scenario data from an AgentWorldModel-format folder, creates a SQLite database,
    generates API tools from the scenario's specification and implementation code,
    and computes rewards using the scenario's verifier.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        config: AgentWorldModelEnvConfig,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int = 25,
        max_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=None,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )

        self.config = config
        self.data_loader = AgentWorldModelDataLoader(config.data_dir)

        # Set after super().__init__ because default reward needs self reference
        self.reward_fn = reward_fn or AgentWorldModelRewardFunction(self)

        # Populated during reset()
        self._tools: list = []
        self._test_client = None
        self._temp_dir: str | None = None
        self.db_path: Path | None = None
        self.initial_db_path: Path | None = None

    @override
    async def reset(self) -> None:
        """Set up the SQLite database and generate API tools for the scenario."""
        # Clean up any previous state
        await self.cleanup()

        scenario = self.config.scenario
        logger.info("Resetting environment for scenario '%s', task %d", scenario, self.config.task_idx)

        # Determine DB directory
        if self.config.db_dir:
            db_dir = self.config.db_dir
            db_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="agent_world_model_env_")
            db_dir = Path(self._temp_dir)

        # Create working database
        self.db_path = db_dir / f"{scenario}.db"
        db_schema = self.data_loader.get_db_schema(scenario)
        sample_data = self.data_loader.get_sample_data(scenario)
        create_database(self.db_path, db_schema, sample_data)

        # Snapshot the initial state for verification
        self.initial_db_path = db_dir / f"{scenario}_initial.db"
        copy_database(self.db_path, self.initial_db_path)

        # Generate API tools from the scenario's code and spec
        full_code = self.data_loader.get_env_code(scenario)
        api_spec = self.data_loader.get_api_spec(scenario)
        self._tools, self._test_client = create_api_tools(full_code, api_spec, self.db_path)

        logger.info("Environment ready: %d tools, DB at %s", len(self._tools), self.db_path)

    @override
    def get_tools(self) -> list:
        """Return the dynamically generated API tools."""
        return self._tools

    @override
    async def cleanup(self) -> None:
        """Release resources: close the test client and remove temp files."""
        # This allows the environment to be reused
        if self._test_client is not None:
            try:
                self._test_client.close()
            except Exception:
                pass
            self._test_client = None

        self._tools = []

        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None

        self.db_path = None
        self.initial_db_path = None
