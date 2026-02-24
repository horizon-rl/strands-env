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

"""AgentWorldModelEvaluator for Agent World Model benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.environments.agent_world_model.data_loader import AgentWorldModelDataLoader

from ..evaluator import AsyncEnvFactory, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


@register_eval("agent-world-model")
class AgentWorldModelEvaluator(Evaluator):
    """Evaluator for AgentWorldModel-format agentic environments.

    Loads scenarios from an AgentWorldModel data folder and creates one Action per
    scenario+task pair. Each action carries ``scenario`` and ``task_idx``
    in its TaskContext extra fields for env_factory to use.
    """

    benchmark_name: str = "agent-world-model"

    def __init__(
        self,
        env_factory: AsyncEnvFactory,
        *,
        data_dir: Path | str,
        scenarios: list[str] | None = None,
        max_tasks_per_scenario: int | None = None,
        max_concurrency: int = 10,
        n_samples_per_prompt: int = 1,
        output_path: Path | str = Path.cwd() / "results.jsonl",
        save_interval: int = 10,
        keep_tokens: bool = False,
    ):
        """Initialize the AgentWorldModelEvaluator.

        Args:
            env_factory: Async factory function that creates a fresh Environment per sample.
            data_dir: Path to AgentWorldModel-format data folder containing JSONL files.
            scenarios: Specific scenario names to evaluate. Defaults to all.
            max_tasks_per_scenario: Cap on tasks per scenario. Defaults to all.
            max_concurrency: Maximum concurrent evaluate_sample() calls.
            n_samples_per_prompt: Number of samples per prompt (for pass@k).
            output_path: Path to JSONL file for saving results.
            save_interval: Flush results to disk every N completed samples.
            keep_tokens: Keep token-level observation in results.
        """
        super().__init__(
            env_factory=env_factory,
            max_concurrency=max_concurrency,
            n_samples_per_prompt=n_samples_per_prompt,
            output_path=output_path,
            save_interval=save_interval,
            keep_tokens=keep_tokens,
        )
        self.data_dir = Path(data_dir)
        self.scenarios = scenarios
        self.max_tasks_per_scenario = max_tasks_per_scenario

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load AgentWorldModel scenarios and yield one Action per scenario+task pair."""
        loader = AgentWorldModelDataLoader(self.data_dir)
        scenario_names = self.scenarios if self.scenarios is not None else loader.list_scenarios()

        for scenario in scenario_names:
            try:
                tasks = loader.get_tasks(scenario)
            except KeyError:
                logger.warning("Scenario '%s' not found in data, skipping", scenario)
                continue

            limit = self.max_tasks_per_scenario if self.max_tasks_per_scenario is not None else len(tasks)
            for task_idx in range(min(limit, len(tasks))):
                yield Action(
                    message=tasks[task_idx],
                    task_context=TaskContext(
                        id=f"{scenario}_{task_idx}",
                        ground_truth=None,
                        scenario=scenario,
                        task_idx=task_idx,
                        data_dir=str(self.data_dir),
                    ),
                )
