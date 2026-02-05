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

"""Evaluator for running agentic benchmarks with `strands-env` environments."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from pydantic import BaseModel

from strands_env.core import Action, Environment, StepResult

logger = logging.getLogger(__name__)

#: Type alias for environment factory function (async).
AsyncEnvFactory = Callable[[Action], Awaitable[Environment]]


class EvalSample(BaseModel):
    """Evaluation sample result."""

    action: Action
    """The action (task) that was evaluated."""

    step_result: StepResult
    """The result of the step (observation, reward, termination reason)."""


class Evaluator:
    """Evaluator for running concurrent environment evaluations."""

    def __init__(
        self,
        env_factory: AsyncEnvFactory,
        *,
        max_concurrency: int = 10,
        n_rollouts: int = 1,
        output_path: Path | str = Path.cwd() / "results.jsonl",
        save_interval: int = 10,
        keep_tokens: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            env_factory: Async factory function that creates a fresh Environment per sample.
            max_concurrency: Maximum concurrent evaluate_sample() calls.
            n_rollouts: Number of rollouts per problem (for pass@k, set to max(k_values)).
            output_path: Path to JSONL file for saving results. Enables resume.
            save_interval: Flush results to disk every N completed samples.
            keep_tokens: Keep token-level observation in results (only valid for `SGLangModel` backends).
        """
        self.env_factory: AsyncEnvFactory = env_factory

        # Configuration
        self.max_concurrency = max_concurrency
        self.n_rollouts = n_rollouts
        self.output_path = Path(output_path)
        self.save_interval = save_interval
        self.keep_tokens = keep_tokens

        # Runtime state: {problem_id: [samples]}
        self.results: dict[str, list[EvalSample]] = defaultdict(list)
        self.completed_ids: set[str] = set()  # Tracks individual sample IDs for checkpoint

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset from file. Override to implement custom dataset loading logic."""
        raise NotImplementedError("Evaluator subclasses must implement load_dataset()")

    def load_results(self) -> None:
        """Load completed samples from results file."""
        if not self.output_path.exists():
            return

        self.results = defaultdict(list)
        self.completed_ids = set()

        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                problem_id = data.pop("problem_id")
                sample = EvalSample.model_validate(data)
                self.results[problem_id].append(sample)
                self.completed_ids.add(sample.action.task_context.id)

        total = sum(len(samples) for samples in self.results.values())
        logger.info(f"Loaded {total} completed samples from: {self.output_path}")

    def save_results(self) -> None:
        """Write all samples to results file."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for problem_id, samples in self.results.items():
                for sample in samples:
                    data = sample.model_dump()
                    data["problem_id"] = problem_id
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

        total = sum(len(samples) for samples in self.results.values())
        logger.info(f"Saved {total} samples to: {self.output_path}")

    async def evaluate_sample(self, action: Action) -> EvalSample:
        """Evaluate a single sample."""
        env = await self.env_factory(action)
        await env.reset()
        step_result = await env.step(action)
        if not self.keep_tokens:
            # Token trajectory is usually not needed for evaluation.
            step_result.observation.tokens = None
        await env.cleanup()
        return EvalSample(action=action, step_result=step_result)

    async def run(self, actions: Iterable[Action]) -> dict[str, list[EvalSample]]:
        """Run evaluation on a collection of actions.

        Each action is duplicated `n_rollouts` times for pass@k computation.
        Completed samples are saved incrementally and can be resumed via output_path.

        Args:
            actions: `Iterable` of `Action`s to evaluate.

        Returns:
            Dict mapping problem_id to list of `EvalSample` rollouts.
        """
        self.load_results()

        # Build list of (problem_id, sample_id, action) for processing
        to_process: list[tuple[str, str, Action]] = []
        for action in actions:
            problem_id = action.task_context.id
            for i in range(self.n_rollouts):
                sample_id = f"{problem_id}_{i}"
                if sample_id not in self.completed_ids:
                    expanded = action.model_copy(deep=True)
                    expanded.task_context.id = sample_id
                    to_process.append((problem_id, sample_id, expanded))

        semaphore = asyncio.Semaphore(self.max_concurrency)
        save_counter = 0
        completed_counter = 0
        total = len(to_process)

        async def process(problem_id: str, sample_id: str, action: Action) -> None:
            nonlocal save_counter, completed_counter
            async with semaphore:
                sample = await self.evaluate_sample(action)
                self.results[problem_id].append(sample)
                self.completed_ids.add(sample_id)
                completed_counter += 1
                save_counter += 1
                if save_counter >= self.save_interval:
                    self.save_results()
                    logger.info(f"Progress: {completed_counter}/{total} samples completed")
                    save_counter = 0

        tasks = [process(pid, sid, action) for pid, sid, action in to_process]
        await asyncio.gather(*tasks)

        self.save_results()
        return dict(self.results)

    @staticmethod
    def _pass_at_k_single(n: int, c: int, k: int) -> float:
        """Compute pass@k for a single problem using unbiased estimator.

        pass@k = 1 - C(n-c, k) / C(n, k)

        Uses log-space for numerical stability with large factorials.
        """
        if n - c < k:
            return 1.0
        if c == 0:
            return 0.0

        log_ratio = 0.0
        for i in range(k):
            log_ratio += math.log(n - c - i) - math.log(n - i)
        return 1.0 - math.exp(log_ratio)

    @staticmethod
    def compute_pass_at_k(
        results: dict[str, list[EvalSample]],
        k_values: list[int] = [1],
        reward_threshold: float = 1.0,
    ) -> dict[int, float]:
        """Compute pass@k metric using unbiased estimator.

        Args:
            results: Dict mapping problem_id to list of sample rollouts.
            k_values: List of k values for pass@k computation.
            reward_threshold: Reward threshold for considering a sample "passed" (default: 1.0).

        Returns:
            Dictionary mapping k to average pass@k score.
        """
        if not results:
            return {k: 0.0 for k in k_values}

        def is_correct(s: EvalSample) -> bool:
            reward = s.step_result.reward
            return reward is not None and reward.reward >= reward_threshold

        # Compute pass@k for each k value
        pass_at_k = {}
        for k in k_values:
            scores = []
            for samples in results.values():
                n = len(samples)
                c = sum(1 for s in samples if is_correct(s))
                if k <= n:
                    scores.append(Evaluator._pass_at_k_single(n, c, k))
            pass_at_k[k] = sum(scores) / len(scores) if scores else 0.0

        return pass_at_k
