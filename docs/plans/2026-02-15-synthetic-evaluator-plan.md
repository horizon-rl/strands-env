# SyntheticEvaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `SyntheticEvaluator(Evaluator)` that runs all 1000 AWM scenarios concurrently with a real LLM and saves full agent trajectories as JSONL.

**Architecture:** Subclass the existing `Evaluator` with a `load_dataset()` that reads AWM JSONL data and yields one `Action` per scenario+task. The evaluator's `env_factory` creates a `SyntheticEnv` per sample, extracting scenario/task from `TaskContext` extra fields. The CLI gets new options for `--data-dir`, `--scenarios`, and `--max-tasks-per-scenario`.

**Tech Stack:** Python 3.10+, strands-env Evaluator framework, AWMDataLoader, SyntheticEnv, pytest, click CLI

---

### Task 1: SyntheticEvaluator — Failing Tests

**Files:**
- Create: `tests/unit/test_synthetic_evaluator.py`

**Step 1: Write the failing tests**

```python
"""Unit tests for SyntheticEvaluator."""

import json
from pathlib import Path

import pytest

from strands_env.core.types import Action, TaskContext

# Reuse test data from test_synthetic_env.py
SCENARIO_NAME = "test_scenario_1"

SAMPLE_SCENARIO = {"name": SCENARIO_NAME, "description": "A test scenario for unit testing."}

SAMPLE_TASKS = {
    "scenario": SCENARIO_NAME,
    "tasks": ["Task 1: do something", "Task 2: do something else", "Task 3: another task"],
}

SAMPLE_DB_SCHEMA = {
    "scenario": SCENARIO_NAME,
    "db_schema": {
        "tables": [
            {
                "name": "users",
                "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
                "indexes": [],
            }
        ]
    },
}

SAMPLE_DATA = {
    "scenario": SCENARIO_NAME,
    "sample_data": {
        "tables": [
            {
                "table_name": "users",
                "insert_statements": [
                    "INSERT INTO users (id, name) VALUES (1, 'Alice');",
                ],
            }
        ]
    },
}

SAMPLE_API_SPEC = {
    "scenario": SCENARIO_NAME,
    "api_spec": {"api_groups": []},
}

SAMPLE_ENV_CODE = {
    "scenario": SCENARIO_NAME,
    "full_code": "from fastapi import FastAPI\napp = FastAPI()\n",
}

SAMPLE_VERIFIER = {
    "scenario": SCENARIO_NAME,
    "task_idx": 0,
    "verification": {
        "code": 'def verify_task_completion(initial_db_path, final_db_path, final_answer=None):\n    return {"result": "complete"}\n',
    },
}


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary AWM data directory."""
    for filename, data in [
        ("gen_scenario.jsonl", SAMPLE_SCENARIO),
        ("gen_tasks.jsonl", SAMPLE_TASKS),
        ("gen_db.jsonl", SAMPLE_DB_SCHEMA),
        ("gen_sample.jsonl", SAMPLE_DATA),
        ("gen_spec.jsonl", SAMPLE_API_SPEC),
        ("gen_envs.jsonl", SAMPLE_ENV_CODE),
        ("gen_verifier.pure_code.jsonl", SAMPLE_VERIFIER),
    ]:
        with open(tmp_path / filename, "w") as f:
            f.write(json.dumps(data) + "\n")
    return tmp_path


@pytest.fixture
def mock_env_factory():
    """A no-op async env factory (evaluator tests don't run step())."""
    from unittest.mock import AsyncMock, MagicMock

    async def factory(action):
        env = MagicMock()
        env.reset = AsyncMock()
        env.step = AsyncMock()
        env.cleanup = AsyncMock()
        return env

    return factory


class TestSyntheticEvaluatorLoadDataset:
    def test_loads_all_actions(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        # 1 scenario x 3 tasks = 3 actions
        assert len(actions) == 3

    def test_action_has_correct_message(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        assert actions[0].message == "Task 1: do something"
        assert actions[1].message == "Task 2: do something else"
        assert actions[2].message == "Task 3: another task"

    def test_action_has_scenario_in_context(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        ctx = actions[0].task_context
        assert ctx.scenario == SCENARIO_NAME
        assert ctx.task_idx == 0
        assert ctx.id == f"{SCENARIO_NAME}_0"

    def test_action_context_ids_are_unique(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        ids = [a.task_context.id for a in actions]
        assert len(ids) == len(set(ids))


class TestSyntheticEvaluatorFiltering:
    def test_scenarios_filter(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            scenarios=["nonexistent_scenario"],
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 0

    def test_scenarios_filter_matches(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            scenarios=[SCENARIO_NAME],
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 3

    def test_max_tasks_per_scenario(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            max_tasks_per_scenario=1,
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 1
        assert actions[0].task_context.task_idx == 0


class TestSyntheticEvaluatorBenchmarkName:
    def test_benchmark_name(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        assert evaluator.benchmark_name == "synthetic"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_synthetic_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strands_env.eval.benchmarks.synthetic'`

**Step 3: Commit**

```bash
git add tests/unit/test_synthetic_evaluator.py
git commit -m "test(eval): add failing tests for SyntheticEvaluator"
```

---

### Task 2: SyntheticEvaluator — Implementation

**Files:**
- Create: `src/strands_env/eval/benchmarks/synthetic.py`

**Step 1: Write the implementation**

```python
"""SyntheticEvaluator for AWM-format (AgentWorldModel) benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.environments.synthetic_env.data_loader import AWMDataLoader

from ..evaluator import AsyncEnvFactory, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


@register_eval("synthetic")
class SyntheticEvaluator(Evaluator):
    """Evaluator for AWM-format synthetic agentic environments.

    Loads scenarios from an AWM data folder and creates one Action per
    scenario+task pair. Each action carries ``scenario`` and ``task_idx``
    in its TaskContext extra fields for env_factory to use.
    """

    benchmark_name: str = "synthetic"

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
        """Load AWM scenarios and yield one Action per scenario+task pair."""
        loader = AWMDataLoader(self.data_dir)
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
                    ),
                )
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_synthetic_evaluator.py -v`
Expected: All 8 tests PASS

**Step 3: Run linting**

Run: `ruff check src/strands_env/eval/benchmarks/synthetic.py`
Expected: Clean

**Step 4: Commit**

```bash
git add src/strands_env/eval/benchmarks/synthetic.py
git commit -m "feat(eval): add SyntheticEvaluator for AWM benchmarks"
```

---

### Task 3: CLI Integration — Failing Tests

**Files:**
- Modify: `tests/unit/test_synthetic_evaluator.py` (append new test class)

**Step 1: Add CLI-related tests**

Append to `tests/unit/test_synthetic_evaluator.py`:

```python
class TestSyntheticEvaluatorRegistered:
    def test_registered_in_registry(self):
        from strands_env.eval import get_benchmark

        cls = get_benchmark("synthetic")
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        assert cls is SyntheticEvaluator

    def test_listed_in_benchmarks(self):
        from strands_env.eval import list_benchmarks

        assert "synthetic" in list_benchmarks()
```

**Step 2: Run tests to verify they pass**

These tests should already pass since `@register_eval("synthetic")` is auto-discovered.

Run: `pytest tests/unit/test_synthetic_evaluator.py::TestSyntheticEvaluatorRegistered -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_synthetic_evaluator.py
git commit -m "test(eval): add registry tests for SyntheticEvaluator"
```

---

### Task 4: CLI Options for Synthetic Benchmark

The current CLI `eval run` command uses hook files (`--env`) for env_factory creation. The `SyntheticEvaluator` needs `data_dir`, `scenarios`, and `max_tasks_per_scenario` passed to its constructor, but the current `run_cmd` at `src/strands_env/cli/eval.py:314` creates the evaluator with only standard Evaluator kwargs.

We need to pass synthetic-specific kwargs to the evaluator constructor.

**Files:**
- Modify: `src/strands_env/cli/eval.py:60-321` (add CLI options + forward kwargs)

**Step 1: Add CLI options**

Add these options to the `@eval_group.command("run")` decorator chain, after line 196 (after `--keep-tokens`):

```python
# Synthetic benchmark specific
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to AWM-format data folder (required for 'synthetic' benchmark).",
)
@click.option(
    "--scenarios",
    type=str,
    default=None,
    help="Comma-separated list of scenario names to evaluate (default: all).",
)
@click.option(
    "--max-tasks-per-scenario",
    type=int,
    default=None,
    help="Maximum number of tasks per scenario (default: all).",
)
```

Add the new parameters to the `run_cmd` function signature (after `keep_tokens: bool`):

```python
    # Synthetic-specific
    data_dir: Path | None,
    scenarios: str | None,
    max_tasks_per_scenario: int | None,
```

**Step 2: Forward kwargs to evaluator constructor**

Replace the evaluator construction block at line 314 with logic that passes extra kwargs for benchmarks that accept them:

```python
    # Build evaluator kwargs
    evaluator_kwargs = dict(
        env_factory=env_factory,
        max_concurrency=eval_config.max_concurrency,
        n_samples_per_prompt=eval_config.n_samples_per_prompt,
        output_path=results_path,
        save_interval=eval_config.save_interval,
        keep_tokens=eval_config.keep_tokens,
    )

    # Pass synthetic-specific kwargs if provided
    if data_dir is not None:
        evaluator_kwargs["data_dir"] = data_dir
    if scenarios is not None:
        evaluator_kwargs["scenarios"] = [s.strip() for s in scenarios.split(",")]
    if max_tasks_per_scenario is not None:
        evaluator_kwargs["max_tasks_per_scenario"] = max_tasks_per_scenario

    # Create evaluator
    evaluator = evaluator_cls(**evaluator_kwargs)
```

**Step 3: Validate data_dir is required for synthetic benchmark**

Add validation after the evaluator class is resolved (around line 263):

```python
    # Validate synthetic-specific requirements
    if benchmark_name == "synthetic" and data_dir is None:
        raise click.ClickException("--data-dir is required for the 'synthetic' benchmark.")
```

**Step 4: Run linting**

Run: `ruff check src/strands_env/cli/eval.py`
Expected: Clean

**Step 5: Commit**

```bash
git add src/strands_env/cli/eval.py
git commit -m "feat(cli): add --data-dir, --scenarios, --max-tasks-per-scenario options for synthetic eval"
```

---

### Task 5: Env Hook File for Synthetic Benchmark

Users need an env hook file to create the `env_factory` that wires `SyntheticEnv` to the CLI. This is a reference/example file.

**Files:**
- Create: `src/strands_env/environments/synthetic_env/env_hook.py`

**Step 1: Write the env hook**

```python
"""Environment hook file for the synthetic benchmark.

Usage:
    strands-env eval run synthetic --env src/strands_env/environments/synthetic_env/env_hook.py --data-dir /path/to/AWM-1K
"""

from __future__ import annotations

from pathlib import Path

from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.synthetic_env import SyntheticEnv, SyntheticEnvConfig


def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create an async env_factory for SyntheticEnv.

    The returned factory extracts ``scenario`` and ``task_idx`` from
    the action's TaskContext extra fields (set by SyntheticEvaluator).
    """

    async def env_factory(action):
        ctx = action.task_context
        config = SyntheticEnvConfig(
            scenario=ctx.scenario,
            task_idx=ctx.task_idx,
            data_dir=Path(ctx.data_dir) if hasattr(ctx, "data_dir") else Path.cwd(),
        )
        return SyntheticEnv(
            model_factory=model_factory,
            config=config,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
        )

    return env_factory
```

**Step 2: Commit**

```bash
git add src/strands_env/environments/synthetic_env/env_hook.py
git commit -m "feat(env): add env hook file for synthetic benchmark CLI usage"
```

---

### Task 6: Pass data_dir Through TaskContext

The env_factory needs `data_dir` to create `SyntheticEnvConfig`. Currently `data_dir` lives on the evaluator. We should pass it through `TaskContext` extra fields so the env hook can access it.

**Files:**
- Modify: `src/strands_env/eval/benchmarks/synthetic.py` (add `data_dir` to TaskContext)

**Step 1: Update load_dataset to include data_dir in TaskContext**

In `load_dataset()`, change the `TaskContext` construction to include `data_dir`:

```python
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
```

**Step 2: Update test to verify data_dir in context**

Add to `TestSyntheticEvaluatorLoadDataset`:

```python
    def test_action_has_data_dir_in_context(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.synthetic import SyntheticEvaluator

        evaluator = SyntheticEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        assert actions[0].task_context.data_dir == str(data_dir)
```

**Step 3: Run tests**

Run: `pytest tests/unit/test_synthetic_evaluator.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/strands_env/eval/benchmarks/synthetic.py tests/unit/test_synthetic_evaluator.py
git commit -m "feat(eval): pass data_dir through TaskContext for env_factory access"
```

---

### Task 7: Full Test Suite Run + Linting

**Step 1: Run all unit tests**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS (existing + new)

**Step 2: Run linting on all new/modified files**

Run: `ruff check src/strands_env/eval/benchmarks/synthetic.py src/strands_env/cli/eval.py src/strands_env/environments/synthetic_env/env_hook.py`
Expected: Clean

Run: `ruff format --check src/strands_env/eval/benchmarks/synthetic.py src/strands_env/cli/eval.py src/strands_env/environments/synthetic_env/env_hook.py`
Expected: Clean

**Step 3: Commit any fixes if needed**

---

### Task 8: Verify CLI Registration

**Step 1: Verify the benchmark shows up in the CLI**

Run: `cd /Users/lyichuan/PycharmProjects/strands-env && python -m strands_env.cli eval list`
Expected: Output includes `synthetic` in the benchmark list

**Step 2: Verify --help shows new options**

Run: `python -m strands_env.cli eval run --help`
Expected: Shows `--data-dir`, `--scenarios`, `--max-tasks-per-scenario` options
