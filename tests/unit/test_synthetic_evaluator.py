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

"""Unit tests for AgentWorldModelEvaluator."""

import json

import pytest

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
    """Create a temporary AgentWorldModel data directory."""
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


class TestAgentWorldModelEvaluatorLoadDataset:
    def test_loads_all_actions(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 3

    def test_action_has_correct_message(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        assert actions[0].message == "Task 1: do something"
        assert actions[1].message == "Task 2: do something else"
        assert actions[2].message == "Task 3: another task"

    def test_action_has_scenario_in_context(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        ctx = actions[0].task_context
        assert ctx.scenario == SCENARIO_NAME
        assert ctx.task_idx == 0
        assert ctx.id == f"{SCENARIO_NAME}_0"

    def test_action_has_data_dir_in_context(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        assert actions[0].task_context.data_dir == str(data_dir)

    def test_action_context_ids_are_unique(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        actions = list(evaluator.load_dataset())
        ids = [a.task_context.id for a in actions]
        assert len(ids) == len(set(ids))


class TestAgentWorldModelEvaluatorFiltering:
    def test_scenarios_filter(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            scenarios=["nonexistent_scenario"],
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 0

    def test_scenarios_filter_matches(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            scenarios=[SCENARIO_NAME],
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 3

    def test_max_tasks_per_scenario(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
            max_tasks_per_scenario=1,
        )
        actions = list(evaluator.load_dataset())
        assert len(actions) == 1
        assert actions[0].task_context.task_idx == 0


class TestAgentWorldModelEvaluatorBenchmarkName:
    def test_benchmark_name(self, data_dir, mock_env_factory):
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        evaluator = AgentWorldModelEvaluator(
            env_factory=mock_env_factory,
            data_dir=data_dir,
        )
        assert evaluator.benchmark_name == "agent-world-model"


class TestAgentWorldModelEvaluatorRegistered:
    def test_registered_in_registry(self):
        from strands_env.eval import get_benchmark
        from strands_env.eval.benchmarks.agent_world_model import AgentWorldModelEvaluator

        cls = get_benchmark("agent-world-model")
        assert cls is AgentWorldModelEvaluator

    def test_listed_in_benchmarks(self):
        from strands_env.eval import list_benchmarks

        assert "agent-world-model" in list_benchmarks()
