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

"""Unit tests for the Agent World Model environment."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from strands_env.core.types import Action, Observation, StepResult, TaskContext, TerminationReason
from strands_env.environments.agent_world_model.data_loader import AgentWorldModelDataLoader
from strands_env.environments.agent_world_model.db import copy_database, create_database
from strands_env.environments.agent_world_model.env import AgentWorldModelEnv, AgentWorldModelEnvConfig
from strands_env.environments.agent_world_model.reward import AgentWorldModelRewardFunction

# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

SCENARIO_NAME = "test_scenario_1"

SAMPLE_SCENARIO = {"name": SCENARIO_NAME, "description": "A test scenario for unit testing."}

SAMPLE_TASKS = {
    "scenario": SCENARIO_NAME,
    "tasks": ["Task 1: do something", "Task 2: do something else"],
}

SAMPLE_DB_SCHEMA = {
    "scenario": SCENARIO_NAME,
    "db_schema": {
        "tables": [
            {
                "name": "users",
                "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE);",
                "indexes": ["CREATE INDEX idx_users_email ON users(email);"],
            },
            {
                "name": "items",
                "ddl": "CREATE TABLE items (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, FOREIGN KEY (user_id) REFERENCES users(id));",
                "indexes": [],
            },
        ]
    },
    "db_path": "./outputs/databases/test.db",
}

SAMPLE_DATA = {
    "scenario": SCENARIO_NAME,
    "tables_count": 2,
    "inserts_count": 4,
    "sample_data": {
        "tables": [
            {
                "table_name": "users",
                "reasoning": "Create test users.",
                "insert_statements": [
                    "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com');",
                    "INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@example.com');",
                ],
            },
            {
                "table_name": "items",
                "reasoning": "Create test items.",
                "insert_statements": [
                    "INSERT INTO items (id, user_id, title) VALUES (1, 1, 'Item A');",
                    "INSERT INTO items (id, user_id, title) VALUES (2, 2, 'Item B');",
                ],
            },
        ]
    },
}

SAMPLE_API_SPEC = {
    "scenario": SCENARIO_NAME,
    "api_spec": {
        "api_groups": [
            {
                "group_name": "Users",
                "endpoints": [
                    {
                        "path": "/api/users",
                        "method": "GET",
                        "summary": "List all users",
                        "description": "Return a list of all users.",
                        "operation_id": "list_users",
                        "tags": ["users"],
                        "request_params": {
                            "limit": {
                                "type": "integer",
                                "param_type": "query",
                                "required": False,
                                "description": "Max number of users to return.",
                            }
                        },
                        "response": {},
                    },
                    {
                        "path": "/api/users/{user_id}",
                        "method": "GET",
                        "summary": "Get user by ID",
                        "description": "Retrieve a user by their ID.",
                        "operation_id": "get_user",
                        "tags": ["users"],
                        "request_params": {
                            "user_id": {
                                "type": "integer",
                                "param_type": "path",
                                "required": True,
                                "description": "The user ID.",
                            }
                        },
                        "response": {},
                    },
                ],
            }
        ]
    },
}

SAMPLE_ENV_CODE = {
    "scenario": SCENARIO_NAME,
    "db_path": "./outputs/databases/test.db",
    "full_code": """
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel
from typing import Optional, List
import os
import sqlite3

DATABASE_PATH = os.getenv("DATABASE_PATH", "sqlite:///test.db")

app = FastAPI(title="Test API")

def get_db_path():
    path = DATABASE_PATH
    if path.startswith("sqlite:///"):
        path = path[len("sqlite:///"):]
    return path

@app.get("/api/users")
async def list_users(limit: int = Query(10)):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"users": rows}

@app.get("/api/users/{user_id}")
async def get_user(user_id: int = Path(...)):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {"user": dict(row) if row else None}
""",
}

SAMPLE_VERIFIER = {
    "scenario": SCENARIO_NAME,
    "task_idx": 0,
    "task": "Task 1: do something",
    "verification": {
        "code": """def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    conn = sqlite3.connect(final_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    if count >= 2:
        return {"result": "complete"}
    return {"result": "others"}
""",
        "raw_response": "",
    },
}


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with all the test JSONL files."""
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
def mock_model_factory():
    """Return a model factory that produces a mock model."""
    mock_model = MagicMock()
    return lambda: mock_model


@pytest.fixture
async def agent_world_model_env(data_dir, mock_model_factory):
    """Create an AgentWorldModelEnv, reset it, yield it, then clean up."""
    config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
    env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
    await env.reset()
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# AgentWorldModelDataLoader tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelDataLoader:
    def test_init_valid_dir(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        assert loader.data_dir == data_dir

    def test_init_invalid_dir(self):
        with pytest.raises(FileNotFoundError):
            AgentWorldModelDataLoader("/nonexistent/path")

    def test_list_scenarios(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        scenarios = loader.list_scenarios()
        assert scenarios == [SCENARIO_NAME]

    def test_get_scenario(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        scenario = loader.get_scenario(SCENARIO_NAME)
        assert scenario["name"] == SCENARIO_NAME
        assert "description" in scenario

    def test_get_scenario_not_found(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        with pytest.raises(KeyError):
            loader.get_scenario("nonexistent")

    def test_get_tasks(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        tasks = loader.get_tasks(SCENARIO_NAME)
        assert len(tasks) == 2
        assert tasks[0] == "Task 1: do something"

    def test_get_db_schema(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        schema = loader.get_db_schema(SCENARIO_NAME)
        assert "tables" in schema
        assert len(schema["tables"]) == 2
        assert schema["tables"][0]["name"] == "users"

    def test_get_sample_data(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        sample = loader.get_sample_data(SCENARIO_NAME)
        assert "tables" in sample
        assert len(sample["tables"]) == 2

    def test_get_api_spec(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        spec = loader.get_api_spec(SCENARIO_NAME)
        assert "api_groups" in spec
        assert len(spec["api_groups"]) == 1
        endpoints = spec["api_groups"][0]["endpoints"]
        assert len(endpoints) == 2

    def test_get_env_code(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        code = loader.get_env_code(SCENARIO_NAME)
        assert "FastAPI" in code
        assert "app" in code

    def test_get_verifier(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        code = loader.get_verifier(SCENARIO_NAME, 0)
        assert "verify_task_completion" in code

    def test_get_verifier_invalid_task_idx(self, data_dir):
        loader = AgentWorldModelDataLoader(data_dir)
        with pytest.raises(KeyError):
            loader.get_verifier(SCENARIO_NAME, 99)


# ---------------------------------------------------------------------------
# Database creation tests
# ---------------------------------------------------------------------------


class TestDatabaseCreation:
    def test_create_database(self, tmp_path):
        db_path = tmp_path / "test.db"
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])

        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "users" in tables
        assert "items" in tables

        # Check data was inserted
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 2

        cursor.execute("SELECT COUNT(*) FROM items")
        assert cursor.fetchone()[0] == 2

        # Check specific data
        cursor.execute("SELECT name FROM users WHERE id = 1")
        assert cursor.fetchone()[0] == "Alice"

        conn.close()

    def test_create_database_overwrites(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Create twice — second should overwrite
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 2  # Not 4
        conn.close()

    def test_copy_database(self, tmp_path):
        db_path = tmp_path / "original.db"
        copy_path = tmp_path / "copy.db"

        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])
        copy_database(db_path, copy_path)

        assert copy_path.exists()

        conn = sqlite3.connect(str(copy_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 2
        conn.close()


# ---------------------------------------------------------------------------
# Tool generation tests
# ---------------------------------------------------------------------------


class TestToolGeneration:
    def test_create_api_tools(self, tmp_path):
        from strands_env.environments.agent_world_model.tools import create_api_tools

        db_path = tmp_path / "test.db"
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])

        tools, client = create_api_tools(
            SAMPLE_ENV_CODE["full_code"],
            SAMPLE_API_SPEC["api_spec"],
            db_path,
        )

        try:
            assert len(tools) == 2

            tool_names = {t.tool_name for t in tools}
            assert "list_users" in tool_names
            assert "get_user" in tool_names

            # Verify tool specs have correct structure
            for t in tools:
                spec = t.tool_spec
                assert "name" in spec
                assert "description" in spec
                assert "inputSchema" in spec
                assert "json" in spec["inputSchema"]
        finally:
            client.close()

    def test_tool_handler_list_users(self, tmp_path):
        from strands_env.environments.agent_world_model.tools import create_api_tools

        db_path = tmp_path / "test.db"
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])

        tools, client = create_api_tools(
            SAMPLE_ENV_CODE["full_code"],
            SAMPLE_API_SPEC["api_spec"],
            db_path,
        )

        try:
            list_tool = next(t for t in tools if t.tool_name == "list_users")
            result = list_tool._tool_func(
                {"toolUseId": "test-123", "input": {"limit": 10}},
            )
            assert result["status"] == "success"
            assert "Alice" in result["content"][0]["text"]
            assert "Bob" in result["content"][0]["text"]
        finally:
            client.close()

    def test_tool_handler_get_user(self, tmp_path):
        from strands_env.environments.agent_world_model.tools import create_api_tools

        db_path = tmp_path / "test.db"
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])

        tools, client = create_api_tools(
            SAMPLE_ENV_CODE["full_code"],
            SAMPLE_API_SPEC["api_spec"],
            db_path,
        )

        try:
            get_tool = next(t for t in tools if t.tool_name == "get_user")
            result = get_tool._tool_func(
                {"toolUseId": "test-456", "input": {"user_id": 1}},
            )
            assert result["status"] == "success"
            assert "Alice" in result["content"][0]["text"]
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Reward / Verifier tests
# ---------------------------------------------------------------------------


class TestVerifier:
    def test_verifier_complete(self, tmp_path):
        """Test that verifier returns 'complete' when conditions are met."""
        db_path = tmp_path / "final.db"
        initial_path = tmp_path / "initial.db"
        create_database(db_path, SAMPLE_DB_SCHEMA["db_schema"], SAMPLE_DATA["sample_data"])
        copy_database(db_path, initial_path)

        # Run the verifier code
        verifier_code = SAMPLE_VERIFIER["verification"]["code"]
        namespace: dict = {}
        exec(verifier_code, namespace)
        verify_fn = namespace["verify_task_completion"]

        result = verify_fn(str(initial_path), str(db_path))
        assert result["result"] == "complete"

    def test_verifier_others(self, tmp_path):
        """Test that verifier returns 'others' when conditions are not met."""
        # Create a DB with no users
        db_path = tmp_path / "final.db"
        initial_path = tmp_path / "initial.db"
        schema = {
            "tables": [
                {"name": "users", "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);", "indexes": []}
            ]
        }
        empty_data = {"tables": []}
        create_database(db_path, schema, empty_data)
        copy_database(db_path, initial_path)

        verifier_code = SAMPLE_VERIFIER["verification"]["code"]
        namespace: dict = {}
        exec(verifier_code, namespace)
        verify_fn = namespace["verify_task_completion"]

        result = verify_fn(str(initial_path), str(db_path))
        assert result["result"] == "others"


# ---------------------------------------------------------------------------
# AgentWorldModelEnvConfig tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelEnvConfig:
    def test_config_creation(self, data_dir):
        config = AgentWorldModelEnvConfig(
            scenario=SCENARIO_NAME,
            task_idx=0,
            data_dir=data_dir,
        )
        assert config.scenario == SCENARIO_NAME
        assert config.task_idx == 0
        assert config.data_dir == data_dir
        assert config.db_dir is None


# ---------------------------------------------------------------------------
# AgentWorldModelEnv.__init__ tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelEnvInit:
    def test_default_system_prompt_loaded(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        assert "You are an agent interacting with a platform" in env.system_prompt

    def test_custom_system_prompt_overrides_default(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config, system_prompt="Custom prompt")
        assert env.system_prompt == "Custom prompt"

    def test_default_reward_fn_created(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        assert isinstance(env.reward_fn, AgentWorldModelRewardFunction)
        assert env.reward_fn._env is env

    def test_custom_reward_fn_used(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        custom_reward = MagicMock()
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config, reward_fn=custom_reward)
        assert env.reward_fn is custom_reward

    def test_config_stored(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        assert env.config is config
        assert env.data_loader.data_dir == data_dir

    def test_max_tool_iters_default(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        assert env.max_tool_iters == 25


# ---------------------------------------------------------------------------
# AgentWorldModelEnv.reset() tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelEnvReset:
    async def test_reset_creates_db_files(self, agent_world_model_env):
        assert agent_world_model_env.db_path is not None
        assert agent_world_model_env.initial_db_path is not None
        assert agent_world_model_env.db_path.exists()
        assert agent_world_model_env.initial_db_path.exists()

    async def test_reset_db_has_correct_data(self, agent_world_model_env):
        conn = sqlite3.connect(str(agent_world_model_env.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 2
        conn.close()

    async def test_reset_initial_db_is_snapshot(self, agent_world_model_env):
        # Modify the working DB
        conn = sqlite3.connect(str(agent_world_model_env.db_path))
        conn.execute("DELETE FROM users")
        conn.commit()
        conn.close()

        # Initial snapshot should be unchanged
        conn = sqlite3.connect(str(agent_world_model_env.initial_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 2
        conn.close()

    async def test_reset_creates_tools(self, agent_world_model_env):
        tools = agent_world_model_env.get_tools()
        assert len(tools) == 2
        tool_names = {t.tool_name for t in tools}
        assert "list_users" in tool_names
        assert "get_user" in tool_names

    async def test_reset_creates_test_client(self, agent_world_model_env):
        assert agent_world_model_env._test_client is not None

    async def test_reset_with_explicit_db_dir(self, data_dir, mock_model_factory, tmp_path):
        db_dir = tmp_path / "explicit_db"
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir, db_dir=db_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.reset()
        try:
            assert env.db_path.parent == db_dir
            assert env._temp_dir is None
        finally:
            await env.cleanup()

    async def test_reset_with_no_db_dir_uses_temp(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.reset()
        try:
            assert env._temp_dir is not None
            assert str(env.db_path).startswith(env._temp_dir)
        finally:
            await env.cleanup()

    async def test_reset_cleans_previous_state(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.reset()
        first_db = env.db_path
        first_temp = env._temp_dir

        await env.reset()
        # New state should be created (old temp dir removed)
        assert env.db_path is not None
        assert env.db_path != first_db or env._temp_dir != first_temp
        assert not Path(first_temp).exists()
        await env.cleanup()


# ---------------------------------------------------------------------------
# AgentWorldModelEnv.get_tools() tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelEnvGetTools:
    async def test_get_tools_empty_before_reset(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        assert env.get_tools() == []

    async def test_get_tools_returns_tools_after_reset(self, agent_world_model_env):
        tools = agent_world_model_env.get_tools()
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# AgentWorldModelEnv.cleanup() tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelEnvCleanup:
    async def test_cleanup_resets_tools(self, agent_world_model_env):
        assert len(agent_world_model_env.get_tools()) > 0
        await agent_world_model_env.cleanup()
        assert agent_world_model_env.get_tools() == []

    async def test_cleanup_clears_db_paths(self, agent_world_model_env):
        assert agent_world_model_env.db_path is not None
        await agent_world_model_env.cleanup()
        assert agent_world_model_env.db_path is None
        assert agent_world_model_env.initial_db_path is None

    async def test_cleanup_removes_temp_dir(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.reset()
        temp_dir = env._temp_dir
        assert temp_dir is not None
        assert Path(temp_dir).exists()
        await env.cleanup()
        assert not Path(temp_dir).exists()

    async def test_cleanup_preserves_explicit_db_dir(self, data_dir, mock_model_factory, tmp_path):
        db_dir = tmp_path / "keep_me"
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir, db_dir=db_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.reset()
        await env.cleanup()
        assert db_dir.exists()

    async def test_cleanup_idempotent(self, agent_world_model_env):
        await agent_world_model_env.cleanup()
        await agent_world_model_env.cleanup()  # Should not raise

    async def test_cleanup_safe_on_fresh_env(self, data_dir, mock_model_factory):
        config = AgentWorldModelEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = AgentWorldModelEnv(model_factory=mock_model_factory, config=config)
        await env.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# AgentWorldModelRewardFunction.compute() tests
# ---------------------------------------------------------------------------


class TestAgentWorldModelRewardFunctionCompute:
    @staticmethod
    def _make_step_result(final_text: str | None = None) -> StepResult:
        messages = []
        if final_text is not None:
            messages = [{"role": "assistant", "content": [{"text": final_text}]}]
        return StepResult(
            observation=Observation(messages=messages),
            termination_reason=TerminationReason.TASK_COMPLETE,
        )

    async def test_compute_reward_complete(self, agent_world_model_env):
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert result.reward == 1.0
        assert result.info.get("result") == "complete"

    async def test_compute_reward_others(self, agent_world_model_env):
        # Delete all users so verifier returns "others"
        conn = sqlite3.connect(str(agent_world_model_env.db_path))
        conn.execute("DELETE FROM users")
        conn.commit()
        conn.close()

        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert result.reward == 0.0
        assert result.info.get("result") == "others"

    async def test_compute_returns_info_dict(self, agent_world_model_env):
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert isinstance(result.info, dict)
        assert "result" in result.info

    async def test_compute_with_missing_verify_fn(self, agent_world_model_env):
        # Patch the verifier to return code without verify_task_completion
        agent_world_model_env.data_loader.get_verifier = lambda s, i: "x = 1"
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert result.reward == 0.0
        assert "error" in result.info

    async def test_compute_with_verifier_exception(self, agent_world_model_env):
        # Patch the verifier to return code that raises
        agent_world_model_env.data_loader.get_verifier = lambda s, i: (
            "def verify_task_completion(**kwargs):\n    raise RuntimeError('boom')\n"
        )
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert result.reward == 0.0
        assert "error" in result.info

    async def test_compute_with_exec_syntax_error(self, agent_world_model_env):
        agent_world_model_env.data_loader.get_verifier = lambda s, i: "def broken(:\n"
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("done")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)
        assert result.reward == 0.0
        assert "error" in result.info

    async def test_compute_passes_correct_paths(self, agent_world_model_env):
        def recording_verifier(s, i):
            return (
                "def verify_task_completion(initial_db_path, final_db_path, final_answer=None):\n"
                "    import json, os\n"
                "    os.environ['_TEST_INITIAL'] = initial_db_path\n"
                "    os.environ['_TEST_FINAL'] = final_db_path\n"
                "    os.environ['_TEST_ANSWER'] = str(final_answer)\n"
                '    return {"result": "complete"}\n'
            )

        agent_world_model_env.data_loader.get_verifier = recording_verifier
        action = Action(message="do something", task_context=TaskContext())
        step_result = self._make_step_result("my answer")
        result = await agent_world_model_env.reward_fn.compute(action, step_result)

        import os

        assert os.environ.get("_TEST_INITIAL") == str(agent_world_model_env.initial_db_path)
        assert os.environ.get("_TEST_FINAL") == str(agent_world_model_env.db_path)
        assert os.environ.get("_TEST_ANSWER") == "my answer"
        assert result.reward == 1.0

        # Clean up env vars
        os.environ.pop("_TEST_INITIAL", None)
        os.environ.pop("_TEST_FINAL", None)
        os.environ.pop("_TEST_ANSWER", None)
