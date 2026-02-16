"""Integration tests for SyntheticEnv with a real SGLang model.

Requires a running SGLang server (default: http://localhost:30000).
Tests exercise the full pipeline: reset → step (LLM + tool use) → reward.
"""

import json

import pytest

from strands_env.core.types import Action, TerminationReason
from strands_env.environments.synthetic_env.env import SyntheticEnv, SyntheticEnvConfig

# ---------------------------------------------------------------------------
# Test scenario data
# ---------------------------------------------------------------------------

SCENARIO_NAME = "integration_test_scenario"

_SCENARIO_DATA = [
    (
        "gen_scenario.jsonl",
        {"name": SCENARIO_NAME, "description": "A simple user management scenario for integration testing."},
    ),
    (
        "gen_tasks.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "tasks": [
                "List all users in the system and report how many there are.",
                "Get the details of the user with ID 1.",
            ],
        },
    ),
    (
        "gen_db.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "db_schema": {
                "tables": [
                    {
                        "name": "users",
                        "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE);",
                        "indexes": ["CREATE INDEX idx_users_email ON users(email);"],
                    }
                ]
            },
            "db_path": "./test.db",
        },
    ),
    (
        "gen_sample.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "tables_count": 1,
            "inserts_count": 3,
            "sample_data": {
                "tables": [
                    {
                        "table_name": "users",
                        "reasoning": "Seed users for testing.",
                        "insert_statements": [
                            "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@test.com');",
                            "INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@test.com');",
                            "INSERT INTO users (id, name, email) VALUES (3, 'Charlie', 'charlie@test.com');",
                        ],
                    }
                ]
            },
        },
    ),
    (
        "gen_spec.jsonl",
        {
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
                                "description": "Return a JSON list of all users in the system.",
                                "operation_id": "list_users",
                                "tags": ["users"],
                                "request_params": {
                                    "limit": {
                                        "type": "integer",
                                        "param_type": "query",
                                        "required": False,
                                        "description": "Maximum number of users to return.",
                                    }
                                },
                                "response": {},
                            },
                            {
                                "path": "/api/users/{user_id}",
                                "method": "GET",
                                "summary": "Get user by ID",
                                "description": "Retrieve a single user by their ID.",
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
        },
    ),
    (
        "gen_envs.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "db_path": "./test.db",
            "full_code": (
                "from fastapi import FastAPI, Query, Path\n"
                "import os, sqlite3\n"
                "\n"
                'DATABASE_PATH = os.getenv("DATABASE_PATH", "sqlite:///test.db")\n'
                'app = FastAPI(title="Test API")\n'
                "\n"
                "def get_db_path():\n"
                "    path = DATABASE_PATH\n"
                '    if path.startswith("sqlite:///"):\n'
                '        path = path[len("sqlite:///"):]\n'
                "    return path\n"
                "\n"
                '@app.get("/api/users")\n'
                "async def list_users(limit: int = Query(100)):\n"
                "    conn = sqlite3.connect(get_db_path())\n"
                "    conn.row_factory = sqlite3.Row\n"
                '    rows = [dict(r) for r in conn.execute("SELECT * FROM users LIMIT ?", (limit,)).fetchall()]\n'
                "    conn.close()\n"
                '    return {"users": rows}\n'
                "\n"
                '@app.get("/api/users/{user_id}")\n'
                "async def get_user(user_id: int = Path(...)):\n"
                "    conn = sqlite3.connect(get_db_path())\n"
                "    conn.row_factory = sqlite3.Row\n"
                '    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()\n'
                "    conn.close()\n"
                '    return {"user": dict(row) if row else None}\n'
            ),
        },
    ),
    (
        "gen_verifier.pure_code.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "task_idx": 0,
            "task": "List all users in the system and report how many there are.",
            "verification": {
                "code": (
                    "def verify_task_completion(initial_db_path: str, final_db_path: str,"
                    " final_answer: str | None = None) -> dict:\n"
                    "    import sqlite3\n"
                    "    conn = sqlite3.connect(final_db_path)\n"
                    '    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]\n'
                    "    conn.close()\n"
                    "    if count >= 3:\n"
                    '        return {"result": "complete"}\n'
                    '    return {"result": "others"}\n'
                ),
                "raw_response": "",
            },
        },
    ),
]

# Also add a verifier for task_idx=1 (get user by ID)
_SCENARIO_DATA.append(
    (
        "gen_verifier.pure_code.jsonl",
        {
            "scenario": SCENARIO_NAME,
            "task_idx": 1,
            "task": "Get the details of the user with ID 1.",
            "verification": {
                "code": (
                    "def verify_task_completion(initial_db_path: str, final_db_path: str,"
                    " final_answer: str | None = None) -> dict:\n"
                    '    if final_answer and "Alice" in final_answer:\n'
                    '        return {"result": "complete"}\n'
                    '    return {"result": "others"}\n'
                ),
                "raw_response": "",
            },
        },
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with all scenario JSONL files."""
    files: dict[str, list[dict]] = {}
    for filename, data in _SCENARIO_DATA:
        files.setdefault(filename, []).append(data)
    for filename, entries in files.items():
        with open(tmp_path / filename, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    return tmp_path


@pytest.fixture
async def synth_env(data_dir, model_factory):
    """Create, reset, and yield a SyntheticEnv; clean up after."""
    config = SyntheticEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
    env = SyntheticEnv(model_factory=model_factory, config=config)
    await env.reset()
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticEnvStep:
    async def test_step_completes(self, synth_env):
        """Agent can call the API tools and complete normally."""
        action = Action(message="List all users in the system and tell me how many there are.")
        result = await synth_env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.messages
        assert result.observation.metrics["message_count"] > 0

    async def test_step_produces_reward(self, synth_env):
        """Reward function runs and attaches a result to the step."""
        action = Action(message="List all users in the system.")
        result = await synth_env.step(action)

        assert result.reward is not None
        assert isinstance(result.reward.reward, float)
        assert result.reward.reward in (0.0, 1.0)
        assert "result" in result.reward.info

    async def test_step_produces_token_observation(self, synth_env):
        """SGLang model produces token-level observations for TITO training."""
        action = Action(message="List all users.")
        result = await synth_env.step(action)

        assert result.observation.tokens is not None
        assert len(result.observation.tokens.token_ids) > 0
        assert result.observation.tokens.prompt_length > 0
        assert len(result.observation.tokens.rollout_token_ids) > 0

    async def test_step_metrics(self, synth_env):
        """Step produces expected metric keys."""
        action = Action(message="How many users are in the system?")
        result = await synth_env.step(action)

        metrics = result.observation.metrics
        assert "message_count" in metrics
        assert "tool_iters" in metrics
        assert "model_calls" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics

    async def test_final_response(self, synth_env):
        """Observation provides final assistant response text."""
        action = Action(message="List all users.")
        result = await synth_env.step(action)

        response = result.observation.final_response
        assert response is not None
        assert len(response) > 0

    async def test_tool_iteration_limit(self, data_dir, model_factory):
        """Environment respects max_tool_iterations."""
        config = SyntheticEnvConfig(scenario=SCENARIO_NAME, task_idx=0, data_dir=data_dir)
        env = SyntheticEnv(model_factory=model_factory, config=config, max_tool_iterations=1)
        await env.reset()
        try:
            action = Action(
                message=(
                    "List all users, then get each user by ID one at a time. "
                    "Make a separate API call for every single user."
                )
            )
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            assert result.observation.metrics["tool_iters"] <= 1
        finally:
            await env.cleanup()

    async def test_reward_complete_when_db_intact(self, synth_env):
        """Verifier returns 'complete' when DB has all 3 users (read-only task)."""
        action = Action(message="List all users in the system and tell me how many there are.")
        result = await synth_env.step(action)

        # The DB is unmodified (read-only tools), so verifier should see 3 users → complete
        assert result.reward is not None
        assert result.reward.reward == 1.0
        assert result.reward.info.get("result") == "complete"

    async def test_get_user_task(self, data_dir, model_factory):
        """Agent can complete a get-user-by-ID task with answer-based verification."""
        config = SyntheticEnvConfig(scenario=SCENARIO_NAME, task_idx=1, data_dir=data_dir)
        env = SyntheticEnv(model_factory=model_factory, config=config)
        await env.reset()
        try:
            action = Action(message="Get the details of the user with ID 1 and tell me their name and email.")
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.TASK_COMPLETE
            assert result.reward is not None
            # Verifier checks if "Alice" appears in final answer
            assert isinstance(result.reward.reward, float)
        finally:
            await env.cleanup()
