# Agent World Model Environment

SQLite-backed tool-use environment from [AgentWorldModel](https://github.com/Snowflake-Labs/agent-world-model) datasets. Dynamically generates API tools from JSONL data folders and computes rewards using per-task verifiers.

## Setup

Requires `fastapi` and `starlette` (included in `strands-env` dependencies).

Data folder must follow the AWM-format with these JSONL files:
- `gen_scenario.jsonl` — scenario descriptions
- `gen_tasks.jsonl` — task lists per scenario
- `gen_db.jsonl` — database schemas
- `gen_sample.jsonl` — seed data
- `gen_spec.jsonl` — API specifications
- `gen_envs.jsonl` — FastAPI implementation code
- `gen_verifier.pure_code.jsonl` — verification functions

## Usage

```python
from strands_env.environments.agent_world_model import SyntheticEnv, SyntheticEnvConfig

config = SyntheticEnvConfig(
    scenario="e_commerce_33",
    task_idx=0,
    data_dir="/path/to/AgentWorldModel-1K",
)
env = SyntheticEnv(model_factory=model_factory, config=config)
await env.reset()
result = await env.step(action)
await env.cleanup()
```

## Tools

Dynamically generated from the scenario's API specification. Each endpoint becomes a tool the agent can call (e.g., `list_users`, `create_order`). Tools route through an in-process FastAPI TestClient backed by SQLite.

## Reward

Built-in `SyntheticEnvRewardFunction` executes the scenario's verifier code against the initial and final database states. Returns `1.0` for "complete", `0.0` otherwise.

## System Prompt

The agent is instructed to interact with the platform through its API tools, calling them with the required parameters to accomplish the given task.
