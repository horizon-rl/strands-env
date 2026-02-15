# SyntheticEvaluator Design

## Problem

Test the synthetic_env implementation by running all 1000 AWM scenarios concurrently with a real LLM, collecting full agent interaction trajectories (messages, tool calls, tool results, reward) as JSONL output.

## Goals

1. **Correctness validation** — verify DB creation, tool routing, and reward computation work across all 1000 scenarios
2. **Trajectory collection** — generate a dataset of agent interaction traces for analysis/training

## Approach: SyntheticEvaluator (Evaluator subclass)

Reuse the existing `Evaluator` framework which provides concurrency control, checkpointing, resume, tqdm progress, and JSONL output.

### Architecture

```
Evaluator.run(actions)
  -> for each action (scenario+task), concurrently:
     -> evaluate_sample(action)
        -> env_factory(action) -> SyntheticEnv(model_factory, config)
        -> env.reset()    [creates SQLite DB, generates API tools via TestClient]
        -> env.step()     [real LLM agent calls tools, reward computed via verifier]
        -> env.cleanup()  [closes TestClient, removes temp files]
        -> EvalSample saved to JSONL checkpoint
```

### SyntheticEvaluator class

- Registered as `@register_eval("synthetic")`
- Located at `src/strands_env/eval/benchmarks/synthetic.py`

#### Constructor parameters

- `env_factory` — standard Evaluator parameter (async callable)
- `data_dir: Path` — path to AWM-format data folder
- `scenarios: list[str] | None` — optional filter for running a subset of scenarios
- `max_tasks_per_scenario: int | None` — optional limit on tasks per scenario (e.g., 1 for smoke test)
- Standard Evaluator params: `max_concurrency`, `n_samples_per_prompt`, `output_path`, etc.

#### load_dataset()

```python
def load_dataset(self) -> Iterable[Action]:
    loader = AWMDataLoader(self.data_dir)
    scenarios = self.scenarios or loader.list_scenarios()
    for scenario in scenarios:
        tasks = loader.get_tasks(scenario)
        limit = self.max_tasks_per_scenario or len(tasks)
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

#### env_factory wiring

The evaluator constructs an `env_factory` closure that extracts `scenario` and `task_idx` from the action's `TaskContext` extra fields:

```python
async def env_factory(action: Action) -> SyntheticEnv:
    ctx = action.task_context
    config = SyntheticEnvConfig(
        scenario=ctx.scenario,
        task_idx=ctx.task_idx,
        data_dir=data_dir,
    )
    return SyntheticEnv(model_factory=model_factory, config=config)
```

### Concurrency safety

- Each sample gets its own `SyntheticEnv` instance via `env_factory`
- Each env creates its own temp directory for SQLite files
- Each env has its own TestClient wrapping an independent FastAPI app instance
- No shared mutable state between concurrent samples

### Output format

Standard Evaluator JSONL checkpoint. Each line:

```json
{
  "prompt_id": "e_commerce_33_0",
  "action": {
    "message": "Find all products under $50...",
    "task_context": {
      "id": "e_commerce_33_0_0",
      "ground_truth": null,
      "scenario": "e_commerce_33",
      "task_idx": 0
    }
  },
  "step_result": {
    "observation": {
      "messages": [...],
      "tokens": null,
      "metrics": {...}
    },
    "reward": {"reward": 1.0, "info": {...}},
    "termination_reason": "task_complete"
  }
}
```

### CLI integration

Runnable via:
```bash
strands-env eval run synthetic \
  --data-dir /path/to/AWM-1K \
  --model sglang \
  --base-url http://localhost:30000 \
  --max-concurrency 10
```

Additional CLI options for `--scenarios` filter and `--max-tasks-per-scenario`.

## File changes

### New files

| File | Purpose |
|------|---------|
| `src/strands_env/eval/benchmarks/synthetic.py` | `SyntheticEvaluator` class |
| `tests/unit/test_synthetic_evaluator.py` | Unit tests for load_dataset, env_factory, filtering |

### Modified files

| File | Change |
|------|--------|
| `src/strands_env/cli/eval.py` | Add `--data-dir`, `--scenarios`, `--max-tasks-per-scenario` options |
| `src/strands_env/cli/config.py` | Add `data_dir` field to `EnvConfig` if needed |

### Unchanged files

- `SyntheticEnv`, `AWMDataLoader`, tools, db, reward — all work as-is
- `Evaluator` base class — SyntheticEvaluator just subclasses it
- Registry — auto-discovered via `@register_eval`

## Test plan

Unit tests with mocked AWM data (inline JSONL fixtures):
- `load_dataset()` produces correct Actions with scenario/task_idx in TaskContext
- `scenarios` filter restricts output
- `max_tasks_per_scenario` limit works
- env_factory creates SyntheticEnv with correct config from action context
