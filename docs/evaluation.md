# Evaluation Guide

This guide covers running benchmark evaluations with `strands-env`.

## CLI Reference

The `strands-env` CLI provides commands for running benchmark evaluations.

### List Benchmarks

```bash
strands-env list
```

### Run Evaluation

```bash
strands-env eval <benchmark> --env <hook_file> [options]
```

**Required arguments:**
- `<benchmark>` - Benchmark name (e.g., `aime-2024`, `aime-2025`)
- `--env`, `-e` - Path to environment hook file

**Model options:**
- `--backend`, `-b` - Model backend: `sglang` (default) or `bedrock`
- `--base-url` - SGLang server URL (default: `http://localhost:30000`)
- `--model-id` - Model ID (auto-detected for SGLang, required for Bedrock)
- `--tokenizer-path` - Tokenizer path (defaults to model_id)
- `--region` - AWS region for Bedrock
- `--profile-name` - AWS profile name for Bedrock
- `--role-arn` - AWS role ARN to assume for Bedrock

**Sampling options:**
- `--temperature` - Sampling temperature (default: 1.0)
- `--max-tokens` - Maximum new tokens (default: 16384)
- `--top-p` - Top-p sampling (default: 1.0)
- `--top-k` - Top-k sampling

**Evaluation options:**
- `--n-samples-per-prompt` - Samples per prompt for pass@k (default: 1)
- `--max-concurrency` - Maximum concurrent evaluations (default: 10)
- `--output`, `-o` - Output directory (default: `{benchmark}_eval/`)
- `--save-interval` - Save results every N samples (default: 10)
- `--keep-tokens` - Keep token-level observations in results

**Other options:**
- `--system-prompt` - Path to system prompt file
- `--max-tool-iterations` - Maximum tool iterations per step (default: 10)
- `--debug` - Enable debug logging

### Examples

```bash
# Basic evaluation with SGLang
strands-env eval aime-2024 --env examples/envs/calculator_env.py --backend sglang

# With Bedrock and assumed role
strands-env eval aime-2024 --env examples/envs/code_sandbox_env.py \
    --backend bedrock \
    --model-id us.anthropic.claude-sonnet-4-20250514 \
    --role-arn ...

# Pass@8 evaluation with high concurrency
strands-env eval aime-2024 --env examples/envs/calculator_env.py \
    --backend sglang \
    --n-samples-per-prompt 8 \
    --max-concurrency 30
```

## Hook Files

Environment hook files define how environments are created for each evaluation sample. They must export a `create_env_factory` function.

### Structure

```python
from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory

def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create an async environment factory.

    Args:
        model_factory: Factory for creating model instances.
        env_config: Environment configuration from CLI.

    Returns:
        Async function that creates an Environment for each action.
    """
    async def env_factory(action):
        return YourEnvironment(
            model_factory=model_factory,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
        )

    return env_factory
```

### Example: Calculator Environment

```python
# examples/envs/calculator_env.py
from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.calculator import CalculatorEnv
from strands_env.rewards.math_reward import MathRewardFunction

def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    reward_fn = MathRewardFunction()

    async def env_factory(_action):
        return CalculatorEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
        )

    return env_factory
```

### Example: Code Sandbox Environment

```python
# examples/envs/code_sandbox_env.py
from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.code_sandbox import CodeMode, CodeSandboxEnv
from strands_env.rewards.math_reward import MathRewardFunction

def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    reward_fn = MathRewardFunction()

    async def env_factory(_action):
        return CodeSandboxEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
            mode=CodeMode.CODE,
        )

    return env_factory
```

## Custom Evaluators

For custom benchmarks, subclass `Evaluator` and use the `@register` decorator.

### Creating an Evaluator

```python
from collections.abc import Iterable

from strands_env.core import Action, TaskContext
from strands_env.eval import Evaluator, register

@register("my-benchmark")
class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset and return Actions for evaluation."""
        for item in load_my_data():
            yield Action(
                message=item["prompt"],
                task_context=TaskContext(
                    id=item["id"],
                    ground_truth=item["answer"],
                ),
            )
```

### Programmatic Usage

```python
async def run_evaluation():
    evaluator = MyEvaluator(
        env_factory=env_factory,
        n_samples_per_prompt=8,
        max_concurrency=30,
        output_path="results.jsonl",
        keep_tokens=False,
    )

    actions = evaluator.load_dataset()
    results = await evaluator.run(actions)
    metrics = evaluator.compute_metrics(results)
    # {"pass@1": 0.75, "pass@8": 0.95, ...}
```

### Custom Metrics

```python
from strands_env.eval import MetricFn, pass_at_k_metric

def my_custom_metric(results: dict) -> dict:
    """Custom metric function."""
    return {"my_metric": compute_something(results)}

evaluator = MyEvaluator(
    env_factory=env_factory,
    metric_fns=[pass_at_k_metric, my_custom_metric],
)
```

## Output Files

Evaluation results are saved to the output directory:

```
{benchmark}_eval/
├── results.jsonl    # Per-sample results (action, step_result, reward)
└── metrics.json     # Aggregated metrics (pass@k, etc.)
```

The evaluator supports checkpointing and resume - if interrupted, it will skip already-completed samples on restart.
