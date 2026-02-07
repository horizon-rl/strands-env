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
# Using a registered benchmark
strands-env eval <benchmark> --env <hook_file> [options]

# Using a custom evaluator hook
strands-env eval --evaluator <evaluator_file> --env <hook_file> [options]
```

**Required arguments:**
- `<benchmark>` - Benchmark name (e.g., `aime-2024`, `aime-2025`), OR
- `--evaluator` - Path to evaluator hook file (mutually exclusive with benchmark)
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
- `--temperature` - Sampling temperature (default: 0.7)
- `--max-tokens` - Maximum new tokens (default: 16384)
- `--top-p` - Top-p sampling (default: 0.95)
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
# Using registered benchmark
strands-env eval aime-2024 --env examples/envs/calculator_env.py --backend sglang

# Using custom evaluator hook (see examples/evaluators/)
strands-env eval --evaluator examples/evaluators/simple_math_evaluator.py \
    --env examples/envs/calculator_env.py --backend sglang

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

For custom benchmarks, subclass `Evaluator`. You can either register it with `@register` or use an evaluator hook file.

### Evaluator Hook File

Create a Python file that exports `EvaluatorClass`:

```python
# my_evaluator.py
from collections.abc import Iterable

from strands_env.core import Action, TaskContext
from strands_env.eval import Evaluator

class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self) -> Iterable[Action]:
        for item in load_my_data():
            yield Action(
                message=item["prompt"],
                task_context=TaskContext(
                    id=item["id"],
                    ground_truth=item["answer"],
                ),
            )

EvaluatorClass = MyEvaluator
```

Then run:
```bash
strands-env eval --evaluator my_evaluator.py --env my_env.py --backend sglang
```

### Registered Evaluator

Alternatively, use `@register` to make it available by name:

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

Override `get_metric_fns()` to customize metrics:

```python
from functools import partial

from strands_env.eval import Evaluator, compute_pass_at_k

class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self):
        ...

    def get_metric_fns(self):
        # Include default pass@k plus custom metrics
        return [
            partial(compute_pass_at_k, k_values=[1, 5, 10], reward_threshold=1.0),
            self.my_custom_metric,
        ]

    def my_custom_metric(self, results: dict) -> dict:
        return {"my_metric": compute_something(results)}
```

## Output Files

Evaluation results are saved to the output directory:

```
{benchmark}_eval/
├── config.json      # CLI configuration for reproducibility
├── results.jsonl    # Per-sample results (action, step_result, reward)
└── metrics.json     # Aggregated metrics (pass@k, etc.)
```

The evaluator supports checkpointing and resume - if interrupted, it will skip already-completed samples on restart.
