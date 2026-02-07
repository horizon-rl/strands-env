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

"""CLI entry point for strands-env."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import click

from strands_env.eval import get_benchmark, list_benchmarks

from .config import EnvConfig, EvalConfig, ModelConfig, SamplingConfig
from .utils import build_model_factory, load_env_hook

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """strands-env: RL environment abstraction for Strands agents."""
    pass


@cli.command("list")
def list_cmd():
    """List registered benchmarks."""
    click.echo("Benchmarks:")
    for name in list_benchmarks():
        click.echo(f"  - {name}")
    if not list_benchmarks():
        click.echo("  (none registered)")


@cli.command("eval")
@click.argument("benchmark")
# Hook file
@click.option(
    "--env",
    "-e",
    "env_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to environment hook file (Python file exporting create_env_factory).",
)
# Model config
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["sglang", "bedrock"]),
    default="sglang",
    help="Model backend.",
)
@click.option(
    "--base-url",
    type=str,
    default="http://localhost:30000",
    help="Base URL for SGLang server.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Model ID. Auto-detected for SGLang if not provided.",
)
@click.option(
    "--tokenizer-path",
    type=str,
    default=None,
    help="Tokenizer path for SGLang. Defaults to model_id if not provided.",
)
@click.option(
    "--region",
    type=str,
    default=None,
    help="AWS region for Bedrock.",
)
@click.option(
    "--profile-name",
    type=str,
    default=None,
    help="AWS profile name for Bedrock.",
)
@click.option(
    "--role-arn",
    type=str,
    default=None,
    help="AWS role ARN for Bedrock (optional).",
)
# Sampling params
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=16384,
    help="Maximum new tokens to generate.",
)
@click.option(
    "--top-p",
    type=float,
    default=1.0,
    help="Top-p sampling parameter.",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-k sampling parameter.",
)
# Environment config
@click.option(
    "--system-prompt",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to system prompt file (overrides environment default).",
)
@click.option(
    "--max-tool-iterations",
    type=int,
    default=10,
    help="Maximum tool iterations per step.",
)
# Eval settings
@click.option(
    "--n-samples-per-prompt",
    type=int,
    default=1,
    help="Number of samples per prompt (for pass@k).",
)
@click.option(
    "--max-concurrency",
    type=int,
    default=10,
    help="Maximum concurrent evaluations.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. Defaults to {benchmark}_eval/.",
)
@click.option(
    "--save-interval",
    type=int,
    default=10,
    help="Save results every N samples.",
)
@click.option(
    "--keep-tokens",
    is_flag=True,
    default=False,
    help="Keep token-level observations in results.",
)
# Debug
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def eval_cmd(
    benchmark: str,
    env_path: Path,
    # Model
    backend: Literal["sglang", "bedrock"],
    base_url: str,
    model_id: str | None,
    tokenizer_path: str | None,
    region: str | None,
    profile_name: str | None,
    role_arn: str | None,
    # Sampling
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
    # Environment
    system_prompt: Path | None,
    max_tool_iterations: int,
    # Eval
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: Path,
    save_interval: int,
    keep_tokens: bool,
    debug: bool,
):
    """Run benchmark evaluation.

    BENCHMARK is the name of the benchmark (e.g., 'aime').

    Example:
        strands-env eval aime --env my_env.py --backend sglang --n-samples 8
    """
    # Setup logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate benchmark first (fail fast)
    try:
        evaluator_cls = get_benchmark(benchmark)
    except KeyError as e:
        raise click.ClickException(str(e))

    # Load hook file (validate before building model factory)
    env_factory_creator = load_env_hook(env_path)

    # Build configs
    sampling_config = SamplingConfig(
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    model_config = ModelConfig(
        backend=backend,
        base_url=base_url,
        model_id=model_id,
        tokenizer_path=tokenizer_path,
        region=region,
        profile_name=profile_name,
        role_arn=role_arn,
        sampling=sampling_config,
    )
    env_config = EnvConfig(
        system_prompt_path=system_prompt,
        max_tool_iterations=max_tool_iterations,
        verbose=False,  # Always False for eval
    )
    eval_config = EvalConfig(
        n_samples_per_prompt=n_samples_per_prompt,
        max_concurrency=max_concurrency,
        output_dir=output,
        save_interval=save_interval,
        keep_tokens=keep_tokens,
    )

    # Build model factory
    model_factory = build_model_factory(model_config, eval_config.max_concurrency)

    # Create env_factory from hook
    env_factory = env_factory_creator(model_factory, env_config)

    # Get output paths based on benchmark name
    output_dir = eval_config.get_output_dir(benchmark)
    results_path = eval_config.get_results_path(benchmark)
    metrics_path = eval_config.get_metrics_path(benchmark)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator
    evaluator = evaluator_cls(
        env_factory=env_factory,
        max_concurrency=eval_config.max_concurrency,
        n_samples_per_prompt=eval_config.n_samples_per_prompt,
        output_path=results_path,
        save_interval=eval_config.save_interval,
        keep_tokens=eval_config.keep_tokens,
    )

    # Load dataset
    actions = evaluator.load_dataset()

    # Run evaluation
    click.echo(f"Running {benchmark} evaluation with {env_path}")
    click.echo(f"  Backend: {backend}, Model: {model_id or '(auto-detect)'}")
    click.echo(f"  Samples per prompt: {n_samples_per_prompt}, Concurrency: {max_concurrency}")
    click.echo(f"  Output directory: {output_dir}")

    results = asyncio.run(evaluator.run(actions))
    metrics = evaluator.compute_metrics(results)

    # Save metrics to JSON
    import json

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Saved metrics to {metrics_path}")


def main():
    cli()


if __name__ == "__main__":
    main()
