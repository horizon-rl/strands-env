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

"""Configuration dataclasses for CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SamplingConfig:
    """Sampling parameters for model generation."""

    temperature: float = 0.7
    max_new_tokens: int = 16384
    top_p: float = 0.95
    top_k: int | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        d = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            d["top_k"] = self.top_k
        return d


@dataclass
class ModelConfig:
    """Model configuration."""

    backend: Literal["sglang", "bedrock"] = "sglang"

    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None  # Auto-detected if None
    tool_parser: str | None = None  # Parser name or path to hook file

    # Bedrock
    model_id: str | None = None
    region: str | None = None
    profile_name: str | None = None  # AWS profile name
    role_arn: str | None = None  # For role assumption

    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # SGLang TITO options
    return_routed_experts: bool = False  # Record MoE routing decisions for routing replay

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "backend": self.backend,
            "base_url": self.base_url,
            "tokenizer_path": self.tokenizer_path,
            "tool_parser": self.tool_parser,
            "model_id": self.model_id,
            "region": self.region,
            "profile_name": self.profile_name,
            "role_arn": self.role_arn,
            "sampling": self.sampling.to_dict(),
            "return_routed_experts": self.return_routed_experts,
        }


@dataclass
class EnvConfig:
    """Environment configuration."""

    system_prompt_path: Path | None = None
    max_tool_iterations: int = 10
    verbose: bool = False

    @property
    def system_prompt(self) -> str | None:
        """Load system prompt from file if path is set."""
        if self.system_prompt_path is None:
            return None
        return self.system_prompt_path.read_text()

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "system_prompt": self.system_prompt,  # Save actual content for reproducibility
            "max_tool_iterations": self.max_tool_iterations,
        }


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    n_samples_per_prompt: int = 1
    max_concurrency: int = 10
    output_dir: Path | None = None  # Defaults to {benchmark}_eval/
    save_interval: int = 10
    keep_tokens: bool = False

    def get_output_dir(self, benchmark_name: str) -> Path:
        """Get output directory, using default if not set."""
        if self.output_dir is not None:
            return self.output_dir
        return Path(f"{benchmark_name}_eval")

    def get_results_path(self, benchmark_name: str) -> Path:
        """Get path for results JSONL file."""
        return self.get_output_dir(benchmark_name) / "results.jsonl"

    def get_metrics_path(self, benchmark_name: str) -> Path:
        """Get path for metrics JSON file."""
        return self.get_output_dir(benchmark_name) / "metrics.json"

    def get_config_path(self, benchmark_name: str) -> Path:
        """Get path for config JSON file."""
        return self.get_output_dir(benchmark_name) / "config.json"

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "max_concurrency": self.max_concurrency,
            "save_interval": self.save_interval,
            "keep_tokens": self.keep_tokens,
        }
