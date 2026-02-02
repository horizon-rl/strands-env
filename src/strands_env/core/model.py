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

"""Base model configuration and sampling parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field
from strands.models import Model


class ClientParams(BaseModel):
    """Client connection parameters for model backends.

    Provides sensible defaults for general use. Override as needed for your use case:
    - Training: Higher max_connections and max_retries
    - Remote APIs: Bounded timeout

    Example:
        >>> params = ClientParams()  # Use defaults
        >>> params = ClientParams(max_connections=1000, max_retries=60)  # Training
        >>> params = ClientParams(timeout=300.0, max_retries=5)  # Remote API
    """

    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds, or None for infinite.",
    )
    connect_timeout: float = Field(
        default=5.0,
        description="TCP connection timeout in seconds.",
    )
    max_connections: int = Field(
        default=100,
        description="Maximum concurrent connections.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts.",
    )


class SamplingParams(BaseModel):
    """Sampling parameters for model policy generation.

    Defines **commonly used sampling parameters**. Additional backend-specific
    parameters can be passed as extra kwargs and will be forwarded to the model.

    Example:
        >>> params = SamplingParams(temperature=0.7, top_p=0.9)
        >>> params = SamplingParams(temperature=0.7, top_k=50)  # extra param
        >>> params.to_dict()
        {'max_new_tokens': 4096, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 50}
    """

    model_config = ConfigDict(extra="allow")

    max_new_tokens: int = Field(
        default=4096,
        description="Maximum new tokens to generate per response.",
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature. Higher = more random.",
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling parameter.",
    )

    def to_dict(self, *, max_tokens_key: str = "max_new_tokens") -> dict:
        """Convert to dict including extra fields.

        Args:
            max_tokens_key: Key name for max tokens. Use "max_new_tokens" for SGLang native,
                "max_tokens" for OpenAI-compatible APIs.

        Returns:
            Dict with all sampling parameters.
        """
        result = {**self.model_dump(), **self.model_extra}
        if max_tokens_key != "max_new_tokens":
            result[max_tokens_key] = result.pop("max_new_tokens")
        return result


class ModelConfig(BaseModel, ABC):
    """Base class for model backend configurations.

    All backend configs inherit from this class and must implement create_model().
    Sampling parameters are shared across all backends.

    Example:
        >>> class MyBackendConfig(ModelConfig):
        ...     custom_param: str
        ...
        ...     def create_model(self) -> Model:
        ...         return MyModel(params=self.sampling.to_dict())
    """

    sampling: SamplingParams = Field(
        default_factory=SamplingParams,
        description="Sampling parameters for generation.",
    )

    @abstractmethod
    def create_model(self) -> Model:
        """Create a model instance for this backend."""
        ...
