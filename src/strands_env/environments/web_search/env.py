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

"""Web search environment with web search and web scraping tools."""

import asyncio
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.tools.web_search import WebSearchToolkit


@dataclass
class SearchConfig:
    timeout: int = 10
    max_concurrency: int = 10
    semaphore: asyncio.Semaphore | None = None
    blocked_domains: list[str] | None = None
    provider: Literal["serper", "google"] = "serper"

    def _search_tool_name(self) -> str:
        return f"{self.provider}_search"


@dataclass
class ScrapeConfig: ...  # TODO: implement scrape config


class WebSearchEnv(Environment):
    """Web search environment with pluggable search providers."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = 3,
        max_tool_calls: int | None = 10,
        verbose: bool = False,
        search_config: SearchConfig = SearchConfig(),
        scrape_config: ScrapeConfig = ScrapeConfig(),
    ):
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=reward_fn,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )
        self.search_toolkit = WebSearchToolkit(**dataclasses.asdict(search_config))

    @override
    def get_tools(self):
        return [getattr(self.search_toolkit, self._search_tool_name())]
