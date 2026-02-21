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

"""Google Custom Search toolkit.

Reads credentials from environment variables by default:
    GOOGLE_API_KEY: Google Custom Search API key
    GOOGLE_CSE_ID:  Custom Search Engine ID (cx)

Example:
    >>> import os
    >>> os.environ["GOOGLE_API_KEY"] = "your-api-key"
    >>> os.environ["GOOGLE_CSE_ID"] = "your-cse-id"
    >>> from strands_env.tools import GoogleSearchToolkit
    >>> toolkit = GoogleSearchToolkit()
    >>> result = toolkit.google_search("Python programming", num_results=3)
"""

from __future__ import annotations

import logging
import os

import aiohttp
from strands import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
MAX_RESULTS = 10

_GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchToolkit:
    """Google Custom Search tools for strands agents.

    Provides a `google_search` tool that searches the web using Google's
    Custom Search API and returns structured results with titles, URLs,
    and snippets.

    Example:
        from strands_env.tools import GoogleSearchToolkit

        toolkit = GoogleSearchToolkit()

        class MyEnv(Environment):
            def __init__(self, ...):
                self.toolkit = GoogleSearchToolkit()

            def get_tools(self):
                return [self.toolkit.google_search]
    """

    def __init__(
        self,
        api_key: str | None = None,
        search_engine_id: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize Google Search Toolkit.

        Args:
            api_key: Google Custom Search API key. Falls back to `GOOGLE_API_KEY` env var.
            search_engine_id: Custom Search Engine ID (cx). Falls back to `GOOGLE_CSE_ID` env var.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("GOOGLE_CSE_ID")
        self.timeout = timeout

    @tool
    async def google_search(self, query: str, top_k: int = 5) -> str:
        f"""Search the web using Google Custom Search.

        Args:
            query: The search query.
            top_k: Number of results to return (max {MAX_RESULTS}).

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info(f"[google_search] query={query}, num_results={top_k}")

        top_k = min(top_k, MAX_RESULTS)

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": top_k,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _GOOGLE_SEARCH_API_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            items = data.get("items", [])
            if not items:
                return "No results found."

            lines = []
            for i, item in enumerate(items, 1):
                title = item.get("title") or "No title available."
                url = item.get("link") or "No URL available."
                snippet = item.get("snippet") or "No snippet available."
                lines.append(f"{i}. {title} ({url}):\n{snippet}")
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"[google_search] error: {e}")
            return f"Search failed: {e}."
