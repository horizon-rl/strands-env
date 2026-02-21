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
    `GOOGLE_API_KEY`: Google Custom Search API key
    `GOOGLE_CSE_ID`: Custom Search Engine ID (cx)

Example:
    >>> import os
    >>> os.environ["GOOGLE_API_KEY"] = "your-api-key"
    >>> os.environ["GOOGLE_CSE_ID"] = "your-cse-id"
    >>> from strands_env.tools import GoogleSearchToolkit
    >>> toolkit = GoogleSearchToolkit()
    >>> result = toolkit.google_search("Python programming", top_k=3)
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from strands import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_CONCURRENCY = 10
MAX_RESULTS = 10

_GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchToolkit:
    """Google Custom Search tools for strands agents.

    Uses a single shared ``aiohttp.ClientSession`` (created lazily on first
    call) and an ``asyncio.Semaphore`` to cap concurrent requests to the
    Google API.  Call :meth:`cleanup` when done to close the session.

    Example:
        from strands_env.tools import GoogleSearchToolkit

        toolkit = GoogleSearchToolkit()

        class MyEnv(Environment):
            def __init__(self, ...):
                self.toolkit = GoogleSearchToolkit()

            def get_tools(self):
                return [self.toolkit.google_search]

            async def cleanup(self):
                await self.toolkit.cleanup()
    """

    def __init__(
        self,
        api_key: str | None = None,
        cse_id: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize Google Search Toolkit.

        Args:
            api_key: Google Custom Search API key. Falls back to ``GOOGLE_API_KEY`` env var.
            cse_id: Custom Search Engine ID (cx). Falls back to ``GOOGLE_CSE_ID`` env var.
            timeout: Request timeout in seconds.
            max_concurrency: Maximum concurrent requests to the Google API (ignored if ``semaphore`` is provided).
            semaphore: Shared semaphore for global rate limiting across multiple toolkit instances in RL training.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        if not self.api_key:
            raise ValueError("Google API key required: pass `api_key` or set `GOOGLE_API_KEY` env var")
        if not self.cse_id:
            raise ValueError("Search engine ID required: pass `cse_id` or set `GOOGLE_CSE_ID` env var")

        self._timeout = timeout
        self._semaphore = semaphore or asyncio.Semaphore(max_concurrency)
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    @tool
    async def google_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using Google Custom Search.

        Args:
            query: The search query.
            top_k: Number of results to return (max 10).

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info(f"[google_search] query={query}, top_k={top_k}")

        top_k = min(top_k, MAX_RESULTS)

        params = {
            "cx": self.cse_id,
            "q": query,
            "num": top_k,
        }

        try:
            async with self._semaphore:
                headers = {"X-goog-api-key": self.api_key}
                async with self._get_session().get(_GOOGLE_SEARCH_API_URL, params=params, headers=headers) as response:
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

    async def cleanup(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
