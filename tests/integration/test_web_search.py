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

"""Integration tests for WebSearchToolkit.

Requires provider credentials via env vars:
    Serper: SERPER_API_KEY
    Google: GOOGLE_API_KEY + GOOGLE_CSE_ID (deprecated in 2027)

Tests for each provider are skipped if the required env vars are not set.
"""

import os
import re

import pytest

from strands_env.tools import WebSearchToolkit

_AUTH_ERROR_RE = re.compile(r"Search failed: (401|403)")


def _skip_on_auth_error(result: str) -> str:
    """Skip the test if the search result indicates invalid credentials."""
    if _AUTH_ERROR_RE.search(result):
        pytest.skip(f"Credentials rejected by API: {result}")
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def toolkit():
    """WebSearchToolkit with default config; cleans up session after test."""
    tk = WebSearchToolkit()
    yield tk
    await tk.cleanup()


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------

serper_available = pytest.mark.skipif(
    not os.getenv("SERPER_API_KEY"),
    reason="SERPER_API_KEY not set",
)


@serper_available
class TestSerperSearch:
    async def test_basic_search(self, toolkit):
        result = _skip_on_auth_error(await toolkit.serper_search("Python programming", top_k=3))
        assert "Search failed" not in result
        assert "1." in result

    async def test_blocked_domains(self):
        tk = WebSearchToolkit(blocked_domains=["example.com"])
        result = _skip_on_auth_error(await tk.serper_search("example.com test", top_k=3))
        assert "Search failed" not in result
        await tk.cleanup()


# ---------------------------------------------------------------------------
# Google Custom Search
# ---------------------------------------------------------------------------

google_available = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    reason="GOOGLE_API_KEY and GOOGLE_CSE_ID not set",
)


@google_available
class TestGoogleSearch:
    async def test_basic_search(self, toolkit):
        result = _skip_on_auth_error(await toolkit.google_search("Python programming", top_k=3))
        assert "Search failed" not in result
        assert "1." in result

    async def test_no_results(self, toolkit):
        result = _skip_on_auth_error(await toolkit.google_search("asdkjhqwelkjhzxcvmnb1234567890", top_k=1))
        assert "No results found" in result or "1." in result

    async def test_blocked_domains(self):
        tk = WebSearchToolkit(blocked_domains=["example.com"])
        result = _skip_on_auth_error(await tk.google_search("example.com test", top_k=3))
        assert "Search failed" not in result
        await tk.cleanup()


# ---------------------------------------------------------------------------
# requires_env validation
# ---------------------------------------------------------------------------


class TestRequiresEnv:
    async def test_google_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        tk = WebSearchToolkit()
        result = await tk.google_search("test")
        assert "missing required environment variable" in result

    async def test_serper_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        tk = WebSearchToolkit()
        result = await tk.serper_search("test")
        assert "SERPER_API_KEY" in result
