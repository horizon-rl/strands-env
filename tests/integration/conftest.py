"""Shared fixtures for integration tests.

All tests in this directory require a running SGLang server.
The model ID is auto-detected from the server via /get_model_info.
Tests are skipped automatically if the server is not reachable.

Configuration (priority: CLI > env var > default):
    pytest --sglang-base-url=http://localhost:30000
    SGLANG_BASE_URL=http://... pytest tests/integration/
"""

import httpx
import pytest
from strands_sglang import SGLangClient
from transformers import AutoTokenizer

from strands_env.core.models import DEFAULT_SAMPLING_PARAMS, sglang_model_factory

# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def sglang_base_url(request):
    """Get SGLang server URL from pytest config."""
    return request.config.getoption("--sglang-base-url")


@pytest.fixture(scope="session")
def sglang_client(sglang_base_url):
    """Shared SGLang client for connection pooling. Skips all tests if server is unreachable."""
    try:
        response = httpx.get(f"{sglang_base_url}/health", timeout=5)
        healthy = response.status_code == 200
    except httpx.HTTPError:
        healthy = False
    if not healthy:
        pytest.skip(f"SGLang server not reachable at {sglang_base_url}")
    return SGLangClient(sglang_base_url)


@pytest.fixture(scope="session")
def sglang_model_id(sglang_base_url, sglang_client):
    """Auto-detect model ID from the running SGLang server."""
    response = httpx.get(f"{sglang_base_url}/get_model_info", timeout=5)
    response.raise_for_status()
    return response.json()["model_path"]


@pytest.fixture(scope="session")
def tokenizer(sglang_model_id):
    """Load tokenizer for the detected model."""
    return AutoTokenizer.from_pretrained(sglang_model_id)


@pytest.fixture
def model_factory(tokenizer, sglang_client, sglang_model_id):
    """Model factory for Environment integration tests."""
    return sglang_model_factory(
        model_id=sglang_model_id,
        tokenizer=tokenizer,
        client=sglang_client,
        sampling_params=DEFAULT_SAMPLING_PARAMS,
    )
