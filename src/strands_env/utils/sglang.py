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

"""Utilities for `SGLang` client caching."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx
from strands_sglang import SGLangClient

_SGLANG_CLIENT_CONFIG = {
    "timeout": 900.0,
    "connect_timeout": 5.0,
    "max_retries": 60,
    "retry_delay": 1.0,
}


@lru_cache(maxsize=None)
def get_cached_client(base_url: str, max_connections: int) -> SGLangClient:
    """Get a shared (cached) `SGLangClient` for connection pooling."""
    return SGLangClient(base_url, max_connections=max_connections, **_SGLANG_CLIENT_CONFIG)


def get_cached_client_from_slime_args(args: Any) -> SGLangClient:
    """Get a shared (cached) `SGLangClient` from `slime`'s training args."""
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    max_connections = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    return get_cached_client(base_url=base_url, max_connections=max_connections)


def clear_clients() -> None:
    """Clear all cached `SGLangClient` instances."""
    get_cached_client.cache_clear()


def get_model_id(base_url: str) -> str:
    """Get the model ID from the SGLang server."""
    response = httpx.get(f"{base_url}/get_model_info", timeout=5)
    response.raise_for_status()
    return response.json()["model_path"]
