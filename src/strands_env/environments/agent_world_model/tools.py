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

"""Dynamic tool generation from Agent World Model API specifications and FastAPI implementation code.

Executes the generated FastAPI code in-process and creates strands tools that route
API calls through a synchronous TestClient for ASGI.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from strands.tools.tools import PythonAgentTool

logger = logging.getLogger(__name__)

# Type mapping from API spec types to JSON Schema types
_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "object": "object",
}


def _map_json_type(api_type: str) -> str:
    """Map an API spec type string to a JSON Schema type."""
    base = api_type.lower().split("[")[0].strip()
    return _TYPE_MAP.get(base, "string")


def _build_tool_spec(endpoint: dict) -> dict:
    """Build a strands ToolSpec dict from an API endpoint specification."""
    op_id = endpoint["operation_id"]
    summary = endpoint.get("summary", op_id)
    description = endpoint.get("description", summary)

    properties: dict[str, dict] = {}
    required: list[str] = []

    params = endpoint.get("request_params", {})
    for param_name, param_spec in params.items():
        prop: dict[str, Any] = {
            "type": _map_json_type(param_spec.get("type", "string")),
        }
        if param_spec.get("description"):
            prop["description"] = param_spec["description"]

        properties[param_name] = prop

        if param_spec.get("required", False):
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return {
        "name": op_id,
        "description": f"{summary}\n\n{description}" if description != summary else summary,
        "inputSchema": {"json": schema},
    }


def _build_request_parts(endpoint: dict, tool_input: dict) -> tuple[dict, dict, dict | None]:
    """Separate tool input into path params, query params, and request body."""
    params_spec = endpoint.get("request_params", {})
    path_params: dict[str, Any] = {}
    query_params: dict[str, Any] = {}
    body_params: dict[str, Any] = {}

    for param_name, value in tool_input.items():
        if value is None:
            continue
        spec = params_spec.get(param_name, {})
        param_type = spec.get("param_type", "query")

        if param_type == "path":
            path_params[param_name] = value
        elif param_type == "body":
            body_params[param_name] = value
        else:
            query_params[param_name] = value

    body = body_params if body_params else None
    return path_params, query_params, body


def _resolve_path(path_template: str, path_params: dict) -> str:
    """Resolve path parameters in a URL template like ``/api/items/{item_id}``."""
    resolved = path_template
    for key, value in path_params.items():
        resolved = resolved.replace(f"{{{key}}}", str(value))
    return resolved


def _create_fastapi_app(full_code: str, db_path: Path | str) -> Any:
    """Execute the generated FastAPI code and return the ``app`` object.

    Injects DATABASE_PATH into both the exec namespace and os.environ atomically
    to support generated code that reads the value via either os.getenv() or as a
    module-level variable. The os.environ mutation is scoped to the exec() call.
    This is safe under asyncio (cooperative scheduling) but would need a lock for
    thread-based concurrency.
    """
    db_url = f"sqlite:///{db_path}"

    # Inject DATABASE_PATH into the namespace so the exec'd code can access it
    # as a module-level variable without relying on os.environ.
    namespace: dict[str, Any] = {"DATABASE_PATH": db_url}

    # Also set os.environ temporarily for code that uses os.getenv("DATABASE_PATH").
    # This is safe under asyncio since exec() is synchronous and won't yield.
    original_env = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = db_url

    try:
        # Strip the __main__ block to avoid running uvicorn
        code = re.sub(r'if\s+__name__\s*==\s*["\']__main__["\'].*', "", full_code, flags=re.DOTALL)
        exec(code, namespace)  # noqa: S102

        # Rebuild Pydantic models defined in the exec'd namespace.
        # Pydantic v2 defers type resolution, and models created via exec()
        # may have unresolved forward references that need model_rebuild().
        try:
            from pydantic import BaseModel as _PydanticBaseModel

            for obj in namespace.values():
                if isinstance(obj, type) and issubclass(obj, _PydanticBaseModel) and obj is not _PydanticBaseModel:
                    try:
                        obj.model_rebuild(_types_namespace=namespace)
                    except Exception:
                        pass  # Some models may not need rebuilding
        except ImportError:
            pass

        app = namespace.get("app")
        if app is None:
            raise RuntimeError("FastAPI 'app' object not found in generated code")
        return app
    finally:
        if original_env is not None:
            os.environ["DATABASE_PATH"] = original_env
        else:
            os.environ.pop("DATABASE_PATH", None)


def create_api_tools(
    full_code: str,
    api_spec: dict,
    db_path: Path | str,
) -> tuple[list[PythonAgentTool], Any]:
    """Create strands tools from an API spec and FastAPI implementation code.

    Executes the generated code to get the FastAPI app, creates a synchronous TestClient,
    and generates a ``PythonAgentTool`` for each API endpoint.

    Requires ``fastapi`` to be installed (see ``requirements.txt``).

    Args:
        full_code: Complete Python source from ``gen_envs.jsonl``.
        api_spec: API specification dict with ``api_groups`` -> ``endpoints``.
        db_path: Path to the SQLite database file.

    Returns:
        Tuple of (list of tools, TestClient). The client should be closed on cleanup.
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError as e:
        raise ImportError(
            "fastapi is required for the agent_world_model environment. Install it with: pip install fastapi"
        ) from e

    app = _create_fastapi_app(full_code, db_path)
    client = TestClient(app, raise_server_exceptions=False)

    tools: list[PythonAgentTool] = []

    for group in api_spec.get("api_groups", []):
        for endpoint in group.get("endpoints", []):
            op_id = endpoint.get("operation_id")
            if not op_id:
                logger.warning("Skipping endpoint without operation_id in group '%s'", group.get("group_name"))
                continue

            tool_spec = _build_tool_spec(endpoint)

            def _make_handler(ep: dict, http_client: Any):
                """Create a synchronous handler closure for a specific endpoint."""

                def handler(tool_use: dict, **kwargs: Any) -> dict:
                    tool_input = tool_use.get("input", {})
                    path_params, query_params, body = _build_request_parts(ep, tool_input)
                    url = _resolve_path(ep["path"], path_params)
                    method = ep["method"].upper()

                    try:
                        response = http_client.request(
                            method,
                            url,
                            params=query_params if query_params else None,
                            json=body,
                        )
                        response_text = response.text
                    except Exception as e:
                        logger.error("API call failed: %s %s — %s", method, url, e)
                        response_text = f'{{"error": "{type(e).__name__}: {e}"}}'

                    return {
                        "toolUseId": tool_use.get("toolUseId", ""),
                        "status": "success",
                        "content": [{"text": response_text}],
                    }

                return handler

            handler = _make_handler(endpoint, client)
            agent_tool = PythonAgentTool(op_id, tool_spec, handler)
            tools.append(agent_tool)
            logger.debug("Created tool: %s (%s %s)", op_id, endpoint["method"].upper(), endpoint["path"])

    logger.info("Created %d API tools from spec", len(tools))
    return tools, client
