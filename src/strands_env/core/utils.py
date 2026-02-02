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

"""Environment utilities for strands-env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands.types.content import Message


def extract_text_content(message: Message) -> str | None:
    """Extract text content from a strands message.

    Handles both strands formats:
    - SGLang/OpenAI: {"role": "...", "content": "string"}
    - Bedrock: {"role": "...", "content": [{"text": "..."}, ...]}

    Args:
        message: A strands message dict with 'content' field.

    Returns:
        Extracted text content, or None if no text found.
    """
    content: str | list[dict] | Any = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
        return "\n".join(texts) if texts else None
    return None


def render_prompt(path: Path | str) -> str:
    """Load a system prompt from file.

    Args:
        path: Path to the prompt file.

    Returns:
        The prompt string.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> prompt = render_prompt(Path(__file__).parent / "system_prompt.md")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()
