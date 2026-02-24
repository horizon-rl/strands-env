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

"""General-purpose data loader for AgentWorldModel-format data folders.

Reads JSONL files and indexes entries by scenario name for efficient lookup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentWorldModelDataLoader:
    """Loads and indexes JSONL data files from an AgentWorldModel-format data folder.

    Each JSONL file contains one JSON object per line, keyed by a ``scenario`` field.
    Data is lazily loaded on first access and cached for subsequent lookups.

    Args:
        data_dir: Path to the data folder (e.g., ``data/AgentWorldModel-1K``).
    """

    # Maps logical names to JSONL filenames
    _FILE_MAP = {
        "scenarios": "gen_scenario.jsonl",
        "tasks": "gen_tasks.jsonl",
        "db_schemas": "gen_db.jsonl",
        "sample_data": "gen_sample.jsonl",
        "api_specs": "gen_spec.jsonl",
        "env_codes": "gen_envs.jsonl",
        "verifiers": "gen_verifier.pure_code.jsonl",
    }

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_jsonl(self, filename: str) -> list[dict]:
        """Read a JSONL file and return a list of parsed JSON objects."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        entries: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at %s:%d — %s", path, line_num, e)
        return entries

    def _index_by_scenario(self, filename: str) -> dict[str, dict]:
        """Read JSONL and build a dict keyed by the ``scenario`` (or ``name``) field."""
        entries = self._read_jsonl(filename)
        index: dict[str, dict] = {}
        for entry in entries:
            key = entry.get("scenario") or entry.get("name")
            if key:
                index[key] = entry
        return index

    def _get_data(self, key: str) -> dict:
        """Lazily load and cache data for the given logical key."""
        if key not in self._cache:
            if key == "verifiers":
                entries = self._read_jsonl(self._FILE_MAP[key])
                index: dict[str, dict] = {}
                for entry in entries:
                    scenario = entry.get("scenario")
                    task_idx = entry.get("task_idx")
                    if scenario is not None and task_idx is not None:
                        index.setdefault(scenario, {})[task_idx] = entry
                self._cache[key] = index
            elif key == "tasks":
                raw = self._index_by_scenario(self._FILE_MAP[key])
                self._cache[key] = {k: v.get("tasks", []) for k, v in raw.items()}
            else:
                self._cache[key] = self._index_by_scenario(self._FILE_MAP[key])
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_scenarios(self) -> list[str]:
        """Return sorted list of all scenario names."""
        return sorted(self._get_data("scenarios").keys())

    def get_scenario(self, scenario: str) -> dict:
        """Return the scenario description dict (``name``, ``description``)."""
        data = self._get_data("scenarios")
        if scenario not in data:
            raise KeyError(f"Scenario '{scenario}' not found. Available: {sorted(data.keys())[:10]}...")
        return data[scenario]

    def get_tasks(self, scenario: str) -> list[str]:
        """Return the list of task strings for a scenario."""
        data = self._get_data("tasks")
        if scenario not in data:
            raise KeyError(f"No tasks found for scenario '{scenario}'")
        return data[scenario]

    def get_db_schema(self, scenario: str) -> dict:
        """Return the database schema dict with ``tables`` containing DDL and indexes."""
        data = self._get_data("db_schemas")
        if scenario not in data:
            raise KeyError(f"No DB schema found for scenario '{scenario}'")
        return data[scenario]["db_schema"]

    def get_sample_data(self, scenario: str) -> dict:
        """Return the sample data dict with INSERT statements per table."""
        data = self._get_data("sample_data")
        if scenario not in data:
            raise KeyError(f"No sample data found for scenario '{scenario}'")
        return data[scenario]["sample_data"]

    def get_api_spec(self, scenario: str) -> dict:
        """Return the API specification dict with ``api_groups`` containing endpoints."""
        data = self._get_data("api_specs")
        if scenario not in data:
            raise KeyError(f"No API spec found for scenario '{scenario}'")
        return data[scenario]["api_spec"]

    def get_env_code(self, scenario: str) -> str:
        """Return the full FastAPI implementation code string."""
        data = self._get_data("env_codes")
        if scenario not in data:
            raise KeyError(f"No environment code found for scenario '{scenario}'")
        return data[scenario]["full_code"]

    def get_verifier(self, scenario: str, task_idx: int) -> str:
        """Return the verification function code string for a specific task."""
        data = self._get_data("verifiers")
        if scenario not in data:
            raise KeyError(f"No verifiers found for scenario '{scenario}'")
        tasks = data[scenario]
        if task_idx not in tasks:
            raise KeyError(
                f"No verifier for scenario '{scenario}' task_idx={task_idx}. Available: {sorted(tasks.keys())}"
            )
        return tasks[task_idx]["verification"]["code"]
