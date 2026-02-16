"""General-purpose data loader for AWM-format (AgentWorldModel) data folders.

Reads JSONL files and indexes entries by scenario name for efficient lookup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AWMDataLoader:
    """Loads and indexes JSONL data files from an AWM-format data folder.

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

        # Lazy caches — populated on first access
        self._scenarios: dict[str, dict] | None = None
        self._tasks: dict[str, list[str]] | None = None
        self._db_schemas: dict[str, dict] | None = None
        self._sample_data: dict[str, dict] | None = None
        self._api_specs: dict[str, dict] | None = None
        self._env_codes: dict[str, dict] | None = None
        self._verifiers: dict[str, dict[int, dict]] | None = None

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

    def _ensure_scenarios(self) -> dict[str, dict]:
        if self._scenarios is None:
            self._scenarios = self._index_by_scenario(self._FILE_MAP["scenarios"])
        return self._scenarios

    def _ensure_tasks(self) -> dict[str, list[str]]:
        if self._tasks is None:
            raw = self._index_by_scenario(self._FILE_MAP["tasks"])
            self._tasks = {k: v.get("tasks", []) for k, v in raw.items()}
        return self._tasks

    def _ensure_db_schemas(self) -> dict[str, dict]:
        if self._db_schemas is None:
            self._db_schemas = self._index_by_scenario(self._FILE_MAP["db_schemas"])
        return self._db_schemas

    def _ensure_sample_data(self) -> dict[str, dict]:
        if self._sample_data is None:
            self._sample_data = self._index_by_scenario(self._FILE_MAP["sample_data"])
        return self._sample_data

    def _ensure_api_specs(self) -> dict[str, dict]:
        if self._api_specs is None:
            self._api_specs = self._index_by_scenario(self._FILE_MAP["api_specs"])
        return self._api_specs

    def _ensure_env_codes(self) -> dict[str, dict]:
        if self._env_codes is None:
            self._env_codes = self._index_by_scenario(self._FILE_MAP["env_codes"])
        return self._env_codes

    def _ensure_verifiers(self) -> dict[str, dict[int, dict]]:
        if self._verifiers is None:
            entries = self._read_jsonl(self._FILE_MAP["verifiers"])
            self._verifiers = {}
            for entry in entries:
                scenario = entry.get("scenario")
                task_idx = entry.get("task_idx")
                if scenario is not None and task_idx is not None:
                    self._verifiers.setdefault(scenario, {})[task_idx] = entry
        return self._verifiers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_scenarios(self) -> list[str]:
        """Return sorted list of all scenario names."""
        return sorted(self._ensure_scenarios().keys())

    def get_scenario(self, scenario: str) -> dict:
        """Return the scenario description dict (``name``, ``description``)."""
        data = self._ensure_scenarios()
        if scenario not in data:
            raise KeyError(f"Scenario '{scenario}' not found. Available: {sorted(data.keys())[:10]}...")
        return data[scenario]

    def get_tasks(self, scenario: str) -> list[str]:
        """Return the list of task strings for a scenario."""
        data = self._ensure_tasks()
        if scenario not in data:
            raise KeyError(f"No tasks found for scenario '{scenario}'")
        return data[scenario]

    def get_db_schema(self, scenario: str) -> dict:
        """Return the database schema dict with ``tables`` containing DDL and indexes."""
        data = self._ensure_db_schemas()
        if scenario not in data:
            raise KeyError(f"No DB schema found for scenario '{scenario}'")
        return data[scenario]["db_schema"]

    def get_sample_data(self, scenario: str) -> dict:
        """Return the sample data dict with INSERT statements per table."""
        data = self._ensure_sample_data()
        if scenario not in data:
            raise KeyError(f"No sample data found for scenario '{scenario}'")
        return data[scenario]["sample_data"]

    def get_api_spec(self, scenario: str) -> dict:
        """Return the API specification dict with ``api_groups`` containing endpoints."""
        data = self._ensure_api_specs()
        if scenario not in data:
            raise KeyError(f"No API spec found for scenario '{scenario}'")
        return data[scenario]["api_spec"]

    def get_env_code(self, scenario: str) -> str:
        """Return the full FastAPI implementation code string."""
        data = self._ensure_env_codes()
        if scenario not in data:
            raise KeyError(f"No environment code found for scenario '{scenario}'")
        return data[scenario]["full_code"]

    def get_verifier(self, scenario: str, task_idx: int) -> str:
        """Return the verification function code string for a specific task."""
        data = self._ensure_verifiers()
        if scenario not in data:
            raise KeyError(f"No verifiers found for scenario '{scenario}'")
        tasks = data[scenario]
        if task_idx not in tasks:
            raise KeyError(
                f"No verifier for scenario '{scenario}' task_idx={task_idx}. Available: {sorted(tasks.keys())}"
            )
        return tasks[task_idx]["verification"]["code"]
