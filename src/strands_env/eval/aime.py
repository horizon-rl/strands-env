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

"""AIME (American Invitational Mathematics Examination) evaluator."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

from datasets import load_dataset

from strands_env.core import Action, TaskContext

from .evaluator import Evaluator

logger = logging.getLogger(__name__)

AIME_HF_PATHS = {
    "2024": {
        "path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_field": "problem",
        "answer_field": "answer",
        "id_field": "id",
    },
    "2025": {
        "path": "MathArena/aime_2025",
        "split": "train",
        "problem_field": "problem",
        "answer_field": "answer",
        "id_field": "id",
    },
}


class AIMEEvaluator(Evaluator):
    """Evaluator for AIME math competition problems."""

    def load_dataset(self, version: Literal["2024", "2025"] = "2024") -> Iterable[Action]:
        """Load AIME dataset from HuggingFace."""
        logger.info(f"Loading AIME {version} dataset from: {AIME_HF_PATHS[version]['path']}")
        dataset = load_dataset(AIME_HF_PATHS[version]["path"], split=AIME_HF_PATHS[version]["split"])

        actions = []
        for i, row in enumerate(dataset):
            problem = row.get(AIME_HF_PATHS[version]["problem_field"], None)
            answer = row.get(AIME_HF_PATHS[version]["answer_field"], None)
            if not all([problem, answer]):
                logger.warning(f"Missing problem or answer fields in row {i}, skipping row")
                continue
            actions.append(Action(message=str(problem), task_context=TaskContext(ground_truth=str(answer))))

        logger.info(f"Loaded {len(actions)}/{len(dataset)} problems from AIME {version} dataset")
        return actions
