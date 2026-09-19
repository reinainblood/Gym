# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize official AgentDojo task selectors for the external agent adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdojo.task_suite.load_suites import get_suites


BENCHMARK_VERSION = "v1.2.2"
SUITES = ("banking", "slack", "travel", "workspace")
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "agentdojo_v1_2_2.jsonl"


def _row(
    *,
    suite: str,
    user_task_id: str,
    prompt: str,
    injection_task_id: str | None = None,
) -> dict[str, Any]:
    attacked = injection_task_id is not None
    return {
        "responses_create_params": {
            # The external adapter reloads the authoritative task by id. Keeping the
            # prompt here makes materialized inputs auditable without duplicating it
            # as an execution source of truth.
            "input": [{"role": "user", "content": prompt}],
        },
        "suite": suite,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "attack": "important_instructions" if attacked else None,
        "defense": None,
        "benchmark_version": BENCHMARK_VERSION,
        "condition": "attacked" if attacked else "clean",
    }


def prepare() -> Path:
    suites = get_suites(BENCHMARK_VERSION)
    rows: list[dict[str, Any]] = []
    for suite_name in SUITES:
        suite = suites[suite_name]
        for user_task_id, user_task in suite.user_tasks.items():
            rows.append(_row(suite=suite_name, user_task_id=user_task_id, prompt=user_task.PROMPT))
            rows.extend(
                _row(
                    suite=suite_name,
                    user_task_id=user_task_id,
                    prompt=user_task.PROMPT,
                    injection_task_id=injection_task_id,
                )
                for injection_task_id in suite.injection_tasks
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} AgentDojo selectors to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
