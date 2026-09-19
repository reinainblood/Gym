# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from agentdojo.task_suite.load_suites import get_suites

from benchmarks.agentdojo.prepare import BENCHMARK_VERSION, SUITES, _row


def test_materialized_matrix_count_is_derived_from_upstream() -> None:
    suites = get_suites(BENCHMARK_VERSION)
    expected = sum(len(suites[name].user_tasks) * (1 + len(suites[name].injection_tasks)) for name in SUITES)

    assert expected == 1046


def test_clean_and_attacked_rows_are_unambiguous() -> None:
    clean = _row(suite="banking", user_task_id="user_task_0", prompt="pay bill")
    attacked = _row(
        suite="banking",
        user_task_id="user_task_0",
        prompt="pay bill",
        injection_task_id="injection_task_0",
    )

    assert clean["condition"] == "clean"
    assert clean["attack"] is None
    assert clean["injection_task_id"] is None
    assert attacked["condition"] == "attacked"
    assert attacked["attack"] == "important_instructions"
    assert attacked["injection_task_id"] == "injection_task_0"
