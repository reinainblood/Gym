# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import csv

import pytest

from benchmarks.harmbench.generate_direct_full import EXPECTED_COUNTS, generate, upstream_cases
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_full_direct_request_executes_pinned_upstream_method_body():
    rows = [
        {"BehaviorID": "standard", "Behavior": "benign request", "ContextString": ""},
        {"BehaviorID": "contextual", "Behavior": "benign request", "ContextString": "context"},
    ]
    cases = upstream_cases(UPSTREAM / "baselines/direct_request/direct_request.py", rows)
    assert cases == {
        "standard": ["benign request"],
        "contextual": ["context\n\n---\n\nbenign request"],
    }


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_full_direct_request_uses_all_400_public_text_behaviors(tmp_path):
    behaviors = UPSTREAM / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    with behaviors.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    assert {category: sum(row["FunctionalCategory"] == category for row in rows) for category in EXPECTED_COUNTS} == (
        EXPECTED_COUNTS
    )
    receipt = generate(upstream=UPSTREAM, run_id="full-400", output_dir=tmp_path / "direct-full")
    assert receipt["behaviors"] == receipt["cases"] == 400
