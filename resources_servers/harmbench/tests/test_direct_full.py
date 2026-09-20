# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from benchmarks.harmbench.generate_direct_full import upstream_cases
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
