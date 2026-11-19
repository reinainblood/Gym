# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the pinned public Anthropic rubrics."""

import hashlib

from resources_servers.even_handedness.upstream_spec import (
    EVEN_HANDEDNESS_PROMPT,
    HEDGING_PROMPT,
    REFUSAL_PROMPT,
)


def _normalized_sha256(text: str) -> str:
    return hashlib.sha256("\n".join(line.rstrip() for line in text.split("\n")).encode()).hexdigest()


def test_rubric_text_matches_the_pinned_upstream_revision() -> None:
    assert _normalized_sha256(EVEN_HANDEDNESS_PROMPT) == (
        "f2aa097e78da365bb311f5a2a63b846b4c1981a13fff1c53e187a57fc5dbff7d"
    )
    assert _normalized_sha256(REFUSAL_PROMPT) == (
        "4553839d237f25481d01f1e79964d86550c8989722fd3557b981023274fbb7dc"
    )
    assert _normalized_sha256(HEDGING_PROMPT) == (
        "63fa4833c3d35120b8d95cd7f34a5b9329c720843f5a56fa2ab2e0aa1db69b0e"
    )
