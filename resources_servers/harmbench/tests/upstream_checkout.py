# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate an optional pinned HarmBench checkout without depending on worktree depth."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def harmbench_upstream() -> Path:
    """Return the configured checkout, or the stable repo-local optional path."""
    configured = os.environ.get("HARMBENCH_UPSTREAM_DIR")
    return Path(configured).expanduser().resolve() if configured else REPOSITORY_ROOT / "reference" / "HarmBench"
