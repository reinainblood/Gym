# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the run package available to the fidelity tests when it has been prepared.

The prompt-fidelity tests compare this adapter's rendered prompts with the ``prompts/*.md``
files the package ships, and the pack checks read its tables. They need the package under
``benchmarks/kora/upstream/``, which ``python -m benchmarks.kora.prepare`` fetches (about
280 MB). It is not fetched here by default so the unit suite stays hermetic; set
``KORA_TEST_FETCH_PACKAGE=1`` to fetch it before collection. Without it those tests skip
and the rest of the suite runs on inline fixtures.
"""

import logging
import os

from benchmarks.kora.prepare import ensure_package


def pytest_configure(config) -> None:
    if os.environ.get("KORA_TEST_FETCH_PACKAGE") == "1":
        try:
            ensure_package()
        except Exception:  # pragma: no cover - a failed fetch only skips the fidelity tests
            logging.getLogger(__name__).warning("Could not fetch the KORA package; fidelity tests will skip")
