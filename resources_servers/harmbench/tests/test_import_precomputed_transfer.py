# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.harmbench.import_precomputed_transfer import filter_cases


def test_transfer_filter_preserves_pinned_source_order_and_reports_extras():
    cases = {"retired": ["x"], "b": ["b-case"], "a": ["a-case"]}
    selected, extra = filter_cases(cases, ["a", "b"])
    assert list(selected) == ["a", "b"]
    assert extra == ["retired"]


def test_transfer_filter_rejects_missing_or_multi_case_behavior():
    with pytest.raises(ValueError, match="missing"):
        filter_cases({"a": ["x"]}, ["a", "b"])
    with pytest.raises(ValueError, match="exactly one"):
        filter_cases({"a": ["x", "y"]}, ["a"])
