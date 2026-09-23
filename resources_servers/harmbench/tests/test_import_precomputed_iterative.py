# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.harmbench.import_precomputed_iterative import merge_cases


def test_iterative_merge_restores_current_order_and_exact_gap():
    merged, missing, extra = merge_cases(
        {"retired": ["r"], "b": ["b-case"]},
        {"a": ["a-case"]},
        ["a", "b"],
    )
    assert list(merged) == ["a", "b"]
    assert missing == ["a"]
    assert extra == ["retired"]


def test_iterative_merge_rejects_wrong_repair_set_and_multiple_cases():
    with pytest.raises(ValueError, match="exact current-corpus gap"):
        merge_cases({"a": ["x"]}, {"different": ["y"]}, ["a", "b"])
    with pytest.raises(ValueError, match="exactly one"):
        merge_cases({}, {"a": ["x", "y"]}, ["a"])
