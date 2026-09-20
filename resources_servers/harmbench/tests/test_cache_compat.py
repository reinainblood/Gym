# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.harmbench.cache_compat import (
    clone_transformers_cache,
    install_gcg_dynamic_cache_adapter,
    repeat_hybrid_cache_batch,
)


class CallableModel:
    def __call__(self, *args, **kwargs):
        raise AssertionError("the unit test does not execute GPU loss math")


class Method:
    model = CallableModel()
    compute_candidates_loss = None


class Layer:
    def __init__(self):
        self.flags = {"ready": True}
        self.values = [1, 2]


class Cache:
    def __init__(self):
        self.layers = [Layer()]


def test_adapter_installs_transformers5_candidate_loss_boundary():
    method = Method()
    assert install_gcg_dynamic_cache_adapter(method) is True
    assert callable(method.compute_candidates_loss)


def test_adapter_rejects_method_without_model_forward():
    with pytest.raises(TypeError, match="callable model"):
        install_gcg_dynamic_cache_adapter(object())


def test_cache_clone_preserves_types_without_aliasing_mutable_state():
    source = Cache()
    cloned = clone_transformers_cache(source)
    assert isinstance(cloned, Cache)
    assert isinstance(cloned.layers[0], Layer)
    assert cloned is not source
    assert cloned.layers[0] is not source.layers[0]
    assert cloned.layers[0].flags is not source.layers[0].flags
    assert cloned.layers[0].values is not source.layers[0].values


def test_hybrid_cache_repeat_requires_positive_count_and_layer_list():
    cache = Cache()
    repeat_hybrid_cache_batch(cache, 2)
    with pytest.raises(ValueError, match="positive"):
        repeat_hybrid_cache_batch(cache, 0)
    with pytest.raises(TypeError, match="layers list"):
        repeat_hybrid_cache_batch(object(), 2)
