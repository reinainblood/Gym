# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from benchmarks.harmbench.kimi_k3_whitebox import (
    MODEL_KEY,
    VISION_WHITEBOX_METHODS,
    WHITEBOX_METHODS,
    expected_full_cases,
    validate_upstream,
    write_runtime_configs,
)
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


def test_kimi_k3_whitebox_catalog_covers_text_and_vision():
    assert set(WHITEBOX_METHODS) == {
        "GCG",
        "GCG-Multi",
        "AutoPrompt",
        "GBDA",
        "PEZ",
        "UAT",
        "AutoDAN",
        "FewShot",
        "MultiModalPGD",
        "MultiModalPGDPatch",
        "MultiModalPGDBlankImage",
    }
    assert set(VISION_WHITEBOX_METHODS) == {
        "MultiModalPGD",
        "MultiModalPGDPatch",
        "MultiModalPGDBlankImage",
    }
    assert sum(expected_full_cases().values()) == 6_730


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_kimi_k3_public_sources_are_hash_locked():
    receipt = validate_upstream(UPSTREAM)
    assert set(receipt["methods"]) == set(WHITEBOX_METHODS)
    assert receipt["methods"]["MultiModalPGD"]["default_hyperparameters"]["num_steps"] == 10
    assert receipt["methods"]["MultiModalPGDPatch"]["default_hyperparameters"]["num_steps"] == 2000
    assert receipt["methods"]["MultiModalPGDBlankImage"]["default_hyperparameters"]["num_steps"] == 1000


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_kimi_k3_target_specific_configs_preserve_public_defaults(tmp_path):
    models, methods = write_runtime_configs(UPSTREAM, tmp_path, Path("/checkpoint/snapshot"))
    model = yaml.safe_load(models.read_text())[MODEL_KEY]
    assert model["model_type"] == "open_source_multimodal"
    assert model["num_gpus"] == 7
    for name in VISION_WHITEBOX_METHODS:
        config = yaml.safe_load(methods[name].read_text())
        assert config[MODEL_KEY]["num_test_cases_per_behavior"] == 1
        assert config[MODEL_KEY]["test_cases_batch_size"] == 1
