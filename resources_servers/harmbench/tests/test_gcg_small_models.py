# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv

import pytest

from benchmarks.harmbench.operations.gcg_small_models_worker import (
    PUBLIC_BEHAVIORS,
    PUBLIC_SEARCH_WIDTH,
    PUBLIC_STEPS,
    TARGETS,
    behavior_artifact_path,
    completed_behavior_count,
    shard_behavior_ids,
)


def _behaviors(path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["BehaviorID"])
        writer.writeheader()
        writer.writerows({"BehaviorID": f"b{index:03d}"} for index in range(PUBLIC_BEHAVIORS))


def test_gcg_target_policy_uses_only_super_and_qwen_full_bf16_weights():
    assert set(TARGETS) == {"super", "qwen"}
    assert TARGETS["super"]["model_revision"] == "hf-ea-0e636f7"
    assert TARGETS["qwen"]["model_revision"] == "dc4d348443bc740c68e2d77492492c11606384d5"  # pragma: allowlist secret
    assert PUBLIC_STEPS == 500
    assert PUBLIC_SEARCH_WIDTH == 512


def test_behavior_shards_partition_the_full_corpus(tmp_path):
    source = tmp_path / "behaviors.csv"
    _behaviors(source)
    shards = [shard_behavior_ids(source, index, 7) for index in range(7)]
    flattened = [behavior_id for shard in shards for behavior_id in shard]
    assert len(flattened) == PUBLIC_BEHAVIORS
    assert len(set(flattened)) == PUBLIC_BEHAVIORS
    assert set(shard_behavior_ids(source, 0, 7, limit=1)) == {"b000"}


def test_behavior_shard_rejects_wrong_corpus_and_selection(tmp_path):
    source = tmp_path / "behaviors.csv"
    _behaviors(source)
    rows = source.read_text(encoding="utf-8").splitlines()
    source.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 400"):
        shard_behavior_ids(source, 0, 2)
    with pytest.raises(ValueError, match="invalid shard"):
        shard_behavior_ids(source, 2, 2)


def test_completion_count_uses_upstream_nested_behavior_layout(tmp_path):
    individual = tmp_path / "test_cases_individual_behaviors"
    expected = ["first", "second"]
    first = behavior_artifact_path(individual, "first")
    first.parent.mkdir(parents=True)
    first.write_text('{"first": ["synthetic"]}', encoding="utf-8")
    # A legacy flat file must not satisfy the upstream artifact contract.
    (individual / "second.json").write_text('{"second": ["synthetic"]}', encoding="utf-8")
    assert completed_behavior_count(individual, expected) == 1
    assert behavior_artifact_path(individual, "second") == individual / "second" / "test_cases.json"
