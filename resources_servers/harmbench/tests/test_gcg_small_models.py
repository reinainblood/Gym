# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json

import pytest

from benchmarks.harmbench.operations.gcg_small_models_worker import (
    EXECUTION_ONLY_DEVIATION,
    GCG_CONFIG_SHA256,
    PIPELINE_SHA256,
    PUBLIC_BEHAVIORS,
    PUBLIC_SEARCH_WIDTH,
    PUBLIC_STEPS,
    TARGETS,
    UPSTREAM_REVISION,
    behavior_artifact_path,
    build_shard_receipt_manifest,
    completed_behavior_count,
    run_generation_if_needed,
    shard_behavior_ids,
    validate_complete_shard_receipts,
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


def test_complete_shard_skips_expensive_generation(monkeypatch, tmp_path):
    individual = tmp_path / "test_cases_individual_behaviors"
    artifact = behavior_artifact_path(individual, "complete")
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"complete": ["synthetic"]}', encoding="utf-8")
    invoked = []
    monkeypatch.setattr(
        "benchmarks.harmbench.operations.gcg_small_models_worker.subprocess.run",
        lambda *args, **kwargs: invoked.append((args, kwargs)),
    )
    run_generation_if_needed(
        individual_dir=individual,
        behavior_ids=["complete"],
        command=["python", "expensive.py"],
        upstream=tmp_path,
    )
    assert invoked == []

    run_generation_if_needed(
        individual_dir=individual,
        behavior_ids=["complete", "missing"],
        command=["python", "expensive.py"],
        upstream=tmp_path,
    )
    assert len(invoked) == 1


def _write_receipt_set(tmp_path, *, target="super", artifact_id="full-run", num_shards=2):
    behaviors = tmp_path / "behaviors.csv"
    _behaviors(behaviors)
    output_root = tmp_path / artifact_id
    receipt_dir = output_root / "shard-receipts"
    receipt_dir.mkdir(parents=True)
    for index in range(num_shards):
        selected = shard_behavior_ids(behaviors, index, num_shards)
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "artifact_id": artifact_id,
            "target": target,
            "model_id": TARGETS[target]["model_id"],
            "model_revision": TARGETS[target]["model_revision"],
            "checkpoint_manifest_sha256": "a" * 64,
            "checkpoint_manifest_status": "verified",
            "upstream": {
                "revision": UPSTREAM_REVISION,
                "pipeline_sha256": PIPELINE_SHA256,
                "gcg_config_sha256": GCG_CONFIG_SHA256,
            },
            "method": "GCG",
            "num_steps": PUBLIC_STEPS,
            "search_width": PUBLIC_SEARCH_WIDTH,
            "starting_search_batch_size": TARGETS[target]["starting_search_batch_size"],
            "execution_only_deviation": EXECUTION_ONLY_DEVIATION,
            "shard_index": index,
            "num_shards": num_shards,
            "selected_behaviors": len(selected),
            "completed_behaviors": len(selected),
            "behavior_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
            "model_config_sha256": "b" * 64,
            "method_config_sha256": "c" * 64,
        }
        path = receipt_dir / f"{target}-{index:02d}-of-{num_shards:02d}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
    return output_root, behaviors


def test_complete_shard_receipts_bind_every_source_order_partition(tmp_path):
    output_root, behaviors = _write_receipt_set(tmp_path)
    receipts = validate_complete_shard_receipts(
        output_root=output_root,
        behaviors=behaviors,
        target="super",
        artifact_id="full-run",
    )
    assert len(receipts) == 2
    assert sum(receipt["completed_behaviors"] for receipt in receipts) == PUBLIC_BEHAVIORS

    manifest, manifest_sha256 = build_shard_receipt_manifest(
        output_root=output_root,
        target="super",
        num_shards=len(receipts),
    )
    assert [row["name"] for row in manifest] == ["super-00-of-02.json", "super-01-of-02.json"]
    assert all(len(row["sha256"]) == 64 for row in manifest)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert manifest_sha256 == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(status="running"), "failed fields status"),
        (lambda receipt: receipt.update(completed_behaviors=0), "failed fields completed_behaviors"),
        (lambda receipt: receipt.update(behavior_ids_sha256="0" * 64), "failed fields behavior_ids_sha256"),
        (
            lambda receipt: receipt.update(checkpoint_manifest_sha256="d" * 64),
            "disagree on checkpoint_manifest_sha256",
        ),
        (
            lambda receipt: receipt["upstream"].update(pipeline_sha256="0" * 64),
            "failed fields upstream.pipeline_sha256",
        ),
    ],
)
def test_complete_shard_receipts_reject_invalid_evidence(tmp_path, mutation, message):
    output_root, behaviors = _write_receipt_set(tmp_path)
    path = output_root / "shard-receipts" / "super-01-of-02.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutation(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_complete_shard_receipts(
            output_root=output_root,
            behaviors=behaviors,
            target="super",
            artifact_id="full-run",
        )


def test_complete_shard_receipts_reject_missing_partition(tmp_path):
    output_root, behaviors = _write_receipt_set(tmp_path)
    (output_root / "shard-receipts" / "super-01-of-02.json").unlink()
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        validate_complete_shard_receipts(
            output_root=output_root,
            behaviors=behaviors,
            target="super",
            artifact_id="full-run",
        )
