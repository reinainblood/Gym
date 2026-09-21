# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json

import pytest

from benchmarks.harmbench.calibrate_client_fresh_generation import (
    TARGET_MODEL,
    TARGET_REVISION,
    sha256,
    validate,
)
from benchmarks.harmbench.client_fresh_generate import UPSTREAM_REVISION


def _write_fixture(tmp_path):
    behaviors = tmp_path / "behaviors.csv"
    with behaviors.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["BehaviorID"])
        writer.writeheader()
        writer.writerows({"BehaviorID": f"b{index:03d}"} for index in range(400))
    root = tmp_path / "artifact"
    root.mkdir()
    merged = {f"b{index:03d}": [f"case-{index}"] for index in range(400)}
    (root / "test_cases.json").write_text(json.dumps(merged), encoding="utf-8")
    manifest = []
    total_calls = 0
    for shard in range(2):
        shard_root = root / "shards" / f"{shard:02d}-of-02"
        generated = shard_root / "generated"
        generated.mkdir(parents=True)
        indexes = list(range(shard, 400, 2))
        shard_cases = {f"b{index:03d}": [f"case-{index}"] for index in indexes}
        cases_path = generated / "test_cases.json"
        target_path = generated / "client-target-receipt.json"
        cases_path.write_text(json.dumps(shard_cases), encoding="utf-8")
        target_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        calls = len(indexes) * 3
        total_calls += calls
        receipt = {
            "status": "completed",
            "method": "PAIR",
            "public_method": "Fresh PAIR against the client model",
            "run_id": "run",
            "shard_index": shard,
            "num_shards": 2,
            "selected_source_indexes": indexes,
            "selected_behaviors": len(indexes),
            "test_cases_sha256": sha256(cases_path),
            "client_target_receipt_sha256": sha256(target_path),
            "client_model": TARGET_MODEL,
            "client_revision": TARGET_REVISION,
            "upstream_revision": UPSTREAM_REVISION,
            "target_calls": calls,
        }
        receipt_path = shard_root / "shard-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        manifest.append({"name": f"{shard:02d}-of-02.json", "sha256": sha256(receipt_path)})
    final = {
        "status": "completed",
        "method": "Fresh PAIR against the client model",
        "upstream_method": "PAIR",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": "super-vl-client-fresh",
        "run_id": "run",
        "test_cases_sha256": sha256(root / "test_cases.json"),
        "behaviors_sha256": sha256(behaviors),
        "behaviors": 400,
        "cases": 400,
        "client_model": TARGET_MODEL,
        "source_target_model": TARGET_MODEL,
        "client_revision": TARGET_REVISION,
        "source_target_revision": TARGET_REVISION,
        "target_calls": total_calls,
        "num_shards": 2,
        "shard_receipts": manifest,
        "shard_receipts_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt_path = root / "generation-receipt.json"
    receipt_path.write_text(json.dumps(final), encoding="utf-8")
    return behaviors, root, receipt_path


def test_client_fresh_control_reconciles_all_shards_and_calls(tmp_path):
    behaviors, root, receipt = _write_fixture(tmp_path)
    control = validate(
        method="PAIR",
        artifact_root=root,
        behaviors=behaviors,
        generation_receipt=receipt,
        output=tmp_path / "control.json",
    )
    assert control["matching_behaviors"] == 400
    assert control["target_calls"] == 1200
    assert len(control["shards"]) == 2


def test_client_fresh_control_rejects_tampered_shard(tmp_path):
    behaviors, root, receipt = _write_fixture(tmp_path)
    shard = root / "shards/00-of-02/shard-receipt.json"
    value = json.loads(shard.read_text())
    value["target_calls"] += 1
    shard.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt hash changed"):
        validate(
            method="PAIR",
            artifact_root=root,
            behaviors=behaviors,
            generation_receipt=receipt,
            output=tmp_path / "control.json",
        )
