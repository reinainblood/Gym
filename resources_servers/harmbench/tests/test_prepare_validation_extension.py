# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.prepare_validation_extension import select


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _full_inputs(tmp_path):
    path = tmp_path / "full.jsonl"
    rows = [
        {
            "harmbench_id": behavior,
            "attack_test_case_index": index,
            "method": "HumanJailbreaks",
        }
        for behavior in ("a", "b", "c")
        for index in range(2)
    ]
    _write_jsonl(path, rows)
    path.with_suffix(".manifest.json").write_text(
        json.dumps({"method": "HumanJailbreaks", "generation_receipt": {"status": "completed"}}),
        encoding="utf-8",
    )
    return path, rows


def test_selects_only_wholly_missing_behaviors_in_source_order(tmp_path):
    full, rows = _full_inputs(tmp_path)
    existing = tmp_path / "existing.jsonl"
    _write_jsonl(
        existing,
        [{**row, "response": {"model": "client-model"}} for row in rows if row["harmbench_id"] in {"a", "b"}],
    )
    output = tmp_path / "missing.jsonl"
    manifest = select(
        full_inputs=full,
        existing_rollouts=existing,
        output=output,
        expected_full_behaviors=3,
        expected_existing_behaviors=2,
        expected_missing_behaviors=1,
        cases_per_behavior=2,
    )
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["harmbench_id"], row["attack_test_case_index"]) for row in selected] == [("c", 0), ("c", 1)]
    assert manifest["rows"] == 2
    assert manifest["existing_models"] == ["client-model"]
    assert manifest["gym_inputs_sha256"]


def test_rejects_partial_existing_behavior(tmp_path):
    full, rows = _full_inputs(tmp_path)
    existing = tmp_path / "existing.jsonl"
    _write_jsonl(existing, [row for row in rows if row["harmbench_id"] == "a"][:1])
    with pytest.raises(ValueError, match="partially covers"):
        select(
            full_inputs=full,
            existing_rollouts=existing,
            output=tmp_path / "missing.jsonl",
            expected_full_behaviors=3,
            expected_existing_behaviors=1,
            expected_missing_behaviors=2,
            cases_per_behavior=2,
        )


def test_rejects_wrong_missing_behavior_count(tmp_path):
    full, rows = _full_inputs(tmp_path)
    existing = tmp_path / "existing.jsonl"
    _write_jsonl(existing, [row for row in rows if row["harmbench_id"] in {"a", "b"}])
    with pytest.raises(ValueError, match="expected 2 missing behaviors"):
        select(
            full_inputs=full,
            existing_rollouts=existing,
            output=tmp_path / "missing.jsonl",
            expected_full_behaviors=3,
            expected_existing_behaviors=2,
            expected_missing_behaviors=2,
            cases_per_behavior=2,
        )
