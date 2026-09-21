# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_gcg_completion_worker import MODEL_ID, MODEL_REVISION, PUBLIC_BEHAVIORS
from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    CLASSIFIER_CLIP_TOKENS,
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    UPSTREAM_REVISION,
)
from benchmarks.harmbench.qwen_gcg_blade import build_rows, validate_bundle, write


def _receipt(*, invalid_index=None):
    cases = []
    for index in range(PUBLIC_BEHAVIORS):
        labels = [1, -1] if index == invalid_index else [int(index == 0), int(index == 0)]
        cases.append(
            {
                "index": index,
                "behavior_id": f"id-{index}",
                "completion_receipt_sha256": f"completion-{index}",
                "generation_sha256": f"generation-{index}",
                "attack_sha256": f"attack-{index}",
                "functional_category": "standard",
                "semantic_category": "synthetic",
                "generation_token_count": 512 if index == 0 else 12,
                "finish_reason": "length" if index == 0 else "stop",
                "labels": labels,
                "classifier_receipts": [
                    {"classifier_prompt_sha256": f"prompt-{index}", "generation_clipped": index == 0},
                    {"classifier_prompt_sha256": f"prompt-{index}", "generation_clipped": index == 0},
                ],
            }
        )
    successes = sum(row["labels"][0] == 1 for row in cases)
    invalid_cases = sum(any(label not in {0, 1} for label in row["labels"]) for row in cases)
    invalid_outputs = sum(label not in {0, 1} for row in cases for label in row["labels"])
    repeat_agreement = sum(len(set(row["labels"])) == 1 for row in cases)
    repeat_valid_agreement = sum(len(set(row["labels"])) == 1 and row["labels"][0] in {0, 1} for row in cases)
    return {
        "artifact_kind": "gcg_canonical_512_classifier_scores",
        "status": "completed",
        "method": "GCG",
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "target_model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "classifier_repeats": 2,
        "classifier_clip_tokens": CLASSIFIER_CLIP_TOKENS,
        "summary": {
            "expected_cases": PUBLIC_BEHAVIORS,
            "scored_cases": PUBLIC_BEHAVIORS,
            "successes": successes,
            "attack_success_rate": successes / PUBLIC_BEHAVIORS,
            "behavior_averaged_attack_success_rate": successes / PUBLIC_BEHAVIORS,
            "invalid_label_cases": invalid_cases,
            "invalid_classifier_outputs": invalid_outputs,
            "repeat_agreement": repeat_agreement,
            "repeat_valid_agreement": repeat_valid_agreement,
        },
        "cases": cases,
    }


def test_qwen_gcg_blade_maps_complete_payload_free_receipt(tmp_path):
    receipt = _receipt()
    rows = build_rows(receipt)
    assert len(rows) == PUBLIC_BEHAVIORS
    assert rows[0]["attack_success"] is True
    assert rows[0]["terminal_condition"] == "target_max_output_tokens"
    assert rows[0]["runtime_identity"]["runtime_modality"] == "text_token_gradient"
    assert rows[1]["attack_success"] is False
    assert all("generation" not in row for row in rows)

    receipt_path = tmp_path / "classifier-scores.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    metrics = write(receipt_path, tmp_path / "blade")
    assert metrics["valid_rows"] == PUBLIC_BEHAVIORS
    assert metrics["methods"]["GCG"]["asr_numerator"] == 1
    assert (tmp_path / "blade/harmbench_blade_rows.jsonl").is_file()
    assert (tmp_path / "blade/harmbench_blade_report.md").is_file()
    manifest = validate_bundle(receipt_path, tmp_path / "blade")
    assert manifest["status"] == "completed"
    assert manifest["rows"] == PUBLIC_BEHAVIORS
    assert len(manifest["outputs"]) == 3

    (tmp_path / "blade/harmbench_blade_metrics.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="readback disagrees"):
        validate_bundle(receipt_path, tmp_path / "blade")


def test_qwen_gcg_blade_excludes_invalid_or_unstable_classifier_case():
    rows = build_rows(_receipt(invalid_index=7))
    assert rows[7]["valid"] is False
    assert rows[7]["asr_denominator"] == 0
    assert rows[7]["reconciliation_disposition"] == "excluded_task_or_verifier_failure"


def test_qwen_gcg_blade_fails_closed_on_wrong_provenance_or_denominator():
    wrong = _receipt()
    wrong["target_revision"] = "different"
    with pytest.raises(ValueError, match="mismatched target_revision"):
        build_rows(wrong)

    short = _receipt()
    short["cases"].pop()
    with pytest.raises(ValueError, match="exactly 400"):
        build_rows(short)

    mismatched = _receipt()
    mismatched["summary"]["successes"] = 99
    with pytest.raises(ValueError, match="summary disagrees"):
        build_rows(mismatched)
