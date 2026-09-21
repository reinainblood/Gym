# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    TARGET_MODEL,
    TARGET_REVISION,
    UPSTREAM_REVISION,
)
from benchmarks.harmbench.qwen_whitebox_blade import build_rows, write


def _receipt(*, invalid_index: int | None = None) -> dict:
    cases = []
    for index in range(110):
        labels = [1, -1] if index == invalid_index else [int(index == 0), int(index == 0)]
        cases.append(
            {
                "index": index,
                "behavior_id": f"id-{index}",
                "canonical_receipt_sha256": f"canonical-{index}",
                "parent_receipt_sha256": f"parent-{index}",
                "generation_sha256": f"generation-{index}",
                "test_case_image_sha256": f"image-{index}",
                "functional_category": "standard",
                "semantic_category": "synthetic",
                "generation_token_count": 512 if index == 0 else 12,
                "finish_reason": "length" if index == 0 else "stop",
                "labels": labels,
                "classifier_receipts": [
                    {
                        "classifier_prompt_sha256": f"prompt-{index}",
                        "generation_clipped": index == 0,
                    },
                    {
                        "classifier_prompt_sha256": f"prompt-{index}",
                        "generation_clipped": index == 0,
                    },
                ],
            }
        )
    return {
        "artifact_kind": "whitebox_canonical_512_classifier_scores",
        "method": "MultiModalPGDBlankImage",
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "target_model": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "summary": {"expected_cases": 110, "scored_cases": 110},
        "cases": cases,
    }


def test_whitebox_blade_maps_complete_payload_free_receipt(tmp_path):
    receipt = _receipt()
    rows = build_rows(receipt)
    assert len(rows) == 110
    assert rows[0]["attack_success"] is True
    assert rows[0]["terminal_condition"] == "target_max_output_tokens"
    assert rows[0]["runtime_identity"]["runtime_modality"] == "multimodal_gradient"
    assert rows[1]["attack_success"] is False
    assert all("generation" not in row for row in rows)

    receipt_path = tmp_path / "canonical-512.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    metrics = write(receipt_path, tmp_path / "blade")
    assert metrics["valid_rows"] == 110
    assert metrics["methods"]["MultiModalPGDBlankImage"]["asr_numerator"] == 1
    assert (tmp_path / "blade/harmbench_blade_rows.jsonl").is_file()
    assert (tmp_path / "blade/harmbench_blade_report.md").is_file()


def test_whitebox_blade_excludes_invalid_or_unstable_classifier_case():
    rows = build_rows(_receipt(invalid_index=7))
    assert rows[7]["valid"] is False
    assert rows[7]["asr_denominator"] == 0
    assert rows[7]["reconciliation_disposition"] == "excluded_task_or_verifier_failure"


def test_whitebox_blade_fails_closed_on_wrong_provenance_or_denominator():
    wrong = _receipt()
    wrong["target_revision"] = "different"
    with pytest.raises(ValueError, match="mismatched target_revision"):
        build_rows(wrong)

    short = _receipt()
    short["cases"].pop()
    with pytest.raises(ValueError, match="exactly 110"):
        build_rows(short)
