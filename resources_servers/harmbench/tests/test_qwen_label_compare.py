# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    score_canonical_receipts,
    sha256,
    summarize_case_rows,
)
from resources_servers.harmbench.app import CONTEXTUAL_PROMPT as APP_CONTEXTUAL_PROMPT
from resources_servers.harmbench.app import STANDARD_PROMPT as APP_STANDARD_PROMPT
from resources_servers.harmbench.prompts import CONTEXTUAL_PROMPT, STANDARD_PROMPT


def _case(
    *,
    parent: tuple[int, int] = (0, 0),
    canonical: tuple[int, int] = (0, 0),
    tokens: int = 100,
    changed: bool = False,
    replicate: tuple[int, int] | tuple[()] = (),
) -> dict:
    return {
        "parent_labels": list(parent),
        "canonical_labels": list(canonical),
        "parent_target_token_count": tokens,
        "text_changed": changed,
        "replicate_labels": list(replicate),
    }


def test_operational_prompts_match_verifier_prompts():
    assert STANDARD_PROMPT == APP_STANDARD_PROMPT
    assert CONTEXTUAL_PROMPT == APP_CONTEXTUAL_PROMPT


def test_summary_separates_text_changes_from_label_changes():
    summary = summarize_case_rows(
        [
            _case(changed=True),
            _case(parent=(0, 0), canonical=(1, 1), changed=True),
            _case(parent=(1, 1), canonical=(1, 1), tokens=256),
        ]
    )
    assert summary["canonical_512"]["successes"] == 2
    assert summary["parent_vs_canonical"] == {
        "label_agreement": 2,
        "cases": 3,
        "label_flips": 1,
        "cap_not_binding_cases": 2,
        "text_changed_when_cap_not_binding": 2,
        "label_flips_when_cap_not_binding": 1,
        "cap_not_binding_label_agreement": 1,
    }


def test_summary_detects_classifier_and_replicate_instability():
    summary = summarize_case_rows(
        [
            _case(parent=(0, 1), canonical=(0, 0), replicate=(0, 1)),
            _case(parent=(1, 1), canonical=(1, 0), replicate=(1, 1)),
        ]
    )
    assert summary["parent"]["repeat_agreement"] == 1
    assert summary["canonical_512"]["repeat_agreement"] == 1
    assert summary["canonical_vs_replicate"] == {
        "cases": 2,
        "label_agreement": 2,
        "label_flips": 0,
        "replicate_repeat_agreement": 1,
    }


def _write_canonical(path, *, index: int, method: str = "MultiModalPGDBlankImage") -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "method": method,
                "index": index,
                "behavior_id": f"id-{index}",
                "behavior": f"synthetic behavior {index}",
                "context": "",
                "model_id": "Qwen/Qwen3.5-122B-A10B",
                "model_revision": "dc4d348443bc740c68e2d77492492c11606384d5",  # pragma: allowlist secret
                "max_new_tokens": 512,
                "generation": f"synthetic generation {index}",
                "generation_sha256": hashlib.sha256(f"synthetic generation {index}".encode()).hexdigest(),
                "parent_receipt_sha256": f"parent-hash-{index}",
                "test_case_image_sha256": f"image-hash-{index}",
                "functional_category": "standard",
                "semantic_category": "synthetic",
                "generation_token_count": 512 if index == 0 else 12,
                "finish_reason": "length" if index == 0 else "stop",
            }
        ),
        encoding="utf-8",
    )


def test_canonical_scoring_reconciles_complete_payload_free_cohort(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_canonical(canonical / "000.json", index=0)
    _write_canonical(canonical / "001.json", index=1)

    calls = []

    def score(behavior: str, context: str, generation: str) -> dict:
        calls.append((behavior, context, generation))
        label = 1 if generation.endswith("0") else 0
        return {
            "label": label,
            "classifier_text_sha256": "classifier-text",
            "classifier_prompt_sha256": "classifier-prompt",
            "classifier_prompt_variant": "standard",
            "generation_token_count": 600 if label else 12,
            "generation_clipped": bool(label),
            "generation_for_classifier_sha256": "clipped-generation",
        }

    receipt = score_canonical_receipts(
        canonical_dir=canonical,
        score=score,
        method="MultiModalPGDBlankImage",
        expected_cases=2,
    )

    assert len(calls) == 4
    assert receipt["artifact_kind"] == "whitebox_canonical_512_classifier_scores"
    assert receipt["summary"] == {
        "expected_cases": 2,
        "scored_cases": 2,
        "successes": 1,
        "attack_success_rate": 0.5,
        "invalid_label_cases": 0,
        "invalid_classifier_outputs": 0,
        "repeat_agreement": 2,
        "repeat_valid_agreement": 2,
        "classifier_clipped_cases": 1,
        "length_finished_cases": 1,
    }
    assert all("generation" not in row for row in receipt["cases"])


def test_canonical_scoring_fails_closed_on_incomplete_or_mismatched_inputs(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_canonical(canonical / "000.json", index=0)

    with pytest.raises(ValueError, match="expected 2 canonical receipts"):
        score_canonical_receipts(
            canonical_dir=canonical,
            score=lambda *_: {"label": 0},
            method="MultiModalPGDBlankImage",
            expected_cases=2,
        )

    with pytest.raises(ValueError, match="unsupported Qwen white-box method"):
        score_canonical_receipts(
            canonical_dir=canonical,
            score=lambda *_: {"label": 0},
            method="GCG",
            expected_cases=1,
        )


def test_canonical_scoring_counts_invalid_output_from_either_repeat(tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_canonical(canonical / "000.json", index=0)
    observed = 0

    def score(*_args) -> dict:
        nonlocal observed
        observed += 1
        return {
            "label": 0 if observed == 1 else -1,
            "generation_clipped": False,
        }

    receipt = score_canonical_receipts(
        canonical_dir=canonical,
        score=score,
        method="MultiModalPGDBlankImage",
        expected_cases=1,
    )

    assert receipt["summary"]["invalid_label_cases"] == 1
    assert receipt["summary"]["invalid_classifier_outputs"] == 1
    assert receipt["summary"]["repeat_agreement"] == 0
    assert receipt["summary"]["repeat_valid_agreement"] == 0


def test_canonical_scoring_rehashes_parent_receipts(tmp_path):
    canonical = tmp_path / "completions-512"
    parents = tmp_path / "cases"
    canonical.mkdir()
    parents.mkdir()
    canonical_path = canonical / "000.json"
    parent_path = parents / "000.json"
    _write_canonical(canonical_path, index=0)
    parent = {
        "status": "completed",
        "index": 0,
        "behavior_id": "id-0",
        "behavior": "synthetic behavior 0",
        "context": "",
        "method": "MultiModalPGDBlankImage",
        "test_case_image_sha256": "image-hash-0",
    }
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    receipt = json.loads(canonical_path.read_text())
    receipt["parent_receipt"] = parent_path.name
    receipt["parent_receipt_sha256"] = sha256(parent_path)
    canonical_path.write_text(json.dumps(receipt), encoding="utf-8")
    scored = score_canonical_receipts(
        canonical_dir=canonical,
        parent_dir=parents,
        score=lambda *_: {"label": 0, "generation_clipped": False},
        method="MultiModalPGDBlankImage",
        expected_cases=1,
    )
    assert scored["summary"]["scored_cases"] == 1

    parent["context"] = "changed"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="parent provenance mismatch"):
        score_canonical_receipts(
            canonical_dir=canonical,
            parent_dir=parents,
            score=lambda *_: {"label": 0, "generation_clipped": False},
            method="MultiModalPGDBlankImage",
            expected_cases=1,
        )
