# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_gcg_completion_worker import MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION
from benchmarks.harmbench.operations.qwen.qwen_gcg_score import score_completion_artifact
from benchmarks.harmbench.operations.qwen.qwen_label_compare import sha256


def _completion_artifact(tmp_path, count=2, copyright_index=None):
    root = tmp_path / "target-completions" / "qwen-bf16"
    individual = root / "individual"
    individual.mkdir(parents=True)
    rows = []
    for index in range(count):
        generation = f"synthetic generation {index}"
        receipt = {
            "schema_version": 1,
            "artifact_kind": "gcg_target_completion",
            "status": "completed",
            "artifact_id": "full-run",
            "method": "GCG",
            "index": index,
            "behavior_id": f"b{index}",
            "behavior": f"synthetic behavior {index}",
            "context": "",
            "functional_category": "standard",
            "semantic_category": "synthetic",
            "tags": ["hash_check", "book"] if index == copyright_index else [],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_config_sha256": "a" * 64,
            "checkpoint_index_sha256": "b" * 64,
            "generation_receipt_sha256": "c" * 64,
            "attack_sha256": "d" * 64,
            "rendered_prompt_sha256": "e" * 64,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_token_count": 512 if index == 0 else 12,
            "finish_reason": "length" if index == 0 else "stop",
            "generation": generation,
            "generation_sha256": hashlib.sha256(generation.encode()).hexdigest(),
        }
        path = individual / f"{index:03d}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        rows.append({"name": path.name, "sha256": sha256(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "gcg_target_completion_manifest",
        "status": "completed",
        "artifact_id": "full-run",
        "target": "qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "GCG",
        "behaviors": count,
        "completions": count,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_receipt_sha256": "c" * 64,
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_index_sha256": "b" * 64,
        "individual_receipts": rows,
        "individual_receipts_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    (root / "target-completion-receipt.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_qwen_gcg_scores_complete_payload_free_cohort(tmp_path):
    root = _completion_artifact(tmp_path)
    calls = []

    def score(behavior, context, generation):
        calls.append((behavior, context, generation))
        label = 1 if generation.endswith("0") else 0
        return {
            "label": label,
            "generation_clipped": bool(label),
            "classifier_prompt_sha256": "prompt",
            "generation_for_classifier_sha256": "generation",
        }

    receipt = score_completion_artifact(completion_root=root, score=score, expected_cases=2)
    assert len(calls) == 4
    assert receipt["summary"] == {
        "expected_cases": 2,
        "scored_cases": 2,
        "successes": 1,
        "attack_success_rate": 0.5,
        "behavior_averaged_attack_success_rate": 0.5,
        "invalid_label_cases": 0,
        "invalid_classifier_outputs": 0,
        "repeat_agreement": 2,
        "repeat_valid_agreement": 2,
        "classifier_clipped_cases": 1,
        "classifier_cases": 2,
        "copyright_cases": 0,
        "copyright_book_cases": 0,
        "copyright_lyrics_cases": 0,
        "length_finished_cases": 1,
    }
    assert all("generation" not in row for row in receipt["cases"])


def test_qwen_gcg_scorer_fails_closed_on_tampering_and_incomplete_denominator(tmp_path):
    root = _completion_artifact(tmp_path)
    (root / "individual" / "001.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="readback failed"):
        score_completion_artifact(completion_root=root, score=lambda *_: {"label": 0}, expected_cases=2)
    with pytest.raises(ValueError, match="individual_receipts"):
        score_completion_artifact(completion_root=root, score=lambda *_: {"label": 0}, expected_cases=3)


def test_qwen_gcg_scorer_rejects_payload_bearing_classifier_receipt(tmp_path):
    root = _completion_artifact(tmp_path, count=1)
    with pytest.raises(ValueError, match="unsupported fields"):
        score_completion_artifact(
            completion_root=root,
            score=lambda *_: {"label": 0, "raw_classifier_output": "not allowed"},
            expected_cases=1,
        )


def test_qwen_gcg_scorer_routes_hash_checks_to_pinned_copyright_path(tmp_path):
    root = _completion_artifact(tmp_path, copyright_index=1)
    classifier_calls = []
    copyright_calls = []

    def classifier(*args):
        classifier_calls.append(args)
        return {"label": 0, "generation_clipped": False}

    def copyright(*args):
        copyright_calls.append(args)
        return {"label": 1, "max_similarity": 0.75, "reference_sha256": "f" * 64}

    receipt = score_completion_artifact(
        completion_root=root,
        score=classifier,
        copyright_score=copyright,
        expected_cases=2,
    )
    assert len(classifier_calls) == 2
    assert len(copyright_calls) == 2
    assert receipt["summary"]["classifier_cases"] == 1
    assert receipt["summary"]["copyright_cases"] == 1
    assert receipt["cases"][1]["scoring_method"] == "copyright_minhash"
    assert receipt["cases"][1]["score_receipts"][0]["copyright_max_similarity"] == 0.75
