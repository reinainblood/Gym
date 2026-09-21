# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_gcg_completion_worker import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PUBLIC_BEHAVIORS,
    PUBLIC_SEARCH_WIDTH,
    PUBLIC_STEPS,
    UPSTREAM_REVISION,
)
from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    CLASSIFIER_CLIP_TOKENS,
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    sha256,
)
from benchmarks.harmbench.qwen_gcg_blade import write as write_blade
from benchmarks.harmbench.qwen_gcg_result import build, write


def _artifacts(tmp_path):
    artifact_id = "qwen-gcg-full-test"
    generation_path = tmp_path / "generation-receipt.json"
    generation = {
        "status": "completed",
        "method": "GCG",
        "upstream_method": "GCG",
        "upstream_revision": UPSTREAM_REVISION,
        "run_id": artifact_id,
        "source_target_model": MODEL_ID,
        "source_target_revision": MODEL_REVISION,
        "behaviors": PUBLIC_BEHAVIORS,
        "cases": PUBLIC_BEHAVIORS,
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
        "num_shards": 2,
    }
    generation_path.write_text(json.dumps(generation), encoding="utf-8")

    completion_path = tmp_path / "target-completion-receipt.json"
    completion = {
        "status": "completed",
        "artifact_kind": "gcg_target_completion_manifest",
        "artifact_id": artifact_id,
        "target": "qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "GCG",
        "behaviors": PUBLIC_BEHAVIORS,
        "completions": PUBLIC_BEHAVIORS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_receipt_sha256": sha256(generation_path),
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    cases = []
    for index in range(PUBLIC_BEHAVIORS):
        copyright = index >= 300
        labels = [int(index == 0), int(index == 0)]
        score_receipts = (
            [
                {
                    "copyright_reference_sha256": "f" * 64,
                    "copyright_max_similarity": 0.0,
                    "generation_clipped": False,
                },
                {
                    "copyright_reference_sha256": "f" * 64,
                    "copyright_max_similarity": 0.0,
                    "generation_clipped": False,
                },
            ]
            if copyright
            else [
                {"classifier_prompt_sha256": f"prompt-{index}", "generation_clipped": False},
                {"classifier_prompt_sha256": f"prompt-{index}", "generation_clipped": False},
            ]
        )
        cases.append(
            {
                "index": index,
                "behavior_id": f"id-{index}",
                "completion_receipt_sha256": f"completion-{index}",
                "generation_sha256": f"generation-{index}",
                "attack_sha256": f"attack-{index}",
                "functional_category": "standard",
                "semantic_category": "synthetic",
                "tags": ["hash_check", "book" if index < 350 else "lyrics"] if copyright else [],
                "generation_token_count": 12,
                "finish_reason": "stop",
                "scoring_method": "copyright_minhash" if copyright else "harmbench_classifier",
                "labels": labels,
                "score_receipts": score_receipts,
            }
        )
    scores_path = tmp_path / "classifier-scores.json"
    scores = {
        "status": "completed",
        "artifact_kind": "gcg_canonical_512_classifier_scores",
        "artifact_id": artifact_id,
        "method": "GCG",
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "classifier_repeats": 2,
        "classifier_clip_tokens": CLASSIFIER_CLIP_TOKENS,
        "target_model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "target_completion_manifest_sha256": sha256(completion_path),
        "summary": {
            "expected_cases": PUBLIC_BEHAVIORS,
            "scored_cases": PUBLIC_BEHAVIORS,
            "successes": 1,
            "attack_success_rate": 1 / PUBLIC_BEHAVIORS,
            "behavior_averaged_attack_success_rate": 1 / PUBLIC_BEHAVIORS,
            "invalid_label_cases": 0,
            "invalid_classifier_outputs": 0,
            "repeat_agreement": PUBLIC_BEHAVIORS,
            "repeat_valid_agreement": PUBLIC_BEHAVIORS,
            "classifier_clipped_cases": 0,
            "classifier_cases": 300,
            "copyright_cases": 100,
            "copyright_book_cases": 50,
            "copyright_lyrics_cases": 50,
            "length_finished_cases": 0,
        },
        "cases": cases,
    }
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    blade_dir = tmp_path / "blade"
    write_blade(scores_path, blade_dir)
    return generation_path, completion_path, scores_path, blade_dir


def test_qwen_gcg_result_packages_all_validated_gates(tmp_path):
    generation, completion, scores, blade = _artifacts(tmp_path)
    manifest, report = build(
        generation_receipt_path=generation,
        completion_receipt_path=completion,
        classifier_scores_path=scores,
        blade_dir=blade,
    )
    assert manifest["status"] == "validated"
    assert manifest["counts"]["expected_cases"] == PUBLIC_BEHAVIORS
    assert manifest["counts"]["blade_model_denominator"] == PUBLIC_BEHAVIORS
    assert manifest["scoring_profile"]["classifier_cases"] == 300
    assert "synthetic generation" not in report

    written = write(
        generation_receipt_path=generation,
        completion_receipt_path=completion,
        classifier_scores_path=scores,
        blade_dir=blade,
        output_dir=tmp_path / "result",
    )
    assert written == manifest
    assert (tmp_path / "result/run-manifest.json").is_file()
    assert (tmp_path / "result/report.md").is_file()
    assert (tmp_path / "result/evidence-manifest.json").is_file()


def test_qwen_gcg_result_rejects_broken_hash_chain(tmp_path):
    generation, completion, scores, blade = _artifacts(tmp_path)
    broken = json.loads(completion.read_text())
    broken["generation_receipt_sha256"] = "0" * 64
    completion.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="generation_receipt_sha256"):
        build(
            generation_receipt_path=generation,
            completion_receipt_path=completion,
            classifier_scores_path=scores,
            blade_dir=blade,
        )
