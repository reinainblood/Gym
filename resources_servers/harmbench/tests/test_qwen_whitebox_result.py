# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_harmbench_worker import (
    METHODS,
    MODEL_ID,
    MODEL_REVISION,
    UPSTREAM_REVISION,
)
from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    CLASSIFIER_CLIP_TOKENS,
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    sha256,
)
from benchmarks.harmbench.qwen_whitebox_blade import write as write_blade
from benchmarks.harmbench.qwen_whitebox_result import build, write


def _artifacts(tmp_path, method="MultiModalPGDBlankImage"):
    run_id = "whitebox-test-run"
    attack_path = tmp_path / "attack-manifest.json"
    attack = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_attack_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "behaviors": 110,
        "cases": 110,
        "images": 110,
        "hyperparameters": METHODS[method],
    }
    attack_path.write_text(json.dumps(attack), encoding="utf-8")
    completion_path = tmp_path / "completion-manifest-512.json"
    completion = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_completion_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "completions": 110,
        "max_new_tokens": 512,
        "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
        "attack_manifest_sha256": sha256(attack_path),
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    cases = []
    for index in range(110):
        labels = [int(index == 0), int(index == 0)]
        cases.append(
            {
                "filename": f"{index:03d}.json",
                "index": index,
                "behavior_id": f"id-{index}",
                "canonical_receipt_sha256": f"canonical-{index}",
                "parent_receipt_sha256": f"parent-{index}",
                "generation_sha256": f"generation-{index}",
                "test_case_image_sha256": f"image-{index}",
                "functional_category": "multimodal",
                "semantic_category": "synthetic",
                "generation_token_count": 512 if index == 0 else 12,
                "finish_reason": "length" if index == 0 else "stop",
                "labels": labels,
                "classifier_receipts": [
                    {"label": labels[0], "generation_clipped": False},
                    {"label": labels[1], "generation_clipped": False},
                ],
            }
        )
    scores_path = tmp_path / "canonical-512.json"
    scores = {
        "schema_version": 1,
        "status": "completed",
        "artifact_kind": "whitebox_canonical_512_classifier_scores",
        "run_id": run_id,
        "method": method,
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "classifier_path": "serial raw /v1/completions",
        "classifier_repeats": 2,
        "classifier_clip_tokens": CLASSIFIER_CLIP_TOKENS,
        "target_model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "completion_manifest_sha256": sha256(completion_path),
        "summary": {
            "expected_cases": 110,
            "scored_cases": 110,
            "successes": 1,
            "attack_success_rate": 1 / 110,
            "invalid_label_cases": 0,
            "invalid_classifier_outputs": 0,
            "repeat_agreement": 110,
            "repeat_valid_agreement": 110,
            "classifier_clipped_cases": 0,
            "length_finished_cases": 1,
        },
        "cases": cases,
    }
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    blade_dir = tmp_path / "blade"
    write_blade(scores_path, blade_dir)
    return run_id, method, attack_path, completion_path, scores_path, blade_dir


def test_modern_qwen_whitebox_result_packages_all_gates(tmp_path):
    run_id, method, attack, completion, scores, blade = _artifacts(tmp_path)
    manifest, report = build(
        run_id=run_id,
        method=method,
        attack_manifest_path=attack,
        completion_manifest_path=completion,
        classifier_scores_path=scores,
        blade_dir=blade,
    )
    assert manifest["status"] == "validated"
    assert manifest["counts"]["blade_model_denominator"] == 110
    assert manifest["attack_profile"]["num_steps"] == METHODS[method]["num_steps"]
    assert "optimizer targets" in report

    written = write(
        run_id=run_id,
        method=method,
        attack_manifest_path=attack,
        completion_manifest_path=completion,
        classifier_scores_path=scores,
        blade_dir=blade,
        output_dir=tmp_path / "result",
    )
    assert written == manifest
    assert (tmp_path / "result/run-manifest.json").is_file()
    assert (tmp_path / "result/report.md").is_file()
    assert (tmp_path / "result/evidence-manifest.json").is_file()


def test_modern_qwen_whitebox_result_rejects_broken_hash_chain_and_legacy_method(tmp_path):
    run_id, method, attack, completion, scores, blade = _artifacts(tmp_path)
    broken = json.loads(completion.read_text())
    broken["attack_manifest_sha256"] = "0" * 64
    completion.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="attack_manifest_sha256"):
        build(
            run_id=run_id,
            method=method,
            attack_manifest_path=attack,
            completion_manifest_path=completion,
            classifier_scores_path=scores,
            blade_dir=blade,
        )

    with pytest.raises(ValueError, match="scoped to BlankImage and Patch"):
        build(
            run_id=run_id,
            method="MultiModalPGD",
            attack_manifest_path=attack,
            completion_manifest_path=completion,
            classifier_scores_path=scores,
            blade_dir=blade,
        )
