# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.compare_gcg_results import BASELINE_MODEL, PRIMARY_MODEL, build, write


def _manifest(model, *, run_id, asr, successes):
    return {
        "artifact_kind": "harmbench_gcg_validated_result",
        "status": "validated",
        "method": "GCG",
        "model": model,
        "target_revision": f"{model}-revision",
        "run_id": run_id,
        "attack_profile": {
            "upstream_revision": "pinned",
            "num_steps": 500,
            "search_width": 512,
            "behaviors": 400,
            "cases": 400,
            "num_shards": 2,
        },
        "completion_profile": {
            "generation_mode": "deterministic",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 512,
        },
        "scoring_profile": {
            "classifier_model": "classifier",
            "classifier_revision": "classifier-revision",
            "classifier_cases": 300,
            "copyright_cases": 100,
            "copyright_book_cases": 50,
            "copyright_lyrics_cases": 50,
        },
        "counts": {
            "expected_cases": 400,
            "scored_cases": 400,
            "failed_cases": 0,
            "missing_cases": 0,
            "successful_cases": successes,
        },
        "metrics": {"native_behavior_averaged_attack_success_rate": asr},
    }


def _inputs(tmp_path):
    primary = tmp_path / "primary.json"
    baseline = tmp_path / "baseline.json"
    primary.write_text(json.dumps(_manifest(PRIMARY_MODEL, run_id="super-run", asr=0.1, successes=40)))
    baseline.write_text(json.dumps(_manifest(BASELINE_MODEL, run_id="qwen-run", asr=0.05, successes=20)))
    return primary, baseline


def test_gcg_comparison_requires_and_records_matched_profiles(tmp_path):
    primary, baseline = _inputs(tmp_path)
    manifest, report = build(primary_path=primary, baseline_path=baseline)
    assert manifest["status"] == "validated"
    assert manifest["observed_asr_delta_primary_minus_baseline"] == pytest.approx(0.05)
    assert manifest["primary"]["model"] == PRIMARY_MODEL
    assert manifest["baseline"]["model"] == BASELINE_MODEL
    assert "causal superiority" in report

    written = write(primary_path=primary, baseline_path=baseline, output_dir=tmp_path / "comparison")
    assert written == manifest
    assert (tmp_path / "comparison/comparison-manifest.json").is_file()
    assert (tmp_path / "comparison/comparison-report.md").is_file()
    assert (tmp_path / "comparison/evidence-manifest.json").is_file()


def test_gcg_comparison_rejects_profile_drift_and_incomplete_inputs(tmp_path):
    primary, baseline = _inputs(tmp_path)
    drifted = json.loads(baseline.read_text())
    drifted["attack_profile"]["search_width"] = 511
    baseline.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="not comparable"):
        build(primary_path=primary, baseline_path=baseline)

    baseline.write_text(json.dumps(_manifest(BASELINE_MODEL, run_id="qwen-run", asr=0.05, successes=20)))
    incomplete = json.loads(primary.read_text())
    incomplete["counts"]["missing_cases"] = 1
    primary.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="failed fields counts"):
        build(primary_path=primary, baseline_path=baseline)
