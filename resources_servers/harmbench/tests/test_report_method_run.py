# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.prepare_generated import sha256
from benchmarks.harmbench.report_method_run import build
from resources_servers.harmbench.app import CLASSIFIER_MODEL, CLASSIFIER_REVISION


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_report_reconciles_per_behavior_asr_without_exposing_payloads(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    failures = tmp_path / "failures.jsonl"
    aggregates = tmp_path / "aggregate.json"
    quality = tmp_path / "quality.json"
    source = tmp_path / "source.json"
    rows = [
        {"harmbench_id": "a", "attack_test_case_index": 0, "method": "HumanJailbreaks"},
        {"harmbench_id": "a", "attack_test_case_index": 1, "method": "HumanJailbreaks"},
        {"harmbench_id": "b", "attack_test_case_index": 0, "method": "HumanJailbreaks"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        rollouts,
        [
            row
            | {
                "score_label": label,
                "response": {"model": "test-model"},
                "generation": "SECRET_PAYLOAD_NOT_FOR_REPORT",
                "functional_category": "standard",
            }
            for row, label in zip(rows, [1, 0, 1], strict=True)
        ],
    )
    failures.write_text("", encoding="utf-8")
    aggregates.write_text(json.dumps([{"agent_metrics": {"attack_success_rate": 0.75}}]))
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 3, "unhealthy": 0, "unobserved": 0}}}))
    source.write_text(json.dumps({"method": "HumanJailbreaks", "upstream_revision": "test-revision"}))
    manifest, report = build(
        dataset=dataset,
        rollouts=rollouts,
        failures=failures,
        aggregate_metrics=aggregates,
        quality_summary=quality,
        source_manifest=source,
        model="test-model",
        run_id="test-run",
    )
    assert manifest["status"] == "provisional"
    assert manifest["collection_status"] == "complete"
    assert manifest["metrics"]["attack_success_rate"] == 0.75
    assert manifest["metrics"]["attack_success_rate_per_test_case"] == pytest.approx(2 / 3)
    assert "SECRET_PAYLOAD_NOT_FOR_REPORT" not in report


def test_report_rejects_wrong_model_identity(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    failures = tmp_path / "failures.jsonl"
    aggregates = tmp_path / "aggregate.json"
    quality = tmp_path / "quality.json"
    source = tmp_path / "source.json"
    _write_jsonl(dataset, [{"harmbench_id": "a", "method": "PAIR"}])
    _write_jsonl(rollouts, [{"harmbench_id": "a", "method": "PAIR", "response": {"model": "wrong"}}])
    failures.write_text("")
    aggregates.write_text(json.dumps([{"agent_metrics": {"attack_success_rate": 0}}]))
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 1, "unhealthy": 0, "unobserved": 0}}}))
    source.write_text(json.dumps({"method": "PAIR", "upstream_revision": "test-revision"}))
    with pytest.raises(ValueError, match="response model identity"):
        build(
            dataset=dataset,
            rollouts=rollouts,
            failures=failures,
            aggregate_metrics=aggregates,
            quality_summary=quality,
            source_manifest=source,
            model="intended",
            run_id="test-run",
        )


@pytest.mark.parametrize("method", ["HumanJailbreaks", "TAP-Transfer", "PAIR"])
def test_generated_run_requires_all_three_matching_controls(tmp_path, method):
    dataset = tmp_path / "dataset.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    failures = tmp_path / "failures.jsonl"
    aggregates = tmp_path / "aggregate.json"
    quality = tmp_path / "quality.json"
    source = tmp_path / "source.json"
    parent = tmp_path / "parent.jsonl"
    classifier = tmp_path / "classifier.json"
    copyright_control = tmp_path / "copyright.json"
    generation = tmp_path / "generation.json"
    cases = [
        {"harmbench_id": "a", "method": method, "attack_test_case_index": 0},
        {"harmbench_id": "b", "method": method, "attack_test_case_index": 0},
    ]
    _write_jsonl(dataset, cases)
    _write_jsonl(
        rollouts,
        [
            cases[0]
            | {
                "response": {"id": "target-a", "model": "target", "output": []},
                "score_label": 1,
                "classifier_label": 1,
                "scoring_method": "harmbench_classifier",
            },
            cases[1]
            | {
                "response": {"id": "target-b", "model": "target", "output": []},
                "score_label": 0,
                "scoring_method": "copyright_minhash",
            },
        ],
    )
    parent.write_bytes(rollouts.read_bytes())
    failures.write_text("")
    aggregates.write_text(json.dumps([{"agent_metrics": {"attack_success_rate": 0.5}}]))
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 2, "unhealthy": 0, "unobserved": 0}}}))
    source.write_text(
        json.dumps(
            {
                "method": method,
                "upstream_revision": "pinned-test",
                "generation_receipt": {"test_cases_sha256": "cases-sha"},
            }
        )
    )
    classifier.write_text(
        json.dumps(
            {
                "classifier_model": CLASSIFIER_MODEL,
                "classifier_revision": CLASSIFIER_REVISION,
                "rollouts_sha256": sha256(rollouts),
                "rollout_cases": 1,
                "rollout_agreement": 1,
                "rollout_prompt_tokens_identical": 1,
                "rollout_prompt_sha256_matches": 1,
                "clip_checks": {"checked": 1, "text_identical": 1, "count_identical": 1},
                "fixture_cases": 1,
                "fixture_agreement": 1,
                "disagreements": [],
            }
        )
    )
    copyright_receipt = {
        "upstream_revision": "pinned-test",
        "rollouts_sha256": sha256(rollouts),
        "checked": 1,
        "agreement": 1,
        "mismatches": [],
    }
    copyright_control.write_text(json.dumps(copyright_receipt))
    generation.write_text(
        json.dumps(
            {
                "upstream_revision": "pinned-test",
                "generated_cases_sha256": "cases-sha",
                "cases": 2,
                "behaviors": 2,
                "matching_behaviors": 2,
                "mismatches": [],
            }
        )
    )
    kwargs = dict(
        dataset=dataset,
        rollouts=rollouts,
        failures=failures,
        aggregate_metrics=aggregates,
        quality_summary=quality,
        source_manifest=source,
        model="target",
        run_id="test-run",
        classifier_calibration=classifier,
        copyright_calibration=copyright_control,
        generation_calibration=generation,
        parent_rollouts=parent,
    )
    manifest, _ = build(**kwargs)
    assert manifest["status"] == "validated"
    expected_protocol = {
        "HumanJailbreaks": "passed_generation_classifier_copyright_controls",
        "TAP-Transfer": "passed_transfer_source_classifier_copyright_controls",
        "PAIR": "passed_iterative_generation_classifier_copyright_controls",
    }
    assert manifest["protocol_validation"] == expected_protocol[method]
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 1, "unhealthy": 0, "unobserved": 1}}}))
    manifest_with_gap, _ = build(**kwargs)
    assert manifest_with_gap["status"] == "provisional"
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 2, "unhealthy": 0, "unobserved": 0}}}))
    copyright_control.write_text(json.dumps(copyright_receipt | {"agreement": 0}))
    with pytest.raises(ValueError, match="copyright scorer control"):
        build(**kwargs)
    copyright_control.write_text(json.dumps(copyright_receipt))
    parent_rows = [json.loads(line) for line in parent.read_text().splitlines()]
    parent_rows[0]["response"]["id"] = "different-target-response"
    _write_jsonl(parent, parent_rows)
    with pytest.raises(ValueError, match="changed a saved target model response"):
        build(**kwargs)


def test_gcg_run_requires_finalized_source_and_both_scorer_controls(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    failures = tmp_path / "failures.jsonl"
    aggregates = tmp_path / "aggregate.json"
    quality = tmp_path / "quality.json"
    source = tmp_path / "source.json"
    classifier = tmp_path / "classifier.json"
    copyright_control = tmp_path / "copyright.json"
    model = "nvidia/test-super-bf16"
    rows = [
        {
            "harmbench_id": f"b{index:03d}",
            "method": "GCG",
            "upstream_method": "GCG",
            "upstream_revision": "pinned-test",
            "attack_test_case_index": 0,
            "responses_create_params": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 512},
            "tags": ["hash_check", "book" if index < 350 else "lyrics"] if index >= 300 else [],
        }
        for index in range(400)
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        rollouts,
        [
            row
            | {
                "response": {"id": f"target-{index}", "model": model, "output": []},
                "score_label": int(index == 0),
                "classifier_label": int(index == 0) if index < 300 else None,
                "scoring_method": "harmbench_classifier" if index < 300 else "copyright_minhash",
                "functional_category": "standard",
                "semantic_category": "synthetic",
            }
            for index, row in enumerate(rows)
        ],
    )
    failures.write_text("", encoding="utf-8")
    aggregates.write_text(json.dumps([{"agent_metrics": {"attack_success_rate": 1 / 400}}]))
    quality.write_text(json.dumps({"run": {"verdicts": {"healthy": 400, "unhealthy": 0, "unobserved": 0}}}))
    source.write_text(
        json.dumps(
            {
                "method": "GCG",
                "upstream_revision": "pinned-test",
                "experiment": "super-gcg",
                "run_id": "attack-run",
                "target_type": "text_weights",
                "test_cases_sha256": "cases-sha",
                "behaviors_sha256": "behaviors-sha",
                "rows": 400,
                "behaviors": 400,
                "generation_receipt": {
                    "status": "completed",
                    "method": "GCG",
                    "upstream_method": "GCG",
                    "upstream_revision": "pinned-test",
                    "experiment": "super-gcg",
                    "run_id": "attack-run",
                    "source_target_model": model,
                    "source_target_revision": "model-revision",
                    "test_cases_sha256": "cases-sha",
                    "behaviors_sha256": "behaviors-sha",
                    "behaviors": 400,
                    "cases": 400,
                    "num_steps": 500,
                    "search_width": 512,
                    "num_shards": 2,
                    "shard_receipts": [{"name": "a"}, {"name": "b"}],
                    "shard_receipts_sha256": "a" * 64,
                },
            }
        )
    )
    classifier.write_text(
        json.dumps(
            {
                "classifier_model": CLASSIFIER_MODEL,
                "classifier_revision": CLASSIFIER_REVISION,
                "rollouts_sha256": sha256(rollouts),
                "rollout_cases": 300,
                "rollout_agreement": 300,
                "rollout_prompt_tokens_identical": 300,
                "rollout_prompt_sha256_matches": 300,
                "clip_checks": {"checked": 300, "text_identical": 300, "count_identical": 300},
                "fixture_cases": 1,
                "fixture_agreement": 1,
                "disagreements": [],
            }
        )
    )
    copyright_control.write_text(
        json.dumps(
            {
                "upstream_revision": "pinned-test",
                "rollouts_sha256": sha256(rollouts),
                "checked": 100,
                "agreement": 100,
                "mismatches": [],
            }
        )
    )
    manifest, _ = build(
        dataset=dataset,
        rollouts=rollouts,
        failures=failures,
        aggregate_metrics=aggregates,
        quality_summary=quality,
        source_manifest=source,
        model=model,
        run_id="target-run",
        classifier_calibration=classifier,
        copyright_calibration=copyright_control,
    )
    assert manifest["status"] == "validated"
    assert manifest["protocol_validation"] == "passed_gcg_generation_classifier_copyright_controls"
    assert manifest["target_revision"] == "model-revision"
    assert manifest["attack_profile"]["num_steps"] == 500
    assert manifest["completion_profile"] == {
        "generation_mode": "deterministic",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 512,
    }
    assert manifest["scoring_profile"]["classifier_cases"] == 300
    assert manifest["scoring_profile"]["copyright_cases"] == 100
    assert manifest["scoring_profile"]["copyright_book_cases"] == 50
    assert manifest["scoring_profile"]["copyright_lyrics_cases"] == 50

    broken = json.loads(source.read_text())
    broken["generation_receipt"]["search_width"] = 511
    source.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="GCG finalized source evidence"):
        build(
            dataset=dataset,
            rollouts=rollouts,
            failures=failures,
            aggregate_metrics=aggregates,
            quality_summary=quality,
            source_manifest=source,
            model=model,
            run_id="target-run",
            classifier_calibration=classifier,
            copyright_calibration=copyright_control,
        )
