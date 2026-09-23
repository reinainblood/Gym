# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.blade_analysis import (
    ROOT_CAUSE_CODES,
    map_failure_sidecar,
    map_rollout,
    summarize,
    validate_bundle,
    write_analysis,
)


def _row(*, task: str, success: int, method: str = "PAP-top5"):
    return {
        "harmbench_id": task,
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
        "method": method,
        "upstream_method": method,
        "upstream_revision": "pinned",
        "functional_category": "standard",
        "semantic_category": "cybercrime_intrusion",
        "attack_test_case_index": 0,
        "attack_source_sha256": "a" * 64,
        "generation": "substantive target output",
        "generation_empty": False,
        "generation_truncated": False,
        "generation_clipped": False,
        "response": {"model": "nvidia/model", "status": "completed"},
        "score_label": success,
        "classifier_label": success,
        "scoring_method": "harmbench_classifier",
        "classifier_prompt_sha256": "b" * 64,
        "classifier_response": {"status": "completed"},
        "failure_reason": None,
        "reward": 1.0 - success,
    }


def test_mapping_preserves_native_score_and_required_blade_fields():
    mapped = map_rollout(_row(task="a", success=1))
    assert mapped["public_method"] == "PAP-top5"
    assert mapped["method"] == "PAP-top5"
    assert mapped["attack_family"] == "language_model_attack"
    assert mapped["attack_success"] is True
    assert mapped["asr_numerator"] == 1
    assert mapped["asr_denominator"] == 1
    assert mapped["score_attribution"] == "model_quality"
    assert mapped["primary_root_cause"] == "BI"
    assert mapped["runtime_identity"] == {
        "target_model": "nvidia/model",
        "target_modality": "text",
        "runtime_modality": "text_api",
        "checkpoint_identity": "nvidia/model",
        "checkpoint_revision": None,
    }
    assert mapped["classifier_result"]["scoring_method"] == "harmbench_classifier"
    assert mapped["classifier_label"] == 1
    assert mapped["validity"] == "valid"
    assert mapped["reconciliation_disposition"] == "included_model_denominator"
    assert mapped["infrastructure_exclusion"] is False


def test_behavior_averaged_asr_and_case_asr_keep_denominators_distinct():
    records = [
        map_rollout(_row(task="a", success=1)),
        map_rollout(_row(task="a", success=0)),
        map_rollout(_row(task="b", success=0)),
    ]
    metrics = summarize(records)["methods"]["PAP-top5"]
    assert metrics["test_case_asr"] == 1 / 3
    assert metrics["behavior_averaged_asr"] == 0.25
    assert metrics["asr_numerator"] == 1
    assert metrics["asr_denominator"] == 3
    assert metrics["behavior_denominator"] == 2


def test_infrastructure_failure_is_excluded_from_model_denominator():
    row = _row(task="a", success=0)
    row["generation"] = ""
    row["failure_reason"] = "provider timeout"
    mapped = map_rollout(row)
    assert mapped["valid"] is False
    assert mapped["score_attribution"] == "infrastructure_excluded"
    assert mapped["primary_root_cause"] == "IR"
    assert mapped["reconciliation_disposition"] == "excluded_infrastructure_failure"
    assert mapped["infrastructure_exclusion"] is True
    metrics = summarize([mapped])["methods"]["PAP-top5"]
    assert metrics["asr_denominator"] == 0
    assert metrics["infrastructure_excluded"] == 1


def test_writer_emits_row_ledger_and_metrics(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(_row(task="a", success=0)) + "\n", encoding="utf-8")
    output = tmp_path / "blade"
    metrics = write_analysis(
        inputs=[source],
        output_dir=output,
        attack_generation_status="completed",
        optimization_outcome=None,
    )
    assert metrics["total_rows"] == 1
    assert (output / "harmbench_blade_rows.jsonl").is_file()
    assert (output / "harmbench_blade_report.md").is_file()
    assert json.loads((output / "harmbench_blade_metrics.json").read_text())["benchmark"] == "HarmBench"
    manifest = validate_bundle(output)
    assert manifest["status"] == "completed"
    assert manifest["rows"] == 1
    assert len(manifest["outputs"]) == 3

    (output / "harmbench_blade_report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="readback disagrees"):
        validate_bundle(output)


def test_failure_sidecar_uses_structured_fields_and_never_enters_asr():
    sidecar = {
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_type": "ClientResponseError",
        "_ng_failure_message": "503 service unavailable",
        "_ng_failure_http_status": 503,
        "_ng_task_index": 4,
        "_ng_rollout_index": 0,
    }
    materialized = _row(task="structured-task", success=0, method="GCG")
    materialized["_ng_task_index"] = 4
    mapped = map_failure_sidecar(sidecar, materialized=materialized)
    assert mapped["asr_numerator"] == 0
    assert mapped["asr_denominator"] == 0
    assert mapped["primary_root_cause"] == "IR"
    assert mapped["terminal_condition"] == "sidecar:agent_run_error:ClientResponseError"
    assert mapped["score_attribution"] == "infrastructure_excluded"


def test_primary_root_cause_is_always_constrained():
    rows = [map_rollout(_row(task="a", success=1)), map_rollout(_row(task="b", success=0))]
    for row in rows:
        assert row["primary_root_cause"] is None or row["primary_root_cause"] in ROOT_CAUSE_CODES


def test_writer_reconciles_failure_sidecars(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(_row(task="a", success=0)) + "\n", encoding="utf-8")
    failures = tmp_path / "rollouts_failures.jsonl"
    failures.write_text(
        json.dumps(
            {
                "_ng_failure_class": "agent_run_error",
                "_ng_failure_type": "TimeoutError",
                "_ng_failure_message": "provider timeout",
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "blade"
    metrics = write_analysis(
        inputs=[source],
        failure_sidecars=[failures],
        materialized_inputs=[],
        output_dir=output,
        attack_generation_status="completed",
        optimization_outcome=None,
    )
    assert metrics["total_rows"] == 2
    assert metrics["valid_rows"] == 1
    assert metrics["reconciliation"]["excluded_infrastructure_failure"] == 1
