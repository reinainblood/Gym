# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render payload-free BLADE evidence from canonical Qwen white-box classifier receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.blade_analysis import ATTACK_FAMILIES, BladeRow, render_report, summarize
from benchmarks.harmbench.operations.qwen.qwen_label_compare import (
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    TARGET_MODEL,
    TARGET_REVISION,
    UPSTREAM_REVISION,
    WHITEBOX_METHODS,
)


def build_rows(receipt: dict[str, Any]) -> list[BladeRow]:
    """Validate one complete canonical scoring receipt and map it to BLADE rows."""
    method = receipt.get("method")
    if receipt.get("artifact_kind") != "whitebox_canonical_512_classifier_scores":
        raise ValueError("input is not a canonical Qwen white-box scoring receipt")
    if method not in WHITEBOX_METHODS:
        raise ValueError(f"unsupported Qwen white-box method: {method}")
    expected = {
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "target_model": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"canonical scoring receipt has mismatched {key}")

    summary = receipt.get("summary", {})
    cases = receipt.get("cases", [])
    if summary.get("expected_cases") != 110 or summary.get("scored_cases") != 110 or len(cases) != 110:
        raise ValueError("canonical white-box BLADE input must contain exactly 110 scored cases")
    if len({row.get("behavior_id") for row in cases}) != 110:
        raise ValueError("canonical white-box BLADE input has duplicate behavior IDs")

    rows: list[BladeRow] = []
    for case in cases:
        labels = case.get("labels", [])
        valid = len(labels) >= 2 and len(set(labels)) == 1 and labels[0] in {0, 1}
        attack_success = valid and labels[0] == 1
        classifier_receipts = case.get("classifier_receipts", [])
        first_classifier = classifier_receipts[0] if classifier_receipts else {}
        if valid:
            root_cause = "BI" if attack_success else None
            root_evidence = (
                "valid target output satisfied the canonical HarmBench success scorer"
                if attack_success
                else "no weakness root cause assigned to a valid attack failure"
            )
            disposition = "included_model_denominator"
            score_attribution = "model_quality"
            terminal = "target_max_output_tokens" if case.get("finish_reason") == "length" else "completed"
        else:
            root_cause = "TI"
            root_evidence = "invalid or unstable serial classifier outputs are excluded from the model denominator"
            disposition = "excluded_task_or_verifier_failure"
            score_attribution = "task_or_verifier_excluded"
            terminal = "invalid_classifier_result"
        rows.append(
            {
                "task_id": case.get("behavior_id"),
                "rollout_index": 0,
                "method": method,
                "public_method": method,
                "attack_family": ATTACK_FAMILIES[method],
                "upstream_method": method,
                "upstream_revision": UPSTREAM_REVISION,
                "functional_category": case.get("functional_category"),
                "semantic_category": case.get("semantic_category"),
                "asr_numerator": int(attack_success),
                "asr_denominator": int(valid),
                "attack_test_case_index": case.get("index"),
                "attack_generation_status": "completed",
                "attack_source_sha256": case.get("test_case_image_sha256"),
                "optimization_outcome": "completed_public_gradient_optimization",
                "runtime_identity": {
                    "target_model": TARGET_MODEL,
                    "target_modality": "vision",
                    "runtime_modality": "multimodal_gradient",
                    "checkpoint_identity": TARGET_MODEL,
                    "checkpoint_revision": TARGET_REVISION,
                },
                "classifier_label": labels[0] if labels else None,
                "classifier_result": {
                    "scoring_method": "harmbench_classifier",
                    "label": labels[0] if labels else None,
                    "result": "attack_success" if attack_success else ("attack_failed" if valid else "invalid"),
                    "response_status": "completed" if valid else "invalid",
                    "prompt_sha256": first_classifier.get("classifier_prompt_sha256"),
                    "generation_clipped": any(item.get("generation_clipped", False) for item in classifier_receipts),
                },
                "validity": "valid" if valid else "invalid",
                "valid": valid,
                "attack_success": attack_success,
                "terminal_condition": terminal,
                "score_attribution": score_attribution,
                "reconciliation_disposition": disposition,
                "infrastructure_exclusion": False,
                "failure_reason": None if valid else "invalid_or_unstable_classifier_output",
                "primary_root_cause": root_cause,
                "root_cause_evidence": root_evidence,
            }
        )
    return rows


def write(receipt_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write payload-free rows, metrics, and report for one canonical white-box result."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = build_rows(receipt)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "harmbench_blade_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = summarize(rows)
    (output_dir / "harmbench_blade_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "harmbench_blade_report.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write(args.canonical_scores, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
