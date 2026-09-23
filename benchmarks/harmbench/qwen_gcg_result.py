# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package a validated, payload-free exact-BF16 Qwen GCG result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    CLASSIFIER_MODEL,
    CLASSIFIER_REVISION,
    sha256,
)
from benchmarks.harmbench.qwen_gcg_blade import validate_bundle


def build(
    *,
    generation_receipt_path: Path,
    completion_receipt_path: Path,
    classifier_scores_path: Path,
    blade_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Validate every Qwen GCG gate and return a normalized manifest and report."""
    generation = json.loads(generation_receipt_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_receipt_path.read_text(encoding="utf-8"))
    scores = json.loads(classifier_scores_path.read_text(encoding="utf-8"))
    generation_expected = {
        "status": "completed",
        "method": "GCG",
        "upstream_method": "GCG",
        "upstream_revision": UPSTREAM_REVISION,
        "source_target_model": MODEL_ID,
        "source_target_revision": MODEL_REVISION,
        "behaviors": PUBLIC_BEHAVIORS,
        "cases": PUBLIC_BEHAVIORS,
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
    }
    completion_expected = {
        "status": "completed",
        "artifact_kind": "gcg_target_completion_manifest",
        "target": "qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "GCG",
        "behaviors": PUBLIC_BEHAVIORS,
        "completions": PUBLIC_BEHAVIORS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_receipt_sha256": sha256(generation_receipt_path),
    }
    scores_expected = {
        "status": "completed",
        "artifact_kind": "gcg_canonical_512_classifier_scores",
        "method": "GCG",
        "classifier_model": CLASSIFIER_MODEL,
        "classifier_revision": CLASSIFIER_REVISION,
        "target_model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "target_completion_manifest_sha256": sha256(completion_receipt_path),
    }
    for label, receipt, expected in (
        ("generation", generation, generation_expected),
        ("completion", completion, completion_expected),
        ("scores", scores, scores_expected),
    ):
        mismatched = [key for key, value in expected.items() if receipt.get(key) != value]
        if mismatched:
            raise ValueError(f"Qwen GCG {label} receipt failed fields {','.join(sorted(mismatched))}")
    artifact_ids = {generation.get("run_id"), completion.get("artifact_id"), scores.get("artifact_id")}
    if len(artifact_ids) != 1 or None in artifact_ids:
        raise ValueError("Qwen GCG artifacts do not share one immutable artifact ID")

    summary = scores.get("summary", {})
    expected_summary = {
        "expected_cases": PUBLIC_BEHAVIORS,
        "scored_cases": PUBLIC_BEHAVIORS,
        "classifier_cases": 300,
        "copyright_cases": 100,
        "copyright_book_cases": 50,
        "copyright_lyrics_cases": 50,
    }
    mismatched_summary = [key for key, value in expected_summary.items() if summary.get(key) != value]
    if mismatched_summary or len(scores.get("cases", [])) != PUBLIC_BEHAVIORS:
        raise ValueError("Qwen GCG score receipt does not reconcile the complete mixed-scoring corpus")

    validate_bundle(classifier_scores_path, blade_dir)
    blade_metrics = json.loads((blade_dir / "harmbench_blade_metrics.json").read_text(encoding="utf-8"))
    method_metrics = blade_metrics.get("methods", {}).get("GCG", {})
    invalid_or_unstable = sum(
        len(set(case.get("labels", []))) != 1 or case.get("labels", [None])[0] not in {0, 1}
        for case in scores["cases"]
    )
    stable_successes = sum(
        len(set(case.get("labels", []))) == 1 and case["labels"][0] == 1 for case in scores["cases"]
    )
    if (
        blade_metrics.get("total_rows") != PUBLIC_BEHAVIORS
        or method_metrics.get("asr_denominator") != PUBLIC_BEHAVIORS - invalid_or_unstable
        or method_metrics.get("asr_numerator") != stable_successes
    ):
        raise ValueError("Qwen GCG BLADE metrics disagree with classifier case evidence")

    artifact_id = artifact_ids.pop()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "harmbench_gcg_validated_result",
        "status": "validated",
        "run_id": artifact_id,
        "method": "GCG",
        "model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "attack_profile": {
            "upstream_revision": UPSTREAM_REVISION,
            "num_steps": PUBLIC_STEPS,
            "search_width": PUBLIC_SEARCH_WIDTH,
            "behaviors": PUBLIC_BEHAVIORS,
            "cases": PUBLIC_BEHAVIORS,
            "num_shards": generation["num_shards"],
        },
        "completion_profile": {
            "generation_mode": "deterministic",
            "max_output_tokens": MAX_NEW_TOKENS,
        },
        "scoring_profile": {
            "classifier_model": CLASSIFIER_MODEL,
            "classifier_revision": CLASSIFIER_REVISION,
            "classifier_cases": 300,
            "copyright_cases": 100,
            "copyright_book_cases": 50,
            "copyright_lyrics_cases": 50,
        },
        "counts": {
            "expected_cases": PUBLIC_BEHAVIORS,
            "scored_cases": PUBLIC_BEHAVIORS,
            "failed_cases": 0,
            "missing_cases": 0,
            "successful_cases": summary["successes"],
            "invalid_or_unstable_cases": invalid_or_unstable,
            "blade_model_denominator": method_metrics["asr_denominator"],
            "blade_successes": method_metrics["asr_numerator"],
        },
        "metrics": {
            "native_attack_success_rate": summary["attack_success_rate"],
            "native_behavior_averaged_attack_success_rate": summary["behavior_averaged_attack_success_rate"],
            "blade_test_case_asr": method_metrics["test_case_asr"],
            "blade_behavior_averaged_asr": method_metrics["behavior_averaged_asr"],
        },
        "provenance": {
            "generation_receipt_sha256": sha256(generation_receipt_path),
            "completion_receipt_sha256": sha256(completion_receipt_path),
            "classifier_scores_sha256": sha256(classifier_scores_path),
            "blade_evidence_manifest_sha256": sha256(blade_dir / "evidence-manifest.json"),
            "blade_metrics_sha256": sha256(blade_dir / "harmbench_blade_metrics.json"),
        },
    }
    report = (
        f"# HarmBench GCG — {MODEL_ID}\n\n"
        f"Status: **validated** · Artifact `{artifact_id}` · Upstream `{UPSTREAM_REVISION}`\n\n"
        f"Native behavior-averaged ASR: **{summary['behavior_averaged_attack_success_rate']:.4%}** "
        f"({summary['successes']}/{PUBLIC_BEHAVIORS}).\n\n"
        f"BLADE valid denominator: **{method_metrics['asr_denominator']}**; excluded invalid or unstable scorer cases: "
        f"**{invalid_or_unstable}**. BLADE behavior-averaged ASR: "
        f"**{method_metrics['behavior_averaged_asr']:.4%}**.\n\n"
        "Scoring routing: 300 pinned HarmBench classifier cases and 100 pinned MinHash copyright cases "
        "(50 book, 50 lyrics). This report contains no attack strings or target generations.\n"
    )
    return manifest, report


def write(
    *,
    generation_receipt_path: Path,
    completion_receipt_path: Path,
    classifier_scores_path: Path,
    blade_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest, report = build(
        generation_receipt_path=generation_receipt_path,
        completion_receipt_path=completion_receipt_path,
        classifier_scores_path=classifier_scores_path,
        blade_dir=blade_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run-manifest.json"
    report_path = output_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "artifact_kind": "harmbench_gcg_result_bundle",
        "status": "completed",
        "inputs": manifest["provenance"],
        "outputs": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (manifest_path, report_path)
        ],
    }
    evidence_path = output_dir / "evidence-manifest.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readback_manifest, readback_report = build(
        generation_receipt_path=generation_receipt_path,
        completion_receipt_path=completion_receipt_path,
        classifier_scores_path=classifier_scores_path,
        blade_dir=blade_dir,
    )
    if json.loads(manifest_path.read_text(encoding="utf-8")) != readback_manifest:
        raise ValueError("Qwen GCG normalized run manifest failed readback")
    if report_path.read_text(encoding="utf-8") != readback_report:
        raise ValueError("Qwen GCG normalized report failed readback")
    if json.loads(evidence_path.read_text(encoding="utf-8")) != evidence:
        raise ValueError("Qwen GCG result evidence manifest failed readback")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument("--completion-receipt", required=True, type=Path)
    parser.add_argument("--classifier-scores", required=True, type=Path)
    parser.add_argument("--blade-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
