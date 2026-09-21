# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package a validated modern Qwen white-box HarmBench method result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.operations.qwen.qwen_harmbench_worker import (
    METHODS,
    MODEL_ID,
    MODEL_REVISION,
    UPSTREAM_REVISION,
)
from benchmarks.harmbench.operations.qwen.qwen_label_compare import sha256, validate_canonical_score_receipt
from benchmarks.harmbench.qwen_whitebox_blade import validate_bundle


MODERN_METHODS = {"MultiModalPGDBlankImage", "MultiModalPGDPatch"}


def build(
    *,
    run_id: str,
    method: str,
    attack_manifest_path: Path,
    completion_manifest_path: Path,
    classifier_scores_path: Path,
    blade_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Validate all modern white-box gates and return a normalized result."""
    if method not in MODERN_METHODS:
        raise ValueError("modern white-box result packaging is scoped to BlankImage and Patch")
    attack = json.loads(attack_manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(classifier_scores_path.read_text(encoding="utf-8"))
    attack_expected = {
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
    completion_expected = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_completion_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "completions": 110,
        "max_new_tokens": 512,
        "attack_manifest_sha256": sha256(attack_manifest_path),
    }
    for label, receipt, expected in (
        ("attack", attack, attack_expected),
        ("completion", completion, completion_expected),
    ):
        mismatched = [key for key, value in expected.items() if receipt.get(key) != value]
        if mismatched:
            raise ValueError(f"Qwen white-box {label} manifest failed fields {','.join(sorted(mismatched))}")
    completion_sha256 = sha256(completion_manifest_path)
    validate_canonical_score_receipt(
        scores,
        run_id=run_id,
        method=method,
        completion_manifest_sha256=completion_sha256,
    )
    validate_bundle(classifier_scores_path, blade_dir)
    blade_metrics = json.loads((blade_dir / "harmbench_blade_metrics.json").read_text(encoding="utf-8"))
    method_metrics = blade_metrics.get("methods", {}).get(method, {})
    invalid_or_unstable = scores["summary"]["invalid_label_cases"] + (
        110 - scores["summary"]["repeat_valid_agreement"] - scores["summary"]["invalid_label_cases"]
    )
    stable_successes = sum(len(set(case["labels"])) == 1 and case["labels"][0] == 1 for case in scores["cases"])
    if (
        blade_metrics.get("total_rows") != 110
        or method_metrics.get("asr_denominator") != 110 - invalid_or_unstable
        or method_metrics.get("asr_numerator") != stable_successes
    ):
        raise ValueError("Qwen white-box BLADE metrics disagree with canonical scorer evidence")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "harmbench_qwen_whitebox_validated_result",
        "status": "validated",
        "run_id": run_id,
        "method": method,
        "model": MODEL_ID,
        "target_revision": MODEL_REVISION,
        "attack_profile": {
            "upstream_revision": UPSTREAM_REVISION,
            **METHODS[method],
            "behaviors": 110,
            "cases": 110,
        },
        "completion_profile": {
            "generation_mode": "deterministic",
            "max_output_tokens": 512,
            "sampling": completion["sampling"],
        },
        "scoring_profile": {
            "classifier_model": scores["classifier_model"],
            "classifier_revision": scores["classifier_revision"],
            "classifier_repeats": scores["classifier_repeats"],
            "classifier_clip_tokens": scores["classifier_clip_tokens"],
        },
        "counts": {
            "expected_cases": 110,
            "scored_cases": 110,
            "successful_cases": scores["summary"]["successes"],
            "invalid_or_unstable_cases": invalid_or_unstable,
            "blade_model_denominator": method_metrics["asr_denominator"],
            "blade_successes": method_metrics["asr_numerator"],
        },
        "metrics": {
            "native_attack_success_rate": scores["summary"]["attack_success_rate"],
            "blade_test_case_asr": method_metrics["test_case_asr"],
            "blade_behavior_averaged_asr": method_metrics["behavior_averaged_asr"],
        },
        "provenance": {
            "attack_manifest_sha256": sha256(attack_manifest_path),
            "completion_manifest_sha256": completion_sha256,
            "classifier_scores_sha256": sha256(classifier_scores_path),
            "blade_evidence_manifest_sha256": sha256(blade_dir / "evidence-manifest.json"),
            "blade_metrics_sha256": sha256(blade_dir / "harmbench_blade_metrics.json"),
        },
    }
    report = (
        f"# HarmBench {method} — {MODEL_ID}\n\n"
        f"Status: **validated** · Run `{run_id}` · Upstream `{UPSTREAM_REVISION}`\n\n"
        f"Native attack success rate: **{scores['summary']['attack_success_rate']:.4%}** "
        f"({scores['summary']['successes']}/110).\n\n"
        f"BLADE valid denominator: **{method_metrics['asr_denominator']}**; excluded invalid or unstable classifier cases: "
        f"**{invalid_or_unstable}**. BLADE ASR: **{method_metrics['behavior_averaged_asr']:.4%}**.\n\n"
        "This report contains no behavior text, optimizer targets, attack images, or target generations.\n"
    )
    return manifest, report


def write(
    *,
    run_id: str,
    method: str,
    attack_manifest_path: Path,
    completion_manifest_path: Path,
    classifier_scores_path: Path,
    blade_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest, report = build(
        run_id=run_id,
        method=method,
        attack_manifest_path=attack_manifest_path,
        completion_manifest_path=completion_manifest_path,
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
        "artifact_kind": "harmbench_qwen_whitebox_result_bundle",
        "status": "completed",
        "inputs": manifest["provenance"],
        "outputs": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (manifest_path, report_path)
        ],
    }
    evidence_path = output_dir / "evidence-manifest.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rebuilt_manifest, rebuilt_report = build(
        run_id=run_id,
        method=method,
        attack_manifest_path=attack_manifest_path,
        completion_manifest_path=completion_manifest_path,
        classifier_scores_path=classifier_scores_path,
        blade_dir=blade_dir,
    )
    if json.loads(manifest_path.read_text(encoding="utf-8")) != rebuilt_manifest:
        raise ValueError("Qwen white-box result manifest failed readback")
    if report_path.read_text(encoding="utf-8") != rebuilt_report:
        raise ValueError("Qwen white-box result report failed readback")
    if json.loads(evidence_path.read_text(encoding="utf-8")) != evidence:
        raise ValueError("Qwen white-box result evidence manifest failed readback")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True, choices=sorted(MODERN_METHODS))
    parser.add_argument("--attack-manifest", required=True, type=Path)
    parser.add_argument("--completion-manifest", required=True, type=Path)
    parser.add_argument("--classifier-scores", required=True, type=Path)
    parser.add_argument("--blade-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
