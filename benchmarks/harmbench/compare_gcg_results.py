# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare validated Super and Qwen GCG manifests under one matched fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.prepare_generated import sha256


PRIMARY_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
BASELINE_MODEL = "Qwen/Qwen3.5-122B-A10B"
PROFILE_KEYS = {
    "attack_profile": ("upstream_revision", "num_steps", "search_width", "behaviors", "cases"),
    "completion_profile": ("generation_mode", "max_output_tokens"),
    "scoring_profile": (
        "classifier_model",
        "classifier_revision",
        "classifier_cases",
        "copyright_cases",
        "copyright_book_cases",
        "copyright_lyrics_cases",
    ),
}


def _profile(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        section: {key: manifest.get(section, {}).get(key) for key in keys} for section, keys in PROFILE_KEYS.items()
    }


def build(*, primary_path: Path, baseline_path: Path) -> tuple[dict[str, Any], str]:
    """Validate comparability and return a payload-free comparison manifest and report."""
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for label, manifest, model in (
        ("primary", primary, PRIMARY_MODEL),
        ("baseline", baseline, BASELINE_MODEL),
    ):
        required = {
            "artifact_kind": "harmbench_gcg_validated_result",
            "status": "validated",
            "method": "GCG",
            "model": model,
        }
        mismatched = [key for key, value in required.items() if manifest.get(key) != value]
        counts = manifest.get("counts", {})
        if (
            counts.get("expected_cases") != 400
            or counts.get("scored_cases") != 400
            or counts.get("failed_cases") != 0
            or counts.get("missing_cases") != 0
        ):
            mismatched.append("counts")
        if mismatched:
            raise ValueError(f"GCG {label} manifest failed fields {','.join(sorted(set(mismatched)))}")

    primary_profile = _profile(primary)
    baseline_profile = _profile(baseline)
    if primary_profile != baseline_profile:
        raise ValueError("GCG primary and baseline profiles are not comparable")
    canonical_profile = json.dumps(primary_profile, sort_keys=True, separators=(",", ":")).encode()
    profile_sha256 = hashlib.sha256(canonical_profile).hexdigest()

    primary_asr = primary.get("metrics", {}).get("native_behavior_averaged_attack_success_rate")
    baseline_asr = baseline.get("metrics", {}).get("native_behavior_averaged_attack_success_rate")
    if not isinstance(primary_asr, int | float) or not isinstance(baseline_asr, int | float):
        raise ValueError("GCG manifests are missing native behavior-averaged ASR")
    for value in (primary_asr, baseline_asr):
        if not 0 <= value <= 1:
            raise ValueError("GCG ASR is outside the valid range")

    comparison = {
        "schema_version": 1,
        "artifact_kind": "harmbench_gcg_matched_comparison",
        "status": "validated",
        "method": "GCG",
        "profile": primary_profile,
        "profile_sha256": profile_sha256,
        "primary": {
            "model": primary["model"],
            "target_revision": primary["target_revision"],
            "run_id": primary["run_id"],
            "behavior_averaged_attack_success_rate": primary_asr,
            "successful_cases": primary["counts"]["successful_cases"],
            "expected_cases": primary["counts"]["expected_cases"],
        },
        "baseline": {
            "model": baseline["model"],
            "target_revision": baseline["target_revision"],
            "run_id": baseline["run_id"],
            "behavior_averaged_attack_success_rate": baseline_asr,
            "successful_cases": baseline["counts"]["successful_cases"],
            "expected_cases": baseline["counts"]["expected_cases"],
        },
        "observed_asr_delta_primary_minus_baseline": primary_asr - baseline_asr,
        "interpretation_boundary": (
            "descriptive matched-profile difference only; no significance or causal superiority claim"
        ),
        "provenance": {
            "primary_manifest_sha256": sha256(primary_path),
            "baseline_manifest_sha256": sha256(baseline_path),
        },
    }
    report = (
        "# HarmBench GCG matched comparison\n\n"
        f"Status: **validated** · Profile `{profile_sha256}`\n\n"
        f"Primary — `{primary['model']}`: **{primary_asr:.4%}** behavior-averaged ASR "
        f"({primary['counts']['successful_cases']}/400 successful cases).\n\n"
        f"Matched baseline — `{baseline['model']}`: **{baseline_asr:.4%}** behavior-averaged ASR "
        f"({baseline['counts']['successful_cases']}/400 successful cases).\n\n"
        f"Observed primary-minus-baseline ASR difference: **{primary_asr - baseline_asr:+.4%}**.\n\n"
        "This is a descriptive matched-profile comparison, not a statistical-significance result or causal superiority "
        "claim. Both inputs independently passed full source, completion, mixed-scorer, reconciliation, and evidence "
        "gates. No attack strings or target generations are included.\n"
    )
    return comparison, report


def write(*, primary_path: Path, baseline_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest, report = build(primary_path=primary_path, baseline_path=baseline_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "comparison-manifest.json"
    report_path = output_dir / "comparison-report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "artifact_kind": "harmbench_gcg_comparison_bundle",
        "status": "completed",
        "inputs": manifest["provenance"],
        "outputs": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (manifest_path, report_path)
        ],
    }
    evidence_path = output_dir / "evidence-manifest.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rebuilt_manifest, rebuilt_report = build(primary_path=primary_path, baseline_path=baseline_path)
    if json.loads(manifest_path.read_text(encoding="utf-8")) != rebuilt_manifest:
        raise ValueError("GCG comparison manifest failed readback")
    if report_path.read_text(encoding="utf-8") != rebuilt_report:
        raise ValueError("GCG comparison report failed readback")
    observed_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if observed_evidence != evidence:
        raise ValueError("GCG comparison evidence manifest failed readback")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            write(primary_path=args.primary, baseline_path=args.baseline, output_dir=args.output_dir), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
