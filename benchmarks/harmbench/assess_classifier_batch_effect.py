# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound one classifier-label mismatch to a measured vLLM batching effect.

This produces a payload-free caveat receipt. It does not change any stored
classifier label or convert a mismatching raw replay into exact parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(*, rollouts: Path, summary: Path, serial: Path, parallel: list[Path], comparison: Path) -> dict[str, Any]:
    calibration = json.loads(summary.read_text(encoding="utf-8"))
    if calibration["rollouts_sha256"] != _sha256(rollouts):
        raise ValueError("classifier replay does not cover these rollouts")
    count = calibration["rollout_cases"]
    if (
        len(calibration["disagreements"]) != 1
        or calibration["rollout_agreement"] != count - 1
        or calibration["rollout_prompt_tokens_identical"] != count
        or calibration["rollout_prompt_sha256_matches"] != count
        or calibration["fixture_agreement"] != calibration["fixture_cases"]
        or any(calibration["clip_checks"][key] != count for key in ("checked", "text_identical", "count_identical"))
    ):
        raise ValueError("classifier control has additional or unexplained parity failures")
    disagreement = calibration["disagreements"][0]
    if disagreement["source"] != "rollout" or disagreement["gym_label"] == disagreement["control_label"]:
        raise ValueError("classifier disagreement is not a one-case label mismatch")
    rows = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    matching = [
        row
        for row in rows
        if row["harmbench_id"] == disagreement["harmbench_id"]
        and row.get("classifier_label") == disagreement["gym_label"]
    ]
    if len(matching) != 1:
        raise ValueError("mismatch identity is ambiguous without a unique saved-label case")
    case = matching[0]
    serial_probe = json.loads(serial.read_text(encoding="utf-8"))
    identity = (case["harmbench_id"], case["attack_test_case_index"], case["classifier_prompt_sha256"])

    def check_probe(probe: dict, *, expected_label: int) -> None:
        if (
            (probe["behavior_id"], probe["attack_test_case_index"], probe["classifier_prompt_sha256"]) != identity
            or probe["rollouts_sha256"] != _sha256(rollouts)
            or probe["saved_label"] != disagreement["gym_label"]
            or not probe["prompt_token_ids_identical"]
            or probe["repeats"] < 10
            or set(probe["raw_labels"]) != {expected_label}
            or set(probe["chat_labels"]) != {expected_label}
        ):
            raise ValueError("classifier stability probe does not prove the claimed case and label")

    if serial_probe["parallel"] != 1:
        raise ValueError("serial probe must use one request at a time")
    check_probe(serial_probe, expected_label=disagreement["control_label"])
    parallel_probes = [json.loads(path.read_text(encoding="utf-8")) for path in parallel]
    if {probe["parallel"] for probe in parallel_probes} != {2, 4, 8}:
        raise ValueError("batch-effect control requires separate 2-, 4-, and 8-way probes")
    for probe in parallel_probes:
        check_probe(probe, expected_label=disagreement["gym_label"])
    diff = json.loads(comparison.read_text(encoding="utf-8"))
    if (
        diff["parent_rollouts_sha256"] != _sha256(rollouts)
        or not diff["target_responses_unchanged"]
        or diff["score_changes"]
        or diff["classifier_clip_changes"]
    ):
        raise ValueError("reverify did not preserve the original saved target responses and labels")
    return {
        "assessment": "one_batch_sensitive_classifier_case",
        "classifier_control_sha256": _sha256(summary),
        "rollouts_sha256": _sha256(rollouts),
        "reverify_comparison_sha256": _sha256(comparison),
        "behavior_id": identity[0],
        "attack_test_case_index": identity[1],
        "classifier_prompt_sha256": identity[2],
        "saved_label": disagreement["gym_label"],
        "serial_label": disagreement["control_label"],
        "batch_concurrencies": [2, 4, 8],
        "serial_repeats": serial_probe["repeats"],
        "batched_repeats": sum(probe["repeats"] for probe in parallel_probes),
        "prompt_token_and_clip_parity_cases": count,
        "raw_replay_agreement": count - 1,
        "classifier_cases": count,
        "no_score_override": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--parallel", required=True, type=Path, action="append")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = assess(
        rollouts=args.rollouts,
        summary=args.summary,
        serial=args.serial,
        parallel=args.parallel,
        comparison=args.comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Bound one classifier mismatch to serial-versus-batched inference; saved score unchanged")


if __name__ == "__main__":
    main()
