# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved target responses and labels before/after Gym reverification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.report_method_run import _asr


def _rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    keyed = {(row["harmbench_id"], int(row.get("attack_test_case_index", 0))): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("reverification comparison found duplicate case identities")
    return keyed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(parent: Path, reverified: Path) -> dict[str, Any]:
    before = _rows(parent)
    after = _rows(reverified)
    if before.keys() != after.keys():
        raise ValueError("reverification changed the case set")
    changed_scores = []
    changed_clips = []
    for key in sorted(before):
        original = before[key]
        current = after[key]
        for field in ("id", "model", "output"):
            if original["response"][field] != current["response"][field]:
                raise ValueError(f"reverification changed saved target response {key} field {field}")
        if original.get("scoring_method") != current.get("scoring_method"):
            raise ValueError(f"reverification changed scorer selection for {key}")
        if original.get("score_label") != current.get("score_label"):
            changed_scores.append(
                {
                    "harmbench_id": key[0],
                    "attack_test_case_index": key[1],
                    "before": original.get("score_label"),
                    "after": current.get("score_label"),
                    "scoring_method": current.get("scoring_method"),
                }
            )
        if original.get("generation_for_classifier") != current.get("generation_for_classifier"):
            changed_clips.append({"harmbench_id": key[0], "attack_test_case_index": key[1]})
    return {
        "parent_rollouts_sha256": _sha256(parent),
        "reverified_rollouts_sha256": _sha256(reverified),
        "cases": len(before),
        "target_responses_unchanged": True,
        "before_behavior_averaged_asr": _asr(list(before.values())),
        "after_behavior_averaged_asr": _asr(list(after.values())),
        "score_changes": changed_scores,
        "classifier_clip_changes": changed_clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--reverified", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.parent, args.reverified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Compared {result['cases']} saved responses: {len(result['score_changes'])} score changes, "
        f"{len(result['classifier_clip_changes'])} clip changes"
    )


if __name__ == "__main__":
    main()
