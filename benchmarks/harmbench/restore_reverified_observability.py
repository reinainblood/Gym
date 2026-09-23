# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore preserved model-call evidence after stateless Gym reverification.

Gym's reverify output keeps target responses and new scores but does not carry
the original `ng_*` observability fields. This creates a new derivative that
reattaches those fields only after verifying each target response is identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


OBSERVABILITY_FIELDS = ("ng_model_call_capture", "ng_trajectory", "ng_perf")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return row["harmbench_id"], int(row.get("attack_test_case_index", 0))


def restore(*, parent: Path, reverified: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    parent_rows = [json.loads(line) for line in parent.read_text(encoding="utf-8").splitlines()]
    reverified_rows = [json.loads(line) for line in reverified.read_text(encoding="utf-8").splitlines()]
    parent_by_case = {_key(row): row for row in parent_rows}
    if len(parent_by_case) != len(parent_rows) or len(reverified_rows) != len(parent_rows):
        raise ValueError("parent or reverified rollouts are missing or duplicate cases")
    observed: list[dict[str, Any]] = []
    for row in reverified_rows:
        original = parent_by_case.get(_key(row))
        if original is None:
            raise ValueError("reverified rollout has no parent case")
        if any(original["response"].get(field) != row["response"].get(field) for field in ("id", "model", "output")):
            raise ValueError("reverification changed a saved target response")
        if original["method"] != row["method"] or original["attack_source_sha256"] != row["attack_source_sha256"]:
            raise ValueError("reverified method or attack source differs from parent")
        if any(field in row for field in OBSERVABILITY_FIELDS):
            raise ValueError("reverified rollout already has observability fields; refusing to overwrite")
        capture = original.get("ng_model_call_capture")
        trajectory = original.get("ng_trajectory")
        if (
            not isinstance(capture, dict)
            or not capture.get("calls")
            or capture.get("gaps")
            or not isinstance(trajectory, dict)
            or not trajectory.get("model_calls")
        ):
            raise ValueError("parent rollout lacks complete captured model-call evidence")
        enriched = copy.deepcopy(row)
        for field in OBSERVABILITY_FIELDS:
            if field in original:
                enriched[field] = copy.deepcopy(original[field])
        observed.append(enriched)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in observed), encoding="utf-8"
    )
    receipt = {
        "operation": "restore_saved_observability_after_reverify",
        "parent_rollouts_sha256": _sha256(parent),
        "reverified_rollouts_sha256": _sha256(reverified),
        "observed_reverified_rollouts_sha256": _sha256(output),
        "cases": len(observed),
        "target_responses_unchanged": True,
        "scores_from_reverified_run": True,
        "observability_from_parent_run": True,
    }
    output.with_name("observability-restore-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--reverified", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = restore(parent=args.parent, reverified=args.reverified, output=args.output)
    print(f"Restored saved observability for {receipt['cases']} unchanged target responses")


if __name__ == "__main__":
    main()
