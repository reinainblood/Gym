# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the exact full-corpus behaviors absent from an existing HarmBench result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def select(
    *,
    full_inputs: Path,
    existing_rollouts: Path,
    output: Path,
    expected_full_behaviors: int = 400,
    expected_existing_behaviors: int = 320,
    expected_missing_behaviors: int = 80,
    cases_per_behavior: int,
) -> dict[str, Any]:
    """Write complete behaviors absent from prior scored rollouts, preserving source order."""
    if cases_per_behavior <= 0:
        raise ValueError("cases_per_behavior must be positive")
    full = load_jsonl(full_inputs)
    existing = load_jsonl(existing_rollouts)
    if not full or not existing:
        raise ValueError("full inputs and existing rollouts must be nonempty")
    methods = {row.get("method") for row in full}
    existing_methods = {row.get("method") for row in existing}
    if len(methods) != 1 or existing_methods != methods:
        raise ValueError("full inputs and existing rollouts must contain one matching method")

    full_by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full:
        full_by_behavior[str(row["harmbench_id"])].append(row)
    existing_keys = Counter((str(row["harmbench_id"]), int(row.get("attack_test_case_index", 0))) for row in existing)
    if any(count != 1 for count in existing_keys.values()):
        raise ValueError("existing rollouts contain duplicate HarmBench cases")
    if len(full_by_behavior) != expected_full_behaviors:
        raise ValueError(f"expected {expected_full_behaviors} full-corpus behaviors, found {len(full_by_behavior)}")
    if any(len(rows) != cases_per_behavior for rows in full_by_behavior.values()):
        raise ValueError("full inputs do not have the expected cases per behavior")

    existing_behavior_ids = {behavior_id for behavior_id, _ in existing_keys}
    if len(existing_behavior_ids) != expected_existing_behaviors:
        raise ValueError(
            f"expected {expected_existing_behaviors} existing behaviors, found {len(existing_behavior_ids)}"
        )
    unknown = existing_behavior_ids - set(full_by_behavior)
    if unknown:
        raise ValueError("existing rollouts contain behaviors outside the full corpus")
    for behavior_id in existing_behavior_ids:
        expected_indexes = {int(row.get("attack_test_case_index", 0)) for row in full_by_behavior[behavior_id]}
        observed_indexes = {index for candidate, index in existing_keys if candidate == behavior_id}
        if observed_indexes != expected_indexes:
            raise ValueError(f"existing result partially covers behavior {behavior_id}")

    missing_ids = set(full_by_behavior) - existing_behavior_ids
    if len(missing_ids) != expected_missing_behaviors:
        raise ValueError(f"expected {expected_missing_behaviors} missing behaviors, found {len(missing_ids)}")
    selected = [row for row in full if str(row["harmbench_id"]) in missing_ids]
    expected_rows = expected_missing_behaviors * cases_per_behavior
    if len(selected) != expected_rows:
        raise ValueError(f"expected {expected_rows} validation rows, found {len(selected)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    full_manifest_path = full_inputs.with_suffix(".manifest.json")
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    models = sorted({row.get("response", {}).get("model") for row in existing if row.get("response", {}).get("model")})
    manifest = {
        **full_manifest,
        "artifact_kind": "harmbench_missing_validation_extension",
        "parent_full_inputs_sha256": sha256(full_inputs),
        "existing_rollouts_sha256": sha256(existing_rollouts),
        "existing_models": models,
        "existing_behaviors": len(existing_behavior_ids),
        "selected_behaviors": len(missing_ids),
        "rows": len(selected),
        "behaviors": len(missing_ids),
        "cases_per_behavior": cases_per_behavior,
        "selection": "complete behaviors absent from the existing scored result",
        "gym_inputs_sha256": sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-inputs", required=True, type=Path)
    parser.add_argument("--existing-rollouts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases-per-behavior", required=True, type=int)
    args = parser.parse_args()
    manifest = select(
        full_inputs=args.full_inputs,
        existing_rollouts=args.existing_rollouts,
        output=args.output,
        cases_per_behavior=args.cases_per_behavior,
    )
    print(
        json.dumps(
            {
                "method": manifest["method"],
                "behaviors": manifest["behaviors"],
                "rows": manifest["rows"],
                "gym_inputs_sha256": manifest["gym_inputs_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
