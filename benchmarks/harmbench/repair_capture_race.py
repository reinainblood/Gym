# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reattach a late model-call capture to one saved Gym rollout.

This preserves the original rollout JSONL and target response, writing a
lineage-receipted derivative rather than rerunning inference or changing a
score. Use only after independently confirming the capture file exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record
from nemo_gym.rollout_collection import _attach_ng_perf, _attach_trajectory_record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(*, rollouts: Path, inputs: Path, capture_dir: Path, task_index: int, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    capture = capture_dir / f"{task_index}-0.capture.jsonl"
    if not capture.is_file():
        raise FileNotFoundError(capture)
    source_lines = rollouts.read_text(encoding="utf-8").splitlines(keepends=True)
    input_rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
    if len(source_lines) != len(input_rows):
        raise ValueError("materialized inputs and saved rollouts have different counts")
    changed = 0
    response_id = None
    for index, line in enumerate(source_lines):
        row = json.loads(line)
        if row.get("_ng_task_index") != task_index or row.get("_ng_rollout_index") != 0:
            continue
        before = copy.deepcopy(row)
        if row.get("ng_model_call_capture", {}).get("gaps") != [{"code": "model_call_capture_no_records"}]:
            raise ValueError("target rollout does not have exactly the expected capture race gap")
        matching_inputs = [item for item in input_rows if item.get("_ng_task_index") == task_index]
        if len(matching_inputs) != 1:
            raise ValueError("target task is absent or duplicated in materialized inputs")
        latency = row.get("ng_perf", {}).get("total_latency_ms")
        merge_model_call_capture_into_record(row, [capture_dir], include_payloads=False)
        if len(row.get("ng_model_call_capture", {}).get("calls", [])) != 1:
            raise ValueError("late capture did not resolve to exactly one target model call")
        if row["ng_model_call_capture"].get("gaps"):
            raise ValueError("late capture still has observation gaps")
        trajectory = row.get("ng_trajectory")
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("gaps"), list):
            raise ValueError("target rollout is missing the original standardized trajectory")
        stale = {"model_call_capture_no_records", "model_calls_unavailable"}
        observed_stale = {gap.get("code") for gap in trajectory["gaps"] if isinstance(gap, dict)} & stale
        if observed_stale != stale:
            raise ValueError("target trajectory lacks the two expected capture-derived gaps")
        trajectory["gaps"] = [gap for gap in trajectory["gaps"] if gap.get("code") not in stale]
        _attach_trajectory_record(matching_inputs[0], row)
        _attach_ng_perf(row, observability_enabled=True, rollout_latency_ms=latency)
        observability_fields = {"ng_model_call_capture", "ng_trajectory", "ng_perf"}
        protected_before = {key: value for key, value in before.items() if key not in observability_fields}
        protected_after = {key: value for key, value in row.items() if key not in observability_fields}
        if protected_after != protected_before:
            raise ValueError("observability repair changed a non-observability field")
        response_id = row["response"]["id"]
        source_lines[index] = json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        changed += 1
    if changed != 1:
        raise ValueError(f"expected one matching rollout; found {changed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(source_lines), encoding="utf-8")
    receipt = {
        "repair": "late_model_call_capture_reattachment",
        "task_index": task_index,
        "response_id": response_id,
        "source_rollouts_sha256": _sha256(rollouts),
        "materialized_inputs_sha256": _sha256(inputs),
        "late_capture_sha256": _sha256(capture),
        "derived_rollouts_sha256": _sha256(output),
        "rows": len(source_lines),
        "target_responses_and_scores_unchanged": True,
    }
    output.with_name("capture-repair-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = repair(
        rollouts=args.rollouts,
        inputs=args.inputs,
        capture_dir=args.capture_dir,
        task_index=args.task_index,
        output=args.output,
    )
    print(f"Reattached one late capture for task {receipt['task_index']} across {receipt['rows']} preserved rows")


if __name__ == "__main__":
    main()
