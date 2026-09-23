# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate finalized client-fresh PAIR/TAP generation without exposing payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


try:
    from benchmarks.harmbench.client_fresh_generate import UPSTREAM_REVISION
except ModuleNotFoundError:  # Standalone Modal image mounts the module at /app.
    from client_fresh_generate import UPSTREAM_REVISION


TARGET_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
TARGET_REVISION = "hf-ea-0e636f7"
PUBLIC_METHODS = {
    "PAIR": "Fresh PAIR against the client model",
    "TAP": "Fresh TAP against the client model",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    *, method: str, artifact_root: Path, behaviors: Path, generation_receipt: Path, output: Path
) -> dict[str, Any]:
    if method not in PUBLIC_METHODS:
        raise ValueError("method must be PAIR or TAP")
    final = json.loads(generation_receipt.read_text(encoding="utf-8"))
    cases_path = artifact_root / "test_cases.json"
    with behaviors.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    ordered_ids = [row["BehaviorID"] for row in source_rows]
    expected = {
        "status": "completed",
        "method": PUBLIC_METHODS[method],
        "upstream_method": method,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": "super-vl-client-fresh",
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behaviors),
        "behaviors": 400,
        "cases": 400,
        "client_model": TARGET_MODEL,
        "source_target_model": TARGET_MODEL,
        "client_revision": TARGET_REVISION,
        "source_target_revision": TARGET_REVISION,
    }
    mismatched = [key for key, value in expected.items() if final.get(key) != value]
    num_shards = final.get("num_shards")
    if not isinstance(num_shards, int) or not 1 <= num_shards <= 16:
        mismatched.append("num_shards")
    if mismatched:
        raise ValueError(f"client-fresh generation receipt failed {','.join(sorted(set(mismatched)))}")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if list(cases) != ordered_ids or any(not isinstance(value, list) or len(value) != 1 for value in cases.values()):
        raise ValueError("client-fresh merged cases do not match the pinned source-order denominator")

    manifest = final.get("shard_receipts")
    if not isinstance(manifest, list) or len(manifest) != num_shards:
        raise ValueError("client-fresh shard manifest is incomplete")
    expected_manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if final.get("shard_receipts_sha256") != expected_manifest_hash:
        raise ValueError("client-fresh shard manifest hash changed")
    observed_indexes: list[int] = []
    total_calls = 0
    shard_evidence = []
    for shard_index, manifest_row in enumerate(manifest):
        expected_manifest_name = f"{shard_index:02d}-of-{num_shards:02d}.json"
        if manifest_row.get("name") != expected_manifest_name:
            raise ValueError("client-fresh shard manifest is not in canonical source-shard order")
        shard_root = artifact_root / "shards" / f"{shard_index:02d}-of-{num_shards:02d}"
        shard_receipt_path = shard_root / "shard-receipt.json"
        shard_behaviors_path = shard_root / "behaviors.csv"
        shard_cases_path = shard_root / "generated" / "test_cases.json"
        target_receipt_path = shard_root / "generated" / "client-target-receipt.json"
        if manifest_row.get("sha256") != sha256(shard_receipt_path):
            raise ValueError(f"client-fresh shard {shard_index} receipt hash changed")
        shard = json.loads(shard_receipt_path.read_text(encoding="utf-8"))
        indexes = list(range(shard_index, len(ordered_ids), num_shards))
        required_shard = {
            "status": "completed",
            "method": method,
            "public_method": PUBLIC_METHODS[method],
            "run_id": final["run_id"],
            "shard_index": shard_index,
            "num_shards": num_shards,
            "selected_source_indexes": indexes,
            "selected_behaviors": len(indexes),
            "behaviors_sha256": sha256(shard_behaviors_path),
            "test_cases_sha256": sha256(shard_cases_path),
            "client_target_receipt_sha256": sha256(target_receipt_path),
            "client_model": TARGET_MODEL,
            "client_revision": TARGET_REVISION,
            "upstream_revision": UPSTREAM_REVISION,
        }
        failed = [key for key, value in required_shard.items() if shard.get(key) != value]
        if failed:
            raise ValueError(f"client-fresh shard {shard_index} failed {','.join(sorted(failed))}")
        shard_cases = json.loads(shard_cases_path.read_text(encoding="utf-8"))
        expected_shard_ids = {ordered_ids[index] for index in indexes}
        if len(shard_cases) != len(indexes) or set(shard_cases) != expected_shard_ids:
            raise ValueError(f"client-fresh shard {shard_index} cases changed or left its source partition")
        target_receipt = json.loads(target_receipt_path.read_text(encoding="utf-8"))
        behavior_ids = [ordered_ids[index] for index in indexes]
        behavior_hashes = target_receipt.get("behavior_receipt_sha256")
        required_target = {
            "status": "completed",
            "client_target_verified": True,
            "client_model": TARGET_MODEL,
            "client_revision": TARGET_REVISION,
            "source_target_model": TARGET_MODEL,
            "behaviors_sha256": shard["behaviors_sha256"],
            "test_cases_sha256": sha256(shard_cases_path),
            "behavior_receipts": len(indexes),
            "target_calls": int(shard["target_calls"]),
        }
        failed_target = [key for key, value in required_target.items() if target_receipt.get(key) != value]
        if failed_target or not isinstance(behavior_hashes, dict) or set(behavior_hashes) != set(behavior_ids):
            raise ValueError(f"client-fresh shard {shard_index} aggregate target receipt changed")
        observed_calls = 0
        for behavior_id in behavior_ids:
            call_path = shard_root / "generated" / "client-call-receipts" / f"{behavior_id}.json"
            if behavior_hashes[behavior_id] != sha256(call_path):
                raise ValueError(f"client-fresh shard {shard_index} behavior-call receipt hash changed")
            call_receipt = json.loads(call_path.read_text(encoding="utf-8"))
            calls = call_receipt.get("calls")
            if (
                call_receipt.get("status") != "completed"
                or call_receipt.get("behavior_id") != behavior_id
                or call_receipt.get("client_model") != TARGET_MODEL
                or call_receipt.get("client_revision") != TARGET_REVISION
                or not isinstance(calls, list)
                or not calls
            ):
                raise ValueError(f"client-fresh shard {shard_index} behavior-call receipt changed")
            for call in calls:
                if (
                    set(call)
                    != {
                        "request_sha256",
                        "response_sha256",
                        "response_id_sha256",
                        "finish_reason",
                        "model",
                    }
                    or call.get("model") != TARGET_MODEL
                ):
                    raise ValueError(f"client-fresh shard {shard_index} call provenance changed")
                if any(
                    not isinstance(call.get(key), str) or len(call[key]) != 64
                    for key in ("request_sha256", "response_sha256", "response_id_sha256")
                ):
                    raise ValueError(f"client-fresh shard {shard_index} call hashes are invalid")
            observed_calls += len(calls)
        if observed_calls != int(shard["target_calls"]):
            raise ValueError(f"client-fresh shard {shard_index} behavior-call counts changed")
        observed_indexes.extend(indexes)
        total_calls += int(shard["target_calls"])
        shard_evidence.append(
            {
                "shard_index": shard_index,
                "selected_behaviors": len(indexes),
                "target_calls": int(shard["target_calls"]),
                "shard_receipt_sha256": sha256(shard_receipt_path),
                "client_target_receipt_sha256": sha256(target_receipt_path),
            }
        )
    if sorted(observed_indexes) != list(range(400)) or final.get("target_calls") != total_calls:
        raise ValueError("client-fresh shards do not reconcile the full source denominator or target-call count")
    control = {
        "method": PUBLIC_METHODS[method],
        "upstream_method": method,
        "upstream_revision": UPSTREAM_REVISION,
        "generated_cases_sha256": sha256(cases_path),
        "generation_receipt_sha256": sha256(generation_receipt),
        "behaviors_sha256": sha256(behaviors),
        "client_model": TARGET_MODEL,
        "client_revision": TARGET_REVISION,
        "source_target_model": TARGET_MODEL,
        "source_target_revision": TARGET_REVISION,
        "cases": 400,
        "behaviors": 400,
        "matching_behaviors": 400,
        "mismatches": [],
        "target_calls": total_calls,
        "num_shards": num_shards,
        "shards": shard_evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(PUBLIC_METHODS))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = validate(
        method=args.method,
        artifact_root=args.artifact_root,
        behaviors=args.behaviors,
        generation_receipt=args.generation_receipt,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "method": control["method"],
                "behaviors": control["behaviors"],
                "cases": control["cases"],
                "target_calls": control["target_calls"],
                "num_shards": control["num_shards"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
