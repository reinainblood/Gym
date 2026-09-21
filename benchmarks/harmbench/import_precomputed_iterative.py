# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge official PAIR/TAP Mixtral cases with exact current-corpus repair cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


EXPERIMENT = "mixtral_8x7b"
SOURCE_EXTRA_IDS_SHA256 = (
    "20b66b06fd4d019222168ab891325000324082b0168c7c17e94999a8508dd25d"  # pragma: allowlist secret
)
ZENODO_DOI = "10.5281/zenodo.10714577"
ZENODO_ARCHIVE = "harmbench_results_initial_release.zip"
ZENODO_ARCHIVE_MD5 = "ac0e8210389b81951f66da6daaa6db3b"  # pragma: allowlist secret
SOURCES = {
    "PAIR": {
        "sha256": "9a85a36ae7aa47fef160eca99746379a1521f75acb2d85749ed083e0aac4af29",  # pragma: allowlist secret
        "member": "harmbench_results_initial_release/results_text/PAIR/mixtral_8x7b/test_cases/test_cases.json",
    },
    "TAP": {
        "sha256": "5b4dd35089a4b02458551b3cfbb6b747787a7bf14012775ef1d441d6aaf24867",  # pragma: allowlist secret
        "member": "harmbench_results_initial_release/results_text/TAP/mixtral_8x7b/test_cases/test_cases.json",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_cases(
    official: dict[str, Any], repair: dict[str, Any], ordered_behavior_ids: list[str]
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    source_ids = set(ordered_behavior_ids)
    extra = sorted(set(official) - source_ids)
    missing = [behavior_id for behavior_id in ordered_behavior_ids if behavior_id not in official]
    if set(repair) != set(missing):
        raise ValueError("repair cases do not cover the exact current-corpus gap")
    merged: dict[str, list[str]] = {}
    for behavior_id in ordered_behavior_ids:
        value = official.get(behavior_id, repair.get(behavior_id))
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str) or not value[0]:
            raise ValueError("PAIR/TAP must contain exactly one nonempty case per behavior")
        merged[behavior_id] = value
    return merged, missing, extra


def import_cases(
    *,
    method: str,
    source: Path,
    repair_cases: Path,
    repair_receipt: Path,
    behaviors: Path,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if method not in SOURCES:
        raise ValueError("method must be PAIR or TAP")
    spec = SOURCES[method]
    if sha256(source) != spec["sha256"]:
        raise ValueError("PAIR/TAP source differs from the official HarmBench 1.0 artifact")
    official = json.loads(source.read_text(encoding="utf-8"))
    repaired = json.loads(repair_cases.read_text(encoding="utf-8"))
    repair_meta = json.loads(repair_receipt.read_text(encoding="utf-8"))
    with behaviors.open(newline="", encoding="utf-8") as stream:
        behavior_rows = list(csv.DictReader(stream))
    ordered_ids = [row["BehaviorID"] for row in behavior_rows]
    merged, missing, extra = merge_cases(official, repaired, ordered_ids)
    if len(official) != 401 or len(ordered_ids) != 400 or len(missing) != 2 or len(extra) != 3:
        raise ValueError("PAIR/TAP official and current corpus denominators changed")
    if hashlib.sha256(json.dumps(extra).encode()).hexdigest() != SOURCE_EXTRA_IDS_SHA256:
        raise ValueError("PAIR/TAP retired-behavior set changed")
    required_repair = {
        "status": "completed",
        "method": method,
        "experiment": EXPERIMENT,
        "upstream_revision": UPSTREAM_REVISION,
        "source_precomputed_sha256": spec["sha256"],
        "repaired_cases_sha256": sha256(repair_cases),
        "repaired_behaviors": 2,
        "repaired_cases": 2,
    }
    if any(repair_meta.get(key) != value for key, value in required_repair.items()):
        raise ValueError("PAIR/TAP repair receipt does not bind the supplemental cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "test_cases.json"
    cases_path.write_text(json.dumps(merged, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": method,
        "upstream_method": method,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "behaviors_sha256": sha256(behaviors),
        "test_cases_sha256": sha256(cases_path),
        "behaviors": len(ordered_ids),
        "cases": len(merged),
        "official_source": {
            "doi": ZENODO_DOI,
            "archive": ZENODO_ARCHIVE,
            "archive_md5": ZENODO_ARCHIVE_MD5,
            "member": spec["member"],
            "member_sha256": spec["sha256"],
            "source_behaviors": len(official),
            "filtered_retired_behaviors": len(extra),
        },
        "repair": {
            "missing_behaviors": len(missing),
            "missing_behavior_ids_sha256": hashlib.sha256(json.dumps(sorted(missing)).encode()).hexdigest(),
            "repair_cases_sha256": sha256(repair_cases),
            "repair_receipt_sha256": sha256(repair_receipt),
        },
    }
    (output_dir / "generation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "generation-upstream-control.json").write_text(
        json.dumps(
            {
                "upstream_revision": UPSTREAM_REVISION,
                "generated_cases_sha256": receipt["test_cases_sha256"],
                "source_member_sha256": spec["sha256"],
                "repair_receipt_sha256": sha256(repair_receipt),
                "cases": len(merged),
                "behaviors": len(ordered_ids),
                "matching_behaviors": len(ordered_ids),
                "mismatches": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(SOURCES))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repair-cases", required=True, type=Path)
    parser.add_argument("--repair-receipt", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = import_cases(
        method=args.method,
        source=args.source,
        repair_cases=args.repair_cases,
        repair_receipt=args.repair_receipt,
        behaviors=args.behaviors,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "method": receipt["method"],
                "run_id": receipt["run_id"],
                "behaviors": receipt["behaviors"],
                "cases": receipt["cases"],
                "test_cases_sha256": receipt["test_cases_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
