# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import the official HarmBench 1.0 TAP-Transfer cases for the pinned corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


METHOD = "TAP-Transfer"
UPSTREAM_METHOD = "TAP-Transfer"
UPSTREAM_CLASS = "TAP"
EXPERIMENT = "gpt-4-0613_judge_gpt-4-1106-preview_target"
SOURCE_SHA256 = "71438be2ca516be44b7ffc03230c01719b8fe3c24c9cae89e17deecd28c92de2"  # pragma: allowlist secret
SOURCE_BEHAVIORS = 403
SOURCE_EXTRA_BEHAVIORS = 3
SOURCE_EXTRA_IDS_SHA256 = (
    "20b66b06fd4d019222168ab891325000324082b0168c7c17e94999a8508dd25d"  # pragma: allowlist secret
)
ZENODO_DOI = "10.5281/zenodo.10714577"
ZENODO_ARCHIVE = "harmbench_results_initial_release.zip"
ZENODO_ARCHIVE_MD5 = "ac0e8210389b81951f66da6daaa6db3b"  # pragma: allowlist secret
ZENODO_MEMBER = f"harmbench_results_initial_release/results_text/TAP/{EXPERIMENT}/test_cases/test_cases.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_cases(cases: dict[str, Any], ordered_behavior_ids: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    missing = set(ordered_behavior_ids) - set(cases)
    if missing:
        raise ValueError("official TAP-Transfer artifact is missing pinned public behaviors")
    extra = sorted(set(cases) - set(ordered_behavior_ids))
    selected: dict[str, list[str]] = {}
    for behavior_id in ordered_behavior_ids:
        value = cases[behavior_id]
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str) or not value[0]:
            raise ValueError("TAP-Transfer must contain exactly one nonempty case per pinned behavior")
        selected[behavior_id] = value
    return selected, extra


def import_cases(*, source: Path, behaviors: Path, run_id: str, output_dir: Path) -> dict[str, Any]:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("TAP-Transfer source differs from the official HarmBench 1.0 artifact")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != SOURCE_BEHAVIORS:
        raise ValueError("official TAP-Transfer artifact has an unexpected denominator")
    with behaviors.open(newline="", encoding="utf-8") as stream:
        behavior_rows = list(csv.DictReader(stream))
    ordered_ids = [row["BehaviorID"] for row in behavior_rows]
    if len(ordered_ids) != 400 or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("pinned HarmBench corpus must contain 400 unique behaviors")
    selected, extra = filter_cases(raw, ordered_ids)
    if (
        len(extra) != SOURCE_EXTRA_BEHAVIORS
        or hashlib.sha256(json.dumps(extra).encode()).hexdigest() != SOURCE_EXTRA_IDS_SHA256
    ):
        raise ValueError("official TAP-Transfer retired-behavior set changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "test_cases.json"
    cases_path.write_text(json.dumps(selected, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": METHOD,
        "upstream_method": UPSTREAM_METHOD,
        "upstream_class": UPSTREAM_CLASS,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": EXPERIMENT,
        "source_experiment": EXPERIMENT,
        "run_id": run_id,
        "behaviors_sha256": sha256(behaviors),
        "test_cases_sha256": sha256(cases_path),
        "behaviors": len(ordered_ids),
        "cases": len(selected),
        "source_artifact": {
            "doi": ZENODO_DOI,
            "archive": ZENODO_ARCHIVE,
            "archive_md5": ZENODO_ARCHIVE_MD5,
            "member": ZENODO_MEMBER,
            "member_sha256": SOURCE_SHA256,
            "source_behaviors": len(raw),
            "filtered_retired_behaviors": len(extra),
            "filtered_retired_behavior_ids_sha256": SOURCE_EXTRA_IDS_SHA256,
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
                "source_member_sha256": SOURCE_SHA256,
                "source_experiment": EXPERIMENT,
                "cases": len(selected),
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
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = import_cases(
        source=args.source,
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
