# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the full 320-behavior DirectRequest set from pinned upstream code.

The existing `harmbench` benchmark stays at its original 240 non-copyright
rows. This opt-in generated-method path includes all 80 copyright behaviors
and relies on the separately validated MinHash scorer.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


SOURCE_SHA256 = "277b6edea5b6aa77b962183dba8a4cace150d85448983e3496222f864d70995d"
EXPECTED_COUNTS = {"standard": 159, "contextual": 81, "copyright": 80}
EXPERIMENT = "default"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_cases(source: Path, behaviors: list[dict[str, str]]) -> dict[str, list[str]]:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("DirectRequest method source differs from pinned upstream")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    upstream_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DirectRequest"
    )
    method = next(
        node
        for node in upstream_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_test_cases"
    )
    isolated = ast.ClassDef(name="DirectRequestControl", bases=[], keywords=[], body=[method], decorator_list=[])
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(source), "exec"),
        namespace,
    )  # noqa: S102 — executes the SHA-checked pinned method body only
    cases, _ = namespace["DirectRequestControl"]().generate_test_cases(behaviors)
    return cases


def generate(*, upstream: Path, run_id: str, output_dir: Path) -> dict[str, Any]:
    if not run_id or output_dir.exists():
        raise ValueError("a new run ID and absent output directory are required")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError("HarmBench checkout is not at the pinned revision")
    source_relative = "baselines/direct_request/direct_request.py"
    behavior_relative = "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    subprocess.run(
        ["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--", source_relative, behavior_relative],
        check=True,
    )
    source_path = upstream / source_relative
    behavior_path = upstream / behavior_relative
    with behavior_path.open(newline="", encoding="utf-8") as source:
        behaviors = list(csv.DictReader(source))
    counts = Counter(row["FunctionalCategory"] for row in behaviors)
    if counts != EXPECTED_COUNTS or len({row["BehaviorID"] for row in behaviors}) != len(behaviors):
        raise ValueError("DirectRequest behavior cohort or IDs differ from the pinned 320-row protocol")
    cases = upstream_cases(source_path, behaviors)
    mismatches = [
        row["BehaviorID"]
        for row in behaviors
        if cases.get(row["BehaviorID"])
        != [f"{row['ContextString']}\n\n---\n\n{row['Behavior']}" if row["ContextString"] else row["Behavior"]]
    ]
    if mismatches or len(cases) != len(behaviors):
        raise ValueError("pinned DirectRequest method disagrees with the independent direct-request formula")
    output_dir.mkdir(parents=True, exist_ok=False)
    cases_path = output_dir / "test_cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "method": "DirectRequest",
        "upstream_method": "DirectRequest",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_method_sha256": SOURCE_SHA256,
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "behaviors_sha256": sha256(behavior_path),
        "test_cases_sha256": sha256(cases_path),
        "behaviors": len(behaviors),
        "cases": len(behaviors),
        "functional_counts": dict(sorted(counts.items())),
    }
    (output_dir / "generation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    control = {
        "method": "DirectRequest",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_method_sha256": SOURCE_SHA256,
        "generated_cases_sha256": receipt["test_cases_sha256"],
        "behaviors_sha256": receipt["behaviors_sha256"],
        "cases": len(behaviors),
        "behaviors": len(behaviors),
        "matching_behaviors": len(behaviors),
        "mismatches": mismatches,
    }
    (output_dir / "generation-upstream-control.json").write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = generate(upstream=args.upstream, run_id=args.run_id, output_dir=args.output_dir)
    print(f"Generated {receipt['cases']} pinned DirectRequest cases across {receipt['behaviors']} behaviors")


if __name__ == "__main__":
    main()
