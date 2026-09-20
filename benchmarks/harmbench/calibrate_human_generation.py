# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native HumanJailbreaks cases with the pinned upstream method body.

The upstream module's top-level imports require Torch/vLLM even though this
method itself does not. We execute only its inspected, pinned ``__init__`` and
``generate_test_cases`` AST nodes with the literal JAILBREAKS list, then compare
all generated cases. No attack text is printed or written by this control.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

from benchmarks.harmbench.generate_human_jailbreaks import _templates
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_human_jailbreaks import BEHAVIORS_SHA256, TEMPLATE_SHA256


def calibrate(*, upstream: Path, generated: Path, output: Path) -> dict:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    source = upstream / "baselines/human_jailbreaks/human_jailbreaks.py"
    templates_path = upstream / "baselines/human_jailbreaks/jailbreaks.py"
    behaviors_path = upstream / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    if hashlib.sha256(templates_path.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("upstream human jailbreak templates changed")
    if hashlib.sha256(behaviors_path.read_bytes()).hexdigest() != BEHAVIORS_SHA256:
        raise ValueError("upstream text test behaviors changed")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    upstream_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HumanJailbreaks"
    )
    methods = [
        node
        for node in upstream_class.body
        if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "generate_test_cases"}
    ]
    if len(methods) != 2:
        raise ValueError("upstream HumanJailbreaks method structure changed")
    isolated = ast.ClassDef(name="HumanJailbreaksControl", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
    namespace: dict = {"random": random, "JAILBREAKS": _templates(templates_path)}
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102
    with behaviors_path.open(newline="", encoding="utf-8") as handle:
        behaviors = list(csv.DictReader(handle))
    control, _ = namespace["HumanJailbreaksControl"](random_subset=5, seed=1).generate_test_cases(behaviors)
    # Upstream's save_test_cases uses json.dump (ensure_ascii=True), then its
    # completion stage reloads test_cases.json. A template contains a Unicode
    # surrogate pair that json.load normalizes to one scalar value; compare the
    # persisted cases actually seen by the target, not transient Python strings.
    control = json.loads(json.dumps(control))
    native = json.loads(generated.read_text(encoding="utf-8"))
    mismatches = sorted(behavior_id for behavior_id in control if control[behavior_id] != native.get(behavior_id))
    mismatches.extend(sorted(set(native) - set(control)))
    receipt = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_method_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "templates_sha256": TEMPLATE_SHA256,
        "behaviors_sha256": BEHAVIORS_SHA256,
        "generated_cases_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "behaviors": len(control),
        "cases": sum(map(len, control.values())),
        "matching_behaviors": len(control) - len(mismatches),
        "mismatches": mismatches,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compared {receipt['cases']} cases across {receipt['behaviors']} behaviors; {len(mismatches)} mismatches")
    if mismatches:
        raise ValueError("native HumanJailbreaks cases differ from the pinned upstream generator")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibrate(upstream=args.upstream, generated=args.generated, output=args.output)
