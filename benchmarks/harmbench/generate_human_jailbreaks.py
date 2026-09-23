# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce pinned HarmBench HumanJailbreaks/random_subset_5 without GPU imports.

The attack templates stay in the separately pinned HarmBench checkout.  We
parse only its literal JAILBREAKS list and reproduce the upstream RNG/shuffle
loop; importing all of HarmBench would eagerly load Torch, vLLM, and Ray.
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

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


def _templates(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "JAILBREAKS" for target in node.targets
        ):
            templates = ast.literal_eval(node.value)
            if (
                not isinstance(templates, list)
                or not templates
                or not all(isinstance(item, str) for item in templates)
            ):
                break
            return templates
    raise ValueError("pinned HumanJailbreaks source does not contain a literal JAILBREAKS list")


def generate(
    behaviors: list[dict[str, str]], templates: list[str], *, seed: int = 1, subset: int = 5
) -> dict[str, list[str]]:
    if subset < 1 or subset > len(templates):
        raise ValueError("subset must be between 1 and the number of upstream jailbreak templates")
    rng = random.Random(seed)
    shuffled = templates.copy()
    test_cases: dict[str, list[str]] = {}
    for row in behaviors:
        behavior_id = row["BehaviorID"]
        if behavior_id in test_cases:
            raise ValueError(f"duplicate behavior ID {behavior_id}")
        rng.shuffle(shuffled)
        request = row["Behavior"]
        if row["ContextString"]:
            request = f"{row['ContextString']}\n\n---\n\n{request}"
        test_cases[behavior_id] = [f"{template}\n\n{request}" for template in shuffled[:subset]]
    return test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="HarmBench checkout at the pinned commit")
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--subset", type=int, default=5)
    args = parser.parse_args()
    upstream_head = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if upstream_head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {upstream_head}, expected {UPSTREAM_REVISION}")
    source = args.upstream / "baselines/human_jailbreaks/jailbreaks.py"
    with args.behaviors.open(newline="", encoding="utf-8") as handle:
        behaviors = list(csv.DictReader(handle))
    cases = generate(behaviors, _templates(source), seed=args.seed, subset=args.subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Upstream templates contain an unpaired Unicode surrogate. JSON escapes it
    # losslessly; writing it as raw UTF-8 would fail after truncating the file.
    args.output.write_text(json.dumps(cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "method": "HumanJailbreaks",
        "upstream_method": "HumanJailbreaks",
        "experiment": f"random_subset_{args.subset}",
        "run_id": f"seed-{args.seed}",
        "upstream_revision": UPSTREAM_REVISION,
        "template_source_sha256": source_hash,
        "test_cases_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "behaviors_sha256": hashlib.sha256(args.behaviors.read_bytes()).hexdigest(),
        "seed": args.seed,
        "subset": args.subset,
        "behaviors": len(cases),
        "cases": sum(map(len, cases.values())),
    }
    args.output.with_suffix(".generation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {manifest['cases']} HumanJailbreaks cases for {manifest['behaviors']} behaviors")


if __name__ == "__main__":
    main()
