# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay generated ZeroShot cases through the pinned upstream method body.

The saved attacker generations are reconstructed from the final cases (after
removing the contextual prefix where applicable). No attack payload is written
to the control receipt or printed to stdout.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import types
from pathlib import Path
from typing import Any

from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.zero_shot import ATTACKER_MODEL, ATTACKER_REVISION, CASES_PER_BEHAVIOR, mixtral_chat_template


SOURCE_SHA256 = "09aa2d210226769c4ca0837fd8600dd4c47ab764e9a4880f89595fa3dd9d0952"
CONTEXT_PREFIX = "\n\n-----\n\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_control(source_path: Path, chat_template: str, generations: list[str]):
    if _sha256(source_path) != SOURCE_SHA256:
        raise ValueError("ZeroShot method source differs from the pinned upstream checkout")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    upstream_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZeroShot")
    method = next(
        node
        for node in upstream_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_test_cases"
    )
    isolated = ast.ClassDef(name="ZeroShotControl", bases=[], keywords=[], body=[method], decorator_list=[])
    names = {"PROMPT_TEMPLATE", "CONTEXTUAL_BEHAVIOR_PROMPT_TEMPLATE"}
    namespace = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    captured_prompts: list[str] = []

    class FakeModel:
        def generate(self, prompts: list[str], sampling_params: object):
            del sampling_params
            captured_prompts.extend(prompts)
            if len(prompts) != len(generations):
                raise ValueError("upstream ZeroShot requested a different number of attacker generations")
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in generations]

    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(source_path), "exec"),
        namespace,
    )  # noqa: S102 — executes only the pinned SHA-checked upstream method node
    control = namespace["ZeroShotControl"]()
    control.num_test_cases_per_behavior = CASES_PER_BEHAVIOR
    control.chat_template = chat_template
    control.model = FakeModel()
    control.sampling_params = object()
    return control, captured_prompts


def calibrate(
    *, upstream: Path, behaviors_path: Path, generated_cases_path: Path, receipt_path: Path, chat_template: str
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "method": "ZeroShot",
        "upstream_method": "ZeroShot",
        "upstream_revision": UPSTREAM_REVISION,
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "test_cases_sha256": _sha256(generated_cases_path),
        "behaviors_sha256": _sha256(behaviors_path),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"ZeroShot generation receipt {key} differs from the pinned source")
    with behaviors_path.open(newline="", encoding="utf-8") as source:
        behaviors = list(csv.DictReader(source))
    saved_cases = json.loads(generated_cases_path.read_text(encoding="utf-8"))
    if list(saved_cases) != [row["BehaviorID"] for row in behaviors]:
        raise ValueError("ZeroShot cases do not follow the pinned behavior order")
    generations: list[str] = []
    for row in behaviors:
        attacks = saved_cases[row["BehaviorID"]]
        if len(attacks) != CASES_PER_BEHAVIOR:
            raise ValueError("ZeroShot generated cases must have five attacks per behavior")
        prefix = row["ContextString"] + CONTEXT_PREFIX if "context" in row["Tags"].split(", ") else ""
        for attack in attacks:
            if not isinstance(attack, str) or not attack.startswith(prefix):
                raise ValueError("ZeroShot saved case has the wrong contextual prefix")
            generations.append(attack[len(prefix) :])
    control, captured_prompts = _upstream_control(
        upstream / "baselines/zeroshot/zeroshot.py", chat_template, generations
    )
    upstream_cases, _ = control.generate_test_cases(behaviors)
    mismatches = [
        row["BehaviorID"] for row in behaviors if upstream_cases[row["BehaviorID"]] != saved_cases[row["BehaviorID"]]
    ]
    return {
        "method": "ZeroShot",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_method_sha256": SOURCE_SHA256,
        "generated_cases_sha256": _sha256(generated_cases_path),
        "behaviors_sha256": _sha256(behaviors_path),
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "attacker_prompt_count": len(captured_prompts),
        "attacker_prompts_sha256": hashlib.sha256(
            json.dumps(captured_prompts, ensure_ascii=True).encode()
        ).hexdigest(),
        "cases": len(generations),
        "behaviors": len(behaviors),
        "matching_behaviors": len(behaviors) - len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--generated-cases", required=True, type=Path)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, revision=ATTACKER_REVISION)
    chat_template = mixtral_chat_template(tokenizer)
    result = calibrate(
        upstream=args.upstream,
        behaviors_path=args.behaviors,
        generated_cases_path=args.generated_cases,
        receipt_path=args.generation_receipt,
        chat_template=chat_template,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"ZeroShot upstream mapping: {result['matching_behaviors']}/{result['behaviors']} behaviors, {result['cases']} cases"
    )
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
