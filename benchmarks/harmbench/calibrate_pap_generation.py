# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay PAP-top5 cases through the SHA-pinned upstream method body."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmarks.harmbench.pap import CASES_PER_BEHAVIOR, SAMPLING, TEMPLATE_SHA256, load_templates
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.zero_shot import ATTACKER_MODEL, ATTACKER_REVISION, mixtral_chat_template


SOURCE_SHA256 = "2dc2a8c6ec3df4a2a2c7d5295556a19eb796ac75e86d436d8358646531da68a4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_control(source_path: Path, templates: tuple[list[dict[str, str]], str, str]):
    if _sha256(source_path) != SOURCE_SHA256:
        raise ValueError("PAP method source differs from the pinned upstream checkout")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    upstream_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PAP")
    method = next(
        node
        for node in upstream_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_test_cases_single_behavior"
    )
    remove_quotes = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "remove_quotes"
    )
    isolated = ast.ClassDef(name="PAPControl", bases=[], keywords=[], body=[method], decorator_list=[])
    namespace: dict[str, Any] = {
        "deepcopy": deepcopy,
        "one_shot_kd": templates[1],
        "one_shot_kd_contexual": templates[2],
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[remove_quotes, isolated], type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )  # noqa: S102 — executes only two SHA-checked upstream AST nodes
    control = namespace["PAPControl"]()
    control.attack_model = object()
    control.attack_gen_config = {
        "max_new_tokens": SAMPLING["max_tokens"],
        "temperature": SAMPLING["temperature"],
        "stop_tokens": SAMPLING["stop"],
    }
    control.persuasion_templates = [
        {"technique": row["ss_technique"], "definition": row["ss_definition"], "example": row["ss_example"]}
        for row in templates[0][:CASES_PER_BEHAVIOR]
    ]
    return control, namespace


def calibrate(
    *,
    upstream: Path,
    behaviors_path: Path,
    generated_cases_path: Path,
    raw_generations_path: Path,
    receipt_path: Path,
    chat_template: str,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "method": "PAP-top5",
        "upstream_method": "PAP-top5",
        "upstream_revision": UPSTREAM_REVISION,
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "test_cases_sha256": _sha256(generated_cases_path),
        "raw_attacker_generations_sha256": _sha256(raw_generations_path),
        "behaviors_sha256": _sha256(behaviors_path),
        "templates_sha256": TEMPLATE_SHA256,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "sampling": SAMPLING,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"PAP generation receipt {key} differs from the pinned source")
    with behaviors_path.open(newline="", encoding="utf-8") as source:
        behaviors = list(csv.DictReader(source))
    saved_cases = json.loads(generated_cases_path.read_text(encoding="utf-8"))
    raw_generations = json.loads(raw_generations_path.read_text(encoding="utf-8"))
    if list(saved_cases) != [row["BehaviorID"] for row in behaviors]:
        raise ValueError("PAP cases do not follow the pinned behavior order")
    if not isinstance(raw_generations, list) or len(raw_generations) != len(behaviors) * CASES_PER_BEHAVIOR:
        raise ValueError("PAP raw attacker generation count disagrees with the pinned protocol")
    templates = load_templates(upstream / "baselines/pap/templates.py")
    control, namespace = _upstream_control(upstream / "baselines/pap/PAP.py", templates)
    captured_inputs: list[str] = []
    mismatches: list[str] = []
    for behavior_index, row in enumerate(behaviors):
        attacks = saved_cases[row["BehaviorID"]]
        if len(attacks) != CASES_PER_BEHAVIOR:
            raise ValueError("PAP-top5 generated cases must have five attacks per behavior")
        generations = raw_generations[behavior_index * CASES_PER_BEHAVIOR : (behavior_index + 1) * CASES_PER_BEHAVIOR]
        if any(not isinstance(value, str) for value in generations):
            raise ValueError("PAP raw attacker generations must be strings")

        def fake_generations(model: object, inputs: list[str], **kwargs: Any) -> list[str]:
            del model
            if kwargs != control.attack_gen_config:
                raise ValueError("upstream PAP sampling config differs from the pinned receipt")
            captured_inputs.extend(inputs)
            return generations

        namespace["get_generations"] = fake_generations
        upstream_cases, _ = control.generate_test_cases_single_behavior(row)
        if upstream_cases != attacks:
            mismatches.append(row["BehaviorID"])
    attacker_prompts = [chat_template.format(instruction=value) for value in captured_inputs]
    return {
        "method": "PAP-top5",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_method_sha256": SOURCE_SHA256,
        "generated_cases_sha256": _sha256(generated_cases_path),
        "raw_attacker_generations_sha256": _sha256(raw_generations_path),
        "behaviors_sha256": _sha256(behaviors_path),
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "templates_sha256": TEMPLATE_SHA256,
        "attacker_prompt_count": len(attacker_prompts),
        "attacker_prompts_sha256": hashlib.sha256(
            json.dumps(attacker_prompts, ensure_ascii=True).encode()
        ).hexdigest(),
        "cases": len(attacker_prompts),
        "behaviors": len(behaviors),
        "matching_behaviors": len(behaviors) - len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--generated-cases", required=True, type=Path)
    parser.add_argument("--raw-generations", required=True, type=Path)
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
        raw_generations_path=args.raw_generations,
        receipt_path=args.generation_receipt,
        chat_template=chat_template,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"PAP upstream mapping: {result['matching_behaviors']}/{result['behaviors']} behaviors, {result['cases']} cases"
    )
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
