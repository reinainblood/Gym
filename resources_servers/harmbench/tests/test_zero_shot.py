# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import hashlib
import json
import types

import pytest

from benchmarks.harmbench.calibrate_zero_shot_generation import calibrate
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.zero_shot import (
    ATTACKER_MODEL,
    ATTACKER_REVISION,
    CASES_PER_BEHAVIOR,
    CONTEXT_PREFIX,
    CONTEXT_QUERY,
    STANDARD_PREFIX,
    STANDARD_QUERY,
    cases,
    mixtral_chat_template,
    queries,
)
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_query_and_case_mapping_match_pinned_upstream_method_body():
    source = UPSTREAM / "baselines/zeroshot/zeroshot.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
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
    assert STANDARD_QUERY == namespace["PROMPT_TEMPLATE"]["prompt"]
    assert STANDARD_PREFIX == namespace["PROMPT_TEMPLATE"]["completion_prefix"]
    assert CONTEXT_QUERY == namespace["CONTEXTUAL_BEHAVIOR_PROMPT_TEMPLATE"]["prompt"]
    assert CONTEXT_PREFIX == namespace["CONTEXTUAL_BEHAVIOR_PROMPT_TEMPLATE"]["completion_prefix"]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(source), "exec"),
        namespace,
    )  # noqa: S102
    behaviors = [
        {"BehaviorID": "standard", "Behavior": "benign behavior", "ContextString": "", "Tags": ""},
        {
            "BehaviorID": "contextual",
            "Behavior": "benign contextual behavior",
            "ContextString": "context",
            "Tags": "context",
        },
    ]
    generated = [f"safe synthetic attack {index}" for index in range(len(behaviors) * CASES_PER_BEHAVIOR)]
    captured = []

    class FakeModel:
        def generate(self, prompts, sampling_params):
            captured.extend(prompts)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in generated]

    control = namespace["ZeroShotControl"]()
    control.num_test_cases_per_behavior = CASES_PER_BEHAVIOR
    control.chat_template = "[INST] {instruction} [/INST]"
    control.model = FakeModel()
    control.sampling_params = object()
    upstream_cases, _ = control.generate_test_cases(behaviors)
    assert captured == queries(behaviors, control.chat_template)
    assert upstream_cases == cases(behaviors, generated)


def test_zero_shot_rejects_missing_attack_generations():
    with pytest.raises(ValueError, match="generation count"):
        cases([{"BehaviorID": "x", "ContextString": "", "Tags": ""}], [])


def test_mixtral_template_mirrors_upstream_bos_removal():
    class Tokenizer:
        bos_token = "<s>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "<s>[INST] " + messages[0]["content"] + " [/INST]"

    assert mixtral_chat_template(Tokenizer()) == "[INST] {instruction} [/INST]"


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_zero_shot_generation_receipt_replays_upstream_contextual_cases(tmp_path):
    row = {
        "BehaviorID": "synthetic",
        "Behavior": "benign behavior",
        "ContextString": "context",
        "Tags": "context",
    }
    behavior_path = tmp_path / "behaviors.csv"
    with behavior_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    generated_cases_path = tmp_path / "test_cases.json"
    generated_cases_path.write_text(
        json.dumps(cases([row], [f"safe synthetic {index}" for index in range(CASES_PER_BEHAVIOR)])),
        encoding="utf-8",
    )
    chat_template = "[INST] {instruction} [/INST]"
    receipt = {
        "method": "ZeroShot",
        "upstream_method": "ZeroShot",
        "upstream_revision": UPSTREAM_REVISION,
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "test_cases_sha256": hashlib.sha256(generated_cases_path.read_bytes()).hexdigest(),
        "behaviors_sha256": hashlib.sha256(behavior_path.read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    receipt_path = tmp_path / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = calibrate(
        upstream=UPSTREAM,
        behaviors_path=behavior_path,
        generated_cases_path=generated_cases_path,
        receipt_path=receipt_path,
        chat_template=chat_template,
    )
    assert result["matching_behaviors"] == 1
    assert result["cases"] == CASES_PER_BEHAVIOR
    assert result["mismatches"] == []
