# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import hashlib
import json
from copy import deepcopy

import pytest

from benchmarks.harmbench.calibrate_pap_generation import calibrate
from benchmarks.harmbench.pap import (
    CASES_PER_BEHAVIOR,
    SAMPLING,
    TEMPLATE_SHA256,
    cases,
    load_templates,
    queries,
    remove_quotes,
)
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.zero_shot import ATTACKER_MODEL, ATTACKER_REVISION
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_pap_top5_queries_and_cases_match_pinned_upstream_method_body():
    source = UPSTREAM / "baselines/pap/PAP.py"
    templates = load_templates(UPSTREAM / "baselines/pap/templates.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    upstream_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PAP")
    method = next(
        node
        for node in upstream_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_test_cases_single_behavior"
    )
    isolated = ast.ClassDef(name="PAPControl", bases=[], keywords=[], body=[method], decorator_list=[])
    namespace = {
        "deepcopy": deepcopy,
        "one_shot_kd": templates[1],
        "one_shot_kd_contexual": templates[2],
        "remove_quotes": remove_quotes,
    }
    captured = []
    generated = ['"safe synthetic A"', "'safe synthetic B'", '"""safe synthetic C"""', "safe D", "safe E"]

    def fake_generations(model, inputs, **kwargs):
        captured.extend(inputs)
        assert kwargs == {
            "max_new_tokens": SAMPLING["max_tokens"],
            "temperature": SAMPLING["temperature"],
            "stop_tokens": SAMPLING["stop"],
        }
        return generated

    namespace["get_generations"] = fake_generations
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(source), "exec"),
        namespace,
    )  # noqa: S102
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
    chat_template = "[INST] {instruction} [/INST]"
    for row in [
        {"BehaviorID": "standard", "Behavior": "benign behavior", "ContextString": ""},
        {"BehaviorID": "contextual", "Behavior": "benign behavior", "ContextString": "context"},
    ]:
        captured.clear()
        upstream_cases, _ = control.generate_test_cases_single_behavior(row)
        assert [chat_template.format(instruction=item) for item in captured] == queries(
            [row], chat_template, templates
        )
        assert upstream_cases == cases([row], generated)[row["BehaviorID"]]


def test_pap_rejects_missing_generations():
    with pytest.raises(ValueError, match="generation count"):
        cases([{"BehaviorID": "x", "ContextString": ""}], [])


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_raw_generation_receipt_preserves_noninvertible_quote_and_whitespace_cases(tmp_path):
    row = {"BehaviorID": "synthetic", "Behavior": "benign behavior", "ContextString": "context"}
    behavior_path = tmp_path / "behaviors.csv"
    with behavior_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    raw = ['"safe A"', '"  spaced B  "', '"""safe C"""', "  safe D  ", "safe E"]
    raw_path = tmp_path / "raw_attacker_generations.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=True), encoding="utf-8")
    cases_path = tmp_path / "test_cases.json"
    cases_path.write_text(json.dumps(cases([row], raw), ensure_ascii=True), encoding="utf-8")
    chat_template = "[INST] {instruction} [/INST]"
    receipt = {
        "method": "PAP-top5",
        "upstream_method": "PAP-top5",
        "upstream_revision": UPSTREAM_REVISION,
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "test_cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "raw_attacker_generations_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "behaviors_sha256": hashlib.sha256(behavior_path.read_bytes()).hexdigest(),
        "templates_sha256": TEMPLATE_SHA256,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "sampling": SAMPLING,
    }
    receipt_path = tmp_path / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = calibrate(
        upstream=UPSTREAM,
        behaviors_path=behavior_path,
        generated_cases_path=cases_path,
        raw_generations_path=raw_path,
        receipt_path=receipt_path,
        chat_template=chat_template,
    )
    assert result["matching_behaviors"] == 1
    assert result["cases"] == CASES_PER_BEHAVIOR
    assert result["mismatches"] == []
