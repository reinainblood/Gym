# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned HarmBench PAP-top5 prompt and output transformations.

The original template file is supplied from the verified upstream checkout.
Its three literal assignments are parsed without importing PAP's Ray/Torch
runtime or copying a second, drift-prone version into Gym.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


TEMPLATE_SHA256 = "4a81b7c635e44eda360f4653edf466be80fc7b682d9fc8271f4b2890a9f5ec1b"
EXPERIMENT = "top_5"
CASES_PER_BEHAVIOR = 5
SAMPLING = {"temperature": 1.0, "max_tokens": 2048, "stop": ["|"]}


def load_templates(path: Path) -> tuple[list[dict[str, str]], str, str]:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("PAP template file differs from pinned HarmBench source")
    values: dict[str, Any] = {}
    selected = {"persuasion_taxonomy", "one_shot_kd", "one_shot_kd_contexual"}
    for node in ast.parse(source, filename=str(path)).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in selected:
            values[target.id] = ast.literal_eval(node.value)
    if set(values) != selected:
        raise ValueError("pinned PAP templates are incomplete")
    taxonomy = values["persuasion_taxonomy"]
    if not isinstance(taxonomy, list) or len(taxonomy) < CASES_PER_BEHAVIOR:
        raise ValueError("pinned PAP taxonomy has fewer than five techniques")
    return taxonomy, values["one_shot_kd"], values["one_shot_kd_contexual"]


def queries(
    behaviors: list[dict[str, str]],
    chat_template: str,
    templates: tuple[list[dict[str, str]], str, str],
) -> list[str]:
    taxonomy, standard, contextual = templates
    output: list[str] = []
    for row in behaviors:
        template = contextual if row["ContextString"] else standard
        for technique in taxonomy[:CASES_PER_BEHAVIOR]:
            input_text = template.format(
                behavior=row["Behavior"],
                context=row["ContextString"],
                technique=technique["ss_technique"],
                definition=technique["ss_definition"],
                example=technique["ss_example"],
            )
            output.append(chat_template.format(instruction=input_text))
    return output


def remove_quotes(value: str) -> str:
    if value.startswith('"""') and value.endswith('"""'):
        return value[3:-3]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def cases(behaviors: list[dict[str, str]], generations: list[str]) -> dict[str, list[str]]:
    if len(generations) != len(behaviors) * CASES_PER_BEHAVIOR:
        raise ValueError("Mixtral PAP generation count differs from the pinned top-5 protocol")
    output: dict[str, list[str]] = {}
    for index, generation in enumerate(generations):
        row = behaviors[index // CASES_PER_BEHAVIOR]
        persuasion = remove_quotes(generation.strip())
        if row["ContextString"]:
            persuasion = f"{row['ContextString']}\n\n---\n\n{persuasion}"
        output.setdefault(row["BehaviorID"], []).append(persuasion)
    if len(output) != len(behaviors):
        raise ValueError("duplicate or missing behavior IDs in PAP-top5 generation")
    return output
