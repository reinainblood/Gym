# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The customer-requested HarmBench method inventory and upstream mappings.

An entry in this catalog is a protocol target, not a claim that the method has
been executed or validated against a particular model.  The pinned upstream
pipeline remains the authority for its class, experiment, and target type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TargetType = Literal["text_api", "text_weights", "vision_api", "vision_weights"]


@dataclass(frozen=True)
class Method:
    name: str
    upstream_key: str
    target_types: tuple[TargetType, ...]
    generation: Literal["upstream", "client_fresh"] = "upstream"


METHODS: dict[str, Method] = {
    method.name: method
    for method in (
        Method("DirectRequest", "DirectRequest", ("text_api", "text_weights")),
        Method("HumanJailbreaks", "HumanJailbreaks", ("text_api", "text_weights")),
        Method("ZeroShot", "ZeroShot", ("text_api", "text_weights")),
        Method("PAP-top5", "PAP-top5", ("text_api", "text_weights")),
        Method("TAP-Transfer", "TAP-Transfer", ("text_api", "text_weights")),
        Method("MultiModalDirectRequest", "MultiModalDirectRequest", ("vision_api", "vision_weights")),
        Method("MultiModalRenderText", "MultiModalRenderText", ("vision_api", "vision_weights")),
        Method("GCG", "GCG", ("text_weights",)),
        Method("GCG-Multi", "GCG-Multi", ("text_weights",)),
        Method("AutoPrompt", "AutoPrompt", ("text_weights",)),
        Method("GBDA", "GBDA", ("text_weights",)),
        Method("PEZ", "PEZ", ("text_weights",)),
        Method("UAT", "UAT", ("text_weights",)),
        Method("AutoDAN", "AutoDAN", ("text_weights",)),
        Method("FewShot", "FewShot", ("text_weights",)),
        Method("PAIR", "PAIR", ("text_api", "text_weights")),
        Method("TAP", "TAP", ("text_api", "text_weights")),
        Method("GCG-Transfer", "GCG-Transfer", ("text_api", "text_weights")),
        Method("Fresh PAIR against the client model", "PAIR", ("text_api", "text_weights"), "client_fresh"),
        Method("Fresh TAP against the client model", "TAP", ("text_api", "text_weights"), "client_fresh"),
        Method("MultiModalPGD", "MultiModalPGD", ("vision_weights",)),
        Method("MultiModalPGDPatch", "MultiModalPGDPatch", ("vision_weights",)),
    )
}


def get_method(name: str) -> Method:
    try:
        return METHODS[name]
    except KeyError as error:
        raise ValueError(f"unknown HarmBench method {name!r}; expected one of {', '.join(METHODS)}") from error
