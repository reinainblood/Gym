# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queue the complete K3 HarmBench white-box campaign on the deployed FDR app."""

from __future__ import annotations

import json

import modal


APP_NAME = "harmbench-kimi-k3-whitebox"
TEXT_CALLS = [
    ("UAT", ""),
    ("GCG", ""),
    ("AutoPrompt", ""),
    ("GBDA", ""),
    ("PEZ", ""),
    ("AutoDAN", ""),
    ("FewShot", ""),
    ("GCG-Multi", "0"),
    ("GCG-Multi", "1"),
    ("GCG-Multi", "2"),
    ("GCG-Multi", "3"),
    ("GCG-Multi", "4"),
]
VISION_CALLS = ["MultiModalPGD", "MultiModalPGDPatch", "MultiModalPGDBlankImage"]


def main() -> None:
    text = modal.Function.from_name(APP_NAME, "run_text_method", environment_name="FDR")
    vision = modal.Function.from_name(APP_NAME, "run_multimodal_method", environment_name="FDR")
    calls = []
    for method, run_id in TEXT_CALLS:
        call = text.spawn(method, run_id)
        calls.append({"method": method, "run_id": run_id or None, "call_id": call.object_id})
    for method in VISION_CALLS:
        call = vision.spawn(method)
        calls.append({"method": method, "run_id": None, "call_id": call.object_id})
    print(
        json.dumps(
            {"app": APP_NAME, "environment": "FDR", "calls": calls},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
