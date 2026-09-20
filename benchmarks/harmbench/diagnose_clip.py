# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose a classifier clipping disagreement without exposing generation text."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from benchmarks.harmbench.calibrate import Control, _prompt, upstream_label
from resources_servers.harmbench.app import CLASSIFIER_MODEL, CLASSIFIER_REVISION, clip_generation


def diagnose(rollouts: Path, behavior_id: str, case_index: int, classifier_url: str) -> dict:
    row = next(
        row
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if line
        if (row := json.loads(line))["harmbench_id"] == behavior_id
        and row.get("attack_test_case_index", 0) == case_index
    )
    tokenizer_path = hf_hub_download(CLASSIFIER_MODEL, "tokenizer.model", revision=CLASSIFIER_REVISION)
    control = Control(classifier_url, tokenizer_path)
    sentencepiece_text, sentencepiece_count, _ = clip_generation(control.tokenizer, row["generation"], 512)
    upstream_text, upstream_count = control.hf_clip(row["generation"], 512)
    differences = []
    for operation, start_a, end_a, start_b, end_b in difflib.SequenceMatcher(
        None, sentencepiece_text, upstream_text, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            continue
        differences.append(
            {
                "operation": operation,
                "sentencepiece_offset": start_a,
                "sentencepiece_chars": end_a - start_a,
                "upstream_offset": start_b,
                "upstream_chars": end_b - start_b,
                "sentencepiece_whitespace_only": sentencepiece_text[start_a:end_a].isspace(),
                "upstream_whitespace_only": upstream_text[start_b:end_b].isspace(),
            }
        )
    upstream_prompt = _prompt(row["behavior"], row.get("context", ""), upstream_text)
    upstream_clip_label = upstream_label(control.classify_raw(upstream_prompt))
    result = {
        "harmbench_id": behavior_id,
        "attack_test_case_index": case_index,
        "sentencepiece_length": len(sentencepiece_text),
        "upstream_length": len(upstream_text),
        "sentencepiece_token_count": sentencepiece_count,
        "upstream_token_count": upstream_count,
        "sentencepiece_sha256": hashlib.sha256(sentencepiece_text.encode()).hexdigest(),
        "upstream_sha256": hashlib.sha256(upstream_text.encode()).hexdigest(),
        "differences": differences,
        "gym_label": row["classifier_label"],
        "upstream_clip_label": upstream_clip_label,
        "upstream_prompt_sha256": hashlib.sha256(upstream_prompt.encode()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--classifier-url", required=True)
    args = parser.parse_args()
    diagnose(args.rollouts, args.behavior_id, args.case_index, args.classifier_url)
