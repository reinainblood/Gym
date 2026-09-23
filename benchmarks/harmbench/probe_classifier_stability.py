# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe one saved HarmBench classifier decision without printing payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.harmbench.calibrate import Control, _prompt, upstream_label


def probe(
    *, rollouts: Path, classifier_url: str, behavior_id: str, attack_index: int, repeats: int, parallel: int = 1
) -> dict:
    if repeats < 1 or repeats > 50:
        raise ValueError("repeats must be between 1 and 50")
    if parallel < 1 or parallel > repeats:
        raise ValueError("parallel must be between 1 and repeats")
    matches = [
        row
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["harmbench_id"] == behavior_id and row["attack_test_case_index"] == attack_index
    ]
    if len(matches) != 1:
        raise ValueError("expected one matching saved rollout")
    row = matches[0]
    if row.get("scoring_method") != "harmbench_classifier":
        raise ValueError("selected case was not classified by HarmBench")
    prompt = _prompt(row["behavior"], row.get("context", ""), row["generation_for_classifier"])
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    if prompt_sha256 != row["classifier_prompt_sha256"]:
        raise ValueError("reconstructed prompt differs from the saved verifier prompt")
    control = Control(classifier_url, None)
    if not control.tokens_identical(prompt):
        raise ValueError("raw and chat classifier token IDs differ")

    def sample(_: int) -> tuple[int, int]:
        return upstream_label(control.classify_raw(prompt)), upstream_label(control.classify_chat(prompt))

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        samples = list(pool.map(sample, range(repeats)))
    raw_labels = [raw for raw, _ in samples]
    chat_labels = [chat for _, chat in samples]
    return {
        "behavior_id": behavior_id,
        "attack_test_case_index": attack_index,
        "rollouts_sha256": hashlib.sha256(rollouts.read_bytes()).hexdigest(),
        "classifier_prompt_sha256": prompt_sha256,
        "saved_label": row["classifier_label"],
        "prompt_token_ids_identical": True,
        "repeats": repeats,
        "parallel": parallel,
        "raw_labels": raw_labels,
        "chat_labels": chat_labels,
        "raw_counts": dict(sorted(Counter(raw_labels).items())),
        "chat_counts": dict(sorted(Counter(chat_labels).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--classifier-url", required=True)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--attack-index", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = probe(
        rollouts=args.rollouts,
        classifier_url=args.classifier_url,
        behavior_id=args.behavior_id,
        attack_index=args.attack_index,
        repeats=args.repeats,
        parallel=args.parallel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Saved label {result['saved_label']}; raw counts {result['raw_counts']}; "
        f"chat counts {result['chat_counts']} across {result['repeats']} repeats"
    )


if __name__ == "__main__":
    main()
