# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the complete pinned OR-Bench Hard plus toxic-control protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path


DATASET_REPOSITORY = "orbench-llm/or-bench"
DATASET_REVISION = "6f776f103b6439a6d8180b20f64b55f491223233"
CODE_REPOSITORY = "orbench/or-bench"
CODE_REVISION = "eef58d7522936e88a7d31f8f8506eaa29ef963fc"
FILES = {
    "hard": {
        "name": "or-bench-hard-1k.csv",
        "sha256": "a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d",
        "rows": 1319,
        "expected_behavior": "answer",
    },
    "toxic": {
        "name": "or-bench-toxic.csv",
        "sha256": "3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057",
        "rows": 655,
        "expected_behavior": "refuse",
    },
}
CATEGORIES = {
    "deception",
    "harassment",
    "harmful",
    "hate",
    "illegal",
    "privacy",
    "self-harm",
    "sexual",
    "unethical",
    "violence",
}
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "or_bench_hard.jsonl"


def _download(name: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/{DATASET_REVISION}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - immutable public dataset
        return response.read()


def render(sources: dict[str, bytes]) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for split, spec in FILES.items():
        content = sources[split]
        digest = hashlib.sha256(content).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"OR-Bench {split} hash drift: {digest} != {spec['sha256']}")
        rows = list(csv.DictReader(content.decode("utf-8").splitlines()))
        if len(rows) != spec["rows"]:
            raise ValueError(f"OR-Bench {split} row-count drift: {len(rows)} != {spec['rows']}")
        if {row["category"] for row in rows} != CATEGORIES:
            raise ValueError(f"OR-Bench {split} category grid drifted")
        for index, row in enumerate(rows):
            prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
            if prompt_hash in seen:
                raise ValueError("OR-Bench contains a duplicate prompt across the public protocol")
            seen.add(prompt_hash)
            record = {
                "or_bench_id": f"or-bench-{split}-{index:04d}",
                "responses_create_params": {"input": [{"role": "user", "content": row["prompt"]}]},
                "verifier_metadata": {
                    "split": split,
                    "category": row["category"],
                    "expected_behavior": spec["expected_behavior"],
                    "source_row_index": index,
                    "prompt_sha256": prompt_hash,
                    "dataset_revision": DATASET_REVISION,
                    "source_file_sha256": spec["sha256"],
                    "code_revision": CODE_REVISION,
                },
            }
            output.append(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if len(output) != 1974 or Counter(json.loads(line)["verifier_metadata"]["split"] for line in output) != {
        "hard": 1319,
        "toxic": 655,
    }:
        raise AssertionError("OR-Bench full public protocol did not reconcile")
    return "".join(output)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare() -> Path:
    sources = {split: _download(str(spec["name"])) for split, spec in FILES.items()}
    _atomic_write(OUTPUT_FPATH, render(sources))
    print(f"Wrote 1,974 OR-Bench Hard plus toxic-control rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
