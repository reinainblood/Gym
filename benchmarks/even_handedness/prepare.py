# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Anthropic's CC-BY-4.0 Political Even-handedness CSV to Gym JSONL."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen


BENCHMARK_DIR = Path(__file__).parent
SOURCE_FPATH = BENCHMARK_DIR / "upstream" / "eval_set.csv"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "even_handedness_eval.jsonl"
UPSTREAM_REVISION = "c5ed67908b56edc0781f47821241ca44114bd4ff"
UPSTREAM_SOURCE_SHA256 = "b02e49e2390c4f03225f176fa7a132858a3fd0d33eced9bae86ecb9f11670cf3"
UPSTREAM_SOURCE_URL = (
    "https://raw.githubusercontent.com/anthropics/political-neutrality-eval/"
    f"{UPSTREAM_REVISION}/eval_set.csv"
)

REQUIRED_COLUMNS = {
    "split",
    "main_category",
    "topic_name",
    "partisan",
    "template_category",
    "template",
    "stance_a",
    "stance_b",
    "prompt_a",
    "prompt_b",
    "prompt_a_group",
    "prompt_b_group",
}


def _validate_source(source_fpath: Path) -> None:
    actual_hash = hashlib.sha256(source_fpath.read_bytes()).hexdigest()
    if actual_hash != UPSTREAM_SOURCE_SHA256:
        raise ValueError(
            f"{source_fpath} has SHA-256 {actual_hash}, expected {UPSTREAM_SOURCE_SHA256} "
            f"from {UPSTREAM_REVISION}"
        )


def _download_source(source_fpath: Path) -> Path:
    if source_fpath.exists():
        _validate_source(source_fpath)
        return source_fpath
    source_fpath.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(UPSTREAM_SOURCE_URL) as response:
        source_fpath.write_bytes(response.read())
    _validate_source(source_fpath)
    return source_fpath


def prepare(source_fpath: Path = SOURCE_FPATH, output_fpath: Path = OUTPUT_FPATH) -> Path:
    """Prepare the pinned Anthropic CC-BY evaluation CSV as Gym JSONL."""
    if source_fpath == SOURCE_FPATH:
        source_fpath = _download_source(source_fpath)
    with source_fpath.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(REQUIRED_COLUMNS)}, got {reader.fieldnames}")
        rows = []
        for index, row in enumerate(reader):
            if not row["prompt_a"].strip() or not row["prompt_b"].strip():
                raise ValueError(f"Row {index} has an empty paired prompt")
            rows.append(
                {
                    "id": f"even_handedness_{index:04d}",
                    **{key: row[key] for key in REQUIRED_COLUMNS},
                    "responses_create_params": {"input": [{"role": "user", "content": row["prompt_a"]}]},
                    "prompt_config": None,
                    "source_revision": UPSTREAM_REVISION,
                    "source_sha256": UPSTREAM_SOURCE_SHA256,
                }
            )
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return output_fpath


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) == 2 else SOURCE_FPATH
    print(prepare(source, OUTPUT_FPATH))
