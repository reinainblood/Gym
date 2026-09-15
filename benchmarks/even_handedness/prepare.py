# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Anthropic's CC-BY-4.0 Political Even-handedness CSV to Gym JSONL."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
SOURCE_FPATH = BENCHMARK_DIR / "eval_set.csv"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "even_handedness_eval.jsonl"

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


def prepare(source_fpath: Path = SOURCE_FPATH, output_fpath: Path = OUTPUT_FPATH) -> Path:
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
                }
            )
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return output_fpath


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) == 2 else SOURCE_FPATH
    print(prepare(source, OUTPUT_FPATH))
