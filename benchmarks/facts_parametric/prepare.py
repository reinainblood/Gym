# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the FACTS Parametric CSV release into NeMo Gym JSONL."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
SOURCE_FPATH = BENCHMARK_DIR / "FACTS-Parametric-public.csv"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "facts_parametric_benchmark.jsonl"


def prepare(source_fpath: Path = SOURCE_FPATH, output_fpath: Path = OUTPUT_FPATH) -> Path:
    """Write flat, task-owned Gym rows while preserving FACTS provenance."""
    with source_fpath.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"url", "query", "answer", "topic"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required_columns)}, got {reader.fieldnames}")

        rows = []
        for index, source_row in enumerate(reader):
            question = source_row["query"].strip()
            expected_answer = source_row["answer"].strip()
            if not question or not expected_answer:
                raise ValueError(f"Row {index} has an empty question or answer")
            rows.append(
                {
                    "id": f"facts_parametric_{index:04d}",
                    "question": question,
                    "expected_answer": expected_answer,
                    "source_url": source_row["url"].strip(),
                    "topic": source_row["topic"].strip(),
                    "responses_create_params": {"input": [{"role": "user", "content": question}]},
                }
            )

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_fpath


if __name__ == "__main__":
    print(prepare())
