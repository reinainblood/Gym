# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from benchmarks.facts_parametric.prepare import prepare


def test_prepare_converts_every_source_row(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "FACTS-Parametric-public.csv"
    output = prepare(source, tmp_path / "facts.jsonl")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1052
    assert rows[0] == {
        "id": "facts_parametric_0000",
        "question": "yair yint aung birthday",
        "expected_answer": "August 18, 1993",
        "source_url": "https://en.wikipedia.org/wiki/Yair_Yint_Aung",
        "topic": "birthday",
        "responses_create_params": {"input": [{"role": "user", "content": "yair yint aung birthday"}]},
    }
