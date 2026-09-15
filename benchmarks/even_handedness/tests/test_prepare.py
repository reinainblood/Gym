# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from benchmarks.even_handedness.prepare import OUTPUT_FPATH, SOURCE_FPATH, prepare


def test_prepare_preserves_paired_prompts(tmp_path: Path) -> None:
    source = tmp_path / "eval_set.csv"
    output = tmp_path / "prepared.jsonl"
    source.write_text(
        "split,main_category,topic_name,partisan,template_category,template,stance_a,stance_b,prompt_a,prompt_b,prompt_a_group,prompt_b_group\n"
        "eval,category,topic,True,reasoning,Argue {stance},left,right,prompt left,prompt right,left_group,right_group\n",
        encoding="utf-8",
    )

    prepare(source, output)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["prompt_a"] == "prompt left"
    assert row["prompt_b"] == "prompt right"
    assert row["responses_create_params"]["input"][0]["content"] == "prompt left"


def test_prepare_has_cli_compatible_defaults() -> None:
    assert prepare.__defaults__ == (SOURCE_FPATH, OUTPUT_FPATH)
