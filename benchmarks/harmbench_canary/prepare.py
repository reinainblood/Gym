# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a hash-bound, two-row copyright canary selection for Gym."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SELECTED = Path(__file__).resolve().parents[1] / "harmbench/data/harmbench_copyright_canary.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> Path:
    selected_path = SELECTED.resolve(strict=True)
    receipt = json.loads(selected_path.with_suffix(".selection.json").read_text(encoding="utf-8"))
    source = Path(receipt["source_path"]).resolve(strict=True)
    if _sha256(source) != receipt["source_sha256"] or _sha256(selected_path) != receipt["output_sha256"]:
        raise ValueError("copyright canary source or selection hash changed")
    rows = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2 or [row["harmbench_id"] for row in rows] != [
        entry["harmbench_id"] for entry in receipt["cases"]
    ]:
        raise ValueError("copyright canary no longer matches its selection receipt")
    if any("hash_check" not in row["tags"] or row["method"] != receipt["method"] for row in rows):
        raise ValueError("copyright canary contains an unexpected method or scoring path")
    if {"book", "lyrics"} != {"book" if "book" in row["tags"] else "lyrics" for row in rows}:
        raise ValueError("copyright canary must contain one book and one lyrics case")
    print(f"Validated two hash-bound {receipt['method']} copyright canary cases: {selected_path}")
    return selected_path


if __name__ == "__main__":
    prepare()
