# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select one book and one lyrics case without printing attack text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def select(source: Path) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "hash_check" not in row.get("tags", []):
                continue
            for kind in ("book", "lyrics"):
                if kind in row["tags"] and kind not in selected:
                    selected[kind] = row
            if len(selected) == 2:
                break
    if set(selected) != {"book", "lyrics"}:
        raise ValueError("dataset lacks a book or lyrics copyright case")
    return [selected["book"], selected["lyrics"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = select(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    receipt = {
        "source_path": str(args.source.resolve(strict=True)),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "method": rows[0]["method"],
        "cases": [{"harmbench_id": row["harmbench_id"], "tags": row["tags"]} for row in rows],
    }
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Selected {len(rows)} copyright canary cases at {args.output}")


if __name__ == "__main__":
    main()
