# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the AgentDojo benchmark and split it into strided shards.

    python -m benchmarks.agentdojo.baseline.shards materialize

A cell's score is a mean over its row set, so N shards score identically to one run. Shards
are strided (row i goes to shard i % N) rather than contiguous: the file is ordered by suite,
and workspace rows are the slow tail, so contiguous shards would finish hours apart.

Writes `benchmarks/agentdojo/data/shards/agentdojo_benchmark.shard{i}of{n}.jsonl` for every
N in SHARD_COUNTS. Deterministic: same input file, same shards.
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.agentdojo.prepare import prepare


SHARD_COUNTS = (1, 2, 3, 4, 5, 6, 8)
SHARD_DIR = Path(__file__).resolve().parents[1] / "data" / "shards"


def shard_path(index: int, shards: int) -> Path:
    return SHARD_DIR / f"agentdojo_benchmark.shard{index}of{shards}.jsonl"


def shard_rows(index: int, shards: int, total: int = 1046) -> int:
    return len(range(index, total, shards))


def materialize() -> None:
    source = prepare()
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for shards in SHARD_COUNTS:
        for index in range(shards):
            shard_path(index, shards).write_text("".join(lines[index::shards]), encoding="utf-8")
    print(f"Wrote shards {SHARD_COUNTS} of {len(lines)} rows to {SHARD_DIR}")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["materialize"]):
        raise SystemExit("usage: python -m benchmarks.agentdojo.baseline.shards [materialize]")
    materialize()
