# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json

from benchmarks.or_bench_hard.prepare import CATEGORIES, FILES, render


def _csv(rows: int) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["prompt", "category"])
    writer.writeheader()
    categories = sorted(CATEGORIES)
    for index in range(rows):
        writer.writerow({"prompt": f"prompt {rows}-{index}", "category": categories[index % len(categories)]})
    return stream.getvalue().encode()


def test_full_public_protocol_counts_and_source_fields(monkeypatch):
    hard, toxic = _csv(1319), _csv(655)
    monkeypatch.setitem(FILES["hard"], "sha256", __import__("hashlib").sha256(hard).hexdigest())
    monkeypatch.setitem(FILES["toxic"], "sha256", __import__("hashlib").sha256(toxic).hexdigest())
    rows = [json.loads(line) for line in render({"hard": hard, "toxic": toxic}).splitlines()]
    assert len(rows) == 1974
    assert sum(row["verifier_metadata"]["split"] == "hard" for row in rows) == 1319
    assert sum(row["verifier_metadata"]["split"] == "toxic" for row in rows) == 655
    assert len({row["or_bench_id"] for row in rows}) == 1974
