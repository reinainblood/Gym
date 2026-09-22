# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shard-merging half of the ledger reader.

A sharded cell is only equivalent to an unsharded one if the summarizer puts the shards back
together exactly: every row counted once, and nothing counted that is not a rollout. Both
halves have failed in practice -- a `-shard*.jsonl` glob once matched Gym's
`_materialized_inputs` sidecar and reported an empty cell as complete -- so both are asserted
here rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agentdyn.summarize_defense_matrix import collect_flat, score_rows


def _rows(count: int, *, attacked: bool, utility: bool, attack_success: bool = False) -> list[dict]:
    return [
        {
            "injection_task_id": f"injection_{index}" if attacked else None,
            "utility": utility,
            "attack_success": attack_success,
            "model_call_count": 3,
        }
        for index in range(count)
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_shards_merge_into_one_cell(tmp_path: Path) -> None:
    _write(tmp_path / "supervl-camel-shard0of2.jsonl", _rows(4, attacked=False, utility=True))
    _write(tmp_path / "supervl-camel-shard1of2.jsonl", _rows(6, attacked=True, utility=False, attack_success=True))

    cell = collect_flat(tmp_path)["nemotron-3-5-super-vl"]["camel"]

    assert cell["shards"] == 2
    assert cell["rows"] == 10
    assert cell["benign_utility"] == 1.0
    assert cell["attack_success_rate"] == 1.0


def test_sidecars_are_not_counted_as_rollouts(tmp_path: Path) -> None:
    _write(tmp_path / "supervl-camel-shard0of2.jsonl", _rows(4, attacked=False, utility=True))
    # Gym writes these beside every rollouts file. `_materialized_inputs` holds one line per
    # selector, so counting it would report far more rows than were ever collected.
    _write(tmp_path / "supervl-camel-shard0of2_materialized_inputs.jsonl", _rows(78, attacked=False, utility=True))
    _write(tmp_path / "supervl-camel-shard0of2_failures.jsonl", _rows(2, attacked=False, utility=False))

    cell = collect_flat(tmp_path)["nemotron-3-5-super-vl"]["camel"]

    assert cell["rows"] == 4
    assert cell["shards"] == 1
    assert cell["adapter_failures"] == 2


def test_shards_win_over_a_stale_whole_cell_file(tmp_path: Path) -> None:
    """A resharded cell leaves its pre-shard partial behind; the shards already cover it.

    Reading both would double-weight whatever selectors the partial and the shards share,
    which shows up as a score no set of rollouts could produce rather than as an error.
    """
    _write(tmp_path / "supervl-camel.jsonl", _rows(5, attacked=True, utility=True, attack_success=True))
    _write(tmp_path / "supervl-camel-shard0of2.jsonl", _rows(4, attacked=True, utility=False))
    _write(tmp_path / "supervl-camel-shard1of2.jsonl", _rows(6, attacked=True, utility=False))

    cell = collect_flat(tmp_path)["nemotron-3-5-super-vl"]["camel"]

    assert cell["rows"] == 10
    assert cell["attack_success_rate"] == 0.0


def test_masked_rollouts_leave_the_denominator(tmp_path: Path) -> None:
    """A masked row is an adapter failure, not a defended one; scoring it as secure flatters."""
    rows = _rows(4, attacked=True, utility=False, attack_success=True)
    rows.extend({"injection_task_id": "injection_x", "mask_sample": True} for _ in range(6))

    scored = score_rows(rows)

    assert scored["scored_rollout_count"] == 4
    assert scored["masked_rollout_count"] == 6
    assert scored["attack_success_rate"] == 1.0
