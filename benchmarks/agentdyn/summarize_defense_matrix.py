# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the AgentDyn defense grid against the undefended baselines.

Reads whatever cells `run_defense_matrix.sh` has finished and prints a markdown ledger plus
a provenance manifest. Incomplete cells are reported as incomplete rather than averaged into
a headline, and a cell that masked rollouts is flagged: a masked row is an adapter failure,
not a secure one.

    python benchmarks/agentdyn/summarize_defense_matrix.py [--results DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
BASELINE_MANIFEST = BENCHMARK_DIR / "baseline-manifest-20260919.json"
EXPECTED_ROWS = 620
# Shard head ports reserved per cell by launch_defense_grid.sh; caps SHARDS at this value.
SHARD_HEAD_PORTS_PER_CELL = 4
# Display order: cheap detectors, then the code/policy defenses, then DRIFT. Matches the
# execution order in run_defense_matrix.sh.
DEFENSE_ORDER = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
    "camel",
    "progent",
    "drift",
)
# Port assignment order, which is NOT the display order above: `launch_defense_grid.sh`
# derives each cell's head port from a defense's index in its own DEFENSE_KEYS list, and that
# list is in cheap-first execution order. Mirror it exactly -- reading it as the display order
# reports the wrong cell as running.
CELL_PORT_DEFENSE_ORDER = (
    "prompt_guard_2_detector",
    "camel",
    "progent",
    "piguard_detector",
    "drift",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
)
DEFENSES_PER_MODEL = len(CELL_PORT_DEFENSE_ORDER)
# Launcher model keys, by slug. The grid scripts take the key; everything else uses the slug.
MODEL_KEYS = {
    "nemotron-3-ultra": "ultra",
    "kimi-k3": "kimi",
    "qwen-3-5-122b-a10b": "qwen",
    "nemotron-3-5-super-vl": "supervl",
}
# Rough policy calls per rollout, from the canaries. Only used to start the most expensive
# cells first, so an interrupted grid is left with the cheap work rather than the long tail.
DEFENSE_COST = {
    "drift": 55,
    "progent": 16,
    "piguard_detector": 14,
    "transformers_pi_detector": 12,
    "prompt_guard_2_detector": 10,
    "tool_filter": 10,
    "spotlighting_with_delimiting": 9,
    "repeat_user_prompt": 9,
    "camel": 4,
}
MODEL_SLUGS = {
    "nemotron-3-ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "kimi-k3": "moonshotai/Kimi-K3",
    "qwen-3-5-122b-a10b": "Qwen/Qwen3.5-122B-A10B-FP8",
    "nemotron-3-5-super-vl": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def _delta(value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return "--"
    return f"{(value - reference) * 100:+.2f}"


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a cell from its rollouts, exactly as the agent's `compute_metrics` does.

    Masked rollouts leave the denominator first -- a masked row is an adapter failure, and
    counting it as secure would flatter the defense -- and then utility and attack success
    are averaged over the benign and attacked subsets. That makes a cell's score a pure
    function of its row set, which is what lets a sharded cell be scored by merging its
    shards: 620 rows collected by four processes score identically to 620 collected by one.
    """
    scored = [row for row in rows if not row.get("mask_sample", False)]
    attacked = [row for row in scored if row.get("injection_task_id") is not None]
    benign = [row for row in scored if row.get("injection_task_id") is None]
    calls = [row.get("model_call_count") for row in scored if row.get("model_call_count") is not None]

    def mean(subset: list[dict[str, Any]], field: str) -> float | None:
        if not subset:
            return None
        return sum(float(bool(row.get(field, False))) for row in subset) / len(subset)

    return {
        "scored_rollout_count": len(scored),
        "masked_rollout_count": len(rows) - len(scored),
        "benign_utility": mean(benign, "utility"),
        "utility_under_attack": mean(attacked, "utility"),
        "attack_success_rate": mean(attacked, "attack_success"),
        "mean_model_calls": (sum(calls) / len(calls)) if calls else None,
    }


def read_cell(rollout_paths: list[Path]) -> dict[str, Any]:
    """Collect one (model, defense) cell's scores and provenance, merging any shards."""
    rows: list[dict[str, Any]] = []
    adapter_failures = 0
    for path in rollout_paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
        failures = Path(f"{path.with_suffix('')}_failures.jsonl")
        if failures.is_file():
            adapter_failures += sum(1 for line in failures.open(encoding="utf-8") if line.strip())

    cell: dict[str, Any] = {
        "rollout_paths": [str(path) for path in rollout_paths],
        "shards": len(rollout_paths),
        "rows": len(rows),
        "adapter_failures": adapter_failures,
        "rollouts_sha256": {str(path): _sha256(path) for path in rollout_paths},
    }
    cell.update(score_rows(rows))
    # A cell can be finished without being full. Gym retires a rollout after three failed
    # attempts and never re-dispatches it, so the runner records that the cell settled short
    # rather than letting the tender relaunch it forever against work that cannot land. Such
    # a cell is done, and its shortfall is reported rather than hidden.
    settled = [path for path in (Path(f"{p.with_suffix('')}.settled") for p in rollout_paths) if path.is_file()]
    cell["settled_short"] = bool(settled) and cell["rows"] < EXPECTED_ROWS
    cell["settled_notes"] = [path.read_text(encoding="utf-8").strip() for path in settled]
    cell["complete"] = cell["rows"] >= EXPECTED_ROWS
    cell["finished"] = cell["complete"] or (bool(settled) and len(settled) == len(rollout_paths))
    return cell


#: Launcher fragment -> results-directory slug. The Modal harness names a cell
#: `<fragment>-<defense>.jsonl` in one flat namespace directory, while a local run writes
#: `<slug>/<defense>.jsonl`. Both layouts hold the same rows, so both are read here rather
#: than making whoever writes the ledger reshuffle files by hand.
FRAGMENT_TO_SLUG = {
    "ultra": "nemotron-3-ultra",
    "kimi": "kimi-k3",
    "qwen": "qwen-3-5-122b-a10b",
    "supervl": "nemotron-3-5-super-vl",
}


def collect_flat(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the Modal harness's flat `<fragment>-<defense>.jsonl` layout.

    A cell too slow to finish in one container is collected as
    `<fragment>-<defense>-shardNofM.jsonl` instead, one file per contiguous slice of the
    selector list. Those merge back here: a cell's score is a mean over its row set, so the
    number of containers that produced the set cannot change it. The whole-cell file and the
    shards are never both read -- a sharded cell's selectors are all covered by its shards,
    so counting a leftover whole-cell file too would double-weight the rows it duplicates.
    """
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for fragment, slug in FRAGMENT_TO_SLUG.items():
        for defense in DEFENSE_ORDER:
            whole = results_dir / f"{fragment}-{defense}.jsonl"
            # Anchored, because `-shard*.jsonl` also matches the sidecars Gym writes beside
            # each rollouts file -- `-shard0of8_materialized_inputs.jsonl` holds one line per
            # selector, so counting it reported an empty shard as complete.
            shard_name = re.compile(rf"^{re.escape(fragment)}-{re.escape(defense)}-shard\d+of\d+\.jsonl$")
            shards = sorted(
                path for path in results_dir.glob(f"{fragment}-{defense}-shard*.jsonl") if shard_name.match(path.name)
            )
            paths = shards or ([whole] if whole.is_file() else [])
            if paths:
                matrix.setdefault(slug, {})[defense] = read_cell(paths)
    return matrix


def collect(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    flat = collect_flat(results_dir)
    if flat:
        return flat
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for slug in MODEL_SLUGS:
        for defense in DEFENSE_ORDER:
            whole = results_dir / slug / f"{defense}.jsonl"
            # Anchored, because `{defense}-shard*of*.jsonl` also matches the sidecars Gym
            # writes beside each rollouts file -- `-shard0of3_materialized_inputs.jsonl` holds
            # one line per selector, so counting it reported an empty cell as complete.
            shard_name = re.compile(rf"^{re.escape(defense)}-shard\d+of\d+\.jsonl$")
            shards = (
                sorted(
                    path
                    for path in (results_dir / slug).glob(f"{defense}-shard*.jsonl")
                    if shard_name.match(path.name)
                )
                if (results_dir / slug).is_dir()
                else []
            )
            paths = ([whole] if whole.is_file() else []) + shards
            if not paths:
                continue
            matrix.setdefault(slug, {})[defense] = read_cell(paths)
    return matrix


def runner_state(results_dir: Path, slug: str, defense: str) -> str:
    """Is a live runner still working this cell?

    `launch_defense_grid.sh` gives each cell a fixed head port, and the runner drops a lock
    holding the driver's pid which its exit trap removes. A SIGKILL cannot run that trap,
    which is what makes a dead driver legible: the lock outlives it. That matters because a
    killed driver is otherwise invisible -- its current `gym eval run` child keeps writing
    rows for a while, so the log's last line looks normal while the cell has stopped
    advancing.
    """
    try:
        model_index = list(MODEL_SLUGS).index(slug)
        defense_index = CELL_PORT_DEFENSE_ORDER.index(defense)
    except ValueError:
        return "unknown"
    cell_index = model_index * DEFENSES_PER_MODEL + defense_index

    def live(head_port: int) -> bool | None:
        lock = results_dir / f".runner.{head_port}.lock"
        if not lock.is_file():
            return None
        try:
            os.kill(int(lock.read_text().strip()), 0)
        except (OSError, ValueError):
            return False
        return True

    # A sharded cell has one lock per shard, on the shard head ports the launcher assigns.
    # It reserves exactly SHARD_HEAD_PORTS_PER_CELL of them, so scanning further would read
    # the next cell's shards as this one's.
    shard_states = [
        state
        for shard in range(SHARD_HEAD_PORTS_PER_CELL)
        if (state := live(11900 + cell_index * SHARD_HEAD_PORTS_PER_CELL + shard)) is not None
    ]
    if shard_states:
        running = sum(shard_states)
        return f"running ({running}/{len(shard_states)} shards)" if running else "DIED (relaunch)"

    whole = live(11820 + model_index * 10 + defense_index)
    if whole is None:
        return "not running"
    return "running" if whole else "DIED (relaunch)"


def baselines() -> dict[str, dict[str, Any]]:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    return {run["model"]: run for run in manifest["runs"]}


def render(matrix: dict[str, dict[str, dict[str, Any]]]) -> str:
    reference = baselines()
    lines: list[str] = []
    for slug, model in MODEL_SLUGS.items():
        cells = matrix.get(slug)
        if not cells:
            continue
        base = reference.get(model, {})
        lines.append(f"### {model}\n")
        lines.append(
            "| Defense | Rows | Benign utility | Utility under attack | ASR | "
            "ASR delta vs undefended | Mean calls | Masked | Adapter errors |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| _(undefended baseline)_ | {EXPECTED_ROWS} | {_pct(base.get('benign_utility'))} | "
            f"{_pct(base.get('utility_under_attack'))} | {_pct(base.get('attack_success_rate'))} | "
            f"-- | -- | 0 | 0 |"
        )
        for defense in DEFENSE_ORDER:
            cell = cells.get(defense)
            if cell is None:
                continue
            if cell["complete"]:
                rows = f"{cell['rows']} / {EXPECTED_ROWS}"
            elif cell["settled_short"]:
                rows = f"{cell['rows']} / {EXPECTED_ROWS} (settled short)"
            else:
                rows = f"{cell['rows']} / {EXPECTED_ROWS} (incomplete)"
            if cell["shards"] > 1:
                rows += f" [{cell['shards']} shards]"
            calls = "--" if cell.get("mean_model_calls") is None else f"{cell['mean_model_calls']:.1f}"
            lines.append(
                f"| `{defense}` | {rows} | {_pct(cell.get('benign_utility'))} | "
                f"{_pct(cell.get('utility_under_attack'))} | {_pct(cell.get('attack_success_rate'))} | "
                f"{_delta(cell.get('attack_success_rate'), base.get('attack_success_rate'))} | {calls} | "
                f"{cell.get('masked_rollout_count', '--')} | {cell['adapter_failures']} |"
            )
        lines.append("")
    if not lines:
        return "No defense cells found yet."
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/agentdyn-defense-matrix"))
    parser.add_argument("--json", type=Path, default=None, help="Write the provenance manifest here.")
    parser.add_argument(
        "--next",
        type=int,
        default=None,
        metavar="N",
        help="Print up to N outstanding cells with no live runner, costliest first, as '<model_key> <defense>'.",
    )
    args = parser.parse_args()

    matrix = collect(args.results)

    if args.next is not None:
        # Machine-readable, for tend_defense_grid.sh; nothing else is printed.
        candidates = [
            (DEFENSE_COST.get(defense, 0), MODEL_KEYS[slug], defense)
            for slug in MODEL_SLUGS
            for defense in DEFENSE_ORDER
            if not (matrix.get(slug, {}).get(defense) or {}).get("finished")
            and runner_state(args.results, slug, defense) == "not running"
        ]
        for _, key, defense in sorted(candidates, reverse=True)[: args.next]:
            print(f"{key} {defense}")
        return

    print(render(matrix))

    completed = sum(1 for cells in matrix.values() for cell in cells.values() if cell["complete"])
    total = len(MODEL_SLUGS) * len(DEFENSE_ORDER)
    collected = sum(cell["rows"] for cells in matrix.values() for cell in cells.values())
    settled_short = sum(1 for cells in matrix.values() for cell in cells.values() if cell["settled_short"])
    print(f"\nCompleted cells: {completed} / {total}")
    if settled_short:
        print(f"Settled short (finished, but missing rows Gym retired): {settled_short}")
    print(f"Rollouts collected: {collected} / {total * EXPECTED_ROWS}")

    pending = [
        (slug, defense, matrix.get(slug, {}).get(defense))
        for slug in MODEL_SLUGS
        for defense in DEFENSE_ORDER
        if not (matrix.get(slug, {}).get(defense) or {}).get("finished")
    ]
    if pending:
        print("\nOutstanding cells (re-run launch_defense_grid.sh to pick up anything not running):")
        for slug, defense, cell in pending:
            rows = cell["rows"] if cell else 0
            state = runner_state(args.results, slug, defense)
            print(f"  {slug:22} {defense:24} {rows:4} / {EXPECTED_ROWS}  {state}")

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "benchmark": "AgentDyn",
                    "benchmark_version": "v1.2.2",
                    "condition": "defended",
                    "selector_sha256": _sha256(BENCHMARK_DIR / "data" / "agentdyn_v1_2_2.jsonl"),
                    "expected_rows_per_cell": EXPECTED_ROWS,
                    "completed_cells": completed,
                    "total_cells": total,
                    "cells": matrix,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote manifest to {args.json}")


if __name__ == "__main__":
    main()
