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
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
BASELINE_MANIFEST = BENCHMARK_DIR / "baseline-manifest-20260919.json"
EXPECTED_ROWS = 620
# Display order: cheap detectors, then the code/policy defenses, then DRIFT. Matches the
# execution order in run_defense_matrix.sh.
DEFENSE_ORDER = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "camel",
    "progent",
    "drift",
)
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


def read_cell(rollouts: Path) -> dict[str, Any]:
    """Collect one (model, defense) cell's scores and provenance from its artifacts."""
    prefix = rollouts.with_suffix("")
    aggregate_path = Path(f"{prefix}_aggregate_metrics.json")
    failures_path = Path(f"{prefix}_failures.jsonl")
    cell: dict[str, Any] = {
        "rollouts_path": str(rollouts),
        "rows": sum(1 for _ in rollouts.open(encoding="utf-8")) if rollouts.is_file() else 0,
        "adapter_failures": sum(1 for _ in failures_path.open(encoding="utf-8")) if failures_path.is_file() else 0,
        "rollouts_sha256": _sha256(rollouts),
        "aggregate_metrics_sha256": _sha256(aggregate_path),
    }
    if aggregate_path.is_file():
        entries = json.loads(aggregate_path.read_text(encoding="utf-8"))
        metrics = entries[0]["agent_metrics"] if entries else {}
        cell.update(
            benign_utility=metrics.get("agentdyn/benign_utility"),
            utility_under_attack=metrics.get("agentdyn/utility_under_attack"),
            attack_success_rate=metrics.get("agentdyn/attack_success_rate"),
            scored_rollout_count=metrics.get("agentdyn/scored_rollout_count"),
            masked_rollout_count=metrics.get("agentdyn/masked_rollout_count"),
            mean_model_calls=metrics.get("mean/model_call_count"),
        )
    cell["complete"] = cell["rows"] >= EXPECTED_ROWS and cell.get("scored_rollout_count") == EXPECTED_ROWS
    return cell


def collect(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for slug in MODEL_SLUGS:
        for defense in DEFENSE_ORDER:
            rollouts = results_dir / slug / f"{defense}.jsonl"
            if not rollouts.is_file():
                continue
            matrix.setdefault(slug, {})[defense] = read_cell(rollouts)
    return matrix


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
            rows = f"{cell['rows']} / {EXPECTED_ROWS}" + ("" if cell["complete"] else " (incomplete)")
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
    args = parser.parse_args()

    matrix = collect(args.results)
    print(render(matrix))

    completed = sum(1 for cells in matrix.values() for cell in cells.values() if cell["complete"])
    total = len(MODEL_SLUGS) * len(DEFENSE_ORDER)
    print(
        f"\nCompleted cells: {completed} / {total}  ({completed * EXPECTED_ROWS} / {total * EXPECTED_ROWS} rollouts)"
    )

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
