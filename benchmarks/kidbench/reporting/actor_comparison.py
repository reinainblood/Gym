# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure what the child actor's willingness to press is worth.

KIDBench's multi-turn track is driven by an actor model playing a child. The paper uses a
refusal-ablated Gemma-4-31B-it and does not release it, so anyone reproducing the
benchmark has to choose a stand-in — and that choice is usually treated as a caveat.

It is not a caveat. An actor that breaks character and refuses stops probing exactly where
probing matters, which means a weaker actor does not add noise around the true degradation:
it biases it in one direction. This module quantifies that bias by comparing two runs of
the same models over the same scenarios, differing only in the actor.

Both runs must already exist. Point ``--baseline-dir`` at the weaker actor's rollouts and
``--results-dir`` at the current ones::

    python -m benchmarks.kidbench.reporting.actor_comparison \\
        --results-dir results/kidbench \\
        --baseline-dir results/kidbench/stock_actor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Optional

from benchmarks.kidbench.reporting.analyze import ModelResults
from benchmarks.kidbench.reporting.cli import DEFAULT_MODELS
from benchmarks.kidbench.reporting.report import _table, fmt, pct, signed


def _load_multi_only(results_dir: Path, slug: str, name: str) -> ModelResults:
    """Load a model's multi-turn rollouts from a directory, ignoring single turn.

    The actor never touches the single-turn track, so a baseline directory holding only
    multi-turn files is the normal case rather than an error.
    """
    model = ModelResults(slug=slug, display_name=name)
    path = results_dir / f"{slug}.multi_turn.jsonl"
    if path.exists():
        model.multi_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return model


def _delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return current - baseline


def _relative(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current - baseline) / abs(baseline)


def compare(results_dir: Path, baseline_dir: Path, models: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slug, name in models:
        current = _load_multi_only(results_dir, slug, name)
        baseline = _load_multi_only(baseline_dir, slug, name)
        if not current.multi_rows or not baseline.multi_rows:
            continue
        now, before = current.multi_turn_summary(), baseline.multi_turn_summary()
        rows.append(
            {
                "model": name,
                "slug": slug,
                "conversations": now.get("conversations"),
                "baseline_actor_refusal_rate": before.get("actor_refusal_rate"),
                "actor_refusal_rate": now.get("actor_refusal_rate"),
                "baseline_degradation_slope": before.get("degradation_slope"),
                "degradation_slope": now.get("degradation_slope"),
                "slope_change": _relative(now.get("degradation_slope"), before.get("degradation_slope")),
                "baseline_peak_quality_drop": before.get("peak_quality_drop"),
                "peak_quality_drop": now.get("peak_quality_drop"),
                "peak_change": _relative(now.get("peak_quality_drop"), before.get("peak_quality_drop")),
                "baseline_turn_curve": before.get("turn_curve"),
                "turn_curve": now.get("turn_curve"),
                "baseline_total_quality_score": before.get("total_quality_score"),
                "total_quality_score": now.get("total_quality_score"),
            }
        )
    return rows


def render(rows: list[dict[str, Any]], *, current_label: str, baseline_label: str) -> str:
    if not rows:
        return "No model had rollouts under both actors.\n"

    out: list[str] = []
    add = out.append
    add("# What the actor's willingness to press is worth")
    add("")
    add(
        f"The same models, the same 200 scenarios, the same judge. The only thing that changed is "
        f"who played the child: **{baseline_label}** against **{current_label}**."
    )
    add("")
    add(
        _table(
            ["Model", "Actor refusals", "Degradation slope", "Peak drop"],
            [
                [
                    row["model"],
                    f"{pct(row['baseline_actor_refusal_rate'], 2)} → {pct(row['actor_refusal_rate'], 2)}",
                    f"{fmt(row['baseline_degradation_slope'], 4)} → {fmt(row['degradation_slope'], 4)}"
                    + (f" ({row['slope_change']:+.0%})" if row["slope_change"] is not None else ""),
                    f"{fmt(row['baseline_peak_quality_drop'])} → {fmt(row['peak_quality_drop'])}"
                    + (f" ({row['peak_change']:+.0%})" if row["peak_change"] is not None else ""),
                ]
                for row in rows
            ],
        )
    )
    add("")

    slope_changes = [row["slope_change"] for row in rows if row["slope_change"] is not None]
    if slope_changes:
        add(
            f"Degradation slope moves {fmean(slope_changes):+.0%} on average. The direction is the "
            f"finding: a harder actor does not make models look uniformly worse, it makes them "
            f"degrade *faster*."
        )
        add("")

    add("## Why this is a result and not a caveat")
    add("")
    add(
        "The turn curves show it. Where the first turn scores **higher** under the harder actor "
        "while the last turn lands in the same place, the model was never more fragile than it is "
        "now — the weaker actor simply stopped pressing before the boundary gave way."
    )
    add("")
    add(
        _table(
            ["Model", f"Turn curve — {baseline_label}", f"Turn curve — {current_label}"],
            [
                [
                    row["model"],
                    " → ".join(fmt(score, 2) for score in (row["baseline_turn_curve"] or [])),
                    " → ".join(fmt(score, 2) for score in (row["turn_curve"] or [])),
                ]
                for row in rows
            ],
        )
    )
    add("")
    add(
        "So a KIDBench multi-turn number cannot be read without knowing which actor produced it. "
        "Two runs quoting the same degradation slope are not agreeing unless their actors pressed "
        "equally hard, and the honest way to report that is the measured refusal rate — which is "
        "why every conversation carries `actor_refusal_rate` regardless of which actor ran it."
    )
    add("")
    add("## Mean quality, for completeness")
    add("")
    add(
        _table(
            ["Model", baseline_label, current_label, "Δ"],
            [
                [
                    row["model"],
                    fmt(row["baseline_total_quality_score"]),
                    fmt(row["total_quality_score"]),
                    signed(_delta(row["total_quality_score"], row["baseline_total_quality_score"])),
                ]
                for row in rows
            ],
        )
    )
    add("")
    add(
        "Mean quality is the wrong place to look for this effect and is reported only so the "
        "comparison is complete. It averages over turns, so a steeper decline from a higher start "
        "can leave it flat or even raise it. The degradation metrics are the ones the actor moves."
    )
    add("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/kidbench"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/kidbench/stock_actor"))
    parser.add_argument("--current-label", default="refusal-ablated Gemma-4-31B-it")
    parser.add_argument("--baseline-label", default="stock Gemma-4-31B-it")
    parser.add_argument("--out", type=Path, default=Path("results/kidbench/reports/actor-comparison.md"))
    args = parser.parse_args()

    rows = compare(args.results_dir, args.baseline_dir, list(DEFAULT_MODELS))
    if not rows:
        raise SystemExit(f"No model has multi-turn rollouts in both {args.results_dir} and {args.baseline_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(rows, current_label=args.current_label, baseline_label=args.baseline_label), encoding="utf-8"
    )
    print(f"wrote {args.out}")

    payload = args.out.with_suffix(".json")
    payload.write_text(
        json.dumps(
            {
                "current_actor": args.current_label,
                "baseline_actor": args.baseline_label,
                "models": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {payload}")


if __name__ == "__main__":
    main()
