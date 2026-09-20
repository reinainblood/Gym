# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the KIDBench BLADE reports and the cross-model comparison page.

Usage::

    python -m benchmarks.kidbench.reporting.cli \
        --results-dir results/kidbench \
        --out-dir results/kidbench/reports
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.kidbench.reporting.analyze import (
    SAFE_THRESHOLD,
    ModelResults,
    leaderboard,
    load_model_results,
    shared_never_safe,
    validate_scale,
)
from benchmarks.kidbench.reporting.report import (
    DIMENSION_LABELS,
    _table,
    fmt,
    model_summary_json,
    pct,
    signed,
    write_model_report,
)
from benchmarks.kidbench.reporting.taxonomy import FAMILY_BY_KEY, ROOT_CAUSES, UNCLASSIFIED
from benchmarks.kidbench.upstream_spec import (
    CORE_METRICS,
    CULTURAL_METRIC,
    PAPER_ARXIV_ID,
    UPSTREAM_REVISION,
)


#: slug -> display name. Ordered as the run script runs them.
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("nemotron-3-ultra-550b", "Nemotron 3 Ultra 550B"),
    ("kimi-k3", "Kimi K3"),
    ("qwen3.5-122b-a10b", "Qwen3.5 122B-A10B"),
    ("nemotron-3.5-super-vl", "Nemotron 3.5 Super VL"),
)


def render_comparison(models: list[ModelResults]) -> str:
    board = leaderboard(models)
    out: list[str] = []
    add = out.append

    add("# KIDBench Leaderboard — Child-Facing Safety, Ages 7–11")
    add("")
    add(
        f"[KIDBench](https://arxiv.org/abs/{PAPER_ARXIV_ID}) scores what a model says to a child "
        f"aged 7–11 on five 1–5 dimensions. It is not a harmful-content benchmark: the failure it "
        f"was built around is the *medically accurate* answer that is still the wrong thing to tell "
        f"a nine-year-old. Judge: `deepseek-v4-pro`, the paper's own. Benchmark pinned at "
        f"`{UPSTREAM_REVISION[:12]}`."
    )
    add("")

    # -- Leaderboard ----------------------------------------------------------
    add("## Overall")
    add("")
    add(
        _table(
            ["#", "Model", "Overall /5", "Unsafe rate", "Weakest dimension"],
            [
                [
                    str(rank),
                    row["model"],
                    fmt(row["total_quality_score"]),
                    pct(row["unsafe_rate"]),
                    _weakest(row),
                ]
                for rank, row in enumerate(board, start=1)
            ],
        )
    )
    add("")
    add(
        f"*Unsafe rate is the share of answers scoring below {SAFE_THRESHOLD:.1f}, the rubric's own "
        f"boundary between “partially appropriate” and “potentially unsafe”. The threshold is this "
        f"adapter's; every mean score above is threshold-free.*"
    )
    add("")

    # -- The headline finding -------------------------------------------------
    add("## Does the model need to be told a child is asking?")
    add("")
    add(
        "The same 500 questions, three ways: asked plainly, asked in a child's own words, and asked "
        "with the system prompt stating the age. A large gap means the model **has** the "
        "child-appropriate answer and only reaches for it when something names the child — which a "
        "real deployment may never do."
    )
    add("")
    add(
        _table(
            ["Model", "No cue", "Implicit cue", "Explicit age", "Cue gap", "Condition-dependent prompts"],
            [
                [
                    row["model"],
                    fmt(row["no_cue"]),
                    fmt(row["implicit_cue"]),
                    fmt(row["explicit_age"]),
                    signed(row["no_cue_penalty"]),
                    pct(row["condition_dependent_rate"]),
                ]
                for row in board
            ],
        )
    )
    add("")

    # -- Dimensions -----------------------------------------------------------
    add("## Rubric dimensions")
    add("")
    add(
        _table(
            ["Model", *[DIMENSION_LABELS[metric] for metric in (*CORE_METRICS, CULTURAL_METRIC)]],
            [[row["model"], *[fmt(row[metric]) for metric in (*CORE_METRICS, CULTURAL_METRIC)]] for row in board],
        )
    )
    add("")

    # -- Multi-turn -----------------------------------------------------------
    add("## Holding the line over five turns")
    add("")
    add(
        "A child actor probes each model for five turns. Degradation slope is quality lost per turn "
        "(higher is worse); peak drop is the largest fall below the model's own turn-1 score — the "
        "number that distinguishes *never had the boundary* from *had it and gave it up*."
    )
    add("")
    add(
        _table(
            ["Model", "Unsafe at any turn", "Degradation slope", "Peak drop", "Conversations"],
            [
                [
                    row["model"],
                    pct(row.get("ever_unsafe")),
                    fmt(row["degradation_slope"]),
                    fmt(row["peak_quality_drop"]),
                    str(row["num_multi"]),
                ]
                for row in board
            ],
        )
    )
    add("")
    add(
        "Rank on the first column. Slope and peak drop measure *change*, which says nothing about "
        "safety without the level — a conversation held at 2.4 for all five turns scores a slope of "
        "zero and reads as stable, while one falling from 5.0 to 3.0 never becomes unsafe at all. "
        "Both patterns are in this data."
    )
    add("")

    # -- Failure modes --------------------------------------------------------
    add("## Where each model actually fails")
    add("")
    add(
        "Failure families are built from the judge's own tags, grouped by meaning. Percentages are "
        "each family's share of that model's total tagged failures."
    )
    add("")
    family_keys = _top_families(models, limit=6)
    add(
        _table(
            ["Model", *[FAMILY_BY_KEY[key].label for key in family_keys]],
            [
                [
                    model.display_name,
                    *[
                        pct(model.failure_families()[key] / max(sum(model.failure_families().values()), 1))
                        for key in family_keys
                    ],
                ]
                for model in models
            ],
        )
    )
    add("")
    for key in family_keys:
        add(f"- **{FAMILY_BY_KEY[key].label}** — {FAMILY_BY_KEY[key].harm_model}")
    add("")

    # -- BLADE codes ----------------------------------------------------------
    add("## What the BLADE codes mean here")
    add("")
    add(
        "BLADE's root-cause letters are defined for benchmarks where a task passes or fails. A "
        "KIDBench row has five 1–5 scores and no pass/fail, so each code is restated below as a "
        "claim about child-facing safety that this run can support or refute."
    )
    add("")
    add(
        _table(
            ["Code", "Generic name", "In KIDBench", "Evidence that earns it"],
            [
                [f"`{cause.code}`", cause.generic_name, f"**{cause.kidbench_name}**", cause.evidence]
                for cause in ROOT_CAUSES
            ],
        )
    )
    add("")

    shared = shared_never_safe(models)
    if shared:
        counts = Counter(category for category, _ in shared)
        add(f"### {len(shared)} questions no model answers safely")
        add("")
        add(
            "These stay below the line for every model in every cue condition. A question the whole "
            "field fails says more about the benchmark's hard edge — or about the topic — than about "
            "any single model."
        )
        add("")
        add(
            _table(
                ["Category", "Prompts"],
                [[category, str(count)] for category, count in counts.most_common()],
            )
        )
        add("")

    add("## Reproducibility")
    add("")
    add(
        _table(
            ["Model", "Single-turn rows", "Multi-turn rows", "Judge failures"],
            [
                [
                    row["model"],
                    f"{row['num_single']:,}",
                    f"{row['num_multi']:,}",
                    pct(row["judge_parse_failure_rate"], 2),
                ]
                for row in board
            ],
        )
    )
    add("")
    add(
        "Two deviations from the published protocol apply to every model equally, so cross-model "
        "comparison is unaffected: the multi-turn child actor is stock `google/gemma-4-31B-it` rather "
        "than the paper's unreleased refusal-ablated checkpoint (which makes every model look "
        "slightly safer in multi-turn), and generation runs at `temperature=0.7` per upstream's "
        "runner scripts rather than the `0` its README states. Per-model reports carry the detail."
    )
    add("")
    return "\n".join(out)


def _weakest(row: dict[str, Any]) -> str:
    scored = [(metric, row[metric]) for metric in CORE_METRICS if row.get(metric) is not None]
    if not scored:
        return "—"
    metric, value = min(scored, key=lambda item: item[1])
    return f"{DIMENSION_LABELS[metric]} ({fmt(value)})"


def _top_families(models: list[ModelResults], *, limit: int) -> list[str]:
    totals: Counter[str] = Counter()
    for model in models:
        for key, count in model.failure_families().items():
            if key != UNCLASSIFIED and key in FAMILY_BY_KEY:
                totals[key] += count
    return [key for key, _ in totals.most_common(limit)]


def comparison_json(models: list[ModelResults]) -> dict[str, Any]:
    return {
        "benchmark": "KIDBench",
        "paper": f"arXiv:{PAPER_ARXIV_ID}",
        "upstream_revision": UPSTREAM_REVISION,
        "safe_threshold": SAFE_THRESHOLD,
        "leaderboard": leaderboard(models),
        "models": {model.slug: model_summary_json(model) for model in models},
        "blade_root_causes": [cause._asdict() for cause in ROOT_CAUSES],
        "failure_families": {
            key: {
                "label": family.label,
                "harm_model": family.harm_model,
                "rubric_dimension": family.rubric_dimension,
            }
            for key, family in FAMILY_BY_KEY.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/kidbench"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/kidbench/reports"))
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="SLUG=Display Name",
        help="Override the model set; repeatable.",
    )
    parser.add_argument("--no-html", action="store_true", help="Skip the standalone HTML page.")
    args = parser.parse_args()

    specs = [tuple(spec.split("=", 1)) for spec in args.model] if args.model else list(DEFAULT_MODELS)
    models = [load_model_results(args.results_dir, slug, name) for slug, name in specs]
    models = [model for model in models if model.num_single or model.num_multi]
    if not models:
        raise SystemExit(f"No rollouts found under {args.results_dir}")

    for problem in validate_scale(models):
        print(f"WARNING: {problem}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        path = write_model_report(model, models, args.out_dir)
        print(f"wrote {path}")

    comparison = args.out_dir / "kidbench-leaderboard.md"
    comparison.write_text(render_comparison(models), encoding="utf-8")
    print(f"wrote {comparison}")

    data = comparison_json(models)

    payload = args.out_dir / "kidbench-leaderboard.json"
    payload.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {payload}")

    if not args.no_html:
        from benchmarks.kidbench.reporting.html_report import render

        page = args.out_dir / "kidbench-leaderboard.html"
        page.write_text(render(data), encoding="utf-8")
        print(f"wrote {page}")


if __name__ == "__main__":
    main()
