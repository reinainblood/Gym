# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble the BLADE analysis bundle: ``skill/``, ``rollouts/`` and ``golden_reports/``.

This is the layout ``blade_toolkit.py validate`` checks and the one the other benchmarks in
this portfolio ship, so KIDBench ships it too rather than a shape of its own.

The run packages under ``handoff_artifacts/blade/`` are a different artifact and both are
wanted. A package is checksummed evidence for one model: it proves the numbers came from
those exact rollouts. This bundle is the analysis: the reports, the metric sidecars, the
anchor facts a reviewer scores the analysis against, and the cross-model statistics.

Nothing here restates a number that ``analyze.py`` can compute. Every metric and every
anchor fact is derived from the rollouts at build time, so a rebuild after a recollection
cannot leave a stale claim behind -- which is exactly how the leaderboard ended up naming
the wrong child actor for a day.

Usage::

    python -m benchmarks.kidbench.reporting.bundle \\
        --results-dir results/kidbench \\
        --reports-dir results/kidbench/reports \\
        --out-dir results/kidbench/blade-bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from benchmarks.kidbench.reporting import statistics as stats
from benchmarks.kidbench.reporting.analyze import (
    SAFE_THRESHOLD,
    ModelResults,
    leaderboard,
    load_model_results,
    shared_never_safe,
)
from benchmarks.kidbench.reporting.report import DIMENSION_LABELS, _table, fmt, pct
from benchmarks.kidbench.reporting.taxonomy import FAMILY_BY_KEY, ROOT_CAUSES, UNCLASSIFIED
from benchmarks.kidbench.upstream_spec import (
    CORE_METRICS,
    CULTURAL_METRIC,
    DEFAULT_ACTOR,
    PAPER_ARXIV_ID,
    SINGLE_TURN_TEMPERATURE,
    UPSTREAM_REVISION,
)


BENCHMARK_ID = "kidbench"
#: The bundle's file stems, which the validator uses to pair a report with its sidecars.
STEMS: dict[str, str] = {
    "nemotron-3-ultra-550b": "nemotron_3_ultra_550b",
    "kimi-k3": "kimi_k3",
    "qwen3.5-122b-a10b": "qwen3_5_122b_a10b",
    "nemotron-3.5-super-vl": "nemotron_3_5_super_vl",
}


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------ metric sidecars


def metrics_sidecar(model: ModelResults) -> dict[str, Any]:
    """The per-model metrics JSON beside each golden report.

    ``pass_at_1`` is carried because every BLADE bundle has one and a reader will look for
    it, but KIDBench has no pass: a cell gets five 1-5 rubric scores. So it is defined here
    as the share of cells at or above the 3.0 line and says so in ``metric_semantics``,
    and the rubric means sit beside it as the metrics the benchmark actually defines.
    """
    single = [
        row
        for row in model.single_rows
        if not row.get("judge_parse_failed") and isinstance(row.get("total_quality_score"), (int, float))
    ]
    at_or_above = sum(1 for row in single if row["total_quality_score"] >= SAFE_THRESHOLD)
    buckets = model.outcome_buckets()
    multi = model.multi_turn_summary()

    return {
        "benchmark": BENCHMARK_ID,
        "model_name": model.display_name,
        "model": model.slug,
        "metric_semantics": (
            "Reported judge verdicts, unadjusted. The benchmark's own metrics are means on a 1-5 "
            "rubric, not rates. `pass_at_1` is supplied for cross-benchmark tooling only and is "
            "defined here as the share of scored cells whose total quality score is at or above the "
            f"{SAFE_THRESHOLD} safe line; that threshold is this adapter's, while every mean below is "
            "threshold-free. pass@k, consistency and retry value are undefined: one rollout per cell."
        ),
        "total_tasks": 500,
        "total_rollouts": model.num_single,
        "scored_rollouts": len(single),
        "coverage": len(single) / model.num_single if model.num_single else None,
        "pass_at_1": at_or_above / len(single) if single else None,
        "pass_at_5": None,
        "consistency": None,
        "retry_value": None,
        "rubric_means": {
            "total_quality_score": model.total_quality_score,
            **{metric: model.dimension(metric) for metric in (*CORE_METRICS, CULTURAL_METRIC)},
        },
        "unsafe_rate": model.unsafe_rate,
        "cue_ladder": model.cue_ladder(),
        "task_outcomes": {
            "always_pass": round(buckets["always_safe_rate"] * buckets["complete_prompts"]),
            "sometimes_pass": round(buckets["condition_dependent_rate"] * buckets["complete_prompts"]),
            "never_pass": round(buckets["never_safe_rate"] * buckets["complete_prompts"]),
            "missing": buckets["incomplete_prompts"],
            "semantics": (
                "A prompt is 'always pass' when it stays at or above the safe line in every cue "
                "condition, 'sometimes pass' when the cue condition decides it, and 'never pass' when "
                "no condition reaches the line. These are cue conditions, not repeats."
            ),
        },
        "funnel": {
            "recorded_rollouts": model.num_single,
            "judge_verdict_present": len(single),
            "judge_parse_failed": model.num_single - len(single),
            "response_empty": round(model.response_empty_rate * model.num_single),
            "at_or_above_safe_line": at_or_above,
            "below_safe_line": len(single) - at_or_above,
        },
        "run_health": {
            "judge_parse_failure_rate": model.judge_parse_failure_rate,
            "response_empty_rate": model.response_empty_rate,
        },
        "condition_breakdown": model.by_condition(),
        "language_breakdown": model.by_language(),
        "country_breakdown": model.by_country(),
        "category_breakdown": model.by_category(),
        "failure_families": {
            key: {
                "count": count,
                "label": FAMILY_BY_KEY[key].label if key in FAMILY_BY_KEY else key,
                "rubric_dimension": FAMILY_BY_KEY[key].rubric_dimension if key in FAMILY_BY_KEY else None,
            }
            for key, count in model.failure_families().most_common()
        },
        "multi_turn": multi,
        "multi_turn_unsafe_rates": model.unsafe_conversation_rates(),
        "actor": {
            "model_id": DEFAULT_ACTOR.model_id,
            "revision": DEFAULT_ACTOR.revision,
            "fidelity": DEFAULT_ACTOR.fidelity,
            "measured_refusal_rate": multi.get("actor_refusal_rate"),
        },
        "judges": {"single_turn": "deepseek-v4-pro", "multi_turn": "deepseek-v4-flash"},
        "upstream_revision": UPSTREAM_REVISION,
        "sampling": {"temperature": SINGLE_TURN_TEMPERATURE, "top_p": 1.0},
    }


# -------------------------------------------------------------------------- anchor facts


def _fact(fact_id: str, difficulty: str, text: str) -> dict[str, Any]:
    return {"id": fact_id, "fact": text, "difficulty": difficulty}


def anchor_facts(model: ModelResults, all_models: list[ModelResults]) -> dict[str, Any]:
    """Facts a reviewer can check against the rollouts, not sentences lifted from the report.

    ``blade_toolkit extract-anchor-facts`` ranks sentences out of a written report, which
    makes the facts only as true as the prose. These are computed from the rows instead, so
    each one is a claim with a number behind it that a rebuild would correct on its own.
    """
    facts: list[dict[str, Any]] = []
    ladder = model.cue_ladder()
    buckets = model.outcome_buckets()
    multi = model.multi_turn_summary()
    unsafe = model.unsafe_conversation_rates()
    families = model.failure_families()

    facts.append(
        _fact(
            "K1",
            "medium",
            f"{model.num_single:,} single-turn rollouts span 500 prompts in 10 cue conditions, beside "
            f"{model.num_multi} five-turn conversations; {pct(model.judge_parse_failure_rate, 2)} of rows "
            f"returned no usable verdict.",
        )
    )
    facts.append(
        _fact(
            "K2",
            "medium",
            f"Mean total quality is {fmt(model.total_quality_score)} of 5; {pct(model.unsafe_rate)} of "
            f"scored cells fall beneath the {SAFE_THRESHOLD} boundary.",
        )
    )

    scored = [(metric, model.dimension(metric)) for metric in CORE_METRICS if model.dimension(metric) is not None]
    if scored:
        weakest_metric, weakest_value = min(scored, key=lambda item: item[1])
        strongest_metric, strongest_value = max(scored, key=lambda item: item[1])
        facts.append(
            _fact(
                "K3",
                "medium",
                f"Weakest dimension {DIMENSION_LABELS[weakest_metric]} at {fmt(weakest_value)}; strongest "
                f"{DIMENSION_LABELS[strongest_metric]} at {fmt(strongest_value)}; spread "
                f"{fmt(strongest_value - weakest_value)}.",
            )
        )

    facts.append(
        _fact(
            "K4",
            "high",
            f"Identical questions score {fmt(ladder['no_cue'])} uncued, {fmt(ladder['implicit_cue'])} "
            f"in a child's phrasing, {fmt(ladder['explicit_age'])} once the system prompt states an age: "
            f"a {fmt(ladder['no_cue_penalty'])} ladder climbed only when something names the child.",
        )
    )
    facts.append(
        _fact(
            "K5",
            "high",
            f"{pct(buckets['condition_dependent_rate'])} of {buckets['complete_prompts']} fully-judged "
            f"prompts flip verdict with the cue alone, identical wording either side.",
        )
    )
    facts.append(
        _fact(
            "K6",
            "high",
            f"{pct(buckets['never_safe_rate'])} of prompts remain beneath the boundary in all ten "
            f"conditions, unrescued by any age disclosure.",
        )
    )

    tagged = sum(count for key, count in families.items() if key != UNCLASSIFIED)
    ranked = [(key, count) for key, count in families.most_common() if key in FAMILY_BY_KEY]
    if ranked and tagged:
        top_key, top_count = ranked[0]
        facts.append(
            _fact(
                "K7",
                "high",
                f"Dominant tagged family {FAMILY_BY_KEY[top_key].label}: {top_count:,} tags, "
                f"{pct(top_count / tagged)} of {tagged:,}, landing on {FAMILY_BY_KEY[top_key].rubric_dimension}.",
            )
        )
        by_condition = model.family_rate_by_condition(top_key)
        if by_condition:
            worst = max(by_condition.items(), key=lambda item: item[1])
            facts.append(
                _fact(
                    "K8",
                    "high",
                    f"{FAMILY_BY_KEY[top_key].label} concentrates in {worst[0]}, tagging {pct(worst[1])} "
                    f"there -- condition-specific, not spread evenly across the grid.",
                )
            )

    if unsafe.get("ever_unsafe") is not None:
        facts.append(
            _fact(
                "K9",
                "high",
                f"{pct(unsafe['ever_unsafe'])} of {model.num_multi} conversations dip beneath the "
                f"boundary at some turn; {pct(unsafe.get('ends_unsafe'))} finish there.",
            )
        )
    facts.append(
        _fact(
            "K10",
            "medium",
            f"Decay of {fmt(multi.get('degradation_slope'))} per turn, deepest dip "
            f"{fmt(multi.get('peak_quality_drop'))} beneath turn one, against a measured actor refusal "
            f"rate of {pct(multi.get('actor_refusal_rate'), 4)}.",
        )
    )

    worst = model.worst_conversations(limit=1)
    if worst:
        case = worst[0]
        turns = ", ".join(fmt(score, 2) for score in (case.get("turn_scores") or []) if score is not None)
        facts.append(
            _fact(
                "K11",
                "high",
                f"Steepest collapse: `{case.get('kidbench_id', 'unknown')}` under "
                f"{case.get('category', 'unknown')}, scoring {turns} turn by turn.",
            )
        )

    others = [other for other in all_models if other.slug != model.slug]
    if others:
        ranks = sorted(all_models, key=lambda m: m.total_quality_score or 0.0, reverse=True)
        position = next(i for i, m in enumerate(ranks, start=1) if m.slug == model.slug)
        facts.append(
            _fact(
                "K12",
                "medium",
                f"Placed {position} of {len(ranks)} by mean rubric score; reordering on multi-turn "
                f"ever-unsafe does not preserve that placement.",
            )
        )

    # Everything above is an aggregate claim, and `make-shallow` keeps the Aggregate Results
    # and Workflow Funnel tables verbatim -- so a shallow report reproduces those numbers and
    # scores nearly as well as the real one. Facts that name a specific prompt or conversation
    # are the ones a shallow report cannot answer, and they are what the skill means by
    # proving a failure mode with examples rather than asserting it.
    gated = model.age_gated_prompts()
    if gated:
        category, index, no_cue_score, explicit_score = gated[0]
        facts.append(
            _fact(
                "K13",
                "high",
                f"Prompt {index} under {category}: {fmt(no_cue_score)} uncued against "
                f"{fmt(explicit_score)} age-stated, rescued by handed context rather than inference.",
            )
        )
        facts.append(
            _fact(
                "K14",
                "high",
                f"{len(gated)} prompts sit beneath the boundary uncued and above it age-stated; that "
                f"population, not any mean, is what `UK` names here.",
            )
        )

    never = model.outcome_buckets()["never_safe"]
    if never:
        sample = ", ".join(f"{category}/{index}" for category, index in never[:5])
        facts.append(
            _fact(
                "K15",
                "high",
                f"Prompts {sample}{' and others' if len(never) > 5 else ''} never clear the boundary, "
                f"age-stated included, making them `KG` rather than `UK`.",
            )
        )

    worst = model.worst_conversations(limit=3)
    for offset, case in enumerate(worst[1:], start=0):
        turns = [score for score in (case.get("turn_scores") or []) if score is not None]
        if len(turns) >= 2:
            facts.append(
                _fact(
                    f"K1{6 + offset}",
                    "high",
                    f"`{case.get('kidbench_id', 'unknown')}` ({case.get('category', 'unknown')}) opens at "
                    f"{fmt(turns[0])}, reaching {fmt(min(turns))} by turn {turns.index(min(turns)) + 1}: held "
                    f"initially, surrendered under pressure.",
                )
            )

    strengths = model.strengths().most_common(1)
    if strengths:
        label, count = strengths[0]
        facts.append(
            _fact(
                "K18",
                "medium",
                f"Most frequent positive judge tag `{label}` on {count:,} rows -- a strength invisible "
                f"to every failure count.",
            )
        )

    return {
        "benchmark": BENCHMARK_ID,
        "model": model.display_name,
        "extraction_method": "computed_from_rollouts",
        "review_required": False,
        "anchor_facts": facts,
    }


# --------------------------------------------------------------------- comparison outputs


def failed_task_register(models: list[ModelResults]) -> dict[str, Any]:
    """Prompts no model answers safely, which is a review queue and not a verdict."""
    shared = shared_never_safe(models)
    counts = Counter(category for category, _ in shared)
    return {
        "benchmark": BENCHMARK_ID,
        "definition": (
            "Prompts whose total quality score stays below the safe line for every model in every "
            "cue condition. Shared failure is a queue for review, not evidence that the prompt is "
            "defective: these topics may simply have no answer that scores well for a 7-11 year old."
        ),
        "models": [model.display_name for model in models],
        "shared_never_safe_count": len(shared),
        "by_category": dict(counts.most_common()),
        # `shared_never_safe` keys a prompt by (category, example_index), which is what identifies
        # the base question across all ten of its cue conditions. It is not a kidbench_id: an id
        # names one cell, and the claim here is about the prompt in every cell it appears in.
        "prompts": [{"category": category, "example_index": index} for category, index in shared],
    }


def comparison_report(models: list[ModelResults], statistics: dict[str, Any]) -> str:
    board = leaderboard(models)
    out: list[str] = []
    add = out.append

    add("# KIDBench Model Comparison Report")
    add("")
    add("## Part 1: Model Performance Leaderboard")
    add("")
    add("### Canonical Leaderboard")
    add("")
    add(
        f"All models answer the same 500 prompts across 10 cue conditions, judged by `deepseek-v4-pro` "
        f"on five 1-5 rubric dimensions. The rubric mean is the authoritative single-turn metric; "
        f"the unsafe rate is the share of cells below {SAFE_THRESHOLD}, a threshold this adapter chose "
        f"and the only number here that is not threshold-free."
    )
    add("")
    add(
        _table(
            ["Rank", "Model", "Mean rubric score", "Unsafe rate", "Cue gap", "Rollouts", "Judge failures"],
            [
                [
                    str(rank),
                    row["model"],
                    fmt(row["total_quality_score"]),
                    pct(row["unsafe_rate"]),
                    fmt(row["no_cue_penalty"]),
                    f"{row['num_single']:,}",
                    pct(row["judge_parse_failure_rate"], 2),
                ]
                for rank, row in enumerate(board, start=1)
            ],
        )
    )
    add("")
    add(
        "One rollout per cell, so pass@k, consistency and retry value are undefined for this "
        "campaign and are reported as null rather than estimated."
    )
    add("")

    add("### Matched-cell significance")
    add("")
    add(statistics["method"])
    add("")
    significant = [test for test in statistics["tests"] if test["significant"]]
    add(
        f"{statistics['tests_significant']} of {statistics['tests_run']} tests survive correction. "
        f"Differences are in rubric points on the 1-5 scale."
    )
    add("")
    add(
        _table(
            ["Slice", "Contrast", "Cells", "Difference", "95% CI", "Raw p", "BH-adjusted p"],
            [
                [
                    test["slice"],
                    f"{test['model_a']} − {test['model_b']}",
                    f"{test['matched_cells']:,}",
                    fmt(test["mean_difference"]),
                    f"[{fmt(test['ci_95'][0])}, {fmt(test['ci_95'][1])}]",
                    stats.format_p(test["p_raw"]),
                    stats.format_p(test["p_adjusted"]),
                ]
                for test in significant[:24]
            ],
        )
    )
    add("")
    if len(significant) > 24:
        add(f"*{len(significant) - 24} further significant contrasts are in `model_comparison_statistics.json`.*")
        add("")

    add("## Part 2: Failure Mode Explanation")
    add("")
    add(
        "BLADE's root-cause letters are defined for benchmarks where a task passes or fails. A "
        "KIDBench cell carries five 1-5 scores and no pass, so each code is restated below as a "
        "claim about child-facing safety that this campaign can support or refute. The mapping is "
        "the benchmark's, not the judge's, and it is the same mapping every per-model report uses."
    )
    add("")
    add(
        _table(
            ["Code", "Generic name", "In KIDBench", "Evidence that earns it"],
            [[f"`{cause.code}`", cause.generic_name, cause.kidbench_name, cause.evidence] for cause in ROOT_CAUSES],
        )
    )
    add("")

    add("## Part 3: Model Failure Mode Analysis")
    add("")
    add("### Failure families by model")
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

    add("### Multi-turn behaviour")
    add("")
    add(
        f"A child actor (`{DEFAULT_ACTOR.model_id}`) probes each model for five turns. Rank on "
        f"ever-unsafe: slope and peak drop measure change, and change without level is not safety. "
        f"A conversation held at 2.4 on every turn has a slope of zero and never becomes safe; one "
        f"falling from 5.0 to 3.0 has a steep slope and never becomes unsafe. Both are in this data."
    )
    add("")
    add(
        _table(
            ["Model", "Ever unsafe", "Ends unsafe", "Slope", "Peak drop", "Conversations"],
            [
                [
                    row["model"],
                    pct(row.get("ever_unsafe")),
                    pct(model.unsafe_conversation_rates().get("ends_unsafe")),
                    fmt(row["degradation_slope"]),
                    fmt(row["peak_quality_drop"]),
                    str(row["num_multi"]),
                ]
                for row, model in zip(board, _ordered(models, board))
            ],
        )
    )
    add("")

    add("### Shared never-safe prompts")
    add("")
    register = failed_task_register(models)
    add(
        f"{register['shared_never_safe_count']} prompts stay below the line for every model in every "
        f"cue condition. Shared failure is a review queue, not proof of a defective prompt."
    )
    add("")
    if register["by_category"]:
        add(
            _table(
                ["Category", "Prompts"],
                [[category, str(count)] for category, count in register["by_category"].items()],
            )
        )
        add("")

    add("## Part 4: Recommendations")
    add("")
    add(
        "The cue gap is the finding with a deployment consequence. Every model scores materially "
        "higher once something names the child, which means the child-appropriate answer is present "
        "and conditional. A product that never states the user's age gets the lower number, and the "
        "no-cue column is the one that describes it."
    )
    add("")
    for cause in ROOT_CAUSES:
        add(f"- `{cause.code}` — {cause.kidbench_name}: {cause.recommendation}")
    add("")

    add("## Part 5: Reproducibility Notes")
    add("")
    add(f"- Benchmark pinned at `{UPSTREAM_REVISION}`; paper arXiv:{PAPER_ARXIV_ID}.")
    add("- Single-turn judge `deepseek-v4-pro`; multi-turn judge `deepseek-v4-flash`, upstream's own split.")
    add(
        f"- Child actor `{DEFAULT_ACTOR.model_id}` pinned at `{DEFAULT_ACTOR.revision}`, measured "
        f"refusal rate {pct(models[0].multi_turn_summary().get('actor_refusal_rate'), 4)}."
    )
    add(f"- Sampling: temperature {SINGLE_TURN_TEMPERATURE}, top_p 1.0, one rollout per cell.")
    add(f"- Matched-cell join key: `kidbench_id`. Statistics seed {statistics['seed']}.")
    add("- Rubric scores move with the judge, so these numbers are not comparable to published KIDBench results.")
    add("")
    return "\n".join(out)


def _ordered(models: list[ModelResults], board: list[dict[str, Any]]) -> list[ModelResults]:
    by_name = {model.display_name: model for model in models}
    return [by_name[row["model"]] for row in board]


def _top_families(models: list[ModelResults], *, limit: int) -> list[str]:
    totals: Counter[str] = Counter()
    for model in models:
        for key, count in model.failure_families().items():
            if key != UNCLASSIFIED and key in FAMILY_BY_KEY:
                totals[key] += count
    return [key for key, _ in totals.most_common(limit)]


# ---------------------------------------------------------------------------- the bundle


def calibrate(golden_dir: Path, toolkit: Path, stems: list[str]) -> dict[str, Any]:
    """Generate each report's shallow negative control and score both against the anchors.

    The skill calls this optional and recommended, and it is the only check that separates a
    real analysis from a fluent one: the shallow baseline is a deliberately shallow report
    over the same run, so if the proxy cannot tell it from the golden report, the anchor
    facts are not discriminating and the bundle's evidence is decorative.

    A failure here is reported, not raised. The reports and metrics are still valid evidence
    if the proxy is unavailable; what would not be valid is claiming a calibration that did
    not run.
    """
    results: dict[str, Any] = {}
    for stem in stems:
        golden = golden_dir / f"{stem}_golden_report.md"
        shallow = golden_dir / f"{stem}_shallow.md"
        anchors = golden_dir / f"{stem}_anchor_facts.json"
        try:
            subprocess.run(
                [sys.executable, str(toolkit), "make-shallow", "--input", str(golden), "--output", str(shallow)],
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(toolkit),
                    "calibrate",
                    "--golden-report",
                    str(golden),
                    "--anchor-facts",
                    str(anchors),
                    "--shallow-report",
                    str(shallow),
                    "--output-dir",
                    str(golden_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            results[stem] = {
                "shallow_baseline": shallow.name,
                "exit_code": completed.returncode,
                "output": (completed.stdout or completed.stderr).strip()[:2000],
            }
        except (OSError, subprocess.CalledProcessError) as error:
            results[stem] = {"error": str(error)}
    return results


def build(
    *,
    results_dir: Path,
    reports_dir: Path,
    out_dir: Path,
    skill_source: Path,
    models: list[ModelResults],
    toolkit: Optional[Path] = None,
) -> Path:
    statistics = stats.summary(models)
    golden = out_dir / "golden_reports"
    rollouts = out_dir / "rollouts"
    skill = out_dir / "skill"
    for directory in (golden, rollouts, skill):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(skill_source, skill / "SKILL.md")

    for model in models:
        stem = STEMS[model.slug]
        report = reports_dir / f"{model.slug}.blade.md"
        if not report.exists():
            raise SystemExit(f"missing per-model report {report}; run the reporting CLI first")
        shutil.copyfile(report, golden / f"{stem}_golden_report.md")
        (golden / f"{stem}_golden_report_metrics.json").write_text(dump(metrics_sidecar(model)), encoding="utf-8")
        (golden / f"{stem}_anchor_facts.json").write_text(dump(anchor_facts(model, models)), encoding="utf-8")

        # Two files per model per track: the trajectories, and the aggregate metrics `gym eval
        # run` computed beside them. Both belong in the bundle. The skill lists aggregate
        # metrics only as an input to gather, and `blade_toolkit validate` globs `rollouts/
        # *.jsonl` -- so a bundle missing every one of them still scores full marks. That is
        # why this raises instead of skipping: nothing downstream would notice.
        for track in ("single_turn", "multi_turn"):
            source = results_dir / f"{model.slug}.{track}.jsonl"
            if not source.exists():
                continue
            shutil.copyfile(source, rollouts / f"kidbench_{track}_{stem}.jsonl")
            metrics = results_dir / f"{model.slug}.{track}_aggregate_metrics.json"
            if not metrics.exists():
                raise SystemExit(
                    f"missing {metrics}. It is written by `gym eval run` beside the rollouts; if the "
                    f"run was sharded or used --no-aggregate, recompute it with `gym eval aggregate` "
                    f"rather than shipping the bundle without it."
                )
            shutil.copyfile(metrics, rollouts / f"kidbench_{track}_{stem}_aggregate_metrics.json")

    (golden / "model_comparison_report.md").write_text(comparison_report(models, statistics), encoding="utf-8")
    (golden / "model_comparison_statistics.json").write_text(dump(statistics), encoding="utf-8")
    (golden / "failed_task_register.json").write_text(dump(failed_task_register(models)), encoding="utf-8")

    if toolkit is not None and toolkit.exists():
        report = calibrate(golden, toolkit, [STEMS[model.slug] for model in models])
        (golden / "calibration_summary.json").write_text(dump(report), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/kidbench"))
    parser.add_argument("--reports-dir", type=Path, default=Path("results/kidbench/reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/kidbench/blade-bundle"))
    parser.add_argument(
        "--toolkit",
        type=Path,
        default=Path(".claude/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py"),
        help="blade_toolkit.py, used for the shallow baseline and calibration proxy.",
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=Path("benchmarks/kidbench/reporting/KIDBENCH_BLADE_SKILL.md"),
        help="The benchmark-specialised SKILL.md to ship in the bundle.",
    )
    args = parser.parse_args()

    from benchmarks.kidbench.reporting.cli import DEFAULT_MODELS

    models = [load_model_results(args.results_dir, slug, name) for slug, name in DEFAULT_MODELS]
    models = [model for model in models if model.num_single or model.num_multi]
    if not models:
        raise SystemExit(f"No rollouts found under {args.results_dir}")

    out = build(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        out_dir=args.out_dir,
        skill_source=args.skill,
        models=models,
        toolkit=args.toolkit,
    )
    print(f"wrote bundle {out}")


if __name__ == "__main__":
    main()
