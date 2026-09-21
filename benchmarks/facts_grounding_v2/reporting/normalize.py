# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize a FACTS Grounding v2 ``gym eval run`` directory into the shared ``NormalizedRun`` contract."""

from __future__ import annotations

import collections
from typing import Any

from benchmarks.facts_grounding_v2 import prepare as prepare_module

from .common import RunArtifacts, excerpt, failure_summaries, reconcile
from .schema import (
    AnchorFact,
    BladeMapping,
    CalibrationSummary,
    MetricValue,
    NormalizedRun,
    ReferenceComparison,
    RolloutSummary,
)


JUDGES = ("gemini-2.5-flash", "gpt-5")
BENCHMARK = {
    "id": "facts_grounding_v2",
    "display_name": "FACTS Grounding v2 (public set)",
    "protocol": "long-form grounded generation, 856 public prompts, eligibility + sentence-level grounding judged by Gemini 2.5 Flash and GPT-5",
    "unit_of_analysis": "one prompt answered once; eligibility settled by the first judge without major issues, grounding averaged over both judges",
    "dataset": {
        "source": prepare_module.KAGGLE_DATASET_URL,
        "revision": f"version {prepare_module.KAGGLE_DATASET_VERSION}",
        "file": prepare_module.CSV_MEMBER,
        "sha256": prepare_module.CSV_SHA256,
        "count": prepare_module.EXPECTED_ROWS,
        "license": prepare_module.LICENSE,
        "cohort": "the 856-prompt public set of FACTS Grounding (Kaggle release v17); 859 private prompts are held by Kaggle; the 2024 Hugging Face mirror carries 4 extra rows",
    },
    "paper": {
        "citation": "Cheng et al., The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality, arXiv:2512.10791v1 (2025), section 6; prompts from Jacovi et al., arXiv:2501.03200",
        "url": "https://arxiv.org/abs/2512.10791v1",
        "sha256": "db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be",
    },
    "upstream": {
        "reference_implementation": "Kaggle starter notebook prathameshbang/facts-grounding-v2-benchmark-starter (v4)",
        "judges": "google/gemini-2.5-flash then openai/gpt-5-2025-08-07",
    },
}
REFERENCES = [
    ReferenceComparison(
        label="Paper Table 1, FACTS Grounding v2 top score",
        value="Gemini 3 Pro leads the suite; per-subset grounding scores are reported with 95% CIs over public+private",
        source="arXiv:2512.10791v1 Table 1",
        comparability="paper numbers pool the public and private halves and use Kaggle's pipeline; this run scores the public half with the same judges",
    ),
]


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("_ng_task_index"))


def receipts_for(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    task = _task_id(rollout)
    rollout_id = f"{rollout.get('_ng_task_index', 0)}-{rollout.get('_ng_rollout_index', 0)}"
    out = []
    for index, receipt in enumerate(rollout.get("judge_receipts", [])):
        out.append(
            {
                "receipt_id": f"{task}/{receipt['stage']}-{receipt['judge']}",
                "task_id": task,
                "rollout_id": rollout_id,
                "index": index,
                **receipt,
            }
        )
    return out


def _m(
    metric_id,
    name,
    value,
    *,
    numerator=None,
    denominator=None,
    unit="rate",
    direction="higher_is_better",
    kind="component",
    definition="",
    slice_=None,
    ci95=None,
) -> MetricValue:
    return MetricValue(
        id=metric_id,
        name=name,
        value=value,
        numerator=numerator,
        denominator=denominator,
        unit=unit,
        direction=direction,
        kind=kind,
        definition=definition,
        slice=slice_,
        ci95=ci95,
    )


def normalize(
    artifacts: RunArtifacts, *, run_info: dict[str, Any], calibration: CalibrationSummary | None, num_repeats: int = 1
) -> NormalizedRun:
    outcomes, scored, unresolved = reconcile(artifacts, task_id_of=_task_id, num_repeats=num_repeats)
    rows = [scored[key] for key in sorted(scored)]
    agg = artifacts.aggregate
    n = len(rows)
    eligible_rows = [row for row in rows if row.get("eligible")]
    n_eligible = len(eligible_rows)
    score_sum = sum(float(row.get("reward", 0.0)) for row in rows)
    graded = [row for row in rows if row.get("unadjusted_score") is not None]
    unadjusted_sum = sum(float(row["unadjusted_score"]) for row in graded)
    ci = (
        (agg.get("factuality_score_ci95_low"), agg.get("factuality_score_ci95_high"))
        if agg.get("factuality_score_ci95_low") is not None
        else None
    )
    grounded_both = sum(1 for row in eligible_rows if row.get("is_grounded_all_judges"))
    disagreements = sum(1 for row in eligible_rows if row.get("judges_agree") is False)
    truncated = sum(1 for row in rows if row.get("generation_truncated"))
    empty = sum(1 for row in rows if row.get("generation_empty"))
    invalid_elig = sum(1 for row in rows if int(row.get("eligibility_invalid_count", 0)) > 0)
    metrics = [
        _m(
            "factuality_score",
            "Factuality score (adjusted)",
            score_sum / n if n else None,
            numerator=score_sum,
            denominator=n,
            kind="primary",
            definition="Mean over prompts of the grounding score (share of judges that found every sentence supported) for eligible answers, 0 for ineligible answers; the leaderboard metric.",
            ci95=ci,
        ),
        _m(
            "unadjusted_factuality_score",
            "Unadjusted factuality score",
            unadjusted_sum / len(graded) if graded else None,
            numerator=unadjusted_sum,
            denominator=len(graded),
            definition="Mean grounding score before disqualifying ineligible answers (both judges were run on every answer).",
        ),
        _m(
            "eligibility_rate",
            "Eligible answers",
            n_eligible / n if n else None,
            numerator=n_eligible,
            denominator=n,
            definition="Answers that at least one judge rated as not having major instruction-following issues versus its own baseline answer.",
        ),
        _m(
            "grounded_all_judges_rate_eligible",
            "Grounded by both judges (eligible answers)",
            grounded_both / n_eligible if n_eligible else None,
            numerator=grounded_both,
            denominator=n_eligible,
            definition="Eligible answers whose every sentence was supported according to both judges.",
        ),
        _m(
            "judge_disagreement_rate_eligible",
            "Judge disagreement (eligible answers)",
            disagreements / n_eligible if n_eligible else None,
            numerator=disagreements,
            denominator=n_eligible,
            direction="neutral",
            kind="diagnostic",
            definition="Eligible answers where exactly one judge found an unsupported sentence.",
        ),
    ]
    for judge in JUDGES:
        verdict_rows = [row for row in eligible_rows if judge in (row.get("grounding_verdicts") or {})]
        grounded = sum(1 for row in verdict_rows if row["grounding_verdicts"][judge])
        metrics.append(
            _m(
                f"grounded_rate_eligible/{judge}",
                f"Grounded per {judge} (eligible answers)",
                grounded / len(verdict_rows) if verdict_rows else None,
                numerator=grounded,
                denominator=len(verdict_rows),
                definition=f"Eligible answers {judge} found fully supported.",
            )
        )
        decided = sum(1 for row in rows if row.get("eligibility_deciding_judge") == judge)
        metrics.append(
            _m(
                f"eligibility_decided_by/{judge}",
                f"Eligibility settled by {judge}",
                decided / n if n else None,
                numerator=decided,
                denominator=n,
                direction="neutral",
                kind="diagnostic",
                definition="Share of answers whose eligibility was settled by this judge (the first judge without a major-issue verdict wins).",
            )
        )
        parsed_rows = [row for row in rows if judge in (row.get("grounding_parsed_sentences") or {})]
        empty_parse = sum(1 for row in parsed_rows if row["grounding_parsed_sentences"][judge] == 0)
        metrics.append(
            _m(
                f"grounding_parse_empty_rate/{judge}",
                f"Unparseable grounding verdicts from {judge}",
                empty_parse / len(parsed_rows) if parsed_rows else None,
                numerator=empty_parse,
                denominator=len(parsed_rows),
                direction="lower_is_better",
                kind="diagnostic",
                definition="Grounding replies with no parseable sentence object (counted as not grounded, as the starter does).",
            )
        )
    metrics += [
        _m(
            "eligibility_invalid_rate",
            "Unparseable eligibility verdicts",
            invalid_elig / n if n else None,
            numerator=invalid_elig,
            denominator=n,
            direction="lower_is_better",
            kind="diagnostic",
            definition="Answers with at least one eligibility reply lacking the verdict JSON (treated as eligible, as the starter does).",
        ),
        _m(
            "generation_truncated_rate",
            "Truncated answers",
            truncated / n if n else None,
            numerator=truncated,
            denominator=n,
            direction="lower_is_better",
            kind="operational",
            definition="Answers cut off by the output-token cap.",
        ),
        _m(
            "generation_empty_rate",
            "Empty answers",
            empty / n if n else None,
            numerator=empty,
            denominator=n,
            direction="lower_is_better",
            kind="operational",
            definition="Answers with no visible text.",
        ),
        _m(
            "policy_input_tokens_max",
            "Longest prompt (tokens)",
            agg.get("policy_input_tokens_max"),
            unit="tokens",
            direction="neutral",
            kind="operational",
            definition="Largest prompt token count seen by the policy endpoint (context window utilisation).",
        ),
        _m(
            "mean_output_tokens",
            "Mean output tokens (reasoning + answer)",
            agg.get("mean/output_tokens"),
            unit="tokens",
            direction="neutral",
            kind="operational",
            definition="Policy output tokens per answer.",
        ),
    ]
    for slice_key in ("domain", "high_level_type"):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            groups[str(row.get(slice_key) or "unknown")].append(row)
        for value, group in sorted(groups.items()):
            total = sum(float(row.get("reward", 0.0)) for row in group)
            metrics.append(
                _m(
                    f"factuality_score/{slice_key}/{value}",
                    f"Factuality score · {slice_key} {value}",
                    total / len(group),
                    numerator=total,
                    denominator=len(group),
                    kind="slice",
                    slice_={slice_key: value},
                    definition="Adjusted factuality score restricted to this slice.",
                )
            )

    rollouts: list[RolloutSummary] = []
    for row in rows:
        task = _task_id(row)
        rollouts.append(
            RolloutSummary(
                task_id=task,
                rollout_id=f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                receipt_ids=[
                    f"{task}/{receipt['stage']}-{receipt['judge']}" for receipt in row.get("judge_receipts", [])
                ],
                slices={
                    "domain": str(row.get("domain") or "unknown"),
                    "high_level_type": str(row.get("high_level_type") or "unknown"),
                    "type": str(row.get("type") or "unknown"),
                },
                outcome_class="scored",
                reward=float(row.get("reward", 0.0)),
                components={
                    "eligible": row.get("eligible"),
                    "eligibility_ratings": row.get("eligibility_ratings"),
                    "grounding_verdicts": row.get("grounding_verdicts"),
                    "grounding_label_counts": row.get("grounding_label_counts"),
                    "unadjusted_score": row.get("unadjusted_score"),
                    "generation_truncated": row.get("generation_truncated"),
                    "policy_input_tokens": row.get("policy_input_tokens"),
                },
                prompt_excerpt=excerpt(row.get("user_request")),
                response_excerpt=excerpt(row.get("generation")),
                judge_excerpt=excerpt(
                    next(
                        (r.get("content") for r in row.get("judge_receipts", []) if r.get("stage") == "grounding"), ""
                    )
                ),
            )
        )
    rollouts += failure_summaries(unresolved, task_id_of=_task_id)
    first = [rollouts[0].rollout_id] if rollouts else ["run"]
    anchors = [
        AnchorFact(
            id="fact-score",
            category="outcome",
            fact=f"The adjusted factuality score was {score_sum / n:.1%} over {n} answers ({n_eligible} eligible; {grounded_both} eligible answers grounded by both judges, {disagreements} with a split verdict); the unadjusted score over all graded answers was {unadjusted_sum / len(graded):.1%}.",
            evidence=first,
        ),
        AnchorFact(
            id="fact-coverage",
            category="coverage",
            fact=f"{outcomes.scored_rollouts} of {outcomes.expected_rollouts} expected answers were judged; {outcomes.judge_failed} judge failures, {outcomes.infrastructure_failed} infrastructure failures, {outcomes.missing_rollouts} missing, {outcomes.replaced_attempts} sidecar attempts superseded.",
            evidence=first,
        ),
        AnchorFact(
            id="fact-eligibility",
            category="invalid",
            fact=f"{n - n_eligible} of {n} answers were ineligible (both judges reported major instruction-following issues) and scored 0; {invalid_elig} answers had an eligibility reply without a parseable verdict.",
            evidence=[r.rollout_id for r in rollouts if r.components.get("eligible") is False][:3] or first,
        ),
        AnchorFact(
            id="fact-truncation",
            category="infrastructure",
            fact=f"{truncated} of {n} answers hit the output-token cap and {empty} were empty; the longest prompt used {agg.get('policy_input_tokens_max', 'n/a')} input tokens against a 1,048,576-token window, so no prompt was clipped.",
            evidence=[r.rollout_id for r in rollouts if r.components.get("generation_truncated")][:3] or first,
        ),
    ]
    for judge in JUDGES:
        verdict_rows = [row for row in eligible_rows if judge in (row.get("grounding_verdicts") or {})]
        grounded = sum(1 for row in verdict_rows if row["grounding_verdicts"][judge])
        anchors.append(
            AnchorFact(
                id=f"fact-judge-{judge}",
                category="outcome",
                fact=f"{judge} found {grounded} of {len(verdict_rows)} eligible answers fully supported.",
                evidence=[
                    r.rollout_id
                    for r in rollouts
                    if (r.components.get("grounding_verdicts") or {}).get(judge) is False
                ][:2]
                or first,
            )
        )
    domain_stats = []
    for value, group in collections.Counter(str(row.get("domain") or "unknown") for row in rows).items():
        subset = [row for row in rows if str(row.get("domain") or "unknown") == value]
        if len(subset) >= 20:
            domain_stats.append(
                (value, sum(float(row.get("reward", 0.0)) for row in subset) / len(subset), len(subset))
            )
    if domain_stats:
        weakest = min(domain_stats, key=lambda item: item[1])
        strongest = max(domain_stats, key=lambda item: item[1])
        anchors.append(
            AnchorFact(
                id="fact-slices",
                category="slice",
                fact=f"Among domains with at least 20 prompts, the adjusted score was lowest for {weakest[0]} ({weakest[1]:.1%}, {weakest[2]} prompts) and highest for {strongest[0]} ({strongest[1]:.1%}, {strongest[2]} prompts).",
                evidence=[r.rollout_id for r in rollouts if r.slices.get("domain") == weakest[0]][:2]
                + [r.rollout_id for r in rollouts if r.slices.get("domain") == strongest[0]][:1],
            )
        )

    def _example(predicate, label):
        for row in rows:
            if predicate(row):
                task = _task_id(row)
                grounding = next(
                    (
                        r
                        for r in row.get("judge_receipts", [])
                        if r.get("stage") == "grounding" and not r.get("grounded")
                    ),
                    None,
                ) or next((r for r in row.get("judge_receipts", []) if r.get("stage") == "grounding"), {})
                unsupported = [s for s in grounding.get("sentences", []) if s.get("label") == "not_supported"][:1]
                detail = (
                    f" One unsupported sentence per {grounding.get('judge')}: '{excerpt(unsupported[0].get('sentence'), 140)}' — {excerpt(unsupported[0].get('rationale'), 160)}"
                    if unsupported
                    else ""
                )
                return AnchorFact(
                    id=f"example-{label}",
                    category="example",
                    fact=f"Prompt {task} ({row.get('domain')}, {row.get('type')}): request '{excerpt(row.get('user_request'), 120)}'; eligibility {row.get('eligibility_ratings')}; verdicts {row.get('grounding_verdicts')}; score {row.get('reward')}.{detail}",
                    evidence=[task, f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}"]
                    + [f"{task}/{r['stage']}-{r['judge']}" for r in row.get("judge_receipts", [])],
                    excerpt=excerpt(row.get("generation"), 300),
                )
        return None

    for predicate, label in (
        (lambda row: row.get("reward") == 1.0, "grounded"),
        (lambda row: row.get("eligible") and row.get("reward") == 0.0, "unsupported"),
        (lambda row: row.get("reward") == 0.5, "split-verdict"),
        (lambda row: not row.get("eligible"), "ineligible"),
    ):
        fact = _example(predicate, label)
        if fact is not None:
            anchors.append(fact)
    if calibration is not None:
        anchors.append(
            AnchorFact(
                id="fact-calibration",
                category="calibration",
                fact=f"Calibration ({calibration.method}): {calibration.agreement} of {calibration.cases} cases agreed with the upstream control; {calibration.disagreements} disagreed.",
                evidence=[str(case) for case in calibration.details.get("case_ids", [])[:3]] or ["calibration"],
            )
        )

    limitations = [
        "Only the 856-prompt public set is scored; the leaderboard pools it with 859 private prompts held by Kaggle, so this number is not a leaderboard submission.",
        "Both judges sample with provider defaults, so grounding verdicts are stochastic; split verdicts (score 0.5) are reported separately.",
        "Eligibility follows the starter: the first judge whose verdict is not 'Major Issue(s)' settles it, and an unparseable verdict counts as eligible.",
        "A judge reply that cannot be parsed into sentence objects counts as 'not grounded' for that judge, as in the starter; the rate is reported.",
        "Reasoning tokens count against the policy's output cap on this endpoint; truncated answers are judged as given.",
    ]
    blade = BladeMapping(
        d1_metrics={
            "pass_at_1": "N/A: the official metric is the eligibility-adjusted mean of two judge verdicts (0, 0.5, 1), not a binary pass; see metric:factuality_score",
            "primary": "metric:factuality_score",
            "consistency": "N/A: one answer per prompt",
        },
        d2_anchor_categories=sorted({fact.category for fact in anchors}),
        d3_notes=[
            "Single-turn grounded generation: no tool calls, no multi-turn funnel; eligibility, per-judge groundedness, judge disagreement, parse validity and truncation are the reported components."
        ],
        not_applicable={
            "pass@k": "single rollout per prompt",
            "tool_call_funnel": "no tools",
            "task_root_cause": "no repeats",
        },
    )
    run = dict(run_info)
    run.setdefault("repeats", num_repeats)
    return NormalizedRun(
        benchmark=BENCHMARK,
        run=run,
        outcomes=outcomes,
        metrics=metrics,
        rollouts=rollouts,
        anchor_facts=anchors,
        calibration=calibration,
        limitations=limitations,
        reference_comparisons=REFERENCES,
        blade=blade,
        reward_semantics="reward = share of judges that found the answer fully grounded (0, 0.5, 1) when the answer is eligible, else 0; the factuality score is its mean.",
    )
