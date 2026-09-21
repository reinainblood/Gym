# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched-prompt inference between two models on the same KIDBench cells.

Every model answers the same 5,000 single-turn cells, so a difference between two of them
is a paired quantity and should be tested as one. An unpaired test on these numbers throws
away the pairing and widens the interval for no reason.

Two things here are deliberate and worth stating, because both are easy to get wrong:

Ranking is on the *mean rubric score*, not on a pass rate. KIDBench scores a response 1-5;
collapsing that to pass/fail at the 3.0 line and then testing the pass rate would measure a
threshold this adapter chose rather than the rubric the paper defines. The unsafe-rate
contrast is available too, but it is reported beside the score contrast rather than instead
of it.

Significance is corrected across the whole family of tests, not per comparison. Six model
pairs across several slices is enough tests that an uncorrected 0.05 would produce
significant-looking noise; Benjamini-Hochberg keeps the false-discovery rate at 0.05 over
every test this module runs at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from benchmarks.kidbench.reporting.analyze import CUE_CONDITIONS, SAFE_THRESHOLD, ModelResults


#: Draws for the sign-flip permutation null. Large enough that the smallest reportable
#: p-value is 1e-5, which is finer than anything this campaign needs to resolve.
PERMUTATIONS: int = 100_000
#: Paired bootstrap resamples for the effect interval.
BOOTSTRAP: int = 50_000
#: Fixed so that a rerun of the same rollouts reproduces the same intervals. It advances per
#: test rather than being reused, so two tests never share a resampling pattern.
SEED: int = 20260920
FDR_ALPHA: float = 0.05


@dataclass(frozen=True)
class PairedTest:
    slice_name: str
    model_a: str
    model_b: str
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_adjusted: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "slice": self.slice_name,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "matched_cells": self.n,
            "mean_difference": self.mean_difference,
            "ci_95": [self.ci_low, self.ci_high],
            "p_raw": self.p_raw,
            "p_adjusted": self.p_adjusted,
            "significant": bool(self.p_adjusted is not None and self.p_adjusted < FDR_ALPHA),
        }


def _cell_scores(model: ModelResults, *, predicate=None) -> dict[str, float]:
    """Matched cell id -> rubric score, over cells this model actually has a verdict for.

    Keyed on ``kidbench_id`` because that is what identifies the same prompt under the same
    cue condition across models. Rows whose judge produced no verdict are absent rather than
    zero-filled: a missing verdict is not a score of zero, and filling it would drag a model's
    mean toward whichever direction had more parse failures.
    """
    scores: dict[str, float] = {}
    for row in model.single_rows:
        if row.get("judge_parse_failed") or not isinstance(row.get("total_quality_score"), (int, float)):
            continue
        if predicate is not None and not predicate(row):
            continue
        key = row.get("kidbench_id")
        if key:
            scores[str(key)] = float(row["total_quality_score"])
    return scores


def paired_test(
    model_a: ModelResults,
    model_b: ModelResults,
    *,
    slice_name: str,
    predicate=None,
    seed_offset: int = 0,
) -> Optional[PairedTest]:
    """Two-sided sign-flip permutation test with a paired bootstrap interval.

    The null is that the two models score the same cell equally, so flipping the sign of a
    per-cell difference is exchangeable under it. That null needs no distributional
    assumption about rubric scores, which are ordinal and bounded and not normal.
    """
    a, b = _cell_scores(model_a, predicate=predicate), _cell_scores(model_b, predicate=predicate)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None

    differences = np.array([a[key] - b[key] for key in shared], dtype=np.float64)
    observed = float(differences.mean())

    rng = np.random.default_rng(SEED + seed_offset)
    # Sign flips in blocks: the full (PERMUTATIONS x n) matrix would be 4 GB at n=5000.
    at_least_as_extreme = 0
    remaining = PERMUTATIONS
    while remaining > 0:
        block = min(remaining, 2000)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(block, differences.size))
        at_least_as_extreme += int((np.abs(signs @ differences / differences.size) >= abs(observed) - 1e-12).sum())
        remaining -= block
    # Add-one so a p-value is never exactly zero: no finite number of draws can prove that.
    p_raw = (at_least_as_extreme + 1) / (PERMUTATIONS + 1)

    # Blocked for the same reason as the permutation: a single (BOOTSTRAP x n) draw is about
    # 2 GB at n=5000, which is enough to push this machine into swap mid-campaign.
    means = np.empty(BOOTSTRAP, dtype=np.float64)
    filled = 0
    while filled < BOOTSTRAP:
        block = min(2000, BOOTSTRAP - filled)
        sample = rng.choice(differences, size=(block, differences.size), replace=True)
        means[filled : filled + block] = sample.mean(axis=1)
        filled += block
    ci_low, ci_high = (float(v) for v in np.percentile(means, [2.5, 97.5]))

    return PairedTest(
        slice_name=slice_name,
        model_a=model_a.display_name,
        model_b=model_b.display_name,
        n=len(shared),
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_raw=p_raw,
    )


def benjamini_hochberg(tests: list[PairedTest], *, alpha: float = FDR_ALPHA) -> list[PairedTest]:
    """Adjust across every test in the family, preserving the caller's order."""
    if not tests:
        return []
    order = sorted(range(len(tests)), key=lambda i: tests[i].p_raw)
    total = len(tests)
    adjusted = [0.0] * total
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, tests[index].p_raw * total / rank)
        adjusted[index] = running
    return [
        PairedTest(
            slice_name=test.slice_name,
            model_a=test.model_a,
            model_b=test.model_b,
            n=test.n,
            mean_difference=test.mean_difference,
            ci_low=test.ci_low,
            ci_high=test.ci_high,
            p_raw=test.p_raw,
            p_adjusted=adjusted[index],
        )
        for index, test in enumerate(tests)
    ]


#: Slices worth testing separately. The cue conditions are the benchmark's own construction
#: and the finding most likely to be quoted, so they are tested rather than eyeballed.
SLICES: tuple[tuple[str, Any], ...] = (
    ("all cells", None),
    # The three ladder rungs are `condition` values, the same field `by_condition` and
    # `outcome_buckets` key on. Deriving them instead from `prompt_type`/`age_conditioned`
    # looks equivalent and is not: those flags are also set on the 2,000 cultural cells, so
    # each rung came out five times too large and averaged in cells from other conditions.
    *((rung.replace("_", " "), (lambda r, c=rung: r.get("condition") == c)) for rung in CUE_CONDITIONS),
    ("cultural cells", lambda row: row.get("condition") == "cultural"),
    ("non-English cells", lambda row: row.get("condition") == "cross_lingual"),
)


def all_pairwise(models: list[ModelResults]) -> list[PairedTest]:
    """Every model pair on every slice, corrected once across the whole family."""
    tests: list[PairedTest] = []
    offset = 0
    for slice_name, predicate in SLICES:
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                offset += 1
                test = paired_test(model_a, model_b, slice_name=slice_name, predicate=predicate, seed_offset=offset)
                if test is not None:
                    tests.append(test)
    return benjamini_hochberg(tests)


def unsafe_rate_contrast(model_a: ModelResults, model_b: ModelResults) -> Optional[dict[str, Any]]:
    """The same pairing, on the share of cells below the safe line rather than on score.

    Reported beside the score contrast because the two can disagree: a model can score
    lower on average while crossing the 3.0 line less often, and a reader who sees only one
    of those numbers will draw the wrong conclusion about deployment risk.
    """
    a, b = _cell_scores(model_a), _cell_scores(model_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    unsafe_a = sum(1 for key in shared if a[key] < SAFE_THRESHOLD) / len(shared)
    unsafe_b = sum(1 for key in shared if b[key] < SAFE_THRESHOLD) / len(shared)
    return {
        "model_a": model_a.display_name,
        "model_b": model_b.display_name,
        "matched_cells": len(shared),
        "unsafe_rate_a": unsafe_a,
        "unsafe_rate_b": unsafe_b,
        "difference": unsafe_a - unsafe_b,
    }


def format_p(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value < 1e-5:
        return "<0.00001"
    if value < 0.0001:
        return f"{value:.6f}".rstrip("0")
    return f"{value:.5f}".rstrip("0")


def summary(models: list[ModelResults]) -> dict[str, Any]:
    tests = all_pairwise(models)
    contrasts = [
        contrast
        for i, model_a in enumerate(models)
        for model_b in models[i + 1 :]
        if (contrast := unsafe_rate_contrast(model_a, model_b)) is not None
    ]
    significant = [test for test in tests if test.p_adjusted is not None and test.p_adjusted < FDR_ALPHA]
    return {
        "method": (
            "Paired by KIDBench cell id. Two-sided sign-flip permutation test with "
            f"{PERMUTATIONS:,} draws; 95% intervals from {BOOTSTRAP:,} paired bootstrap resamples; "
            f"Benjamini-Hochberg across all {len(tests)} tests at a {FDR_ALPHA} false-discovery rate."
        ),
        "seed": SEED,
        "unit": "mean total quality score (1-5), model_a minus model_b",
        "safe_threshold": SAFE_THRESHOLD,
        "tests_run": len(tests),
        "tests_significant": len(significant),
        "tests": [test.to_json() for test in tests],
        "unsafe_rate_contrasts": contrasts,
        "not_applicable": {
            "pass_at_k": "one rollout per cell; k>1 metrics are undefined for this campaign",
            "consistency": "requires repeats of the same cell",
            "retry_value": "requires repeats of the same cell",
        },
    }


def is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
