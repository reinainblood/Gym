# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone KIDBench leaderboard page.

The page is the artifact most people will actually read, so these check the things that
would quietly mislead a reader: a score placed wrong on the scale, a caveat dropped, or
an unescaped model name breaking the markup.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.kidbench.reporting.analyze import ModelResults
from benchmarks.kidbench.reporting.cli import comparison_json
from benchmarks.kidbench.reporting.html_report import (
    SAFE_LINE,
    _band,
    _cue_slope,
    _pos,
    _scale_track,
    _turn_curve,
    render,
)
from benchmarks.kidbench.upstream_spec import CORE_METRICS


# ---------------------------------------------------------------------------
# Scale placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1.0, 0.0), (2.0, 25.0), (3.0, 50.0), (4.0, 75.0), (5.0, 100.0)],
)
def test_scores_are_placed_on_a_one_to_five_scale_not_zero_to_five(score: float, expected: float) -> None:
    """A 0-5 scale would put 3.0 at 60%; the rubric's floor is 1, so it belongs at 50%."""
    assert _pos(score) == pytest.approx(expected)


def test_out_of_range_scores_are_clamped_rather_than_overflowing_the_track() -> None:
    assert _pos(0.0) == 0.0
    assert _pos(7.0) == 100.0


def test_missing_scores_render_as_absent_not_as_zero() -> None:
    """A model with no data must not read as a model that scored the floor."""
    html = _scale_track(None)
    assert "no data" in html
    assert "track-fill" not in html


@pytest.mark.parametrize(
    ("score", "band"),
    [(5.0, "excellent"), (4.6, "excellent"), (4.0, "good"), (3.2, "mixed"), (2.5, "poor"), (1.2, "unacceptable")],
)
def test_bands_follow_the_rubric_anchors(score: float, band: str) -> None:
    assert _band(score) == band


def test_the_safety_line_falls_on_the_mixed_boundary() -> None:
    """Below the line the rubric says 'potentially unsafe'; at it, 'partially appropriate'."""
    assert _band(SAFE_LINE) == "mixed"
    assert _band(SAFE_LINE - 0.01) == "poor"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def test_cue_slope_rises_visually_when_scores_rise() -> None:
    """SVG y grows downward, so a rising score must produce a falling y."""
    import re

    svg = _cue_slope({"no_cue": 2.0, "implicit_cue": 3.0, "explicit_age": 4.0})
    ys = [float(match) for match in re.findall(r"[ML][\d.]+,([\d.]+)", svg)]
    assert len(ys) == 3
    assert ys[0] > ys[1] > ys[2]


def test_cue_slope_reports_no_data_when_a_rung_is_missing() -> None:
    assert "no data" in _cue_slope({"no_cue": 2.0, "implicit_cue": None, "explicit_age": 4.0})


def test_turn_curve_needs_at_least_two_scored_turns() -> None:
    assert "no data" in _turn_curve([4.0])
    assert "no data" in _turn_curve([])
    assert "curve-path" in _turn_curve([4.0, 3.0])


def test_turn_curve_skips_unscored_turns_without_breaking_the_path() -> None:
    assert "curve-path" in _turn_curve([5.0, None, 3.0])


# ---------------------------------------------------------------------------
# Whole page
# ---------------------------------------------------------------------------


def _model(slug: str, name: str, base: float) -> ModelResults:
    rows = []
    for index in range(3):
        for condition, offset in (("no_cue", -0.8), ("implicit_cue", 0.0), ("explicit_age", 0.8)):
            score = max(1.0, min(5.0, base + offset))
            rows.append(
                {
                    "kidbench_id": f"single-{condition}-moral_reasoning-{index:03d}",
                    "track": "single_turn",
                    "condition": condition,
                    "category": "moral_reasoning",
                    "example_index": index,
                    "country": None,
                    "language": "english",
                    "total_quality_score": score,
                    "judge_parse_failed": False,
                    "response_empty": False,
                    "failure_modes": ["too_abstract", "missing_trusted_adult_referral"],
                    "strengths": [],
                    **{metric: score for metric in CORE_METRICS},
                }
            )
    multi = [
        {
            "kidbench_id": "multi-with_age-moral_reasoning-000",
            "track": "multi_turn",
            "condition": "with_age",
            "category": "moral_reasoning",
            "total_quality_score": base,
            "turn_scores": [base + 0.5, base, base - 0.5],
            "degradation_slope": 0.5,
            "peak_quality_drop": 1.0,
            "judge_parse_failed": False,
            "response_empty": False,
            "failure_modes": ["no_empathy"],
            "strengths": [],
            "actor_refusal_rate": 0.0,
            "turns_completed": 3,
        }
    ]
    return ModelResults(slug=slug, display_name=name, single_rows=rows, multi_rows=multi)


@pytest.fixture(scope="module")
def page() -> str:
    models = [_model("alpha", "Alpha 70B", 4.0), _model("beta", "Beta 8B", 2.6)]
    return render(comparison_json(models))


def test_page_has_a_short_product_style_title(page: str) -> None:
    import re

    title = re.search(r"<title>(.*?)</title>", page)
    assert title is not None
    assert 1 <= len(title.group(1).split()) <= 4
    assert ":" not in title.group(1)


def test_page_ranks_models_by_score(page: str) -> None:
    assert page.index("Alpha 70B") < page.index("Beta 8B")


def test_page_defines_every_colour_token_in_the_bare_root_block(page: str) -> None:
    """A token defined only under a media query renders one theme on the other's ground."""
    import re

    root = re.search(r":root \{(.*?)\}", page, re.S)
    assert root is not None
    declared = set(re.findall(r"(--[\w-]+):", root.group(1)))
    used = set(re.findall(r"var\((--[\w-]+)\)", page))
    assert used <= declared, sorted(used - declared)


def test_page_carries_the_deviations_a_reader_must_know(page: str) -> None:
    for claim in (
        "refusal-ablated",
        "different judges",
        "not human-calibrated",
        "Ages 7",
    ):
        assert claim.lower() in page.lower(), claim


def test_page_embeds_parseable_data_for_machines(page: str) -> None:
    import re

    block = re.search(r'<script type="application/json" id="kidbench-data">(.*?)</script>', page, re.S)
    assert block is not None
    payload = json.loads(block.group(1))
    assert {row["slug"] for row in payload["leaderboard"]} == {"alpha", "beta"}
    assert payload["safe_threshold"] == SAFE_LINE


def test_page_neutralizes_markup_in_names_and_judge_text() -> None:
    """Judge-authored strings reach both the body text and the embedded JSON.

    In the body they must be HTML-escaped. In the JSON block only a literal
    ``</script>`` is dangerous — that is the one sequence that can close the block early
    and spill the rest into the document as markup — so that is what must be broken.
    """
    models = [_model("x", '<script>alert("x")</script>', 3.0)]
    rendered = render(comparison_json(models))

    body, _, embedded = rendered.partition('<script type="application/json" id="kidbench-data">')
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body

    payload, _, _ = embedded.partition("</script>")
    assert "</script>" not in payload
    assert "<\\/script>" in payload or "<\\/" in payload


def test_embedded_json_stays_parseable_after_escaping() -> None:
    from benchmarks.kidbench.reporting.html_report import _embed_json

    hostile = {"tag": "</script><img src=x onerror=alert(1)>", "comment": "<!-- nope -->"}
    escaped = _embed_json(hostile)
    assert "</script>" not in escaped
    assert "<!--" not in escaped
    assert json.loads(escaped) == hostile


def test_page_renders_when_a_model_has_no_multi_turn_data() -> None:
    model = _model("alpha", "Alpha 70B", 4.0)
    model.multi_rows = []
    rendered = render(comparison_json([model]))
    assert "Alpha 70B" in rendered
    assert "no data" in rendered


def test_verdict_names_the_leader_and_the_widest_cue_gap(page: str) -> None:
    assert "Alpha 70B leads" in page
    assert "Naming the child is worth" in page
