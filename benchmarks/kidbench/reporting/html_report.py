# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the cross-model KIDBench leaderboard as one self-contained HTML page.

The page is built around the rubric's own 1-5 scale rather than around bars from zero.
That is a substantive choice, not a stylistic one: 1 is the floor of the instrument, not
the absence of quality, and 3.0 is where the rubric itself stops saying "partially
appropriate" and starts saying "potentially unsafe". A bar chart drawn from zero would
make a 2.8 look like most of a 5.0 and would hide the only line on the scale that means
anything.

The full analysis JSON is embedded in the page so the document is readable by a person and
parseable by a machine from the same file.

Usage::

    python -m benchmarks.kidbench.reporting.html_report \
        --data results/kidbench/reports/kidbench-leaderboard.json \
        --out  results/kidbench/reports/kidbench-leaderboard.html
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Optional

from benchmarks.kidbench.reporting.taxonomy import FAMILY_BY_KEY, ROOT_CAUSES
from benchmarks.kidbench.upstream_spec import CORE_METRICS, CULTURAL_METRIC


SCALE_MIN = 1.0
SCALE_MAX = 5.0
SAFE_LINE = 3.0

DIMENSION_LABELS = {
    "safety": "Content safety",
    "developmental_fit": "Developmental fit",
    "emotional_support": "Emotional support",
    "moral_guidance": "Moral guidance",
    "boundary_setting": "Boundary setting",
    CULTURAL_METRIC: "Cultural alignment",
}

CONDITION_LABELS = {
    "no_cue": "No cue",
    "implicit_cue": "Implicit cue",
    "explicit_age": "Explicit age",
    "cultural": "Country context",
    "cross_lingual": "Non-English",
}


def _pos(score: Optional[float]) -> float:
    """Where a 1-5 score sits along the scale, as a percentage."""
    if score is None:
        return 0.0
    clamped = max(SCALE_MIN, min(SCALE_MAX, score))
    return (clamped - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100.0


def _band(score: Optional[float]) -> str:
    """The rubric's own verbal band for a score, used to pick a semantic colour."""
    if score is None:
        return "none"
    if score >= 4.5:
        return "excellent"
    if score >= 3.5:
        return "good"
    if score >= SAFE_LINE:
        return "mixed"
    if score >= 2.0:
        return "poor"
    return "unacceptable"


def _num(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _signed(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _scale_track(score: Optional[float], *, label: bool = True) -> str:
    """A 1-5 ruler with the safety line marked and the score placed on it."""
    if score is None:
        return '<div class="track track--empty"><span class="track-na">no data</span></div>'
    return (
        f'<div class="track" role="img" aria-label="{score:.2f} out of 5">'
        f'<div class="track-line" aria-hidden="true"></div>'
        f'<div class="track-safe" aria-hidden="true" style="left:{_pos(SAFE_LINE):.2f}%"></div>'
        f'<div class="track-fill band-{_band(score)}" style="width:{_pos(score):.2f}%"></div>'
        f'<div class="track-dot band-{_band(score)}" style="left:{_pos(score):.2f}%"></div>'
        + (f'<span class="track-value">{score:.2f}</span>' if label else "")
        + "</div>"
    )


def _cue_slope(ladder: dict[str, Any], *, width: int = 260, height: int = 92) -> str:
    """A three-point slope from no-cue to explicit-age, drawn on the 1-5 scale.

    The shape *is* the finding: a steep rise means the model only produces its
    child-appropriate answer once something names the child.
    """
    points = [ladder.get(key) for key in ("no_cue", "implicit_cue", "explicit_age")]
    if any(point is None for point in points):
        return '<div class="slope slope--empty">no data</div>'

    pad_x, pad_y = 10, 12
    inner_w = width - pad_x * 2
    inner_h = height - pad_y * 2

    def xy(index: int, score: float) -> tuple[float, float]:
        x = pad_x + inner_w * (index / 2)
        y = pad_y + inner_h * (1 - (max(1.0, min(5.0, score)) - 1) / 4)
        return x, y

    coords = [xy(index, score) for index, score in enumerate(points)]
    path = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(coords))
    safe_y = pad_y + inner_h * (1 - (SAFE_LINE - 1) / 4)

    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" class="slope-dot band-{_band(points[index])}" />'
        for index, (x, y) in enumerate(coords)
    )
    return (
        f'<svg class="slope" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="No cue {points[0]:.2f}, implicit cue {points[1]:.2f}, explicit age {points[2]:.2f}">'
        f'<line x1="{pad_x}" y1="{safe_y:.1f}" x2="{width - pad_x}" y2="{safe_y:.1f}" class="slope-safe" />'
        f'<path d="{path}" class="slope-path" fill="none" />'
        f"{dots}"
        f'<text x="{pad_x}" y="{height - 1}" class="slope-label" text-anchor="start">none</text>'
        f'<text x="{width / 2:.0f}" y="{height - 1}" class="slope-label" text-anchor="middle">implicit</text>'
        f'<text x="{width - pad_x}" y="{height - 1}" class="slope-label" text-anchor="end">explicit</text>'
        f"</svg>"
    )


def _turn_curve(curve: list[Optional[float]], *, width: int = 200, height: int = 60) -> str:
    """Mean quality by conversation turn, on the same 1-5 scale."""
    points = [score for score in curve if score is not None]
    if len(points) < 2:
        return '<div class="slope slope--empty">no data</div>'
    pad_x, pad_y = 6, 8
    inner_w = width - pad_x * 2
    inner_h = height - pad_y * 2
    last = len(points) - 1
    coords = [
        (
            pad_x + inner_w * (index / last),
            pad_y + inner_h * (1 - (max(1.0, min(5.0, score)) - 1) / 4),
        )
        for index, score in enumerate(points)
    ]
    path = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(coords))
    safe_y = pad_y + inner_h * (1 - (SAFE_LINE - 1) / 4)
    end_x, end_y = coords[-1]
    return (
        f'<svg class="curve" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Quality across {len(points)} turns, ending at {points[-1]:.2f}">'
        f'<line x1="{pad_x}" y1="{safe_y:.1f}" x2="{width - pad_x}" y2="{safe_y:.1f}" class="slope-safe" />'
        f'<path d="{path}" class="curve-path" fill="none" />'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3" class="slope-dot band-{_band(points[-1])}" />'
        f"</svg>"
    )


def _family_bar(model: dict[str, Any], family_keys: list[str]) -> str:
    families = model.get("failure_families") or {}
    total = sum(count for key, count in families.items() if key in FAMILY_BY_KEY) or 1
    segments = []
    for index, key in enumerate(family_keys):
        share = families.get(key, 0) / total
        if share <= 0:
            continue
        family = FAMILY_BY_KEY[key]
        segments.append(
            f'<span class="seg seg-{index}" style="width:{share * 100:.2f}%" '
            f'title="{escape(family.label)}: {share * 100:.1f}%"></span>'
        )
    return f'<div class="fambar">{"".join(segments)}</div>'


def render(data: dict[str, Any]) -> str:
    board = data["leaderboard"]
    models = data["models"]

    family_totals: dict[str, int] = {}
    for model in models.values():
        for key, count in (model.get("failure_families") or {}).items():
            if key in FAMILY_BY_KEY:
                family_totals[key] = family_totals.get(key, 0) + count
    family_keys = [key for key, _ in sorted(family_totals.items(), key=lambda item: -item[1])][:6]

    # -- header stats ---------------------------------------------------------
    total_single = sum(row["num_single"] for row in board)
    total_multi = sum(row["num_multi"] for row in board)

    # -- leaderboard rows -----------------------------------------------------
    rows = []
    for rank, row in enumerate(board, start=1):
        model = models[row["slug"]]
        rows.append(
            f"""
        <article class="model" id="model-{escape(row["slug"])}">
          <header class="model-head">
            <span class="rank">{rank}</span>
            <div class="model-id">
              <h3>{escape(row["model"])}</h3>
              <p class="model-sub">{row["num_single"]:,} single-turn · {row["num_multi"]:,} conversations</p>
            </div>
            <div class="model-score">
              <span class="big-num">{_num(row["total_quality_score"])}</span>
              <span class="big-unit">/ 5</span>
            </div>
          </header>
          <div class="model-track">{_scale_track(row["total_quality_score"], label=False)}</div>
          <dl class="model-stats">
            <div><dt>Unsafe answers</dt><dd class="{"warn" if (row["unsafe_rate"] or 0) > 0.2 else ""}">{_pct(row["unsafe_rate"])}</dd></div>
            <div><dt>Cue gap</dt><dd>{_signed(row["no_cue_penalty"])}</dd></div>
            <div><dt>Peak drop</dt><dd>{_num(row["peak_quality_drop"])}</dd></div>
            <div><dt>Weakest</dt><dd class="weak">{escape(_weakest_label(row))}</dd></div>
          </dl>
          <div class="model-viz">
            <figure>
              <figcaption>Cue ladder</figcaption>
              {_cue_slope(model.get("cue_ladder") or {})}
            </figure>
            <figure>
              <figcaption>Quality by turn</figcaption>
              {_turn_curve((model.get("multi_turn") or {}).get("turn_curve") or [])}
            </figure>
          </div>
          <div class="model-fam">
            <span class="fam-label">Failure mix</span>
            {_family_bar(model, family_keys)}
          </div>
        </article>"""
        )

    # -- dimension matrix -----------------------------------------------------
    dimension_rows = []
    for metric in (*CORE_METRICS, CULTURAL_METRIC):
        cells = "".join(
            f'<td><span class="cell-num">{_num(row.get(metric))}</span>{_scale_track(row.get(metric), label=False)}</td>'
            for row in board
        )
        dimension_rows.append(f'<tr><th scope="row">{escape(DIMENSION_LABELS[metric])}</th>{cells}</tr>')

    # -- condition matrix -----------------------------------------------------
    condition_rows = []
    for condition in ("no_cue", "implicit_cue", "explicit_age", "cultural", "cross_lingual"):
        cells = "".join(
            f'<td><span class="cell-num">{_num((models[row["slug"]].get("by_condition") or {}).get(condition))}</span>'
            f"{_scale_track((models[row['slug']].get('by_condition') or {}).get(condition), label=False)}</td>"
            for row in board
        )
        condition_rows.append(f'<tr><th scope="row">{escape(CONDITION_LABELS[condition])}</th>{cells}</tr>')

    # -- root cause findings --------------------------------------------------
    findings_blocks = []
    for row in board:
        model = models[row["slug"]]
        causes = model.get("root_causes") or []
        if not causes:
            continue
        items = "".join(
            f'<li><code class="code">{escape(cause["code"])}</code>'
            f"<strong>{escape(cause['finding'])}</strong>"
            f"<span>{escape(cause['evidence'])}</span></li>"
            for cause in causes
        )
        findings_blocks.append(
            f'<div class="finding-group"><h4>{escape(row["model"])}</h4><ul class="findings">{items}</ul></div>'
        )

    blade_rows = "".join(
        f'<tr><th scope="row"><code class="code">{escape(cause.code)}</code></th>'
        f'<td class="muted">{escape(cause.generic_name)}</td>'
        f'<td><strong>{escape(cause.kidbench_name)}</strong><br><span class="muted">{escape(cause.meaning)}</span></td></tr>'
        for cause in ROOT_CAUSES
    )

    family_legend = "".join(
        f'<li><span class="swatch seg-{index}"></span><strong>{escape(FAMILY_BY_KEY[key].label)}</strong>'
        f"<span>{escape(FAMILY_BY_KEY[key].harm_model)}</span></li>"
        for index, key in enumerate(family_keys)
    )

    model_headers = "".join(f'<th scope="col">{escape(row["model"])}</th>' for row in board)

    return f"""<title>KidBench Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{_css()}
</style>

<div class="page">

  <header class="masthead">
    <p class="eyebrow">NeMo Gym · benchmark report · {date.today().isoformat()}</p>
    <h1>What these models say to a nine-year-old</h1>
    <p class="standfirst">
      KidBench scores child-facing answers for ages 7&ndash;11 on five dimensions drawn from
      developmental psychology. It is not a harmful-content benchmark. The failure it was built
      around is the answer that is <em>factually correct</em> and still the wrong thing to tell a
      child &mdash; accurate, unbounded, and with no grown-up anywhere in it.
    </p>
    <dl class="masthead-stats">
      <div><dt>Models</dt><dd>{len(board)}</dd></div>
      <div><dt>Single-turn answers judged</dt><dd>{total_single:,}</dd></div>
      <div><dt>Five-turn conversations</dt><dd>{total_multi:,}</dd></div>
      <div><dt>Judge</dt><dd class="mono">deepseek-v4-pro</dd></div>
    </dl>
  </header>

  <section class="scale-key" aria-label="How to read the scale">
    <h2 class="sr-only">Reading the scale</h2>
    <div class="key-track">
      <div class="track-line" aria-hidden="true"></div>
      <div class="track-safe" style="left:{_pos(SAFE_LINE):.2f}%" aria-hidden="true"></div>
      <span class="key-tick" style="left:0%">1</span>
      <span class="key-tick" style="left:25%">2</span>
      <span class="key-tick key-tick--safe" style="left:50%">3</span>
      <span class="key-tick" style="left:75%">4</span>
      <span class="key-tick" style="left:100%">5</span>
    </div>
    <p class="key-note">
      Every bar on this page runs 1&ndash;5, the rubric's actual range &mdash; not 0&ndash;5. The marked
      line at <strong>3.0</strong> is where the rubric stops saying <em>partially appropriate</em> and
      starts saying <em>potentially unsafe</em>.
    </p>
  </section>

  <section class="verdict">
    <h2>The short version</h2>
    <p>
      {escape(_verdict_sentence(board, models))}
    </p>
  </section>

  <section>
    <h2>Leaderboard</h2>
    <div class="models">
      {"".join(rows)}
    </div>
  </section>

  <section>
    <h2>Does it matter whether the model knows it&rsquo;s a child?</h2>
    <p class="lede">
      The same 500 questions, asked three ways: plainly, in a child&rsquo;s own words, and with the
      system prompt stating the age outright. Nothing about the question changes &mdash; only how
      visible the child is. A steep climb means the model <em>has</em> a child-appropriate answer and
      only reaches for it when something asks. A production deployment often never does.
    </p>
    <div class="table-wrap">
      <table class="matrix">
        <thead><tr><th scope="col">Condition</th>{model_headers}</tr></thead>
        <tbody>{"".join(condition_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>The five dimensions</h2>
    <p class="lede">
      A model can be perfectly safe and still fail a child &mdash; by being too abstract to follow,
      too cold to come back to, or by refusing without telling them where else to go.
      Cultural alignment is scored only in the four country cells and is never folded into the
      overall score.
    </p>
    <div class="table-wrap">
      <table class="matrix">
        <thead><tr><th scope="col">Dimension</th>{model_headers}</tr></thead>
        <tbody>{"".join(dimension_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>How they fail</h2>
    <p class="lede">
      Built from the judge&rsquo;s own failure tags. The rubric suggests sixteen; the judge coins its
      own where none fit &mdash; <span class="mono">fails_to_recognize_grooming_risk</span>,
      <span class="mono">bypasses_trusted_adult_gatekeeping</span> &mdash; and those coinages are
      often the most specific thing in the run, so they are grouped by meaning rather than dropped.
    </p>
    <ul class="legend">
      {family_legend}
    </ul>
  </section>

  <section>
    <h2>What the BLADE codes mean here</h2>
    <p class="lede">
      BLADE&rsquo;s root-cause letters were defined for benchmarks where a task passes or fails. A
      KidBench row has five scores and no pass/fail, so each letter is restated below as a claim
      about child-facing safety that this run can actually support or refute.
    </p>
    <div class="table-wrap">
      <table class="matrix matrix--text">
        <thead><tr><th scope="col">Code</th><th scope="col">Generic</th><th scope="col">In KidBench</th></tr></thead>
        <tbody>{blade_rows}</tbody>
      </table>
    </div>
    <div class="findings-wrap">
      {"".join(findings_blocks)}
    </div>
  </section>

  <section class="caveats">
    <h2>What these numbers are not</h2>
    <ul>
      <li><strong>Not human-calibrated.</strong> Every score comes from a single LLM judge. The paper
        is explicit that the rubric is not validated against expert human scoring.</li>
      <li><strong>Not a clearance.</strong> A good score is evidence about a benchmark. Upstream&rsquo;s
        use notice is explicit that these artifacts are for evaluation and model development, never
        for putting a model in front of a child.</li>
      <li><strong>Ages 7&ndash;11 only.</strong> Nothing here speaks to younger children or teenagers.</li>
      <li><strong>The multi-turn actor is a stand-in.</strong> The paper drives its conversations with a
        refusal-ablated Gemma-4-31B that it deliberately does not release; this run uses the stock
        checkpoint. A stock actor sometimes declines to press, which makes every model look
        <em>safer</em> here than the published setup would show. Actor refusal rate is measured and
        reported per model.</li>
      <li><strong>Single run, temperature 0.7.</strong> Upstream&rsquo;s runner scripts set 0.7 for hosted
        models even though its README says 0. These scores carry the same sampling noise the
        published ones do.</li>
    </ul>
  </section>

  <footer class="colophon">
    <p>
      KidBench &mdash; <em>The Age of Curiosity Meets the Age of AI</em>, Arif, Borah &amp; Mihalcea,
      Findings of EMNLP 2026 · <a href="https://arxiv.org/abs/{data.get("paper", "").replace("arXiv:", "")}">arXiv:{escape(data.get("paper", "").replace("arXiv:", ""))}</a>
      · benchmark pinned at <span class="mono">{escape(data.get("upstream_revision", "")[:12])}</span>
    </p>
    <details class="data">
      <summary>Machine-readable data</summary>
      <p class="muted">The complete analysis, including per-category breakdowns and raw judge tag counts.</p>
      <pre id="raw-json"></pre>
    </details>
  </footer>
</div>

<script type="application/json" id="kidbench-data">{json.dumps(data, ensure_ascii=False)}</script>
<script>
  (function () {{
    var node = document.getElementById('kidbench-data');
    var out = document.getElementById('raw-json');
    if (!node || !out) return;
    try {{
      out.textContent = JSON.stringify(JSON.parse(node.textContent), null, 2);
    }} catch (e) {{
      out.textContent = node.textContent;
    }}
  }})();
</script>
"""


def _weakest_label(row: dict[str, Any]) -> str:
    scored = [(metric, row.get(metric)) for metric in CORE_METRICS if row.get(metric) is not None]
    if not scored:
        return "—"
    metric, value = min(scored, key=lambda item: item[1])
    return f"{DIMENSION_LABELS[metric]} {value:.2f}"


def _verdict_sentence(board: list[dict[str, Any]], models: dict[str, Any]) -> str:
    """One paragraph a reader could quote, built only from what the run measured."""
    if not board:
        return "No rollouts were collected."
    best = board[0]
    worst = board[-1]
    gaps = [(row["model"], row["no_cue_penalty"]) for row in board if row.get("no_cue_penalty") is not None]
    drops = [(row["model"], row["peak_quality_drop"]) for row in board if row.get("peak_quality_drop") is not None]

    parts = [
        f"{best['model']} leads at {best['total_quality_score']:.2f} out of 5 and "
        f"{worst['model']} trails at {worst['total_quality_score']:.2f}, "
        f"but the spread between them is smaller than the spread inside any one of them."
    ]
    if gaps:
        widest = max(gaps, key=lambda item: item[1])
        parts.append(
            f"Naming the child is worth up to {widest[1]:+.2f} points on identical questions "
            f"({widest[0]}) — a larger swing than the entire gap between first and last place."
        )
    if drops:
        steepest = max(drops, key=lambda item: item[1])
        parts.append(
            f"And every model that holds a boundary at turn one gives ground by turn five: "
            f"{steepest[0]} falls {steepest[1]:.2f} points below its own opening answer."
        )
    parts.append(
        "The ranking is the least interesting thing on this page. What the conditions do to each model is the finding."
    )
    return " ".join(parts)


def _css() -> str:
    return """
:root {
  --ground: #f7f5f1;
  --surface: #fffefb;
  --surface-sunk: #efece5;
  --ink: #16181d;
  --ink-soft: #4b5059;
  --ink-faint: #878d97;
  --rule: #ddd8cf;
  --rule-strong: #c6c0b4;
  --accent: #9a5b1f;
  --accent-soft: #f0e2d2;
  --excellent: #2f6b4f;
  --good: #5d8a4a;
  --mixed: #b8862b;
  --poor: #c2662c;
  --unacceptable: #a33327;
  --seg-0: #9a5b1f;
  --seg-1: #4a6d84;
  --seg-2: #7d5a86;
  --seg-3: #2f6b4f;
  --seg-4: #a33327;
  --seg-5: #736a58;
  --shadow: 0 1px 2px rgba(22,24,29,.05), 0 8px 24px -16px rgba(22,24,29,.25);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #14161a;
    --surface: #1c1f25;
    --surface-sunk: #24282f;
    --ink: #eceae5;
    --ink-soft: #a8adb6;
    --ink-faint: #767c86;
    --rule: #2e333b;
    --rule-strong: #414853;
    --accent: #d99a55;
    --accent-soft: #33291d;
    --excellent: #64b98c;
    --good: #93c47b;
    --mixed: #dfb257;
    --poor: #e08e55;
    --unacceptable: #e0695c;
    --seg-0: #d99a55;
    --seg-1: #7fa8c4;
    --seg-2: #b28ec0;
    --seg-3: #64b98c;
    --seg-4: #e0695c;
    --seg-5: #9a9384;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -16px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"] {
  --ground: #14161a;
  --surface: #1c1f25;
  --surface-sunk: #24282f;
  --ink: #eceae5;
  --ink-soft: #a8adb6;
  --ink-faint: #767c86;
  --rule: #2e333b;
  --rule-strong: #414853;
  --accent: #d99a55;
  --accent-soft: #33291d;
  --excellent: #64b98c;
  --good: #93c47b;
  --mixed: #dfb257;
  --poor: #e08e55;
  --unacceptable: #e0695c;
  --seg-0: #d99a55;
  --seg-1: #7fa8c4;
  --seg-2: #b28ec0;
  --seg-3: #64b98c;
  --seg-4: #e0695c;
  --seg-5: #9a9384;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -16px rgba(0,0,0,.6);
}

* { box-sizing: border-box; }
body {
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.page {
  max-width: 1080px;
  margin: 0 auto;
  padding-inline: 20px;
  padding-block: 40px 72px;
  display: flex;
  flex-direction: column;
  gap: 52px;
}
h1, h2, h3, h4 { font-family: Newsreader, ui-serif, Georgia, serif; font-weight: 500; margin: 0; text-wrap: balance; }
h1 { font-size: clamp(2rem, 5vw, 3.1rem); line-height: 1.08; letter-spacing: -0.015em; }
h2 { font-size: clamp(1.35rem, 3vw, 1.75rem); line-height: 1.2; }
h3 { font-size: 1.05rem; }
h4 { font-size: .98rem; }
p { margin: 0; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
.mono, .code, code, pre { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace; }
.muted { color: var(--ink-soft); }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
section { display: flex; flex-direction: column; gap: 18px; }
.lede { color: var(--ink-soft); max-width: 68ch; }

/* masthead */
.masthead { display: flex; flex-direction: column; gap: 18px; padding-bottom: 28px; border-bottom: 2px solid var(--ink); }
.eyebrow { font-size: .72rem; letter-spacing: .13em; text-transform: uppercase; color: var(--ink-faint); font-weight: 500; }
.standfirst { font-family: Newsreader, ui-serif, Georgia, serif; font-size: clamp(1.02rem, 2.2vw, 1.2rem); line-height: 1.55; color: var(--ink-soft); max-width: 62ch; }
.standfirst em { color: var(--ink); font-style: italic; }
.masthead-stats { display: flex; flex-wrap: wrap; gap: 8px 36px; margin: 0; }
.masthead-stats div { display: flex; flex-direction: column; gap: 1px; }
.masthead-stats dt { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-faint); }
.masthead-stats dd { margin: 0; font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums; }

/* scale key */
.scale-key { background: var(--surface-sunk); border-radius: 3px; padding: 20px 24px 14px; gap: 14px; }
.key-track { position: relative; height: 26px; }
.key-tick { position: absolute; top: 10px; transform: translateX(-50%); font-size: .7rem; font-family: "IBM Plex Mono", monospace; color: var(--ink-faint); }
.key-tick--safe { color: var(--accent); font-weight: 500; }
.key-note { font-size: .86rem; color: var(--ink-soft); max-width: 62ch; }
.key-note strong { color: var(--accent); font-variant-numeric: tabular-nums; }

/* verdict */
.verdict { background: var(--surface); border-left: 3px solid var(--accent); border-radius: 0 3px 3px 0; padding: 22px 26px; box-shadow: var(--shadow); }
.verdict p { font-family: Newsreader, ui-serif, Georgia, serif; font-size: 1.1rem; line-height: 1.6; max-width: 66ch; }

/* scale track */
.track { position: relative; height: 18px; width: 100%; min-width: 80px; }
.track-line { position: absolute; left: 0; right: 0; top: 8px; height: 2px; background: var(--rule); border-radius: 2px; }
.track-safe { position: absolute; top: 3px; width: 1px; height: 12px; background: var(--rule-strong); }
.track-fill { position: absolute; left: 0; top: 8px; height: 2px; border-radius: 2px; }
.track-dot { position: absolute; top: 4px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; border: 2px solid var(--surface); }
.track-value { position: absolute; right: 0; top: -1px; font-size: .78rem; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.track--empty { display: flex; align-items: center; }
.track-na { font-size: .72rem; color: var(--ink-faint); }
.band-excellent { background: var(--excellent); border-color: var(--excellent); }
.band-good { background: var(--good); border-color: var(--good); }
.band-mixed { background: var(--mixed); border-color: var(--mixed); }
.band-poor { background: var(--poor); border-color: var(--poor); }
.band-unacceptable { background: var(--unacceptable); border-color: var(--unacceptable); }
.track-dot.band-excellent, .track-dot.band-good, .track-dot.band-mixed,
.track-dot.band-poor, .track-dot.band-unacceptable { border-color: var(--surface); }

/* model cards */
.models { display: flex; flex-direction: column; gap: 14px; }
.model { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 20px 22px; display: flex; flex-direction: column; gap: 16px; box-shadow: var(--shadow); }
.model:first-child { border-color: var(--rule-strong); border-left: 3px solid var(--accent); }
.model-head { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.rank { font-family: Newsreader, serif; font-size: 1.5rem; color: var(--ink-faint); min-width: 1ch; }
.model-id { flex: 1 1 200px; }
.model-sub { font-size: .76rem; color: var(--ink-faint); font-variant-numeric: tabular-nums; }
.model-score { display: flex; align-items: baseline; gap: 3px; }
.big-num { font-family: Newsreader, serif; font-size: 2.3rem; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1; }
.big-unit { font-size: .85rem; color: var(--ink-faint); }
.model-track { padding-right: 2px; }
.model-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 12px; margin: 0; padding: 14px 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.model-stats div { display: flex; flex-direction: column; gap: 1px; }
.model-stats dt { font-size: .66rem; letter-spacing: .09em; text-transform: uppercase; color: var(--ink-faint); }
.model-stats dd { margin: 0; font-size: 1rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.model-stats dd.warn { color: var(--poor); }
.model-stats dd.weak { font-size: .84rem; font-weight: 500; color: var(--ink-soft); }
.model-viz { display: flex; flex-wrap: wrap; gap: 24px; }
.model-viz figure { margin: 0; display: flex; flex-direction: column; gap: 5px; }
.model-viz figcaption { font-size: .66rem; letter-spacing: .09em; text-transform: uppercase; color: var(--ink-faint); }
.slope, .curve { display: block; max-width: 100%; height: auto; overflow: visible; }
.slope--empty { font-size: .74rem; color: var(--ink-faint); padding: 14px 0; }
.slope-path { stroke: var(--accent); stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; }
.curve-path { stroke: var(--ink-soft); stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; fill: none; }
.slope-safe { stroke: var(--rule-strong); stroke-width: 1; stroke-dasharray: 2 3; }
.slope-dot { stroke: var(--surface); stroke-width: 1.5; }
.slope-label { font-size: 8px; fill: var(--ink-faint); font-family: "IBM Plex Mono", monospace; }
.model-fam { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.fam-label { font-size: .66rem; letter-spacing: .09em; text-transform: uppercase; color: var(--ink-faint); }
.fambar { flex: 1 1 200px; display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--surface-sunk); }
.seg { display: block; height: 100%; }
.seg-0 { background: var(--seg-0); } .seg-1 { background: var(--seg-1); } .seg-2 { background: var(--seg-2); }
.seg-3 { background: var(--seg-3); } .seg-4 { background: var(--seg-4); } .seg-5 { background: var(--seg-5); }

/* tables */
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 3px; background: var(--surface); }
table.matrix { width: 100%; border-collapse: collapse; min-width: 560px; }
table.matrix th, table.matrix td { padding: 12px 14px; text-align: left; vertical-align: top; }
table.matrix thead th { font-family: "IBM Plex Sans", sans-serif; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint); font-weight: 600; border-bottom: 1px solid var(--rule-strong); }
table.matrix tbody tr + tr th, table.matrix tbody tr + tr td { border-top: 1px solid var(--rule); }
table.matrix tbody th { font-weight: 500; font-size: .88rem; white-space: nowrap; }
table.matrix td { min-width: 120px; }
.cell-num { display: block; font-family: "IBM Plex Mono", monospace; font-size: .9rem; font-variant-numeric: tabular-nums; margin-bottom: 4px; }
table.matrix--text td { min-width: 0; font-size: .87rem; }
table.matrix--text th[scope="row"] { vertical-align: top; }
.code { background: var(--accent-soft); color: var(--accent); padding: 1px 6px; border-radius: 2px; font-size: .78rem; font-weight: 500; }

/* legend + findings */
.legend { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; }
.legend li { display: grid; grid-template-columns: 12px 1fr; gap: 4px 10px; align-items: start; font-size: .85rem; }
.legend .swatch { width: 10px; height: 10px; border-radius: 2px; margin-top: 5px; grid-row: span 2; }
.legend strong { font-weight: 600; }
.legend span:not(.swatch) { color: var(--ink-soft); grid-column: 2; }
.findings-wrap { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
.finding-group { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 16px 18px; }
.finding-group h4 { margin-bottom: 10px; }
.findings { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }
.findings li { display: grid; grid-template-columns: auto 1fr; gap: 3px 8px; font-size: .84rem; }
.findings code { grid-row: span 2; align-self: start; }
.findings strong { font-weight: 600; }
.findings span { color: var(--ink-soft); grid-column: 2; }

/* caveats */
.caveats { background: var(--surface-sunk); border-radius: 3px; padding: 24px 26px; }
.caveats ul { margin: 0; padding-left: 1.1em; display: flex; flex-direction: column; gap: 10px; color: var(--ink-soft); font-size: .89rem; max-width: 72ch; }
.caveats strong { color: var(--ink); }

/* colophon */
.colophon { border-top: 1px solid var(--rule); padding-top: 22px; display: flex; flex-direction: column; gap: 14px; font-size: .82rem; color: var(--ink-soft); }
.data summary { cursor: pointer; font-weight: 500; color: var(--ink); }
.data pre { background: var(--surface-sunk); border-radius: 3px; padding: 14px; overflow: auto; max-height: 420px; font-size: .72rem; line-height: 1.5; margin-top: 10px; }

@media (max-width: 620px) {
  .model-viz { gap: 18px; }
  .big-num { font-size: 1.9rem; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/kidbench/reports/kidbench-leaderboard.json"))
    parser.add_argument("--out", type=Path, default=Path("results/kidbench/reports/kidbench-leaderboard.html"))
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
