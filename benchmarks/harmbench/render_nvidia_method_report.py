# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a compact, payload-free Snorkel x NVIDIA HarmBench method page."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


THEME = Path(__file__).with_name("report_theme.css")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render(manifest: dict) -> str:
    counts = manifest["counts"]
    metrics = manifest["metrics"]
    provenance = manifest["provenance"]
    model = manifest["model"]
    method = manifest["method"]
    functional = metrics["functional_category"]
    display_model = (
        "Nemotron 3.5 Super VL"
        if "Nemotron-3.5-Super-VL" in model
        else "Nemotron 3 Ultra"
        if "Nemotron-3-Ultra" in model
        else model
    )
    group_name = "semantic" if len(functional) == 1 else "functional"
    displayed_slices = metrics[f"{group_name}_category"]
    slices = "".join(
        "<tr>"
        f"<td>{_escape(name.replace('_', ' ').title())}</td>"
        f"<td>{int(info['behaviors'])}</td>"
        f"<td>{int(info['cases'])}</td>"
        f"<td>{float(info['asr']):.1%}</td>"
        "</tr>"
        for name, info in displayed_slices.items()
    )
    status = manifest["status"]
    success_detail = (
        "Classifier-labeled successes"
        if set(manifest["score_methods"]) == {"harmbench_classifier"}
        else "Classifier or copyright match"
    )
    css = THEME.read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HarmBench {_escape(method)} — {_escape(model)}</title><style>{css}
.page {{ width: 8.5in; min-height: 11in; margin: 0 auto; background: white; box-shadow: 0 0 26px #111c2d18; }}
.header {{ padding: .42in .56in .38in; min-height: 2.25in; }}
.header h1 {{ position: relative; margin: .42in 0 0; font-size: 33px; font-weight: 300; line-height: 1.1; letter-spacing: -.025em; }}
.header h1 span {{ display: block; color: var(--nv-green-soft); }}
.header .sub {{ position: relative; margin-top: 14px; color: #ffffffb5; font-size: 13px; }}
.header .nv-accent-rule {{ position: relative; margin-top: 24px; }}
.body {{ padding: .34in .56in .38in; }}
.stat-grid {{ display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 12px; }}
.stat .label {{ font: 10px "Geist Mono", ui-monospace, monospace; text-transform: uppercase; letter-spacing: .11em; color: #526070; }}
.stat .value {{ display: block; margin: 10px 0 3px; font-size: 28px; font-weight: 300; font-variant-numeric: tabular-nums; }}
.stat.primary .value {{ color: var(--nv-green); font-size: 39px; }}
.stat .detail {{ color: #526070; font-size: 11px; }}
.section {{ margin-top: 26px; }}
.section h2 {{ margin: 0 0 12px; color: var(--nv-heading-blue); font-size: 20px; font-weight: 400; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th {{ text-align: left; color: #526070; font: 10px "Geist Mono", ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; border-bottom: 1px solid #cfd7dc; padding: 7px 6px; }}
td {{ border-bottom: 1px solid #e1e5e7; padding: 7px 6px; font-variant-numeric: tabular-nums; }}
th:nth-child(n+2), td:nth-child(n+2) {{ text-align: right; }}
.integrity {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.integrity .nv-card {{ font-size: 11px; line-height: 1.35; min-height: 74px; }}
.integrity b {{ display: block; color: var(--nv-green); font-size: 15px; margin: 5px 0; }}
.fine {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #d8dde1; font-size: 9px; }}
@media screen {{ body {{ background: #e8ebed; padding: 24px; }} }}
@media print {{ body {{ background: white; }} .page {{ box-shadow: none; }} }}
</style></head><body><main class="page">
<header class="nv-dark-header header"><div class="nv-green-glow"></div>
<div class="nv-brand">SNORKEL AI <strong>x</strong> NVIDIA · SAFETY EVALUATION</div>
<h1>{_escape(display_model)}<span>HarmBench {_escape(method)}</span></h1>
<div class="sub">One method · {counts["behaviors"]} evaluated behaviors · {counts["scored_cases"]} scored attack cases</div>
<div class="nv-accent-rule"></div></header>
<section class="nv-paper-page body">
<div class="stat-grid">
  <div class="nv-card stat primary"><div class="label">Attack success rate</div><span class="value">{metrics["attack_success_rate"]:.1%}</span><div class="detail">Behavior-averaged; lower is better</div></div>
  <div class="nv-card stat"><div class="label">Successful cases</div><span class="value">{counts["successful_cases"]}/{counts["scored_cases"]}</span><div class="detail">{_escape(success_detail)}</div></div>
  <div class="nv-card stat"><div class="label">Run integrity</div><span class="value">{counts["scored_cases"]}/{counts["expected_cases"]}</span><div class="detail">{manifest["health"]["healthy"]} healthy · {counts["failed_cases"]} infra failures</div></div>
</div>
<div class="section"><h2>Observed by behavior type</h2><table><thead><tr><th>{group_name.title()} category</th><th>Behaviors</th><th>Cases</th><th>ASR</th></tr></thead><tbody>{slices}</tbody></table></div>
<div class="section"><h2>Completion and scorer checks</h2><div class="integrity">
  <div class="nv-card">Generation cap<b>{counts["generation_truncated"]} truncated</b>{counts["generation_empty"]} empty</div>
  <div class="nv-card">Classifier input<b>{counts["generation_clipped"]} clipped</b>512 classifier-token limit</div>
  <div class="nv-card">Protocol controls<b>{_escape(status.replace("_", " ").title())}</b>{_escape(manifest["protocol_validation"].replace("_", " "))}</div>
</div></div>
<p class="nv-fine-print fine">Performance: {_escape(model)}; NeMo Gym simple_agent, single turn and one rollout per attack case; HarmBench {_escape(method)}, upstream {_escape(provenance["upstream_revision"])}; {counts["behaviors"]} behavior cohort, {counts["scored_cases"]} cases; ASR is the mean of per-behavior attack success rates. The HarmBench classifier is {_escape(manifest["classifier"]["model"])} at {_escape(manifest["classifier"]["revision"][:12])}; copyright cases, when present, use the pinned MinHash references. {counts["failed_cases"]} infrastructure failures, {counts["missing_cases"]} missing cases. Run {_escape(manifest["run_id"])}; source dataset SHA-256 {_escape(provenance["dataset_sha256"][:16])}…; rollout SHA-256 {_escape(provenance["rollouts_sha256"][:16])}… .</p>
</section></main></body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(manifest), encoding="utf-8")
    print(f"Rendered {_escape(manifest['method'])} one-page HTML report: {args.output}")
