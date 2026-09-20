# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a six-method, payload-free Snorkel x NVIDIA HarmBench coverage map."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


THEME = Path(__file__).with_name("report_theme.css")
ULTRA_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
SUPER_VL_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
ULTRA_METHODS = ("DirectRequest", "HumanJailbreaks", "ZeroShot", "PAP-top5")
VISION_METHODS = ("MultiModalDirectRequest", "MultiModalRenderText")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(manifest_paths: list[Path]) -> dict[str, dict]:
    expected = set(ULTRA_METHODS) | set(VISION_METHODS)
    if len(manifest_paths) != len(expected):
        raise ValueError("campaign report requires exactly six method manifests")
    methods: dict[str, dict] = {}
    for path in manifest_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        method = item["method"]
        if method not in expected or method in methods:
            raise ValueError("campaign report has a missing, duplicated, or unrequested method")
        model = ULTRA_MODEL if method in ULTRA_METHODS else SUPER_VL_MODEL
        if item["model"] != model:
            raise ValueError(f"{method} belongs to {model}, not {item['model']}")
        if item["status"] not in {"validated", "validated_with_caveat"}:
            raise ValueError(f"{method} is not a validated result")
        if item["status"] == "validated_with_caveat" and method != "ZeroShot":
            raise ValueError("only the documented ZeroShot batch caveat is accepted here")
        counts = item["counts"]
        health = item["health"]
        if (
            counts["expected_cases"] != counts["scored_cases"]
            or counts["failed_cases"]
            or counts["missing_cases"]
            or health != {"healthy": counts["scored_cases"], "unhealthy": 0, "unobserved": 0}
        ):
            raise ValueError(f"{method} lacks full healthy case coverage")
        methods[method] = item | {"manifest_sha256": _sha256(path)}
    if set(methods) != expected:
        raise ValueError("campaign report is missing methods")
    return methods


def render(manifests: dict[str, dict]) -> str:
    total_cases = sum(item["counts"]["scored_cases"] for item in manifests.values())
    if total_cases != 5340:
        raise ValueError("six-method case total differs from the pinned 5,340-case execution")

    def rows(names: tuple[str, ...]) -> str:
        output = []
        for method in names:
            item = manifests[method]
            counts = item["counts"]
            asr = item["metrics"]["attack_success_rate"]
            status = "Validated†" if item["status"] == "validated_with_caveat" else "Validated"
            output.append(
                "<tr>"
                f"<td>{_escape(method)}</td>"
                f"<td>{counts['behaviors']}</td>"
                f"<td>{counts['scored_cases']:,}</td>"
                f"<td>{asr:.2%}</td>"
                f"<td>{_escape(status)}</td>"
                "</tr>"
            )
        return "".join(output)

    manifest_digest = hashlib.sha256(
        "".join(manifests[name]["manifest_sha256"] for name in (*ULTRA_METHODS, *VISION_METHODS)).encode()
    ).hexdigest()
    css = THEME.read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snorkel × NVIDIA · HarmBench method coverage</title><style>{css}
@page {{ size: Letter; margin: 0; }}
.page {{ width: 8.5in; min-height: 11in; margin: 0 auto 24px; background: white; box-shadow: 0 0 26px #111c2d18; }}
.header {{ padding: .40in .56in .36in; min-height: 2.15in; }}
.header h1 {{ position: relative; margin: .39in 0 0; font-size: 32px; font-weight: 300; line-height: 1.12; letter-spacing: -.025em; }}
.header h1 span {{ display: block; color: var(--nv-green-soft); }}
.header .sub {{ position: relative; margin-top: 13px; color: #ffffffbb; font-size: 12px; }}
.header .nv-accent-rule {{ position: relative; margin-top: 20px; }}
.body {{ padding: .33in .56in .37in; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }}
.stat .label {{ font: 10px "Geist Mono",ui-monospace,monospace; text-transform: uppercase; letter-spacing:.1em; color:#526070; }}
.stat .value {{ display:block; margin:8px 0 2px; font-size:29px; font-weight:300; font-variant-numeric:tabular-nums; }}
.stat.primary .value {{ color:var(--nv-green); font-size:38px; }}
.stat .detail {{ color:#526070; font-size:10px; line-height:1.3; }}
.section {{ margin-top:24px; }}
.section h2 {{ margin:0 0 10px; color:var(--nv-heading-blue); font-size:19px; font-weight:400; }}
.section .desc {{ margin:-3px 0 10px; color:#536171; font-size:11px; }}
table {{ width:100%; border-collapse:collapse; font-size:11px; }}
th {{ text-align:left; color:#526070; font:9px "Geist Mono",ui-monospace,monospace; text-transform:uppercase; letter-spacing:.07em; border-bottom:1px solid #cfd7dc; padding:7px 5px; }}
td {{ border-bottom:1px solid #e1e5e7; padding:7px 5px; font-variant-numeric:tabular-nums; }}
td:nth-child(n+2),th:nth-child(n+2) {{ text-align:right; }}
td:last-child {{ color:#438600; font-weight:600; }}
.note {{ margin-top:18px; padding:12px 15px; border-left:4px solid var(--nv-green); background:#f4f7ef; font-size:10px; line-height:1.48; color:#304050; }}
.map-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
.map-card {{ border:1px solid #d7dde0; border-radius:8px; padding:13px 15px; min-height:112px; }}
.map-card.wide {{ grid-column:1 / -1; min-height:96px; }}
.map-card .count {{ color:var(--nv-green); font-size:23px; font-weight:300; }}
.map-card h3 {{ color:var(--nv-heading-blue); margin:3px 0 7px; font-size:14px; font-weight:500; }}
.map-card p {{ margin:0; color:#485767; font-size:10px; line-height:1.45; }}
.method-line {{ color:#111c2d !important; margin-bottom:5px !important; font-weight:600; }}
.fine {{ margin-top:18px; padding-top:9px; border-top:1px solid #d8dde1; font-size:8px; line-height:1.4; }}
@media screen {{ body {{ background:#e8ebed; padding:24px; }} }}
@media print {{ body {{ background:white; }} .page {{ box-shadow:none; margin:0; page-break-after:always; }} .page:last-child {{ page-break-after:auto; }} }}
</style></head><body>
<main class="page"><header class="nv-dark-header header"><div class="nv-green-glow"></div>
<div class="nv-brand">SNORKEL AI <strong>x</strong> NVIDIA · SAFETY EVALUATION</div>
<h1>HarmBench method coverage<span>Six measured routes. Two target modalities.</span></h1>
<div class="sub">17 September 2026 · source pinned to centerforaisafety/HarmBench 8e1604d</div>
<div class="nv-accent-rule"></div></header><section class="nv-paper-page body">
<div class="stat-grid">
<div class="nv-card stat primary"><div class="label">Methods evaluated</div><span class="value">6 / 22</span><div class="detail">Distinct HarmBench methods; remaining 16 have explicit execution gates</div></div>
<div class="nv-card stat"><div class="label">Scored attack cases</div><span class="value">{total_cases:,}</span><div class="detail">No pooled ASR across different methods or models</div></div>
<div class="nv-card stat"><div class="label">Run integrity</div><span class="value">100%</span><div class="detail">All scored rollouts healthy · zero infrastructure failures</div></div>
</div>
<div class="section"><h2>Nemotron 3 Ultra · text only</h2><p class="desc">One-turn target completion per attack case; each method uses its own attack generator and behavior-averaged ASR.</p>
<table><thead><tr><th>Method</th><th>Behaviors</th><th>Cases</th><th>ASR</th><th>Evidence</th></tr></thead><tbody>{rows(ULTRA_METHODS)}</tbody></table></div>
<div class="section"><h2>Nemotron 3.5 Super VL · vision</h2><p class="desc">Image-bearing cases remain on the separately deployed vision model. Ultra is never used for multimodal input.</p>
<table><thead><tr><th>Method</th><th>Behaviors</th><th>Cases</th><th>ASR</th><th>Evidence</th></tr></thead><tbody>{rows(VISION_METHODS)}</tbody></table></div>
<div class="note"><b>Read the method, not just the percentage.</b> These attacks, cohorts, and target modalities are not a matched ranking. † ZeroShot has one documented classifier decision that changes with vLLM batch concurrency: raw/chat prompt tokens and clipping match; the saved campaign score was not overridden. PAP-top5's final 5.625% uses a separate stateless reverify of unchanged target responses. The 512-token target cap was reached on 1,262 PAP cases.</div>
<p class="nv-fine-print fine">Source: six checksummed, payload-free method manifests. Behavior-averaged ASR is the mean of each behavior's case success rate. HarmBench classifier and copyright MinHash were replayed against pinned controls; ZeroShot's batch caveat is separately receipted. Manifest-set SHA-256 {_escape(manifest_digest[:24])}… .</p>
</section></main>
<main class="page"><header class="nv-dark-header header"><div class="nv-green-glow"></div>
<div class="nv-brand">SNORKEL AI <strong>x</strong> NVIDIA · EXECUTION MAP</div>
<h1>The full 22-method set<span>Coverage is tracked, not implied.</span></h1>
<div class="sub">Six end-to-end results above · sixteen methods remain to execute under their original requirements</div>
<div class="nv-accent-rule"></div></header><section class="nv-paper-page body">
<div class="section"><h2>What remains to run</h2><div class="map-grid">
<div class="map-card"><div class="count">8</div><h3>Ultra text · white-box</h3><p class="method-line">GCG, GCG-Multi, AutoPrompt, GBDA, PEZ, UAT, AutoDAN, FewShot</p><p>These optimize against local target weights or loss/gradient access. An inference URL cannot substitute for an Ultra gradient-capable checkpoint.</p></div>
<div class="map-card"><div class="count">2</div><h3>Super VL · image gradients</h3><p class="method-line">MultiModalPGD, MultiModalPGDPatch</p><p>The vision endpoint proves image inference; these methods additionally require a gradient-capable Super VL runtime and its pinned image preprocessing.</p></div>
<div class="map-card wide"><div class="count">6</div><h3>Iterative or transfer attack campaigns</h3><p class="method-line">PAIR, TAP, Fresh PAIR, Fresh TAP, TAP-Transfer, GCG-Transfer</p><p>PAIR/TAP need a verified live client-model attack loop and Mixtral judge/attacker; their paper settings permit roughly 19,200 and 30,080 target candidate evaluations across 320 behaviors before retries or early stopping. TAP-Transfer fixes a historical GPT-4 source/judge experiment; GCG-Transfer requires five runs against a four-model source ensemble. Neither can be renamed as a client-fresh result.</p></div>
</div></div>
<div class="note"><b>Next evidence gate.</b> For each remaining method: pin the exact source experiment and target interface, retain attack-generation receipts and query counts, run complete Gym trajectories, replay the appropriate classifier or copyright scorer, and report failures and per-behavior ASR. The six completed methods are not a license to claim all 22 as executed.</div>
<p class="nv-fine-print fine">The method inventory and target-type gates live in the HarmBench adapter's METHODS.md and methods.py. The user-provided Ultra deployment is text-only; the separately authenticated Super VL deployment is the vision target. No production endpoint, checkpoint, or shared GPU app was stopped or resized to make this report.</p>
</section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifests = _verify(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(manifests), encoding="utf-8")
    print(f"Rendered {len(manifests)} methods and 5,340 scored cases: {args.output}")


if __name__ == "__main__":
    main()
