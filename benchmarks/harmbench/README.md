# HarmBench in NeMo Gym

The calibrated baseline is the DirectRequest text-classifier protocol (159 standard + 81 contextual behaviors).
The expansion has an independently reconciled 320-row held-out-test run for HumanJailbreaks on text-only Nemotron 3 Ultra and
MultiModalDirectRequest and MultiModalRenderText on Nemotron 3.5 Super VL. It also adds the separate copyright
MinHash verifier and a provenance-checked bridge for other pinned upstream attack classes. **A catalog entry or
prepared dataset is not a completed evaluation.** Method-by-method status and execution requirements are in
[`METHODS.md`](METHODS.md).

Full text-corpus execution means upstream's default 400-row
`harmbench_behaviors_text_all.csv` (320 held-out test + 80 validation). The
400-row launchers are hash- and cardinality-locked; historical 320-row results
remain valid test-split evidence but are not labeled full-corpus runs.

- Environment and run instructions: [`resources_servers/harmbench/README.md`](../../resources_servers/harmbench/README.md)
- Metrics, provenance, calibration, and reading guide: [`METRICS.md`](METRICS.md)
- Paper metadata and fetch script: [`paper/PAPER.md`](paper/PAPER.md), `fetch_paper.py`
- Calibration against upstream's classifier path: `calibrate.py`
- Run packages, BLADE exports, and model-card reports: `reporting/`
- Pinned upstream method launcher and Gym materializer: `run_upstream_generation.py`, `prepare_generated.py`
- Method-specific, payload-free evidence reconciliation and NVIDIA one-page rendering: `report_method_run.py`,
  `render_nvidia_method_report.py`
