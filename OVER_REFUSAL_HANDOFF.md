# Over Refusal Detection — four-model handoff

**Snapshot:** 2026-09-20 CDT
**Branch:** `codex/over-refusal-detection-handoff`
**Implementation:** `resources_servers/over_refusal_detection/`

This branch exists so this environment's implementation and evidence state are
separate from XSTest. It evaluates whether a model inappropriately refuses safe
requests using the environment's LLM-judge compliance labels.

## Evidence status

The implementation, focused tests, config variants, and example fixtures are
in this checkout. I searched the local NVIDIA checkout and sibling worktrees,
then inspected the FDR app and volume inventory on 2026-09-20. **No production
four-model run JSONL, metrics file, or result archive was located.**

The committed `data/example*` files are fixtures. They are not evidence that
the target matrix ran. Do not report a benchmark score until a result package
records all four model IDs, its input set, judge route, denominator, and failure
sidecar.

## Required target matrix

1. `moonshotai/Kimi-K3`
2. `Qwen/Qwen3.5-122B-A10B-FP8`
3. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
4. `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`

## Restart checklist

1. Pin and record the input dataset revision and row count.
2. Record an authenticated `/v1/models` receipt for the exact served model.
3. Run a small model-and-judge smoke and inspect verdict parsing.
4. Collect all planned rows. Keep provider, judge, and infrastructure errors in
   separate sidecars rather than silently retrying them into the score.
5. Save raw outputs, normalized/verdict rows, aggregate metrics, resolved
   config, and a machine-readable manifest under a non-ignored handoff artifact
   directory before reporting.

New Modal work is FDR only. Never commit endpoint credentials, API keys, or
private judge-access material.
