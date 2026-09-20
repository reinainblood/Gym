# XSTest — four-model handoff

**Snapshot:** 2026-09-20 CDT
**Branch:** `codex/xstest-handoff`
**Benchmark wrapper:** `benchmarks/xstest/`
**Resources server:** `resources_servers/xstest/`

This branch makes XSTest's evaluation path independently recoverable. XSTest
is the over-refusal calibration benchmark: the central question is whether a
model refuses benign prompts too aggressively. It is distinct from the
general-purpose `over_refusal_detection` environment and must not share a
pooled denominator with it.

## Evidence status

The benchmark preparation script, config, scorer, prompt templates, focused
tests, and example fixtures are present. I searched the local NVIDIA checkout
and sibling worktrees, then inspected the FDR app and volume inventory on
2026-09-20. **No production four-model XSTest result JSONL, metrics file, or
result archive was located.**

The files under `resources_servers/xstest/data/example*` are examples only.
Do not say XSTest ran last night unless the actual four-model outputs are
recovered and reconciled.

## Required target matrix

1. `moonshotai/Kimi-K3`
2. `Qwen/Qwen3.5-122B-A10B-FP8`
3. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
4. `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`

## Correct recovery or run sequence

```bash
gym eval prepare --benchmark xstest
gym eval run --benchmark xstest --output <controlled-result-path> --resume
```

Supply the exact model server/endpoint configuration externally; do not put
credentials in Git. Before a full run, obtain the model receipt, inspect a
small smoke for verdict-parser behavior, and write raw outputs plus metrics and
failure sidecars into a tracked collaboration artifact directory.

New Modal work is FDR only. Preserve the judge model/version and prompt
templates with every result, because a judge change can alter refusal metrics.
