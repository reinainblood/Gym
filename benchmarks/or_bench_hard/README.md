# OR-Bench Hard

Full public OR-Bench Hard evaluation with its toxic safety control:

- 1,319 seemingly toxic but benign hard prompts, where rejection is over-refusal.
- 655 genuinely toxic prompts, where rejection is the desired behavior.
- Ten matched public categories in each split.
- One raw single-turn response per prompt.
- Exact public three-class response-checking prompt and parsing behavior.

The benchmark's primary output is a two-axis result—Hard over-refusal rate and
Toxic rejection rate—not a single scalar. The supplemental balanced accuracy is
reported without replacing that pair.

- Dataset: `orbench-llm/or-bench@6f776f103b6439a6d8180b20f64b55f491223233`
- Official code: `orbench/or-bench@eef58d7522936e88a7d31f8f8506eaa29ef963fc`
- Hard CSV SHA-256: `a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d`
- Toxic CSV SHA-256: `3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057`
- Dataset license: CC BY 4.0; official code license: Apache-2.0.

See [METRICS.md](METRICS.md) for scoring and the full run plan.

```bash
gym eval prepare --benchmark or_bench_hard

OR_BENCH_JUDGE_BASE_URL="$JUDGE_BASE_URL" \
OR_BENCH_JUDGE_API_KEY="$JUDGE_API_KEY" \
OR_BENCH_JUDGE_MODEL="$JUDGE_MODEL" \
gym eval run --benchmark or_bench_hard --split benchmark \
  --model-type inference_provider --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY" \
  --output results/or-bench-hard/rollouts.jsonl \
  +route_failures_to_sidecar=true +observability_enabled=true
```

The released checker used `gpt-4-turbo-2024-04-09` at temperature zero. If that
retired model is unavailable, the selected replacement must be named, source
prompts must remain byte-identical, and a fixed released-response calibration
must be retained alongside the new run.
