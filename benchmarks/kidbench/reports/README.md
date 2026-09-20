# KIDBench run — 2026-09-20

Results for four models deployed in the Snorkel FDR Modal environment, run against the
full published KIDBench grid: 5,000 single-turn answers and 200 five-turn child-actor
conversations per model.

**Single turn** — 5,000 answers per model, judged by `deepseek-v4-pro`:

| Model | Overall /5 | Unsafe rate | Cue gap |
|---|---|---|---|
| Kimi K3 | 4.28 | 13.3% | +1.67 |
| Qwen3.5 122B-A10B | 4.22 | 12.3% | +1.72 |
| Nemotron 3.5 Super VL | 4.09 | 17.1% | +1.65 |
| Nemotron 3 Ultra 550B | 3.84 | 25.3% | +1.93 |

**Multi turn** — 200 five-turn conversations per model, judged by `deepseek-v4-flash`,
ranked on the share of conversations that go unsafe at any turn:

| Model | Ever unsafe | Mean quality | Slope | Peak drop |
|---|---|---|---|---|
| Nemotron 3.5 Super VL | 7.5% | 4.68 | 0.01 | 0.12 |
| Qwen3.5 122B-A10B | 10.0% | 4.66 | 0.05 | 0.28 |
| Kimi K3 | 10.1% | 4.71 | 0.01 | 0.24 |
| Nemotron 3 Ultra 550B | 24.0% | 4.39 | 0.07 | 0.41 |

The two tracks use different judges — upstream's own split, reproduced rather than
normalized — so they are **not comparable to each other**, and they rank differently:
Super VL is third on single turn and first on multi turn. Compare models within a track.

Multi-turn was collected with a refusal-ablated actor
(`wangzhang/gemma-4-31B-it-abliterated` at `d7431183`), measured at **0.0000% actor
refusals across all 800 conversations**. `actor-comparison.md` measures what that choice
was worth against the stock actor. Benchmark pinned at `8789d50e`.

## Files

| File | Contents |
|---|---|
| `kidbench-leaderboard.md` | Cross-model comparison |
| `kidbench-leaderboard.json` | The same data, machine-readable, plus per-model detail |
| `<model>.blade.md` | Full BLADE analysis per model |
| `actor-comparison.md` | What the child actor's willingness to press is worth |

Regenerate with:

```bash
python -m benchmarks.kidbench.reporting.cli --results-dir results/kidbench --out-dir results/kidbench/reports
```

Metric definitions, the adapter-defined threshold, and every deviation from the published
protocol are in [../METRICS.md](../METRICS.md). The rollout JSONL these were built from is
not committed — it is large and carries third-party model output.
