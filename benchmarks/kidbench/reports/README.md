# KIDBench run — 2026-09-20

Results for four models deployed in the Snorkel FDR Modal environment, run against the
full published KIDBench grid: 5,000 single-turn answers and 200 five-turn child-actor
conversations per model.

| Model | Overall /5 | Unsafe rate | Cue gap | Peak drop |
|---|---|---|---|---|
| Kimi K3 | 4.28 | 13.3% | +1.67 | 0.24 |
| Qwen3.5 122B-A10B | 4.22 | 12.3% | +1.72 | 0.32 |
| Nemotron 3.5 Super VL | 4.09 | 17.1% | +1.65 | 0.17 |
| Nemotron 3 Ultra 550B | 3.84 | 25.3% | +1.93 | 0.35 |

Judges: `deepseek-v4-pro` (single turn) and `deepseek-v4-flash` (multi turn), matching
upstream's own per-track split. Benchmark pinned at `8789d50e`.

## Files

| File | Contents |
|---|---|
| `kidbench-leaderboard.md` | Cross-model comparison |
| `kidbench-leaderboard.json` | The same data, machine-readable, plus per-model detail |
| `<model>.blade.md` | Full BLADE analysis per model |

Regenerate with:

```bash
python -m benchmarks.kidbench.reporting.cli --results-dir results/kidbench --out-dir results/kidbench/reports
```

Metric definitions, the adapter-defined threshold, and every deviation from the published
protocol are in [../METRICS.md](../METRICS.md). The rollout JSONL these were built from is
not committed — it is large and carries third-party model output.
