# AgentDyn

AgentDyn is integrated as a separate pinned backend behind the shared AgentDojo-family adapter contract. The primary
dataset contains 60 clean tasks and 560 attacked task pairs across `shopping`, `github`, and `dailylife` only.

```bash
gym eval prepare --benchmark agentdyn
```

Official AgentDojo's `banking`, `slack`, `travel`, and `workspace` suites are maintained as a separate control
benchmark and are never averaged into AgentDyn's headline metrics.

The upstream revision is pinned to `5353cf7615b135cace8d07c8f12dac53a16b6db3`. Defense-specific configs select
one of PromptGuard2, PIGuard, CaMeL, Progent, or DRIFT as a pipeline treatment over the same task matrix. They remain
unverified until their auxiliary-model routing and upstream parity are individually exercised.

The first undefended live smoke receipt is recorded in [`LIVE-VALIDATION.md`](LIVE-VALIDATION.md).
Per-defense runtime status is tracked in [`DEFENSE-VALIDATION.md`](DEFENSE-VALIDATION.md).

## Running the grid

Undefended baselines for all four models are in [`BASELINE-VALIDATION.md`](BASELINE-VALIDATION.md). The defense grid
runs the same 620 selectors under each treatment:

```bash
bash benchmarks/agentdyn/run_defense_matrix.sh ultra      # one model, all five defenses
LIMIT=2 OUT_SUFFIX=canary DEFENSES=drift \
    bash benchmarks/agentdyn/run_defense_matrix.sh supervl   # smoke one cell
python benchmarks/agentdyn/summarize_defense_matrix.py --json defense-matrix-manifest.json
```

The runner selects a treatment with `default_defense` on the agent server rather than by rewriting the rows, so every
cell scores the byte-identical selector file the baselines used. Each model key owns a fixed head port and port block,
so one invocation per model can run in parallel; every stage passes `--resume`.
