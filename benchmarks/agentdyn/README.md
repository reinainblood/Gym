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
