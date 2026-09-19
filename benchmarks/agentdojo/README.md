# AgentDojo

This benchmark wraps official AgentDojo `v0.1.35` / benchmark version `v1.2.2` at the agent-server boundary. See
[`PROVENANCE.md`](PROVENANCE.md) for the AgentDyn fork audit and [`ADAPTER-CONTRACT.md`](ADAPTER-CONTRACT.md) for the
shared AgentDojo-family boundary.

Prepare the complete clean and `important_instructions` task matrix with:

```bash
gym eval prepare --benchmark agentdojo
```

The committed agent example contains one clean `banking` task and one attacked pair. The benchmark remains
unverified until the full benchmark is baselined. The first upstream/Gym live smoke evidence is recorded in
[`LIVE-VALIDATION.md`](LIVE-VALIDATION.md).
