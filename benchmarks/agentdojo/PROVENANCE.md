# AgentDojo and AgentDyn provenance

The official AgentDojo dependency is pinned to tag `v0.1.35`, commit
`a75aba7631d3ca5fb7ab938965c97ead2f9ff84b` (2025-10-27). The adapter uses benchmark version `v1.2.2` and the
four official suites: `banking`, `slack`, `travel`, and `workspace`.

The AgentDyn repository inspected for the planned extension is pinned to
`5353cf7615b135cace8d07c8f12dac53a16b6db3` (2026-05-19). AgentDyn is not a Git fork with AgentDojo ancestry: its
history begins with an AgentDyn-only README, followed by a copied code snapshot. Its package metadata retains the
name `agentdojo` and version `0.1.35`, which identifies the corresponding official AgentDojo release but does not
make the distributions interchangeable.

Comparing AgentDyn's first code-bearing commit (`de473e58748dc19f9ac2c392fa650185f35b0d83`) with official AgentDojo
`v0.1.35` shows that the copied snapshot had already changed generic framework behavior in:

- `agent_pipeline/agent_pipeline.py`
- `agent_pipeline/llms/openai_llm.py`
- `agent_pipeline/pi_detector.py`
- `benchmark.py`
- `data/system_messages.yaml`
- `models.py`
- `task_suite/load_suites.py`
- `default_suites/v1/tools/types.py`

It also added the AgentDyn suites and dynamic tool implementations. Later AgentDyn commits add and repair CaMeL,
Progent, and DRIFT. Consequently, official AgentDojo and AgentDyn must be treated as separate pinned backends behind
one adapter contract, not installed together as if one were an additive Python plugin: both install the same
top-level `agentdojo` package.
