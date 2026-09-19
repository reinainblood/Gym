# AgentDojo agent adapter

This external-benchmark agent runs the official AgentDojo orchestration and verifiers while routing every policy
model call through NeMo Gym's configured model server. The official `v0.1.35` tag is pinned by commit in
`pyproject.toml`.

AgentDojo remains authoritative for suite loading, dynamic environment state, tool execution, attack placement,
and utility/security verification. NeMo Gym owns model routing, rollout identity, result serialization, token
capture, and aggregate metrics.

The initial adapter intentionally supports only the four official AgentDojo suites: `banking`, `slack`, `travel`,
and `workspace`. AgentDyn will be added as a separate backend behind the same request/result contract after
upstream AgentDojo parity is demonstrated.

The upstream `important_instructions` attack derives a human-readable victim model family from `pipeline.name`.
`attack_model_alias` makes that input explicit. Its default, `local`, reproduces AgentDojo's own generic local-model
route; set it to an upstream-recognized model id when reproducing a named hosted-model result.

## Result contract

Every successful rollout reports `utility`, `security`, `attack_success`, `reward_utility`, and `reward_security`.
The scalar training reward is `reward_utility * reward_security`. Adapter and infrastructure failures are returned
with `mask_sample=true`; they are not counted as successful defenses.

AgentDojo's internal injection verifier returns true when the attacker goal succeeds despite calling that result
`security`. The Gym adapter normalizes this explicitly: `attack_success` preserves the upstream meaning and
`security = not attack_success`.

## Local checks

```bash
gym env test +entrypoint=responses_api_agents/agentdojo_agent
```

The committed example dataset contains one clean banking task and the matching user task paired with one injection
task. A live parity run is deliberately a separate step: first run the same pair with upstream AgentDojo, then run
it through Gym against the same model and compare the trajectories and two verifier decisions.
