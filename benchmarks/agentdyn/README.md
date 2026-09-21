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

## Reproducing the PromptGuard2 cells without Meta access

`prompt_guard_2_detector` loads `meta-llama/Llama-Prompt-Guard-2-86M`, which is **gated**: running those cells needs
an `HF_TOKEN` for an account Meta has granted. Without one the download fails outright rather than silently.

There is a free ungated alternative, and it is not a substitute of unknown quality -- it is the same weights:

```text
project-free-llama/Llama-Prompt-Guard-2-86M
revision 43882965632dcb7b20299530f6436ac759d07fd9
```

All five files, `model.safetensors` included, hash identically to canonical
`meta-llama/Llama-Prompt-Guard-2-86M@a8ded8e697ce7c355e395a0df51f94adb4a2fd27`. Reproduce that comparison rather than
taking it on trust:

```bash
HF_TOKEN=... python benchmarks/agentdyn/check_prompt_guard_provenance.py   # exits 0 when every file matches
```

To use the mirror, point the two config keys at it:

```yaml
prompt_guard_2_model_name: project-free-llama/Llama-Prompt-Guard-2-86M
prompt_guard_2_model_revision: 43882965632dcb7b20299530f6436ac759d07fd9
```

Each rollout records the repository and revision that classified it, so a run against the mirror is self-describing
rather than indistinguishable from a canonical one. Some of this campaign's PromptGuard2 rows were collected that
way, before access was granted; see [`DEFENSE-VALIDATION.md`](DEFENSE-VALIDATION.md).

## Running the grid

Undefended baselines for all four models are in [`BASELINE-VALIDATION.md`](BASELINE-VALIDATION.md). The defense grid
runs the same 620 selectors under each treatment:

```bash
bash benchmarks/agentdyn/launch_defense_grid.sh              # every cell, one process each
SHARDS=3 DEFENSES=drift bash benchmarks/agentdyn/launch_defense_grid.sh kimi qwen
LIMIT=2 OUT_SUFFIX=canary DEFENSES=drift \
    bash benchmarks/agentdyn/run_defense_matrix.sh supervl   # smoke one cell
python benchmarks/agentdyn/summarize_defense_matrix.py --json defense-matrix-manifest.json
```

The runner selects a treatment with `default_defense` on the agent server rather than by rewriting the rows, so every
cell scores the byte-identical selector file the baselines used. Every stage passes `--resume`, and the launcher skips
any cell a live runner already owns -- so re-running `launch_defense_grid.sh` is also how you restart whatever died,
and the summarizer's status listing says which cells those are.

Parallelism comes from processes, not from `--concurrency`. The agent server runs rollouts behind
`asyncio.Semaphore(concurrency)` with concurrency 1, so a higher `--concurrency` only queues requests at the agent.
That semaphore is load-bearing: setting up a defense monkeypatches process globals (`TransformersBasedPIDetector`,
`openai.OpenAI`, `os.environ`), and overlapping `mock.patch` scopes inside one process restore each other's values out
of order, which would hand a rollout the unpatched client and point it at the public OpenAI endpoint. So the launcher
gives each cell its own process, and `SHARDS=N` splits one cell's selectors across N more. Sharding is safe because a
cell's score is a pure function of its row set: masked rollouts leave the denominator, then utility and attack success
are averaged over the benign and attacked subsets. DRIFT is the cell that needs it, at roughly fifty policy calls per
rollout against nine undefended.
