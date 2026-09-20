# Defense grid

Date opened: 2026-09-20 CDT

The undefended baselines in [`BASELINE-VALIDATION.md`](BASELINE-VALIDATION.md) are complete for all four models. This
ledger covers the defended arm: every defense over the same 620 selectors, per model.

```text
4 models x 5 defenses x 620 selectors = 12,400 rollouts
```

**This ledger is open.** Cells are recorded here only when they have all 620 rows. Live progress, including which
cells a runner still owns, comes from:

```bash
python benchmarks/agentdyn/summarize_defense_matrix.py --json defense-matrix-manifest.json
```

## Method

Every cell scores the byte-identical selector file the baselines used, `sha256 819443fe...`. The treatment is chosen
with `default_defense` on the agent server rather than by rewriting the rows, so the input never varies across the
grid and the defended numbers are read against the undefended ones on the same matrix.

A masked rollout is an adapter failure, not a secure outcome. Masked rows leave the denominator before utility and
attack success are averaged over the benign and attacked subsets, and both the masked count and the adapter-failure
count are reported per cell. A cell with masked rows is not publishable as-is.

Because that score is a pure function of the row set, a cell may be collected by several processes and merged. DRIFT
is collected this way -- at roughly fifty policy calls per rollout against nine undefended, a single process projects
past forty hours for one cell.

## Deviations from the baseline arm, disclosed

- **Concurrency is not the baselines' 4.** The baseline manifest records concurrency 4, but that number never bound
  anything: the agent server runs rollouts behind `asyncio.Semaphore(1)`, so every rollout in both arms was collected
  one at a time per stack. The grid uses one process per cell, and three per cell for DRIFT. Per-sample isolation and
  deterministic verification are unchanged; only the number of stacks differs.
- **PromptGuard2 uses a mirror.** `project-free-llama/Llama-Prompt-Guard-2-86M` at revision
  `43882965632dcb7b20299530f6436ac759d07fd9`, because Meta's canonical repository is gated behind manual approval and
  this host has no Hugging Face token. Every PromptGuard2 rollout records the repository and revision that classified
  it. `check_prompt_guard_provenance.py` performs the comparison once access exists; until then no canonical claim is
  made.
- **PIGuard is pinned** to `dd78b24e330193a22d2293ac66922dd4f982f563` rather than resolving a moving Hub `main`,
  which matters because upstream loads it with `trust_remote_code=True`.
- **The routed defenses were fixed mid-campaign.** CaMeL, Progent, and DRIFT were reading Gym's `<think>` envelope as
  if it were the model's answer; see the correction in [`DEFENSE-VALIDATION.md`](DEFENSE-VALIDATION.md). Every row in
  this ledger is collected after that fix. Pre-fix rows were discarded rather than merged.

## Results

No cell has reached 620 rows yet. Nothing is recorded here until one does.
