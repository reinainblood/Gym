# HarmBench monitoring handoff — 2026-09-21

This handoff is deliberately operational and payload-free. The adversarial method adapters are already written. The next
agent should monitor, resume exact missing indexes, run existing finalizers and scorers, and preserve evidence; it should not
redesign GCG, substitute models, lower public budgets, inspect harmful payloads, or publish upstream.

## Git and PR

- Worktree: `/Users/kruge/Documents/ChatGPT/NVIDIA/.claude/worktrees/harmbench-adapter-handoff-6c5cbc`
- Branch: `codex/harmbench-full-attacks`
- Fork PR: `https://github.com/reinainblood/Gym/pull/10`
- Push only to `reinainblood/Gym:codex/harmbench-full-attacks`. Never open or push an NVIDIA upstream PR.

## Binding GCG target policy

- Primary: unreleased `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, version `hf-ea-0e636f7`.
- Comparison baseline: `Qwen/Qwen3.5-122B-A10B`, revision `dc4d348443bc740c68e2d77492492c11606384d5`.
- Excluded from new GCG work and denominators: Nemotron 3 Ultra and Kimi K3. Retain historical artifacts only.
- Both selected checkpoints are complete BF16 safetensors already verified in FDR. Do not download them again.

## Persistent FDR runtime

- App: `harmbench-gcg-super-qwen`
- Deployed app ID at handoff: `ap-FaAhbV7ZzMVBW4isDD3CZ4`
- Results Volume: `harmbench-gcg-super-qwen-results`
- Super weights Volume: `nemotron-3-5-super-vl-ea-09112026`
- Qwen weights Volume: `qwen3-5-122b-a10b-bf16-dc4d3484`
- Active Super canary artifact: `super-gcg-canary-20260921c`
- Active Super function call: `fc-01M3182YVSBTQ96M4TZ8DT25HV`

The deployed image pins public HarmBench commit `8e1604d1171fe8a48d8febecd22f600e462bdcdd`, 500 GCG steps, search width
512, compiled causal-conv1d/Flash Linear Attention/Mamba kernels, and exact verified checkpoints. Prefix-cache reuse is
disabled for wrapper compatibility; candidate-loss microbatching starts at eight. Neither changes public inputs, token-gradient
math, search width, candidate losses, or global argmin.

## Monitoring and continuation

1. Confirm explicit FDR state with `modal app list --env=FDR`, `modal container list --env=FDR`, and bounded logs.
2. The Super canary passes only when the Volume contains a completed one-behavior individual GCG artifact and
   `super-gcg-canary-20260921c/shard-receipts/super-00-of-01.json`. A loaded model or running container is not success.
3. After that receipt exists, spawn the full Super corpus through the deployed `run_super` function with a new immutable artifact
   ID and bounded sharding. Preserve exact source order; the worker skips existing behavior files on resume.
4. Run one matched Qwen canary, then the full Qwen baseline through deployed `run_qwen` under the same public method profile.
5. The `finalize(target, artifact_id)` function is present in the branch but was added after the active canary deployment. After the
   canary ends, redeploy this exact branch to the same FDR app, then call `finalize` only after all 400 individual behaviors exist. It
   merges cases and writes a `generation-receipt.json` accepted by `prepare_generated.py`.
6. Materialize via the existing generated-method bridge, collect target completions, score through the pinned HarmBench classifier,
   calibrate, reconcile exact denominators, build BLADE/report evidence, run tests/pre-commit, commit with DCO, and push PR #10.
7. Stay metadata-only in commentary and logs. Do not print attack strings, targets, generations, credentials, or secrets.

Separately, the existing Qwen MultiModalPGDBlankImage workers and later Patch completion remain governed by the active Volume
`harmbench-qwen35-122b-whitebox-results`; never duplicate active shards or overwrite completed receipts.
