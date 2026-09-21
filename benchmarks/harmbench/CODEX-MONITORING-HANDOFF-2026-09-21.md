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

## Codex takeover update — 2026-09-21 05:02 CDT

- Super canary `super-gcg-canary-20260921c` passed. The original call produced the valid upstream nested
  one-behavior artifact but the old wrapper counted a flat filename and ended with `GCG shard saved 0/1`.
  Commits `fc1acb8` and `fe69585` fixed the nested path and added a pre-model-load completeness short-circuit.
  Repair call `fc-01M31PJBPVP065CRB3ZC02EHS1` reused the existing artifact without rerunning optimization and
  persisted `shard-receipts/super-00-of-01.json`.
- Full Super campaign artifact: `super-gcg-full-20260921a`. Exactly two source-order shards are active:
  shard 0 `fc-01M31PKEVXNAFW3ZG082K5FQ3D`; shard 1 `fc-01M31PKEZX35EG5DF0RDMJGQ76`.
  Do not launch another campaign or duplicate either shard while its call/container is live. Resume only missing
  nested behavior artifacts with the same artifact ID and shard indexes after terminal calls and zero attack workers.
- Qwen MultiModalPGD now has complete 110/110 canonical 512-token scoring and BLADE evidence. The long-running
  MultiModalPGDBlankImage campaign remains active under app `ap-07zo5VR8b4cd4UHMb6VHBM`; Patch remains incomplete
  and must not be resumed until active BlankImage workers drain and exact missing indexes are reconciled.
- Before any BlankImage or Patch resume after worker drain, redeploy the current branch to the same white-box app. The
  resume path now skips an existing index only after validating its case receipt, paired image and hash, exact method and
  checkpoint identity, public hyperparameters, generation hash, step accounting, and optimization-checkpoint files. A
  present-but-stale or truncated JSON is not completion evidence and must fail closed rather than suppress repair.
- The separate 512-token completion runtime now follows the same rule: existing completions are skipped only after parent
  receipt/image hashes, BF16 checkpoint and processor identities, prompt hash, deterministic sampling, token cap/count,
  finish reason, and generation hash validate. It writes new completion receipts atomically. Canonical scoring explicitly
  rehashes the sibling parent receipts and rejects any parent/canonical filename or provenance drift before classification.
- After attack workers drain and the current branch is redeployed, call the white-box app's `finalize_method` for a method
  only at 110/110. It revalidates all case/image/checkpoint chains plus the gradient/checkpoint receipt and writes
  `attack-manifest.json`. The separate completion app now refuses to start canonical or replicate completion without
  rehashing that exact 110-case manifest and its files.
- After all 110 canonical 512-token completions exist, call the completion app's own `finalize_method`. It requires exact
  parent/canonical filename parity, revalidates every completion against its parent and attack manifest, enforces one
  checkpoint/processor/sampling identity, and writes `completion-manifest-512.json`. The scorer refuses to run until that
  manifest and all 110 completion hashes pass readback.
- Canonical scoring binds its payload-free receipt to the completion-manifest SHA-256, recomputes all case-level counts and
  summary metrics before reuse, rejects duplicate indexes/behaviors or unexpected classifier fields, and writes the score
  receipt atomically. An existing `canonical-512.json` is not reusable merely because it has a summary.
- BLADE requires that modern score provenance for BlankImage and Patch. The completed PGD receipt predates the new
  completion-manifest field and is accepted only as an explicit legacy `MultiModalPGD` format backed by its independently
  checksummed and remotely read-back bundle; partial modern provenance is rejected for every method.
- The served Super endpoint `ap-Y6sMOWq4rnrM5EU6efe8Zf` was reverified at three live instances and must not be
  confused with or scaled down for the separate GCG runtime.

## Qwen GCG preflight artifact — 2026-09-21 05:24 CDT

- Volume prefix `qwen-gcg-canary-20260921a` already exists from an earlier incomplete attempt. At this audit it
  contained only runtime configuration and one 349-byte operational log last modified at 00:24:53 CDT. It had no
  nested behavior `test_cases.json`, no shard receipt, no completion marker, and no live Qwen GCG container.
- This prefix is **not** a passed canary. Do not duplicate or resume it while the full Super campaign is active.
  After Super reaches 400/400, both Super calls are terminal, its two receipts exist, and finalization succeeds,
  recheck ownership and live containers, then resume this same immutable Qwen canary prefix through `run_qwen`.

## Exact Qwen BF16 completion stage

- The matched Qwen GCG result must not use the separately served FP8 endpoint for target completions. The branch now
  contains `operations/qwen/modal_qwen_gcg_completion.py`, a separate resumable FDR runtime bound to the same exact
  BF16 Volume and revision used by GCG optimization. It reproduces upstream HarmBench's default tokenizer chat-template
  branch and 512-token deterministic completion profile.
- Do not deploy or call this runtime before the Qwen full GCG artifact is finalized. Then launch two source-order
  completion shards, require 400 validated individual receipts and both completion-shard receipts, call its `finalize`
  function, and require `target-completions/qwen-bf16/target-completion-receipt.json` before classifier scoring.
- The completion runner returns only indexes, counts, statuses, and call IDs. Its private Volume receipts contain model
  generations needed by the scorer; never print or download their payload fields into commentary or logs. A bounded
  `limit` call is a payload-producing canary but intentionally writes no canonical completion-shard receipt; only an
  unbounded full shard may satisfy the finalizer's shard evidence gate.
- After the completion finalizer passes, deploy `operations/qwen/modal_qwen_gcg_score.py` in FDR and call its `score`
  function once. It rehashes the 400 private completion receipts, routes ordinary behaviors through two serial passes of
  the pinned raw HarmBench classifier, and routes `hash_check` book/lyrics behaviors through the pinned upstream MinHash
  references after the same pinned classifier-tokenizer 512-token clip used by the resource server. Require the pinned
  split of 300 classifier cases plus 100 copyright cases (50 book, 50 lyrics), with clip counts and hashes. It writes
  only payload-free `classifier-scores.json`. Download that single receipt and run
  `qwen_gcg_blade.py`; require 400 reconciled source indexes, exact model/classifier revisions, stable denominator
  accounting, and successful readback of its JSONL, metrics JSON, Markdown report, and `evidence-manifest.json` before
  reporting the baseline. The manifest binds the classifier-score input and all three rendered outputs by size and SHA-256.
- Run `qwen_gcg_result.py` only after all those gates pass. It binds the attack, exact-BF16 completion, mixed scorer, and
  validated BLADE bundle into the normalized payload-free `run-manifest.json` used for the final matched comparison. Do
  not compare the primary and baseline from standalone ASR numbers or draft Markdown.
- Once both normalized manifests have `status: validated`, run `compare_gcg_results.py`. It refuses mismatched upstream,
  attack, completion, classifier, or copyright fingerprints and any nonzero failure/missing count. Its ASR delta is
  descriptive only: do not turn it into a significance, causality, or model-superiority claim without separate analysis.

## Super GCG reporting gate

- After Super finalization, materialize through `prepare_generated.py`, collect all 400 target completions from the exact
  served Super identity, and run both classifier and copyright calibration controls. `report_method_run.py` now has an
  explicit GCG validation branch: it requires the finalized 400-row source receipt, matching source target, public 500/512
  profile, bound shard evidence, complete healthy collection, the pinned 300 classifier / 100 copyright routing split,
  and both scorer controls before it can emit `status: validated`.
  Its validated run manifest includes the same normalized attack, completion, scoring, count, metric, and target-revision
  fingerprints required by the Qwen result package.
