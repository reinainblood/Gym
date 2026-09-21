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
- After both calls are terminal and workers are zero, redeploy the current branch and call the GCG app's CPU-only
  `status_campaign("super", "super-gcg-full-20260921a", 2)`. It validates nested artifacts without returning attacks,
  distinguishes missing from malformed source indexes, counts unexpected entries, verifies the complete shard-receipt set,
  and returns the minimal `resume_shards`. Launch only those shard IDs; call `finalize` only when `ready_to_finalize` is true.
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
- Before resuming, call the white-box app's CPU-only `status_method`. It returns only counts and source indexes, classifies
  every index as valid/missing/invalid, counts extra case/image files, and reports the minimal four-way `resume_shards` set.
  Launch only those shards; do not infer missing work from filename counts alone.
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
- Run `qwen_whitebox_result.py` after a modern BlankImage or Patch BLADE bundle passes. It verifies the complete
  attack→completion→score→BLADE hash chain, keeps native and BLADE denominators distinct, and emits checksummed
  `run-manifest.json`, `report.md`, and `evidence-manifest.json`. It intentionally does not relabel historical PGD.
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

## Live campaign milestone — 2026-09-21 10:11 CDT

- The full Super campaign produced its first individual behavior artifact under
  `super-gcg-full-20260921a`. A payload-free in-place readback confirmed valid JSON with exactly one behavior mapping,
  a nonempty attack list, 296 bytes, and SHA-256
  `c13789eccb40ee8ec8b66e03c54cce824769438dc8bdc97448650be84873b6c5`.
- Both source-order Super function calls remained live and the app still had exactly two attack workers. The campaign had
  1/400 individual artifacts and 0/2 shard receipts. This is partial execution evidence only; do not finalize, redeploy,
  or launch Qwen GCG yet.
- The served Super endpoint remained at three live instances. The independent Qwen BlankImage campaign had reached 87/110
  completed cases with three optimization streams still advancing, so Patch remained gated behind worker drain.

### Two-shard liveness update — 2026-09-21 10:34 CDT

- The full Super campaign reached 2/400 individual artifacts while both function calls remained live. Payload-free
  provenance checks mapped the artifacts to pinned public source indexes 0 and 1 and confirmed they belong to stride
  shards 0 and 1, respectively. Both artifacts contain exactly one behavior mapping with a nonempty attack list.
- Source index 0 is 361 bytes with SHA-256
  `bcbd32c7bb90c4ece1be068a0285b310519411a23d127b82c095f0271dc39d24`; source index 1 remains 296 bytes with the
  previously recorded hash. This proves that both full-campaign workers have persisted a correctly partitioned behavior;
  it does not satisfy either 200-behavior shard receipt or the 400-behavior finalization gate.
- Qwen BlankImage reached 90/110 completed cases and immediately assigned replacement work to the three existing streams.
  Patch remains gated until those workers drain and the current branch can be safely redeployed for exact reconciliation.

### BlankImage shard-drain update — 2026-09-21 11:51 CDT

- Qwen MultiModalPGDBlankImage reached 96/110 case files. Filename-level source partition accounting showed modulo-4
  shards 1 and 3 complete with 28/28 and 27/27 cases, while shards 0 and 2 had 22/28 and 19/27, leaving 6 and 8 cases.
- Two unfinished checkpoint streams remained visible and advancing, matching the two incomplete shards. The drop from
  three visible streams to two was therefore normal shard drain, not a lost worker. Do not relaunch completed shards.
- Continue to wait for both remaining streams and all attack workers to drain before redeploying the current branch and
  calling `status_method`; filename counts are progress evidence only and do not replace that exact validity audit.

## Full Super fixed/black-box milestone — 2026-09-21 14:18 CDT

- The served Super endpoint `ap-Y6sMOWq4rnrM5EU6efe8Zf` was scaled in place to six live TP4 replicas, above the
  requested floor of three. The independently served classifier app `ap-Z2MFk4t6FPLu9onmrrtkEd` has two live replicas.
  The GCG app remains separate with exactly two attack workers; neither GCG shard was duplicated or redeployed.
- DirectRequest, HumanJailbreaks, ZeroShot, and PAP-top5 now have complete full-corpus Super target runs with exact
  expected/scored/failed/missing counts of 400/400/0/0, 2,000/2,000/0/0, 2,000/2,000/0/0, and 2,000/2,000/0/0.
  A combined source-order collection bound 6,400 unique rollout IDs to 6,400 complete model-call captures and reconciled
  6,400 healthy, zero unhealthy, zero unobserved trajectories before method-specific splitting.
- All four runs passed their pinned upstream generation controls, exact raw-classifier replay, copyright replay, strict
  `report_method_run.py` validation, and deterministic BLADE output readback. The BLADE writer now emits and validates an
  evidence manifest that binds row JSONL, metrics JSON, and Markdown by byte size and SHA-256 and fails on tampering.
- A resumable client-fresh FDR runtime now preserves upstream PAIR/TAP Mixtral attacker/judge settings and changes only
  the independently receipt-bound Super target. Independent target calls within an upstream batch may execute in parallel,
  while results and hashed call receipts remain in source order. The final one-behavior canaries passed under calls
  `fc-01M32Q6ACBVR9F2HBJR2JK6H74` (PAIR: 60 target calls) and `fc-01M32Q6A437YR5MYPTEM76ZTNK` (TAP: 20 target calls).
  Both receipts bind the exact Super model/revision. Earlier attempts failed closed before writing attack artifacts: first on
  a missing eager upstream provider dependency, then on the endpoint's thinking-mode response shape. The runtime now carries
  the complete public dependency set and explicitly preserves the campaign's non-thinking target profile.
- The existing Qwen BlankImage campaign reached 105 filename-level case files with its existing workers still live. This
  remains progress only. Continue to wait for worker drain, then redeploy and use `status_method` before any repair.

## TAP-Transfer full Super milestone — 2026-09-21 14:42 CDT

- The official HarmBench 1.0 Zenodo archive (DOI `10.5281/zenodo.10714577`, archive MD5
  `ac0e8210389b81951f66da6daaa6db3b`) contains the pinned TAP-Transfer source experiment. The 394,358-byte member was
  range-read without downloading the 10.4 GB archive and matched SHA-256
  `71438be2ca516be44b7ffc03230c01719b8fe3c24c9cae89e17deecd28c92de2`.
- The historical source held 403 behaviors. Import validation proved it covers all 400 behaviors in the current pinned corpus
  plus exactly three retired IDs bound by a stable aggregate hash. Filtering only those retired rows and restoring current
  source order produced a 400-case artifact with SHA-256
  `0b75db8c0151fefb550479f3f27f6976eef578a0f68cc8774f6a6ac3df08ce0e`.
- Fresh Super collection and scoring completed 400/400 healthy cases with zero failures/missing, 26 successes, and 6.5% ASR.
  Raw classifier replay matched 300/300; copyright replay matched 100/100. The strict method report and deterministic BLADE
  bundle readback both passed. Do not reuse any target responses from the historical archive; only its fixed attack cases enter
  this result.

## Client-fresh full campaigns — 2026-09-21 14:47 CDT

- Persistent FDR app `harmbench-client-fresh-super`, app ID `ap-HsL55FPL5a3QhChWI6N2X5`, was deployed from pushed
  commit `f8d8b46`. Results Volume: `harmbench-client-fresh-super-results`; Mixtral cache Volume:
  `harmbench-mixtral-attacker-cache`. The exact Super target remains the independently receipt-bound served endpoint.
- Full PAIR artifact `pair-super-full-400-20260921a` has exactly three source-order shards:
  `fc-01M32R45HPJ483RX4R4E1EWVCW`, `fc-01M32R45MYM788VC98TNWKYMN3`, and
  `fc-01M32R45RM8AK9CQDGV6HA9NQ2`.
- Full TAP artifact `tap-super-full-400-20260921a` has exactly three source-order shards:
  `fc-01M32R45XXW0G3X2T90NVQ6ZY3`, `fc-01M32R4610ATD46JKC6W9P3483`, and
  `fc-01M32R4657KWVFY4MMZZAKRCFE`.
- All six workers are live. Together they issue at most 120 concurrent target calls, matching the six-replica served Super
  capacity. Do not duplicate a shard while its call/container remains live. A behavior is complete only when its private
  upstream test-case artifact and hashed client-call receipt both persist; a full method is complete only after all three
  shard receipts validate and the CPU finalizer writes the canonical 400-behavior generation receipt.

## GCG-Transfer exact-source blocker — 2026-09-21 15:22 CDT

- The pinned pipeline requires the four-model Llama 2 7B Chat / Vicuna 7B v1.5 / Llama 2 13B Chat / Vicuna 13B v1.5
  ensemble at 1,000 optimization steps, search width 512, and run IDs 0–4. This yields the 2,000 source cases that must then
  receive fresh Super completions and scoring.
- The official HarmBench 1.0 Zenodo archive was range-inspected without downloading the 10.4 GB ZIP. Its matching four-model
  ensemble artifact is bound to 500 steps, not 1,000. Preserve it as historical evidence but never use it to clear the pinned
  GCG-Transfer denominator.
- Authenticated Hugging Face metadata resolved the exact current checkpoint revisions. Vicuna 7B v1.5 revision
  `3321f76e3f527bd14065daf69dad9344000a201d` and Vicuna 13B v1.5 revision
  `c8327bf999adbd2efe2e75f6509fa01436100dc2` are accessible. Weight preflight for Meta Llama 2 7B Chat revision
  `f5db02db724555f92da89c216ac04704f23d4590` and Meta Llama 2 13B Chat revision
  `a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8` returns access denied because both repositories are manually gated.
  No exact checkpoint copy exists in the local cache or FDR Volumes. The existing FDR secret `agentdyn-hf-gated-access` was
  also tested through CPU-only authenticated safetensors HEAD requests and returned `GatedRepoError` for both exact revisions.
  Categorize this row as `gated_model`; do not substitute a mirror, another Llama revision, the 500-step release artifact, or
  a reduced budget.

## Ordinary PAIR full Super milestone — 2026-09-21 15:31 CDT

- The official HarmBench 1.0 `PAIR/mixtral_8x7b` source member is 402,090 bytes with SHA-256
  `9a85a36ae7aa47fef160eca99746379a1521f75acb2d85749ed083e0aac4af29`. It contains 401 historical behaviors:
  398 current-corpus rows plus the same three retired IDs present in other release artifacts.
- FDR repair call `fc-01M32SEPCVZW566NJ8AYZFYNWC` regenerated exactly the two missing current behaviors through the pinned
  upstream PAIR class, `mixtral_8x7b` experiment, and exact public method/model configs. Its completed receipt binds two
  repaired behaviors and two cases. Merging official and repaired sources in current source order yielded 400 cases with
  SHA-256 `1b0da501a209eea988b0d7a1c41c86980f354787a2abadd90ddd0b5024cafdc2`.
- Fresh Super target collection scored 400/400 healthy cases with zero failures/missing, 21 successes, and 5.25% ASR. Raw
  classifier replay matched 300/300 and copyright replay matched 100/100. The strict method report and deterministic BLADE
  readback passed. Ordinary TAP repair call `fc-01M32SEPNMVQFE36D2HDR27JGS` remains live; do not infer completion from PAIR.

## Ordinary TAP full Super milestone — 2026-09-21 15:41 CDT

- The official HarmBench 1.0 `TAP/mixtral_8x7b` source member is 408,632 bytes with SHA-256
  `5b4dd35089a4b02458551b3cfbb6b747787a7bf14012775ef1d441d6aaf24867`. It contains 401 historical behaviors:
  398 current-corpus rows plus the same three retired IDs present in other release artifacts.
- FDR repair call `fc-01M32SEPNMVQFE36D2HDR27JGS` regenerated exactly the two missing current behaviors through the pinned
  upstream TAP class, `mixtral_8x7b` experiment, and exact public method/model configs. Its completed receipt binds two
  repaired behaviors and two cases. Merging official and repaired sources in current source order yielded 400 cases with
  SHA-256 `a1c30c304c3fb92e37d8f16941a507069f9f6bd082dc35df6171f36342fb7db2`.
- Fresh Super target collection scored 400/400 healthy cases with zero failures/missing, 35 successes, and 8.75% ASR. Raw
  classifier replay matched 300/300 and copyright replay matched 100/100. The strict method report and deterministic BLADE
  readback passed. Ordinary PAIR and TAP are now both validated; do not rerun either source or target result.
