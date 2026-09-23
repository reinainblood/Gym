# HarmBench completion handoff for Claude

Snapshot time: 2026-09-20, America/Chicago. This is an evidence ledger and
continuation guide. A present adapter, a launched Modal task, generated attack
cases, scored rollouts, reconciliation, and a validated BLADE/report artifact
are separate gates throughout this document.

## Repository state

- Repository: `reinainblood/Gym`
- Working branch: `codex/harmbench-full-attacks`
- Starting commit before this consolidation:
  `8fed5091ff52106e726043a0b7df6740c4082418`
- Upstream public HarmBench commit:
  `8e1604d1171fe8a48d8febecd22f600e462bdcdd`
- Push target: `origin` (`https://github.com/reinainblood/Gym.git`)
- `upstream` is fetch-only and push-disabled. Do not open or push an NVIDIA
  upstream PR; Kirsten owns that publication step manually.
- No internal or NVIDIA PR is created by this handoff.

After checkout, use `git rev-parse HEAD` and `git status --short` as the
authoritative branch and cleanliness check. Generated result corpora remain
ignored. Small payload-free reports, manifests, and calibration receipts needed
to verify headline claims are committed under `benchmarks/harmbench/evidence/`;
they do not replace the external case-level artifacts bound by their hashes.

## Implemented method surface

The local package covers or binds the following public methods:

- API/fixed attacks: DirectRequest, HumanJailbreaks, ZeroShot, PAP-top5,
  MultiModalDirectRequest, MultiModalRenderText.
- Local-weight white-box attacks: GCG, GCG-Multi, AutoPrompt, GBDA, PEZ, UAT,
  AutoDAN, FewShot, MultiModalPGD, MultiModalPGDPatch, and
  MultiModalPGDBlankImage.
- Transfer/iterative catalog and input bridge: TAP-Transfer, GCG-Transfer,
  PAIR, TAP, Fresh PAIR, and Fresh TAP.
- The pinned public repository's `MultiModalPGDBlankImage` entry is represented
  in the shared catalog and the Qwen/Kimi operational launchers. Its pinned
  pipeline maps to the `MultiModalPGD` class with a separate 1,000-step config;
  preserve and audit that source behavior rather than silently substituting a
  different implementation.

Primary code:

- `resources_servers/harmbench/app.py`: canonical HarmBench and copyright
  scoring boundary.
- `benchmarks/harmbench/methods.py`: public method/target-type catalog.
- `benchmarks/harmbench/prepare_generated.py`: generated-case provenance and
  Gym input bridge.
- `benchmarks/harmbench/run_upstream_generation.py`: immutable public-method
  execution bridge.
- `benchmarks/harmbench/zero_shot.py` and `pap.py`: pinned Mixtral attacker
  transformations.
- `benchmarks/harmbench/ultra_whitebox.py` and `kimi_k3_whitebox.py`:
  source/config binding and command construction.
- `benchmarks/harmbench/blade_analysis.py` and `report_method_run.py`:
  reconciliation and payload-free evidence reporting.
- `benchmarks/harmbench/operations/`: checkpoint and full white-box FDR
  workflows copied into the repository by this handoff.

## Completed held-out test runs: validated, but not the full 400-row corpus

### Ultra ZeroShot

- Target: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
- 320 behaviors, five attacks each, 1,600/1,600 scored and healthy.
- 154 successes; behavior-averaged ASR 9.625%.
- 1,200 HarmBench-classifier cases plus 400 copyright-MinHash cases.
- Status: `validated_with_caveat`. The one raw replay disagreement is a
  measured vLLM batch-sensitive classifier case with identical prompt tokens
  and clipping. No label was manually overridden.
- Portable payload-free report:
  `benchmarks/harmbench/evidence/ultra-zeroshot/report.md`.
- Portable run manifest and control receipts:
  `benchmarks/harmbench/evidence/ultra-zeroshot/`.
- Observed rollout SHA-256:
  `339dfcc15260e5adbb98de60ab7bc8b785bbf0883d0bd30b3fc43f4a570f2b35`.

### Ultra PAP-top5

- Target: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
- 320 behaviors, five attacks each, 1,600/1,600 scored and healthy.
- 90 successes; behavior-averaged ASR 5.625%.
- Canonical classifier control passed 1,200/1,200; copyright control passed
  400/400; pinned upstream PAP mapping passed 1,600/1,600.
- Status: `validated` after stateless re-verification of unchanged target
  responses.
- Portable payload-free report:
  `benchmarks/harmbench/evidence/ultra-pap-top5/report.md`.
- Portable run manifest and control receipts:
  `benchmarks/harmbench/evidence/ultra-pap-top5/`.
- Reconciled rollout SHA-256:
  `8c5226f5b0e61562a17d9c76c103459efc97105c4ee4853ef78ee691472d31ba`.

Do not regenerate either Ultra campaign merely because the old implementation
was untracked. Rerun only for a named different target/checkpoint or a declared
profile change. ZeroShot and PAP-top5 are attacker-model/fixed-case methods;
they do not require target gradients.

The receipts above are complete for HarmBench's 320-row held-out text test
split. They are not complete executions of upstream's default 400-row
`harmbench_behaviors_text_all.csv` corpus. To satisfy the full-benchmark scope,
generate, target-complete, score, and reconcile the missing 80 validation rows
for each method without changing or discarding the 320-row evidence.

There is no method named `PAPSmear`, `PAP-Smear`, or `PAP smear` in the pinned
HarmBench repository. Resolve that phrase to an exact public source/config
before spending compute. It may be shorthand for PAP-top5 or a behavior about
an online smear campaign.

## Incomplete white-box work

### Ultra 3

- Exact differentiable checkpoint:
  `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16@77df655d5e9f8362164ed14dd8b48f8bce657498`.
- Eight-rank gradient receipt passed: finite/nonzero gradient and one signed
  step reduced loss from 5.5970458984375 to 5.502708911895752.
- Modal Volume: `harmbench-ultra-whitebox-results`.
- The directory named `ultra-gcg-full-20260918a` is not full. It contains only
  four individual behavior artifacts versus 400 required GCG cases.
- GCG must be resumed/recovered, then target-completed, canonically scored,
  reconciled, and reported. GCG-Multi, AutoPrompt, GBDA, PEZ, UAT, AutoDAN,
  and FewShot have no completed full Ultra campaign.
- Use `operations/ultra/incremental_generate.py`; do not discard compatible
  per-behavior outputs or reinterpret canaries as results.

### Qwen 3.5 122B-A10B

- Served FP8 weights were rejected for white-box optimization because the
  fused grouped-MoE operator has no registered autograd formula. No
  straight-through estimator was used.
- Official differentiable checkpoint:
  `Qwen/Qwen3.5-122B-A10B@dc4d348443bc740c68e2d77492492c11606384d5`.
- Gradient receipt passed on two B200s: processor parity `5.91e-8`, finite
  nonzero raw-pixel gradient, and loss 9.778717041015625 -> 8.790602684020996.
- Results Volume: `harmbench-qwen35-122b-whitebox-results`.
- Run ID: `qwen35-122b-harmbench-whitebox-20260919`.
- MultiModalPGD attack generation: 110/110 complete, 110 unique behaviors,
  nonempty target generations, exact indexes 0-109. Combined receipt-set hash
  over canonical sorted JSON objects (portable canonical-content algorithm):
  `98c971691d7109ae5bc057a71cb5defd5403a161a39c1278fc5da0fe0ac2db70`.
- The earlier audit recorded
  `2c58f30795612defc5ee9514869ceb2bc8bb5194aa4d6487d5214a4cd8d26299`
  by hashing `shasum` output containing temporary absolute paths. That value is
  retained as a historical, path-dependent receipt; it is not expected to
  reproduce on another checkout. The canonical hash above supersedes it for
  portable verification. Both algorithms and values are committed in
  `evidence/qwen-whitebox-pgd/receipt-set-hash-migration.json`.
- MultiModalPGDPatch: 9/110 complete at the reconciliation snapshot.
- MultiModalPGDBlankImage: 0/110 complete at the audit snapshot.
- Four Qwen worker tasks were active at the snapshot. Recheck FDR before any
  restart; long methods commit one case at a time and resume by skipping
  existing receipts.
- The 110 PGD cases are attack-generation artifacts, not a final method
  result. They still require canonical target/scorer reconciliation and BLADE
  reporting.
- The eight public text white-box methods have not been executed for Qwen.

Qwen source-image filenames use the checked correction artifacts:

- corrected CSV SHA-256:
  `ed85ca605a07ea7e66eb03726d2ac5df0444926e6709721ffaa3c33671557950`;
- seven-row correction manifest SHA-256:
  `7b25da589c49ac46ff4454e343dac64beb5059356ef8bb0c1cae2c0a9765bf07`.

Do not return to the raw upstream CSV for those seven rows: it names `.jpeg`
or `.jpg` files while the released assets are `.png`.

### Kimi K3

- Exact served checkpoint:
  `moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Checkpoint Volume: `endpoint-ep-WD4gnzaeXMyM7DfrzTDYjW`.
- Results Volume: `harmbench-kimi-k3-whitebox-results`.
- 118 files, 96 weight shards, 1,560,998,984,390 bytes; full file hashes are
  recorded in the Volume checkpoint manifest.
- The public checkpoint is packed MXFP4 inference weights. On the first target
  forward, the public compressed-tensors path attempted full decompression and
  exhausted GPU 0 on eight B300s after 5,645 of 247,296 quantized modules.
- No forward completed, no gradient was observed, and no methods were launched.
- Payload-free blocker receipt:
  `operations/kimi/receipts/k3-text-gradient-20260919g-failure.json`.
- Do not use a tiny model, different checkpoint, calibration proxy,
  straight-through estimator, or reduced attack and call it Kimi K3. Clearing
  this blocker requires an official differentiable Kimi checkpoint or a
  packed-kernel autograd implementation faithful to the public method.
- `MultiModalPGDBlankImage` is now cataloged and launchable, but remains
  unexecuted behind the same differentiability blocker.

### Super 3.5 VL

- Eight-H200 raw-pixel gradient receipt passed for model version
  `hf-ea-0e636f7`: finite/nonzero gradient and loss 8.278218269348145 ->
  7.962818622589111 on every rank.
- The full 110-case MultiModalPGD, 110-case MultiModalPGDPatch, and 110-case
  MultiModalPGDBlankImage campaigns have not been executed. A gradient receipt
  is not a completed benchmark.

## Already completed black-box multimodal Qwen results

These are separate from white-box PGD and should not be rerun accidentally:

- `MultiModalDirectRequest`: 110/110 healthy, canonical reverified ASR
  46/110 = 41.82%, zero excluded rows.
- `MultiModalRenderText`: 110/110 healthy, canonical reverified ASR
  35/110 = 31.82%, zero excluded rows.
- Portable payload-free reports, manifests, classifier controls, reverify
  comparisons, and combined BLADE metrics are under
  `benchmarks/harmbench/evidence/qwen-multimodal/`.

On the original execution host, the complete ignored run packages are under
`/Users/kruge/Documents/ChatGPT/nvidia/worktrees/harmbench/results/`. A fresh
clone will not contain those large case-level artifacts. If that absolute path
is unavailable, use the committed hashes and named Modal Volumes/delivery
packages to recover them; do not describe an absent ignored directory as if it
were committed branch content.

## Safe continuation order

1. Inspect live FDR state before mutation:

   ```bash
   modal app list --env FDR --json
   modal volume ls harmbench-qwen35-122b-whitebox-results / --env FDR
   modal volume ls harmbench-kimi-k3-whitebox-results / --env FDR
   modal volume ls harmbench-ultra-whitebox-results / --env FDR
   ```

2. Preserve active Qwen work. Count case receipts for PGD, PGD-Patch, and
   BlankImage. Resume only missing indexes with the same exact checkpoint,
   source revision, corrected filename manifest, preprocessing, seeds, and
   public hyperparameters.
3. Finish Qwen attack generation, then run the canonical HarmBench classifier
   and copyright scorer, reconcile exact denominators, and build BLADE evidence.
4. Resume Ultra GCG from compatible individual behavior artifacts. Complete
   and reconcile all 400 cases before moving through the remaining seven text
   white-box methods.
5. Keep Kimi blocked unless a scientifically valid differentiable runtime
   becomes available. Retain the failed attempt outside model-quality
   denominators.
6. Execute full Super VL PGD/Patch on the already-proven differentiable
   runtime.
7. Run ZeroShot/PAP for another target only if the target is explicitly named;
   do not overwrite or duplicate the validated Ultra artifacts.
8. After each method: expected/scored/failure accounting, rollout health,
   classifier/copyright parity as applicable, BLADE generation and readback,
   agent review, and Kirsten review. Do not collapse those gates into “done.”

## Validation commands

Complete HarmBench adapter and reporting suite:

```bash
uv sync --extra dev
(
  cd resources_servers/harmbench
  uv pip install --python ../../.venv/bin/python -r requirements.txt
)
export HARMBENCH_UPSTREAM_DIR=/absolute/path/to/HarmBench
test "$(git -C "$HARMBENCH_UPSTREAM_DIR" rev-parse HEAD)" = \
  8e1604d1171fe8a48d8febecd22f600e462bdcdd
.venv/bin/pytest -q resources_servers/harmbench/tests
.venv/bin/ruff check benchmarks/harmbench resources_servers/harmbench
.venv/bin/ruff format --check benchmarks/harmbench resources_servers/harmbench
git diff --check
```

Without `HARMBENCH_UPSTREAM_DIR`, source-parity tests skip consistently from
any checkout layout. With the pinned checkout configured, they must execute and
pass; a checkout-depth-specific path is never assumed.

Repository policy additionally requires scoped pre-commit and a DCO-signed
commit before PR review. This handoff creates no PR.

## Sensitive-data boundary

- Public benchmark prompts and row metadata are permitted in the local result
  artifacts, but result corpora are intentionally ignored from Git.
- Never commit API keys, Modal proxy tokens, Keychain values, raw credentials,
  model weights, Hugging Face caches, or private/gated data.
- The committed receipts are payload-free infrastructure/provenance records.

## Codex takeover update — 2026-09-20 23:30 CDT

- Qwen MultiModalPGD now has a complete parent-versus-512-versus-replicate
  classifier comparison. The protocol-correct 512-token completions scored
  31/110 (28.18% ASR); both parent and canonical labels were stable on two
  serial raw-classifier passes for all 110 cases. Parent versus canonical had
  seven label flips, including two among 66 cases where the old 256-token cap
  did not bind. The 12-case fresh-container replicate matched 12/12 canonical
  labels. Treat the 512 result as valid with a generation-sensitivity caveat,
  not as invalid or classifier-noisy.
- The full 351,508-byte comparison receipt is retained in
  `harmbench-qwen35-122b-whitebox-results` at
  `qwen35-122b-harmbench-whitebox-20260919/MultiModalPGD/label-comparison/parent-vs-512-vs-r1.json`
  with SHA-256
  `b8612e5e76ddb3b48ea785f2c5bb2ceaebecb59efc02f1626c5e7ee0b6cf31d3`.
- Contrary to the earlier snapshot, MultiModalPGDBlankImage is not queued.
  Four FDR workers were live and writing resumable artifacts during takeover.
  MultiModalPGDPatch and BlankImage remain incomplete and must be recounted from
  the Volume before any launch or restart.
