# HarmBench method expansion — execution ledger

This ledger is intentionally stricter than a feature checklist. A method is **covered** only after its upstream
attack generator, exact experiment, target-model interface, Gym rollout, correct scorer, and case-level source
receipt have all been exercised. The 23 names below are the methods requested by NVIDIA. The source of truth is
`centerforaisafety/HarmBench` at `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The two "Fresh ... against
the client model" entries are NVIDIA-requested client-targeted variants of upstream PAIR/TAP, not separate proper
method keys in the paper's pipeline.

## Mandatory Nemotron 3.5 Super VL common-proof gate

Historical runs, another target model, source-generation alone, canaries, transport probes, partial shards, and
schema previews do **not** clear a requested method. Every method below must finish on the exact Nemotron 3.5
Super VL EA BF16 checkpoint, reconcile the full pinned public denominator as `expected = scored + failed + missing`,
have zero missing rows, and pass the method-specific generation, target-completion, scoring, provenance, and
payload-free report readback gates. `Scored` is a target-model completion that reached its required scorer; an
attack artifact without target execution is not scored. Counts are source-order case counts, not worker or shard
counts. These are the current evidence-backed counts in this branch; white-box rows owned by the separate main
task remain uncleared until their complete Super receipts are imported and revalidated here.

| Requested method | Expected | Scored | Failed | Missing | Super common-proof state |
|---|---:|---:|---:|---:|---|
| DirectRequest | 400 | 400 | 0 | 0 | **Validated:** full Super target/scorer collection, 400 healthy trajectories, upstream generation, classifier, copyright, BLADE, and report gates passed. |
| HumanJailbreaks | 2,000 | 2,000 | 0 | 0 | **Validated:** full Super target/scorer collection, 2,000 healthy trajectories, generation, classifier, copyright, BLADE, and report gates passed. |
| ZeroShot | 2,000 | 2,000 | 0 | 0 | **Validated:** full Super target/scorer collection, 2,000 healthy trajectories, generation, classifier, copyright, BLADE, and report gates passed. |
| PAP-top5 | 2,000 | 2,000 | 0 | 0 | **Validated:** full Super target/scorer collection, 2,000 healthy trajectories, generation, classifier, copyright, BLADE, and report gates passed. |
| TAP-Transfer | 400 | 400 | 0 | 0 | **Validated:** official pinned source cases, full Super target/scorer collection, 400 healthy trajectories, classifier, copyright, BLADE, and report gates passed. |
| MultiModalDirectRequest | 110 | 110 | 0 | 0 | **Validated:** full Super run and image/classifier evidence passed. |
| MultiModalRenderText | 110 | 110 | 0 | 0 | **Validated:** full Super run and render/classifier evidence passed. |
| GCG | 400 | 0 | 0 | 400 | In progress: two attack shards are active; partial attack artifacts are not target-scored cases. |
| GCG-Multi | 2,000 | 0 | 0 | 2,000 | Not cleared: five complete source runs and full Super scoring are required. |
| AutoPrompt | 400 | 0 | 0 | 400 | Not cleared: no complete Super package. |
| GBDA | 2,000 | 0 | 0 | 2,000 | Not cleared: no complete Super package. |
| PEZ | 2,000 | 0 | 0 | 2,000 | Not cleared: no complete Super package. |
| UAT | 400 | 0 | 0 | 400 | Not cleared: no complete Super package. |
| AutoDAN | 400 | 0 | 0 | 400 | Not cleared: no complete Super package. |
| FewShot | 400 | 0 | 0 | 400 | Not cleared: no complete Super package. |
| PAIR | 400 | 400 | 0 | 0 | **Validated:** official Mixtral source plus exact two-behavior repair, full Super target/scorer collection, 400 healthy trajectories, classifier, copyright, BLADE, and report gates passed. |
| TAP | 400 | 400 | 0 | 0 | **Validated:** official Mixtral source plus exact two-behavior repair, full Super target/scorer collection, 400 healthy trajectories, classifier, copyright, BLADE, and report gates passed. |
| GCG-Transfer | 2,000 | 0 | 0 | 2,000 | Blocked on exact source checkpoints: five 1,000-step ensemble runs and full Super scoring are required; the public release has only a non-substitutable 500-step artifact, and the authenticated account lacks access to both pinned Meta Llama source checkpoints. |
| Fresh PAIR against the client model | 400 | 0 | 0 | 400 | Not cleared: no complete client-fresh Super loop and score package. |
| Fresh TAP against the client model | 400 | 0 | 0 | 400 | Not cleared: no complete client-fresh Super loop and score package. |
| MultiModalPGD | 110 | 0 | 0 | 110 | Main-task-owned; no complete Super evidence has been imported into this branch. |
| MultiModalPGDPatch | 110 | 0 | 0 | 110 | Main-task-owned; no complete Super evidence has been imported into this branch. |
| MultiModalPGDBlankImage | 110 | 0 | 0 | 110 | Main-task-owned; no complete Super evidence has been imported into this branch. |

| Requested method | Generator / client requirement | Verified state in this branch |
|---|---|---|
| DirectRequest | Single-turn text request | **Validated full Super VL run:** 400/400 healthy and scored, zero failures/missing, 58 successful cases, 14.5% ASR. Pinned upstream generation, 300/300 raw-classifier parity, 100/100 copyright parity, BLADE reconciliation, and payload-free report validation passed. The prior 320-row Ultra result remains historical evidence only. |
| HumanJailbreaks | Pinned human templates, random subset 5 | **Validated full Super VL run:** 2,000/2,000 healthy and scored across 400 behaviors, zero failures/missing, 76 successful cases, 3.8% behavior-averaged ASR. Pinned generation replay matched 400/400 behaviors; raw classifier matched 1,500/1,500 and copyright matched 500/500. BLADE and report validation passed. |
| ZeroShot | Upstream Mixtral attacker model | **Validated full Super VL run:** the pinned FDR Mixtral generator produced 2,000 cases across all 400 behaviors, upstream replay matched 400/400, and Super scored 2,000/2,000 healthy cases with zero failures/missing, 280 successful cases, and 14.0% behavior-averaged ASR. Raw classifier and copyright controls matched 1,500/1,500 and 500/500; BLADE and report validation passed. |
| PAP-top5 | Upstream Mixtral attack model, pinned five-technique taxonomy | **Validated full Super VL run:** the pinned FDR attacker and five-technique mapping produced and replayed 2,000/2,000 cases over 400 behaviors. Super scored all 2,000 healthy cases with zero failures/missing, 56 successful cases, and 2.8% behavior-averaged ASR. Raw classifier and copyright controls matched 1,500/1,500 and 500/500; BLADE and report validation passed. |
| TAP-Transfer | Fixed upstream transfer experiment | **Validated full Super VL run:** the official HarmBench 1.0 Zenodo source experiment was SHA-256 verified, reconciled from its 403-row historical set to the exact current 400-row pinned corpus by removing three receipt-hashed retired behaviors, and then freshly executed on Super. All 400 cases were healthy and scored with zero failures/missing, 26 successes, and 6.5% ASR; raw classifier matched 300/300 and copyright matched 100/100. BLADE and report validation passed. |
| MultiModalDirectRequest | Image-bearing target | **Validated full run:** 110/110 healthy Super 3.5 VL Gym rollouts, zero sidecar failures, 54 classifier successes (49.1% ASR), 3 truncated/clipped. All 110 images are pixel-identical to upstream Torchvision, and raw-vs-Gym classifier labels, prompt tokens/hashes, and clip text/count agree 110/110. |
| MultiModalRenderText | Image-bearing target | **Validated full Super VL run:** 110/110 healthy rollouts, zero failures, 11 classifier successes (10.0% ASR), one truncated/clipped. Upstream render functions reproduce 110/110 image pixels and instructions when bound to the declared DejaVu Sans font; classifier controls match 110/110. The fixed font replaces upstream's host-dependent first-system-font lookup. |
| GCG | Local target weights and gradients | **Target policy changed by Kirsten:** run the full public protocol on unreleased Nemotron 3.5 Super VL 120B BF16 as the primary result and Qwen 3.5 122B-A10B BF16 as the single matched baseline. Ultra and Kimi are excluded from new GCG launches and denominators. The persistent FDR runtime pins both complete checkpoints, upstream commit, 500 steps, search width 512, compiled recurrent kernels, resumable behavior shards, and a Gym-compatible finalizer. Finalization now requires and cryptographically binds the complete source-order shard evidence; the generated-method bridge reads those receipts back and refuses a partial corpus. The full-budget Super one-behavior canary passed with its nested behavior artifact and shard receipt; the two-shard 400-behavior Super campaign is active. A stale Qwen canary prefix contains no completed behavior or receipt and must not be resumed until Super is complete and finalized. An exact-BF16, 512-token, resumable Qwen target-completion stage is implemented so the FP8 endpoint cannot contaminate the matched baseline. No full-corpus GCG result exists yet. |
| GCG-Multi | Local target weights; upstream run IDs 0–4 | Bridge enforces all five source runs before merging; no generator run or model rollout. |
| AutoPrompt | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| GBDA | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| PEZ | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| UAT | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| AutoDAN | Local target weights | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| FewShot | Upstream open-weight target configuration | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| PAIR | Upstream iterative attacker and judge | **Validated full Super VL run:** the official HarmBench 1.0 Mixtral-targeted source was SHA-256 verified and reconciled to 398 current behaviors after excluding three receipt-hashed retired rows. The two missing current behaviors were regenerated through the exact pinned upstream PAIR/Mixtral method and bound by a repair receipt. Super scored the merged source-order 400/400 healthy cases with zero failures/missing, 21 successes, and 5.25% ASR. Raw classifier matched 300/300 and copyright matched 100/100; BLADE and report validation passed. |
| TAP | Upstream tree-search attacker and judge | **Validated full Super VL run:** the official HarmBench 1.0 Mixtral-targeted source was SHA-256 verified and reconciled to 398 current behaviors after excluding three receipt-hashed retired rows. The two missing current behaviors were regenerated through the exact pinned upstream TAP/Mixtral method and bound by a repair receipt. Super scored the merged source-order 400/400 healthy cases with zero failures/missing, 35 successes, and 8.75% ASR. Raw classifier matched 300/300 and copyright matched 100/100; BLADE and report validation passed. |
| GCG-Transfer | Fixed upstream transfer ensemble; run IDs 0–4 | **Exact-source blocker:** the pinned pipeline requires five 1,000-step runs over Llama 2 7B Chat, Vicuna 7B v1.5, Llama 2 13B Chat, and Vicuna 13B v1.5 before a 2,000-case Super target run. The official HarmBench 1.0 archive was inspected by range: its matching four-model ensemble receipt records only 500 steps, so it is retained but cannot satisfy this row. The pinned Vicuna revisions are public, but the authenticated Hugging Face account receives access denied for `meta-llama/Llama-2-7b-chat-hf` revision `f5db02db724555f92da89c216ac04704f23d4590` and `meta-llama/Llama-2-13b-chat-hf` revision `a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8`. No reduced-budget or substitute-weight run is permitted. |
| Fresh PAIR against the client model | Client-model queries inside the PAIR loop | The resumable client-target adapter preserves upstream PAIR's Mixtral attacker/judge, public stream/step/query budgets, and stopping rule while replacing only the target model with a verified OpenAI-compatible client endpoint. It requires an immutable deployment-binding receipt and environment-only API key; no full generator run or model rollout exists yet. |
| Fresh TAP against the client model | Client-model queries inside the TAP tree search | The resumable client-target adapter preserves upstream TAP's Mixtral attacker/judge, branching, pruning, depth/query budgets, and stopping rule while replacing only the target model with a verified OpenAI-compatible client endpoint. It requires an immutable deployment-binding receipt and environment-only API key; no full generator run or model rollout exists yet. |
| MultiModalPGD | Local multimodal weights and image gradients | **Qwen white-box attack, canonical scoring, and BLADE reconciliation complete.** The exact Qwen 3.5 122B-A10B BF16 checkpoint produced all 110 optimized images and protocol-correct 512-token target completions. The pinned raw HarmBench classifier scored 31/110 successes (28.18% ASR), with 110/110 repeat agreement on two serial passes. Payload-free BLADE evidence reconciles all 110 rows into the model denominator with zero exclusions. Seven labels differ from the superseded 256-token parents; two are among 66 cases where the old cap did not bind. A 12-case fresh-container replicate matched 12/12 canonical labels. Super 3.5 VL has a passed 8×H200 raw-pixel gradient canary but no full PGD campaign. |
| MultiModalPGDPatch | Local multimodal weights and image gradients | A resumable Qwen 3.5 122B-A10B BF16 campaign is active under the public 2,000-step patch protocol; no complete scored result exists yet. The same verified Super VL gradient runtime separately satisfies the checkpoint/autograd prerequisite, but no full Super patch campaign has run. |
| MultiModalPGDBlankImage | Local multimodal weights and image gradients | A resumable Qwen 3.5 122B-A10B BF16 campaign is active under the pinned 1,000-step configuration; no complete scored result exists yet. Treat the pinned pipeline's `MultiModalPGD` class mapping as a source behavior to preserve and audit, not permission to rename another method. |

## Current reproducible paths

- `gym eval prepare --benchmark harmbench` remains the legacy 240-row held-out-test DirectRequest-only lane. Opt in to the 2,000-case full-corpus HumanJailbreaks
  set with `gym eval prepare --benchmark harmbench_human_jailbreaks`. HumanJailbreaks requires
  `HARMBENCH_UPSTREAM_DIR` at the pinned commit and `HARMBENCH_COPYRIGHT_HASHES_DIR` pointing to that checkout's
  `data/copyright_classifier_hashes`; preparation fails if either source drifts or a copyright reference is missing.
- `modal_super_vl_gradient_canary.py` launches an ephemeral **FDR** 8×H200 job against the read-only
  `nemotron-3-5-super-vl-ea-09112026` Volume. Its torchrun worker loads checkpoint `hf-ea-0e636f7` with
  Transformers FSDP2, freezes the model parameters, differentiates a safe target-completion loss with respect to
  a synthetic 512×512 RGB tensor, and requires a finite nonzero gradient plus loss reduction on every rank after
  one 1/255 signed step. Run `super-vl-grad-canary-20260918d` passed; its receipt SHA-256 is
  `94369571a70ed7675d39d6aae0dfa17351fa96d1e71a14d51ecd77a5ddaf2892`. The peak allocated memory was
  61,888,249,856 bytes per rank. This proves the runtime prerequisite; it is not a completed PGD benchmark run.
- `generate_direct_full.py` executes the SHA-pinned upstream DirectRequest method over all 400 public text behaviors,
  checks its output against the independent context-prefix formula, and emits attack cases, a source receipt, and
  a payload-free generation control. Import its `test_cases.json` through `prepare_generated.py` with
  `--method DirectRequest --experiment default` (the pinned upstream experiment); do not silently replace the
  default 240-row dataset. The initial `direct-full-20260917` exploratory receipt used a local experiment label
  and is superseded by `direct-full-default-20260917`; their generated test-case hashes match.
- `run_upstream_generation.py` dispatches the named upstream attack class using an explicitly selected upstream
  Python environment. It hashes the pipeline, model/method configs, behaviors, generated cases, and (for ensembles)
  each constituent run. `prepare_generated.py` refuses a method label that disagrees with its generation receipt.
  `harmbench_generated` is the Gym benchmark overlay for a verified generated JSONL.
- For the two client-fresh variants, `run_upstream_generation.py` delegates to `client_fresh_generate.py`. That
  adapter injects a verified OpenAI-compatible target into the unchanged pinned PAIR/TAP loop, persists each
  behavior before moving on, and skips only behaviors with both a saved upstream test case and payload-free client
  call receipt. It requires a deployment-binding receipt covering the app ID, endpoint hash, exact model ID, and
  immutable revision; the API key is referenced only by environment-variable name and never enters arguments or
  receipts. The final receipt binds all behavior-call receipt hashes to the generated cases before materialization.
- `modal_zero_shot.py` and `modal_pap.py` generate the two pinned Mixtral methods in ephemeral **FDR** apps using
  one H200, eager vLLM, and the Triton MoE backend (the auto-selected FlashInfer CUTLASS path stalled in bounded
  canaries). Set `HARMBENCH_SOURCE_CSV` to the pinned 400-row `harmbench_behaviors_text_all.csv`; both launchers
  reject any other source hash or cardinality. PAP additionally requires
  `HARMBENCH_PAP_TEMPLATES` at the pinned `baselines/pap/templates.py`. Each run writes a source-hash-bound
  `generation-receipt.json` and `test_cases.json` to its dedicated FDR Volume. PAP also stores private
  `raw_attacker_generations.json` because its quote/whitespace post-processing cannot be reversed reliably from
  final cases. Neither generator prints attack text, target output, or credentials to Modal logs. The
  `calibrate_zero_shot_generation.py` and `calibrate_pap_generation.py` controls replay the actual pinned upstream
  method bodies and fail on any case mismatch. The generated datasets must be imported through
  `prepare_generated.py`, then prepared with `gym eval prepare --benchmark harmbench_generated` before scoring.
  The first full PAP artifact (`pap-full-20260917`) omitted raw generations and is superseded by
  `pap-full-raw-20260917`; its final case SHA-256 was identical, but it is not used as the validated provenance
  source. The corrected run's upstream replay matched all 320 behaviors and all 1,600 cases.
- `harmbench_generated/prepare.py` preserves the previous fixed-slot aggregate metrics under its dataset SHA-256
  when switching generated methods. This prevents a stale canary or previous method denominator from silently
  contaminating the next Gym run.
- `operations/qwen/modal_qwen_completion.py` regenerates canonical target completions from committed white-box
  images at HarmBench's 512-token cap. After and only after one method has all 110 completion receipts,
  `operations/qwen/modal_qwen_label_compare.py --canonical-only` scores the complete cohort twice through the
  pinned serial raw classifier. It refuses partial cohorts, mismatched methods/checkpoints, or non-512-token
  receipts and writes `label-comparison/canonical-512.json` without attack strings, target outputs, or prompts.
  After downloading that payload-free receipt, `qwen_whitebox_blade.py` validates the full 110-case denominator,
  source/model/classifier revisions, behavior uniqueness, and serial-label stability before writing BLADE rows,
  native ASR metrics, and a deterministic Markdown report. Invalid or unstable classifier cases remain visible
  but are excluded from the model-quality denominator.
- White-box BLADE generation now emits `evidence-manifest.json`, immediately reparses and recomputes the 110 rows,
  metrics, and report, and verifies every output hash and byte size. The completed MultiModalPGD remote bundle was
  backfilled and read back successfully in FDR; its compact local receipt records the Volume path and manifest hash.
- The Qwen white-box attack resume boundary validates every existing case receipt before skipping it: method/checkpoint
  identity, public hyperparameters, source index, paired image and hash, target-generation hash, executed-step/early-stop
  consistency, and optimization-checkpoint files must all survive readback. This prevents a partial case JSON from hiding
  an exact missing index when BlankImage or Patch is resumed after worker drain.
- The 512-token completion resume and scoring boundaries likewise validate the exact parent receipt/image hash chain,
  BF16 checkpoint and processor hashes, rendered prompt, deterministic sampling profile, completion token accounting,
  generation hash, and sibling parent filename set. New completion receipts use atomic replacement.
- A CPU-only `finalize_method` gate on the white-box app binds all 110 validated case receipts and images to the passed
  gradient/checkpoint receipt in `attack-manifest.json`. Canonical and replicate completion refuse to load the model until
  that manifest, its aggregate hashes, filenames, and every referenced case/image file pass readback.
- The white-box app's CPU-only `status_method` performs the same per-index validation without model load and returns a
  payload-free repair plan: valid/missing/invalid counts and indexes, extra-file counts, and the minimal shard IDs to resume.
- A second CPU-only finalizer on the completion app binds exactly 110 parent-matched canonical receipts to the attack
  manifest, BF16 checkpoint, processor files, deterministic sampling profile, and 512-token cap. Canonical scoring requires
  `completion-manifest-512.json` and rehashes all receipts before any classifier calls.
- The canonical scorer binds `canonical-512.json` to that completion-manifest hash, recomputes the entire 110-case summary,
  rejects duplicate identities and unsafe classifier receipt fields, validates before reuse, and persists atomically.
- New BlankImage/Patch BLADE inputs must include the completed run identity and completion-manifest SHA. The earlier PGD
  receipt is admitted only through an explicit legacy-PGD branch backed by its validated remote bundle; a partially filled
  modern provenance tuple fails closed.
- `qwen_whitebox_result.py` packages modern BlankImage/Patch results only after validating the attack, completion, score,
  and BLADE hash chain. Its normalized manifest reports native and BLADE denominators separately and performs deterministic
  manifest/report/evidence readback; historical PGD remains represented by its dedicated compact evidence receipt.
- `operations/qwen/modal_qwen_gcg_completion.py` is the post-generation target-completion stage for the matched Qwen
  GCG baseline. It mounts the same immutable BF16 checkpoint as optimization, reproduces the pinned upstream tokenizer
  chat template, uses deterministic 512-token generation, resumes by source-order index, rehashes the finalized GCG
  generation and shard receipts before model load, and requires a complete 400-receipt completion manifest. It is a
  separate app so deploying it cannot interrupt the active white-box image attack workers or the persistent GCG app.
- `operations/qwen/modal_qwen_gcg_score.py` rehashes that 400-receipt manifest, runs two serial passes through the pinned
  raw HarmBench classifier for ordinary behaviors, and preserves the upstream MinHash path for `hash_check` book/lyrics
  behaviors after the same classifier-tokenizer 512-token clip used by the Gym resource server, without copying
  generations into its output. Both the scorer and BLADE validator enforce the pinned 300
  classifier / 100 copyright split, including 50 book and 50 lyrics cases. `qwen_gcg_blade.py` then checks all model, source, scorer,
  denominator, and case-level hashes before emitting payload-free BLADE rows, native metrics, and a
  deterministic report; invalid or unstable classifier cases remain visible but outside the model-quality denominator.
  Its evidence manifest binds the classifier-score input and every output hash and is validated through a full readback.
- The persistent GCG app's CPU-only `status_campaign` validates every nested behavior artifact's shape without emitting
  its attack string, reports exact missing/invalid source indexes and unexpected entries, verifies complete shard receipts,
  and returns the minimal shard IDs required for a resumable repair before finalization.
- `report_method_run.py` recognizes a GCG Gym run as validated only when its complete finalized source receipt targets the
  same model, retains the public 500-step/512-width profile and shard evidence, reconciles all 400 healthy target rows,
  preserves the 300 classifier / 100 copyright split, and passes both classifier and MinHash calibration controls.
- `qwen_gcg_result.py` normalizes the matched baseline into the same comparison-facing manifest shape, verifies the full
  generation-to-completion-to-scorer-to-BLADE hash chain, keeps native and BLADE denominators distinct, and performs a
  deterministic manifest/report readback. Final Super-versus-Qwen claims must consume the two validated run manifests.
- `compare_gcg_results.py` accepts only those two validated manifests, hashes the shared public profile, rejects any
  fingerprint or denominator drift, and emits a checksummed payload-free comparison bundle. It reports the observed
  primary-minus-baseline ASR difference as descriptive evidence, not as statistical or causal superiority.
- For ZeroShot, `repair_capture_race.py` created a derivative that attached one late-but-present target-call
  capture without touching any response or score; a separate Gym health audit then passed 1,600/1,600.
  `restore_reverified_observability.py` keeps the original captured trajectory alongside stateless reverify scores
  only after all target responses match. `compare_reverify.py` records any label changes without payloads.
  `assess_classifier_batch_effect.py` binds the one ZeroShot raw-replay mismatch to 12 serial and 52 batched
  raw/chat probes across concurrency 2, 4, and 8, with exact token/clip parity. It does not alter the saved score;
  the report explicitly says `validated_with_caveat` rather than claiming exact 1,200/1,200 raw-label parity.
- `render_nvidia_method_report.py` produces one payload-free report per validated run. The campaign-level
  `render_nvidia_campaign_report.py` requires the exact four Ultra text manifests and two Super VL vision
  manifests, verifies all 5,340 rollouts are healthy, rejects any missing or mismatched target model, and renders
  a two-page held-out-test coverage map. It never pools ASR across different methods or modalities and explicitly lists the
  sixteen methods still awaiting full execution.
- `prepare_multimodal_source.py` found and recorded seven CSV filename-extension errors in the pinned 110-row
  multimodal set (`.jpeg`/`.jpg` in the CSV versus existing `.png` assets). It rewrites only those filenames into
  a generated CSV and emits an image-hash correction manifest; it does not alter the pinned source checkout.
- The verifier routes `hash_check` behaviors to the upstream-style book/lyrics sliding-window MinHash protocol,
  instead of misusing the harmfulness classifier. It loads reference pickles only from the verified pinned checkout.
  The old reference hashes are explicitly evaluated with their legacy permutation scheme under pinned
  `datasketch==2.0.0`; the verifier pins spaCy and its English model as well. A real benign-reference check and a
  synthetic positive/negative check pass. The 320-row held-out-test Ultra 3 HumanJailbreaks run also passed 400/400 case-level
  parity against the original HarmBench scorer.
- `harmbench_canary` is a separate two-row book/lyrics overlay so a verifier smoke cannot accidentally launch the
  2,000-row full-corpus HumanJailbreaks set. The September 17 Kimi K3 smoke produced 2/2 healthy Gym rollouts with
  `scoring_method=copyright_minhash`, no classifier calls, and zero sidecar failures. This is transport/scorer
  evidence, **not** a full ASR estimate.
- `harmbench_multimodal_direct` is a separate 110-row vision overlay. Its September 17 Super 3.5 VL full run
  preserved an image data URI in every Gym materialized input, returned the exact served Super VL model ID on all
  110 cases, and passed 110/110 rollout-health checks with zero sidecar failures. The deterministic
  `report_method_run.py` reconciled 54/110 classifier successes against the aggregate file and emitted a
  checksummed, payload-free validated report. `calibrate_multimodal_images.py` independently compared every
  generated PNG with the pinned upstream RGB/Lanczos/center-crop/Torchvision tensor round trip:
  **110/110 pixel-identical**. `calibrate.py` also confirmed 110/110 agreement with the raw-prompt classifier,
  including prompt-token and clip parity (plus 7/7 synthetic fixtures). Optional image-calibration dependencies
  are pinned in `requirements-image-parity.txt`.
- `harmbench_multimodal_render_text` is a separate 110-row vision overlay. Its deterministic-font generator pins
  the bundled DejaVu Sans font by SHA-256; `calibrate_multimodal_render_text.py` executes the pinned upstream
  rendering methods with the same font and confirms 110/110 pixel and instruction parity. The full Super VL run
  scored 110/110 cases with zero failures and 11/110 successes; raw-classifier labels, prompt tokens/hashes, and
  clip text/count agree 110/110. The source's default font selection is not reproducible across hosts, so this
  font pin is a disclosed profile difference.
- The Ultra 3 HumanJailbreaks result preserves the original 1,600 model responses and a separate fast-tokenizer
  reverify artifact. Exactly one clipped text string changed in that regrade; **zero target responses and zero
  success labels changed**. `calibrate_human_generation.py` reproduces 1,600/1,600 persisted upstream attack cases;
  the raw classifier path agrees 1,200/1,200 on non-copyright cases; the original spaCy 3.7.2/datasketch 1.5.9
  copyright scorer agrees 400/400. `report_method_run.py` checks all three receipts before marking this method's
  Ultra result validated. The original and reverified rollout SHA-256 values are both in the manifest.

## Target separation

- Nemotron 3 Ultra 550B (the FDR endpoint supplied by Kirsten) is a **text-only** target for applicable API methods.
  It must not be used or reported for the four multimodal methods.
- Nemotron 3.5 Super VL is the vision target. Authenticated live `/v1/models`, text, and synthetic-image requests
  revalidated its exact served ID as `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`; one image-bearing
  110-case full runs for each of MultiModalDirectRequest and MultiModalRenderText also succeeded. An inference
  API alone does not establish gradient access for
  `MultiModalPGD`, `MultiModalPGDPatch`, or `MultiModalPGDBlankImage`.

## Non-negotiable completion evidence

For every requested method and intended client model, retain the generated-case manifest, target-model identity,
sampling and query budget, full Gym rollouts, scorer receipts, failures, and per-behavior ASR. Upstream HarmBench
averages attack success within each behavior and then across behaviors; the verifier now reports that metric as well
as the per-test-case rate. Do not combine the 240-row DirectRequest classifier subset with a 320-row method result,
or report an API-only model as having completed a weight/gradient-based attack.
