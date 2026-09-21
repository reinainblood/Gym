# HarmBench method expansion — execution ledger

This ledger is intentionally stricter than a feature checklist. A method is **covered** only after its upstream
attack generator, exact experiment, target-model interface, Gym rollout, correct scorer, and case-level source
receipt have all been exercised. The 23 names below are the methods requested by NVIDIA. The source of truth is
`centerforaisafety/HarmBench` at `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The two "Fresh ... against
the client model" entries are NVIDIA-requested client-targeted variants of upstream PAIR/TAP, not separate proper
method keys in the paper's pipeline.

| Requested method | Generator / client requirement | Verified state in this branch |
|---|---|---|
| DirectRequest | Single-turn text request | **Validated held-out-test Ultra 3 run:** pinned upstream `default` experiment over all 320 test behaviors (159 standard, 81 contextual, 80 copyright), 320/320 healthy rollouts, 27 successful cases, 8.4375% behavior-averaged ASR, 240/240 raw-classifier and 80/80 copyright scorer agreement. The full 400-row source and exact 80-row validation extension are generated, set-difference reconciled against the prior rollouts, and stored in FDR. Validation collection is blocked only on restoring the original Ultra endpoint credential binding. |
| HumanJailbreaks | Pinned human templates, random subset 5 | **Validated held-out-test Ultra 3 run:** 1,600/1,600 healthy rollouts across all 320 text-test behaviors; 73 successful cases (4.5625% behavior-averaged ASR), 315 truncated and 355 classifier-clipped. The full 2,000-case source and exact 80-behavior / 400-case validation extension are generated, set-difference reconciled against the prior rollouts, and stored in FDR. Validation collection is blocked only on restoring the original Ultra endpoint credential binding. |
| ZeroShot | Upstream Mixtral attacker model | **Validated held-out-test result with explicit classifier-batching caveat:** FDR Mixtral/Triton generated all 1,600 attacks across 320 text behaviors; upstream generation replay matched 320/320. Ultra scored 1,600/1,600 healthy cases with 9.625% behavior-averaged ASR and 154 successful cases. Copyright parity passed 400/400. Raw classifier replay matched 1,199/1,200; the sole mismatch had identical prompt token IDs and clipping and was proven to flip with vLLM batch concurrency (serial Yes, 2-/4-/8-way No in both raw and chat paths). No score was overridden; reverify preserved all target responses and labels. The missing 80 validation behaviors still require generation and scoring for a full-corpus result. |
| PAP-top5 | Upstream Mixtral attack model, pinned five-technique taxonomy | **Validated held-out-test Ultra 3 run:** the corrected FDR attacker retained all 1,600 raw generations, and the pinned upstream PAP method body reproduced 1,600/1,600 cases. After a one-label stateless reverify, the saved target responses yielded 1,600/1,600 healthy rollouts, 90 successful cases and 5.625% behavior-averaged ASR. Raw classifier and copyright controls matched 1,200/1,200 and 400/400 respectively. The 512-token target cap was reached on 1,262 cases. The generator now requires the full 400-row source; the missing 80 validation behaviors remain unrun. |
| TAP-Transfer | Fixed upstream transfer experiment | Bridge and source-experiment receipt only; no generator run or model rollout. |
| MultiModalDirectRequest | Image-bearing target | **Validated full run:** 110/110 healthy Super 3.5 VL Gym rollouts, zero sidecar failures, 54 classifier successes (49.1% ASR), 3 truncated/clipped. All 110 images are pixel-identical to upstream Torchvision, and raw-vs-Gym classifier labels, prompt tokens/hashes, and clip text/count agree 110/110. |
| MultiModalRenderText | Image-bearing target | **Validated full Super VL run:** 110/110 healthy rollouts, zero failures, 11 classifier successes (10.0% ASR), one truncated/clipped. Upstream render functions reproduce 110/110 image pixels and instructions when bound to the declared DejaVu Sans font; classifier controls match 110/110. The fixed font replaces upstream's host-dependent first-system-font lookup. |
| GCG | Local target weights and gradients | **Target policy changed by Kirsten:** run the full public protocol on unreleased Nemotron 3.5 Super VL 120B BF16 as the primary result and Qwen 3.5 122B-A10B BF16 as the single matched baseline. Ultra and Kimi are excluded from new GCG launches and denominators. The persistent FDR runtime pins both complete checkpoints, upstream commit, 500 steps, search width 512, compiled recurrent kernels, resumable behavior shards, and a Gym-compatible finalizer. A full-budget Super one-behavior canary is active; no full-corpus result exists yet. |
| GCG-Multi | Local target weights; upstream run IDs 0–4 | Bridge enforces all five source runs before merging; no generator run or model rollout. |
| AutoPrompt | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| GBDA | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| PEZ | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| UAT | Local target weights and gradients | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| AutoDAN | Local target weights | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| FewShot | Upstream open-weight target configuration | Bridge and weight-only compatibility gate; no generator run or model rollout. |
| PAIR | Upstream iterative attacker and judge | Bridge and target-type mapping only; no generator run or model rollout. |
| TAP | Upstream tree-search attacker and judge | Bridge and target-type mapping only; no generator run or model rollout. |
| GCG-Transfer | Fixed upstream transfer ensemble; run IDs 0–4 | Bridge enforces all five source runs before merging; no generator run or model rollout. |
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
