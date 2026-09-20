# HarmBench ZeroShot — nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4

Status: **validated_with_caveat** · Rollout collection: **complete** · Protocol control: `passed_with_documented_batch_sensitive_classifier_case` · Run `zeroshot-full-20260917` · Upstream `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

Behavior-averaged attack success rate: **9.6%**. Raw successful cases: **154/1600** across **320** scored behaviors.

Cohort: HarmBench **320-row held-out text test split**, not the full 400-row public corpus (`harmbench_behaviors_text_all.csv` = 320 held-out + 80 validation). This rate is a held-out-test result and must not be presented or tabulated as a full-corpus result.

Expected/scored/failure/missing cases: **1600/1600/0/0**. Health: **1600 healthy**, 0 unhealthy, 0 unobserved.

Generation diagnostics: 0 empty, 870 truncated, 944 classifier-clipped. Score methods: {'copyright_minhash': 400, 'harmbench_classifier': 1200}.

One classifier case is batch-sensitive: its serial raw replay differs from the saved label, while raw and chat probes at 2-, 4-, and 8-way concurrency agree with the saved label. No score was overridden, and a separate reverify preserved every target response and score. See the checksummed batch-effect receipt.

This report contains no attack prompts or model generations. The checksummed run manifest and raw local artifacts are the case-level evidence. Do not compare this rate with a different method, cohort, target sampling profile, or uncalibrated image transform as though they were matched.
