# HarmBench PAP-top5 — nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4

Status: **validated** · Rollout collection: **complete** · Protocol control: `passed_upstream_pap_classifier_copyright_controls` · Run `pap-full-raw-20260917` · Upstream `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

Behavior-averaged attack success rate: **5.6%**. Raw successful cases: **90/1600** across **320** scored behaviors.

Cohort: HarmBench **320-row held-out text test split**, not the full 400-row public corpus (`harmbench_behaviors_text_all.csv` = 320 held-out + 80 validation). This rate is a held-out-test result and must not be presented or tabulated as a full-corpus result.

Expected/scored/failure/missing cases: **1600/1600/0/0**. Health: **1600 healthy**, 0 unhealthy, 0 unobserved.

Generation diagnostics: 0 empty, 1262 truncated, 1311 classifier-clipped. Score methods: {'copyright_minhash': 400, 'harmbench_classifier': 1200}.

This report contains no attack prompts or model generations. The checksummed run manifest and raw local artifacts are the case-level evidence. Do not compare this rate with a different method, cohort, target sampling profile, or uncalibrated image transform as though they were matched.
