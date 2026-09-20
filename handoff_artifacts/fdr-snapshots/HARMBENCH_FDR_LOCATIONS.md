# HarmBench FDR artifact locations

Checked read-only in Modal environment `FDR` on 2026-09-20. These are preserved
run/attack artifacts, not automatically scored benchmark rows.

| Volume | Exact path(s) observed | State |
| --- | --- | --- |
| `harmbench-ultra-whitebox-results` | `ultra-bf16-gradient-20260918-v5/`, `ultra-gcg-canary-20260918b/`, `ultra-gcg-canary-20260918c/`, `ultra-gcg-canary-20260918d/`, `ultra-gcg-full-20260918a/` | Ultra GCG has runtime configs and a partial/full-run checkpoint directory. |
| `harmbench-qwen35-122b-whitebox-results` | `qwen35-122b-harmbench-whitebox-20260919/gradient-receipt.json`, `MultiModalPGD/`, `MultiModalPGDPatch/` | Gradient receipt plus in-progress multimodal PGD/patch case, image, and checkpoint directories. |
| `harmbench-kimi-k3-whitebox-results` | `checkpoint-manifest.json`, `checkpoint-preflight.json`, `failures/k3-text-gradient-20260919g.json` | Preflight/checkpoint metadata and a failure receipt; not a method score. |
| `harmbench-super-vl-gradient-canary-results` | volume root | Existing Super-VL gradient canary volume; inspect before reuse. |

The active/deployed FDR apps observed in this same snapshot were:

- `harmbench-ultra-whitebox` — `ap-5ZFVUo30h7lpDWz1KmbsgE`
- `harmbench-kimi-k3-whitebox` — `ap-oFHR4MrhEMJoNzswfvtBPD`
- `harmbench-qwen35-122b-whitebox` — `ap-07zo5VR8b4cd4UHMb6VHBM`

Before any continuation, re-check live task counts and artifact modification
times. Do not treat a deployed app, checkpoint, or case JSON as a completed
method outcome.
