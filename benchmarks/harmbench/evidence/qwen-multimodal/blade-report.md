# HarmBench BLADE Deterministic Report

## Reconciliation

- Total observed rows: **220**
- Valid model-quality denominator rows: **220**
- Excluded rows: **0**

## Per-Method Native Metrics

| Public method | ASR numerator | ASR denominator | Test-case ASR | Behavior-averaged ASR | Excluded |
|---|---:|---:|---:|---:|---:|
| MultiModalDirectRequest | 46 | 110 | 41.8182% | 41.8182% | 0 |
| MultiModalRenderText | 35 | 110 | 31.8182% | 31.8182% | 0 |

## Attribution Boundary

Only rows with `reconciliation_disposition=included_model_denominator` enter ASR. Infrastructure, task/verifier, and data-artifact exclusions remain separately counted.
