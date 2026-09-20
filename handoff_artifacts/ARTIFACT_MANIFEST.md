# Artifact manifest

All paths below are relative to this handoff branch. Raw artifacts were checked
for obvious bearer/API-key patterns before staging. They are controlled
evaluation artifacts, not public-release assets.

## Locally recovered raw packages

| Path | What it contains | Result status |
| --- | --- | --- |
| `local-results/agentdyn-baselines/` | Four full 620-selector undefended baselines, materialized inputs, rollout rows, score/agent metrics, and failures sidecars. | Complete baseline evidence; defense treatments not included. |
| `local-results/facts_grounding_v2_qwen_openrouter_20260919/direct_rollouts.jsonl` | Qwen recovery rollouts. | Requires refreshed-judge provenance reconciliation. |
| `local-results/facts_parametric_qwen_recovery_20260919/` | Qwen recovery plus retries/failure sidecars. | Use `resolved_recovery.jsonl`; zero-byte failure sidecars are expected. |

## FDR snapshots collected at handoff

| Path | FDR source volume | What it contains | Interpretation |
| --- | --- | --- | --- |
| `fdr-snapshots/safe-child-qwen-verify.jsonl` and `safe-child-super-vl.jsonl` | `safe-child-annotation-data/runs/` | Qwen and Super five-round response JSONL files. | Raw responses only; no completed human-annotation score. |
| `fdr-snapshots/kidbench-super-vl-ara-results` | `kidbench-super-vl-ara-results` | Super-VL multi-turn actor output, metrics, and review. | Single-model multi-turn evidence; separate from four-model single-turn leaderboard. |
| `fdr-snapshots/HARMBENCH_FDR_LOCATIONS.md` | Three white-box result volumes | Exact paths to Ultra, Qwen, and Kimi work-in-progress artifacts. | Snapshot/locator rather than a claim of 22-method coverage. |

## External source packages

See `EVAN_DRIVE_BUNDLES.md`. The original archives remain in shared Drive.
They are too large for ordinary Git (and this checkout has no Git LFS), so this
branch stores exact retrieval metadata instead of a misleading incomplete copy.

## Over-refusal audit result

The FDR app and volume inventory contained no run named `xstest`,
`over_refusal`, or an equivalent result archive. The local checkout contained
only committed example fixtures. Dedicated branch ledgers say **no four-model
production result located** until an artifact is recovered.
