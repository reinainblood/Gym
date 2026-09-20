# NVIDIA Safety — master Claude handoff

**Snapshot:** 2026-09-20 CDT
**Repository:** `https://github.com/reinainblood/Gym.git`
**Handoff branch:** `codex/nvidia-safety-claude-handoff`

This is the continuity point for the NVIDIA safety portfolio. It is deliberately
evidence-first: a row is only called complete when its output is present and
its model, harness, denominator, and scoring route are recoverable. An adapter,
a report template, or a deployed endpoint is not a completed benchmark result.

## First five minutes

```bash
git fetch origin
git switch codex/nvidia-safety-claude-handoff
git submodule update --init --recursive
```

Read this file, then `handoff_artifacts/ARTIFACT_MANIFEST.md`. Raw shareable
artifacts collected from this Mac and FDR live under `handoff_artifacts/`.
No API key, access code, cookie secret, or endpoint credential belongs in Git.

New Modal work is **FDR only**. Existing shared endpoints are dependencies:
do not stop, resize, or redeploy them while evaluating a model.

## Portfolio ledger

| Benchmark | Code / evidence branch | Verified state at handoff | Claude's next concrete move |
| --- | --- | --- | --- |
| AgentDyn | `codex/feature-add-agentdyn` @ `a85739cc` | Four 620-row undefended baselines are complete and copied into this handoff. Defense treatments remain separate/incomplete. | Use copied raw results; continue only defense rows with explicit treatment labels. |
| HarmBench | `feature-add-harmbench` @ `8fed5091` | Six historical methods are reportable. FDR holds gradient artifacts for Ultra, Qwen, and Kimi; they are snapshots, not a 22-method completion claim. | Reconcile copied volume snapshots; preserve native method semantics and keep white-box/transfer lanes distinct. |
| FACTS Parametric | `feature-add-facts-parametric` @ `6377e3d9` | Evan's BLADE bundle exists; a local Qwen recovery artifact is copied here. Earlier empty-Qwen rows must not enter a result table. | Audit recovery package and Evan bundle, then compute/report only reconciled denominators. |
| FACTS Search | existing implementation / prior report | Previously validated Super lane; no new raw local artifact was found in this checkout. | Locate the prior run package before expanding claims. |
| FACTS Grounding v2 | `feature-add-facts-grounding-v2` @ `e7eb5367` | Valid Kimi report lineage and a Qwen/OpenRouter recovery artifact are copied here. A paper-pinned old judge is unavailable; refreshed judge use must be labeled. | Reconcile Qwen/refresh provenance and score only after the judge route is recorded. |
| FACTS Multimodal | `feature-add-facts-multimodal` @ `329c88c6` | Evan's 8.52 GB Super bundle is indexed below, not vendored. | Audit/download from Drive; do not invent missing rows from an adapter. |
| Political Even-handedness | `feature-add-even-handedness` @ `8daa4ed8` | Evan's 139 MB Super bundle is indexed below. | Audit its run and maintain the option-token-probability versus discrete-choice boundary. |
| ToolAlignBench | `feature-add-toolalignbench` @ `54bdccfa` | Evan's 241 MB Super bundle is indexed below; earlier validation was not five-trial coverage. | Recover bundle, then run/reconcile five matched trials. |
| InjecAgent | `feature-add-injecagent` @ `bb1512e1` | Kimi reportable line exists. | Preserve first-tool-call protocol; expand models only with a matched denominator. |
| KIDBench | `feature-add-kid-bench` @ `7f6091b8` | Four-model scored leaderboard/reports are committed on the branch. A Super-VL multi-turn FDR artifact is copied here. | Use existing reports; do not merge multi-turn actor results with single-turn results. |
| Safe-Child-LLM | `codex/safe-child-llm` @ `06e6bd93`; portal code `codex/safe-child-annotation-app` @ `d48d1dab` | Qwen and Super five-round response files are copied from FDR; Kimi and Ultra raw files were not present in the volume. Human annotation is incomplete. | Obtain the private access code through the project channel; preserve model × prompt × round. |
| VERA-MH | `feature-add-vera-mh` @ `6c4e0b7f` | Kimi reportable line exists. | Use official rubric route and retain calibration provenance. |
| ExploitBench | `codex/add-exploitbench` @ `02bc198c` | Code is committed; no reconciled local run artifact was found in this checkout. | Locate/reconcile external run receipts before calling it scored. |
| Agent Security Bench (ASB) | `codex/asb-modal-campaign` @ `8d93a635` | Source/input/runtime preflight only; **zero score rows or trajectories** at this snapshot. | Read the existing `handoff.md` on that branch. Build/run the public matrix; do not call a receipt a benchmark score. |

## Evan's package: exact durable source

Evan's Drive folder is [BLADE Bundles](https://drive.google.com/drive/folders/1tCLHYqBO4-OMUG0OdZp8_j74Bf4INobK).
The package inventory, IDs, sizes, and direct Drive URLs are checked into
`handoff_artifacts/EVAN_DRIVE_BUNDLES.md`.

The large multimodal ZIP is **8,515,569,157 bytes (8.52 GB)**. It cannot be
checked into ordinary GitHub Git: this repository has no Git LFS installed and
GitHub rejects normal blobs above 100 MB. The manifest is the checked-in,
versioned retrieval contract; do not split an 8.5 GB archive into Git blobs.
The three smaller Evan ZIPs are also intentionally left in Drive so their
original shared location, ownership, and revision remain intact.

## Over-refusal: two dedicated branches, and the honest result state

The two requested environments are now isolated on dedicated documentation and
result-ledger branches:

- `codex/over-refusal-detection-handoff`
- `codex/xstest-handoff`

Both point to the exact repository implementations and describe the proper
launch/configuration. I searched this checkout, its sibling worktrees, and FDR
app/volume names on 2026-09-20. I found only fixture/example files, **not a
four-model production result artifact for either benchmark**. Do not claim
they ran last night until a JSONL/metrics receipt with all target model IDs is
recovered. Their dedicated branches make that absence obvious rather than
burying it in the master handoff.

Target matrix whenever those runs are found or re-run:

1. `moonshotai/Kimi-K3`
2. `Qwen/Qwen3.5-122B-A10B-FP8`
3. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
4. `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`

## Important guardrails for Claude

- Never conflate a model response with a judged result, a completed adapter
  with executed coverage, or an FDR/provider failure with model behavior.
- Preserve exact model ID, source revision, input/selector hash, agent,
  sampling, judge/scorer, repeat count, raw output, and failure sidecar.
- Keep ordinary model output and child-safety human annotations private except
  where the controlled internal repo is the agreed collaboration channel.
- Do not commit secrets or share the Safe-Child portal access code in a file.
- Do not publish, open a PR, message benchmark authors, or alter a shared
  deployment unless Kirsten explicitly requests it.

## What is actually in this branch

- `handoff_artifacts/local-results/agentdyn-baselines/`: all four complete
  baseline raw/metric packages collected on this Mac.
- `handoff_artifacts/local-results/facts_grounding_v2_qwen_openrouter_20260919/`:
  Qwen recovery rollout artifact.
- `handoff_artifacts/local-results/facts_parametric_qwen_recovery_20260919/`:
  Qwen recovery/retry artifacts.
- `handoff_artifacts/fdr-snapshots/`: FDR Safe-Child and KIDBench response
  snapshots copied at handoff time, plus a precise HarmBench source-volume
  manifest for in-progress artifacts.
- `handoff_artifacts/EVAN_DRIVE_BUNDLES.md`: Evan's source bundles and exact
  recovery locations.
- `handoff_artifacts/ARTIFACT_MANIFEST.md`: source locations and contents
  needed to decide what is a result versus a diagnostic.
