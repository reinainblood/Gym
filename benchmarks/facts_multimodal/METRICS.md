# FACTS Multimodal metrics

The headline metric is binary accuracy: essential-fact coverage must be strictly greater than 0.5 and the response must receive a no-contradiction verdict. Coverage and factuality are reported separately to show which gate failed.

## Protocol fidelity

- Dataset: the 711-row Apache-2.0 public CSV, pinned by SHA-256 in `prepare.py`.
- Sampling: one rollout per runnable task, matching the paper's per-question evaluation.
- Image transport: identical validated image bytes are embedded for the policy model and factuality judge.
- Coverage: one yes/no decision per essential rubric fact, aggregated by code.
- Factuality: one image-aware contradiction verdict over all rubric facts.
- Exclusions: only image download or validation failures; every exclusion is written to `image_download_report.csv`.

## Known divergence from the leaderboard

The paper does not release the production autorater model or exact prompts. This adapter uses prompts adapted from the public Kaggle example and makes the judge model explicit in Gym configuration. Public-split results must name that judge and the prepared-dataset hash. They are not directly comparable to the combined public-and-private leaderboard values.

The committed example rollouts are five real Kimi K3 outcomes from the recovered baseline and include both passing and failing rows. Full model results belong in external run packages rather than this source tree.
