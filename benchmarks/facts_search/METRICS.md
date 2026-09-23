# FACTS Search metrics

The primary metric is FACTS F1, the harmonic mean of overall accuracy and attempted accuracy. The grader labels an
answer `A` for correct, `B` for incorrect, or `C` for not attempted. Attempted accuracy is `A / (A + B)`, and hedging
rate is the fraction labeled `C`.

Average search turns matches the notebook's `n_hops` counter; parallel queries in the same model turn count as one
search turn. The adapter also reports the number of individual queries, forced-final rate, empty and truncated answer
rates, judge-format validity, and a 95% interval from 1,000 full-size bootstrap resamples with seed 0.

The leaderboard parser reads the first uppercase `A`, `B`, or `C` in the grader response and otherwise assigns `C`. A
second, strict parser records whether the grader returned the requested one-letter response. Judge transport failures
are written to Gym's failure sidecar.

## NVIDIA model results

We ran FACTS Search on four models using all 890 downloadable Search-On questions, Brave Web Search, the grader prompt
pinned in this repository, Gemini 3.5 Flash, and one rollout per question.

| Model | F1 | 95% CI | Correct | Incorrect | Not attempted | Valid judge rows | Empty | Truncated | Average hops | Average queries | Forced final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moonshotai/Kimi-K3` | **91.21%** | 89.41%–92.91% | 799 | 63 | 28 | 889/890 | 0 | 0 | 2.34 | 6.34 | 10.79% |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | **81.08%** | 78.69%–83.41% | 645 | 56 | 189 | 890/890 | 0 | 0 | 4.07 | 4.45 | 31.35% |
| `Qwen/Qwen3.5-122B-A10B-FP8` | **80.49%** | 78.03%–83.04% | 660 | 90 | 140 | 890/890 | 52 | 67 | 3.10 | 6.32 | 14.49% |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | **80.18%** | 77.71%–82.47% | 641 | 68 | 181 | 890/890 | 0 | 0 | 4.17 | 6.57 | 33.60% |
