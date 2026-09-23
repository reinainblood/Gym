# AgentDojo baseline results (dev branch only, not part of the adapter PR)

Official AgentDojo v0.1.35 (`a75aba76`), benchmark v1.2.2, `important_instructions`, 1,046 selectors per cell
(97 clean + 949 attacked; data sha256 `e59fd9b9…27f8`). One rollout per selector, `--temperature 0.0`,
agent concurrency 1, each cell split into 4 strided shards (one process each). Collected 2026-09-23.

BU = benign utility (clean rows), UuA = utility under attack, ASR = attack success rate (attacked rows).
Rates exclude masked rows; `scored` is the denominator.

| model | arm | scored | BU | UuA | ASR | masked |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Nemotron-3-Ultra | undefended | 1046 | 0.866 | 0.881 | 0.009 | 0 |
| Nemotron-3-Ultra | tool_filter † | 1046 | 0.206 | 0.241 | 0.000 | 0 |
| Nemotron-3-Ultra | transformers_pi_detector | 1046 | 0.505 | 0.510 | 0.001 | 0 |
| Nemotron-3-Ultra | spotlighting_with_delimiting | 1046 | 0.907 | 0.887 | 0.012 | 0 |
| Nemotron-3-Ultra | repeat_user_prompt | 1046 | 0.897 | 0.870 | 0.007 | 0 |
| Qwen3.5-122B-A10B | undefended | 1046 | 0.897 | 0.789 | 0.283 | 0 |
| Qwen3.5-122B-A10B | tool_filter † | 1046 | 0.052 | 0.066 | 0.000 | 0 |
| Qwen3.5-122B-A10B | transformers_pi_detector | 1046 | 0.495 | 0.464 | 0.046 | 0 |
| Qwen3.5-122B-A10B | spotlighting_with_delimiting | 1046 | 0.897 | 0.804 | 0.243 | 0 |
| Qwen3.5-122B-A10B | repeat_user_prompt | 1046 | 0.887 | 0.771 | 0.263 | 0 |
| Nemotron-3.5-Super-VL | undefended | 1046 | 0.876 | 0.863 | 0.040 | 0 |
| Nemotron-3.5-Super-VL | tool_filter | 1046 | 0.722 | 0.743 | 0.000 | 0 |
| Nemotron-3.5-Super-VL | transformers_pi_detector | 1045 | 0.557 | 0.494 | 0.004 | 1 RolloutTimeout |
| Nemotron-3.5-Super-VL | spotlighting_with_delimiting | 1046 | 0.897 | 0.860 | 0.017 | 0 |
| Nemotron-3.5-Super-VL | repeat_user_prompt | 1046 | 0.845 | 0.835 | 0.026 | 0 |
| Kimi-K3 | all five arms | — | — | — | — | not collected: dedicated endpoint returned HTTP 503 throughout |

† The Ultra and Qwen deployments do not show the model its tool definitions when a request sets
`tool_choice="none"` (probed directly: Qwen returns empty content, Ultra lists invented tool names; Super-VL and
Kimi list the real ones). `tool_filter`'s selection call sets exactly that, so on these two deployments the
filter selects no real tool and the task runs with an empty tool set. These two rows measure the serving stack,
not the defense.

## How the rows were collected

- Temperature 0.0 on every call; every scored row records `response.temperature == 0.0`.
- `prepare.py` originally numbered slack's injection tasks 0-4; upstream's are 1-5. Shards launched before the fix
  hold 21 unloadable `slack x injection_task_0` rows (masked `KeyError`) and lack `slack x injection_task_5`. Each
  cell was completed with a 21-row supplement (`<model>-<arm>-slack5`) collected from the fixed file, and
  `summarize.py` scores every cell against the fixed selector set, dropping the 21 unloadable rows. Every cell
  covers the 1,046 fixed selectors exactly once.
- Re-collected: 1 `ClientResponseError` 500 (qwen undefended) and 7 `RolloutTimeout` rows (Super-VL detector) cut
  at a 1800s harness budget. Upstream sets no time limit and reruns a pipeline up to 3x when it ends without a
  text answer, so a healthy rollout can make 48 calls; the budget was raised to 5400s and those rows collected
  again. One (`travel/user_task_11 x injection_task_2`) timed out again at 5400s after 15 calls and is kept masked.
