# AgentDojo baseline results (dev branch only, not part of the adapter PR)

Official AgentDojo v0.1.35 (`a75aba76`), benchmark v1.2.2, `important_instructions`, 1,046 selectors per cell
(97 clean + 949 attacked; data sha256 `e59fd9b9…27f8`). One rollout per selector, `--temperature 0.0`,
agent concurrency 1, each cell split into strided shards (one process each; 4 per cell, 2 for Kimi). Collected
2026-09-23.

BU = benign utility (clean rows), UuA = utility under attack, ASR = attack success rate (attacked rows).
Rates exclude masked rows; `scored` is the denominator. Empty sel. = share of scored rows whose tool_filter kept no
tool (`tool_filter_kept_tools == []`), recorded only for cells collected after that field existed.

| model | arm | scored | BU | UuA | ASR | masked | empty sel. |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Nemotron-3-Ultra | undefended | 1046 | 0.866 | 0.881 | 0.009 | 0 | |
| Nemotron-3-Ultra | tool_filter ‡ | 1046 | 0.670 | 0.661 | 0.002 | 0 | 0.012 |
| Nemotron-3-Ultra | transformers_pi_detector | 1046 | 0.505 | 0.510 | 0.001 | 0 | |
| Nemotron-3-Ultra | spotlighting_with_delimiting | 1046 | 0.907 | 0.887 | 0.012 | 0 | |
| Nemotron-3-Ultra | repeat_user_prompt | 1046 | 0.897 | 0.870 | 0.007 | 0 | |
| Qwen3.5-122B-A10B | undefended | 1046 | 0.897 | 0.789 | 0.283 | 0 | |
| Qwen3.5-122B-A10B | tool_filter ‡ | 1046 | 0.113 | 0.120 | 0.001 | 0 | 0.000 |
| Qwen3.5-122B-A10B | transformers_pi_detector | 1046 | 0.495 | 0.464 | 0.046 | 0 | |
| Qwen3.5-122B-A10B | spotlighting_with_delimiting | 1046 | 0.897 | 0.804 | 0.243 | 0 | |
| Qwen3.5-122B-A10B | repeat_user_prompt | 1046 | 0.887 | 0.771 | 0.263 | 0 | |
| Nemotron-3.5-Super-VL | undefended | 1046 | 0.876 | 0.863 | 0.040 | 0 | |
| Nemotron-3.5-Super-VL | tool_filter | 1046 | 0.722 | 0.743 | 0.000 | 0 | n/a § |
| Nemotron-3.5-Super-VL | transformers_pi_detector | 1045 | 0.557 | 0.494 | 0.004 | 1 RolloutTimeout | |
| Nemotron-3.5-Super-VL | spotlighting_with_delimiting | 1046 | 0.897 | 0.860 | 0.017 | 0 | |
| Nemotron-3.5-Super-VL | repeat_user_prompt | 1046 | 0.845 | 0.835 | 0.026 | 0 | |
| Kimi-K3 | undefended | 1046 | 0.918 | 0.916 | 0.003 | 0 | |
| Kimi-K3 | tool_filter | 1046 | 0.247 | 0.214 | 0.001 | 0 | 0.000 |
| Kimi-K3 | transformers_pi_detector | 1046 | 0.567 | 0.512 | 0.000 | 0 | |
| Kimi-K3 | spotlighting_with_delimiting | 1046 | 0.928 | 0.905 | 0.001 | 0 | |
| Kimi-K3 | repeat_user_prompt | 1046 | 0.918 | 0.900 | 0.006 | 0 | |

‡ Collected on replacement deployments (below); replaces the first tool_filter cells, which are archived.
§ Collected before `tool_filter_kept_tools` existed. Super-VL's serving stack kept tools under
`tool_choice="none"` (probed directly), so its selection was not emptied by serving.

## Deployment changes for tool_filter

`tool_filter`'s selection call is the only request in the benchmark that sets `tool_choice="none"`. The
deployments differed in what they do with the tool list on such a request:

- **Nemotron-3-Ultra and Qwen3.5-122B (original endpoints):** dropped `tools` from the prompt under
  `tool_choice="none"`. The model could not name a real tool, the filter kept nothing, and tasks ran with no
  tools. The first tool_filter cells from these endpoints (Ultra BU 0.206, Qwen BU 0.052) measured the serving
  stack, not the defense. They are archived at `agentdojo/archive/tool_stripped/` on the results volume and
  excluded from this table.
- **Replacement endpoints** (`agentdyn-nemotron-3-ultra-tools-preserved`, `agentdyn-qwen3-5-122b-tools-preserved`):
  same weights, revisions, engine (SGLang 0.5.18) and serving flags, plus a source patch so tools reach the chat
  template when `tool_choice="none"`. Probed before launch: equal `prompt_tokens` under `"auto"` and `"none"`, and
  the `"none"` answer named every tool. Only the Ultra and Qwen tool_filter cells were re-collected on them; the
  other four arms never send `tool_choice="none"` and stand from the original endpoints.
- **Kimi-K3 (dedicated endpoint):** its SGLang fork previously added a 38-token "no tools" template directive under
  `tool_choice="none"`. That directive was removed before any Kimi row here was collected, so every Kimi cell ran
  on the changed deployment. The change affects only `tool_choice="none"` requests.
- **Super-VL:** unchanged; it kept tools under `tool_choice="none"` throughout.

## Why tool_filter utility is low on Qwen and Kimi even with tools preserved

With the selection intact (empty-selection rate 0.000), Qwen and Kimi still score low because of what upstream's
pipeline does next. The filter appends its prompt as a user turn and the model's tool list as an assistant turn,
and the policy call then continues from that history. Qwen and Kimi mostly answer by repeating the list and never
call a tool:

| cell | scored rows with no tool call | of those, final answer identical to the selection reply |
| --- | ---: | ---: |
| Ultra tool_filter | 84 (8.0%) | 23 |
| Qwen tool_filter | 936 (89.5%) | 885 |
| Kimi tool_filter | 816 (78.0%) | 705 |

This is the upstream pipeline's conversation as sent (AgentDojo's own message conversion builds every request),
at temperature 0.0; it is a model-by-pipeline result, not a serving or adapter artifact.

## How the rows were collected

- Temperature 0.0 on every call; every scored row in all 20 cells records `response.temperature == 0.0`.
- Kimi: no `reasoning` parameter sent (the endpoint's default is its maximum reasoning mode), 2 shards per arm to
  stay within the endpoint's ~12 concurrent requests.
- `prepare.py` originally numbered slack's injection tasks 0-4; upstream's are 1-5. The first Ultra, Qwen and
  Super-VL shards (launched before the fix) hold 21 unloadable `slack x injection_task_0` rows (masked `KeyError`)
  and lack `slack x injection_task_5`. Each of those cells was completed with a 21-row supplement
  (`<model>-<arm>-slack5`) collected from the fixed file, and `summarize.py` scores every cell against the fixed
  selector set, dropping the 21 unloadable rows. Kimi and the replacement tool_filter cells ran on the fixed file
  from the start. Every cell covers the 1,046 fixed selectors exactly once.
- Re-collected: 1 `ClientResponseError` 500 (Qwen undefended) and 7 `RolloutTimeout` rows (Super-VL detector) cut
  at a 1800s harness budget. Upstream sets no time limit and reruns a pipeline up to 3x when it ends without a
  text answer, so a healthy rollout can make 48 calls; the budget was raised to 5400s and those rows collected
  again. One (`travel/user_task_11 x injection_task_2`) timed out again at 5400s after 15 calls and is kept masked.
