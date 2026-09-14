# FACTS Grounding v2 — metrics and reading guide

## 1. What the benchmark measures

FACTS Grounding tests whether a model can write a long-form answer to a user request using *only* a supplied
context document (up to 32k tokens; finance, technology, retail, medical, legal domains; Q&A, summarization and
rewriting tasks). Version 2 (FACTS Benchmark Suite, section 6) keeps the v1 prompts and replaces the judges: Gemini
2.5 Flash and GPT-5 with a revised prompt. Two things are judged and kept distinct:

- eligibility: does the response actually address the request (guards against short, evasive answers that would be
  trivially "grounded")?
- groundedness: is every information-bearing sentence supported by the document?

- Unit of analysis: one prompt, answered once. Judges are consulted in a fixed order.
- Public protocol scope: the 856-prompt public set (Kaggle release v17). The 859-prompt private set is held by
  Kaggle; nothing here is a leaderboard submission. The December 2024 Hugging Face mirror (860 rows) contains four
  rows that are absent from the current official release; those four are not scored.
- Slices: `domain` (Medical 236, Legal 192, Financial 156, Internet/Technology 156, Retail/Product 92, Unknown 24),
  `high_level_type` (Q&A 508, Text Transformation 348), `type` (10 task types).

## 2. Sources, pins, and licenses

| Item | Value |
|---|---|
| Paper | Cheng et al., arXiv:2512.10791v1 (CC BY 4.0), section 6 "FACTS Grounding v2"; prompts from Jacovi et al., arXiv:2501.03200 (`paper/PAPER.md`; PDF SHA-256 `db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be`) |
| Public data | Kaggle `deepmind/FACTS-grounding-examples` v17 (2026-01-07), Apache 2.0 / CC-BY 4.0; `examples.csv` SHA-256 `66990aea86a91d40edcda6ebcd0d686674246dc7c07338ed8c17e1c275b1e096`, 856 rows; `evaluation_prompts.csv` (v1 prompts, SHA-256 `43a4bc5083275ca1a3d95eed8b26c0760a50dacac5ee19856af887b36d120069`, identical to the Hugging Face copy) |
| Reference implementation | Kaggle starter `prathameshbang/facts-grounding-v2-benchmark-starter` v4 (author on the paper); v2 prompts vendored byte-exactly under `resources_servers/facts_grounding_v2/prompts/` (SHA-256 pinned in `app.py`); parsers vendored verbatim in `upstream_control.py` (cell hashes recorded there) |
| Judges | `google/gemini-2.5-flash` then `openai/gpt-5-2025-08-07` (starter `JUDGES`; paper section 6.2) |
| Gym revision | `1e668906d2e69a9e8ee9aaafc60050a4025d9688` plus this branch |

## 3. Metric dictionary

Per answer, with judges J = (gemini-2.5-flash, gpt-5):

- eligibility: for each judge in order, the judge writes its own baseline answer to `full_prompt`, then rates the
  policy answer against that baseline with the v2 "Instruction Following" prompt (context redacted). The verdict is the
  first well-formed `{"Instruction Following": ...}` object; `Invalid` if none. The answer is eligible as soon as a
  judge's verdict is not `Major Issue(s)` (so `Invalid` counts as eligible); later judges are not asked.
- grounding: each judge labels every sentence (`supported`, `not_supported`, `no_rad`; missing label -> `unknown`);
  a judge's verdict is *grounded* iff at least one sentence parsed and none is `not_supported`; an unparseable line
  containing `"not_supported"` counts as a not_supported sentence; a reply with no parseable sentence is *not grounded*.
- score = mean of the judges' grounded verdicts if eligible, else 0.

| Metric | Formula / rule | Range, direction | Denominator, exclusions | Role |
|---|---|---|---|---|
| `factuality_score` (paper "adjusted" / leaderboard score) | mean over scored answers of `reward` = (eligible ? mean judge verdict : 0) | 0-1, higher better | scored answers; judge-failed rows are in the sidecar and excluded | primary |
| `unadjusted_factuality_score` (paper "unadjusted") | mean judge verdict over all graded answers, ineligible included | 0-1, higher better | answers with grounding verdicts (all, because `grade_ineligible_responses: true`) | component |
| `eligibility_rate` | eligible answers / N | 0-1, higher better | N | component |
| `grounded_rate_eligible/<judge>` | eligible answers the judge found fully supported / eligible answers | 0-1, higher better | eligible answers | component |
| `grounded_all_judges_rate_eligible` | eligible answers grounded by both judges / eligible | 0-1 | eligible | component |
| `judge_disagreement_rate_eligible` | eligible answers with exactly one grounded verdict / eligible | 0-1, neutral | eligible | diagnostic |
| `eligibility_decided_by/<judge>` | answers whose eligibility was settled by that judge / N | 0-1 | N | diagnostic |
| `eligibility_rating/<judge>/<rating>` | rating share among answers that judge rated | 0-1 | rated by that judge | diagnostic |
| `eligibility_invalid_rate` | answers with an `Invalid` eligibility verdict / N | 0-1, lower better | N | diagnostic |
| `grounding_parse_empty_rate/<judge>`, `grounding_unparseable_line_rate/<judge>` | replies with no parsed sentence / replies with at least one unparseable line | 0-1, lower better | answers graded by that judge | diagnostic |
| `factuality_score_ci95_low/high` | bootstrap (2,000 resamples, fixed seed) of the mean reward | - | N | uncertainty |
| `generation_truncated_rate`, `generation_empty_rate`, `policy_input_tokens_max/mean` | output-cap and context accounting | - | N | operational |
| `factuality_score/<domain|high_level_type|type>/<v>` | adjusted score within the slice | 0-1 | answers in the slice | slice |

Invalid handling: judge transport/HTTP failures raise `JudgeError` -> failures sidecar, excluded, retried on
`--resume`. Unparseable eligibility verdicts (`Invalid`) and unparseable grounding replies are *not* failures: they
follow the starter's rules (eligible / not grounded) and are counted in the diagnostics.

## 4. Paper/source field to Gym mapping

| Source | Gym |
|---|---|
| CSV `full_prompt` | `responses_create_params.input` = one user message (starter `llm.prompt(full_prompt)`) |
| CSV `user_request`, `context_document` | eligibility `{user_request}`; grounding `{user_query}`, `{context}` |
| CSV `system_instruction`, `domain`, `type`, `high_level_type` | row fields (system_instruction is part of `full_prompt`; slices) |
| starter `judge.prompt(full_prompt)` (baseline) | `judge_receipts[stage=baseline]` |
| starter `_evaluate_quality_no_context` | `judge_receipts[stage=eligibility]`, `eligibility_ratings`, `eligible`, `eligibility_deciding_judge` |
| starter `classify_grounding` / `parse_ufg_rev21_verdict` | `judge_receipts[stage=grounding]` with `sentences`, `grounding_verdicts`, `grounding_label_counts`, parse counters |
| starter `grounding_score` | `grounding_score`, `unadjusted_score`, `reward` |

## 5. Calibration

`calibrate.py` writes `calibration/upstream-vs-gym.jsonl` and `calibration/summary.json`:

- `replay`: every eligibility/grounding receipt is re-parsed with the vendored starter parsers and the starter's
  eligibility and score rules are re-applied; must agree exactly with the Gym decisions.
- `live`: a stratified subset (ineligible, grounded, split, unsupported) is re-judged with the starter loop against the
  same judge endpoints; eligibility, per-judge verdicts and scores are compared (judges sample; this measures
  stability).

## 6. Kimi K3 run fingerprint

<!-- fingerprint:start -->
- Run id `20260914a`; model `moonshotai/Kimi-K3`; finished `2026-09-14T07:06:21Z`; NeMo Gym revision `35aec1975823b5248f724e0147b1582b762501ad`.
- Endpoint: OpenAI-compatible chat completions via a loopback proxy on 127.0.0.1 to the shared Modal Kimi K3 endpoint (moonshotai/Kimi-K3, 1,048,576-token context, reasoning enabled by the proxy).
- Sampling: temperature 0.0, top_p 1.0, reasoning enabled by the shared endpoint; limits: max_output_tokens 16384 per answer (reasoning tokens count against it); prompts up to 153k characters against a 1,048,576-token window; harness `simple_agent`; repeats 1.
- Judge/verifier: Gemini 2.5 Flash then GPT-5 (gpt-5-2025-08-07): judge baseline answer + v2 no-context eligibility prompt (first judge without Major Issue(s) wins), v2 UFG_REV21 sentence-level grounding per judge, provider-default sampling; judge identities observed: {'gemini-2.5-flash': 2568, 'gpt-5-2025-08-07': 878}.
- Coverage: 856 of 856 expected rollouts scored over 856 tasks; judge failed 0, simulator failed 0, infrastructure failed 0, missing 0, duplicates 0, superseded sidecar attempts 0.

| Metric | Value | Numerator / denominator | Role |
|---|---:|---:|---|
| `factuality_score` Factuality score (adjusted) | 66.5% (95% CI 64.1%–68.9%) | 569 / 856 | primary |
| `unadjusted_factuality_score` Unadjusted factuality score | 66.8% | 571.5 / 856 | component |
| `eligibility_rate` Eligible answers | 99.4% | 851 / 856 | component |
| `grounded_all_judges_rate_eligible` Grounded by both judges (eligible answers) | 48.9% | 416 / 851 | component |
| `judge_disagreement_rate_eligible` Judge disagreement (eligible answers) | 36.0% | 306 / 851 | diagnostic |
| `grounded_rate_eligible/gemini-2.5-flash` Grounded per gemini-2.5-flash (eligible answers) | 82.5% | 702 / 851 | component |
| `eligibility_decided_by/gemini-2.5-flash` Eligibility settled by gemini-2.5-flash | 98.7% | 845 / 856 | diagnostic |
| `grounding_parse_empty_rate/gemini-2.5-flash` Unparseable grounding verdicts from gemini-2.5-flash | 0.0% | 0 / 856 | diagnostic |
| `grounded_rate_eligible/gpt-5` Grounded per gpt-5 (eligible answers) | 51.2% | 436 / 851 | component |
| `eligibility_decided_by/gpt-5` Eligibility settled by gpt-5 | 0.7% | 6 / 856 | diagnostic |
| `grounding_parse_empty_rate/gpt-5` Unparseable grounding verdicts from gpt-5 | 0.0% | 0 / 856 | diagnostic |
| `eligibility_invalid_rate` Unparseable eligibility verdicts | 0.0% | 0 / 856 | diagnostic |
| `generation_truncated_rate` Truncated answers | 0.0% | 0 / 856 | operational |
| `generation_empty_rate` Empty answers | 0.0% | 0 / 856 | operational |
| `policy_input_tokens_max` Longest prompt (tokens) | 29445 | - | operational |
| `mean_output_tokens` Mean output tokens (reasoning + answer) | 1145.9 | - | operational |

- Calibration: replay of every judge receipt through the starter's verbatim parsers and eligibility/score rules; plus a live stratified re-judge with the starter loop: 886 of 896 cases agree; replay: 856 of 856 cases agree; live: 30 of 40 cases agree; eligibility agreement 38 of 40; per-judge verdict agreement 86.2%.
- Package: `facts_grounding_v2-kimi-k3-20260914a`; checksums.sha256 SHA-256 `e1f7c0e8909ee7b5d63d8b9df23569f8d564b8bf86f57b4c46ce89d4531433f7`; status: complete public protocol run, `verified: false` (single model baselined).
<!-- fingerprint:end -->

## 7. How to read the result

- Quote `factuality_score` (adjusted) with its interval and denominator; say "public set" and name both judges.
- A score of 0.5 on an answer means the two judges disagreed; `judge_disagreement_rate_eligible` tells how much of
  the score is judge noise rather than model behaviour.
- The unadjusted score isolates groundedness from eligibility; the gap between the two is the share of answers
  disqualified for not addressing the request.
- Common false readings: comparing with the paper's Table 1 (public+private, Kaggle pipeline); calling `Invalid`
  eligibility verdicts model failures; reading `not_grounded` as "hallucinated" when a single sentence lacked support.

## 8. BLADE mapping

- D1: `pass_at_1` N/A (three-valued eligibility-adjusted score); `primary` -> `factuality_score`; consistency N/A.
  Funnel: expected -> materialized -> answered -> judged -> scored.
- D2 anchor facts: outcome (score, per-judge grounded rates), coverage, invalid (eligibility/parse), infrastructure
  (truncation, context utilisation), slice (domain), example (grounded / unsupported / split / ineligible), calibration.
- D3: shallow baseline vs BLADE report. Not applicable: pass@k, tool-call funnel, task root cause.

## 9. What belongs in the 1-2 page report

`factuality_score` with CI; `unadjusted_factuality_score`; `eligibility_rate`; per-judge grounded rates and
disagreement; parse validity; truncation and the longest prompt vs the context window; two to four trace-linked
examples (grounded, unsupported with the judge's unsupported sentence, split verdict, ineligible); calibration
counts; the limitations in section 7; the provenance footer.
