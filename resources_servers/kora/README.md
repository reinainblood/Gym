# KORA resources server

Grades a conversation between a simulated child and the model under test on the KORA
rubric: one overall safety grade, `failing` / `adequate` / `exemplary`, and the same grade
for each of seven conversation behaviours, from two structured-output calls to the
published judge.

Full benchmark documentation, metric definitions, and protocol deviations live in
[`benchmarks/kora/`](../../benchmarks/kora/README.md).

## Before first use

```bash
python -m benchmarks.kora.prepare
```

The server reads the taxonomy and the behaviour rubrics from `benchmarks/kora/upstream/pack.json`,
which this command exports from the run package. It does not fetch them itself: a server
booting inside a large worker pool would otherwise race on a 280 MB download, and the rubric
text is part of the measurement, so it is pinned deliberately rather than acquired at startup.

## Agent

| Agent | Dataset |
|---|---|
| `kora_child_actor_agent` | `data/kora.jsonl` (1,562 rows: 781 scenarios x {adult, child}) |

The agent plays the child for the scenario's `planned_turns` exchanges and sends the
transcript here. `data/example.jsonl` holds five rows from the least sensitive risks for
smoke tests.

## What `verify()` returns

`reward` is the overall grade over two, 0.0 / 0.5 / 1.0, because Gym aggregates rewards on
[0, 1]; `grade` and `grade_score` keep the raw verdict. `behaviors` carries each
behaviour's grade, occurrence count and `not_triggered` flag; a not-triggered behaviour is
score-neutral. `judge_parse_failed`, `turn_shortfall`, `assistant_turn_issue` and
`truncated_turns` flag rows the aggregate should be read with.

A judge call that fails at the transport layer raises `JudgeError` and the row is routed to
the failures sidecar, excluded from aggregates. A judge reply that arrives but will not
parse is a scoring outcome: the row scores 0.0 and is counted in `judge_parse_failure_rate`,
and excluded from `score_pct`.

## Judge wiring

`judge_model_server` should point at the published judge, GPT-5.2 at medium reasoning
effort through the Responses API. Swapping it is supported but changes what the scores
mean, so `judge_name` is recorded on every row.

```yaml
judge_model_server:
  type: responses_api_models
  name: kora_judge_model
judge_responses_create_params:
  input: []
  max_output_tokens: 26000
judge_structured_output: true
```

## Tests

```bash
gym env test --resources-server kora
```

The prompt-fidelity tests compare every rendered prompt with the package's `prompts/*.md`
and need the package under `benchmarks/kora/upstream/`; they skip when it is absent. Set
`KORA_TEST_FETCH_PACKAGE=1` to fetch it before collection.

## Use restrictions

The dataset contains simulated children discussing self-harm, sexual grooming, abuse, hate
speech and other sensitive topics. It is published under KORA's Permitted Use terms for
research, internal evaluation and benchmarking, with attribution, and never for direct
exposure to children. See the benchmark README for the full terms.
