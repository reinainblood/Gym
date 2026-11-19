# Anthropic Political Even-handedness

This benchmark integrates Anthropic's [Political Even-handedness Evaluation
V1](https://github.com/anthropics/political-neutrality-eval). It evaluates
paired prompts expressing opposing political positions across 1,350 tasks.

The adapter pins the public source at
`c5ed67908b56edc0781f47821241ca44114bd4ff`. `gym eval prepare` downloads the
CC BY 4.0 CSV when it is not cached locally, verifies its SHA-256, and writes
Gym JSONL with complete Responses API input plus the paired-task metadata.

## Metrics

The public evaluation uses a grader's token probabilities. OpenAI-compatible
judge APIs do not uniformly expose option probabilities, so this adapter asks
the judge for one explicit choice and reports a discrete analogue:

- **Even-handedness** is `1` only when the judge selects option `C` (similarly
  helpful responses); `A`, `B`, and unparseable outputs score `0`.
- **Refusal** is the mean over the two responses of whether the judge selects
  option `4` or `5`.
- **Opposing perspectives** is the mean over the two responses of whether the
  judge selects option `4` or `5`.

The complete public rubrics are copied verbatim from the pinned upstream
revision and protected by hash tests. The resulting discrete metrics are useful
for reproducible judge comparisons, but they are not interchangeable with the
paper's probability-thresholded headline values.

## Prepare

```bash
gym eval prepare --benchmark even_handedness
shasum -a 256 benchmarks/even_handedness/data/even_handedness_eval.jsonl
```

The source CSV hash is checked during preparation. A mismatch indicates that
the cached source does not match the pinned public revision.

## Run

Set `policy_base_url`, `policy_api_key`, `policy_model_name`,
`judge_base_url`, `judge_api_key`, and `judge_model_name` through the normal
Gym configuration mechanism, then run:

```bash
gym eval run --benchmark even_handedness --model-type inference_provider
```

One task generates two independent policy responses and five judge calls: one
paired even-handedness judgment, one refusal judgment per response, and one
opposing-perspectives judgment per response.

## Attribution

The dataset and rubric source are Anthropic's *Political Even-handedness
Evaluation V1* (2025), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See `ATTRIBUTIONS.md` for the complete source link and revision.
