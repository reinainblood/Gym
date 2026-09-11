# FACTS Parametric

FACTS Parametric ([technical report](https://storage.googleapis.com/deepmind-media/FACTS/FACTS_benchmark_suite_paper.pdf),
[public examples](https://www.kaggle.com/datasets/kaggle/facts-parametric-public-examples)) measures
parametric factual recall with short, atomic questions. This integration's downloaded public CSV
snapshot contains **1,052** examples with a question, canonical answer, topic, and source URL. It covers facts such as
birthdays, people, releases, records, population, and addresses; `other` is the largest topic
bucket.

The benchmark uses NeMo Gym's built-in **`simple_agent`** harness: one policy-model response per
question, followed by three independent semantic-equivalence judgements. The judge sees the
question, gold answer, and model response, and returns one of four labels:

- **`correct`** — response conveys the requested gold fact without contradiction.
- **`incorrect`** — response contradicts the gold fact.
- **`not-attempted`** — response does not answer, without contradicting it.
- **`unknown`** — the judge cannot reliably determine semantic alignment.

The rollout reward is the fraction of the three labels that are `correct`. For example, two
`correct` labels and one `unknown` produces a reward of `2/3`. This is not exact match and does
not apply numeric tolerances. The judge assesses whether the response means the same thing as the
gold answer.

## Configuration

Chains to the `facts_parametric` resources server with the **`simple_agent`** harness. See the
[server README](../../resources_servers/facts_parametric/README.md) for the verifier and metric
definitions.

The benchmark defines two independent `inference_provider` instances: `policy_model` for the
model under test and `judge_model` for the semantic judge. 

## Usage

The Kaggle CSV is a local source download, not benchmark data committed to Git. Convert it into
Gym JSONL before a full run:

```bash
# Converts FACTS-Parametric-public.csv into 1,052 task rows.
gym eval prepare --benchmark facts_parametric

# Validate the five committed smoke-test rows and the simple_agent wiring.
gym dataset collate \
    --config benchmarks/facts_parametric/config.yaml \
    --output-dir /tmp/facts-parametric \
    --mode example_validation
```

For a five-row smoke test (status should show 4 healthy):

```bash
gym env start --benchmark facts_parametric

gym env status

gym eval run --no-serve \
    --agent facts_parametric_benchmark_simple_agent \
    --input resources_servers/facts_parametric/data/example.jsonl \
    --output results/facts_parametric_smoke.jsonl \
    --num-repeats 1
```

Collect full-benchmark rollouts on long-lived servers (status should show 4 healthy):

```bash
gym env start --benchmark facts_parametric

gym env status

gym eval run --no-serve \
    --agent facts_parametric_benchmark_simple_agent \
    --input benchmarks/facts_parametric/data/facts_parametric_benchmark.jsonl \
    --output results/facts_parametric.jsonl \
    --num-repeats 1 \
    --concurrency 8
```



## Endpoints

Credentials belong in a gitignored root `env.yaml`, which overrides settings in `config_paths`.
Configure the tested model and judge independently. The optional
[`judge_gemini_2_5_pro.yaml`](examples/judge_gemini_2_5_pro.yaml) overlay supplies Gemini 2.5 Pro
as an example judge through Google's OpenAI-compatible endpoint; it is not loaded by default.
Google's [OpenAI compatibility documentation](https://ai.google.dev/gemini-api/docs/openai)
describes that endpoint and API-key setup.

For a policy model served through `inference_provider`, use the normal root-level policy settings:

```yaml
# env.yaml
# Judge model: receives the question, gold answer, and policy response.
judge_base_url: https://your-judge-endpoint/v1
judge_api_key: ${oc.env:JUDGE_API_KEY}
judge_model_name: your-judge-model

# Tested policy model: receives only the benchmark question.
policy_base_url: https://your-policy-endpoint/v1
policy_api_key: ${oc.env:POLICY_API_KEY}
policy_model_name: your-policy-model

# Raw model-call capture. Each FACTS episode writes one JSONL file containing
# the policy-model call and its three judge calls. The directory must be
# absolute and should be gitignored.
observability_enabled: true
model_call_capture_dir: /absolute/path/to/facts-parametric-model-calls

# Preserve provider reasoning separately in rollout records. FACTS scores only
# the final assistant answer, so this does not change the verifier's behaviour.
policy_model:
  responses_api_models:
    inference_provider:
      uses_reasoning_parser: true
```

## Validation Runs

*GLM-5.3 was used as the judge instead of Gemini 2.5 Pro.*

| Model | Metric | Paper Value | NeMo Gym Value | Notes |
|-------|--------|-------------|----------------|-------|
| GPT-4.1 | F1 | 52.5 | 52.8 | From Paper Table 6 |
| GPT-4.1 | Accuracy | 51.5 | 51.3 | From Paper Table 6 |
| GPT-4.1 | Attempted Accuracy | 53.6 | 54.3 | From Paper Table 6 |
| GPT-4.1 | Hedging Rate | 3.8 | 5.6 |From Paper Table 6 |
| DeepSeek-R1 | F1 | 21.2 | 22.4 | From Leaderboard |
| DeepSeek-R1 | Accuracy | 22.1 | 21.9 | From Leaderboard "Public Score"  |
| DeepSeek-R1 | Attempted Accuracy | 21.6 | 23.0 | From Leaderboard |
| DeepSeek-R1 | Hedging Rate | 3.7 | 5.1 | From Leaderboard |

[Source](https://www.kaggle.com/benchmarks/google/facts-parametric/versions/1)

## Validation Notes

**Public data, unreleased scorer:** The downloaded public package supplies only the CSV examples.
It does not include the authors' exact judge prompt, scorer code, sampling parameters beyond the
reported three grades, or a reference set of judged responses. The prompt in
`resources_servers/facts_parametric/prompts/judge.yaml` implements the paper's described protocol;
it is not claimed to be a byte-for-byte reproduction of an unreleased prompt.

**Dataset-version mismatch:** The current
[official leaderboard](https://www.kaggle.com/benchmarks/google/facts-parametric/leaderboard)
describes 2,104 total examples, with 1,102 public and 1,102 private. The local source snapshot
has 1,052 rows. This integration intentionally preserves the downloaded snapshot; update the
source CSV and rerun `prepare.py` when adopting a newer public release.

**Unknown is grader uncertainty:** `unknown` is retained independently rather than silently mapped
to `incorrect` or `not-attempted`. It receives zero reward, contributes to `unknown_rate`, and is
counted as attempted when calculating attempted accuracy:

```text
accuracy           = correct / total
hedging_rate       = not-attempted / total
attempted_accuracy = correct / (correct + incorrect + unknown)
F1                 = harmonic_mean(accuracy, attempted_accuracy)
```

Each episode contributes fractions from its three judge labels. For example, one
`not-attempted` label out of three contributes `1/3` to `hedging_rate`.

**Three judge samples per episode:** One episode (or trial) of a question contains one policy
response, judged three times (`judge_samples: 3`). If `--num-repeats` is greater than one, each
additional episode of that question produces its own policy response and its own three judge calls.

**Judge-output failures:** Empty replies and judge transport, authentication, or HTTP failures are
recorded as `judge_failed` and excluded from scores. A received reply with harmless prose or
Markdown around a valid label is parsed normally. If no label can be recovered, it falls back to
`unknown` rather than failing the entire rollout.

**Verbose judge replies:** The judghe output parser has been hardened in this implementation as some 
judge providers return an explanation despite the one-label instruction. The parser gives priority to
a leading label such as `**incorrect**`, then to an explicit `Label: incorrect`, `Verdict: incorrect`,
or `Grade: incorrect` field. This avoids misreading a later phrase such as “the correct fact” in an 
explanation as the verdict. 

## Licensing

The local configuration uses the framework's `TBD` license placeholder because the public CSV
release does not supply a license in the downloaded package. Confirm the dataset license and
replace that placeholder before publishing or distributing a prepared dataset. Integration code is
Apache-2.0.
