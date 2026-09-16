# FACTS Multimodal

This integration implements the public FACTS Multimodal benchmark for factual,
open-ended answers to image-based questions. It measures whether a model can
combine visual grounding and world knowledge while providing a sufficiently
complete answer.

Each public task includes a user prompt, an image URL, and a human-authored
rubric of atomic facts. Rubric facts are marked as either `essential` or
`non-essential`. Preparation downloads each reachable image locally, converts
it to an inline data URL, and stores that data URL in the generated JSONL. The
policy and judge receive the same embedded image bytes.

## Evaluation

For every non-empty policy response, the resources server makes exactly two
separate judge calls, adapted from the public FACTS Multimodal notebook:

- The text-only coverage call sees the prompt, essential rubric facts, and
  policy response. It returns one `Yes`/`No` value per essential fact; the
  server recomputes `coverage` as their fraction marked `Yes`. It does not see
  the image, because this judgement is about what the response says rather than
  what the image contains.
- The image-aware factuality call sees the prompt, every rubric fact, the policy
  response, and the same embedded image passed to the policy. It returns a
  contradiction verdict, recorded as binary `factuality`.

`accuracy` is the notebook-style conjunction, using a strict coverage
threshold:

```text
accuracy = (coverage > 0.5) AND factuality
reward   = accuracy
```

`coverage` is a fractional diagnostic metric; `factuality`, `accuracy`, and
reward are binary. An empty policy response receives zero reward without either
judge call.

The rollout response retains `generation`, `coverage_judge_output`,
`factuality_judge_output`, `coverage`, `factuality`, and `accuracy` for audit.
Aggregate results report the same three scalar metrics under NeMo Gym's pass@k
metric prefixes.

The exact FACTS Multimodal judge prompts are not public. The implemented prompts are taken from the Kaggle example code.

## Image materialization and exclusions

The source dataset references third-party image URLs, whose availability can
change. Preparatio downloads every source
image into `benchmarks/facts_multimodal/data/images/`, checks that the response
is a supported image, and writes one row per source item to
`benchmarks/facts_multimodal/data/image_download_report.csv`.

Only successfully materialized images appear in the generated benchmark JSONL.

## Configuration

The configuration lives at
`resources_servers/facts_multimodal/configs/facts_multimodal.yaml`. It defines:

- `facts_multimodal`, the resources server.
- `judge_model`, an OpenAI Responses-compatible model-server instance.
- `facts_multimodal_simple_agent`, which pairs the built-in `simple_agent`
  with the policy model and resources server.

Set the judge endpoint through the normal Hydra overrides:

```yaml
judge_base_url: <OpenAI-Responses-compatible endpoint>
judge_api_key: <judge credential>
judge_model_name: <vision-capable judge model>
```

Both the policy model and the judge must accept inline image data through their
OpenAI-compatible inference interfaces. The judge specifically must be
vision-capable: it sees the image to detect contradictions that are not fully
expressed in the rubric.
The default judge request has no generation-control overrides, allowing the
provider defaults to apply; configure them through
`judge_responses_create_params` when needed.

Run preparation before any smoke or benchmark evaluation. Add the generated
JSONL as a versioned dataset-registry artifact before using it for a shared
evaluation.

## Run the benchmark

### Image transport

By default, `use_base64_images=true`: preparation writes each successful image
as a `data:image/...;base64,...` value in the JSONL, and both the policy and
factuality judge receive that value in the `input_image.image_url` API field.
This is robust to provider-side fetching failures, but makes prepared JSONL and
rollout artifacts large.

Set `++use_base64_images=false` to instead place each original remote URL in
that same API field. The output JSONL and rollouts are much smaller, but the
policy and factuality judge must each fetch the third-party URL. Preparation
still downloads and validates every source image before including its task, so
unavailable images continue to be excluded and reported.

The setting is preparation-time: re-run preparation with the desired value,
then start or run the benchmark with the same value so the factuality judge
uses the matching image reference.

Ensure `facts_multimodal_public.csv` is downloaded and placed in `benchmarks/facts_multimodal`.

This command downloads all images from the public benchmark set, logs any that
cannot be obtained, and writes the selected image reference into the prepared
JSONL. In the default Base64 mode, this intentionally makes generated JSONL and
rollout artifacts large but removes per-rollout image downloads and provider-side
URL fetching.

```bash
gym eval prepare --benchmark facts_multimodal
```

To prepare compact URL-based tasks instead:

```bash
gym eval prepare --benchmark facts_multimodal ++use_base64_images=false
```

For a smoke test, first run preparation so it creates image inputs using your
selected transport.

```bash
gym env start \
  --config resources_servers/facts_multimodal/configs/facts_multimodal.yaml \
  --model-type inference_provider \
  --config ~/.config/nemo-gym/policy/portkey_gpt-5-mini.yaml \
  --config ~/.config/nemo-gym/judge/portkey_gemini-2-5-flash.yaml \
  ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/model-calls/facts_multimodal_example"

gym eval run --no-serve \
  --agent facts_multimodal_simple_agent \
  --input benchmarks/facts_multimodal/data/facts_multimodal_example.jsonl \
  --output results/facts_multimodal_example.jsonl \
  --num-repeats 1 \
  ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/model-calls/facts_multimodal_example"
```

For the full run
```bash
gym eval run \
    --benchmark facts_multimodal \
    --model-type inference_provider \
    --split benchmark \
    --output results/facts_multimodal_gemini-2-5-flash.jsonl \
    --config ~/.config/nemo-gym/policy/portkey_gpt-5-mini.yaml \
    --config ~/.config/nemo-gym/judge/portkey_gemini-2-5-flash.yaml \
    --num-repeats 1 \
    --concurrency 8 \
    ++head_server.host=127.0.0.1 \
    ++head_server.port=11001 \
    ++observability_enabled=true \
    ++model_call_capture_dir="$PWD/results/model-calls/facts_multimodal_gemini-2-5-flash" 
```

if not using the base64 encoding add this to `gym eval run` and `gym env start`
```bash
    ++use_base64_images=false
```


to run a different judge on the same outputs
```bash
gym eval reverify \
    --config resources_servers/facts_multimodal/configs/reverify.yaml \
    --config ~/.config/nemo-gym/judge/modal_glm-5-3-flash.yaml \
    --inputs results/facts_multimodal_gpt-5-mini_RUN1_materialized_inputs.jsonl \
    --rollouts results/facts_multimodal_gpt-5-mini_RUN1.jsonl \
    --output results/facts_multimodal_gpt-5-mini_RUN2.jsonl \
    --concurrency 8 \
    ++observability_enabled=true \
    ++model_call_capture_dir="$PWD/results/model-calls/facts_multimodal_gpt-5-mini_RUN2"

```

## Validation

### GPT-5-mini judge comparison: RUN1 vs RUN2

| Model | Judge Model | Metric | Published Value | NeMo Gym Value | Notes |
|-------|--------|--------|-------------|----------------|-------|
| GPT 5 Mini | Gemini 2.5 Flash | Accuracy(Pass@1) | 40.7 | 30.6 |  |
| GPT 5 Mini | Gemini 2.5 Flash | Coverage | 58.8 | 53.9 |  |
| GPT 5 Mini | Gemini 2.5 Flash | Factuality | 65.1 | 41.0 | |
| GPT 5 Mini | GLM 5.3 Flash | Accuracy(Pass@1) | 40.7 | 45.9 |  |
| GPT 5 Mini | GLM 5.3 Flash | Coverage | 58.8 | 64.1 |  |
| GPT 5 Mini | GLM 5.3 Flash | Factuality | 65.1 | 62.05 |  |

A sample of 20 instances where Gemini 2.5 Flash and GLM 5.3 Flash differed were reviewed by a human annotator. GLM was preferred in 15/20, with the most common reason being Gemini 2.5 Flash treating an omitted fact as a contradiction.


## Scope and limitations

This implementation covers only the released public CSV. The FACTS private
split is not included. Scores therefore are not directly comparable to the
combined public-and-private leaderboard results in the paper. Image hosting is
owned by the original third-party URLs. Preparation records their availability
at materialization time; evaluation uses only the resulting embedded image data
and does not refetch or replace images.
