# KIDBench

Child-facing LLM safety for ages 7–11, as a NeMo Gym environment.

> **The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language
> Models** — Samee Arif, Angana Borah, Rada Mihalcea. Findings of EMNLP 2026.
> [arXiv:2605.25510](https://arxiv.org/abs/2605.25510) ·
> [`MichiganNLP/kidbench`](https://github.com/MichiganNLP/kidbench)

Most safety evaluation is adult-facing and stops at harmful-content avoidance. KIDBench
starts from a sharper problem: a medically accurate answer to *"How are babies made?"* can
still be the wrong thing to tell a nine-year-old. It scores child-facing responses on five
1–5 dimensions drawn from developmental psychology — Piaget on concrete explanation,
Vygotsky on scaffolding and trusted adults, Bandura on what a child will imitate — rather
than marking answers right or wrong.

## What this adapter runs

| Track | Rows | What happens |
|---|---|---|
| Single turn | 5,000 | The ten-cell condition grid: 500 prompts × {no cue, implicit cue, explicit age, 4 country contexts, 3 non-English languages} |
| Multi turn | 200 | 100 scenario/child-goal pairs × {age stated, age not stated}, five turns each against a child-actor model |

Both are judged with the paper's own rubric by the paper's own judge.

## Quickstart

```bash
python -m benchmarks.kidbench.prepare
```

This clones upstream at the pinned revision into `benchmarks/kidbench/upstream/`
(gitignored) and writes the Gym JSONL into `resources_servers/kidbench/data/`. Nothing
upstream is vendored into this repository — its use notice requires that the child-safety
context travel with any redistribution, so the adapter references it in place.

Copy the run wiring and fill in your endpoints:

```bash
cp benchmarks/kidbench/configs/env.yaml.example env.yaml
```

Then start the servers and collect:

```bash
gym env start --config resources_servers/kidbench/configs/kidbench.yaml
```

```bash
gym eval run --no-serve --agent kidbench_simple_agent --input resources_servers/kidbench/data/single_turn.jsonl --output results/kidbench/my-model.single_turn.jsonl
```

```bash
gym eval run --no-serve --agent kidbench_multi_turn_agent --input resources_servers/kidbench/data/multi_turn.jsonl --output results/kidbench/my-model.multi_turn.jsonl
```

To sweep several models in one go, `benchmarks/kidbench/run_all_models.sh` restarts the
stack per model and runs both tracks with `--resume`.

## Reports

```bash
python -m benchmarks.kidbench.reporting.cli --results-dir results/kidbench --out-dir results/kidbench/reports
```

Writes a BLADE-structured markdown report and a JSON sidecar per model, plus a cross-model
leaderboard in both formats.

The reporting layer does one thing beyond aggregation that is worth knowing about. BLADE's
root-cause codes (`KG`, `UK`, `BI`, …) are defined for benchmarks where a task passes or
fails, and they mean nothing on their own for a benchmark whose unit is "did a nine-year-old
get a usable answer". `benchmarks/kidbench/reporting/taxonomy.py` restates each code as a
falsifiable claim about child-facing safety — `UK` becomes *safety is gated on the child
being named*, `BI` becomes *the boundary eroded under child-like pressure* — and names the
evidence that earns it. The letters stay so BLADE tooling keeps working.

## Moving parts

| Component | Where | Role |
|---|---|---|
| Condition grid and decoding constants | `benchmarks/kidbench/upstream_spec.py` | The published protocol, transcribed, with its source line cited |
| Data preparation | `benchmarks/kidbench/prepare.py` | Pins upstream, expands the grid, validates row counts |
| Verifier | `resources_servers/kidbench/app.py` | Runs the rubric, parses verdicts, derives degradation metrics |
| Multi-turn harness | `responses_api_agents/kidbench_child_actor/app.py` | The five-turn actor loop, plus actor-fidelity screening |
| Reporting | `benchmarks/kidbench/reporting/` | Aggregation, failure taxonomy, BLADE reports |

Metric definitions, threshold choices, and every deviation from the published protocol are
in [METRICS.md](METRICS.md).

## What you need to run it

- **A judge.** `deepseek-v4-pro` for single turn and `deepseek-v4-flash` for multi turn —
  upstream's own split, visible in its evaluation scripts. Any OpenAI-compatible provider
  that serves them works; the judge identity is recorded on every scored row.
- **A child-actor model** for the multi-turn track only. The paper uses a refusal-ablated
  Gemma-4-31B-it that it deliberately does not release. Two options, in descending
  fidelity:
  - **Self-hosted** (default, and what the published runs used):
    [`wangzhang/gemma-4-31B-it-abliterated`](https://huggingface.co/wangzhang/gemma-4-31B-it-abliterated)
    pinned at `d7431183`, a third-party ablation of the same base model.
    `deploy/modal_actor.py` deploys it; 62.6 GB of bf16 weights, two H100s.
  - **Hosted API**: stock `google/gemma-4-31b-it` from OpenRouter or any OpenAI-compatible
    provider. No GPU needed, but it refuses some actor turns, which biases results towards
    the model under test. `configs/env.yaml.example` carries both blocks.

  Either way the agent screens every actor turn and reports `actor_refusal_rate`, so the
  gap between the two is measured rather than assumed. See METRICS.md.

## Licensing

Upstream's code is MIT. Its data, rubrics, and rules carry a bespoke research-use and
child-safety notice with no entry in Gym's license vocabulary, so the dataset entries are
marked `TBD` and the real terms are these:

- For research on child-facing AI safety evaluation and model development.
- **Not** to be deployed directly to children as an assistant, toy, or companion.
- Not to be used to generate harmful content targeting children.
- The child-safety context — rubrics, rules, and these notices — must not be stripped when
  redistributing.

These artifacts are not a substitute for parental, educational, medical, legal, or
emergency support. A good score on this benchmark is evidence about a benchmark, not
clearance to put a model in front of a child.
