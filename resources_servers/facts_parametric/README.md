# FACTS Parametric

This integration grades short factual responses with an LLM semantic-equivalence
judge. For every `(question, gold answer, model response)` triplet it requests
three independent labels: `correct`, `incorrect`, `not-attempted`, or `unknown`.

Only `correct` receives reward. The reward is the fraction of the three labels
that are correct, so a response judged correct twice receives `2/3`.

`unknown` is retained as grader uncertainty. It is considered attempted for
attempted accuracy because it is not a model abstention:

- accuracy: mean `correct` fraction
- hedging rate: mean `not-attempted` fraction
- attempted accuracy: `correct / (correct + incorrect + unknown)`
- F1: harmonic mean of accuracy and attempted accuracy

The benchmark uses NeMo Gym's built-in `simple_agent`: it sends each question to
the policy model once, then the resources server sends the completed answer to
the judge three times. Run preparation from the repository root:

```bash
python benchmarks/facts_parametric/prepare.py
gym dataset collate --config benchmarks/facts_parametric/config.yaml \
  --output-dir /tmp/facts-parametric --mode example_validation
```

The released CSV is a local source download and is ignored by git. Preparation
writes the full JSONL to `data/`, which is ignored as well; five representative
examples are committed for smoke tests.

The judge uses Gemini's OpenAI-compatible Chat Completions API through a NeMo
Gym model server. The endpoint is
`https://generativelanguage.googleapis.com/v1beta/openai/`; configure its API
key as `GEMINI_API_KEY` in the model-server configuration. The prompt is an
implementation of the published grading description, not a claim to reproduce
an unreleased reference prompt byte-for-byte.
