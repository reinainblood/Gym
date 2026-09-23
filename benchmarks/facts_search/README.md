# FACTS Search

FACTS Search measures whether a model can answer factual questions by searching the web, combining evidence across
queries, and returning a concise answer. This benchmark implements Google's public Search-On protocol over all 890
currently downloadable questions.

```bash
gym eval prepare --benchmark facts_search
gym eval run \
  --benchmark facts_search \
  --agent facts_search_public \
  --split benchmark \
  --model-type inference_provider \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY"
```

Required runtime variables:

- `FACTS_SEARCH_BRAVE_API_KEY`: Brave Search API subscription token.
- `FACTS_SEARCH_JUDGE_BASE_URL`, `FACTS_SEARCH_JUDGE_API_KEY`, and optionally `FACTS_SEARCH_JUDGE_MODEL` (default
  `google/gemini-3.5-flash`): an OpenAI-compatible endpoint exposing Gemini 3.5 Flash.

See [METHODS.md](METHODS.md) for the protocol and the gap between the advertised and downloadable public data. See
[METRICS.md](METRICS.md) for metric definitions and baseline results.
