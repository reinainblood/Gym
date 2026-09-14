# vera_mh

VERA-MH (Validation of Ethical and Responsible AI in Mental Health; Spring Health; arXiv:2605.13318;
https://github.com/SpringCare/VERA-MH, commit `2c9d1fc`, rubric v1.2): simulated conversations between clinician-authored
user personas in suicide-risk scenarios and the chatbot under evaluation, judged with a clinical rubric that is
navigated one question at a time.

Two servers implement the benchmark:

- `responses_api_agents/vera_mh_agent` simulates the conversation. The persona (user simulator) and the provider (policy)
  are separately named model servers (`user_model_servers`, `model_server`); persona speaks first, 30 turns (provider
  last), early stop on `<END OF CONVERSATION>`, upstream message construction and role reminder, upstream transcript
  format.
- `resources_servers/vera_mh` (this server) judges the transcript with the pinned v1.2 rubric (`rubric/`, hashes in
  `rubric/PROVENANCE.md`): GPT 5.4 with `reasoning_effort=low`, structured `{answer, reasoning}` replies, upstream GOTO
  navigation, special cases, fallback matching, severity-based dimension ratings, and the VERA-MH v1.2 score.

## Recommended profile (upstream README "Recommended settings")

100 personas x (GPT 5.2, Claude Opus 4.5) user agents x 1 run x 30 turns, judged by GPT 5.4 (low reasoning), pooled.
`benchmarks/vera_mh/prepare.py` materializes the 200 tasks from the pinned `personas.tsv`.

## Running

```bash
export VERA_MH_JUDGE_BASE_URL=https://.../v1 VERA_MH_JUDGE_API_KEY=... VERA_MH_JUDGE_MODEL=gpt-5.4
export VERA_MH_USER_GPT52_BASE_URL=... VERA_MH_USER_GPT52_API_KEY=... VERA_MH_USER_GPT52_MODEL=gpt-5.2
export VERA_MH_USER_OPUS45_BASE_URL=... VERA_MH_USER_OPUS45_API_KEY=... VERA_MH_USER_OPUS45_MODEL=claude-opus-4-5-20251101

gym eval run --benchmark vera_mh --config <policy model config> --split benchmark --max-output-tokens 8192 \
  --output results/vera_mh.jsonl +route_failures_to_sidecar=true +observability_enabled=true
```

Simulator failures become sidecar rows (`vera_mh_simulation_failed`), judge failures become `judge_failed` rows;
neither is a target-model safety failure. See [`benchmarks/vera_mh/METRICS.md`](../../benchmarks/vera_mh/METRICS.md).

## Tests

```bash
gym env test --resources-server vera_mh
gym env test --entrypoint responses_api_agents/vera_mh_agent   # agent tests; its instance config lives in resources_servers/vera_mh/configs/vera_mh.yaml
```
