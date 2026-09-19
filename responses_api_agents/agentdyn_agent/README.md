# AgentDyn agent adapter

This backend reuses the shared AgentDojo-family model bridge and result contract while installing AgentDyn in an
isolated agent environment. Isolation is required because official AgentDojo and AgentDyn both install the same
top-level Python package, `agentdojo`.

The primary benchmark exposes only AgentDyn's `shopping`, `github`, and `dailylife` suites. The original `banking`,
`slack`, `travel`, and `workspace` suites remain in the separate official AgentDojo control benchmark; they are not
mixed into the AgentDyn aggregate.

The request contract admits the five requested defenses: PromptGuard2, PIGuard, CaMeL, Progent, and DRIFT. Each run
selects exactly one defense; defenses are not stacked. The base config starts only the undefended agent. A complete
defense treatment can set `default_defense` in its run config without duplicating the task matrix. Filter and
system-defense runtime parity must be established independently before defense-specific configs are published.

PromptGuard2's detector source is configurable independently of the defense name. During gated-access validation,
`prompt_guard_2_model_name` and `prompt_guard_2_model_revision` may identify a provenance-recorded mirror while the
run remains labeled `prompt_guard_2_detector`. Before publication, replace the mirror with Meta's canonical model,
verify the model checksum, and rerun the same treatment. `prompt_guard_2_local_path` is a development-only cache
override; it does not replace the source repository and revision recorded in each rollout.

Some OpenAI-compatible endpoints, including the Qwen3.5 SGLang deployment used for baselining, reject the newer
`developer` message role. Set `model_system_role: system` for those endpoints. The default remains `developer`, and
the bridge applies the configured role consistently to ordinary policy calls and defense-owned auxiliary clients.

CaMeL, Progent, and DRIFT construct auxiliary OpenAI-compatible clients upstream. The adapter routes those clients to
the rollout-prefixed NeMo policy-model URL rather than allowing them to use unrelated provider credentials. A
separate compatibility alias selects the upstream OpenAI code path; the actual served model remains the model named
by the NeMo model server and is preserved in run provenance.

The shared adapter records the outer upstream pipeline transcript even when a system defense bypasses the ordinary
model bridge. `model_call_count` is exact for ordinary/filter pipelines and a lower bound for defenses that hide
auxiliary planner or policy-model calls inside their own clients; NeMo model-server observability remains the
authoritative call ledger for those treatments.
