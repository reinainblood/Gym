# vera_mh_agent

Conversation simulator for VERA-MH: a persona model server plays the user, the policy model server plays the
chatbot, and the resources server `vera_mh` judges the transcript. See
[`resources_servers/vera_mh/README.md`](../../resources_servers/vera_mh/README.md).

Config keys: `model_server` (provider under evaluation), `user_model_servers` (map from the dataset row's
`user_simulator` to a model server), `user_responses_create_params` (optional per-simulator overrides), `max_turns`
(30; rounded so the provider speaks last), `persona_speaks_first`, `start_prompt`, `termination_signal`.
