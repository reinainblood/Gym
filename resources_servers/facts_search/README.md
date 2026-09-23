# FACTS Search resources server

The server exposes `brave_search`, backed by Brave Web Search. Each request returns five results with the title, URL,
description, and all extra snippets in the format used by Google's public implementation.

The verifier grades the final answer with Gemini 3.5 Flash and the public A/B/C rubric:

- `A`: correct
- `B`: incorrect
- `C`: not attempted

The reward is 1 for `A` and 0 otherwise. Aggregate metrics include FACTS F1, accuracy, attempted accuracy, hedging rate,
search hops, query count, forced-final rate, empty or truncated answers, and judge-format validity.

The default configuration requires Brave and Gemini 3.5 Flash. Missing credentials fail the evaluation instead of
switching providers.
