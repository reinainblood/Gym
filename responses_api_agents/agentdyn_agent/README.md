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
