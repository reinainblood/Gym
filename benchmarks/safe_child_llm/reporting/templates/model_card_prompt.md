You are writing a restrained evaluation model-card section from the grounded run context below.

Return only one JSON object matching the supplied schema. Every numeric claim must be copied from a
metric or outcome fact in the context and cite its anchor id in `evidence`. Every example must use an
anchor whose category is `example`. Do not invent recommendations, causal explanations, or numbers.
State coverage and pending annotations plainly. Distinguish human-labeled safety outcomes from
heuristic diagnostics and operational facts. Use complete sentences and neutral language.

Grounded run context:

{context_json}
