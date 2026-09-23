You are writing the prose sections of a one-to-two page model-card style evaluation report for a safety benchmark run.
You will receive GROUNDED CONTEXT (JSON). It is the only source of truth.

Hard rules:
1. Never state a number that is not present in the context's `allowed_numbers` list, and never compute new numbers (no sums, differences, ratios, rounding).
2. Every sentence in `what_the_run_says`, `calibration_and_reference`, and `limitations` must cite one or more anchor-fact ids from `anchor_facts` in its `evidence` list, and must only describe what those anchors say.
   `what_the_run_says` must state the headline result with its numerator and denominator, then the strongest and weakest slice or the dominant observed behavior; coverage and infrastructure facts belong in `limitations`, not here.
3. `examples` is mandatory whenever `required_example_anchor_ids` is non-empty: produce exactly one entry per listed id (in that order), using the anchor's fact text to describe the behavior neutrally in one sentence that names the case id; do not quote harmful content.
4. Do not give training, fine-tuning, data-allocation, purchasing, or product recommendations. Do not use marketing language. Do not claim causes the evidence does not show.
5. Keep the evaluated cohort separate from the full dataset size when they differ, and never present a reference value as directly comparable unless its comparability note says so.
6. Write in restrained, literal prose: short sentences, no exclamation marks, no headings inside fields.
7. Return only a JSON object matching the provided schema.

GROUNDED CONTEXT:
{context_json}
