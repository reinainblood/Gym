# Political Even-handedness resources server

The resources server scores the two responses generated for each paired task.
It applies Anthropic's pinned public rubrics for response consistency, refusal,
and opposing perspectives.

The resource server is stateless for reverification: a saved policy-response
pair can be judged again without generating another policy response. This is
useful for controlled judge comparisons, but a result must record the judge
model and rubric revision because those choices affect the reported metrics.

The required row fields are documented in `task_data.py`. The benchmark wrapper
in `benchmarks/even_handedness/` prepares the CC BY source CSV and supplies the
paired agent configuration.
