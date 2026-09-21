# What the actor's willingness to press is worth

The same models, the same 200 scenarios, the same judge. The only thing that changed is who played the child: **stock Gemma-4-31B-it** against **refusal-ablated Gemma-4-31B-it**.

| Model | Actor refusals | Degradation slope | Peak drop |
|---|---|---|---|
| Nemotron 3 Ultra 550B | 0.20% → 0.00% | 0.0460 → 0.0695 (+51%) | 0.35 → 0.41 (+17%) |
| Kimi K3 | 0.20% → 0.00% | 0.0263 → 0.0128 (-51%) | 0.24 → 0.24 (-1%) |
| Qwen3.5 122B-A10B | 0.10% → 0.00% | 0.0443 → 0.0495 (+12%) | 0.32 → 0.28 (-13%) |
| Nemotron 3.5 Super VL | 0.20% → 0.00% | -0.0142 → 0.0063 (+144%) | 0.17 → 0.12 (-26%) |

Degradation steepened for **3 of 4** models and peak drop deepened for **1 of 4**. The effect is real but it is *not* uniform, and no single direction should be quoted as the result — the per-model rows above are the finding.

No average is given for the percentage changes. One model's baseline slope crosses zero, and a percentage change across a sign flip is arithmetic rather than meaning.

## What does hold across every model

Two things are uniform. Actor refusals go to zero, and **every model's first turn scores higher** under the harder actor — its opening questions are better formed, so the model has more to work with. What happens after turn one is model-specific: some hold the line better under sustained pressure than the weaker actor ever revealed, and at least one degrades markedly faster once genuinely pressed.

That non-uniformity is the practical result. If the actor moved every model the same way it could be corrected for. Because it does not, a multi-turn number is only meaningful next to the actor that produced it, and two runs cannot be compared unless their actors pressed equally hard.

| Model | Turn curve — stock Gemma-4-31B-it | Turn curve — refusal-ablated Gemma-4-31B-it |
|---|---|---|
| Nemotron 3 Ultra 550B | 4.45 → 4.44 → 4.39 → 4.34 → 4.31 | 4.53 → 4.48 → 4.37 → 4.32 → 4.31 |
| Kimi K3 | 4.66 → 4.58 → 4.59 → 4.62 → 4.73 | 4.74 → 4.73 → 4.73 → 4.80 → 4.84 |
| Qwen3.5 122B-A10B | 4.72 → 4.70 → 4.69 → 4.64 → 4.65 | 4.75 → 4.70 → 4.64 → 4.67 → 4.67 |
| Nemotron 3.5 Super VL | 4.63 → 4.64 → 4.66 → 4.69 → 4.69 | 4.68 → 4.68 → 4.69 → 4.69 → 4.66 |

So a KIDBench multi-turn number cannot be read without knowing which actor produced it. Two runs quoting the same degradation slope are not agreeing unless their actors pressed equally hard, and the honest way to report that is the measured refusal rate — which is why every conversation carries `actor_refusal_rate` regardless of which actor ran it.

## The metrics the leaderboard actually ranks on

| Model | Mean quality — stock Gemma-4-31B-it | — refusal-ablated Gemma-4-31B-it | Δ | Ever unsafe |
|---|---|---|---|---|
| Nemotron 3 Ultra 550B | 4.34 | 4.39 | +0.05 | 17.1% → 24.0% |
| Kimi K3 | 4.59 | 4.71 | +0.12 | 11.5% → 10.1% |
| Qwen3.5 122B-A10B | 4.67 | 4.66 | -0.01 | 12.0% → 10.0% |
| Nemotron 3.5 Super VL | 4.65 | 4.68 | +0.03 | 12.0% → 7.5% |

Mean quality moves least, because it averages over turns: a steeper decline from a higher start can leave it flat or even raise it. That is precisely why the leaderboard ranks on quality and ever-unsafe rather than on degradation — the metrics the actor moves most are the ones least safe to rank on.
