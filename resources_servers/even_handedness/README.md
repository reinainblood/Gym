# Anthropic Political Even-handedness

This environment evaluates a policy on paired prompts representing opposing political positions. The paired agent independently generates an answer for each prompt; the resources server uses a judge to classify consistency, refusal, and consideration of opposing perspectives. The scalar Gym reward is even-handedness (option C).

The data and evaluation design are from Anthropic's *Political Even-handedness Evaluation V1* (2025), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This implementation uses discrete judge choices because OpenAI-compatible judge APIs do not consistently expose the option-token probabilities used in the original release.

Generate the full dataset from the release CSV, then upload it to the dataset registry:

```bash
python benchmarks/even_handedness/prepare.py /path/to/eval_set.csv
gym dataset upload --storage gitlab --name even_handedness --revision 0.0.1 --input benchmarks/even_handedness/data/even_handedness_eval.jsonl
```
