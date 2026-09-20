# Nemotron 3 Ultra white-box HarmBench execution

This execution lane binds all eight public text white-box methods to the aligned
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` checkpoint at immutable
revision `77df655d5e9f8362164ed14dd8b48f8bce657498`.

The served `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` checkpoint was
downloaded and hash verified separately. Its ModelOpt packed tensors are an
inference checkpoint: current Transformers does not recognize its `modelopt`
quantization adapter, and the packed expert dimensions do not match the
differentiable Nemotron-H modules. It therefore cannot execute the public
HarmBench input-gradient methods. BF16 is NVIDIA's public aligned replacement
checkpoint for white-box evaluation and is recorded under its own identity:

- GCG
- GCG-Multi, including public run IDs 0, 1, 2, 3, and 4
- AutoPrompt
- GBDA
- PEZ
- UAT
- AutoDAN
- FewShot, using the public Mixtral 8x7B attacker

`ultra_whitebox.py` executes the classes in the pinned public HarmBench checkout
at commit `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The adapter rejects a different
checkout, pipeline, method config, repetition set, incomplete checkpoint, model
revision, behavior set, or case count. Attack step counts, search widths,
initial strings, learning rates, target strings, sampling settings, and early
stopping settings come directly from the hash-locked upstream YAML files.

Generation logs and test cases remain private run artifacts. The generated
receipt records the public source and config hashes, exact checkpoint manifest,
behavior source, repetitions, and final case hash. Generated cases are imported
into Gym with `benchmarks/harmbench/run_upstream_generation.py` and
`benchmarks/harmbench/prepare_generated.py`, then evaluated with the ordinary
HarmBench target and classifier paths.

The full public text corpus has 400 behaviors (the 320-row held-out test split
plus the 80-row validation split). Expected generated rows are 400 for GCG,
2,000 for the five GCG-Multi repetitions, 400 for AutoPrompt, 2,000 for GBDA,
2,000 for PEZ, 400 for UAT, 400 for AutoDAN, and 400 for FewShot: 8,000
white-box attack cases before target completion and classification. The runtime
uses upstream's default `harmbench_behaviors_text_all.csv`; a 320-row test-only
campaign must not be labeled the full benchmark.

The Modal execution shape uses one eight-B200 node. Seven devices are available
to the target checkpoint; FewShot runs its unchanged public Mixtral 8x7B
attacker on the remaining B200. HarmBench's two-A100 Mixtral allocation is a
hardware capacity setting and is reduced to one B200; its model, revision,
sampling settings, prompts, candidate count, and query loop are unchanged.
