# Kimi K3 HarmBench white-box execution

This lane binds the current FDR-served `moonshotai/Kimi-K3` checkpoint to the
immutable public HarmBench implementation at
`8e1604d1171fe8a48d8febecd22f600e462bdcdd`. It does not substitute proxy
models, calibration prompts, reduced algorithms, or synthetic benchmark cases.

## Checkpoint provenance

- Hugging Face revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- FDR source Volume: `endpoint-ep-WD4gnzaeXMyM7DfrzTDYjW`
- Exact size: 118 files and 1,560,998,984,390 bytes
- Full per-file LFS SHA verification receipt:
  `harmbench-kimi-k3-whitebox-results:/checkpoint-manifest.json`
- Receipt SHA-256:
  `98c4642076606cf1dd3f83706c901f867288febed93f45719c21982514b2ba2b`

K3 is a native image-and-text model (`KimiK3ForConditionalGeneration`) with a
27-layer MoonViT tower. Accordingly, all ten applicable public white-box
methods are in scope; no multimodal method is excluded.

## Public methods and full case counts

| Public method | Public class | Full cases |
| --- | --- | ---: |
| GCG | `GCG` | 320 |
| GCG-Multi | `EnsembleGCG`, public run IDs 0-4 | 1,600 |
| AutoPrompt | `AutoPrompt` | 320 |
| GBDA | `GBDA` | 1,600 |
| PEZ | `PEZ` | 1,600 |
| UAT | `UAT` | 320 |
| AutoDAN | `AutoDAN` | 320 |
| FewShot | `FewShot` | 320 |
| MultiModalPGD | `MultiModalPGD` | 110 |
| MultiModalPGDPatch | `MultiModalPGDPatch` | 110 |

The total full workload is 6,620 generated cases. Method YAMLs, hashes,
published step counts, search widths, repetitions, target files, and behavior
files are checked by `kimi_k3_whitebox.py` before dispatch.

## Runtime

FDR app `harmbench-kimi-k3-whitebox` mounts the production endpoint cache
read-only and writes only to the dedicated
`harmbench-kimi-k3-whitebox-results` Volume. Text attacks instantiate the
original public HarmBench classes. The K3 multimodal adapter exposes native
K3 preprocessing and differentiable pixel-to-target loss to the original
public `MultiModalPGD` and `MultiModalPGDPatch` classes; it does not replace
their attack loops or hyperparameters.

The served checkpoint's custom model code imports the retired Transformers
`OutputRecorder` metadata class, while its serving image contains the newer,
quantization-compatible Transformers release. The runtime keeps that serving
release and restores the former dataclass interface solely for immutable-model
code compatibility. The exact public method implementations remain unmodified.

The snapshot's compression metadata also omits five projection families from
its uncompressed ignore list even though those exact tensors are stored as
ordinary `.weight` values. The runtime creates a read-only loader view whose
only content change is the corresponding ignore regex; all 96 verified weight
shards are symlinked byte-for-byte from the source Volume. Loader-config hashes
are retained with execution receipts.

Outputs are committed to the results Volume after every completed behavior
where the upstream method permits it. Failures include the method, behavior
ID, exception type, message, and traceback; they are not converted into model
scores or silently retried as successful cases.

## Current execution blocker

The exact public checkpoint completes module construction and loads all
497,220 tensor entries across eight B300s. At the first target forward,
`compressed-tensors==0.17.0` invokes its model-wide decompression hook. This
cannot fit: GPU 0 had 267.69 GiB total and 267.65 GiB in use before a further
42 MiB allocation failed, after only 5,645 of 247,296 quantized modules had
been decompressed.

The official Moonshot Hub catalog contains no full-size BF16 Kimi K3
checkpoint, and FDR contains no Kimi K3 BF16 Volume. The serving vLLM MXFP4
kernels operate on the packed checkpoint but do not expose the target-input
autograd path required by HarmBench white-box attacks. Therefore none of the
ten methods is reported as run. Tiny-random fixtures, another model, a
straight-through estimator, or reduced attacks are not valid substitutes.

The retained failure receipt is
`harmbench-kimi-k3-whitebox-results:/failures/k3-text-gradient-20260919g.json`.
