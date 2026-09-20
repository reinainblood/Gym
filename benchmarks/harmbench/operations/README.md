# HarmBench FDR operations

These scripts preserve the Modal FDR checkpoint, gradient, and full-attack
workflows used during the NVIDIA HarmBench evaluation. They are operational
companions to the reusable Gym benchmark and resources server; they are not
substitutes for `gym eval run`, reconciliation, or report validation.

All Modal resources are selected explicitly in the `FDR` environment. The
scripts never contain credentials. Modal authentication must already be
configured on the invoking host.

## Pinned upstream checkout

The white-box launchers require the public HarmBench repository at commit
`8e1604d1171fe8a48d8febecd22f600e462bdcdd`. Set its local path before a Modal
build:

```bash
export HARMBENCH_UPSTREAM=/absolute/path/to/HarmBench
test "$(git -C "$HARMBENCH_UPSTREAM" rev-parse HEAD)" = \
  8e1604d1171fe8a48d8febecd22f600e462bdcdd
```

The checkout itself is intentionally not vendored. The launchers copy only the
required pinned public source, configuration, and data into their Modal images.

## Contents

- `qwen/`: hydrate the official Qwen 3.5 122B-A10B BF16 checkpoint, prove a
  real raw-pixel gradient, and incrementally execute public multimodal PGD
  methods.
- `kimi/`: verify the exact served Kimi K3 checkpoint and preserve the native
  text/vision gradient and public-method runners. The included failure receipt
  documents why the public MXFP4 checkpoint cannot currently execute the
  differentiable methods on eight B300s without an unsupported surrogate.
- `ultra/`: hydrate the exact Ultra checkpoints, prove BF16 input gradients,
  and incrementally execute the pinned public text white-box methods.

Every long-running method persists individual behavior outputs to a named
Modal Volume. Inspect and reconcile those receipts before resuming; directory
names and deployed apps alone are not completion evidence.
