# Payload-free HarmBench evidence

This directory contains the small, payload-free receipts needed to verify the
headline Ultra ZeroShot, Ultra PAP-top5, and Qwen multimodal results from a
fresh Git checkout. The files were copied from the original run packages after
their hashes and reconciliation counts were validated.

These receipts intentionally contain no attack prompts, target generations,
model weights, credentials, or private data. Full rollout JSONL, call captures,
optimized images, and other large case-level artifacts remain outside Git in
the original ignored result tree and/or named Modal Volumes. Their hashes in
the committed manifests bind those external artifacts without publishing the
payloads.

The existence of a report or manifest establishes only the gate described by
that receipt. It does not imply a four-model weekly batch, Kirsten review,
upstream PR, or merge.

`SHA256SUMS` binds every committed evidence file in this directory. Verify it
from the repository root with `shasum -a 256 -c benchmarks/harmbench/evidence/SHA256SUMS`.

- `ultra-zeroshot/`: validated-with-caveat 1,600-case Ultra ZeroShot receipts.
- `ultra-pap-top5/`: validated 1,600-case Ultra PAP-top5 receipts.
- `qwen-multimodal/`: corrected Qwen MultiModalDirectRequest and
  MultiModalRenderText reports, classifier controls, reverify comparisons, and
  combined BLADE metrics.
- `qwen-whitebox-pgd/`: payload-free receipt-set hash migration record for the
  110 completed Qwen MultiModalPGD generation artifacts. It retains the legacy
  path-dependent audit hash and the portable canonical-content hash with their
  distinct algorithms; it is not a scored HarmBench result.
