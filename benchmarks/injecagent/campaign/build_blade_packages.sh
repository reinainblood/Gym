#!/usr/bin/env bash
# Build one validated BLADE run package per model, plus a cross-model index.
#
# The package builder is per-run by design: a package is the evidence for one model on one
# dataset, checksummed end to end. A four-model deliverable is therefore four packages and
# an index that states the denominator they share — not a merged package, which would lose
# the per-run manifests the checksums cover.
#
# Usage:  bash build_blade_packages.sh <output-dir>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

OUT="${1:?usage: build_blade_packages.sh <output-dir>}"
RESULTS_DIR="${RESULTS_DIR:-results/injecagent}"
DATASET="benchmarks/injecagent/data/injecagent_base.jsonl"
RUN_DATE="${RUN_DATE:-20260920}"
# Pinned so a rebuild from the same inputs is byte-identical.
GENERATED_AT="${GENERATED_AT:-2026-09-20T18:00:00Z}"
ENDPOINT_TYPE="modal-fdr-openai-compatible"

# slug | model id | package suffix
MODELS=(
"nemotron-3-ultra-550b|nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4|nemotron-3-ultra-550b"
"qwen3.5-122b-a10b|Qwen/Qwen3.5-122B-A10B-FP8|qwen3-5-122b-a10b"
"kimi-k3|moonshotai/Kimi-K3|kimi-k3"
"nemotron-3.5-super-vl|nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16|nemotron-3-5-super-vl"
)

mkdir -p "$OUT"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r slug model suffix <<< "$entry"
    rollouts="${RESULTS_DIR}/${slug}.jsonl"
    if [ ! -s "$rollouts" ]; then
        echo "SKIP ${model}: no rollouts at ${rollouts}" >&2
        continue
    fi
    rows=$(wc -l < "$rollouts")
    if [ "$rows" -lt 1054 ]; then
        echo "SKIP ${model}: incomplete (${rows}/1054)" >&2
        continue
    fi

    echo "building ${model} (${rows} rollouts)"
    .venv/bin/python -m benchmarks.injecagent.reporting.build_package \
        --rollouts "$rollouts" \
        --failures "${RESULTS_DIR}/${slug}_failures.jsonl" \
        --materialized-inputs "${RESULTS_DIR}/${slug}_materialized_inputs.jsonl" \
        --aggregate-metrics "${RESULTS_DIR}/${slug}_aggregate_metrics.json" \
        --dataset "$DATASET" \
        --run-id "${RUN_DATE}-${suffix}" \
        --model "$model" \
        --endpoint-type "$ENDPOINT_TYPE" \
        --generated-at "$GENERATED_AT" \
        --output "${OUT}/injecagent-${suffix}-${RUN_DATE}"
done

echo "packages in ${OUT}:"
ls -1 "$OUT"
