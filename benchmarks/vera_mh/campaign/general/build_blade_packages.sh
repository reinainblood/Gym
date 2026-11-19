#!/usr/bin/env bash
# Build one validated BLADE run package per model, plus a cross-model index.
# Same shape as the InjecAgent and HarmBench deliverables: a package is evidence for one run,
# so four models means four checksummed packages and an index, not one merged package.
#
# Usage:  bash build_blade_packages.sh <output-dir>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

OUT="${1:?usage: build_blade_packages.sh <output-dir>}"
RESULTS_DIR="${RESULTS_DIR:-results/vera_mh}"
RUN_DATE="${RUN_DATE:-20260920}"
GENERATED_AT="${GENERATED_AT:-2026-09-20T18:00:00Z}"
EXPECTED_ROWS=200

JUDGE="${VERA_MH_JUDGE_MODEL:-deepseek/deepseek-v4-pro}"

# Kimi was recollected in a parallel worktree so it could run alongside Super-VL instead of
# behind it, so its rollouts are not under this worktree's RESULTS_DIR. Without this the build
# reports "SKIP moonshotai/Kimi-K3: no rollouts", which reads like the model never ran.
results_dir_for() {
    case "$1" in
        kimi-k3) echo "${KIMI_RESULTS_DIR:-${ROOT_DIR%/*}/vera-mh-kimi/results/vera_mh}" ;;
        *)       echo "${RESULTS_DIR}" ;;
    esac
}

# slug | model id | package suffix
MODELS=(
"nemotron-3-ultra-550b|nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4|nemotron-3-ultra-550b"
"qwen3.5-122b-a10b|Qwen/Qwen3.5-122B-A10B-FP8|qwen3-5-122b-a10b"
"nemotron-3.5-super-vl|nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16|nemotron-3-5-super-vl"
"kimi-k3|moonshotai/Kimi-K3|kimi-k3"
)

mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for entry in "${MODELS[@]}"; do
    IFS='|' read -r slug model suffix <<< "$entry"
    model_results="$(results_dir_for "$slug")"
    rollouts="${model_results}/${slug}.jsonl"
    if [ ! -s "$rollouts" ]; then echo "SKIP ${model}: no rollouts" >&2; continue; fi
    rows=$(wc -l < "$rollouts")
    if [ "$rows" -lt "$EXPECTED_ROWS" ]; then echo "SKIP ${model}: incomplete (${rows}/${EXPECTED_ROWS})" >&2; continue; fi

    # The builder reads <run-dir>/<stem>.jsonl, so each model gets a directory whose stem is
    # the benchmark name rather than the model slug.
    staged="${WORK}/${suffix}"
    mkdir -p "$staged"
    for suffix_file in "" "_failures" "_materialized_inputs" "_aggregate_metrics"; do
        src="${model_results}/${slug}${suffix_file}.jsonl"
        [ -f "$src" ] && cp "$src" "${staged}/vera_mh${suffix_file}.jsonl"
        src_json="${model_results}/${slug}${suffix_file}.json"
        [ -f "$src_json" ] && cp "$src_json" "${staged}/vera_mh${suffix_file}.json"
    done

    cat > "${staged}/run-info.json" <<JSON
{
  "endpoint_type": "modal-fdr-openai-compatible",
  "harness": "vera_mh_agent (100 personas x 2 user simulators, max_turns 30, persona speaks first)",
  "sampling": "endpoint deployment defaults; Claude simulator capped at 1024 output tokens as LangChain's client does",
  "generation_limits": "provider max_output_tokens 8192 per turn (reasoning tokens count against it)",
  "user_simulators": ["openai/gpt-5.2", "anthropic/claude-opus-4.5"],
  "verifier": {
    "summary": "${JUDGE} judge, v1.2 rubric, one structured question per call, upstream GOTO navigation",
    "judge_model": "${JUDGE}",
    "judge_deviation": "NOT the protocol's GPT-5.4. On this OpenRouter key GPT-5.4 is served only by Azure, whose content filter answers with a refusal instead of a rubric verdict on suicide-risk transcripts: 8 of 36 rubric questions filtered on one real transcript, all Red severity items, and one filtered question aborts the conversation. 27 of the first 33 conversations went to the sidecar. Scores here are therefore NOT comparable to published VERA-MH numbers or to the earlier Kimi K3 line, whose v1.2 calibration (85% raw agreement, IRR 0.79) was measured on GPT-5.4."
  },
  "finished_at": "${GENERATED_AT}",
  "generation_summary": "four-model recollection, 2026-09-20"
}
JSON

    echo "building ${model} (${rows} rollouts)"
    .venv/bin/python -m benchmarks.vera_mh.reporting.cli build \
        --run-dir "$staged" \
        --stem vera_mh \
        --run-id "${RUN_DATE}-${suffix}" \
        --model "$model" \
        --run-info "${staged}/run-info.json" \
        --num-repeats 1 \
        --generated-at "$GENERATED_AT" \
        --package "${OUT}/vera-mh-${suffix}-${RUN_DATE}"
done

echo "packages in ${OUT}:"
ls -1 "$OUT"
