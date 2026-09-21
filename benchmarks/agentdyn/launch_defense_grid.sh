#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fan the defense grid out over processes: one `run_defense_matrix.sh` invocation per
# (model, defense) cell, each with its own head port and port block.
#
# WHY ONE PROCESS PER CELL
#
# The agent server runs rollouts behind `asyncio.Semaphore(concurrency)` with concurrency 1,
# so `gym eval run --concurrency N` does not make a cell faster -- the requests just queue at
# the agent. That semaphore is not arbitrary: setting up a defense monkeypatches process
# globals (`TransformersBasedPIDetector`, `openai.OpenAI`, `os.environ`), and overlapping
# `mock.patch` scopes in one process restore each other's values out of order, which would
# hand a rollout the unpatched client and send it at the public OpenAI endpoint. Raising the
# semaphore is therefore not a safe lever. Separate processes share no globals, so this is
# the way to use the machine.
#
# Usage:  bash benchmarks/agentdyn/launch_defense_grid.sh [model_key ...]
#         DEFENSES="drift" bash benchmarks/agentdyn/launch_defense_grid.sh ultra kimi
#
# Each cell writes results/agentdyn-defense-matrix/<slug>/<defense>.jsonl and resumes, so
# re-running after an interruption costs nothing already collected.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/agentdyn-defense-matrix}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
SHARD_DIR="${SHARD_DIR:-${RESULTS_DIR}/shards}"
DEFENSES="${DEFENSES:-prompt_guard_2_detector camel progent piguard_detector drift transformers_pi_detector spotlighting_with_delimiting repeat_user_prompt tool_filter}"
INPUT="${INPUT:-benchmarks/agentdyn/data/agentdyn_v1_2_2.jsonl}"
# SHARDS splits one cell's 620 selectors across that many processes. A cell's score is a
# pure function of its row set -- masked rows leave the denominator, then utility and attack
# success are averaged over the benign and attacked subsets -- so splitting the rows and
# recombining them scores identically to one process doing all 620. Worth it for DRIFT,
# which at ~50 policy calls per rollout projects past forty hours as a single process.
SHARDS="${SHARDS:-1}"
# Each cell reserves four shard head ports; more shards would take the next cell's, and the
# summarizer would then report another cell's shards as this one's.
if [ "$SHARDS" -gt 4 ]; then
    echo "ERROR: SHARDS=${SHARDS} exceeds the four head ports reserved per cell." >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$SHARD_DIR"

# Model index fixes the port neighbourhood; see run_defense_matrix.sh for the block registry.
MODEL_KEYS=(ultra kimi qwen supervl)
# Order fixes each cell's ports; append new defenses rather than reordering.
DEFENSE_KEYS=(prompt_guard_2_detector camel progent piguard_detector drift \
              transformers_pi_detector spotlighting_with_delimiting repeat_user_prompt tool_filter)
DEFENSES_PER_MODEL=${#DEFENSE_KEYS[@]}
MODEL_PORT_BASE=(45001 46001 47001 48001)

index_of() {
    local needle="$1"; shift
    local i=0
    for item in "$@"; do
        [ "$item" = "$needle" ] && { echo "$i"; return 0; }
        i=$(( i + 1 ))
    done
    return 1
}

WANTED=("$@")
[ ${#WANTED[@]} -eq 0 ] && WANTED=("${MODEL_KEYS[@]}")

for key in "${WANTED[@]}"; do
    model_index=$(index_of "$key" "${MODEL_KEYS[@]}") || { echo "unknown model key: $key" >&2; exit 1; }
    for defense in $DEFENSES; do
        defense_index=$(index_of "$defense" "${DEFENSE_KEYS[@]}") || {
            echo "unknown defense: $defense" >&2; exit 1; }
        # Head ports 11820-11853, one per cell, clear of every campaign block on this host.
        head_port=$(( 11820 + model_index * 10 + defense_index ))
        # 100 ports per cell so nine cells fit inside each model's 900-port block. A stack
        # binds two or three, so the width is ample.
        port_low=$(( MODEL_PORT_BASE[model_index] + defense_index * 100 ))
        port_high=$(( port_low + 99 ))
        log="${LOG_DIR}/${key}.${defense}.cell.log"

        if [ -f "${RESULTS_DIR}/.runner.${head_port}.lock" ] \
           && kill -0 "$(cat "${RESULTS_DIR}/.runner.${head_port}.lock" 2>/dev/null)" 2>/dev/null; then
            echo "skip ${key}/${defense}: already running on head port ${head_port}"
            continue
        fi

        if [ "$SHARDS" -le 1 ]; then
            echo "launch ${key}/${defense}  head=${head_port} ports=${port_low}-${port_high}  -> ${log}"
            HEAD_PORT="$head_port" PORT_LOW="$port_low" PORT_HIGH="$port_high" \
            ENV_YAML="${RESULTS_DIR}/env.${head_port}.yaml" \
            DEFENSES="$defense" \
            nohup bash benchmarks/agentdyn/run_defense_matrix.sh "$key" > "$log" 2>&1 &
            sleep 2
            continue
        fi

        cell_index=$(( model_index * DEFENSES_PER_MODEL + defense_index ))
        for shard in $(seq 0 $(( SHARDS - 1 ))); do
            shard_input="${SHARD_DIR}/$(basename "${INPUT%.jsonl}").shard${shard}of${SHARDS}.jsonl"
            # Contiguous split, written once and reused, so a relaunch resumes the same rows.
            if [ ! -s "$shard_input" ]; then
                .venv/bin/python - "$INPUT" "$shard_input" "$shard" "$SHARDS" <<'SPLIT'
import sys
from pathlib import Path

source, target, index, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = source.read_text(encoding="utf-8").splitlines(keepends=True)
size, remainder = divmod(len(rows), count)
start = index * size + min(index, remainder)
stop = start + size + (1 if index < remainder else 0)
target.write_text("".join(rows[start:stop]), encoding="utf-8")
print(f"{target} rows={stop - start}")
SPLIT
            fi
            shard_rows=$(wc -l < "$shard_input" | tr -d ' ')
            shard_head=$(( 11900 + cell_index * 4 + shard ))
            shard_low=$(( 50001 + cell_index * 200 + shard * 50 ))
            shard_log="${LOG_DIR}/${key}.${defense}.shard${shard}.cell.log"

            if [ -f "${RESULTS_DIR}/.runner.${shard_head}.lock" ] \
               && kill -0 "$(cat "${RESULTS_DIR}/.runner.${shard_head}.lock" 2>/dev/null)" 2>/dev/null; then
                echo "skip ${key}/${defense} shard ${shard}: already running on head port ${shard_head}"
                continue
            fi

            echo "launch ${key}/${defense} shard ${shard}/${SHARDS} (${shard_rows} rows)  head=${shard_head}"
            HEAD_PORT="$shard_head" PORT_LOW="$shard_low" PORT_HIGH="$(( shard_low + 49 ))" \
            ENV_YAML="${RESULTS_DIR}/env.${shard_head}.yaml" \
            DEFENSES="$defense" INPUT="$shard_input" OUT_SUFFIX="shard${shard}of${SHARDS}" \
            EXPECTED="$shard_rows" \
            nohup bash benchmarks/agentdyn/run_defense_matrix.sh "$key" > "$shard_log" 2>&1 &
            sleep 2
        done
    done
done

echo "All requested cells launched. Watch with:"
echo "  python benchmarks/agentdyn/summarize_defense_matrix.py"
