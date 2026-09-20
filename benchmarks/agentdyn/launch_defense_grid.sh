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
DEFENSES="${DEFENSES:-prompt_guard_2_detector camel progent piguard_detector drift}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Model index fixes the port neighbourhood; see run_defense_matrix.sh for the block registry.
MODEL_KEYS=(ultra kimi qwen supervl)
DEFENSE_KEYS=(prompt_guard_2_detector camel progent piguard_detector drift)
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
        port_low=$(( MODEL_PORT_BASE[model_index] + defense_index * 150 ))
        port_high=$(( port_low + 149 ))
        log="${LOG_DIR}/${key}.${defense}.cell.log"

        if [ -f "${RESULTS_DIR}/.runner.${head_port}.lock" ] \
           && kill -0 "$(cat "${RESULTS_DIR}/.runner.${head_port}.lock" 2>/dev/null)" 2>/dev/null; then
            echo "skip ${key}/${defense}: already running on head port ${head_port}"
            continue
        fi

        echo "launch ${key}/${defense}  head=${head_port} ports=${port_low}-${port_high}  -> ${log}"
        HEAD_PORT="$head_port" PORT_LOW="$port_low" PORT_HIGH="$port_high" \
        ENV_YAML="${RESULTS_DIR}/env.${head_port}.yaml" \
        DEFENSES="$defense" \
        nohup bash benchmarks/agentdyn/run_defense_matrix.sh "$key" > "$log" 2>&1 &
        sleep 2
    done
done

echo "All requested cells launched. Watch with:"
echo "  python benchmarks/agentdyn/summarize_defense_matrix.py"
