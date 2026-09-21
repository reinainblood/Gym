#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Keep the defense grid moving without a human in the loop: as cells finish and free their
# stack, start the next outstanding one, costliest first.
#
# WHY A TENDER RATHER THAN LAUNCHING EVERYTHING
#
# All twenty cells at once does not fit. Each stack holds an AgentDyn environment and, for the
# detector defenses, a CPU classifier; twenty of those alongside the other campaigns on this
# host pushed it to 47G of 51G with 20G in the compressor, and a machine under that kind of
# memory pressure starts killing processes -- including the neighbours' runs. So the grid runs
# a bounded number of stacks and refills as they drain.
#
# It also holds a model back while that model's endpoint is saturated. Qwen and Super-VL are
# single-replica and queue rather than erroring, so a saturated endpoint shows up as 20-30x
# latency and no errors at all: starting a cell there would crawl and slow whoever already has
# the endpoint. The probe below is the only way to tell that apart from a slow client.
#
# Usage:  nohup bash benchmarks/agentdyn/tend_defense_grid.sh > results/agentdyn-defense-matrix/logs/tender.log 2>&1 &
#         MAX_STACKS=10 bash benchmarks/agentdyn/tend_defense_grid.sh
#
# Stop it with `kill <pid>`; it never kills a running cell, so stopping the tender just stops
# new ones from starting.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/agentdyn-defense-matrix}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
# Deliberately small. Each stack is a full Ray cluster, not just two FastAPI servers, and
# this host is shared with other benchmark campaigns. Fourteen of these plus their neighbours
# took the machine to 15MB free and ~8GB across ~466 processes, which puts every campaign on
# it at risk of an OOM kill, not only this one. Raise it only when the box is otherwise idle.
MAX_STACKS="${MAX_STACKS:-5}"
# Gate on what the OS itself thinks, not on "Pages free". macOS keeps free pages near zero by
# design and holds the rest as active/inactive, so a free-pages threshold either never trips or
# never clears. `memory_pressure` reports the number that actually tracks headroom here.
MIN_FREE_PCT="${MIN_FREE_PCT:-25}"
INTERVAL="${INTERVAL:-300}"
# Above this, an endpoint is queueing behind someone else's campaign, not merely slow.
MAX_ENDPOINT_SECONDS="${MAX_ENDPOINT_SECONDS:-5}"
DRIFT_SHARDS="${DRIFT_SHARDS:-3}"
mkdir -p "$LOG_DIR"

set -a; source "${ENV_FILE:-/Users/kruge/Documents/ChatGPT/NVIDIA/.env}"; set +a

live_stacks() {
    local count=0 lock pid
    for lock in "${RESULTS_DIR}"/.runner.*.lock; do
        [ -e "$lock" ] || continue
        pid=$(cat "$lock" 2>/dev/null)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && count=$(( count + 1 ))
    done
    echo "$count"
}

free_pct() {
    memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print int($2)}'
}

# 1 normal, 2 warning, 4 critical. At critical the kernel is already choosing processes to kill,
# and the neighbouring campaigns are as likely to be chosen as this one.
pressure_critical() {
    [ "$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)" -ge 4 ]
}

endpoint_seconds() {
    local key="$1" url token model
    case "$key" in
        ultra) url=https://snorkelai-fdr--ep-nvidia-nemotron-3-ultra-550b-a55b-nvfp-63eebc.us-west.modal.direct/v1
               model=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4; token="${MODAL_PROXY_TOKEN:-}" ;;
        kimi)  url=https://snorkelai-fdr--ep-kimi-k3-server.us-west.modal.direct/v1
               model=moonshotai/Kimi-K3; token="${MODAL_PROXY_TOKEN:-}" ;;
        qwen)  url=https://snorkelai-fdr--ep-qwen3-5-122b-a10b-fp8-server.us-west.modal.direct/v1
               model=Qwen/Qwen3.5-122B-A10B-FP8; token="${MODAL_PROXY_TOKEN:-}" ;;
        supervl) url=https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct/v1
               model=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16; token="${SUPER_VL_MODAL_TOKEN:-}" ;;
        *) echo 999; return ;;
    esac
    local seconds
    seconds=$(curl -s -m 60 -o /dev/null -w "%{time_total}" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}" \
        "${url}/chat/completions" 2>/dev/null)
    echo "${seconds:-999}"
}

echo "[$(date +%H:%M:%S)] tending: max ${MAX_STACKS} stacks, >= ${MIN_FREE_PCT}% memory free, every ${INTERVAL}s"

while true; do
    stacks=$(live_stacks)
    free=$(free_pct); free="${free:-0}"
    remaining=$(.venv/bin/python benchmarks/agentdyn/summarize_defense_matrix.py --next 99 | wc -l | tr -d ' ')

    if [ "$remaining" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] nothing outstanding without a runner; ${stacks} stacks still working"
        [ "$stacks" -eq 0 ] && { echo "[$(date +%H:%M:%S)] grid drained; tender exiting"; exit 0; }
    elif [ "$stacks" -ge "$MAX_STACKS" ]; then
        echo "[$(date +%H:%M:%S)] holding: ${stacks}/${MAX_STACKS} stacks, ${free}% free, ${remaining} cells waiting"
    elif pressure_critical; then
        echo "[$(date +%H:%M:%S)] holding: kernel reports critical memory pressure, ${stacks} stacks"
    elif [ "$free" -lt "$MIN_FREE_PCT" ]; then
        echo "[$(date +%H:%M:%S)] holding: ${free}% memory free (want ${MIN_FREE_PCT}%), ${stacks} stacks"
    else
        started=0
        while read -r key defense; do
            [ -z "$key" ] && continue
            seconds=$(endpoint_seconds "$key")
            if awk "BEGIN{exit !($seconds > $MAX_ENDPOINT_SECONDS)}"; then
                echo "[$(date +%H:%M:%S)] skipping ${key}/${defense}: endpoint at ${seconds}s, still busy"
                continue
            fi
            echo "[$(date +%H:%M:%S)] starting ${key}/${defense} (endpoint ${seconds}s, ${stacks} stacks, ${free}% free)"
            if [ "$defense" = "drift" ]; then
                SHARDS="$DRIFT_SHARDS" DEFENSES="$defense" bash benchmarks/agentdyn/launch_defense_grid.sh "$key"
            else
                DEFENSES="$defense" bash benchmarks/agentdyn/launch_defense_grid.sh "$key"
            fi
            started=1
            break
        done < <(.venv/bin/python benchmarks/agentdyn/summarize_defense_matrix.py --next 5)
        [ "$started" -eq 0 ] && echo "[$(date +%H:%M:%S)] nothing startable this pass"
    fi

    sleep "$INTERVAL"
done
