#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the AgentDyn defense grid: every defense treatment over the same 620 selectors that
# produced the undefended baselines in BASELINE-VALIDATION.md.
#
#   4 models x 5 defenses x 620 selectors = 12,400 rollouts
#
# The selector file is NOT regenerated per defense. A treatment is selected with
# `default_defense` on the agent server, so every cell scores the byte-identical matrix
# (sha256 819443fe...) the baselines used and the comparison stays honest.
#
# Usage:  bash benchmarks/agentdyn/run_defense_matrix.sh [model_key ...]
#         DEFENSES="camel drift" bash benchmarks/agentdyn/run_defense_matrix.sh qwen
#         LIMIT=4 OUT_SUFFIX=canary bash benchmarks/agentdyn/run_defense_matrix.sh supervl
#
# One invocation per model may run in parallel: each model key owns a fixed head port and
# port block below, so two invocations never share a stack and the port-scoped cleanup in
# benchmark_runner_lib.sh cannot kill a neighbouring campaign. Passing several keys to one
# invocation runs them sequentially through that first key's ports.
#
# Defenses run cheap-first. DRIFT issues ~50 policy calls per rollout against ~9 undefended,
# so it costs more than the other four combined; ordering it last means an interrupted
# campaign still leaves complete, comparable slices rather than one model half-covered.
#
# Every stage passes --resume, so re-running after an interruption picks up where it
# stopped instead of re-spending GPU time on rollouts that already landed.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/agentdyn-defense-matrix}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
INPUT="${INPUT:-benchmarks/agentdyn/data/agentdyn_v1_2_2.jsonl}"
# Per-model, from the MODELS table below, because the deployments do not absorb the same
# load. Set CONCURRENCY to force one value for every model. Concurrency is a throughput
# knob, not an experimental variable: each rollout runs an isolated suite environment and
# is verified deterministically, so a cell scores the same at 4 as at 16. Whatever a cell
# actually used is recorded with its result.
CONCURRENCY="${CONCURRENCY:-}"
EXPECTED="${EXPECTED:-620}"
LIMIT="${LIMIT:-}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
DEFENSES="${DEFENSES:-prompt_guard_2_detector camel progent piguard_detector drift}"
GYM_RUNNER_NAME="agentdyn"
# The detector defenses download their classifiers from the Hub. Gym defaults HF_HOME to
# <checkout>/cache/huggingface, which for a worktree means re-downloading what the main
# checkout already holds; point every cell at the one shared cache instead.
export HF_HOME="${HF_HOME:-/Users/kruge/Documents/ChatGPT/NVIDIA/cache/huggingface}"
# The detector classifiers run on CPU here (no CUDA on this host), and torch otherwise
# grabs a thread per core in every concurrent rollout of every parallel model campaign.
# One thread each keeps the detector cells from starving the neighbouring campaigns; the
# classifiers are small enough that per-call latency barely moves.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# key | slug | base_url | model id | token env var | head port | port low | port high | concurrency
#
# Concurrency is measured, not guessed (FDR endpoints, 2026-09-20, ~32k ASB rollouts):
# ultra and kimi hold up well past 64 concurrent requests, while qwen and supervl are
# single-replica and queue rather than erroring -- their latency inflates 20-30x under load
# with zero 503s, so pushing more client concurrency at them buys nothing and only starves
# the neighbouring campaigns. The ceilings here are further limited by this host: every
# rollout runs an AgentDyn suite in-process, and the detector defenses hold a CPU
# classifier per concurrent rollout.
#
# Ports are disjoint from the campaign blocks registered in scripts/benchmark_runner_lib.sh
# (asb 20001-29000, kidbench 30001-34000, xstest 34001-38000, over_refusal 38001-42000) and
# from Gym's default 10001-20000, so a parallel AgentDyn campaign cannot sweep up a
# neighbour's servers on cleanup.
MODELS=(
"ultra|nemotron-3-ultra|https://snorkelai-fdr--ep-nvidia-nemotron-3-ultra-550b-a55b-nvfp-63eebc.us-west.modal.direct/v1|nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4|MODAL_PROXY_TOKEN|11810|45001|45900|16"
"kimi|kimi-k3|https://snorkelai-fdr--ep-kimi-k3-server.us-west.modal.direct/v1|moonshotai/Kimi-K3|MODAL_PROXY_TOKEN|11811|46001|46900|16"
"qwen|qwen-3-5-122b-a10b|https://snorkelai-fdr--ep-qwen3-5-122b-a10b-fp8-server.us-west.modal.direct/v1|Qwen/Qwen3.5-122B-A10B-FP8|MODAL_PROXY_TOKEN|11812|47001|47900|6"
"supervl|nemotron-3-5-super-vl|https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct/v1|nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16|SUPER_VL_MODAL_TOKEN|11813|48001|48900|6"
)

source "${ROOT_DIR}/scripts/benchmark_runner_lib.sh"
set -a; source "${ENV_FILE:-/Users/kruge/Documents/ChatGPT/NVIDIA/.env}"; set +a

write_env_yaml() {
    local base_url="$1" model="$2" token_var="$3" defense="$4" system_role="$5"
    {
    cat <<YAML
# Generated by benchmarks/agentdyn/run_defense_matrix.sh -- do not edit by hand.
YAML
    gym_runner_ports_yaml
    cat <<YAML

# The treatment. Setting it on the server rather than on the rows keeps the selector file
# byte-identical to the one the undefended baselines scored.
agentdyn_benchmark:
  responses_api_agents:
    agentdyn_agent:
      default_defense: ${defense}
      model_system_role: ${system_role}

policy_model:
  responses_api_models:
    inference_provider:
      entrypoint: app.py
      base_url: ${base_url}
      api_key: \${oc.env:${token_var},""}
      model: ${model}
      uses_reasoning_parser: true
      num_concurrent_requests: 64
YAML
    } > "$ENV_YAML"
}

# Wait until every child server answers, not just the head. The head binds immediately
# while `gym env start` is still building the agent's sub-venv, so a head-only wait hands
# `gym eval run` a dead agent port and the run burns its retry budget on ClientOSError.
# First start of a fresh checkout installs AgentDyn and torch, hence the long ceiling.
wait_for_all_servers() {
    local log="$1" max_wait="${2:-1800}" waited=0 urls ready total
    while [ "$waited" -lt "$max_wait" ]; do
        if [ -n "$log" ] && grep -q "Head server finished unexpectedly" "$log" 2>/dev/null; then
            echo "  ERROR: server stack died (see ${log})" >&2
            return 1
        fi
        urls=$(curl -s -m 5 "http://127.0.0.1:${HEAD_PORT}/server_instances" 2>/dev/null \
               | .venv/bin/python -c "
import json, sys
try:
    print(' '.join(i['url'] for i in json.load(sys.stdin) if i.get('url')))
except Exception:
    pass" 2>/dev/null)
        if [ -n "$urls" ]; then
            ready=0; total=0
            for url in $urls; do
                total=$(( total + 1 ))
                curl -sf -m 5 "${url}/docs" >/dev/null 2>&1 && ready=$(( ready + 1 ))
            done
            if [ "$total" -gt 0 ] && [ "$ready" -eq "$total" ]; then
                echo "  all ${total} servers ready after ${waited}s"
                return 0
            fi
        fi
        sleep 10
        waited=$(( waited + 10 ))
    done
    echo "  ERROR: servers did not all come up within ${max_wait}s (see ${log})" >&2
    return 1
}

start_servers() {
    local tag="$1"
    nohup .venv/bin/gym env start \
        --config benchmarks/agentdyn/config.yaml --config "$ENV_YAML" \
        > "${LOG_DIR}/${tag}.servers.log" 2>&1 &
    # Poll our own head port rather than `gym env status`, which takes no --config and so
    # reports whatever is on the default port 11000 -- another campaign's stack, not ours.
    gym_runner_wait_for_head "${LOG_DIR}/${tag}.servers.log" || return 1
    wait_for_all_servers "${LOG_DIR}/${tag}.servers.log"
}

run_cell() {
    local slug="$1" defense="$2" concurrency="$3"
    local tag="${slug}.${defense}${OUT_SUFFIX:+.${OUT_SUFFIX}}"
    local out="${RESULTS_DIR}/${slug}/${defense}${OUT_SUFFIX:+-${OUT_SUFFIX}}.jsonl"
    local attempt status rows expected="$EXPECTED"
    [ -n "$LIMIT" ] && expected="$LIMIT"
    mkdir -p "$(dirname "$out")"

    for attempt in 1 2 3; do
        if ! gym_runner_head_is_up; then
            echo "  [$(date +%H:%M:%S)] ${tag} head server down; restarting stack"
            gym_runner_stop_servers
            start_servers "$tag" || return 1
        fi
        echo "  [$(date +%H:%M:%S)] ${tag} attempt ${attempt} (concurrency ${concurrency}) -> ${out}"
        .venv/bin/gym eval run --no-serve --resume \
            --config benchmarks/agentdyn/config.yaml --config "$ENV_YAML" \
            --agent agentdyn_benchmark \
            --input "$INPUT" \
            --output "$out" \
            --num-repeats 1 \
            --concurrency "$concurrency" \
            ${LIMIT:+--limit "$LIMIT"} \
            > "${LOG_DIR}/${tag}.attempt${attempt}.log" 2>&1
        status=$?
        rows=$(wc -l < "$out" 2>/dev/null || echo 0)
        echo "  [$(date +%H:%M:%S)] ${tag} attempt ${attempt} exit=${status} rows=${rows}/${expected}"

        if [ "$status" -eq 0 ] && [ "$rows" -ge "$expected" ]; then
            echo "  [$(date +%H:%M:%S)] ${tag} complete"
            return 0
        fi

        # A 503 from an overloaded router means back off, not push harder. Halve the
        # ceiling and let the endpoint's circuit breakers reset before trying again.
        concurrency=$(( concurrency / 2 ))
        [ "$concurrency" -lt 2 ] && concurrency=2
        echo "  [$(date +%H:%M:%S)] ${tag} backing off to concurrency ${concurrency}"
        sleep 60
    done

    echo "  [$(date +%H:%M:%S)] ${tag} INCOMPLETE after 3 attempts (${rows}/${expected})"
    return 1
}

WANTED=("$@")
SELECTED=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r key _rest <<< "$entry"
    if [ ${#WANTED[@]} -gt 0 ] && [[ ! " ${WANTED[*]} " =~ " ${key} " ]]; then
        continue
    fi
    SELECTED+=("$entry")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "ERROR: no model keys matched: ${WANTED[*]:-<all>}" >&2
    exit 1
fi

# Ports come from the first selected model, so `... ultra kimi` (sequential, one stack at a
# time) and two parallel single-key invocations both get a coherent, non-overlapping block.
# An explicit HEAD_PORT/PORT_LOW/PORT_HIGH wins, which is how the grid runs several cells of
# one model at once: the agent server serializes rollouts behind a semaphore of one (its
# defense setup monkeypatches process globals, so concurrent rollouts inside one process
# would race over them), and the only safe way to go faster is more processes.
IFS='|' read -r _k _s _u _m _t table_head table_low table_high _c <<< "${SELECTED[0]}"
HEAD_PORT="${HEAD_PORT:-$table_head}"
PORT_LOW="${PORT_LOW:-$table_low}"
PORT_HIGH="${PORT_HIGH:-$table_high}"
export HEAD_PORT PORT_LOW PORT_HIGH
ENV_YAML="${ENV_YAML:-${RESULTS_DIR}/env.${HEAD_PORT}.yaml}"

gym_runner_lock "$RESULTS_DIR" || exit 1
gym_runner_clear_shadowing_env_yaml || exit 1

for entry in "${SELECTED[@]}"; do
    IFS='|' read -r key slug base_url model token_var _hp _pl _ph model_concurrency <<< "$entry"
    concurrency="${CONCURRENCY:-$model_concurrency}"
    # The Qwen SGLang deployment rejects the `developer` role with "Unexpected message
    # role."; see BASELINE-VALIDATION.md. Every other endpoint takes the default.
    system_role="developer"
    [ "$key" = "qwen" ] && system_role="system"

    echo "==============================================================="
    echo "[$(date +%H:%M:%S)] MODEL ${key} (${model}) on head port ${HEAD_PORT}"
    echo "==============================================================="

    for defense in $DEFENSES; do
        echo "[$(date +%H:%M:%S)] DEFENSE ${defense}"
        write_env_yaml "$base_url" "$model" "$token_var" "$defense" "$system_role"
        gym_runner_stop_servers
        start_servers "${slug}.${defense}" || continue
        run_cell "$slug" "$defense" "$concurrency"
    done
done

gym_runner_stop_servers
echo "[$(date +%H:%M:%S)] All requested cells finished."
