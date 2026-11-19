#!/usr/bin/env bash
# Run VERA-MH's recommended public protocol against the target models.
#
# 100 personas x 2 user simulators = 200 conversations of up to 30 turns each, judged with
# the v1.2 rubric. The simulators and judge are pinned to the same models the existing
# Kimi K3 line used, because VERA-MH scores are only comparable across runs that share the
# rubric, personas, simulators, and judge.
#
# Usage:  bash run_all_models.sh [model_key ...]
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/vera_mh}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

set -a; source "${ENV_FILE:-/Users/kruge/Documents/ChatGPT/NVIDIA/.env}"; set +a

# Other Gym campaigns share this Mac. Pin a head-server port away from the 11000 default so
# two campaigns do not collide on it, and never pattern-kill `python app.py`.
GYM_HEAD_SERVER_PORT="${GYM_HEAD_SERVER_PORT:-11791}"

stop_my_servers() {
    # Select by the process's own working directory, never by a path in argv.
    #
    # These scripts invoke `.venv/bin/gym` and `--output results/...` RELATIVELY, so the
    # worktree path never appears in the command line. A pattern like
    # `gym eval run.*${ROOT_DIR}/${RESULTS_DIR}` therefore matches nothing at all -- it looks
    # correctly scoped and is silently a no-op, which leaves this campaign's own collector
    # alive holding the head-server port, so the next launch dies with "address already in
    # use". The earlier relative pattern had the opposite failure: it matched a sibling
    # worktree's collector and SIGKILLed it. cwd is the only thing that actually identifies
    # which campaign a process belongs to.
    local pid cwd
    for pid in $(pgrep -f "python app.py" 2>/dev/null) $(pgrep -f "gym eval run" 2>/dev/null); do
        cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | tail -1)"
        case "$cwd" in
            "n${ROOT_DIR}"|"n${ROOT_DIR}/"*) kill -9 "$pid" 2>/dev/null ;;
        esac
    done
    sleep 5
}

# Killing the driver does NOT kill the collector it launched: the child keeps running and
# keeps the head-server port bound, so the next driver's first attempt dies with "address
# already in use" and burns a retry. Clean up our own children on any exit, including a
# signal. Scoped to this worktree by stop_my_servers, so a sibling campaign is untouched.
trap 'stop_my_servers' EXIT INT TERM

# The upstream recommended profile, reached through OpenRouter. Model identities are
# recorded per run from what the endpoint actually served, not from these names.
export VERA_MH_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export VERA_MH_JUDGE_API_KEY="${OPENROUTER_API_KEY_SNORKEL}"
# NOT the protocol's GPT-5.4, and the substitution is the whole story of this lane.
#
# VERA-MH v1.2 recommends GPT-5.4 at reasoning_effort=low, and its published calibration
# (85% raw agreement, IRR 0.79 against clinicians) was measured with it. On this OpenRouter
# key GPT-5.4 is served only by Azure, whose content filter answers "I'm sorry, but I cannot
# assist with that request" instead of a rubric verdict when the transcript is a suicide-risk
# conversation. Measured on one real transcript: 8 of 36 rubric questions were content
# filtered (1, 3, 9, 15, 21, 22, 25, 33 -- the Red severity items, which are the ones that
# decide the score). One filtered question aborts the whole conversation, and 27 of the first
# 33 conversations landed in the failures sidecar. `provider: {only: [openai]}` is refused by
# this account's guardrail and data policy, so there is no unfiltered GPT-5.4 route here.
#
# DeepSeek-V4-Pro answers all 36 on the same transcript, and is already the judge for this
# portfolio's XSTest and KidBench lanes. The cost is real: these scores are NOT comparable to
# the published VERA-MH numbers, nor to the earlier Kimi K3 line, which is why Kimi is
# recollected here rather than carried over.
export VERA_MH_JUDGE_MODEL="${VERA_MH_JUDGE_MODEL:-deepseek/deepseek-v4-pro}"
export VERA_MH_JUDGE_CONCURRENCY="${VERA_MH_JUDGE_CONCURRENCY:-32}"

export VERA_MH_USER_GPT52_BASE_URL="https://openrouter.ai/api/v1"
export VERA_MH_USER_GPT52_API_KEY="${OPENROUTER_API_KEY_SNORKEL}"
export VERA_MH_USER_GPT52_MODEL="openai/gpt-5.2"

export VERA_MH_USER_OPUS45_BASE_URL="https://openrouter.ai/api/v1"
export VERA_MH_USER_OPUS45_API_KEY="${OPENROUTER_API_KEY_SNORKEL}"
export VERA_MH_USER_OPUS45_MODEL="anthropic/claude-opus-4.5"
export VERA_MH_USER_CONCURRENCY="${VERA_MH_USER_CONCURRENCY:-48}"

EXPECTED_ROWS=200

# key | slug | base_url | model id | token env var | conversation concurrency
MODELS=(
"ultra|nemotron-3-ultra-550b|https://snorkelai-fdr--ep-nvidia-nemotron-3-ultra-550b-a55b-nvfp-63eebc.us-west.modal.direct/v1|nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4|MODAL_PROXY_TOKEN|24"
"qwen|qwen3.5-122b-a10b|https://snorkelai-fdr--ep-qwen3-5-122b-a10b-fp8-server.us-west.modal.direct/v1|Qwen/Qwen3.5-122B-A10B-FP8|MODAL_PROXY_TOKEN|24"
"supervl|nemotron-3.5-super-vl|https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct/v1|nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16|SUPER_VL_MODAL_TOKEN|12"
"kimi|kimi-k3|https://snorkelai-fdr--ep-kimi-k3-server.us-west.modal.direct/v1|moonshotai/Kimi-K3|MODAL_PROXY_TOKEN|12"
)

write_env_yaml() {
    local base_url="$1" model="$2" token_var="$3"
    cat > env.yaml <<YAML
# Generated by run_all_models.sh — do not edit by hand.
head_server:
  # host is not defaulted when the key is present, so both fields must be given together.
  host: 127.0.0.1
  port: ${GYM_HEAD_SERVER_PORT}

policy_model:
  responses_api_models:
    inference_provider:
      entrypoint: app.py
      base_url: ${base_url}
      api_key: \${oc.env:${token_var},""}
      model: ${model}
      uses_reasoning_parser: true
      num_concurrent_requests: 64
      # Matches the Kimi K3 line's per-turn budget; reasoning tokens count against it.
      responses_create_params:
        max_output_tokens: 8192
YAML
}

run_model() {
    local slug="$1" concurrency="$2"
    local out="${RESULTS_DIR}/${slug}.jsonl"
    local attempt status rows

    for attempt in 1 2 3; do
        echo "  [$(date +%H:%M:%S)] ${slug} attempt ${attempt} (concurrency ${concurrency}) -> ${out}"
        .venv/bin/gym eval run --resume \
            --benchmark vera_mh \
            --agent vera_mh_recommended \
            --split benchmark \
            --output "$out" \
            --concurrency "$concurrency" \
            > "${LOG_DIR}/${slug}.attempt${attempt}.log" 2>&1
        status=$?
        rows=$(wc -l < "$out" 2>/dev/null || echo 0)
        echo "  [$(date +%H:%M:%S)] ${slug} attempt ${attempt} exit=${status} rows=${rows}/${EXPECTED_ROWS}"

        if [ "$status" -eq 0 ] && [ "$rows" -ge "$EXPECTED_ROWS" ]; then
            echo "  [$(date +%H:%M:%S)] ${slug} complete"
            return 0
        fi

        concurrency=$(( concurrency / 2 ))
        [ "$concurrency" -lt 4 ] && concurrency=4
        echo "  [$(date +%H:%M:%S)] ${slug} backing off to concurrency ${concurrency}"
        stop_my_servers
        sleep 60
    done

    echo "  [$(date +%H:%M:%S)] ${slug} INCOMPLETE after 3 attempts (${rows}/${EXPECTED_ROWS})"
    return 1
}

WANTED=("$@")
for entry in "${MODELS[@]}"; do
    IFS='|' read -r key slug base_url model token_var model_concurrency <<< "$entry"
    if [ ${#WANTED[@]} -gt 0 ] && [[ ! " ${WANTED[*]} " =~ " ${key} " ]]; then
        continue
    fi

    echo "==============================================================="
    echo "[$(date +%H:%M:%S)] MODEL ${key} (${model})"
    echo "  simulators: ${VERA_MH_USER_GPT52_MODEL} + ${VERA_MH_USER_OPUS45_MODEL}   judge: ${VERA_MH_JUDGE_MODEL}"
    echo "==============================================================="

    write_env_yaml "$base_url" "$model" "$token_var"
    stop_my_servers
    run_model "$slug" "${CONCURRENCY_OVERRIDE:-$model_concurrency}"
done

stop_my_servers
echo "[$(date +%H:%M:%S)] All requested VERA-MH models finished."
