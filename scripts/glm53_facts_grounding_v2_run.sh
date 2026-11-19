#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_ID="bc4d9953-3971-49f7-96e6-85ec0057f150"
ROOT="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3661"
OUTPUT_DIR="$ROOT/results/glm53-safety/facts-grounding-v2/$RUN_ID"
GYM="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3588/.venv/bin/gym"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
set -a
source /Users/kruge/Documents/ChatGPT/NVIDIA/.env
set +a
: "${MODAL_PROXY_TOKEN:?MODAL_PROXY_TOKEN is required for the GLM-5.3 policy route}"
: "${PORTKEY_API_KEY:?PORTKEY_API_KEY is required for the refreshed grounding judge route}"
export MODAL_ENVIRONMENT="FDR"
export PYTHONWARNINGS="ignore"
export FACTS_GROUNDING_CAPTURE_DIR="$OUTPUT_DIR/model-calls"

exec "$GYM" eval run \
  --benchmark facts_grounding_v2 \
  --model-type inference_provider \
  --split benchmark \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-output-tokens 16384 \
  --output "$OUTPUT_DIR/rollouts.jsonl" \
  --concurrency 24 \
  --resume \
  +route_failures_to_sidecar=true \
  >>"$OUTPUT_DIR/full-run.log" 2>&1
