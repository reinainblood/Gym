#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_ID="2a252e00-0ce1-43d8-a7dc-e2ad0569b7e0"
ROOT="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3588"
OUTPUT_DIR="$ROOT/results/glm53-safety/facts-parametric/$RUN_ID"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
set -a
source /Users/kruge/Documents/ChatGPT/NVIDIA/.env
set +a
: "${PORTKEY_API_KEY:?PORTKEY_API_KEY is required for the Gemini 2.5 Pro judge route}"
export PYTHONWARNINGS="ignore"

exec .venv/bin/gym eval run \
  --benchmark facts_parametric \
  --model-type inference_provider \
  --split benchmark \
  --output "$OUTPUT_DIR/rollouts.jsonl" \
  --concurrency 28 \
  --resume \
  +route_failures_to_sidecar=true \
  >"$OUTPUT_DIR/full-run.log" 2>&1
