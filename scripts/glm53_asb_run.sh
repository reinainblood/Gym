#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_ID="2d5991e8-482f-4ee8-95fe-d6e1d40c28f8"
ROOT="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3567"
OUTPUT_DIR="$ROOT/results/glm53-safety/asb/$RUN_ID"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
set -a
source /Users/kruge/Documents/ChatGPT/NVIDIA/.env
set +a
export OPENROUTER_API_KEY_SNORKEL="$(security find-generic-password -s 'OPENROUTER_API_KEY' -w)"

exec .venv/bin/gym eval run \
  --benchmark asb \
  --model-type inference_provider \
  --split benchmark \
  --output "$OUTPUT_DIR/rollouts.jsonl" \
  --concurrency 32 \
  --resume \
  +route_failures_to_sidecar=true \
  >"$OUTPUT_DIR/full-run.log" 2>&1
