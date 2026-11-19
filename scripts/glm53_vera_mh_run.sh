#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_ID="${RUN_ID:-f10962f9-be94-49f0-ba82-a80482adb0a5}"
ROOT="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3620"
OUTPUT_DIR="$ROOT/results/glm53-safety/vera-mh/$RUN_ID"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
set -a
source /Users/kruge/Documents/ChatGPT/NVIDIA/.env
set +a
: "${PORTKEY_API_KEY:?PORTKEY_API_KEY is required for the VERA-MH hosted routes}"
# The inherited OpenRouter attempts occupy indices 0 and 1. A ceiling of 3
# permits exactly one Portkey recovery attempt without reopening completed rows.
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-3}"
# Pydantic serializer warnings include snippets of the simulated crisis transcript.
# Keep those sensitive prompt fragments out of operational logs.
export PYTHONWARNINGS="ignore"

exec .venv/bin/gym eval run \
  --benchmark vera_mh \
  --model-type inference_provider \
  --split benchmark \
  --output "$OUTPUT_DIR/rollouts.jsonl" \
  --concurrency 12 \
  --resume \
  +route_failures_to_sidecar=true \
  >"$OUTPUT_DIR/full-run.log" 2>&1
