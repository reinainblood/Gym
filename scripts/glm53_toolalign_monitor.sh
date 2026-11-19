#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_ID="56d396eb-cc7c-4586-b389-954f882d2e7e"
WORKER_ID="codex-glm53-babysitter"
GYM_PID="${1:?gym process PID is required}"
ROOT="/Users/kruge/Documents/ChatGPT/NVIDIA/worktrees/pr-3595"
OUTPUT="$ROOT/results/glm53-safety/toolalignbench/$RUN_ID/rollouts.jsonl"
CONTROL_ROOM="https://glm53-safety-control-room-production.up.railway.app"
EXPECTED_ROWS=640
start_epoch="$(date +%s)"

heartbeat() {
  local rows failures now elapsed throughput eta token message payload
  rows="$(wc -l < "$OUTPUT" 2>/dev/null || printf '0')"
  failures="$(wc -l < "${OUTPUT%.jsonl}_failures.jsonl" 2>/dev/null || printf '0')"
  now="$(date +%s)"
  elapsed="$((now - start_epoch))"
  throughput="$(awk -v rows="$rows" -v elapsed="$elapsed" 'BEGIN { if (elapsed > 0) printf "%.6f", rows * 60 / elapsed; else print "0" }')"
  eta="$(awk -v expected="$EXPECTED_ROWS" -v rows="$rows" -v rate="$throughput" 'BEGIN { if (rate > 0) printf "%.0f", (expected - rows) * 60 / rate; else print "0" }')"
  message="ToolAlignBench full run active: ${rows}/${EXPECTED_ROWS} accounted, ${failures} sidecar failures, ${throughput} rows/min, measured ETA ${eta}s."
  payload="$(jq -n \
    --arg stage running \
    --argjson completed_rows "$rows" \
    --argjson failed_rows "$failures" \
    --argjson throughput_rows_per_minute "$throughput" \
    --arg message "$message" \
    --arg external_ref "local-process://$(hostname)/$GYM_PID" \
    '{stage:$stage,completed_rows:$completed_rows,failed_rows:$failed_rows,throughput_rows_per_minute:$throughput_rows_per_minute,message:$message,external_ref:$external_ref,lease_seconds:900}')"
  token="$(security find-generic-password -s 'GLM-5.3 Safety Control Room' -w)"
  curl -fsS \
    -H "Authorization: Bearer $token" \
    -H "X-Worker-ID: $WORKER_ID" \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "$CONTROL_ROOM/api/runs/$RUN_ID/heartbeat" \
    >/dev/null
}

heartbeat
while kill -0 "$GYM_PID" 2>/dev/null; do
  sleep 300
  heartbeat || true
done
