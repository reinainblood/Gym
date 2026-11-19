# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume-safe target generation for a sharded, client-fresh HarmBench TAP run."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import aiohttp


def load_rows(test_cases_dir: Path, behaviors_dir: Path) -> list[dict[str, object]]:
    behaviors: dict[str, dict[str, str]] = {}
    for path in sorted(behaviors_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                behavior_id = row["BehaviorID"]
                if behavior_id in behaviors:
                    raise ValueError(f"duplicate behavior metadata: {behavior_id}")
                behaviors[behavior_id] = row

    cases: dict[str, str] = {}
    for path in sorted(test_cases_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        for behavior_id, prompts in payload.items():
            if behavior_id in cases:
                raise ValueError(f"duplicate generated test case: {behavior_id}")
            if not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str):
                raise ValueError(f"unexpected test-case shape for {behavior_id}")
            cases[behavior_id] = prompts[0]

    if len(cases) != 400 or set(cases) != set(behaviors):
        raise ValueError(f"expected 400 matched rows, got cases={len(cases)} metadata={len(behaviors)}")
    return [{"behavior_id": key, "test_case": cases[key], "metadata": behaviors[key]} for key in sorted(cases)]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["behavior_id"] for line in path.open() if line.strip()}


async def generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, object],
    *,
    url: str,
    model: str,
    api_key: str,
) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": row["test_case"]}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    last_status: int | None = None
    started = time.monotonic()
    for attempt in range(1, 4):
        try:
            async with semaphore:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as response:
                    last_status = response.status
                    if response.status != 200:
                        await response.read()
                        if response.status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < 3:
                            await asyncio.sleep(2**attempt)
                            continue
                        return {
                            "behavior_id": row["behavior_id"],
                            "failure": "http_error",
                            "http_status": response.status,
                            "attempts": attempt,
                        }
                    payload = await response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            return {
                "behavior_id": row["behavior_id"],
                "metadata": row["metadata"],
                "test_case": row["test_case"],
                "generation": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"),
                "response_id": payload.get("id"),
                "model": payload.get("model", model),
                "http_status": 200,
                "attempts": attempt,
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < 3:
                await asyncio.sleep(2**attempt)
                continue
            return {
                "behavior_id": row["behavior_id"],
                "failure": "transport_error",
                "http_status": last_status,
                "attempts": attempt,
            }
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_cases_dir, args.behaviors_dir)
    done = completed_ids(args.output)
    pending = [row for row in rows if row["behavior_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, limit_per_host=args.concurrency)
    headers = {"Content-Type": "application/json"}
    counts = Counter()
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [
            asyncio.create_task(
                generate_one(
                    session,
                    semaphore,
                    row,
                    url=args.url,
                    model=args.model,
                    api_key=os.environ[args.api_key_env],
                )
            )
            for row in pending
        ]
        with args.output.open("a") as output, args.failures.open("a") as failures:
            for index, task in enumerate(asyncio.as_completed(tasks), 1):
                result = await task
                if result.get("failure"):
                    failures.write(json.dumps(result, separators=(",", ":")) + "\n")
                    failures.flush()
                    counts[str(result["failure"])] += 1
                else:
                    output.write(json.dumps(result, separators=(",", ":")) + "\n")
                    output.flush()
                    counts["success"] += 1
                if index % 25 == 0 or index == len(tasks):
                    print(
                        json.dumps(
                            {
                                "attempted_this_run": index,
                                "remaining_this_run": len(tasks) - index,
                                "success_this_run": counts["success"],
                                "failures_this_run": index - counts["success"],
                            }
                        ),
                        flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases-dir", type=Path, required=True)
    parser.add_argument("--behaviors-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODAL_PROXY_TOKEN")
    parser.add_argument("--concurrency", type=int, default=54)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
