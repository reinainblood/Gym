# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-grade KORA's own published conversations through this adapter's verifier.

The run package carries every graded conversation of the published leaderboard
(``messages.parquet``) beside the grade each received (``results.parquet``). Feeding a
sample of those transcripts to a running ``kora`` resources server, with no policy or
child model in the loop, isolates the judge: if the grades agree, later differences on
fresh rollouts are the conversation, not the grader. This is the adapter's counterpart to
upstream's ``kora reassess`` command.

Start the server (judge only; the policy and child are not needed)::

    gym env start --config resources_servers/kora/configs/kora.yaml \\
        +kora_judge_base_url=https://api.openai.com/v1 +kora_judge_api_key=$OPENAI_API_KEY

then::

    python -m benchmarks.kora.regrade --verify-url http://localhost:<port>/verify \\
        --targets gpt-5-nano-high,claude-haiku-4.5-high --per-target 50 \\
        --output results/kora_regrade.jsonl

The report prints exact grade agreement, the confusion matrix, and the pooled score of
the published grades against ours on the same conversations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import aiohttp

from benchmarks.kora.prepare import DEFAULT_UPSTREAM_DIR, iter_rows, resolve_package
from benchmarks.kora.upstream_spec import GRADES, TIERS


logger = logging.getLogger(__name__)


def _read_table(package_dir: Path, name: str, columns: Optional[list[str]] = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(package_dir / f"{name}.parquet", columns=columns).to_pylist()


def _response_of(assistant_messages: list[str]) -> dict[str, Any]:
    """A minimal Responses object carrying the published assistant turns."""
    return {
        "created_at": 0,
        "id": "resp_regrade",
        "model": "published",
        "object": "response",
        "output": [
            {
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "id": f"msg_{index}",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
            for index, text in enumerate(assistant_messages)
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def sample_results(
    package_dir: Path, *, targets: Optional[list[str]], per_target: int, seed: int
) -> list[dict[str, Any]]:
    results = _read_table(package_dir, "results", ["result_id", "target_id", "tier", "scenario_id", "grade"])
    if targets:
        results = [row for row in results if row["target_id"] in targets]
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for target_id in sorted({row["target_id"] for row in results}):
        for tier in TIERS:
            pool = [row for row in results if row["target_id"] == target_id and row["tier"] == tier]
            rng.shuffle(pool)
            chosen.extend(pool[: per_target // len(TIERS)])
    return chosen


def build_requests(package_dir: Path, sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = {row["kora_id"]: row for row in iter_rows(package_dir)}
    wanted = {row["result_id"] for row in sample}
    messages: dict[str, list[dict[str, Any]]] = {}
    for message in _read_table(package_dir, "messages"):
        if message["result_id"] in wanted:
            messages.setdefault(message["result_id"], []).append(message)
    requests = []
    for result in sample:
        transcript = sorted(messages[result["result_id"]], key=lambda m: m["message_index"])
        conversation = [{"role": m["role"], "content": m["content"]} for m in transcript]
        row = rows[f"{result['scenario_id']}:{result['tier']}"]
        requests.append(
            row
            | {
                "result_id": result["result_id"],
                "target_id": result["target_id"],
                "published_grade": result["grade"],
                "conversation": conversation,
                "child_turns": [m["content"] for m in conversation if m["role"] == "user"],
                "turns_completed": sum(1 for m in conversation if m["role"] == "user"),
                "response": _response_of([m["content"] for m in conversation if m["role"] == "assistant"]),
            }
        )
    return requests


async def _verify_all(verify_url: str, requests: list[dict[str, Any]], *, concurrency: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)

    async def one(session: aiohttp.ClientSession, payload: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            async with session.post(verify_url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(*(one(session, payload) for payload in requests))


def _pooled(grades: list[str]) -> Optional[float]:
    scored = [g for g in grades if g in GRADES]
    if not scored:
        return None
    return sum({"failing": 0, "adequate": 1, "exemplary": 2}[g] for g in scored) / (2 * len(scored)) * 100


def report(results: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in results if row.get("grade") in GRADES]
    agreement = sum(1 for row in graded if row["grade"] == row["published_grade"]) / len(graded) if graded else None
    confusion = Counter((row["published_grade"], row["grade"]) for row in graded)
    per_target: dict[str, dict[str, Any]] = {}
    for target_id in sorted({row["target_id"] for row in results}):
        rows = [row for row in results if row["target_id"] == target_id]
        per_target[target_id] = {
            "n": len(rows),
            "published_score_pct": _pooled([row["published_grade"] for row in rows]),
            "regraded_score_pct": _pooled([row.get("grade") for row in rows]),
        }
    return {
        "n": len(results),
        "n_graded": len(graded),
        "exact_agreement": agreement,
        "confusion_published_vs_regraded": {f"{a}->{b}": count for (a, b), count in sorted(confusion.items())},
        "per_target": per_target,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-url", required=True, help="The running kora resources server's /verify URL.")
    parser.add_argument("--targets", default=None, help="Comma-separated target_ids from the package; default all 41.")
    parser.add_argument("--per-target", type=int, default=20, help="Conversations per target, split across tiers.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSONL of verifier responses with the published grade."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    package_dir = resolve_package(args.upstream_dir)
    targets = args.targets.split(",") if args.targets else None
    sample = sample_results(package_dir, targets=targets, per_target=args.per_target, seed=args.seed)
    logger.info("Re-grading %d published conversations", len(sample))
    requests = build_requests(package_dir, sample)
    results = asyncio.run(_verify_all(args.verify_url, requests, concurrency=args.concurrency))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = report(results)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
