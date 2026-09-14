# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the Gym FACTS Grounding v2 verifier against the official starter implementation.

``replay``
    Re-parses every eligibility and grounding receipt from a Gym run with the starter's verbatim parsers
    (``extract_instruction_json``, ``parse_ufg_rev21_verdict``), recomputes the starter's eligibility rule
    ("Major Issue(s)" not in rating, first eligible judge wins) and grounding score (mean of judge booleans, 0 when
    ineligible), and compares with the Gym decisions case by case. This isolates parsing and aggregation from judge
    sampling and must agree exactly.

``live``
    Re-runs the starter's judging loop (judge baseline answer, eligibility prompt, grounding prompt, same judge
    order and provider defaults) through the judge endpoints for a stratified subset of the Gym answers and compares
    eligibility, per-judge verdicts and scores. The judges sample, so this measures verdict stability.

Endpoint settings for ``live`` come from the environment variables the Gym judge model servers read
(``FACTS_GROUNDING_JUDGE_GEMINI_*`` and ``FACTS_GROUNDING_JUDGE_GPT5_*``).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.facts_grounding_v2.upstream_control import extract_instruction_json, parse_ufg_rev21_verdict
from resources_servers.facts_grounding_v2.app import load_prompts


JUDGES = [("gemini-2.5-flash", "FACTS_GROUNDING_JUDGE_GEMINI"), ("gpt-5", "FACTS_GROUNDING_JUDGE_GPT5")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def starter_decisions(
    receipts_by_stage: dict[tuple[str, str], str], judges: list[str], *, grade_ineligible: bool
) -> dict[str, Any]:
    """Apply the starter's rules to raw judge texts keyed by (stage, judge)."""
    eligible = False
    deciding = None
    ratings = {}
    for judge in judges:
        text = receipts_by_stage.get(("eligibility", judge))
        if text is None:
            break
        rating = extract_instruction_json(text)["Instruction Following"]
        ratings[judge] = rating
        if "Major Issue(s)" not in rating:
            eligible = True
            deciding = judge
            break
    verdicts = {}
    if eligible or grade_ineligible:
        for judge in judges:
            text = receipts_by_stage.get(("grounding", judge))
            if text is not None:
                verdicts[judge], _ = parse_ufg_rev21_verdict(text, judge)
    grounding = (sum(verdicts.values()) / len(verdicts)) if verdicts else None
    return {
        "eligible": eligible,
        "deciding": deciding,
        "ratings": ratings,
        "verdicts": verdicts,
        "score": (grounding if eligible and grounding is not None else 0.0),
        "unadjusted": grounding,
    }


def replay(rollouts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    agreements = 0
    for row in rollouts:
        by_stage = {(r["stage"], r["judge"]): (r.get("content") or "") for r in row.get("judge_receipts", [])}
        judges = [label for label, _ in JUDGES]
        starter = starter_decisions(by_stage, judges, grade_ineligible=True)
        agree = (
            starter["eligible"] == bool(row.get("eligible"))
            and starter["deciding"] == row.get("eligibility_deciding_judge")
            and starter["ratings"] == row.get("eligibility_ratings")
            and starter["verdicts"] == {k: bool(v) for k, v in (row.get("grounding_verdicts") or {}).items()}
            and abs(float(starter["score"]) - float(row.get("reward", 0.0))) < 1e-9
        )
        agreements += agree
        cases.append(
            {
                "case_id": f"{row.get('id')}/replay",
                "control": "replay",
                "task_id": row.get("id"),
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream": starter,
                "gym": {
                    "eligible": row.get("eligible"),
                    "deciding": row.get("eligibility_deciding_judge"),
                    "ratings": row.get("eligibility_ratings"),
                    "verdicts": row.get("grounding_verdicts"),
                    "score": row.get("reward"),
                },
                "agree": agree,
            }
        )
    return cases, {
        "control": "replay",
        "cases": len(cases),
        "agreement": agreements,
        "disagreements": len(cases) - agreements,
    }


class Judge:
    def __init__(self, label: str, env_prefix: str) -> None:
        self.label = label
        self.base_url = os.environ[f"{env_prefix}_BASE_URL"].rstrip("/")
        self.api_key = os.environ[f"{env_prefix}_API_KEY"]
        self.model = os.environ[f"{env_prefix}_MODEL"]

    def prompt(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "user-agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"].get("content") or ""
        return {
            "model": payload.get("model"),
            "response_id": payload.get("id"),
            "content": content,
            "usage": payload.get("usage"),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }


# Some gateways sit behind a WAF that rejects urllib's default user agent (Cloudflare error 1010).
USER_AGENT = "nemo-gym-facts-grounding-v2-calibrate/1.0"
LIVE_WORKERS = int(os.environ.get("FACTS_GROUNDING_JUDGE_CONCURRENCY", "6"))


def stratified_subset(rollouts: list[dict[str, Any]], size: int, seed: int = 20251211) -> list[dict[str, Any]]:
    def stratum(row):
        if not row.get("eligible"):
            return "ineligible"
        if row.get("reward") == 1.0:
            return "grounded"
        if row.get("reward") == 0.5:
            return "split"
        return "unsupported"

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rollouts:
        groups[stratum(row)].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    per_group = max(1, size // max(1, len(groups)))
    for name in sorted(groups):
        pool = sorted(groups[name], key=lambda row: str(row.get("id")))
        rng.shuffle(pool)
        picked.extend(pool[:per_group])
    remaining = [row for row in rollouts if row not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, size - len(picked))])
    return picked[:size]


def live(rollouts: list[dict[str, Any]], size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = load_prompts()
    judges = [Judge(label, prefix) for label, prefix in JUDGES]
    cases = []
    agree_eligible = agree_score = 0
    verdict_agree = verdict_total = 0
    subset = stratified_subset(rollouts, size)

    def rejudge(row: dict[str, Any]) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
        receipts: dict[tuple[str, str], str] = {}
        raw: list[dict[str, Any]] = []
        response = row.get("generation", "")
        for judge in judges:
            baseline = judge.prompt([{"role": "user", "content": row["full_prompt"]}])
            raw.append({"stage": "baseline", "judge": judge.label, **baseline})
            prompt = prompts["eligibility_prompt"].format(
                user_request=row["user_request"], response_a=response, response_b=baseline["content"]
            )
            rating = judge.prompt(
                [
                    {"role": "system", "content": prompts["eligibility_system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
            )
            raw.append({"stage": "eligibility", "judge": judge.label, **rating})
            receipts[("eligibility", judge.label)] = rating["content"]
            if "Major Issue(s)" not in extract_instruction_json(rating["content"])["Instruction Following"]:
                break
        for judge in judges:
            prompt = prompts["grounding_prompt"].format(
                user_query=row["user_request"], context=row["context_document"], response=response
            )
            grounding = judge.prompt(
                [
                    {"role": "system", "content": prompts["grounding_system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
            )
            raw.append({"stage": "grounding", "judge": judge.label, **grounding})
            receipts[("grounding", judge.label)] = grounding["content"]
        return receipts, raw

    with ThreadPoolExecutor(max_workers=LIVE_WORKERS) as pool:
        rejudged = list(pool.map(rejudge, subset))
    for row, (receipts, raw) in zip(subset, rejudged):
        starter = starter_decisions(receipts, [j.label for j in judges], grade_ineligible=True)
        gym_verdicts = {k: bool(v) for k, v in (row.get("grounding_verdicts") or {}).items()}
        agree_eligible += starter["eligible"] == bool(row.get("eligible"))
        agree_score += abs(float(starter["score"]) - float(row.get("reward", 0.0))) < 1e-9
        for judge in gym_verdicts:
            if judge in starter["verdicts"]:
                verdict_total += 1
                verdict_agree += starter["verdicts"][judge] == gym_verdicts[judge]
        cases.append(
            {
                "case_id": f"{row.get('id')}/live",
                "control": "live",
                "task_id": row.get("id"),
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream": starter,
                "upstream_receipts": raw,
                "gym": {
                    "eligible": row.get("eligible"),
                    "ratings": row.get("eligibility_ratings"),
                    "verdicts": row.get("grounding_verdicts"),
                    "score": row.get("reward"),
                },
                "agree": abs(float(starter["score"]) - float(row.get("reward", 0.0))) < 1e-9,
            }
        )
    return cases, {
        "control": "live",
        "cases": len(cases),
        "agreement": agree_score,
        "disagreements": len(cases) - agree_score,
        "eligibility_agreement": agree_eligible,
        "verdict_agreement_rate": verdict_agree / verdict_total if verdict_total else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live-size", type=int, default=0)
    args = parser.parse_args(argv)
    rollouts = _read_jsonl(args.rollouts)
    args.out.mkdir(parents=True, exist_ok=True)
    cases, summary = replay(rollouts)
    summaries = [summary]
    if args.live_size:
        live_cases, live_summary = live(rollouts, args.live_size)
        cases += live_cases
        summaries.append(live_summary)
    with (args.out / "upstream-vs-gym.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    normalized = {
        "method": "replay of every judge receipt through the starter's verbatim parsers and eligibility/score rules"
        + ("; plus a live stratified re-judge with the starter loop" if args.live_size else ""),
        "cases": sum(s["cases"] for s in summaries),
        "agreement": sum(s["agreement"] for s in summaries),
        "disagreements": sum(s["disagreements"] for s in summaries),
        "notes": [
            f"{s['control']}: {s['agreement']} of {s['cases']} cases agree"
            + (
                f"; eligibility agreement {s['eligibility_agreement']} of {s['cases']}; per-judge verdict agreement {s['verdict_agreement_rate']:.1%}"
                if s["control"] == "live" and s.get("verdict_agreement_rate") is not None
                else ""
            )
            for s in summaries
        ],
        "details": {
            "controls": summaries,
            "case_ids": [c["case_id"] for c in cases if not c["agree"]][:20] or [c["case_id"] for c in cases[:3]],
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps({"controls": summaries, "normalized": normalized}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
