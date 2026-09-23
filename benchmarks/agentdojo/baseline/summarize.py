# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a directory of AgentDojo baseline shard files, and drop infrastructure masks.

    python -m benchmarks.agentdojo.baseline.summarize table <dir>
    python -m benchmarks.agentdojo.baseline.summarize recollect <shard.jsonl> [--apply]

Only files named exactly `<model>-<arm>-shard<i>of<n>.jsonl` are read. Gym writes
`_materialized_inputs.jsonl` and `_failures.jsonl` sidecars next to every rollout file, and a
loose glob picks them up as rollouts. A cell is complete only when every one of its N shards
exists and the union covers each of the 1,046 selectors exactly once.

Masked rows are excluded from every rate. `recollect` removes rows masked for infrastructure
reasons (the request failed below the model) so that `gym eval run --resume` collects them
again; `RolloutTimeout` and unrecognised causes stay, since they may be real outcomes.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


TOTAL_ROWS = 1046
SHARD_FILE = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<arm>[a-z_]+)-shard(?P<index>\d+)of(?P<shards>\d+)\.jsonl$")
#: Completion of a cell whose shards predate the slack-numbering fix; see baseline/shards.py.
SUPPLEMENT_FILE = re.compile(r"^(?P<model>[a-z0-9]+)-(?P<arm>[a-z_]+)-slack5\.jsonl$")


def canonical_selectors() -> set[tuple[str, str, str | None]]:
    """The selector set of the fixed prepare(): what every cell is scored against."""
    from benchmarks.agentdojo.prepare import prepare

    return {selector(json.loads(line)) for line in prepare().read_text(encoding="utf-8").splitlines()}


INFRASTRUCTURE_PREFIXES = (
    "ClientResponseError",
    "ClientOSError",
    "ClientConnectorError",
    "ClientPayloadError",
    "ServerDisconnectedError",
    "TimeoutError",
    "APIConnectionError",
    "InternalServerError",
)
RESULT_PREFIXES = ("RolloutTimeout",)


def mask_cause(row: dict) -> str | None:
    if not row.get("mask_sample"):
        return None
    error = row.get("adapter_error") or row.get("failure_reason") or "unknown"
    return error.split(":", 1)[0].strip() or "unknown"


def classify(row: dict) -> str:
    cause = mask_cause(row)
    if cause is None:
        return "scored"
    if cause.startswith(RESULT_PREFIXES):
        return "masked-result"
    if cause.startswith(INFRASTRUCTURE_PREFIXES):
        return "masked-infrastructure"
    return "masked-other"


def selector(row: dict) -> tuple[str, str, str | None]:
    return (row["suite"], row["user_task_id"], row.get("injection_task_id"))


def load_cells(directory: Path) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict[int, dict[int, Path]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    supplements: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.iterdir()):
        match = SHARD_FILE.match(path.name)
        if match:
            key = (match["model"], match["arm"])
            groups[key][int(match["shards"])][int(match["index"])] = path
        match = SUPPLEMENT_FILE.match(path.name)
        if match:
            supplements[(match["model"], match["arm"])] = path
    canonical = canonical_selectors()
    cells = {}
    for key, by_count in groups.items():
        if len(by_count) != 1:
            raise SystemExit(f"{key}: files from more than one shard count {sorted(by_count)}; resolve before scoring")
        ((shards, files),) = by_count.items()
        paths = [files[index] for index in sorted(files)] + ([supplements[key]] if key in supplements else [])
        rows = []
        for path in paths:
            rows.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        kept = [row for row in rows if selector(row) in canonical]
        cells[key] = {
            "shards": shards,
            "present": sorted(files),
            "supplement": key in supplements,
            "rows": kept,
            "dropped_noncanonical": len(rows) - len(kept),
        }
    return cells


def summarize(cell: dict) -> dict:
    rows = cell["rows"]
    counts = collections.Counter(selector(row) for row in rows)
    scored = [row for row in rows if not row.get("mask_sample")]
    benign = [row for row in scored if row.get("injection_task_id") is None]
    attacked = [row for row in scored if row.get("injection_task_id") is not None]
    temperatures = collections.Counter(json.dumps((row.get("response") or {}).get("temperature")) for row in rows)
    rate = lambda part, key: sum(bool(row.get(key)) for row in part) / len(part) if part else float("nan")
    return {
        "shards": f"{len(cell['present'])}/{cell['shards']}" + ("+s" if cell["supplement"] else ""),
        "dropped_noncanonical": cell["dropped_noncanonical"],
        "rows": len(rows),
        "unique": len(counts),
        "duplicates": sum(n - 1 for n in counts.values() if n > 1),
        "scored": len(scored),
        "scored_benign": len(benign),
        "scored_attacked": len(attacked),
        "benign_utility": rate(benign, "utility"),
        "utility_under_attack": rate(attacked, "utility"),
        "attack_success_rate": rate(attacked, "attack_success"),
        "masked": dict(collections.Counter(mask_cause(row) for row in rows if row.get("mask_sample"))),
        "temperatures": dict(temperatures),
        "complete": len(counts) == TOTAL_ROWS and len(cell["present"]) == cell["shards"],
    }


def table(directory: Path, as_json: bool) -> None:
    cells = load_cells(directory)
    out = {f"{model}-{arm}": summarize(cell) for (model, arm), cell in sorted(cells.items())}
    if as_json:
        print(json.dumps(out, indent=2))
        return
    header = f"{'cell':34} {'shards':>8} {'drop':>4} {'rows':>5} {'scored':>6} {'BU':>6} {'UuA':>6} {'ASR':>6}  temp  masked"
    print(header)
    for name, s in out.items():
        print(
            f"{name:34} {s['shards']:>8} {s['dropped_noncanonical']:>4} {s['unique']:>5} {s['scored']:>6} "
            f"{s['benign_utility']:6.3f} {s['utility_under_attack']:6.3f} {s['attack_success_rate']:6.3f}  "
            f"{','.join(s['temperatures'])}  {s['masked'] or ''}{'' if s['complete'] else '  INCOMPLETE'}"
            f"{'  DUPES ' + str(s['duplicates']) if s['duplicates'] else ''}"
        )


def recollect(path: Path, apply: bool, timeouts: bool = False) -> None:
    """`timeouts` also drops `RolloutTimeout` rows, for a re-run under a larger budget.

    Only for a timeout the harness imposed on a healthy rollout: upstream sets no time limit,
    and its task runner reruns a pipeline up to 3x (up to 48 model calls), so a slow endpoint
    can push a healthy rollout past a budget calibrated on fast ones.
    """
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    counts = collections.Counter(classify(row) for row in rows)
    drop = {"masked-infrastructure"} | ({"masked-result"} if timeouts else set())
    keep = [row for row in rows if classify(row) not in drop]
    print(f"{path.name}: {dict(counts)} -> drop {len(rows) - len(keep)}, keep {len(keep)}")
    if apply and len(keep) != len(rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in keep), encoding="utf-8")
        print(f"  rewrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table")
    t.add_argument("directory", type=Path)
    t.add_argument("--json", action="store_true")
    r = sub.add_parser("recollect")
    r.add_argument("paths", type=Path, nargs="+")
    r.add_argument("--apply", action="store_true")
    r.add_argument("--timeouts", action="store_true", help="Also drop RolloutTimeout rows.")
    args = parser.parse_args()
    if args.command == "table":
        table(args.directory, args.json)
    else:
        for path in args.paths:
            recollect(path, args.apply, args.timeouts)


if __name__ == "__main__":
    main()
