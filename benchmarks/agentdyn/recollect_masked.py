# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drop infrastructure-masked rollouts from a finished cell so `--resume` re-collects them.

A masked row still occupies its slot in the 620, so a cell can reach its row target while
being scored on fewer. Masks caused by the serving stack -- a 500 from the model server, a
connection reset -- are not results: the request never reached the model, and the right
remedy is to collect the row again rather than to report a smaller denominator.

Masks that *are* results stay. A `RolloutTimeout` means the defense and model between them
could not finish the task inside the budget, which is an outcome of the treatment rather
than of the infrastructure; deleting it would quietly convert a real failure into a re-roll
until it passed.

    python benchmarks/agentdyn/recollect_masked.py results/agentdyn/supervl-tool_filter.jsonl
    python benchmarks/agentdyn/recollect_masked.py <file> --apply

Dry run by default. Run it only when no container holds the cell, and re-upload the
rewritten file before relaunching.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


#: Masks worth collecting again: the request failed below the model, so there is no result.
INFRASTRUCTURE_PREFIXES = (
    "ClientResponseError",
    "ClientOSError",
    "ClientConnectorError",
    "ServerDisconnectedError",
    "TimeoutError",
)
#: Masks that are results in their own right and must survive.
RESULT_PREFIXES = ("RolloutTimeout",)


def classify(row: dict) -> str:
    if not row.get("mask_sample"):
        return "scored"
    error = row.get("adapter_error") or ""
    if error.startswith(RESULT_PREFIXES):
        return "masked-result"
    if error.startswith(INFRASTRUCTURE_PREFIXES):
        return "masked-infrastructure"
    return "masked-other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--apply", action="store_true", help="Rewrite the file; otherwise report only.")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.rollouts.open(encoding="utf-8") if line.strip()]
    counts = collections.Counter(classify(row) for row in rows)
    keep = [row for row in rows if classify(row) != "masked-infrastructure"]
    drop = len(rows) - len(keep)

    print(f"  {args.rollouts}")
    for name in ("scored", "masked-result", "masked-infrastructure", "masked-other"):
        if counts[name]:
            print(f"    {name:24} {counts[name]:4}")
    print(f"    -> would re-collect {drop}, leaving {len(keep)} rows for resume to top up")
    if counts["masked-other"]:
        print("    masked-other is left in place: an unrecognised cause is not safe to assume transient")

    if not args.apply:
        print("    dry run; pass --apply to rewrite")
        return 0
    if not drop:
        print("    nothing to do")
        return 0
    args.rollouts.write_text("".join(json.dumps(row) + "\n" for row in keep), encoding="utf-8")
    print(f"    rewrote {args.rollouts} with {len(keep)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
