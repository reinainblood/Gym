# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay copyright cases through pinned upstream spaCy/datasketch code.

Run this with a separate Python 3.10 environment containing the paper-era
spaCy 3.7.2, en_core_web_sm 3.7.1, and datasketch 1.5.9. It compares every
copyright label without printing protected text or model generations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import datasketch
import spacy


UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"


def calibrate(*, upstream: Path, rollouts: Path, output: Path) -> dict:
    if spacy.__version__ != "3.7.2" or datasketch.__version__ != "1.5.9":
        raise ValueError("upstream control requires spaCy 3.7.2 and datasketch 1.5.9")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    subprocess.run(
        ["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--", "data/copyright_classifier_hashes"],
        check=True,
    )
    source = upstream / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        behaviors = {row["BehaviorID"]: row for row in csv.DictReader(handle)}
    scored = [
        json.loads(line)
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if line and '"copyright_minhash"' in line
    ]
    if not scored:
        raise ValueError("no copyright-scored rollouts were found")
    sys.path.insert(0, str(upstream))
    import eval_utils  # type: ignore[import-not-found]  # noqa: PLC0415

    eval_utils.tqdm = lambda iterable: iterable  # suppress one progress bar per copyright case
    if eval_utils.nlp.meta.get("version") != "3.7.1":
        raise ValueError("upstream English tokenizer model is not en_core_web_sm 3.7.1")
    old_cwd = Path.cwd()
    mismatches = []
    try:
        os.chdir(upstream)  # original compute_results_hashing resolves references relative to cwd
        for row in scored:
            behavior_id = row["harmbench_id"]
            upstream_result = eval_utils.compute_results_hashing(
                behaviors[behavior_id], [{"generation": row["generation_for_classifier"]}]
            )[0]
            if upstream_result["label"] != row["score_label"]:
                mismatches.append(behavior_id)
    finally:
        os.chdir(old_cwd)
    receipt = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_code": "eval_utils.compute_results_hashing",
        "rollouts_sha256": hashlib.sha256(rollouts.read_bytes()).hexdigest(),
        "behaviors_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checked": len(scored),
        "agreement": len(scored) - len(mismatches),
        "mismatches": mismatches,
        "runtime": {
            "python": sys.version.split()[0],
            "spacy": spacy.__version__,
            "datasketch": datasketch.__version__,
            "english_model": eval_utils.nlp.meta["version"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Checked {len(scored)} copyright rollouts; {len(mismatches)} upstream scorer disagreements")
    if mismatches:
        raise ValueError(f"copyright scorer disagreed with upstream on {len(mismatches)} cases")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibrate(upstream=args.upstream, rollouts=args.rollouts, output=args.output)
