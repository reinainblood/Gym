# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned HarmBench HumanJailbreaks/random_subset_5 benchmark.

Set HARMBENCH_UPSTREAM_DIR to a checkout of the pinned HarmBench commit and
HARMBENCH_COPYRIGHT_HASHES_DIR to its data/copyright_classifier_hashes folder.
The latter is required because the full 400-behavior text corpus includes
100 copyright cases, which use MinHash rather than the Llama classifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from benchmarks.harmbench.generate_human_jailbreaks import _templates, generate
from benchmarks.harmbench.prepare import DATA_DIR, UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import materialize


OUTPUT = DATA_DIR / "human_jailbreaks_random_subset_5.jsonl"
TEST_CASES = DATA_DIR / "human_jailbreaks_random_subset_5" / "test_cases.json"
TEMPLATE_SHA256 = "62e53558de1da3e67d1aec15c0dff59c848c0b374fd262ffad8f723fda0cd099"
BEHAVIORS_SHA256 = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"


def _require_upstream(upstream: Path) -> tuple[Path, Path]:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    templates = upstream / "baselines/human_jailbreaks/jailbreaks.py"
    behaviors = upstream / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    for path, expected in ((templates, TEMPLATE_SHA256), (behaviors, BEHAVIORS_SHA256)):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned HarmBench source changed: {path}")
    return templates, behaviors


def prepare(upstream_dir: str | None = None) -> Path:
    upstream_value = upstream_dir or os.environ.get("HARMBENCH_UPSTREAM_DIR")
    if not upstream_value:
        raise ValueError("set HARMBENCH_UPSTREAM_DIR to the pinned HarmBench checkout")
    upstream = Path(upstream_value).resolve(strict=True)
    template_path, behaviors_path = _require_upstream(upstream)
    references = upstream / "data/copyright_classifier_hashes"
    configured_references = os.environ.get("HARMBENCH_COPYRIGHT_HASHES_DIR")
    if not configured_references or Path(configured_references).resolve() != references.resolve():
        raise ValueError("set HARMBENCH_COPYRIGHT_HASHES_DIR to the pinned checkout's copyright_classifier_hashes")
    with behaviors_path.open(newline="", encoding="utf-8") as source:
        behaviors = list(csv.DictReader(source))
    if len(behaviors) != 400:
        raise ValueError(f"expected 400 full-corpus text behaviors, found {len(behaviors)}")
    for row in behaviors:
        if "hash_check" in row["Tags"].split(", ") and not (references / f"{row['BehaviorID']}.pkl").is_file():
            raise FileNotFoundError(f"missing copyright reference for {row['BehaviorID']}")

    cases = generate(behaviors, _templates(template_path))
    TEST_CASES.parent.mkdir(parents=True, exist_ok=True)
    TEST_CASES.write_text(json.dumps(cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "method": "HumanJailbreaks",
        "upstream_method": "HumanJailbreaks",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": "random_subset_5",
        "run_id": "seed-1",
        "test_cases_sha256": hashlib.sha256(TEST_CASES.read_bytes()).hexdigest(),
        "behaviors_sha256": BEHAVIORS_SHA256,
        "template_source_sha256": TEMPLATE_SHA256,
        "generator": "pinned HumanJailbreaks/random_subset_5 algorithm",
    }
    TEST_CASES.with_suffix(".generation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows, manifest = materialize(
        method_name="HumanJailbreaks",
        test_cases_path=TEST_CASES,
        behaviors_path=behaviors_path,
        experiment="random_subset_5",
        run_id="seed-1",
        target_type="text_api",
        generation_receipt=receipt,
    )
    if len(rows) != 2000:
        raise ValueError(f"expected 2,000 attack cases, found {len(rows)}")
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["gym_inputs_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} HumanJailbreaks cases across {len(behaviors)} behaviors at {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    prepare()
