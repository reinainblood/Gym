# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight a generated HarmBench method dataset before Gym evaluates it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.harmbench.methods import get_method
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import sha256


ACTIVE = Path(__file__).resolve().parent / "data/active.jsonl"


def _preserve_previous_metrics(new_dataset_sha256: str) -> None:
    """Keep Gym's fixed-slot aggregate cache when changing generated methods."""
    previous_manifest = ACTIVE.with_suffix(".manifest.json")
    old_metrics = ACTIVE.with_name("active_metrics.json")
    old_conflict = ACTIVE.with_name("active_metrics_conflict.json")
    if not old_metrics.exists() and not old_conflict.exists():
        return
    if not previous_manifest.is_file():
        raise ValueError("active metrics exist without a source manifest; preserve them manually before switching")
    old_dataset_sha256 = json.loads(previous_manifest.read_text(encoding="utf-8"))["gym_inputs_sha256"]
    if old_dataset_sha256 == new_dataset_sha256:
        return
    if ACTIVE.is_file() and sha256(ACTIVE) != old_dataset_sha256:
        raise ValueError("active dataset disagrees with the manifest beside its cached metrics")
    history = ACTIVE.parent / "metrics-history"
    history.mkdir(parents=True, exist_ok=True)
    for source, suffix in ((old_metrics, "metrics"), (old_conflict, "conflict")):
        if not source.is_file():
            continue
        destination = history / f"{old_dataset_sha256}-{suffix}.json"
        if destination.exists():
            raise FileExistsError(f"metric history already exists; inspect before switching: {destination}")
        source.rename(destination)


def prepare() -> Path:
    value = os.environ.get("HARMBENCH_GENERATED_JSONL")
    if not value:
        raise ValueError("set HARMBENCH_GENERATED_JSONL to run_upstream_generation.py's gym-inputs.jsonl")
    dataset = Path(value).resolve(strict=True)
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    method = get_method(manifest["method"])
    receipt = manifest["generation_receipt"]
    if method.generation == "client_fresh":
        raise ValueError("fresh client-targeted attacks require live client binding validation")
    if manifest["gym_inputs_sha256"] != sha256(dataset):
        raise ValueError("generated HarmBench dataset hash disagrees with its manifest")
    if receipt["upstream_revision"] != UPSTREAM_REVISION or receipt["upstream_method"] != method.upstream_key:
        raise ValueError("generated HarmBench method provenance does not match pinned upstream")
    if manifest["test_cases_sha256"] != receipt["test_cases_sha256"]:
        raise ValueError("generated HarmBench attack-source digest disagrees with its receipt")
    count = 0
    behavior_ids: set[str] = set()
    copyright_ids: set[str] = set()
    with dataset.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["method"] != method.name or row["attack_source_sha256"] != receipt["test_cases_sha256"]:
                raise ValueError("generated HarmBench row mismatches its method receipt")
            behavior_ids.add(row["harmbench_id"])
            if "hash_check" in row.get("tags", []):
                copyright_ids.add(row["harmbench_id"])
            count += 1
    if not count or count != manifest["rows"] or len(behavior_ids) != manifest["behaviors"]:
        raise ValueError("generated HarmBench dataset counts disagree with its manifest")
    if copyright_ids:
        reference_value = os.environ.get("HARMBENCH_COPYRIGHT_HASHES_DIR")
        if not reference_value:
            raise ValueError("HARMBENCH_COPYRIGHT_HASHES_DIR is required for generated copyright behaviors")
        references = Path(reference_value).resolve(strict=True)
        upstream = references.parent.parent
        if references != upstream / "data/copyright_classifier_hashes":
            raise ValueError("copyright references must be in the pinned HarmBench checkout")
        head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if head != UPSTREAM_REVISION:
            raise ValueError("copyright reference checkout does not match the pinned upstream revision")
        subprocess.run(
            ["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--", "data/copyright_classifier_hashes"],
            check=True,
        )
        missing = sorted(
            behavior_id for behavior_id in copyright_ids if not (references / f"{behavior_id}.pkl").is_file()
        )
        if missing:
            raise FileNotFoundError(f"copyright references missing for {missing[:5]}")
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    if dataset != ACTIVE:
        # The original generated run remains in its method-specific output
        # directory. This fixed path is only Gym's active, opt-in overlay.
        _preserve_previous_metrics(manifest["gym_inputs_sha256"])
        shutil.copyfile(dataset, ACTIVE)
        shutil.copyfile(dataset.with_suffix(".manifest.json"), ACTIVE.with_suffix(".manifest.json"))
    print(f"Validated {count} {method.name} attack cases from {len(behavior_ids)} behaviors: {ACTIVE}")
    return ACTIVE


if __name__ == "__main__":
    prepare()
