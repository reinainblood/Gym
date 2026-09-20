# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run original public HarmBench multimodal PGD classes through native K3."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import yaml
from kimi_k3_harmbench_text_runner import (
    install_kimi_transformers_compatibility,
    prepare_loader_view,
)
from kimi_k3_multimodal_adapter import KimiK3HarmBenchModel


UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
METHODS = {
    "MultiModalPGD": (
        "MultiModalPGD",
        "MultiModalPGD_config.yaml",
        "5056a2c2fae1d41f74c15b4d47d2920e586025e012cdb6cc84602b2e52e1d8d4",
    ),
    "MultiModalPGDPatch": (
        "MultiModalPGDPatch",
        "MultiModalPGDPatch_config.yaml",
        "ae8e45a178c72e81643911eb1fed3b6a38a932292a0fa3d1c034885f5974426a",
    ),
}


def run(
    *,
    upstream: Path,
    snapshot: Path,
    output_root: Path,
    method_name: str,
    commit: Callable[[], None],
) -> dict[str, object]:
    if method_name not in METHODS:
        raise ValueError(f"unknown multimodal method: {method_name}")
    revision = (upstream / ".pinned-revision").read_text(encoding="utf-8").strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision is {revision}, expected {UPSTREAM_REVISION}")
    class_name, config_name, config_hash = METHODS[method_name]
    config_path = upstream / "configs/method_configs" / config_name
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_hash:
        raise ValueError(f"public config hash mismatch for {method_name}")
    method_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["default_method_hyperparameters"]
    method_config.update(
        {
            "target_model": {"model_name_or_path": str(snapshot)},
            "test_cases_batch_size": 1,
            "num_test_cases_per_behavior": 1,
        }
    )

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    install_kimi_transformers_compatibility()
    from baselines import get_method_class
    from baselines.multimodalpgd import multimodalpgd as public_module

    native_model = KimiK3HarmBenchModel(prepare_loader_view(snapshot))
    public_module.load_multimodal_model = lambda _model_name_or_path: native_model
    method_class = get_method_class(class_name)
    method = method_class(**method_config)
    output_dir = output_root / method_name / "default"
    output_dir.mkdir(parents=True, exist_ok=True)
    behaviors_path = upstream / "data/behavior_datasets/harmbench_behaviors_multimodal_all.csv"
    with behaviors_path.open(newline="", encoding="utf-8") as stream:
        behaviors = list(csv.DictReader(stream))
    started = time.time()
    try:
        for index, behavior in enumerate(behaviors):
            behavior_id = behavior["BehaviorID"]
            existing = Path(method_class.get_output_file_path(str(output_dir), behavior_id, "test_cases"))
            if existing.exists():
                continue
            try:
                test_cases, logs = method.generate_test_cases(behaviors=[behavior], verbose=True)
                method.save_test_cases(str(output_dir), test_cases, logs, method_config=method_config)
                commit()
            except Exception as exc:
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "method": method_name,
                    "behavior_id": behavior_id,
                    "behavior_index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                path = output_dir / "failures" / f"{behavior_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                commit()
                raise
    finally:
        progress = {
            "schema_version": 1,
            "status": "running_or_complete",
            "method": method_name,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "upstream_revision": revision,
            "public_config_sha256": config_hash,
            "expected_behaviors": len(behaviors),
            "present_outputs": sum(
                Path(method_class.get_output_file_path(str(output_dir), b["BehaviorID"], "test_cases")).exists()
                for b in behaviors
            ),
            "elapsed_seconds": time.time() - started,
        }
        (output_dir / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        commit()
    return progress
