# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-locked public HarmBench white-box contract for the served Kimi K3."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.ultra_whitebox import (
    PIPELINE_SHA256,
    WhiteboxMethod,
    sha256,
)
from benchmarks.harmbench.ultra_whitebox import WHITEBOX_METHODS as TEXT_WHITEBOX_METHODS


MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
MODEL_TOTAL_BYTES = 1_560_998_984_390
MODEL_FILE_COUNT = 118
MODEL_KEY = "kimi_k3_served_9f62e4e9"
SOURCE_VOLUME = "endpoint-ep-WD4gnzaeXMyM7DfrzTDYjW"
SNAPSHOT_PATH = (
    "/checkpoint/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)

VISION_WHITEBOX_METHODS = {
    method.public_name: method
    for method in (
        WhiteboxMethod(
            "MultiModalPGD",
            "MultiModalPGD",
            "MultiModalPGD_config.yaml",
            "5056a2c2fae1d41f74c15b4d47d2920e586025e012cdb6cc84602b2e52e1d8d4",
            1,
        ),
        WhiteboxMethod(
            "MultiModalPGDPatch",
            "MultiModalPGDPatch",
            "MultiModalPGDPatch_config.yaml",
            "ae8e45a178c72e81643911eb1fed3b6a38a932292a0fa3d1c034885f5974426a",
            1,
        ),
    )
}
WHITEBOX_METHODS = dict(TEXT_WHITEBOX_METHODS) | VISION_WHITEBOX_METHODS


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def validate_upstream(upstream: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    pipeline_path = upstream / "configs/pipeline_configs/run_pipeline.yaml"
    if sha256(pipeline_path) != PIPELINE_SHA256:
        raise ValueError("public pipeline config hash mismatch")
    pipeline = _yaml(pipeline_path)
    methods = {}
    for name, spec in WHITEBOX_METHODS.items():
        entry = pipeline.get(name)
        if not isinstance(entry, dict) or entry.get("class_name") != spec.class_name:
            raise ValueError(f"public pipeline mapping changed for {name}")
        expected_type = "open_source_multimodal" if name in VISION_WHITEBOX_METHODS else "open_source"
        if expected_type not in entry.get("allowed_target_model_types", []):
            raise ValueError(f"public target type changed for {name}")
        config = upstream / "configs/method_configs" / spec.config_name
        if sha256(config) != spec.config_sha256:
            raise ValueError(f"public method config hash mismatch for {name}")
        methods[name] = {
            "class_name": spec.class_name,
            "config_sha256": spec.config_sha256,
            "default_hyperparameters": _yaml(config)["default_method_hyperparameters"],
            "run_ids": list(spec.run_ids),
            "cases_per_behavior": spec.cases_per_behavior,
        }
    return {"upstream_revision": head, "pipeline_sha256": PIPELINE_SHA256, "methods": methods}


def validate_checkpoint_manifest(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "verified",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "source_volume": SOURCE_VOLUME,
        "file_count": MODEL_FILE_COUNT,
        "total_bytes": MODEL_TOTAL_BYTES,
    }
    mismatch = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise ValueError(f"Kimi K3 checkpoint manifest mismatch: {mismatch}")
    return receipt


def write_runtime_configs(upstream: Path, output: Path, checkpoint: Path) -> tuple[Path, dict[str, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    model = {
        MODEL_KEY: {
            "model": {
                "model_name_or_path": str(checkpoint),
                "trust_remote_code": True,
                "use_fast_tokenizer": True,
                "dtype": "auto",
                "device_map": "auto",
            },
            "num_gpus": 7,
            "model_type": "open_source_multimodal",
        }
    }
    models_path = output / "models-kimi-k3.yaml"
    models_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    effective = {}
    for name, spec in WHITEBOX_METHODS.items():
        source = upstream / "configs/method_configs" / spec.config_name
        if name not in {"AutoDAN", "FewShot", *VISION_WHITEBOX_METHODS}:
            effective[name] = source
            continue
        config = _yaml(source)
        config[MODEL_KEY] = {
            "target_model": {
                "model_name_or_path": str(checkpoint),
                "trust_remote_code": True,
                "dtype": "auto",
                "device_map": "auto",
            }
        }
        if name == "AutoDAN":
            config[MODEL_KEY]["target_model"].update({"model_short_name": "Kimi K3", "developer_name": "Moonshot AI"})
        if name == "FewShot":
            config["default_method_hyperparameters"]["attack_model"]["num_gpus"] = 1
        if name in VISION_WHITEBOX_METHODS:
            config[MODEL_KEY].update({"test_cases_batch_size": 1, "num_test_cases_per_behavior": 1})
        path = output / f"{name}-kimi-k3.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        effective[name] = path
    return models_path, effective


def expected_full_cases() -> dict[str, int]:
    return {
        name: spec.cases_per_behavior * (110 if name in VISION_WHITEBOX_METHODS else 320)
        for name, spec in WHITEBOX_METHODS.items()
    }
