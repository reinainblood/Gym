# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned HarmBench GCG against the verified Qwen and Super BF16 checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
PIPELINE_SHA256 = "14071e847282121ab3cc89a0fb99af82a2bce138a26e4badf997dcc6d4c37710"  # pragma: allowlist secret
GCG_CONFIG_SHA256 = "f046d305c30938a29ee51af011cee6380c833d6a11246d7a68f78288da8bfc99"  # pragma: allowlist secret
PUBLIC_DATASET = "harmbench_behaviors_text_all.csv"
PUBLIC_BEHAVIORS = 400
PUBLIC_STEPS = 500
PUBLIC_SEARCH_WIDTH = 512

TARGETS: dict[str, dict[str, Any]] = {
    "super": {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "model_revision": "hf-ea-0e636f7",
        "model_subdir": "nemotron_3_5_super_ea_09112026_vhf-ea-0e636f7",
        "manifest": "manifest.json",
        "manifest_identity": {"status": "verified", "version": "hf-ea-0e636f7"},
        "trust_remote_code": True,
        "starting_search_batch_size": 8,
    },
    "qwen": {
        "model_id": "Qwen/Qwen3.5-122B-A10B",
        "model_revision": "dc4d348443bc740c68e2d77492492c11606384d5",  # pragma: allowlist secret
        "model_subdir": "snapshot",
        "manifest": "manifest.json",
        "manifest_identity": {
            "status": "verified",
            "model_id": "Qwen/Qwen3.5-122B-A10B",
            "revision": "dc4d348443bc740c68e2d77492492c11606384d5",  # pragma: allowlist secret
        },
        "trust_remote_code": False,
        "starting_search_batch_size": 8,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def behavior_artifact_path(individual_dir: Path, behavior_id: str) -> Path:
    """Return upstream SingleBehaviorRedTeamingMethod's per-behavior case path."""
    return individual_dir / behavior_id / "test_cases.json"


def completed_behavior_count(individual_dir: Path, behavior_ids: list[str]) -> int:
    """Count only complete upstream per-behavior case files."""
    return sum(behavior_artifact_path(individual_dir, behavior_id).is_file() for behavior_id in behavior_ids)


def run_generation_if_needed(
    *, individual_dir: Path, behavior_ids: list[str], command: list[str], upstream: Path
) -> None:
    """Avoid reloading frontier weights when a resumable shard is already complete."""
    if completed_behavior_count(individual_dir, behavior_ids) != len(behavior_ids):
        subprocess.run(command, cwd=upstream, check=True)


def validate_upstream(upstream: Path) -> dict[str, str]:
    """Validate the exact public pipeline and GCG method configuration."""
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    pipeline = upstream / "configs/pipeline_configs/run_pipeline.yaml"
    gcg_config = upstream / "configs/method_configs/GCG_config.yaml"
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {revision}, expected {UPSTREAM_REVISION}")
    if sha256(pipeline) != PIPELINE_SHA256:
        raise ValueError("public HarmBench pipeline config hash mismatch")
    if sha256(gcg_config) != GCG_CONFIG_SHA256:
        raise ValueError("public HarmBench GCG config hash mismatch")
    mapping = yaml.safe_load(pipeline.read_text(encoding="utf-8"))["GCG"]
    if mapping.get("class_name") != "GCG" or mapping.get("allowed_target_model_types") != ["open_source"]:
        raise ValueError("public HarmBench GCG pipeline mapping changed")
    defaults = yaml.safe_load(gcg_config.read_text(encoding="utf-8"))["default_method_hyperparameters"]
    if defaults.get("num_steps") != PUBLIC_STEPS or defaults.get("search_width") != PUBLIC_SEARCH_WIDTH:
        raise ValueError("public HarmBench GCG hyperparameters changed")
    return {
        "revision": revision,
        "pipeline_sha256": PIPELINE_SHA256,
        "gcg_config_sha256": GCG_CONFIG_SHA256,
    }


def validate_checkpoint(target: str, weights_root: Path) -> tuple[dict[str, Any], Path]:
    """Bind the run to one already verified immutable checkpoint volume."""
    spec = TARGETS[target]
    manifest_path = weights_root / spec["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in spec["manifest_identity"].items():
        if manifest.get(key) != expected:
            raise ValueError(f"{target} checkpoint manifest failed {key}")
    model_dir = weights_root / spec["model_subdir"]
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"{target} checkpoint directory is incomplete")
    if target == "super":
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        keys = set(index.get("weight_map", {}))
        required_language_weights = {
            "language_model.backbone.embeddings.weight",
            "language_model.lm_head.weight",
        }
        if not required_language_weights.issubset(keys):
            raise ValueError("Super checkpoint index is missing the language embedding or LM head")
        # The EA checkpoint also publishes top-level aliases of these tied weights. Transformers
        # reports those aliases as unexpected after loading the actual language_model.* tensors.
        alias_weights = {"backbone.embeddings.weight", "lm_head.weight"}
        if not alias_weights.issubset(keys):
            raise ValueError("Super checkpoint no longer contains the documented tied-weight aliases")
    return manifest, model_dir


def shard_behavior_ids(behaviors: Path, shard_index: int, num_shards: int, limit: int = 0) -> list[str]:
    """Return the stable source-order behavior partition for one resumable shard."""
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard selection")
    with behaviors.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != PUBLIC_BEHAVIORS:
        raise ValueError(f"expected {PUBLIC_BEHAVIORS} public behaviors, found {len(rows)}")
    selected = [row["BehaviorID"] for row in rows[shard_index::num_shards]]
    return selected[:limit] if limit else selected


def write_runtime_configs(upstream: Path, output_root: Path, target: str, model_dir: Path) -> tuple[Path, Path]:
    """Bind a target while changing only execution microbatching from public GCG."""
    spec = TARGETS[target]
    runtime_dir = output_root / "runtime-config"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_key = f"nemotron_3_5_{target}_gcg"
    models_path = runtime_dir / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                model_key: {
                    "model": {
                        "model_name_or_path": str(model_dir),
                        "dtype": "bfloat16",
                        "device_map": "auto",
                        "trust_remote_code": bool(spec["trust_remote_code"]),
                        "use_fast_tokenizer": True,
                    },
                    "num_gpus": 1,
                    "model_type": "open_source",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load((upstream / "configs/method_configs/GCG_config.yaml").read_text(encoding="utf-8"))
    config["default_method_hyperparameters"]["starting_search_batch_size"] = spec["starting_search_batch_size"]
    # The public Qwen profile disables prefix caching because its cache format is not compatible
    # with the original tuple expansion. Super's wrapper likewise keeps cache configuration on its
    # inner language model. Recomputing the fixed prefix preserves the exact loss and argmin math.
    config["default_method_hyperparameters"]["use_prefix_cache"] = False
    method_path = runtime_dir / "GCG.yaml"
    method_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return models_path, method_path


def run_shard(
    *,
    target: str,
    artifact_id: str,
    shard_index: int,
    num_shards: int,
    limit: int = 0,
) -> dict[str, Any]:
    """Run one resumable source-order GCG shard and write a payload-free receipt."""
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    upstream = Path("/app/HarmBench")
    outputs = Path("/outputs")
    output_root = outputs / artifact_id
    upstream_receipt = validate_upstream(upstream)
    checkpoint_manifest, model_dir = validate_checkpoint(target, Path("/weights"))
    behaviors = upstream / "data/behavior_datasets" / PUBLIC_DATASET
    selected = shard_behavior_ids(behaviors, shard_index, num_shards, limit)
    models_path, method_path = write_runtime_configs(upstream, output_root, target, model_dir)
    model_key = f"nemotron_3_5_{target}_gcg"
    output_dir = output_root / "GCG" / model_key / "test_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "test_cases_individual_behaviors"
    command = [
        "python",
        "/app/incremental_generate.py",
        "--upstream",
        str(upstream),
        "--models-config",
        str(models_path),
        "--method-config",
        str(method_path),
        "--behaviors",
        str(behaviors),
        "--output-dir",
        str(output_dir),
        "--public-name",
        "GCG",
        "--class-name",
        "GCG",
        "--experiment",
        model_key,
        "--behavior-ids",
        ",".join(selected),
        "--quiet-upstream",
    ]
    # Resume and receipt repair must not reload a frontier checkpoint when every selected
    # upstream behavior artifact is already present.
    run_generation_if_needed(individual_dir=individual_dir, behavior_ids=selected, command=command, upstream=upstream)
    completed = completed_behavior_count(individual_dir, selected)
    if completed != len(selected):
        raise ValueError(f"GCG shard saved {completed}/{len(selected)} expected behavior artifacts")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "artifact_id": artifact_id,
        "target": target,
        "model_id": TARGETS[target]["model_id"],
        "model_revision": TARGETS[target]["model_revision"],
        "checkpoint_manifest_sha256": sha256(Path("/weights") / TARGETS[target]["manifest"]),
        "checkpoint_manifest_status": checkpoint_manifest.get("status"),
        "upstream": upstream_receipt,
        "method": "GCG",
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
        "starting_search_batch_size": TARGETS[target]["starting_search_batch_size"],
        "execution_only_deviation": (
            "candidate loss microbatching and disabled prefix reuse; public inputs, 500 steps, "
            "512-candidate search, token gradients, losses, and global argmin preserved"
        ),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "selected_behaviors": len(selected),
        "completed_behaviors": completed,
        "behavior_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "model_config_sha256": sha256(models_path),
        "method_config_sha256": sha256(method_path),
    }
    receipt_path = output_root / "shard-receipts" / f"{target}-{shard_index:02d}-of-{num_shards:02d}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def finalize_artifact(*, target: str, artifact_id: str) -> dict[str, Any]:
    """Merge a complete 400-behavior campaign and write the Gym-compatible generation receipt."""
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    upstream = Path("/app/HarmBench")
    output_root = Path("/outputs") / artifact_id
    model_key = f"nemotron_3_5_{target}_gcg"
    output_dir = output_root / "GCG" / model_key / "test_cases"
    individual_dir = output_dir / "test_cases_individual_behaviors"
    behaviors = upstream / "data/behavior_datasets" / PUBLIC_DATASET
    selected = shard_behavior_ids(behaviors, 0, 1)
    missing = [
        behavior_id for behavior_id in selected if not behavior_artifact_path(individual_dir, behavior_id).is_file()
    ]
    if missing:
        raise ValueError(f"cannot finalize: {len(missing)} GCG behaviors are missing")
    subprocess.run(
        ["python", "merge_test_cases.py", "--method_name", "GCG", "--save_dir", str(output_dir)],
        cwd=upstream,
        check=True,
    )
    cases_path = output_dir / "test_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if set(cases) != set(selected) or any(not isinstance(value, list) or len(value) != 1 for value in cases.values()):
        raise ValueError("merged GCG cases do not contain exactly one case for every public behavior")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": "GCG",
        "upstream_method": "GCG",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": model_key,
        "run_id": artifact_id,
        "target_type": "text_weights",
        "source_target_model": TARGETS[target]["model_id"],
        "source_target_revision": TARGETS[target]["model_revision"],
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behaviors),
        "behaviors": PUBLIC_BEHAVIORS,
        "cases": PUBLIC_BEHAVIORS,
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
        "execution_only_deviation": (
            "candidate loss microbatching and disabled prefix reuse; public inputs, 500 steps, "
            "512-candidate search, token gradients, losses, and global argmin preserved"
        ),
    }
    receipt_path = output_root / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
