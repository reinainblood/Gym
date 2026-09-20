# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the canonical HarmBench white-box methods against Nemotron 3 Ultra.

This module is an execution adapter for the public HarmBench implementation. It
does not reimplement or approximate any attack. It validates the immutable
upstream checkout and method configs, binds the exact local checkpoint, and
dispatches the upstream attack classes with their published defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


SERVED_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
SERVED_MODEL_REVISION = "02462641f13d3af838b904f48195b9bb8a1e4ebc"
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"
MODEL_TOTAL_BYTES = 1_121_078_448_762
MODEL_KEY = "nemotron_3_ultra_550b_a55b_bf16"
PIPELINE_SHA256 = "14071e847282121ab3cc89a0fb99af82a2bce138a26e4badf997dcc6d4c37710"
PUBLIC_FULL_TEXT_DATASET = "harmbench_behaviors_text_all.csv"
PUBLIC_FULL_TEXT_BEHAVIORS = 400


@dataclass(frozen=True)
class WhiteboxMethod:
    public_name: str
    class_name: str
    config_name: str
    config_sha256: str
    cases_per_behavior: int
    run_ids: tuple[str, ...] = ()


WHITEBOX_METHODS: dict[str, WhiteboxMethod] = {
    method.public_name: method
    for method in (
        WhiteboxMethod(
            "GCG",
            "GCG",
            "GCG_config.yaml",
            "f046d305c30938a29ee51af011cee6380c833d6a11246d7a68f78288da8bfc99",
            1,
        ),
        WhiteboxMethod(
            "GCG-Multi",
            "EnsembleGCG",
            "EnsembleGCG_config.yaml",
            "eece35ed068032df9f12c6eaf852078dcfae3d00a92b1ee0b2f75fd2a9c64d4f",
            5,
            ("0", "1", "2", "3", "4"),
        ),
        WhiteboxMethod(
            "AutoPrompt",
            "AutoPrompt",
            "AutoPrompt_config.yaml",
            "0887dcb1fb8637b05b44adb475cb1eefb9ff7356cc2a7aa96ec745e155dc4972",
            1,
        ),
        WhiteboxMethod(
            "GBDA",
            "GBDA",
            "GBDA_config.yaml",
            "14572275354078065f67f8eb95b7b023f5525e72ce13efcf6e9a13d728327c2d",
            5,
        ),
        WhiteboxMethod(
            "PEZ",
            "PEZ",
            "PEZ_config.yaml",
            "4ccb3c01ecb39f0579b347f7f4b15e40d9d73c5d6d7a9072c67b2211013253b0",
            5,
        ),
        WhiteboxMethod(
            "UAT",
            "UAT",
            "UAT_config.yaml",
            "4fc01891c14d81383a2c9be002e5b521d15806ab60a7c50342844b86212622b3",
            1,
        ),
        WhiteboxMethod(
            "AutoDAN",
            "AutoDAN",
            "AutoDAN_config.yaml",
            "beaba0873558e1de3b9e61301c9948abedf63e0227ea5d365e8a2fe891e7bfaf",
            1,
        ),
        WhiteboxMethod(
            "FewShot",
            "FewShot",
            "FewShot_config.yaml",
            "11919d2720a357e0a68f175e66d54db2b7e0245319b90b09c96eabf79eed3c65",
            1,
        ),
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def validate_upstream(upstream: Path) -> dict[str, Any]:
    """Prove that every method resolves to the immutable public implementation."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        marker = upstream / ".pinned-revision"
        if not marker.is_file():
            raise ValueError("HarmBench checkout has neither git identity nor a pinned revision marker") from None
        head = marker.read_text(encoding="utf-8").strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    pipeline_path = upstream / "configs/pipeline_configs/run_pipeline.yaml"
    if sha256(pipeline_path) != PIPELINE_SHA256:
        raise ValueError("public HarmBench pipeline config hash mismatch")
    pipeline = _load_yaml(pipeline_path)
    methods: dict[str, Any] = {}
    for public_name, spec in WHITEBOX_METHODS.items():
        entry = pipeline.get(public_name)
        if not isinstance(entry, dict) or entry.get("class_name") != spec.class_name:
            raise ValueError(f"public pipeline mapping changed for {public_name}")
        if entry.get("allowed_target_model_types") != ["open_source"]:
            raise ValueError(f"{public_name} is no longer the public open-source white-box protocol")
        run_ids = tuple(str(value) for value in entry.get("run_ids", ()))
        if run_ids != spec.run_ids:
            raise ValueError(f"public repetitions changed for {public_name}: {run_ids}")
        config_path = upstream / "configs/method_configs" / spec.config_name
        actual_config_hash = sha256(config_path)
        if actual_config_hash != spec.config_sha256:
            raise ValueError(f"public method config hash mismatch for {public_name}")
        methods[public_name] = {
            "class_name": spec.class_name,
            "config_path": str(config_path),
            "config_sha256": actual_config_hash,
            "default_hyperparameters": _load_yaml(config_path)["default_method_hyperparameters"],
            "run_ids": list(spec.run_ids),
            "cases_per_behavior": spec.cases_per_behavior,
        }
    return {
        "upstream_repository": "centerforaisafety/HarmBench",
        "upstream_revision": head,
        "pipeline_sha256": PIPELINE_SHA256,
        "methods": methods,
    }


def validate_checkpoint_manifest(manifest_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Require the complete immutable checkpoint receipt before a GPU run."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "total_bytes": MODEL_TOTAL_BYTES,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"checkpoint manifest identity mismatch: {mismatches}")
    if manifest.get("gated") not in (False, "false", None) or manifest.get("private") is not False:
        raise ValueError("checkpoint receipt does not prove public, ungated access")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest.get("file_count"):
        raise ValueError("checkpoint file manifest is incomplete")
    missing = [item["path"] for item in files if not (checkpoint_path / item["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing {len(missing)} files; first missing path: {missing[0]}")
    return manifest


def write_runtime_configs(upstream: Path, runtime_dir: Path, checkpoint_path: Path) -> tuple[Path, dict[str, Path]]:
    """Bind Ultra while preserving every public default method hyperparameter."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        MODEL_KEY: {
            "model": {
                "model_name_or_path": str(checkpoint_path),
                "use_fast_tokenizer": True,
                "dtype": "bfloat16",
                "device_map": "auto",
            },
            # Seven B200s hold the BF16 target; the eighth holds Mixtral. The
            # public config's two A100s for Mixtral are one B200 here.
            "num_gpus": 7,
            "model_type": "open_source",
        }
    }
    models_path = runtime_dir / "models-ultra.yaml"
    models_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    method_configs: dict[str, Path] = {}
    for public_name, spec in WHITEBOX_METHODS.items():
        source = upstream / "configs/method_configs" / spec.config_name
        if public_name not in {"GCG", "AutoDAN", "FewShot"}:
            method_configs[public_name] = source
            continue
        config = _load_yaml(source)
        if public_name == "GCG":
            # Execution-only microbatch control. The public search width remains
            # 512 and every candidate participates in the same global argmin.
            # Beginning at 512 spends minutes replicating the hybrid cache before
            # Accelerate can observe and recover from an OOM.
            config["default_method_hyperparameters"]["starting_search_batch_size"] = 8
            effective_path = runtime_dir / "GCG-ultra.yaml"
            effective_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            method_configs[public_name] = effective_path
            continue
        # These two public configs list only the paper's original target models,
        # so add the Ultra target experiment. AutoDAN also needs its two prompt
        # substitutions. FewShot's two-A100 attacker fits on one B200; every
        # algorithm and sampling parameter remains the public default.
        config[MODEL_KEY] = {
            "target_model": {
                "model_name_or_path": str(checkpoint_path),
                "use_fast_tokenizer": True,
                "dtype": "bfloat16",
                "device_map": "auto",
            }
        }
        if public_name == "AutoDAN":
            config[MODEL_KEY]["target_model"].update(
                {"model_short_name": "Nemotron 3 Ultra", "developer_name": "NVIDIA"}
            )
        else:
            config["default_method_hyperparameters"]["attack_model"]["num_gpus"] = 1
        effective_path = runtime_dir / f"{public_name}-ultra.yaml"
        effective_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        method_configs[public_name] = effective_path
    return models_path, method_configs


def build_generate_command(
    *,
    upstream: Path,
    python: Path,
    behaviors: Path,
    output_dir: Path,
    method_name: str,
    models_config: Path,
    method_config: Path,
    run_id: str | None = None,
) -> list[str]:
    spec = WHITEBOX_METHODS[method_name]
    if spec.run_ids and run_id not in spec.run_ids:
        raise ValueError(f"{method_name} requires one of public run IDs {spec.run_ids}")
    if not spec.run_ids and run_id is not None:
        raise ValueError(f"{method_name} has no repeated public runs")
    command = [
        str(python),
        "generate_test_cases.py",
        "--method_name",
        spec.class_name,
        "--experiment_name",
        MODEL_KEY,
        "--behaviors_path",
        str(behaviors),
        "--save_dir",
        str(output_dir),
        "--models_config_file",
        str(models_config),
        "--method_config_file",
        str(method_config),
    ]
    if run_id is not None:
        command.extend(["--run_id", run_id])
    return command


def _expected_behavior_ids(behaviors: Path) -> set[str]:
    with behaviors.open(newline="", encoding="utf-8") as stream:
        return {row["BehaviorID"] for row in csv.DictReader(stream)}


def _validate_cases(cases_path: Path, behaviors: Path, expected_cases: int) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    expected_ids = _expected_behavior_ids(behaviors)
    if set(cases) != expected_ids:
        raise ValueError(f"case behavior IDs differ: missing={sorted(expected_ids - set(cases))[:3]}")
    wrong = {behavior_id: len(values) for behavior_id, values in cases.items() if len(values) != expected_cases}
    if wrong:
        raise ValueError(f"wrong case repetitions: {dict(list(wrong.items())[:3])}")
    return {"behaviors": len(cases), "cases": sum(len(values) for values in cases.values())}


def run_method(
    *,
    upstream: Path,
    checkpoint_path: Path,
    checkpoint_manifest: Path,
    behaviors: Path,
    output_root: Path,
    method_name: str,
    python: Path,
    run_id: str | None = None,
) -> Path | None:
    upstream_receipt = validate_upstream(upstream)
    model_receipt = validate_checkpoint_manifest(checkpoint_manifest, checkpoint_path)
    runtime_dir = output_root / "runtime-config"
    models_config, method_configs = write_runtime_configs(upstream, runtime_dir, checkpoint_path)
    spec = WHITEBOX_METHODS[method_name]
    output_dir = output_root / method_name / MODEL_KEY / "test_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_generate_command(
        upstream=upstream,
        python=python,
        behaviors=behaviors,
        output_dir=output_dir,
        method_name=method_name,
        models_config=models_config,
        method_config=method_configs[method_name],
        run_id=run_id,
    )
    subprocess.run(command, cwd=upstream, check=True)

    if spec.run_ids:
        missing = [value for value in spec.run_ids if not (output_dir / f"test_cases_{value}.json").is_file()]
        if missing:
            return None
    subprocess.run(
        [str(python), "merge_test_cases.py", "--method_name", spec.class_name, "--save_dir", str(output_dir)],
        cwd=upstream,
        check=True,
    )
    cases_path = output_dir / "test_cases.json"
    counts = _validate_cases(cases_path, behaviors, spec.cases_per_behavior)
    receipt = {
        "schema_version": 1,
        "method": method_name,
        "upstream_class": spec.class_name,
        "experiment": MODEL_KEY,
        "run_ids": list(spec.run_ids) if spec.run_ids else ["upstream-default"],
        "upstream": upstream_receipt,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
        "checkpoint_file_count": model_receipt["file_count"],
        "checkpoint_total_bytes": model_receipt["total_bytes"],
        "behaviors_sha256": sha256(behaviors),
        "models_config_sha256": sha256(models_config),
        "effective_method_config_sha256": sha256(method_configs[method_name]),
        "cases_sha256": sha256(cases_path),
        **counts,
    }
    receipt_path = output_dir / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=WHITEBOX_METHODS)
    parser.add_argument("--run-id")
    parser.add_argument("--python", type=Path, default=Path("/usr/bin/python3"))
    args = parser.parse_args()
    result = run_method(
        upstream=args.upstream,
        checkpoint_path=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        behaviors=args.behaviors,
        output_root=args.output_root,
        method_name=args.method,
        python=args.python,
        run_id=args.run_id,
    )
    print(result or "waiting for remaining public GCG-Multi run IDs")


if __name__ == "__main__":
    main()
