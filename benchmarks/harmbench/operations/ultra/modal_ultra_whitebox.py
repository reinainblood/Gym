# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FDR-only eight-B200 runtime for the exact Ultra white-box checkpoint."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "harmbench-ultra-whitebox"
MODEL_VOLUME = "nemotron-3-ultra-550b-a55b-bf16-77df655d"
OUTPUT_VOLUME = "harmbench-ultra-whitebox-results"
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"
WORLD_SIZE = 8
LOCAL = Path(__file__).parent
UPSTREAM = Path(os.environ.get("HARMBENCH_UPSTREAM", LOCAL / "upstream-harmbench"))
GYM_ROOT = LOCAL.parents[3]

weights = modal.Volume.from_name(MODEL_VOLUME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, environment_name="FDR", create_if_missing=False)
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.22.0",
        setup_dockerfile_commands=[
            "RUN command -v python3 && python3 --version && "
            "(command -v python || ln -s /usr/bin/python3 /usr/local/bin/python) && "
            "(command -v pip || ln -s /usr/bin/pip3 /usr/local/bin/pip)"
        ],
    )
    .entrypoint([])
    .apt_install("git")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .uv_pip_install(
        "accelerate>=1.12.0",
        "bpe",
        "confection==0.1.4",
        "datasketch==2.0.0",
        "fschat",
        "numpy==1.26.4",
        "transformers==5.17.0",
        "pandas<3",
        "pyyaml",
        "ray",
        "sentence-transformers",
        "spacy==3.7.2",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    )
    .add_local_file(
        Path(__file__).with_name("ultra_gradient_canary_worker.py"),
        "/app/ultra_gradient_canary_worker.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("incremental_generate.py"),
        "/app/incremental_generate.py",
        copy=True,
    )
    .add_local_dir(
        UPSTREAM,
        "/app/HarmBench",
        copy=True,
    )
    .add_local_file(
        LOCAL / "upstream-harmbench-revision.txt",
        "/app/HarmBench/.pinned-revision",
        copy=True,
    )
    .add_local_dir(
        GYM_ROOT / "benchmarks",
        "/app/Gym/benchmarks",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("gym_benchmarks_init.py"),
        "/app/Gym/benchmarks/__init__.py",
        copy=True,
    )
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=2,
    memory=8_192,
    timeout=600,
    volumes={"/checkpoint": weights.with_mount_options(read_only=True)},
)
def package_probe() -> dict[str, object]:
    """Check the pinned container's architecture and distributed-loader support."""
    import torch
    import transformers
    import vllm
    from transformers import AutoConfig
    from transformers.distributed.configuration_utils import DistributedConfig

    config = AutoConfig.from_pretrained("/checkpoint/model", local_files_only=True)
    distributed = DistributedConfig(fsdp_size=WORLD_SIZE)
    result = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "vllm": str(vllm.__version__),
        "modelopt": "not_required_for_bf16_runtime",
        "architecture": config.architectures,
        "model_type": config.model_type,
        "quant_method": (getattr(config, "quantization_config", None) or {}).get("quant_method"),
        "distributed_fsdp_size": distributed.fsdp_size,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu=f"B200:{WORLD_SIZE}",
    cpu=32,
    memory=524_288,
    timeout=7_200,
    startup_timeout=3_600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/checkpoint": weights.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def gradient_canary(run_id: str) -> dict[str, object]:
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must contain lowercase letters, digits, and hyphens only")
    manifest = Path("/checkpoint/checkpoint-manifest.json")
    model = Path("/checkpoint/model")
    if not manifest.is_file():
        raise FileNotFoundError("verified checkpoint manifest is not present yet")
    receipt = Path("/outputs") / run_id / "gradient-canary-receipt.json"
    if receipt.exists():
        raise FileExistsError("choose a new gradient canary run ID")
    subprocess.run(
        [
            "python",
            "/app/ultra_gradient_canary_worker.py",
            "--model-dir",
            str(model),
            "--manifest",
            str(manifest),
            "--output",
            str(receipt),
        ],
        check=True,
    )
    result = json.loads(receipt.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "world_size": WORLD_SIZE,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "loss_decreased_all_ranks": True,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"gradient canary receipt failed {key}")
    outputs.commit()
    return {
        "run_id": run_id,
        "status": result["status"],
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "loss_before": result["loss_before"],
        "loss_after_one_step": result["loss_after_one_step"],
        "output_volume": OUTPUT_VOLUME,
    }


@app.function(
    image=image,
    gpu=f"B200:{WORLD_SIZE}",
    cpu=32,
    memory=524_288,
    timeout=86_400,
    startup_timeout=3_600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/checkpoint": weights.with_mount_options(read_only=True),
        "/outputs": outputs,
        "/root/.cache/huggingface": modal.Volume.from_name(
            "harmbench-mixtral-attacker-cache",
            environment_name="FDR",
            create_if_missing=False,
        ),
    },
)
def run_attack(
    artifact_id: str,
    method_name: str,
    public_run_id: str = "",
    behavior_ids: str = "",
) -> dict[str, object]:
    """Run one full public method or a source-row-preserving bounded canary."""
    if not re.fullmatch(r"[a-z0-9-]+", artifact_id):
        raise ValueError("artifact_id must contain lowercase letters, digits, and hyphens only")
    sys.path.insert(0, "/app/Gym")
    from benchmarks.harmbench.ultra_whitebox import (
        MODEL_KEY,
        WHITEBOX_METHODS,
        _validate_cases,
        sha256,
        validate_checkpoint_manifest,
        validate_upstream,
        write_runtime_configs,
    )

    if method_name not in WHITEBOX_METHODS:
        raise ValueError(f"unsupported white-box method: {method_name}")
    spec = WHITEBOX_METHODS[method_name]
    run_id = public_run_id or None
    if spec.run_ids and run_id not in spec.run_ids:
        raise ValueError(f"{method_name} requires public_run_id in {spec.run_ids}")
    if not spec.run_ids and run_id is not None:
        raise ValueError(f"{method_name} does not use a public run ID")

    upstream = Path("/app/HarmBench")
    checkpoint = Path("/checkpoint/model")
    checkpoint_manifest = Path("/checkpoint/checkpoint-manifest.json")
    output_root = Path("/outputs") / artifact_id
    if (output_root / "completed-receipt.json").exists():
        raise FileExistsError("artifact ID already has a completed receipt")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_receipt = validate_upstream(upstream)
    model_receipt = validate_checkpoint_manifest(checkpoint_manifest, checkpoint)
    models_config, method_configs = write_runtime_configs(upstream, output_root / "runtime-config", checkpoint)

    canonical_behaviors = upstream / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    behaviors = canonical_behaviors
    selected_ids = [value for value in behavior_ids.split(",") if value]
    if selected_ids:
        import csv

        with canonical_behaviors.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = [row for row in reader if row["BehaviorID"] in set(selected_ids)]
            fieldnames = reader.fieldnames
        if len(rows) != len(set(selected_ids)):
            raise ValueError("one or more requested behavior IDs are absent from the canonical text test set")
        behaviors = output_root / "selected-behaviors.csv"
        with behaviors.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    output_dir = output_root / method_name / MODEL_KEY / "test_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "/app/incremental_generate.py",
        "--upstream",
        str(upstream),
        "--models-config",
        str(models_config),
        "--method-config",
        str(method_configs[method_name]),
        "--behaviors",
        str(behaviors),
        "--output-dir",
        str(output_dir),
        "--public-name",
        method_name,
        "--class-name",
        spec.class_name,
        "--experiment",
        MODEL_KEY,
    ]
    if run_id is not None:
        command.extend(["--run-id", run_id])
    subprocess.run(command, cwd=upstream, check=True)
    outputs.commit()

    if spec.run_ids:
        missing = [value for value in spec.run_ids if not (output_dir / f"test_cases_{value}.json").is_file()]
        if missing:
            return {
                "artifact_id": artifact_id,
                "method": method_name,
                "public_run_id": run_id,
                "status": "waiting_for_public_repetitions",
                "remaining_run_ids": missing,
                "output_volume": OUTPUT_VOLUME,
            }
    subprocess.run(
        ["python", "merge_test_cases.py", "--method_name", spec.class_name, "--save_dir", str(output_dir)],
        cwd=upstream,
        check=True,
    )
    cases_path = output_dir / "test_cases.json"
    counts = _validate_cases(cases_path, behaviors, spec.cases_per_behavior)
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "scope": "bounded_canary" if selected_ids else "full_public_text_test",
        "behavior_ids": selected_ids or None,
        "artifact_id": artifact_id,
        "method": method_name,
        "upstream_class": spec.class_name,
        "public_run_ids": list(spec.run_ids) if spec.run_ids else ["upstream-default"],
        "upstream": upstream_receipt,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
        "checkpoint_file_count": model_receipt["file_count"],
        "checkpoint_total_bytes": model_receipt["total_bytes"],
        "behaviors_sha256": sha256(behaviors),
        "canonical_behaviors_sha256": sha256(canonical_behaviors),
        "models_config_sha256": sha256(models_config),
        "effective_method_config_sha256": sha256(method_configs[method_name]),
        "cases_sha256": sha256(cases_path),
        **counts,
    }
    receipt_path = output_root / "completed-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.commit()
    return {
        "artifact_id": artifact_id,
        "method": method_name,
        "status": "completed",
        "scope": receipt["scope"],
        "behaviors": counts["behaviors"],
        "cases": counts["cases"],
        "cases_sha256": receipt["cases_sha256"],
        "output_volume": OUTPUT_VOLUME,
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    print(json.dumps(gradient_canary.remote(run_id), sort_keys=True))
