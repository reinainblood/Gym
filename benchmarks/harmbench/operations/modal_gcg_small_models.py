# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run public HarmBench GCG on the verified Super and Qwen BF16 weights in FDR."""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-gcg-super-qwen"
RESULTS_VOLUME = "harmbench-gcg-super-qwen-results"
SUPER_VOLUME = "nemotron-3-5-super-vl-ea-09112026"
QWEN_VOLUME = "qwen3-5-122b-a10b-bf16-dc4d3484"
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
LOCAL = Path(__file__).parent

super_weights = modal.Volume.from_name(SUPER_VOLUME, environment_name="FDR", create_if_missing=False)
qwen_weights = modal.Volume.from_name(QWEN_VOLUME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=True)
base_image = (
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
            "PYTHONPATH": "/app:/app/Gym:/app/HarmBench",
        }
    )
    .uv_pip_install(
        "accelerate>=1.12.0",
        "bpe",
        "datasketch==2.0.0",
        "fschat",
        "numpy==1.26.4",
        "pandas<3",
        "pyyaml",
        "ray",
        "sentence-transformers",
        "spacy==3.7.2",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    )
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0 flash-linear-attention==0.5.2")
    .run_commands("pip install --no-build-isolation mamba-ssm==2.3.2.post1")
    .run_commands(
        "git init /app/HarmBench && "
        "git -C /app/HarmBench remote add origin https://github.com/centerforaisafety/HarmBench.git && "
        f"git -C /app/HarmBench fetch --depth 1 origin {UPSTREAM_REVISION} && "
        "git -C /app/HarmBench checkout --detach FETCH_HEAD"
    )
)
if modal.is_local():
    gym_root = LOCAL.parents[2]
    image = (
        base_image.add_local_file(
            LOCAL / "gcg_small_models_worker.py",
            "/app/gcg_small_models_worker.py",
        )
        .add_local_file(
            LOCAL / "ultra" / "incremental_generate.py",
            "/app/incremental_generate.py",
        )
        .add_local_file(
            gym_root / "benchmarks" / "harmbench" / "cache_compat.py",
            "/app/Gym/benchmarks/harmbench/cache_compat.py",
        )
        .add_local_file(
            gym_root / "benchmarks" / "harmbench" / "__init__.py",
            "/app/Gym/benchmarks/harmbench/__init__.py",
        )
        .add_local_file(
            LOCAL / "ultra" / "gym_benchmarks_init.py",
            "/app/Gym/benchmarks/__init__.py",
        )
    )
else:
    image = base_image
app = modal.App(APP_NAME)


def _validate_call(artifact_id: str, shard_index: int, num_shards: int, limit: int) -> None:
    if not re.fullmatch(r"[a-z0-9-]+", artifact_id):
        raise ValueError("artifact_id must contain lowercase letters, digits, and hyphens only")
    if not 0 <= shard_index < num_shards <= 16:
        raise ValueError("invalid shard selection")
    if limit < 0:
        raise ValueError("limit cannot be negative")


def _run(target: str, artifact_id: str, shard_index: int, num_shards: int, limit: int) -> dict:
    from gcg_small_models_worker import run_shard

    _validate_call(artifact_id, shard_index, num_shards, limit)
    result = run_shard(
        target=target,
        artifact_id=artifact_id,
        shard_index=shard_index,
        num_shards=num_shards,
        limit=limit,
    )
    outputs.commit()
    return {
        "artifact_id": artifact_id,
        "target": target,
        "status": result["status"],
        "completed_behaviors": result["completed_behaviors"],
        "selected_behaviors": result["selected_behaviors"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "output_volume": RESULTS_VOLUME,
    }


@app.function(
    image=image,
    gpu="H200:4",
    cpu=32,
    memory=524288,
    timeout=86400,
    startup_timeout=3600,
    min_containers=0,
    max_containers=8,
    volumes={"/weights": super_weights.with_mount_options(read_only=True), "/outputs": outputs},
)
def run_super(artifact_id: str, shard_index: int, num_shards: int, limit: int = 0) -> dict:
    """Run one resumable Super GCG shard."""
    return _run("super", artifact_id, shard_index, num_shards, limit)


@app.function(
    image=image,
    gpu="B200:2",
    cpu=24,
    memory=393216,
    timeout=86400,
    startup_timeout=3600,
    min_containers=0,
    max_containers=4,
    volumes={"/weights": qwen_weights.with_mount_options(read_only=True), "/outputs": outputs},
)
def run_qwen(artifact_id: str, shard_index: int, num_shards: int, limit: int = 0) -> dict:
    """Run one resumable Qwen GCG shard."""
    return _run("qwen", artifact_id, shard_index, num_shards, limit)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/outputs": outputs},
)
def finalize(target: str, artifact_id: str) -> dict:
    """Merge a finished campaign and emit the Gym-compatible generation receipt."""
    from gcg_small_models_worker import finalize_artifact

    _validate_call(artifact_id, 0, 1, 0)
    result = finalize_artifact(target=target, artifact_id=artifact_id)
    outputs.commit()
    return {
        "artifact_id": artifact_id,
        "target": target,
        "status": result["status"],
        "behaviors": result["behaviors"],
        "cases": result["cases"],
        "test_cases_sha256": result["test_cases_sha256"],
        "output_volume": RESULTS_VOLUME,
    }


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/outputs": outputs},
)
def status_campaign(target: str, artifact_id: str, num_shards: int) -> dict:
    """Return exact missing/invalid indexes and minimal repair shards without loading weights."""
    from gcg_small_models_worker import PUBLIC_DATASET, reconcile_campaign_status

    if target not in {"super", "qwen"}:
        raise ValueError("target must be super or qwen")
    _validate_call(artifact_id, 0, num_shards, 0)
    return reconcile_campaign_status(
        output_root=Path("/outputs") / artifact_id,
        behaviors=Path("/app/HarmBench/data/behavior_datasets") / PUBLIC_DATASET,
        target=target,
        artifact_id=artifact_id,
        num_shards=num_shards,
    )


@app.local_entrypoint()
def main(target: str, artifact_id: str, num_shards: int = 1, limit: int = 1, detach: bool = False) -> None:
    """Launch a bounded canary or a full set of resumable shards."""
    if target not in {"super", "qwen"}:
        raise ValueError("target must be super or qwen")
    function = run_super if target == "super" else run_qwen
    calls = []
    for shard_index in range(num_shards):
        call = function.spawn(artifact_id, shard_index, num_shards, limit)
        calls.append({"shard_index": shard_index, "function_call_id": call.object_id})
    print(json.dumps({"target": target, "artifact_id": artifact_id, "calls": calls}, indent=2, sort_keys=True))
    if not detach:
        results = [modal.FunctionCall.from_id(row["function_call_id"]).get() for row in calls]
        print(json.dumps(results, indent=2, sort_keys=True))
