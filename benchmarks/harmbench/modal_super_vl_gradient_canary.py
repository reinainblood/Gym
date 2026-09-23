# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one ephemeral, eight-H200 Super VL input-gradient canary in FDR."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import modal


APP_NAME = "harmbench-super-vl-gradient-canary"
MODEL_VOLUME = "nemotron-3-5-super-vl-ea-09112026"
OUTPUT_VOLUME = "harmbench-super-vl-gradient-canary-results"
MODEL_SUBDIR = "nemotron_3_5_super_ea_09112026_vhf-ea-0e636f7"
MODEL_VERSION = "hf-ea-0e636f7"
VLLM_IMAGE = "vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e"
WORLD_SIZE = 8

weights = modal.Volume.from_name(MODEL_VOLUME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, environment_name="FDR", create_if_missing=True)
image = (
    modal.Image.from_registry(
        VLLM_IMAGE,
        setup_dockerfile_commands=[
            "RUN command -v python3 && python3 --version && "
            "(command -v python || ln -s /usr/bin/python3 /usr/local/bin/python) && "
            "(command -v pip || ln -s /usr/bin/pip3 /usr/local/bin/pip)"
        ],
    )
    .entrypoint([])
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install("accelerate>=1.12.0", "pillow>=11.0.0")
    .add_local_file(
        Path(__file__).with_name("super_vl_gradient_canary_worker.py"),
        "/app/super_vl_gradient_canary_worker.py",
    )
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=f"H200:{WORLD_SIZE}",
    cpu=32,
    memory=524288,
    timeout=3600,
    startup_timeout=1800,
    min_containers=0,
    max_containers=1,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def gradient_canary(run_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must contain lowercase letters, digits, and hyphens only")
    manifest_path = Path("/weights/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified" or manifest.get("version") != MODEL_VERSION:
        raise ValueError("Super VL checkpoint manifest is not the verified EA version")
    model_dir = Path("/weights") / MODEL_SUBDIR
    if Path(manifest["model_path"]) != model_dir:
        raise ValueError("Super VL checkpoint manifest points to an unexpected model directory")
    receipt = Path("/outputs") / run_id / "gradient-canary-receipt.json"
    if receipt.exists():
        raise FileExistsError("choose a new gradient canary run ID")
    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        "/app/super_vl_gradient_canary_worker.py",
        "--model-dir",
        str(model_dir),
        "--output",
        str(receipt),
    ]
    subprocess.run(command, check=True)
    result = json.loads(receipt.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "model_version": MODEL_VERSION,
        "world_size": WORLD_SIZE,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "loss_changed": True,
        "loss_decreased_all_ranks": True,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"gradient canary receipt failed {key}")
    outputs.commit()
    return {
        "run_id": run_id,
        "status": result["status"],
        "world_size": result["world_size"],
        "loss_before": result["loss_before"],
        "loss_after_one_step": result["loss_after_one_step"],
        "gradient_nonzero": result["gradient_nonzero"],
        "output_volume": OUTPUT_VOLUME,
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    print(json.dumps(gradient_canary.remote(run_id), sort_keys=True))
