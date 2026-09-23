# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modal FDR runtime for full Qwen 3.5 HarmBench multimodal white-box methods."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-qwen35-122b-whitebox"
MODEL_VOLUME = "qwen3-5-122b-a10b-bf16-dc4d3484"
RESULTS_VOLUME = "harmbench-qwen35-122b-whitebox-results"
LOCAL = Path(__file__).parent
UPSTREAM = Path(os.environ.get("HARMBENCH_UPSTREAM", LOCAL / "upstream-harmbench"))
GYM_DATA = LOCAL.parents[1] / "data"

weights = modal.Volume.from_name(MODEL_VOLUME, environment_name="FDR", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=True)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel")
    .entrypoint([])
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/app",
        }
    )
    .uv_pip_install(
        "transformers==5.17.0",
        "accelerate>=1.12.0",
        "torchvision==0.24.1",
        "pillow>=11.0.0",
        "safetensors>=0.6.2",
        "packaging>=26.0",
        "ninja>=1.13.0",
    )
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0 flash-linear-attention==0.5.2")
    .add_local_file(LOCAL / "qwen_harmbench_worker.py", "/app/qwen_harmbench_worker.py")
    .add_local_dir(UPSTREAM / "data", "/harmbench/data")
    .add_local_dir(UPSTREAM / "configs/method_configs", "/harmbench/configs")
    .add_local_file(
        GYM_DATA / "harmbench_behaviors_multimodal_corrected.csv",
        "/harmbench/harmbench_behaviors_multimodal_corrected.csv",
    )
    .add_local_file(
        GYM_DATA / "harmbench_behaviors_multimodal_corrected.corrections.json",
        "/harmbench/harmbench_behaviors_multimodal_corrected.corrections.json",
    )
)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="B200:2",
    cpu=24,
    memory=393216,
    timeout=86400,
    startup_timeout=3600,
    min_containers=0,
    max_containers=4,
    scaledown_window=900,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/results": results,
    },
)
class QwenWhitebox:
    @modal.enter()
    def load(self):
        self.runtime = None

    def ensure_runtime(self):
        from qwen_harmbench_worker import QwenHarmBenchRuntime

        if self.runtime is None:
            self.runtime = QwenHarmBenchRuntime()
        return self.runtime

    @modal.method()
    def gradient_probe(self, run_id: str) -> str:
        if not re.fullmatch(r"[a-z0-9-]+", run_id):
            raise ValueError("invalid run id")
        output = Path("/results") / run_id / "gradient-receipt.json"
        if output.exists():
            return output.read_text()
        receipt = self.ensure_runtime().gradient_probe(output)
        results.commit()
        return json.dumps(receipt, sort_keys=True)

    @modal.method()
    def run_shard(self, run_id: str, method: str, shard_index: int, num_shards: int) -> str:
        from qwen_harmbench_worker import METHODS

        if not re.fullmatch(r"[a-z0-9-]+", run_id):
            raise ValueError("invalid run id")
        if method not in METHODS:
            raise ValueError("unknown public method")
        if not 0 <= shard_index < num_shards <= 16:
            raise ValueError("invalid shard selection")
        run_dir = Path("/results") / run_id / method
        completed = []
        runtime = self.ensure_runtime()
        for index in range(shard_index, 110, num_shards):
            completed.append(runtime.attack_case(method, index, run_dir))
            results.commit()
        return json.dumps(
            {"run_id": run_id, "method": method, "shard": shard_index, "num_shards": num_shards, "cases": completed},
            sort_keys=True,
        )


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/results": results},
)
def finalize_method(run_id: str, method: str) -> dict:
    from qwen_harmbench_worker import CORRECTED_BEHAVIORS, METHODS, finalize_method_artifact

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run id")
    if method not in METHODS:
        raise ValueError("unknown public method")
    run_root = Path("/results") / run_id
    manifest = finalize_method_artifact(
        run_dir=run_root / method,
        method=method,
        behaviors_path=CORRECTED_BEHAVIORS,
        gradient_receipt_path=run_root / "gradient-receipt.json",
    )
    results.commit()
    return {
        "run_id": run_id,
        "method": method,
        "status": manifest["status"],
        "cases": manifest["cases"],
        "images": manifest["images"],
    }


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/results": results},
)
def status_method(run_id: str, method: str, num_shards: int = 4) -> dict:
    from qwen_harmbench_worker import CORRECTED_BEHAVIORS, METHODS, reconcile_method_status

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run id")
    if method not in METHODS:
        raise ValueError("unknown public method")
    return reconcile_method_status(
        run_dir=Path("/results") / run_id / method,
        method=method,
        behaviors_path=CORRECTED_BEHAVIORS,
        num_shards=num_shards,
    )


@app.local_entrypoint()
def main(run_id: str = "qwen35-122b-harmbench-whitebox-20260919", launch_full: bool = True) -> None:
    runtime = QwenWhitebox()
    probe = json.loads(runtime.gradient_probe.remote(run_id))
    required = {
        "status": "passed",
        "gradient_finite": True,
        "gradient_nonzero": True,
        "loss_decreased": True,
    }
    for key, expected in required.items():
        if probe.get(key) != expected:
            raise ValueError(f"gradient receipt failed {key}")
    launched = []
    if launch_full:
        for method in ("MultiModalPGD", "MultiModalPGDPatch", "MultiModalPGDBlankImage"):
            for shard in range(4):
                call = runtime.run_shard.spawn(run_id, method, shard, 4)
                launched.append({"method": method, "shard": shard, "function_call_id": call.object_id})
    print(json.dumps({"probe": probe, "launched": launched}, sort_keys=True))
