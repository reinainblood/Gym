# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modal FDR runtime that regenerates Qwen white-box completions at the 512-token protocol cap.

Deployed as its own app so the long-running attack-optimization containers in
``harmbench-qwen35-122b-whitebox`` are never redeployed or interrupted. It mounts the same exact
BF16 weights volume and the same results volume, and it reads each case's published PNG rather than
any in-memory tensor, so completion is reproducible from committed artifacts alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-qwen35-122b-completion"
MODEL_VOLUME = "qwen3-5-122b-a10b-bf16-dc4d3484"
RESULTS_VOLUME = "harmbench-qwen35-122b-whitebox-results"
LOCAL = Path(__file__).parent
METHODS = ("MultiModalPGD", "MultiModalPGDPatch", "MultiModalPGDBlankImage")

weights = modal.Volume.from_name(MODEL_VOLUME, environment_name="FDR", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=False)
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
    .add_local_file(LOCAL / "qwen_completion_worker.py", "/app/qwen_completion_worker.py")
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
class QwenCompletion:
    @modal.enter()
    def load(self):
        self.runtime = None

    def ensure_runtime(self):
        from qwen_completion_worker import QwenCompletionRuntime

        if self.runtime is None:
            self.runtime = QwenCompletionRuntime()
        return self.runtime

    @modal.method()
    def complete_shard(
        self,
        run_id: str,
        method: str,
        shard_index: int,
        num_shards: int,
        replicate: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Complete one shard. ``replicate`` writes to a sibling directory for determinism checks."""
        from qwen_completion_worker import COMPLETIONS_DIRNAME, validate_attack_manifest

        if not re.fullmatch(r"[a-z0-9-]+", run_id):
            raise ValueError("invalid run id")
        if method not in METHODS:
            raise ValueError(f"unknown public method: {method}")
        if not 0 <= shard_index < num_shards <= 16:
            raise ValueError("invalid shard selection")
        if replicate is not None and not re.fullmatch(r"[a-z0-9-]+", replicate):
            raise ValueError("invalid replicate label")

        method_dir = Path("/results") / run_id / method
        validate_attack_manifest(method_dir, method)
        case_dir = method_dir / "cases"
        if not case_dir.is_dir():
            raise FileNotFoundError(f"no completed attack cases for {method}")
        # A replicate never touches the canonical directory, so the protocol result cannot be
        # overwritten by a diagnostic re-run.
        output_dir = method_dir / (
            f"{COMPLETIONS_DIRNAME}-replicate-{replicate}" if replicate else COMPLETIONS_DIRNAME
        )

        cases = sorted(case_dir.glob("*.json"))
        if limit is not None:
            cases = cases[:limit]
        runtime = self.ensure_runtime()
        completed = []
        for case_path in cases[shard_index::num_shards]:
            completed.append(runtime.complete_case(case_path, output_dir))
            results.commit()
        return json.dumps(
            {
                "run_id": run_id,
                "method": method,
                "shard": shard_index,
                "num_shards": num_shards,
                "replicate": replicate,
                "observed_cases": len(cases),
                "results": completed,
            },
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
    from qwen_completion_worker import WHITEBOX_METHODS, finalize_completion_artifact

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run id")
    if method not in WHITEBOX_METHODS:
        raise ValueError("unknown public method")
    manifest = finalize_completion_artifact(method_dir=Path("/results") / run_id / method, method=method)
    results.commit()
    return {
        "run_id": run_id,
        "method": method,
        "status": manifest["status"],
        "completions": manifest["completions"],
    }


@app.local_entrypoint()
def main(
    run_id: str = "qwen35-122b-harmbench-whitebox-20260919",
    method: str = "MultiModalPGD",
    num_shards: int = 4,
    replicate: str = "",
    limit: int = 0,
) -> None:
    """Spawn completion shards for one method; attack optimization is never touched."""
    runtime = QwenCompletion()
    launched = []
    for shard in range(num_shards):
        call = runtime.complete_shard.spawn(run_id, method, shard, num_shards, replicate or None, limit or None)
        launched.append({"method": method, "shard": shard, "function_call_id": call.object_id})
    print(json.dumps({"run_id": run_id, "method": method, "launched": launched}, indent=2, sort_keys=True))
