# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resumable exact-BF16 Qwen target completions for finalized GCG artifacts in FDR."""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-qwen-gcg-completion"
MODEL_VOLUME = "qwen3-5-122b-a10b-bf16-dc4d3484"
RESULTS_VOLUME = "harmbench-gcg-super-qwen-results"
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
LOCAL = Path(__file__).parent

weights = modal.Volume.from_name(MODEL_VOLUME, environment_name="FDR", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=False)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel")
    .entrypoint([])
    .apt_install("git")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/app:/app/HarmBench",
        }
    )
    .uv_pip_install(
        "transformers==5.17.0",
        "accelerate>=1.12.0",
        "safetensors>=0.6.2",
        "packaging>=26.0",
        "ninja>=1.13.0",
    )
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0 flash-linear-attention==0.5.2")
    .run_commands("pip install --no-build-isolation mamba-ssm==2.3.2.post1")
    .run_commands(
        "git init /app/HarmBench && "
        "git -C /app/HarmBench remote add origin https://github.com/centerforaisafety/HarmBench.git && "
        f"git -C /app/HarmBench fetch --depth 1 origin {UPSTREAM_REVISION} && "
        "git -C /app/HarmBench checkout --detach FETCH_HEAD"
    )
    .add_local_file(LOCAL / "qwen_gcg_completion_worker.py", "/app/qwen_gcg_completion_worker.py")
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
    max_containers=2,
    scaledown_window=900,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/outputs": results,
    },
)
class QwenGCGCompletion:
    @modal.enter()
    def load(self):
        self.runtime = None

    def ensure_runtime(self):
        from qwen_gcg_completion_worker import QwenGCGCompletionRuntime

        if self.runtime is None:
            self.runtime = QwenGCGCompletionRuntime()
        return self.runtime

    @modal.method()
    def complete_shard(self, artifact_id: str, shard_index: int, num_shards: int, limit: int = 0) -> str:
        from qwen_gcg_completion_worker import (
            BEHAVIORS,
            generation_paths,
            load_generation_artifact,
            sha256,
            shard_indexes,
            write_completion_shard_receipt,
        )

        if not re.fullmatch(r"[a-z0-9-]+", artifact_id):
            raise ValueError("invalid artifact id")
        selected = shard_indexes(shard_index, num_shards, limit)
        output_root = Path("/outputs") / artifact_id
        behaviors, cases, _ = load_generation_artifact(
            output_root=output_root,
            behaviors_path=BEHAVIORS,
            artifact_id=artifact_id,
        )
        completion_dir = output_root / "target-completions" / "qwen-bf16" / "individual"
        if all((completion_dir / f"{index:03d}.json").is_file() for index in selected):
            receipt = write_completion_shard_receipt(
                output_root=output_root,
                artifact_id=artifact_id,
                shard_index=shard_index,
                num_shards=num_shards,
                selected=selected,
            )
            results.commit()
            return json.dumps(receipt, sort_keys=True)

        runtime = self.ensure_runtime()
        _, generation_receipt_path = generation_paths(output_root)
        generation_receipt_sha256 = sha256(generation_receipt_path)
        completed = 0
        for index in selected:
            behavior = behaviors[index]
            attack = cases[behavior["BehaviorID"]][0]
            runtime.complete(
                artifact_id=artifact_id,
                index=index,
                behavior=behavior,
                attack=attack,
                generation_receipt_sha256=generation_receipt_sha256,
                output_dir=completion_dir,
            )
            results.commit()
            completed += 1
        receipt = write_completion_shard_receipt(
            output_root=output_root,
            artifact_id=artifact_id,
            shard_index=shard_index,
            num_shards=num_shards,
            selected=selected,
            behaviors_path=BEHAVIORS,
        )
        results.commit()
        return json.dumps(
            {
                "artifact_id": artifact_id,
                "target": "qwen",
                "status": receipt["status"],
                "shard_index": shard_index,
                "num_shards": num_shards,
                "completed_behaviors": completed,
            },
            sort_keys=True,
        )


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/outputs": results},
)
def finalize(artifact_id: str) -> dict:
    from qwen_gcg_completion_worker import BEHAVIORS, finalize_completion_artifact

    if not re.fullmatch(r"[a-z0-9-]+", artifact_id):
        raise ValueError("invalid artifact id")
    final = finalize_completion_artifact(
        output_root=Path("/outputs") / artifact_id,
        behaviors_path=BEHAVIORS,
        artifact_id=artifact_id,
    )
    results.commit()
    return {
        "artifact_id": artifact_id,
        "target": "qwen",
        "status": final["status"],
        "behaviors": final["behaviors"],
        "completions": final["completions"],
    }


@app.local_entrypoint()
def main(artifact_id: str, num_shards: int = 2, limit: int = 0) -> None:
    runtime = QwenGCGCompletion()
    launched = []
    for shard_index in range(num_shards):
        call = runtime.complete_shard.spawn(artifact_id, shard_index, num_shards, limit)
        launched.append({"shard_index": shard_index, "function_call_id": call.object_id})
    print(json.dumps({"artifact_id": artifact_id, "target": "qwen", "calls": launched}, sort_keys=True))
