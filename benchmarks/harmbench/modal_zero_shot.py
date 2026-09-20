# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ephemeral FDR Mixtral generator for HarmBench ZeroShot test cases.

Set HARMBENCH_SOURCE_CSV to the pinned text-test CSV before `modal run
--env=FDR`. This app is not deployed or kept warm; a single generation call
uses 1×H200, commits its output to a dedicated FDR Volume, then exits. No
attack prompts, model generations, or credentials are printed to Modal logs.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
APP_NAME = "harmbench-zero-shot-generator"
MODEL_VOLUME_NAME = "harmbench-mixtral-attacker-cache"
OUTPUT_VOLUME_NAME = "harmbench-zero-shot-cases"

model_cache = modal.Volume.from_name(MODEL_VOLUME_NAME, environment_name="FDR", create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, environment_name="FDR", create_if_missing=True)
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==4.57.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/root/.cache/huggingface"})
)
if modal.is_local():
    source_csv = Path(os.environ["HARMBENCH_SOURCE_CSV"]).resolve(strict=True)
    image = base_image.add_local_file(HERE / "zero_shot.py", "/app/zero_shot.py").add_local_file(
        source_csv, "/app/behaviors.csv"
    )
else:
    # Modal imports this module again inside the container after local files
    # are baked into the image; no Mac path exists or should be consulted there.
    image = base_image
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H200",
    timeout=3600,
    startup_timeout=1200,
    min_containers=0,
    max_containers=1,
    volumes={"/root/.cache/huggingface": model_cache, "/outputs": output_volume},
)
def generate(run_id: str, behavior_id: str = "") -> dict:
    import csv
    import hashlib
    import json
    import re
    import sys

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, "/app")
    from zero_shot import (
        ATTACKER_MODEL,
        ATTACKER_REVISION,
        CASES_PER_BEHAVIOR,
        EXPERIMENT,
        SAMPLING,
        cases,
        mixtral_chat_template,
        queries,
    )

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must be lowercase letters, digits, or hyphens")
    output_dir = Path("/outputs") / run_id
    output_path = output_dir / "test_cases.json"
    if output_path.exists():
        raise FileExistsError("ZeroShot output already exists; choose a new run ID")
    with Path("/app/behaviors.csv").open(newline="", encoding="utf-8") as handle:
        all_behaviors = list(csv.DictReader(handle))
    selected = [row for row in all_behaviors if row["BehaviorID"] == behavior_id] if behavior_id else all_behaviors
    if not selected or (behavior_id and len(selected) != 1):
        raise ValueError("requested behavior is absent from the pinned text-test CSV")
    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, revision=ATTACKER_REVISION)
    chat_template = mixtral_chat_template(tokenizer)
    prompts = queries(selected, chat_template)
    model = LLM(
        model=ATTACKER_MODEL,
        revision=ATTACKER_REVISION,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        moe_backend="triton",
        seed=1,
    )
    generations = [
        output.outputs[0].text for output in model.generate(prompts, SamplingParams(**SAMPLING), use_tqdm=False)
    ]
    generated_cases = cases(selected, generations)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path.write_text(json.dumps(generated_cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "method": "ZeroShot",
        "upstream_method": "ZeroShot",
        "upstream_revision": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "attacker_seed": 1,
        "gpu": "H200",
        "tensor_parallel_size": 1,
        "vllm_version": vllm.__version__,
        "vllm_enforce_eager": True,
        "moe_backend": "triton",
        "sampling": SAMPLING,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "behaviors_sha256": hashlib.sha256(Path("/app/behaviors.csv").read_bytes()).hexdigest(),
        "test_cases_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "behaviors": len(selected),
        "cases": len(selected) * CASES_PER_BEHAVIOR,
    }
    (output_dir / "generation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_cache.commit()
    output_volume.commit()
    return {
        "run_id": run_id,
        "behaviors": receipt["behaviors"],
        "cases": receipt["cases"],
        "attacker_model": ATTACKER_MODEL,
        "attacker_revision": ATTACKER_REVISION,
        "test_cases_sha256": receipt["test_cases_sha256"],
        "output_volume": OUTPUT_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(run_id: str, behavior_id: str = "") -> None:
    import json

    print(json.dumps(generate.remote(run_id, behavior_id), sort_keys=True))
