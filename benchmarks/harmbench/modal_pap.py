# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ephemeral FDR Mixtral generator for pinned HarmBench PAP-top5 cases.

Set HARMBENCH_SOURCE_CSV and HARMBENCH_PAP_TEMPLATES to the verified pinned
HarmBench checkout before ``modal run --env=FDR``. The app is never deployed;
it logs no prompts, generations, or credentials and commits receipts to its
dedicated output Volume.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
APP_NAME = "harmbench-pap-top5-generator"
MODEL_VOLUME_NAME = "harmbench-mixtral-attacker-cache"
OUTPUT_VOLUME_NAME = "harmbench-pap-top5-cases"

model_cache = modal.Volume.from_name(MODEL_VOLUME_NAME, environment_name="FDR", create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, environment_name="FDR", create_if_missing=True)
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==4.57.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/root/.cache/huggingface"})
)
if modal.is_local():
    source_csv = Path(os.environ["HARMBENCH_SOURCE_CSV"]).resolve(strict=True)
    source_templates = Path(os.environ["HARMBENCH_PAP_TEMPLATES"]).resolve(strict=True)
    image = (
        base_image.add_local_file(HERE / "pap.py", "/app/pap.py")
        .add_local_file(HERE / "zero_shot.py", "/app/zero_shot.py")
        .add_local_file(source_csv, "/app/behaviors.csv")
        .add_local_file(source_templates, "/app/templates.py")
    )
else:
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
    from pap import CASES_PER_BEHAVIOR, EXPERIMENT, SAMPLING, TEMPLATE_SHA256, cases, load_templates, queries
    from zero_shot import (
        ATTACKER_MODEL,
        ATTACKER_REVISION,
        FULL_TEXT_BEHAVIORS,
        FULL_TEXT_BEHAVIORS_SHA256,
        mixtral_chat_template,
    )

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must be lowercase letters, digits, or hyphens")
    output_dir = Path("/outputs") / run_id
    output_path = output_dir / "test_cases.json"
    if output_path.exists():
        raise FileExistsError("PAP output already exists; choose a new run ID")
    source_path = Path("/app/behaviors.csv")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != FULL_TEXT_BEHAVIORS_SHA256:
        raise ValueError("behavior source is not the pinned public full text corpus")
    with source_path.open(newline="", encoding="utf-8") as handle:
        all_behaviors = list(csv.DictReader(handle))
    if len(all_behaviors) != FULL_TEXT_BEHAVIORS:
        raise ValueError(f"expected {FULL_TEXT_BEHAVIORS} full-corpus behaviors, found {len(all_behaviors)}")
    selected = [row for row in all_behaviors if row["BehaviorID"] == behavior_id] if behavior_id else all_behaviors
    if not selected or (behavior_id and len(selected) != 1):
        raise ValueError("requested behavior is absent from the pinned full text corpus")
    templates = load_templates(Path("/app/templates.py"))
    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, revision=ATTACKER_REVISION)
    chat_template = mixtral_chat_template(tokenizer)
    prompts = queries(selected, chat_template, templates)
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
    raw_path = output_dir / "raw_attacker_generations.json"
    raw_path.write_text(json.dumps(generations, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "method": "PAP-top5",
        "upstream_method": "PAP-top5",
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
        "templates_sha256": TEMPLATE_SHA256,
        "behaviors_sha256": FULL_TEXT_BEHAVIORS_SHA256,
        "test_cases_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "raw_attacker_generations_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
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
