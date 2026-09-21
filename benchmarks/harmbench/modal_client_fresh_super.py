# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resumable FDR PAIR/TAP generators against the served Super VL EA target.

The attacker and judge remain the pinned public Mixtral configuration. Only the
target is replaced by the independently bound OpenAI-compatible Super endpoint.
Outputs stay on a private Modal Volume; function results and logs are metadata-only.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
APP_NAME = "harmbench-client-fresh-super"
MODEL_VOLUME_NAME = "harmbench-mixtral-attacker-cache"
OUTPUT_VOLUME_NAME = "harmbench-client-fresh-super-results"
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
TARGET_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
TARGET_REVISION = "hf-ea-0e636f7"
TARGET_BASE_URL = "https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct"
PUBLIC_METHOD = {
    "PAIR": "Fresh PAIR against the client model",
    "TAP": "Fresh TAP against the client model",
}

model_cache = modal.Volume.from_name(MODEL_VOLUME_NAME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME_NAME, environment_name="FDR", create_if_missing=True)
service_secret = modal.Secret.from_name("nemotron-super-vl-service", environment_name="FDR")
base_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.21.0",
        setup_dockerfile_commands=[
            "RUN command -v python3 && python3 --version && "
            "(command -v python || ln -s /usr/bin/python3 /usr/local/bin/python) && "
            "(command -v pip || ln -s /usr/bin/pip3 /usr/local/bin/pip)"
        ],
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.12.0",
        "anthropic",
        "art",
        "boto3",
        "bpe",
        "datasketch==2.0.0",
        "fschat",
        "google-cloud-aiplatform",
        "google-generativeai",
        "matplotlib",
        "mistralai",
        "numpy==1.26.4",
        "openai>=1.25.1",
        "pandas<3",
        "pyyaml",
        "ray",
        "sentence-transformers",
        "spacy==3.7.2",
        "tenacity",
        "transformers==4.57.1",
    )
    .run_commands(
        "git init /app/HarmBench && "
        "git -C /app/HarmBench remote add origin https://github.com/centerforaisafety/HarmBench.git && "
        f"git -C /app/HarmBench fetch --depth 1 origin {UPSTREAM_REVISION} && "
        "git -C /app/HarmBench checkout --detach FETCH_HEAD"
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HARMBENCH_CLIENT_TARGET_CONCURRENCY": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/app:/app/HarmBench",
        }
    )
)
if modal.is_local():
    source_csv = Path(os.environ["HARMBENCH_SOURCE_CSV"]).resolve(strict=True)
    image = (
        base_image.add_local_file(HERE / "client_fresh_generate.py", "/app/client_fresh_generate.py")
        .add_local_file(HERE / "evidence" / "super-vl-client-binding.yaml", "/app/client-binding.yaml")
        .add_local_file(source_csv, "/app/behaviors.csv")
    )
else:
    image = base_image

app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(method: str, run_id: str, shard_index: int, num_shards: int, limit: int = 0) -> None:
    import re

    if method not in PUBLIC_METHOD:
        raise ValueError("method must be PAIR or TAP")
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must contain lowercase letters, digits, or hyphens")
    if not 0 <= shard_index < num_shards <= 16:
        raise ValueError("invalid shard selection")
    if limit < 0:
        raise ValueError("limit cannot be negative")


@app.function(
    image=image,
    gpu="H200:2",
    cpu=16,
    memory=262144,
    timeout=86400,
    startup_timeout=1800,
    min_containers=0,
    max_containers=8,
    secrets=[service_secret],
    volumes={"/root/.cache/huggingface": model_cache, "/outputs": outputs},
)
def run_shard(method: str, run_id: str, shard_index: int, num_shards: int, limit: int = 0) -> dict:
    import csv
    import json
    import os
    import sys

    _validate(method, run_id, shard_index, num_shards, limit)
    sys.path.insert(0, "/app")
    from client_fresh_generate import run, sha256

    source = Path("/app/behaviors.csv")
    with source.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    if len(all_rows) != 400:
        raise ValueError("client-fresh source must contain all 400 public text behaviors")
    selected = all_rows[shard_index::num_shards]
    if limit:
        selected = selected[:limit]
    shard_root = Path("/outputs") / run_id / method.lower() / "shards" / f"{shard_index:02d}-of-{num_shards:02d}"
    completed = shard_root / "shard-receipt.json"
    if completed.is_file():
        value = json.loads(completed.read_text(encoding="utf-8"))
        if value.get("status") != "completed" or value.get("selected_behaviors") != len(selected):
            raise ValueError("existing shard receipt is invalid")
        return {
            "method": method,
            "run_id": run_id,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "selected_behaviors": len(selected),
            "status": "completed",
            "output_volume": OUTPUT_VOLUME_NAME,
        }
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_csv = shard_root / "behaviors.csv"
    with shard_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    previous_cwd = Path.cwd()
    os.chdir("/app/HarmBench")
    try:
        target_receipt = run(
            upstream=Path("/app/HarmBench"),
            method=method,
            behaviors=shard_csv,
            output_dir=shard_root / "generated",
            client_base_url=TARGET_BASE_URL,
            client_model=TARGET_MODEL,
            client_revision=TARGET_REVISION,
            api_key_env="VLLM_API_KEY",
            client_binding_receipt=Path("/app/client-binding.yaml"),
        )
    finally:
        os.chdir(previous_cwd)
    generated_cases = shard_root / "generated" / "test_cases.json"
    target_value = json.loads(target_receipt.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": method,
        "public_method": PUBLIC_METHOD[method],
        "run_id": run_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "limit": limit,
        "selected_source_indexes": list(range(shard_index, len(all_rows), num_shards)),
        "selected_behaviors": len(selected),
        "behaviors_sha256": sha256(shard_csv),
        "test_cases_sha256": sha256(generated_cases),
        "client_target_receipt_sha256": sha256(target_receipt),
        "client_model": target_value["client_model"],
        "client_revision": target_value["client_revision"],
        "target_calls": target_value["target_calls"],
        "upstream_revision": UPSTREAM_REVISION,
    }
    completed.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_cache.commit()
    outputs.commit()
    return {
        "method": method,
        "run_id": run_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "selected_behaviors": len(selected),
        "target_calls": receipt["target_calls"],
        "status": "completed",
        "output_volume": OUTPUT_VOLUME_NAME,
    }


def _status(method: str, run_id: str, num_shards: int) -> dict:
    import csv
    import json

    _validate(method, run_id, 0, num_shards)
    with Path("/app/behaviors.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    root = Path("/outputs") / run_id / method.lower()
    valid_shards = []
    for shard_index in range(num_shards):
        shard_root = root / "shards" / f"{shard_index:02d}-of-{num_shards:02d}"
        receipt_path = shard_root / "shard-receipt.json"
        cases_path = shard_root / "generated" / "test_cases.json"
        if not receipt_path.is_file() or not cases_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_indexes = list(range(shard_index, len(rows), num_shards))
        if (
            receipt.get("status") == "completed"
            and receipt.get("method") == method
            and receipt.get("run_id") == run_id
            and receipt.get("shard_index") == shard_index
            and receipt.get("num_shards") == num_shards
            and receipt.get("selected_source_indexes") == expected_indexes
            and receipt.get("test_cases_sha256") == _sha256(cases_path)
        ):
            valid_shards.append(shard_index)
    missing = sorted(set(range(num_shards)) - set(valid_shards))
    return {
        "method": method,
        "run_id": run_id,
        "num_shards": num_shards,
        "valid_shards": valid_shards,
        "missing_shards": missing,
        "ready_to_finalize": not missing,
    }


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={"/outputs": outputs})
def status(method: str, run_id: str, num_shards: int) -> dict:
    return _status(method, run_id, num_shards)


@app.function(image=image, cpu=2, memory=16384, timeout=1800, volumes={"/outputs": outputs})
def finalize(method: str, run_id: str, num_shards: int) -> dict:
    import csv
    import hashlib
    import json

    state = _status(method, run_id, num_shards)
    if not state["ready_to_finalize"]:
        raise ValueError("client-fresh campaign is incomplete")
    source = Path("/app/behaviors.csv")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ordered_ids = [row["BehaviorID"] for row in rows]
    root = Path("/outputs") / run_id / method.lower()
    merged: dict[str, list[str]] = {}
    shard_manifest = []
    total_calls = 0
    for shard_index in range(num_shards):
        shard_root = root / "shards" / f"{shard_index:02d}-of-{num_shards:02d}"
        receipt_path = shard_root / "shard-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        shard_cases = json.loads((shard_root / "generated" / "test_cases.json").read_text(encoding="utf-8"))
        if set(shard_cases) != {ordered_ids[index] for index in receipt["selected_source_indexes"]}:
            raise ValueError("client-fresh shard cases disagree with source partition")
        if any(not isinstance(value, list) or len(value) != 1 for value in shard_cases.values()):
            raise ValueError("client-fresh method must yield exactly one attack case per behavior")
        merged.update(shard_cases)
        total_calls += int(receipt["target_calls"])
        shard_manifest.append(
            {
                "name": receipt_path.name.replace("shard-receipt", f"{shard_index:02d}-of-{num_shards:02d}"),
                "sha256": _sha256(receipt_path),
            }
        )
    if set(merged) != set(ordered_ids):
        raise ValueError("client-fresh merged cases do not cover the public corpus")
    canonical = {behavior_id: merged[behavior_id] for behavior_id in ordered_ids}
    cases_path = root / "test_cases.json"
    cases_path.write_text(json.dumps(canonical, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    canonical_manifest = json.dumps(shard_manifest, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": PUBLIC_METHOD[method],
        "upstream_method": method,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": "super-vl-client-fresh",
        "run_id": run_id,
        "test_cases_sha256": _sha256(cases_path),
        "behaviors_sha256": _sha256(source),
        "behaviors": len(rows),
        "cases": len(rows),
        "client_model": TARGET_MODEL,
        "source_target_model": TARGET_MODEL,
        "client_revision": TARGET_REVISION,
        "source_target_revision": TARGET_REVISION,
        "target_calls": total_calls,
        "num_shards": num_shards,
        "shard_receipts": shard_manifest,
        "shard_receipts_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
    }
    receipt_path = root / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.commit()
    return {
        "method": method,
        "run_id": run_id,
        "status": "completed",
        "behaviors": len(rows),
        "cases": len(rows),
        "target_calls": total_calls,
        "test_cases_sha256": receipt["test_cases_sha256"],
        "output_volume": OUTPUT_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(method: str, run_id: str, num_shards: int = 4, limit: int = 0) -> None:
    import json

    _validate(method, run_id, 0, num_shards, limit)
    calls = []
    for shard_index in range(num_shards):
        call = run_shard.spawn(method, run_id, shard_index, num_shards, limit)
        calls.append({"shard_index": shard_index, "function_call_id": call.object_id})
    print(json.dumps({"method": method, "run_id": run_id, "calls": calls}, sort_keys=True))
