# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FDR-only Kimi K3 white-box gradient and attack runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "harmbench-kimi-k3-whitebox"
SOURCE_VOLUME = "endpoint-ep-WD4gnzaeXMyM7DfrzTDYjW"
OUTPUT_VOLUME = "harmbench-kimi-k3-whitebox-results"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
MODEL_ID = "moonshotai/Kimi-K3"
MODEL_FILE_COUNT = 118
MODEL_TOTAL_BYTES = 1_560_998_984_390
CONFIG_SHA256 = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
INDEX_SHA256 = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
MANIFEST_SHA256 = "98c4642076606cf1dd3f83706c901f867288febed93f45719c21982514b2ba2b"
SNAPSHOT = Path("/checkpoint/huggingface/hub/models--moonshotai--Kimi-K3/snapshots") / REVISION
LOCAL = Path(__file__).parent
UPSTREAM = Path(os.environ.get("HARMBENCH_UPSTREAM", LOCAL / "upstream-harmbench"))

source = modal.Volume.from_name(SOURCE_VOLUME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, environment_name="FDR", create_if_missing=False)
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:kimi-k3",
        setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/bin/python"],
    )
    .entrypoint([])
    .run_commands(
        "uv pip install --system 'fschat==0.2.36' 'ray==2.58.0' 'sentence-transformers==5.2.2' 'nltk==3.9.3'"
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(
        Path(__file__).with_name("kimi_k3_text_gradient.py"),
        "/app/kimi_k3_text_gradient.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("sitecustomize.py"),
        "/usr/local/lib/python3.12/dist-packages/sitecustomize.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("kimi_k3_harmbench_text_runner.py"),
        "/app/kimi_k3_harmbench_text_runner.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("kimi_k3_multimodal_adapter.py"),
        "/app/kimi_k3_multimodal_adapter.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("kimi_k3_harmbench_multimodal_runner.py"),
        "/app/kimi_k3_harmbench_multimodal_runner.py",
        copy=True,
    )
    .add_local_dir(
        UPSTREAM / "baselines",
        "/app/HarmBench/baselines",
        copy=True,
    )
    .add_local_dir(
        UPSTREAM / "configs",
        "/app/HarmBench/configs",
        copy=True,
    )
    .add_local_dir(
        UPSTREAM / "data",
        "/app/HarmBench/data",
        copy=True,
    )
    .add_local_file(
        LOCAL / "upstream-pinned-revision.txt",
        "/app/HarmBench/.pinned-revision",
        copy=True,
    )
)
app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_final_manifest() -> dict[str, object]:
    path = Path("/outputs/checkpoint-manifest.json")
    if _sha256(path) != MANIFEST_SHA256:
        raise ValueError("final checkpoint manifest hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "verified",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": REVISION,
        "source_volume": SOURCE_VOLUME,
        "file_count": MODEL_FILE_COUNT,
        "total_bytes": MODEL_TOTAL_BYTES,
    }
    mismatch = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if mismatch or len(receipt.get("files", [])) != MODEL_FILE_COUNT:
        raise ValueError(f"final checkpoint manifest mismatch: {mismatch}")
    return receipt


@app.function(
    image=image,
    cpu=32,
    memory=344_064,
    timeout=600,
    volumes={
        "/checkpoint": source.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def package_probe() -> dict[str, object]:
    import compressed_tensors
    import fastchat
    import ray
    import torch
    import transformers
    from transformers import AutoConfig

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/HarmBench")
    from kimi_k3_harmbench_text_runner import install_kimi_transformers_compatibility

    install_kimi_transformers_compatibility()
    import vllm
    from baselines import get_method_class

    public_classes = {
        name: get_method_class(name).__name__
        for name in (
            "GCG",
            "EnsembleGCG",
            "AutoPrompt",
            "GBDA",
            "PEZ",
            "UAT",
            "AutoDAN",
            "FewShot",
        )
    }

    config = AutoConfig.from_pretrained(SNAPSHOT, trust_remote_code=True, local_files_only=True)
    result = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "compressed_tensors": str(compressed_tensors.__version__),
        "fastchat": str(fastchat.__version__),
        "ray": str(ray.__version__),
        "vllm": str(vllm.__version__),
        "public_classes": public_classes,
        "architecture": config.architectures,
        "model_type": config.model_type,
        "quant_method": config.quantization_config.get("quant_method"),
        "vision_layers": config.vision_config.vt_num_hidden_layers,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    cpu=32,
    memory=344_064,
    timeout=1_800,
    volumes={
        "/checkpoint": source.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def checkpoint_preflight() -> dict[str, object]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise ValueError("public checkpoint identity or access state changed")
    siblings = sorted(info.siblings or [], key=lambda item: item.rfilename)
    local_total = 0
    for sibling in siblings:
        path = SNAPSHOT / sibling.rfilename
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if sibling.size is not None and size != sibling.size:
            raise ValueError(f"size mismatch for {sibling.rfilename}: {size} != {sibling.size}")
        local_total += size
    if len(siblings) != MODEL_FILE_COUNT or local_total != MODEL_TOTAL_BYTES:
        raise ValueError("checkpoint file-count or byte-total mismatch")
    critical = {
        "config.json": _sha256(SNAPSHOT / "config.json"),
        "model.safetensors.index.json": _sha256(SNAPSHOT / "model.safetensors.index.json"),
    }
    if critical != {
        "config.json": CONFIG_SHA256,
        "model.safetensors.index.json": INDEX_SHA256,
    }:
        raise ValueError("critical checkpoint manifest hash mismatch")
    receipt = {
        "schema_version": 1,
        "status": "preflight_verified",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": info.sha,
        "private": info.private,
        "gated": info.gated,
        "source_volume": SOURCE_VOLUME,
        "snapshot_path": str(SNAPSHOT),
        "file_count": len(siblings),
        "total_bytes": local_total,
        "critical_sha256": critical,
        "full_content_hash_manifest_pending": True,
    }
    receipt_path = Path("/outputs/checkpoint-preflight.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.commit()
    return {**receipt, "receipt_sha256": _sha256(receipt_path)}


@app.function(
    image=image,
    gpu="B300:8",
    cpu=32,
    memory=344_064,
    timeout=14_400,
    startup_timeout=3_600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/checkpoint": source.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def text_gradient(run_id: str) -> dict[str, object]:
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run ID")
    final_manifest = Path("/outputs/checkpoint-manifest.json")
    manifest = final_manifest if final_manifest.is_file() else Path("/outputs/checkpoint-preflight.json")
    receipt = Path("/outputs") / run_id / "text-gradient-receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    subprocess.run(
        [
            "python",
            "/app/kimi_k3_text_gradient.py",
            "--snapshot",
            str(SNAPSHOT),
            "--manifest",
            str(manifest),
            "--output",
            str(receipt),
        ],
        check=True,
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or not value.get("gradient_nonzero"):
        raise ValueError("text gradient receipt failed")
    outputs.commit()
    return {
        "run_id": run_id,
        "status": value["status"],
        "loss": value["loss"],
        "gradient_l2": value["gradient_l2"],
        "output_volume": OUTPUT_VOLUME,
    }


@app.function(
    image=image,
    gpu="B300:8",
    cpu=32,
    memory=344_064,
    timeout=86_400,
    startup_timeout=3_600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/checkpoint": source.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def run_text_method(method_name: str, run_id: str = "") -> dict[str, object]:
    _require_final_manifest()
    sys.path.insert(0, "/app")
    from kimi_k3_harmbench_text_runner import run

    return run(
        upstream=Path("/app/HarmBench"),
        snapshot=SNAPSHOT,
        output_root=Path("/outputs/full-runs"),
        method_name=method_name,
        run_id=run_id or None,
        commit=outputs.commit,
    )


@app.function(
    image=image,
    gpu="B300:8",
    cpu=32,
    memory=344_064,
    timeout=86_400,
    startup_timeout=3_600,
    min_containers=0,
    max_containers=1,
    volumes={
        "/checkpoint": source.with_mount_options(read_only=True),
        "/outputs": outputs,
    },
)
def run_multimodal_method(method_name: str) -> dict[str, object]:
    _require_final_manifest()
    sys.path.insert(0, "/app")
    from kimi_k3_harmbench_multimodal_runner import run

    return run(
        upstream=Path("/app/HarmBench"),
        snapshot=SNAPSHOT,
        output_root=Path("/outputs/full-runs"),
        method_name=method_name,
        commit=outputs.commit,
    )


@app.local_entrypoint()
def main(run_id: str = "", probe: bool = False) -> None:
    if run_id:
        print(json.dumps(text_gradient.remote(run_id), sort_keys=True))
    elif probe:
        print(json.dumps(package_probe.remote(), sort_keys=True))
    else:
        print(json.dumps(checkpoint_preflight.remote(), sort_keys=True))
