# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hydrate and verify the exact public Qwen 3.5 122B BF16 checkpoint in FDR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


APP_NAME = "qwen35-122b-bf16-hydrate"
VOLUME_NAME = "qwen3-5-122b-a10b-bf16-dc4d3484"
MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"
volume = modal.Volume.from_name(VOLUME_NAME, environment_name="FDR", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface-hub==1.32.0")
app = modal.App(APP_NAME)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@app.function(image=image, cpu=16, memory=65536, timeout=21600, volumes={"/weights": volume})
def hydrate() -> dict:
    from huggingface_hub import HfApi, snapshot_download

    target = Path("/weights/snapshot")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=target,
        max_workers=16,
    )
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("resolved Qwen BF16 revision mismatch")
    expected = {item.rfilename: item.size for item in info.siblings}
    missing = [name for name in expected if not (target / name).is_file()]
    wrong_size = [
        name
        for name, size in expected.items()
        if size is not None and (target / name).is_file() and (target / name).stat().st_size != size
    ]
    if missing or wrong_size:
        raise ValueError(f"incomplete Qwen BF16 snapshot: missing={len(missing)} wrong_size={len(wrong_size)}")
    index = json.loads((target / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files": len(expected),
        "checkpoint_shards": len(shards),
        "checkpoint_bytes": sum((target / name).stat().st_size for name in shards),
        "config_sha256": digest(target / "config.json"),
        "index_sha256": digest(target / "model.safetensors.index.json"),
        "all_expected_files_present": True,
        "all_expected_sizes_match": True,
    }
    (Path("/weights") / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(hydrate.remote(), sort_keys=True))
