# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and verify the aligned Nemotron 3 Ultra BF16 checkpoint in FDR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"
VOLUME_NAME = "nemotron-3-ultra-550b-a55b-bf16-77df655d"
MOUNT = Path("/checkpoint")

app = modal.App("harmbench-ultra-whitebox-bf16-checkpoint")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=43_200, cpu=8, memory=32_768)
def download_and_verify() -> dict[str, object]:
    from huggingface_hub import HfApi, snapshot_download

    model_dir = MOUNT / "model"
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    hub_files = []
    for sibling in sorted(info.siblings or [], key=lambda item: item.rfilename):
        lfs = sibling.lfs
        hub_files.append(
            {
                "path": sibling.rfilename,
                "size": sibling.size,
                "lfs_sha256": lfs.sha256 if lfs is not None else None,
                "blob_id": sibling.blob_id,
            }
        )

    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=model_dir, max_workers=16)
    verified_files = []
    for expected in hub_files:
        path = model_dir / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if expected["size"] is not None and size != expected["size"]:
            raise ValueError(f"size mismatch for {expected['path']}: {size} != {expected['size']}")
        digest = _sha256(path)
        if expected["lfs_sha256"] is not None and digest != expected["lfs_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {expected['path']}")
        verified_files.append(expected | {"local_sha256": digest})

    receipt = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": info.sha,
        "private": info.private,
        "gated": info.gated,
        "volume_name": VOLUME_NAME,
        "model_path": str(model_dir),
        "file_count": len(verified_files),
        "total_bytes": sum(item["size"] or 0 for item in verified_files),
        "files": verified_files,
    }
    receipt_path = MOUNT / "checkpoint-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "volume_name": VOLUME_NAME,
        "file_count": receipt["file_count"],
        "total_bytes": receipt["total_bytes"],
        "manifest_sha256": _sha256(receipt_path),
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(download_and_verify.remote(), sort_keys=True))
