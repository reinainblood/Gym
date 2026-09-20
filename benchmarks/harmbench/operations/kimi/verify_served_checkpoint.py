# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the exact Kimi K3 checkpoint already cached by the live FDR endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


MODEL_ID = "moonshotai/Kimi-K3"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
SOURCE_VOLUME = "endpoint-ep-WD4gnzaeXMyM7DfrzTDYjW"
OUTPUT_VOLUME = "harmbench-kimi-k3-whitebox-results"
CACHE_ROOT = Path("/cache")
OUTPUT_ROOT = Path("/outputs")
SNAPSHOT = CACHE_ROOT / "huggingface/hub/models--moonshotai--Kimi-K3/snapshots" / REVISION

app = modal.App("harmbench-kimi-k3-whitebox-checkpoint")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0")
source = modal.Volume.from_name(SOURCE_VOLUME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, environment_name="FDR", create_if_missing=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    timeout=43_200,
    volumes={
        str(CACHE_ROOT): source.with_mount_options(read_only=True),
        str(OUTPUT_ROOT): outputs,
    },
)
def verify() -> dict[str, object]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise ValueError("public checkpoint identity or access state changed")
    verified = []
    for sibling in sorted(info.siblings or [], key=lambda item: item.rfilename):
        path = SNAPSHOT / sibling.rfilename
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if sibling.size is not None and size != sibling.size:
            raise ValueError(f"size mismatch for {sibling.rfilename}: {size} != {sibling.size}")
        digest = _sha256(path)
        lfs_sha = sibling.lfs.sha256 if sibling.lfs is not None else None
        if lfs_sha is not None and digest != lfs_sha:
            raise ValueError(f"SHA-256 mismatch for {sibling.rfilename}")
        verified.append(
            {
                "path": sibling.rfilename,
                "size": sibling.size,
                "lfs_sha256": lfs_sha,
                "blob_id": sibling.blob_id,
                "local_sha256": digest,
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": info.sha,
        "private": info.private,
        "gated": info.gated,
        "source_volume": SOURCE_VOLUME,
        "snapshot_path": str(SNAPSHOT),
        "file_count": len(verified),
        "total_bytes": sum(item["size"] or 0 for item in verified),
        "files": verified,
    }
    receipt_path = OUTPUT_ROOT / "checkpoint-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.commit()
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source_volume": SOURCE_VOLUME,
        "output_volume": OUTPUT_VOLUME,
        "file_count": receipt["file_count"],
        "total_bytes": receipt["total_bytes"],
        "manifest_sha256": _sha256(receipt_path),
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(verify.remote(), sort_keys=True))
