# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every Gym MultiModalDirectRequest image with upstream Torchvision pixels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import PIL
import torch
import torchvision
from PIL import Image, ImageChops
from torchvision import transforms

from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import sha256


def upstream_render(source: Path, *, width: int, height: int) -> Image.Image:
    """The pinned MultiModalDirectRequest transformation, including tensor round trip."""
    with Image.open(source) as image_file:
        image = image_file.convert("RGB")
    original_width, original_height = image.size
    aspect_ratio = min(width / original_width, height / original_height)
    new_width = int(original_width * aspect_ratio)
    new_height = int(original_height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) / 2
    top = (new_height - height) / 2
    right = (new_width + width) / 2
    bottom = (new_height + height) / 2
    image = image.crop((left, top, right, bottom))
    return transforms.ToPILImage()(transforms.ToTensor()(image))


def calibrate(*, upstream: Path, corrected_csv: Path, prepared_images: Path, output: Path) -> dict:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    with corrected_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    pixel_digests = []
    for row in rows:
        behavior_id = row["BehaviorID"]
        expected = upstream_render(
            upstream / "data/multimodal_behavior_images" / row["ImageFileName"], width=224, height=224
        )
        with Image.open(prepared_images / f"{behavior_id}.png") as image_file:
            actual = image_file.convert("RGB")
        if actual.size != (224, 224) or ImageChops.difference(expected, actual).getbbox() is not None:
            mismatches.append(behavior_id)
        pixel_digests.append(hashlib.sha256(actual.tobytes()).hexdigest())
    receipt = {
        "upstream_revision": UPSTREAM_REVISION,
        "corrected_csv_sha256": sha256(corrected_csv),
        "checked": len(rows),
        "pixel_identical": len(rows) - len(mismatches),
        "mismatches": mismatches,
        "pixel_digest_set_sha256": hashlib.sha256("\n".join(pixel_digests).encode()).hexdigest(),
        "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Checked {len(rows)} multimodal images; {len(mismatches)} pixel mismatches")
    if mismatches:
        raise ValueError(f"Gym images differ from upstream Torchvision for {len(mismatches)} behaviors")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--corrected-csv", required=True, type=Path)
    parser.add_argument("--prepared-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibrate(
        upstream=args.upstream,
        corrected_csv=args.corrected_csv,
        prepared_images=args.prepared_images,
        output=args.output,
    )
