# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare upstream-equivalent MultiModalDirectRequest images for an API VLM.

HarmBench's gpt4v experiment uses 224x224 images. This CPU-only port mirrors
its RGB/Lanczos/center-crop path; a cross-runtime pixel-parity control against
Torchvision is still required before calling this a calibrated paper result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from PIL import Image

from benchmarks.harmbench.prepare import DATA_DIR, UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import materialize, sha256
from benchmarks.harmbench.prepare_multimodal_source import prepare as prepare_source


OUTPUT = DATA_DIR / "multimodal_direct_request_gpt4v.jsonl"
SOURCE = DATA_DIR / "harmbench_behaviors_multimodal_corrected.csv"
GENERATED = DATA_DIR / "multimodal_direct_request_gpt4v"


def _image(original: Path, output: Path, *, width: int, height: int) -> None:
    with Image.open(original) as source:
        image = source.convert("RGB")
    original_width, original_height = image.size
    aspect_ratio = min(width / original_width, height / original_height)
    new_width = int(original_width * aspect_ratio)
    new_height = int(original_height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) / 2
    top = (new_height - height) / 2
    image = image.crop((left, top, left + width, top + height))
    if image.size != (width, height):
        raise ValueError(f"image crop has wrong size for {original.name}")
    image.save(output, format="PNG")


def prepare(upstream_dir: str | None = None) -> Path:
    upstream_value = upstream_dir or os.environ.get("HARMBENCH_UPSTREAM_DIR")
    if not upstream_value:
        raise ValueError("set HARMBENCH_UPSTREAM_DIR to the pinned HarmBench checkout")
    upstream = Path(upstream_value).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    prepare_source(upstream, SOURCE)
    config_path = upstream / "configs/method_configs/MultiModalDirectRequest_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["gpt4v"]
    width, height = config["image_width"], config["image_height"]
    if (width, height) != (224, 224):
        raise ValueError("pinned gpt4v image profile changed")
    targets = json.loads((upstream / "data/optimizer_targets/harmbench_targets_multimodal.json").read_text())
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    images_dir = GENERATED / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    test_cases: dict[str, list[list[str]]] = {}
    for row in rows:
        behavior_id = row["BehaviorID"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", behavior_id) or behavior_id not in targets:
            raise ValueError(f"invalid or untargeted multimodal behavior {behavior_id}")
        filename = f"{behavior_id}.png"
        _image(
            upstream / "data/multimodal_behavior_images" / row["ImageFileName"],
            images_dir / filename,
            width=width,
            height=height,
        )
        test_cases[behavior_id] = [[filename, row["Behavior"] + " "]]
    if len(test_cases) != 110:
        raise ValueError(f"expected 110 multimodal cases, found {len(test_cases)}")
    cases_path = GENERATED / "test_cases.json"
    cases_path.write_text(json.dumps(test_cases, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "method": "MultiModalDirectRequest",
        "upstream_method": "MultiModalDirectRequest",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": "gpt4v",
        "run_id": "gym-native-224",
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(SOURCE),
        "method_config_sha256": sha256(config_path),
        "behavior_corrections_sha256": sha256(SOURCE.with_suffix(".corrections.json")),
        "image_profile": {"width": width, "height": height, "resampling": "Lanczos", "crop": "center"},
        "generator": "Gym-native CPU port of upstream MultiModalDirectRequest/gpt4v",
    }
    GENERATED.joinpath("generation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    materialized, manifest = materialize(
        method_name="MultiModalDirectRequest",
        test_cases_path=cases_path,
        behaviors_path=SOURCE,
        experiment="gpt4v",
        run_id="gym-native-224",
        target_type="vision_api",
        generation_receipt=receipt,
        image_dir=images_dir,
    )
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in materialized), encoding="utf-8"
    )
    manifest["gym_inputs_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(materialized)} MultiModalDirectRequest/gpt4v image cases at {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    prepare()
