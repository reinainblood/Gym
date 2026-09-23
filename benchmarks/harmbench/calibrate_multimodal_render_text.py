# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare deterministic render-text images with the pinned upstream method.

The upstream module imports optional GPU stacks at top level. This control
executes only its inspected method bodies, with the first-system-font lookup
bound to the declared DejaVu Sans file, then compares every PNG pixel and text
instruction. It does not print prompt or image contents.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib
import torch
import torchvision
from matplotlib import font_manager
from PIL import Image, ImageChops, ImageDraw, ImageFont
from torchvision import transforms

from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import sha256


def calibrate(
    *, upstream: Path, corrected_csv: Path, generated_cases: Path, image_dir: Path, font: Path, output: Path
) -> dict:
    upstream = upstream.resolve(strict=True)
    corrected_csv = corrected_csv.resolve(strict=True)
    generated_cases = generated_cases.resolve(strict=True)
    image_dir = image_dir.resolve(strict=True)
    font = font.resolve(strict=True)
    output = output.resolve()
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    source = upstream / "baselines/multimodalrendertext/multimodalrendertext.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    upstream_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiModalRenderText"
    )
    names = {"__init__", "estimate_char_capacity", "draw_text_in_roi", "generate_test_cases_single_behavior"}
    methods = [node for node in upstream_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise ValueError("upstream render-text method structure changed")
    isolated = ast.ClassDef(name="RenderTextControl", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
    namespace = {
        "json": json,
        "os": os,
        "font_manager": font_manager,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "transforms": transforms,
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102
    config = json.loads(generated_cases.read_text(encoding="utf-8"))
    with corrected_csv.open(newline="", encoding="utf-8") as handle:
        behaviors = list(csv.DictReader(handle))
    control = namespace["RenderTextControl"](
        targets_path=str(upstream / "data/optimizer_targets/harmbench_targets_multimodal.json"),
        image_width=224,
        image_height=224,
        roi_x=0,
        roi_y=0,
        roi_width=224,
        roi_height=224,
        test_cases_batch_size=1,
        num_test_cases_per_behavior=1,
    )
    original_font_lookup = font_manager.findSystemFonts
    original_cwd = Path.cwd()
    pixel_mismatches = []
    instruction_mismatches = []
    try:
        font_manager.findSystemFonts = lambda fontpaths=None, fontext="ttf": [str(font)]
        os.chdir(upstream)
        for behavior in behaviors:
            behavior_id = behavior["BehaviorID"]
            test_cases, _ = control.generate_test_cases_single_behavior(behavior, num_generate=1)
            tensor, instruction = test_cases[0]
            expected = transforms.ToPILImage()(tensor)
            with Image.open(image_dir / f"{behavior_id}.png") as image_file:
                actual = image_file.convert("RGB")
            if actual.size != expected.size or ImageChops.difference(expected, actual).getbbox() is not None:
                pixel_mismatches.append(behavior_id)
            if instruction != config[behavior_id][0][1]:
                instruction_mismatches.append(behavior_id)
    finally:
        font_manager.findSystemFonts = original_font_lookup
        os.chdir(original_cwd)
    receipt = {
        "upstream_revision": UPSTREAM_REVISION,
        "method_source_sha256": sha256(source),
        "corrected_csv_sha256": sha256(corrected_csv),
        "generated_cases_sha256": sha256(generated_cases),
        "font_sha256": sha256(font),
        "checked": len(behaviors),
        "pixel_identical": len(behaviors) - len(pixel_mismatches),
        "instruction_identical": len(behaviors) - len(instruction_mismatches),
        "pixel_mismatches": pixel_mismatches,
        "instruction_mismatches": instruction_mismatches,
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Compared {len(behaviors)} rendered image cases; {len(pixel_mismatches)} pixel and {len(instruction_mismatches)} instruction mismatches"
    )
    if pixel_mismatches or instruction_mismatches:
        raise ValueError("render-text port differs from the pinned upstream method")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--corrected-csv", required=True, type=Path)
    parser.add_argument("--generated-cases", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibrate(
        upstream=args.upstream,
        corrected_csv=args.corrected_csv,
        generated_cases=args.generated_cases,
        image_dir=args.image_dir,
        font=args.font,
        output=args.output,
    )
