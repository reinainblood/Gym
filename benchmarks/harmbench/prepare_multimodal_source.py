# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair pinned HarmBench multimodal CSV image extensions with an audit log.

Seven rows in the upstream CSV request JPEG filenames, while their checked-in
images at the same commit are PNGs with identical stems. This changes only the
filename extension used to locate those already-pinned assets; every change is
recorded with the source and image digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from benchmarks.harmbench.prepare import UPSTREAM_REVISION


SOURCE_SHA256 = "0a4dd6bd0df2a52d1d9ee25f3d018af49b52b61ee2ccdaaac16b9660f753145d"
EXPECTED_BEHAVIORS = 110
EXPECTED_EXTENSION_FIXES = 7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected_rows(rows: list[dict[str, str]], image_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    output: list[dict[str, str]] = []
    corrections: list[dict[str, str]] = []
    for original in rows:
        row = original.copy()
        filename = row["ImageFileName"]
        if Path(filename).name != filename:
            raise ValueError(f"invalid upstream image filename for {row['BehaviorID']}")
        image = image_dir / filename
        if not image.is_file():
            replacement = image_dir / f"{Path(filename).stem}.png"
            if not replacement.is_file() or replacement.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise FileNotFoundError(f"no valid pinned image for {row['BehaviorID']}: {filename}")
            row["ImageFileName"] = replacement.name
            corrections.append(
                {
                    "behavior_id": row["BehaviorID"],
                    "source_filename": filename,
                    "resolved_filename": replacement.name,
                    "image_sha256": _sha256(replacement),
                }
            )
        output.append(row)
    return output, corrections


def prepare(upstream: Path, output: Path) -> Path:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    source = upstream / "data/behavior_datasets/harmbench_behaviors_multimodal_all.csv"
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("pinned upstream multimodal CSV digest changed")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(rows) != EXPECTED_BEHAVIORS:
        raise ValueError(f"expected {EXPECTED_BEHAVIORS} pinned multimodal behaviors")
    corrected, corrections = corrected_rows(rows, upstream / "data/multimodal_behavior_images")
    if len(corrections) != EXPECTED_EXTENSION_FIXES:
        raise ValueError(f"expected {EXPECTED_EXTENSION_FIXES} upstream filename fixes, found {len(corrections)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corrected)
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "source_csv_sha256": SOURCE_SHA256,
        "corrected_csv_sha256": _sha256(output),
        "behaviors": len(corrected),
        "corrections": corrections,
    }
    output.with_suffix(".corrections.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(corrected)} multimodal behaviors with {len(corrections)} recorded filename corrections")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.upstream, args.output)
