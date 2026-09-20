# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize pinned upstream HarmBench test cases as NeMo Gym benchmark rows.

This does *not* generate attacks.  The input must be the test_cases.json written
by HarmBench's actual method implementation.  Keeping generation separate makes
its model requirements and provenance visible instead of silently substituting
an unrelated prompt list.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from benchmarks.harmbench.methods import get_method
from benchmarks.harmbench.prepare import UPSTREAM_REVISION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_content(test_case: Any, image_dir: Path | None) -> tuple[str | list[dict[str, str]], str | None]:
    if isinstance(test_case, str):
        if not test_case.strip():
            raise ValueError("upstream produced an empty text test case")
        return test_case, None
    if not isinstance(test_case, list) or len(test_case) != 2:
        raise ValueError("upstream test case must be text or [image filename, text]")
    image_name, prompt = test_case
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise ValueError("multimodal image must be a filename inside the generated images directory")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("multimodal test case requires non-empty text")
    if image_dir is None:
        raise ValueError("--image-dir is required for multimodal test cases")
    image_path = image_dir / image_name
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    mime, _ = mimetypes.guess_type(image_name)
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"unsupported generated image type: {image_name}")
    uri = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    return [
        {"type": "input_image", "image_url": uri, "detail": "high"},
        {"type": "input_text", "text": prompt},
    ], sha256(image_path)


def materialize(
    *,
    method_name: str,
    test_cases_path: Path,
    behaviors_path: Path,
    experiment: str,
    run_id: str,
    target_type: str,
    generation_receipt: dict[str, Any],
    image_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = get_method(method_name)
    if target_type not in method.target_types:
        raise ValueError(f"{method_name} does not support {target_type}; supported: {method.target_types}")
    if not experiment or not run_id:
        raise ValueError("experiment and run_id are required for attack provenance")
    cases_sha256 = sha256(test_cases_path)
    behavior_sha256 = sha256(behaviors_path)
    required_receipt = {
        "method": method.name,
        "upstream_method": method.upstream_key,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": experiment,
        "run_id": run_id,
        "test_cases_sha256": cases_sha256,
        "behaviors_sha256": behavior_sha256,
    }
    for key, expected in required_receipt.items():
        if generation_receipt.get(key) != expected:
            raise ValueError(f"generation receipt {key} does not match the requested {method.name} source")
    if method.generation == "client_fresh" and (
        not generation_receipt.get("client_model")
        or generation_receipt.get("source_target_model") != generation_receipt.get("client_model")
    ):
        raise ValueError("fresh PAIR/TAP requires proof that generation targeted the client model")
    if method.name in {"TAP-Transfer", "GCG-Transfer"} and not generation_receipt.get("source_experiment"):
        raise ValueError("transfer attacks require their source experiment in the generation receipt")
    with behaviors_path.open(newline="", encoding="utf-8") as source:
        behaviors = {row["BehaviorID"]: row for row in csv.DictReader(source)}
    attack_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    if not isinstance(attack_cases, dict) or not attack_cases:
        raise ValueError("upstream test_cases.json must contain a nonempty behavior-to-cases mapping")
    unknown = set(attack_cases) - set(behaviors)
    if unknown:
        raise ValueError(f"attack cases contain unknown behavior IDs: {sorted(unknown)[:5]}")

    rows: list[dict[str, Any]] = []
    for behavior_id, cases in attack_cases.items():
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{behavior_id} has no generated attack cases")
        behavior = behaviors[behavior_id]
        tags = [tag for tag in behavior["Tags"].split(", ") if tag]
        for index, test_case in enumerate(cases):
            content, image_sha256 = _input_content(test_case, image_dir)
            is_image = image_sha256 is not None
            if is_image != ("multimodal" in tags):
                raise ValueError(f"{behavior_id} modality disagrees with the upstream behavior tags")
            rows.append(
                {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": content}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_output_tokens": 512,
                    },
                    "harmbench_id": behavior_id,
                    "behavior": behavior["Behavior"],
                    "context": behavior.get("ContextString", "") or behavior.get("RedactedImageDescription", ""),
                    "functional_category": behavior["FunctionalCategory"],
                    "semantic_category": behavior["SemanticCategory"],
                    "tags": tags,
                    "method": method.name,
                    "upstream_method": method.upstream_key,
                    "attack_experiment": experiment,
                    "attack_run_id": run_id,
                    "attack_test_case_index": index,
                    "attack_source_sha256": cases_sha256,
                    "attack_image_sha256": image_sha256,
                    "upstream_revision": UPSTREAM_REVISION,
                }
            )
    manifest = {
        "method": method.name,
        "upstream_method": method.upstream_key,
        "generation": method.generation,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": experiment,
        "run_id": run_id,
        "target_type": target_type,
        "test_cases_sha256": cases_sha256,
        "behaviors_sha256": sha256(behaviors_path),
        "generation_receipt": generation_receipt,
        "rows": len(rows),
        "behaviors": len(attack_cases),
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--test-cases", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-receipt", required=True, type=Path)
    parser.add_argument(
        "--target-type", required=True, choices=["text_api", "text_weights", "vision_api", "vision_weights"]
    )
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, manifest = materialize(
        method_name=args.method,
        test_cases_path=args.test_cases,
        behaviors_path=args.behaviors,
        experiment=args.experiment,
        run_id=args.run_id,
        target_type=args.target_type,
        generation_receipt=json.loads(args.generation_receipt.read_text(encoding="utf-8")),
        image_dir=args.image_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve malformed-but-present Unicode in pinned upstream attack strings.
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["gym_inputs_sha256"] = sha256(args.output)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} {args.method} rows from {len({r['harmbench_id'] for r in rows})} behaviors to {args.output}"
    )


if __name__ == "__main__":
    main()
