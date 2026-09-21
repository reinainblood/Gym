# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen PGD completion-label comparison inside Modal FDR."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-qwen-label-compare"
RESULTS_VOLUME = "harmbench-qwen35-122b-whitebox-results"
CLASSIFIER_URL = "https://snorkelai-fdr--harmbench-classifier-serve.modal.run"
LOCAL = Path(__file__).parent
METHODS = ("MultiModalPGD", "MultiModalPGDPatch", "MultiModalPGDBlankImage")

results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=False)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/app"})
    .uv_pip_install(
        "transformers==5.1.0",
        "tokenizers>=0.22.2",
        "huggingface-hub>=0.35.3",
        "sentencepiece>=0.2.1",
    )
)
if modal.is_local():
    prompts = LOCAL.parents[3] / "resources_servers" / "harmbench" / "prompts.py"
    image = base_image.add_local_file(LOCAL / "qwen_label_compare.py", "/app/qwen_label_compare.py").add_local_file(
        prompts, "/app/harmbench_prompts.py"
    )
else:
    image = base_image
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/results": results},
    secrets=[modal.Secret.from_name("harmbench-classifier-key", environment_name="FDR")],
)
def compare(run_id: str, method: str = "MultiModalPGD", replicate: str = "r1") -> dict:
    """Score all matched receipts twice and persist one payload-free comparison receipt."""
    from qwen_label_compare import RawClassifier, compare_receipts

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run id")
    if method != "MultiModalPGD":
        raise ValueError("this comparison is intentionally scoped to completed MultiModalPGD")
    if not re.fullmatch(r"[a-z0-9-]+", replicate):
        raise ValueError("invalid replicate label")

    method_dir = Path("/results") / run_id / method
    output_dir = method_dir / "label-comparison"
    output_path = output_dir / f"parent-vs-512-vs-{replicate}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return {"status": "already_complete", "output": str(output_path), "summary": existing["summary"]}

    classifier = RawClassifier(CLASSIFIER_URL, os.environ["HARMBENCH_CLASSIFIER_API_KEY"])
    receipt = compare_receipts(
        parent_dir=method_dir / "cases",
        canonical_dir=method_dir / "completions-512",
        replicate_dir=method_dir / f"completions-512-replicate-{replicate}",
        score=classifier.score,
        target_token_count=classifier.target_token_count,
        repeats=2,
    )
    receipt.update({"run_id": run_id, "method": method, "replicate": replicate})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    results.commit()
    return {"status": "completed", "output": str(output_path), "summary": receipt["summary"]}


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/results": results},
    secrets=[modal.Secret.from_name("harmbench-classifier-key", environment_name="FDR")],
)
def score_canonical(run_id: str, method: str) -> dict:
    """Score one complete 110-case 512-token white-box cohort and persist payload-free evidence."""
    from qwen_label_compare import RawClassifier, score_canonical_receipts

    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("invalid run id")
    if method not in METHODS:
        raise ValueError(f"unknown public method: {method}")

    method_dir = Path("/results") / run_id / method
    output_dir = method_dir / "label-comparison"
    output_path = output_dir / "canonical-512.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return {"status": "already_complete", "output": str(output_path), "summary": existing["summary"]}

    classifier = RawClassifier(CLASSIFIER_URL, os.environ["HARMBENCH_CLASSIFIER_API_KEY"])
    receipt = score_canonical_receipts(
        canonical_dir=method_dir / "completions-512",
        score=classifier.score,
        method=method,
        repeats=2,
        expected_cases=110,
    )
    receipt.update({"run_id": run_id})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    results.commit()
    return {"status": "completed", "output": str(output_path), "summary": receipt["summary"]}


@app.local_entrypoint()
def main(
    run_id: str = "qwen35-122b-harmbench-whitebox-20260919",
    method: str = "MultiModalPGD",
    replicate: str = "r1",
    canonical_only: bool = False,
) -> None:
    """Run comparison or canonical scoring and print only its payload-free summary."""
    result = score_canonical.remote(run_id, method) if canonical_only else compare.remote(run_id, method, replicate)
    print(json.dumps(result, indent=2, sort_keys=True))
