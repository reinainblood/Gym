# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score finalized exact-BF16 Qwen GCG completions with the pinned classifier in FDR."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal


APP_NAME = "harmbench-qwen-gcg-score"
RESULTS_VOLUME = "harmbench-gcg-super-qwen-results"
CLASSIFIER_URL = "https://snorkelai-fdr--harmbench-classifier-serve.modal.run"
LOCAL = Path(__file__).parent

results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/app"})
    .uv_pip_install(
        "transformers==5.1.0",
        "tokenizers>=0.22.2",
        "huggingface-hub>=0.35.3",
        "sentencepiece>=0.2.1",
    )
    .add_local_file(LOCAL / "qwen_gcg_score.py", "/app/qwen_gcg_score.py")
    .add_local_file(LOCAL / "qwen_label_compare.py", "/app/qwen_label_compare.py")
    .add_local_file(LOCAL.parents[3] / "resources_servers" / "harmbench" / "prompts.py", "/app/harmbench_prompts.py")
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=7200,
    volumes={"/outputs": results},
    secrets=[modal.Secret.from_name("harmbench-classifier-key", environment_name="FDR")],
)
def score(artifact_id: str) -> dict:
    from qwen_gcg_score import score_completion_artifact
    from qwen_label_compare import RawClassifier, sha256

    if not re.fullmatch(r"[a-z0-9-]+", artifact_id):
        raise ValueError("invalid artifact id")
    completion_root = Path("/outputs") / artifact_id / "target-completions" / "qwen-bf16"
    output_path = completion_root / "classifier-scores.json"
    completion_manifest = completion_root / "target-completion-receipt.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "completed"
            or existing.get("artifact_id") != artifact_id
            or existing.get("target_completion_manifest_sha256") != sha256(completion_manifest)
        ):
            raise ValueError("existing Qwen GCG classifier receipt is stale or invalid")
        return {"artifact_id": artifact_id, "status": "already_complete", "summary": existing["summary"]}

    classifier = RawClassifier(CLASSIFIER_URL, os.environ["HARMBENCH_CLASSIFIER_API_KEY"])
    receipt = score_completion_artifact(completion_root=completion_root, score=classifier.score)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results.commit()
    return {"artifact_id": artifact_id, "status": receipt["status"], "summary": receipt["summary"]}


@app.local_entrypoint()
def main(artifact_id: str) -> None:
    print(json.dumps(score.remote(artifact_id), indent=2, sort_keys=True))
