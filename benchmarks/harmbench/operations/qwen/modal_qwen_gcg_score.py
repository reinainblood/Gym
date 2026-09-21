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
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
LOCAL = Path(__file__).parent
ROOT = LOCAL.parents[3]

results = modal.Volume.from_name(RESULTS_VOLUME, environment_name="FDR", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"PYTHONPATH": "/app:/app/Gym"})
    .uv_pip_install(
        "transformers==5.1.0",
        "tokenizers>=0.22.2",
        "huggingface-hub>=0.35.3",
        "sentencepiece>=0.2.1",
        "datasketch==2.0.0",
        "spacy==3.7.2",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    )
    .run_commands(
        "git init /app/HarmBench && "
        "git -C /app/HarmBench remote add origin https://github.com/centerforaisafety/HarmBench.git && "
        f"git -C /app/HarmBench fetch --depth 1 origin {UPSTREAM_REVISION} && "
        "git -C /app/HarmBench checkout --detach FETCH_HEAD"
    )
    .add_local_file(LOCAL / "qwen_gcg_score.py", "/app/qwen_gcg_score.py")
    .add_local_file(LOCAL / "qwen_label_compare.py", "/app/qwen_label_compare.py")
    .add_local_file(ROOT / "resources_servers" / "harmbench" / "prompts.py", "/app/harmbench_prompts.py")
    .add_local_file(
        ROOT / "resources_servers" / "harmbench" / "__init__.py", "/app/Gym/resources_servers/harmbench/__init__.py"
    )
    .add_local_file(
        ROOT / "resources_servers" / "harmbench" / "copyright.py", "/app/Gym/resources_servers/harmbench/copyright.py"
    )
    .add_local_file(ROOT / "benchmarks" / "harmbench" / "__init__.py", "/app/Gym/benchmarks/harmbench/__init__.py")
    .add_local_file(ROOT / "benchmarks" / "harmbench" / "prepare.py", "/app/Gym/benchmarks/harmbench/prepare.py")
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

    from resources_servers.harmbench.copyright import CopyrightScorer

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
    copyright = CopyrightScorer(Path("/app/HarmBench/data/copyright_classifier_hashes"))

    def score_copyright(behavior_id: str, generation: str, tags: list[str]) -> dict:
        result = copyright.score(behavior_id, generation, tags)
        return {
            "label": result.label,
            "max_similarity": result.max_similarity,
            "reference_sha256": result.reference_sha256,
        }

    receipt = score_completion_artifact(
        completion_root=completion_root,
        score=classifier.score,
        copyright_score=score_copyright,
    )
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results.commit()
    return {"artifact_id": artifact_id, "status": receipt["status"], "summary": receipt["summary"]}


@app.local_entrypoint()
def main(artifact_id: str) -> None:
    print(json.dumps(score.remote(artifact_id), indent=2, sort_keys=True))
