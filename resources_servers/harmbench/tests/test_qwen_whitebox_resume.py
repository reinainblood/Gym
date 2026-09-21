# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harmbench.operations.qwen.qwen_harmbench_worker import (
    CORRECTED_BEHAVIORS_SHA256,
    CORRECTIONS_SHA256,
    METHODS,
    MODEL_ID,
    MODEL_REVISION,
    UPSTREAM_REVISION,
    sha256,
    validate_case_receipt,
)


def _case(tmp_path, method="MultiModalPGDBlankImage"):
    index = 7
    behavior_id = "synthetic_behavior"
    run_dir = tmp_path / method
    image = run_dir / "images" / f"{index:03d}-{behavior_id}.png"
    checkpoint = run_dir / "optimization-checkpoints" / f"{index:03d}-{behavior_id}" / "step-0000.png"
    case = run_dir / "cases" / f"{index:03d}-{behavior_id}.json"
    image.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    case.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    checkpoint.write_bytes(b"checkpoint")
    row = {
        "BehaviorID": behavior_id,
        "Behavior": "synthetic behavior",
        "ContextString": "",
        "FunctionalCategory": "multimodal",
        "SemanticCategory": "synthetic",
        "ImageFileName": "synthetic.png",
    }
    generation = "synthetic generation"
    steps = METHODS[method]["num_steps"]
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": method,
        "public_method_class": "MultiModalPGDPatch" if method == "MultiModalPGDPatch" else "MultiModalPGD",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "corrected_behaviors_sha256": CORRECTED_BEHAVIORS_SHA256,
        "filename_corrections_sha256": CORRECTIONS_SHA256,
        "index": index,
        "behavior_id": behavior_id,
        "behavior": row["Behavior"],
        "context": "",
        "functional_category": "multimodal",
        "semantic_category": "synthetic",
        "image_file_name": "synthetic.png",
        "target_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "test_case_image": str(image),
        "test_case_image_sha256": sha256(image),
        "generation": generation,
        "generation_sha256": hashlib.sha256(generation.encode()).hexdigest(),
        "hyperparameters": METHODS[method],
        "seed": index,
        "patch": {"start_x": 0, "start_y": 0, "width": 1, "height": 1} if method == "MultiModalPGDPatch" else None,
        "processor_pixel_parity_max_abs_error": 0.0,
        "target_token_count": 1,
        "steps_executed": steps,
        "stopped_early": False,
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "minimum_loss": 0.5,
        "all_losses": [1.0, 0.5],
        "optimization_checkpoints": [{"step": 0, "path": str(checkpoint)}],
        "elapsed_seconds": 1.0,
    }
    case.write_text(json.dumps(receipt), encoding="utf-8")
    return case, row, image


@pytest.mark.parametrize("method", ["MultiModalPGDBlankImage", "MultiModalPGDPatch"])
def test_whitebox_resume_accepts_only_complete_matching_case_artifacts(tmp_path, method):
    case, row, _ = _case(tmp_path, method)
    receipt = validate_case_receipt(case, method=method, index=7, row=row)
    assert receipt["status"] == "completed"


def test_whitebox_resume_rejects_missing_or_tampered_image(tmp_path):
    case, row, image = _case(tmp_path)
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="test_case_image_sha256"):
        validate_case_receipt(case, method="MultiModalPGDBlankImage", index=7, row=row)
    image.unlink()
    with pytest.raises(ValueError, match="test_case_image"):
        validate_case_receipt(case, method="MultiModalPGDBlankImage", index=7, row=row)


def test_whitebox_resume_rejects_wrong_method_or_incomplete_checkpoint(tmp_path):
    case, row, _ = _case(tmp_path)
    receipt = json.loads(case.read_text())
    receipt["method"] = "MultiModalPGD"
    case.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="failed fields method"):
        validate_case_receipt(case, method="MultiModalPGDBlankImage", index=7, row=row)

    case, row, _ = _case(tmp_path)
    checkpoint = Path(json.loads(case.read_text())["optimization_checkpoints"][0]["path"])
    checkpoint.unlink()
    with pytest.raises(ValueError, match="optimization_checkpoints"):
        validate_case_receipt(case, method="MultiModalPGDBlankImage", index=7, row=row)
