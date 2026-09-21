# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_completion_worker import (
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_MAX_NEW_TOKENS,
    sha256,
    text_sha256,
    validate_completion_receipt,
)


def _receipts(tmp_path):
    image = tmp_path / "images" / "000-case.png"
    parent = tmp_path / "cases" / "000-case.json"
    output = tmp_path / "completions-512" / "000-case.json"
    image.parent.mkdir(parents=True)
    parent.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    parent_receipt = {
        "status": "completed",
        "method": "MultiModalPGDBlankImage",
        "public_method_class": "MultiModalPGD",
        "index": 0,
        "behavior_id": "case",
        "behavior": "synthetic behavior",
        "context": "",
        "functional_category": "multimodal",
        "semantic_category": "synthetic",
        "generation_sha256": "a" * 64,
        "test_case_image": str(image),
        "test_case_image_sha256": sha256(image),
    }
    parent.write_text(json.dumps(parent_receipt), encoding="utf-8")
    generation = "synthetic generation"
    completion = {
        "schema_version": 1,
        "artifact_kind": "target_completion",
        "status": "completed",
        "method": parent_receipt["method"],
        "public_method_class": parent_receipt["public_method_class"],
        "index": 0,
        "behavior_id": "case",
        "behavior": parent_receipt["behavior"],
        "context": "",
        "functional_category": "multimodal",
        "semantic_category": "synthetic",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_config_sha256": "b" * 64,
        "checkpoint_index_sha256": "c" * 64,
        "processor_file_sha256": {"tokenizer.json": "d" * 64},
        "parent_receipt": parent.name,
        "parent_receipt_sha256": sha256(parent),
        "parent_generation_sha256": "a" * 64,
        "parent_max_new_tokens": 256,
        "test_case_image": str(image),
        "test_case_image_sha256": sha256(image),
        "rendered_prompt_sha256": "e" * 64,
        "prompt_token_count": 10,
        "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
        "generation_token_count": 12,
        "finish_reason": "stop",
        "generation": generation,
        "generation_sha256": text_sha256(generation),
        "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
    }
    output.write_text(json.dumps(completion), encoding="utf-8")
    kwargs = {
        "checkpoint_config_sha256": "b" * 64,
        "checkpoint_index_sha256": "c" * 64,
        "processor_file_sha256": {"tokenizer.json": "d" * 64},
        "rendered_prompt_sha256": "e" * 64,
    }
    return output, parent, image, kwargs


def test_canonical_completion_resume_accepts_full_hash_chain(tmp_path):
    output, parent, _, kwargs = _receipts(tmp_path)
    receipt = validate_completion_receipt(output, parent, **kwargs)
    assert receipt["status"] == "completed"


def test_canonical_completion_resume_rejects_parent_or_image_drift(tmp_path):
    output, parent, image, kwargs = _receipts(tmp_path)
    parent_receipt = json.loads(parent.read_text())
    parent_receipt["context"] = "changed"
    parent.write_text(json.dumps(parent_receipt))
    with pytest.raises(ValueError, match="parent_receipt_sha256"):
        validate_completion_receipt(output, parent, **kwargs)

    output, parent, image, kwargs = _receipts(tmp_path / "image-drift")
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="test_case_image_sha256"):
        validate_completion_receipt(output, parent, **kwargs)


def test_canonical_completion_resume_rejects_generation_or_protocol_drift(tmp_path):
    output, parent, _, kwargs = _receipts(tmp_path)
    receipt = json.loads(output.read_text())
    receipt["generation_sha256"] = "0" * 64
    output.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="generation_sha256"):
        validate_completion_receipt(output, parent, **kwargs)

    output, parent, _, kwargs = _receipts(tmp_path / "protocol-drift")
    receipt = json.loads(output.read_text())
    receipt["max_new_tokens"] = 256
    output.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_completion_receipt(output, parent, **kwargs)
