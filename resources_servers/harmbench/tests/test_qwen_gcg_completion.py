# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json

import pytest

from benchmarks.harmbench.operations.qwen.qwen_gcg_completion_worker import (
    EXPERIMENT,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PUBLIC_BEHAVIORS,
    PUBLIC_SEARCH_WIDTH,
    PUBLIC_STEPS,
    UPSTREAM_REVISION,
    finalize_completion_artifact,
    generation_paths,
    load_generation_artifact,
    render_upstream_prompt,
    sha256,
    shard_indexes,
    write_completion_shard_receipt,
)


def _artifact(tmp_path):
    artifact_id = "qwen-gcg-full-test"
    output_root = tmp_path / artifact_id
    behaviors_path = tmp_path / "behaviors.csv"
    with behaviors_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "BehaviorID",
                "Behavior",
                "ContextString",
                "FunctionalCategory",
                "SemanticCategory",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "BehaviorID": f"b{index:03d}",
                "Behavior": "synthetic behavior",
                "ContextString": "",
                "FunctionalCategory": "standard",
                "SemanticCategory": "synthetic",
            }
            for index in range(PUBLIC_BEHAVIORS)
        )
    cases_path, receipt_path = generation_paths(output_root)
    cases_path.parent.mkdir(parents=True)
    cases = {f"b{index:03d}": [f"synthetic case {index}"] for index in range(PUBLIC_BEHAVIORS)}
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    shard_dir = output_root / "shard-receipts"
    shard_dir.mkdir()
    shard_path = shard_dir / "qwen-00-of-01.json"
    shard_path.write_text('{"status":"completed"}', encoding="utf-8")
    shard_manifest = [{"name": shard_path.name, "sha256": sha256(shard_path)}]
    receipt = {
        "status": "completed",
        "method": "GCG",
        "upstream_method": "GCG",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": EXPERIMENT,
        "run_id": artifact_id,
        "target_type": "text_weights",
        "source_target_model": MODEL_ID,
        "source_target_revision": MODEL_REVISION,
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behaviors_path),
        "behaviors": PUBLIC_BEHAVIORS,
        "cases": PUBLIC_BEHAVIORS,
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
        "num_shards": 1,
        "shard_receipts": shard_manifest,
        "shard_receipts_sha256": hashlib.sha256(
            json.dumps(shard_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return artifact_id, output_root, behaviors_path, cases_path, receipt_path


def test_load_generation_artifact_requires_exact_full_qwen_source(tmp_path):
    artifact_id, output_root, behaviors_path, cases_path, receipt_path = _artifact(tmp_path)
    behaviors, cases, receipt = load_generation_artifact(
        output_root=output_root,
        behaviors_path=behaviors_path,
        artifact_id=artifact_id,
    )
    assert len(behaviors) == len(cases) == PUBLIC_BEHAVIORS
    assert receipt["test_cases_sha256"] == sha256(cases_path)

    receipt["source_target_revision"] = "wrong"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source_target_revision"):
        load_generation_artifact(output_root=output_root, behaviors_path=behaviors_path, artifact_id=artifact_id)


def test_load_generation_artifact_rehashes_attack_shard_receipts(tmp_path):
    artifact_id, output_root, behaviors_path, _, _ = _artifact(tmp_path)
    (output_root / "shard-receipts" / "qwen-00-of-01.json").write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="readback failed"):
        load_generation_artifact(output_root=output_root, behaviors_path=behaviors_path, artifact_id=artifact_id)


def test_qwen_completion_shards_cover_source_order_and_require_receipts(tmp_path):
    artifact_id, output_root, behaviors_path, _, receipt_path = _artifact(tmp_path)
    first, second = shard_indexes(0, 2), shard_indexes(1, 2)
    assert first[:3] == [0, 2, 4]
    assert second[:3] == [1, 3, 5]
    assert sorted(first + second) == list(range(PUBLIC_BEHAVIORS))
    completion_dir = output_root / "target-completions" / "qwen-bf16" / "individual"
    completion_dir.mkdir(parents=True)
    generation_receipt_sha256 = sha256(receipt_path)
    for index in range(PUBLIC_BEHAVIORS):
        completion = {
            "schema_version": 1,
            "artifact_kind": "gcg_target_completion",
            "status": "completed",
            "artifact_id": artifact_id,
            "method": "GCG",
            "index": index,
            "behavior_id": f"b{index:03d}",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "generation_receipt_sha256": generation_receipt_sha256,
            "attack_sha256": hashlib.sha256(f"synthetic case {index}".encode()).hexdigest(),
            "rendered_prompt_sha256": "d" * 64,
            "max_new_tokens": MAX_NEW_TOKENS,
            "checkpoint_config_sha256": "a" * 64,
            "checkpoint_index_sha256": "b" * 64,
            "generation": "synthetic generation",
            "generation_sha256": hashlib.sha256(b"synthetic generation").hexdigest(),
        }
        (completion_dir / f"{index:03d}.json").write_text(json.dumps(completion), encoding="utf-8")
    receipt = write_completion_shard_receipt(
        output_root=output_root,
        artifact_id=artifact_id,
        shard_index=0,
        num_shards=2,
        selected=first,
        behaviors_path=behaviors_path,
    )
    assert receipt["completed_behaviors"] == len(first)
    assert receipt["max_new_tokens"] == MAX_NEW_TOKENS
    second_receipt = write_completion_shard_receipt(
        output_root=output_root,
        artifact_id=artifact_id,
        shard_index=1,
        num_shards=2,
        selected=second,
        behaviors_path=behaviors_path,
    )
    assert second_receipt["individual_receipts"][0]["name"] == "001.json"
    final = finalize_completion_artifact(
        output_root=output_root,
        behaviors_path=behaviors_path,
        artifact_id=artifact_id,
    )
    assert final["completions"] == PUBLIC_BEHAVIORS
    assert len(final["individual_receipts"]) == PUBLIC_BEHAVIORS

    (completion_dir / "001.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="failed fields"):
        finalize_completion_artifact(
            output_root=output_root,
            behaviors_path=behaviors_path,
            artifact_id=artifact_id,
        )


def test_qwen_completion_prompt_matches_upstream_default_template_branch():
    class Tokenizer:
        bos_token = "<bos>"

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "{instruction}"}]
            assert tokenize is False and add_generation_prompt is True
            return "<bos><user>{instruction}</user><assistant>"

    assert render_upstream_prompt(Tokenizer(), "synthetic input") == "<user>synthetic input</user><assistant>"
