# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import csv
import hashlib
import importlib
import json

import pytest
from PIL import Image

from benchmarks.harmbench.generate_human_jailbreaks import generate
from benchmarks.harmbench.methods import METHODS, get_method
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import materialize, sha256
from benchmarks.harmbench.prepare_human_jailbreaks import BEHAVIORS_SHA256, _require_upstream
from benchmarks.harmbench.prepare_multimodal_direct import _image
from benchmarks.harmbench.prepare_multimodal_source import corrected_rows
from benchmarks.harmbench.select_copyright_canary import select
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


REQUESTED_METHODS = {
    "DirectRequest",
    "HumanJailbreaks",
    "ZeroShot",
    "PAP-top5",
    "TAP-Transfer",
    "MultiModalDirectRequest",
    "MultiModalRenderText",
    "GCG",
    "GCG-Multi",
    "AutoPrompt",
    "GBDA",
    "PEZ",
    "UAT",
    "AutoDAN",
    "FewShot",
    "PAIR",
    "TAP",
    "GCG-Transfer",
    "Fresh PAIR against the client model",
    "Fresh TAP against the client model",
    "MultiModalPGD",
    "MultiModalPGDPatch",
    "MultiModalPGDBlankImage",
}
UPSTREAM = harmbench_upstream()


def test_generated_overlay_preserves_metrics_when_dataset_changes(tmp_path, monkeypatch):
    overlay = importlib.import_module("benchmarks.harmbench_generated.prepare")
    active = tmp_path / "active.jsonl"
    active.write_text('{"previous":true}\n', encoding="utf-8")
    old_hash = sha256(active)
    active.with_suffix(".manifest.json").write_text(json.dumps({"gym_inputs_sha256": old_hash}), encoding="utf-8")
    active.with_name("active_metrics.json").write_text('{"old":"metrics"}\n', encoding="utf-8")
    active.with_name("active_metrics_conflict.json").write_text('{"old":"conflict"}\n', encoding="utf-8")
    monkeypatch.setattr(overlay, "ACTIVE", active)

    overlay._preserve_previous_metrics("f" * 64)

    assert not active.with_name("active_metrics.json").exists()
    assert not active.with_name("active_metrics_conflict.json").exists()
    assert (tmp_path / "metrics-history" / f"{old_hash}-metrics.json").read_text() == '{"old":"metrics"}\n'
    assert (tmp_path / "metrics-history" / f"{old_hash}-conflict.json").read_text() == '{"old":"conflict"}\n'


def _behaviors(path):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "BehaviorID",
                "Behavior",
                "ContextString",
                "FunctionalCategory",
                "SemanticCategory",
                "Tags",
                "RedactedImageDescription",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "BehaviorID": "b1",
                "Behavior": "benign test behavior",
                "ContextString": "test context",
                "FunctionalCategory": "contextual",
                "SemanticCategory": "test",
                "Tags": "context",
            }
        )
        writer.writerow(
            {
                "BehaviorID": "v1",
                "Behavior": "describe the test image",
                "FunctionalCategory": "multimodal",
                "SemanticCategory": "test",
                "Tags": "multimodal",
                "RedactedImageDescription": "redacted description",
            }
        )


def _receipt(method, cases_path, behavior_path, experiment, run_id):
    selected = get_method(method)
    return {
        "method": method,
        "upstream_method": selected.upstream_key,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": experiment,
        "run_id": run_id,
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behavior_path),
    }


def test_catalog_is_exactly_the_requested_method_set():
    assert set(METHODS) == REQUESTED_METHODS
    assert get_method("Fresh PAIR against the client model").upstream_key == "PAIR"
    assert get_method("Fresh TAP against the client model").generation == "client_fresh"
    assert get_method("GCG").target_types == ("text_weights",)
    assert get_method("MultiModalPGD").target_types == ("vision_weights",)
    assert get_method("MultiModalPGDBlankImage").target_types == ("vision_weights",)


def test_materializes_multiple_text_cases_with_source_receipts(tmp_path):
    behavior_path = tmp_path / "behaviors.csv"
    cases_path = tmp_path / "test_cases.json"
    _behaviors(behavior_path)
    cases_path.write_text(json.dumps({"b1": ["case one", "case two"]}), encoding="utf-8")
    rows, manifest = materialize(
        method_name="HumanJailbreaks",
        test_cases_path=cases_path,
        behaviors_path=behavior_path,
        experiment="random_subset_5",
        run_id="run-1",
        target_type="text_api",
        generation_receipt=_receipt("HumanJailbreaks", cases_path, behavior_path, "random_subset_5", "run-1"),
    )
    assert len(rows) == 2
    assert [row["attack_test_case_index"] for row in rows] == [0, 1]
    assert rows[0]["responses_create_params"]["input"] == [{"role": "user", "content": "case one"}]
    assert rows[0]["attack_source_sha256"] == sha256(cases_path)
    assert rows[0]["context"] == "test context"
    assert manifest["test_cases_sha256"] == sha256(cases_path)
    assert manifest["behaviors"] == 1 and manifest["rows"] == 2


def test_materializes_image_without_leaking_local_file_path(tmp_path):
    behavior_path = tmp_path / "behaviors.csv"
    cases_path = tmp_path / "test_cases.json"
    images = tmp_path / "images"
    images.mkdir()
    (images / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\ncontents")
    _behaviors(behavior_path)
    cases_path.write_text(json.dumps({"v1": [["sample.png", "describe image"]]}), encoding="utf-8")
    rows, _ = materialize(
        method_name="MultiModalDirectRequest",
        test_cases_path=cases_path,
        behaviors_path=behavior_path,
        experiment="default",
        run_id="run-2",
        target_type="vision_api",
        generation_receipt=_receipt("MultiModalDirectRequest", cases_path, behavior_path, "default", "run-2"),
        image_dir=images,
    )
    content = rows[0]["responses_create_params"]["input"][0]["content"]
    assert (
        content[0]["image_url"]
        == "data:image/png;base64," + base64.b64encode((images / "sample.png").read_bytes()).decode()
    )
    assert content[1] == {"type": "input_text", "text": "describe image"}
    assert rows[0]["context"] == "redacted description"
    assert rows[0]["attack_image_sha256"] == sha256(images / "sample.png")


def test_rejects_incompatible_target_and_unmatched_behaviors(tmp_path):
    behavior_path = tmp_path / "behaviors.csv"
    cases_path = tmp_path / "test_cases.json"
    _behaviors(behavior_path)
    cases_path.write_text(json.dumps({"missing": ["test"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not support"):
        materialize(
            method_name="GCG",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="model-1",
            run_id="run-3",
            target_type="text_api",
            generation_receipt=_receipt("GCG", cases_path, behavior_path, "model-1", "run-3"),
        )
    with pytest.raises(ValueError, match="unknown behavior IDs"):
        materialize(
            method_name="PAIR",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="model-1",
            run_id="run-3",
            target_type="text_api",
            generation_receipt=_receipt("PAIR", cases_path, behavior_path, "model-1", "run-3"),
        )


def test_gcg_requires_full_corpus_and_rehashes_shard_receipts(tmp_path):
    behavior_path = tmp_path / "behaviors.csv"
    cases_path = tmp_path / "test_cases.json"
    receipt_path = tmp_path / "generation-receipt.json"
    with behavior_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "BehaviorID",
                "Behavior",
                "ContextString",
                "FunctionalCategory",
                "SemanticCategory",
                "Tags",
                "RedactedImageDescription",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "BehaviorID": f"b{index:03d}",
                "Behavior": "synthetic behavior",
                "FunctionalCategory": "standard",
                "SemanticCategory": "test",
                "Tags": "",
            }
            for index in range(400)
        )
    cases = {f"b{index:03d}": ["synthetic case"] for index in range(400)}
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    shard_dir = tmp_path / "shard-receipts"
    shard_dir.mkdir()
    manifest = []
    for index in range(2):
        path = shard_dir / f"super-{index:02d}-of-02.json"
        path.write_text(json.dumps({"shard": index}), encoding="utf-8")
        manifest.append({"name": path.name, "sha256": sha256(path)})
    receipt = _receipt("GCG", cases_path, behavior_path, "nemotron_3_5_super_gcg", "full-run")
    receipt.update(
        {
            "source_target_model": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            "source_target_revision": "hf-ea-0e636f7",
            "behaviors": 400,
            "cases": 400,
            "num_steps": 500,
            "search_width": 512,
            "num_shards": 2,
            "shard_receipts": manifest,
            "shard_receipts_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    rows, _ = materialize(
        method_name="GCG",
        test_cases_path=cases_path,
        behaviors_path=behavior_path,
        experiment="nemotron_3_5_super_gcg",
        run_id="full-run",
        target_type="text_weights",
        generation_receipt=receipt,
        generation_receipt_path=receipt_path,
    )
    assert len(rows) == 400

    (shard_dir / "super-01-of-02.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="readback failed"):
        materialize(
            method_name="GCG",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="nemotron_3_5_super_gcg",
            run_id="full-run",
            target_type="text_weights",
            generation_receipt=receipt,
            generation_receipt_path=receipt_path,
        )

    partial_cases = dict(list(cases.items())[:-1])
    cases_path.write_text(json.dumps(partial_cases), encoding="utf-8")
    receipt["test_cases_sha256"] = sha256(cases_path)
    with pytest.raises(ValueError, match="exactly one generated case"):
        materialize(
            method_name="GCG",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="nemotron_3_5_super_gcg",
            run_id="full-run",
            target_type="text_weights",
            generation_receipt=receipt,
        )


def test_rejects_mislabeled_and_unbound_fresh_attacks(tmp_path):
    behavior_path = tmp_path / "behaviors.csv"
    cases_path = tmp_path / "test_cases.json"
    _behaviors(behavior_path)
    cases_path.write_text(json.dumps({"b1": ["test"]}), encoding="utf-8")
    receipt = _receipt("HumanJailbreaks", cases_path, behavior_path, "random_subset_5", "run-1")
    with pytest.raises(ValueError, match="generation receipt method"):
        materialize(
            method_name="ZeroShot",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="random_subset_5",
            run_id="run-1",
            target_type="text_api",
            generation_receipt=receipt,
        )
    fresh_receipt = _receipt("Fresh PAIR against the client model", cases_path, behavior_path, "client-1", "run-1")
    with pytest.raises(ValueError, match="requires proof"):
        materialize(
            method_name="Fresh PAIR against the client model",
            test_cases_path=cases_path,
            behaviors_path=behavior_path,
            experiment="client-1",
            run_id="run-1",
            target_type="text_api",
            generation_receipt=fresh_receipt,
        )


def test_human_jailbreak_selection_reproduces_upstream_in_place_shuffles():
    templates = ["a", "b", "c", "d", "e", "f"]
    behaviors = [
        {"BehaviorID": "one", "Behavior": "first", "ContextString": ""},
        {"BehaviorID": "two", "Behavior": "second", "ContextString": "context"},
    ]
    import random

    expected_templates = templates.copy()
    random_generator = random.Random(1)
    random_generator.shuffle(expected_templates)
    first = [f"{template}\n\nfirst" for template in expected_templates[:5]]
    random_generator.shuffle(expected_templates)
    second = [f"{template}\n\ncontext\n\n---\n\nsecond" for template in expected_templates[:5]]
    assert generate(behaviors, templates) == {"one": first, "two": second}


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_human_jailbreak_full_source_is_all_400_public_text_behaviors():
    import hashlib

    _, behaviors = _require_upstream(UPSTREAM)
    assert behaviors.name == "harmbench_behaviors_text_all.csv"
    assert hashlib.sha256(behaviors.read_bytes()).hexdigest() == BEHAVIORS_SHA256
    with behaviors.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 400


def test_multimodal_source_extension_correction_is_explicit_and_hash_bound(tmp_path):
    (tmp_path / "present.png").write_bytes(b"\x89PNG\r\n\x1a\ncurrent")
    (tmp_path / "missing.png").write_bytes(b"\x89PNG\r\n\x1a\nreplacement")
    rows, corrections = corrected_rows(
        [
            {"BehaviorID": "a", "ImageFileName": "present.png"},
            {"BehaviorID": "b", "ImageFileName": "missing.jpeg"},
        ],
        tmp_path,
    )
    assert [row["ImageFileName"] for row in rows] == ["present.png", "missing.png"]
    assert corrections == [
        {
            "behavior_id": "b",
            "source_filename": "missing.jpeg",
            "resolved_filename": "missing.png",
            "image_sha256": sha256(tmp_path / "missing.png"),
        }
    ]


def test_copyright_canary_selects_one_of_each_kind(tmp_path):
    source = tmp_path / "inputs.jsonl"
    cases = [
        {"harmbench_id": "standard", "tags": []},
        {"harmbench_id": "lyrics", "tags": ["lyrics", "hash_check"]},
        {"harmbench_id": "book", "tags": ["book", "hash_check"]},
        {"harmbench_id": "book-again", "tags": ["book", "hash_check"]},
    ]
    source.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    assert [case["harmbench_id"] for case in select(source)] == ["book", "lyrics"]


def test_multimodal_direct_image_port_produces_expected_rgb_crop(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (8, 4), (12, 34, 56)).save(source)
    _image(source, output, width=4, height=4)
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)
        assert image.getpixel((2, 2)) == (12, 34, 56)
