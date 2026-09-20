# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.ultra_whitebox import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MODEL_TOTAL_BYTES,
    PUBLIC_FULL_TEXT_BEHAVIORS,
    PUBLIC_FULL_TEXT_DATASET,
    WHITEBOX_METHODS,
    build_generate_command,
    validate_checkpoint_manifest,
    validate_upstream,
    write_runtime_configs,
)
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


def test_whitebox_catalog_is_complete_and_preserves_public_repetitions():
    assert set(WHITEBOX_METHODS) == {
        "GCG",
        "GCG-Multi",
        "AutoPrompt",
        "GBDA",
        "PEZ",
        "UAT",
        "AutoDAN",
        "FewShot",
    }
    assert WHITEBOX_METHODS["GCG-Multi"].run_ids == ("0", "1", "2", "3", "4")
    assert sum(spec.cases_per_behavior for spec in WHITEBOX_METHODS.values()) == 20
    assert PUBLIC_FULL_TEXT_DATASET == "harmbench_behaviors_text_all.csv"
    assert PUBLIC_FULL_TEXT_BEHAVIORS == 400


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_public_full_text_dataset_has_locked_cardinality():
    path = UPSTREAM / "data/behavior_datasets" / PUBLIC_FULL_TEXT_DATASET
    with path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == PUBLIC_FULL_TEXT_BEHAVIORS


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_public_sources_and_default_hyperparameters_are_hash_locked():
    receipt = validate_upstream(UPSTREAM)
    assert receipt["upstream_revision"] == UPSTREAM_REVISION
    assert set(receipt["methods"]) == set(WHITEBOX_METHODS)
    assert receipt["methods"]["GCG"]["default_hyperparameters"]["num_steps"] == 500
    assert receipt["methods"]["GCG"]["default_hyperparameters"]["search_width"] == 512
    assert receipt["methods"]["GBDA"]["default_hyperparameters"]["num_test_cases_per_behavior"] == 5
    assert receipt["methods"]["PEZ"]["default_hyperparameters"]["num_steps"] == 500
    assert receipt["methods"]["UAT"]["default_hyperparameters"]["num_steps"] == 100
    assert receipt["methods"]["AutoDAN"]["default_hyperparameters"]["num_steps"] == 100
    assert receipt["methods"]["FewShot"]["default_hyperparameters"]["sample_size"] == 64
    assert receipt["methods"]["FewShot"]["default_hyperparameters"]["num_steps"] == 50


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_runtime_binding_changes_only_the_target_identity(tmp_path):
    models_path, methods = write_runtime_configs(UPSTREAM, tmp_path, Path("/checkpoint/model"))
    model = yaml.safe_load(models_path.read_text())[MODEL_KEY]
    assert model["model_type"] == "open_source"
    assert model["num_gpus"] == 7
    assert model["model"]["dtype"] == "bfloat16"
    assert model["model"]["model_name_or_path"] == "/checkpoint/model"
    for name, spec in WHITEBOX_METHODS.items():
        source = UPSTREAM / "configs/method_configs" / spec.config_name
        if name not in {"GCG", "AutoDAN", "FewShot"}:
            assert methods[name] == source
    public_gcg = yaml.safe_load((UPSTREAM / "configs/method_configs/GCG_config.yaml").read_text())
    effective_gcg = yaml.safe_load(methods["GCG"].read_text())
    assert effective_gcg["default_method_hyperparameters"]["search_width"] == 512
    assert effective_gcg["default_method_hyperparameters"]["starting_search_batch_size"] == 8
    del effective_gcg["default_method_hyperparameters"]["starting_search_batch_size"]
    assert effective_gcg == public_gcg
    public_autodan = yaml.safe_load((UPSTREAM / "configs/method_configs/AutoDAN_config.yaml").read_text())
    effective_autodan = yaml.safe_load(methods["AutoDAN"].read_text())
    assert effective_autodan["default_method_hyperparameters"] == public_autodan["default_method_hyperparameters"]
    assert effective_autodan[MODEL_KEY]["target_model"]["developer_name"] == "NVIDIA"
    public_fewshot = yaml.safe_load((UPSTREAM / "configs/method_configs/FewShot_config.yaml").read_text())
    effective_fewshot = yaml.safe_load(methods["FewShot"].read_text())
    assert effective_fewshot[MODEL_KEY]["target_model"]["model_name_or_path"] == "/checkpoint/model"
    assert effective_fewshot["default_method_hyperparameters"]["attack_model"]["num_gpus"] == 1
    effective_fewshot["default_method_hyperparameters"]["attack_model"]["num_gpus"] = 2
    del effective_fewshot[MODEL_KEY]
    assert effective_fewshot == public_fewshot


def test_commands_require_all_five_gcg_multi_runs(tmp_path):
    kwargs = {
        "upstream": tmp_path,
        "python": Path("/usr/bin/python3"),
        "behaviors": tmp_path / "behaviors.csv",
        "output_dir": tmp_path / "output",
        "method_name": "GCG-Multi",
        "models_config": tmp_path / "models.yaml",
        "method_config": tmp_path / "method.yaml",
    }
    with pytest.raises(ValueError, match="requires one of public run IDs"):
        build_generate_command(**kwargs)
    commands = [build_generate_command(**kwargs, run_id=str(index)) for index in range(5)]
    assert [command[-1] for command in commands] == ["0", "1", "2", "3", "4"]
    assert all(command[command.index("--method_name") + 1] == "EnsembleGCG" for command in commands)


def test_checkpoint_identity_and_completeness_are_required(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    manifest = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "private": False,
        "gated": False,
        "total_bytes": MODEL_TOTAL_BYTES,
        "file_count": 1,
        "files": [{"path": "config.json"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert validate_checkpoint_manifest(path, checkpoint)["model_id"] == MODEL_ID
    manifest["revision"] = "main"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_manifest(path, checkpoint)


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_upstream_revision_marker_is_accepted_only_with_all_source_hashes(tmp_path, monkeypatch):
    copied = tmp_path / "HarmBench"
    for relative in ("configs/pipeline_configs", "configs/method_configs"):
        source = UPSTREAM / relative
        destination = copied / relative
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                (destination / path.name).write_bytes(path.read_bytes())
    (copied / ".pinned-revision").write_text(UPSTREAM_REVISION + "\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert validate_upstream(copied)["upstream_revision"] == UPSTREAM_REVISION
    (copied / "configs/method_configs/GCG_config.yaml").write_text("drift: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_upstream(copied)
