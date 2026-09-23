# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

import pytest
import yaml

from benchmarks.harmbench.methods import METHODS
from benchmarks.harmbench.run_upstream_generation import (
    ENSEMBLE_RUNS,
    audit_method_catalog,
    build_generate_command,
    run,
)
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_every_requested_method_matches_pinned_upstream_pipeline(tmp_path):
    assert audit_method_catalog(UPSTREAM) == 23
    pipeline = yaml.safe_load((UPSTREAM / "configs/pipeline_configs/run_pipeline.yaml").read_text())
    behaviors = UPSTREAM / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    for method in METHODS.values():
        if method.generation == "client_fresh":
            binding = tmp_path / f"{method.upstream_key}-binding.json"
            binding.write_text("{}", encoding="utf-8")
            command, class_name, digest = build_generate_command(
                upstream=UPSTREAM,
                method_name=method.name,
                experiment="client",
                behaviors=behaviors,
                output_dir=tmp_path,
                upstream_python=Path(__file__),
                client_base_url="https://client.example",
                client_model="client-model",
                client_revision="client-revision",
                client_api_key_env="CLIENT_API_KEY",
                client_binding_receipt=binding,
            )
            assert command[1].endswith("client_fresh_generate.py")
            assert command[command.index("--method") + 1] == method.upstream_key
            assert command[command.index("--api-key-env") + 1] == "CLIENT_API_KEY"
            assert class_name == pipeline[method.upstream_key]["class_name"]
            assert len(digest) == 64
            continue
        experiment = pipeline[method.upstream_key]["experiment_name_template"].replace("<model_name>", "test-model")
        command, class_name, digest = build_generate_command(
            upstream=UPSTREAM,
            method_name=method.name,
            experiment=experiment,
            behaviors=behaviors,
            output_dir=tmp_path,
            upstream_python=Path(__file__),
            run_id="0" if method.name in ENSEMBLE_RUNS else None,
        )
        assert command[0] == str(Path(__file__).resolve())
        assert command[1:4] == ["generate_test_cases.py", "--method_name", class_name]
        assert class_name == pipeline[method.upstream_key]["class_name"]
        assert len(digest) == 64


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_ensemble_requires_five_matching_receipts_before_materializing(monkeypatch, tmp_path):
    import benchmarks.harmbench.run_upstream_generation as runner

    monkeypatch.setattr(
        runner, "_upstream_config", lambda upstream, method_name, experiment: ("EnsembleGCG", "a" * 64)
    )
    behaviors = UPSTREAM / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    with behaviors.open(newline="", encoding="utf-8") as handle:
        behavior_id = next(csv.DictReader(handle))["BehaviorID"]
    output_dir = tmp_path / "ensemble"

    def fake_run(command, *, cwd, check, stdout, stderr):
        if command[1] == "generate_test_cases.py":
            index = command[command.index("--run_id") + 1]
            (output_dir / f"test_cases_{index}.json").write_text(json.dumps({behavior_id: [f"case {index}"]}))
        elif command[1] == "merge_test_cases.py":
            (output_dir / "test_cases.json").write_text(json.dumps({behavior_id: [f"case {i}" for i in range(5)]}))
        else:
            raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for index in range(4):
        result = run(
            upstream=UPSTREAM,
            method_name="GCG-Transfer",
            experiment="llama2_7b_vicuna_7b_llama2_13b_vicuna_13b_multibehavior_1000steps",
            behaviors=behaviors,
            output_dir=output_dir,
            target_type="text_api",
            upstream_python=Path(__file__),
            run_id=str(index),
        )
        assert result is None
        assert (output_dir / f"run_{index}.receipt.json").is_file()
    result = run(
        upstream=UPSTREAM,
        method_name="GCG-Transfer",
        experiment="llama2_7b_vicuna_7b_llama2_13b_vicuna_13b_multibehavior_1000steps",
        behaviors=behaviors,
        output_dir=output_dir,
        target_type="text_api",
        upstream_python=Path(__file__),
        run_id="4",
    )
    assert result is not None and result.is_file()
    assert sum(1 for _ in result.open()) == 5
    receipt = json.loads((output_dir / "generation-receipt.json").read_text())
    assert set(receipt["ensemble_run_hashes"]) == {"0", "1", "2", "3", "4"}


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_client_fresh_command_is_resumable_and_never_receives_secret_value(monkeypatch, tmp_path):
    import benchmarks.harmbench.run_upstream_generation as runner

    behaviors = UPSTREAM / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    output_dir = tmp_path / "partial-client-fresh"
    output_dir.mkdir()
    (output_dir / "client-call-receipts").mkdir()
    observed = {}
    binding = tmp_path / "binding.json"
    binding.write_text("{}", encoding="utf-8")

    def fake_run(command, *, cwd, check, stdout, stderr):
        observed["command"] = command
        with behaviors.open(newline="", encoding="utf-8") as handle:
            behavior_id = next(csv.DictReader(handle))["BehaviorID"]
        cases = {behavior_id: ["synthetic case"]}
        (output_dir / "test_cases.json").write_text(json.dumps(cases), encoding="utf-8")
        (output_dir / "client-target-receipt.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "client_target_verified": True,
                    "client_model": "client-model",
                    "client_revision": "client-revision",
                    "source_target_model": "client-model",
                    "behaviors_sha256": runner.sha256(behaviors),
                    "test_cases_sha256": runner.sha256(output_dir / "test_cases.json"),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "_upstream_config", lambda *_: ("PAIR", "a" * 64))
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("CLIENT_API_KEY", "must-not-enter-command")
    result = run(
        upstream=UPSTREAM,
        method_name="Fresh PAIR against the client model",
        experiment="client-model",
        behaviors=behaviors,
        output_dir=output_dir,
        target_type="text_api",
        upstream_python=Path(__file__),
        client_base_url="https://client.example",
        client_model="client-model",
        client_revision="client-revision",
        client_api_key_env="CLIENT_API_KEY",
        client_binding_receipt=binding,
    )
    assert result is not None and result.is_file()
    assert "must-not-enter-command" not in observed["command"]
    assert observed["command"][observed["command"].index("--api-key-env") + 1] == "CLIENT_API_KEY"
