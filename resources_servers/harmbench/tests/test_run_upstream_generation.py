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


UPSTREAM = Path(__file__).resolve().parents[5] / "reference/HarmBench"


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
def test_every_requested_method_matches_pinned_upstream_pipeline(tmp_path):
    assert audit_method_catalog(UPSTREAM) == 22
    pipeline = yaml.safe_load((UPSTREAM / "configs/pipeline_configs/run_pipeline.yaml").read_text())
    behaviors = UPSTREAM / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    for method in METHODS.values():
        if method.generation == "client_fresh":
            with pytest.raises(ValueError, match="client-model target binding"):
                build_generate_command(
                    upstream=UPSTREAM,
                    method_name=method.name,
                    experiment="client",
                    behaviors=behaviors,
                    output_dir=tmp_path,
                    upstream_python=Path(__file__),
                )
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
