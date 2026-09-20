# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a method from the pinned HarmBench generator, then issue a Gym receipt.

This dispatches the *actual upstream attack class* in a separately prepared
HarmBench environment. It does not substitute prompts, target models, or
hyperparameters. Methods requiring local weights/GPU must run on such a host.
The two client-fresh variants need a separately validated client-model binding
and are intentionally rejected here until that path is built and tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from benchmarks.harmbench.methods import METHODS, get_method
from benchmarks.harmbench.prepare import UPSTREAM_REVISION
from benchmarks.harmbench.prepare_generated import materialize, sha256


ENSEMBLE_RUNS = {"GCG-Multi", "GCG-Transfer"}
UPSTREAM_TARGET_TYPES = {
    "text_api": "closed_source",
    "text_weights": "open_source",
    "vision_api": "closed_source_multimodal",
    "vision_weights": "open_source_multimodal",
}


def audit_method_catalog(upstream: Path) -> int:
    """Fail if any customer method drifts from the pinned upstream pipeline."""
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    pipeline = yaml.safe_load((upstream / "configs/pipeline_configs/run_pipeline.yaml").read_text(encoding="utf-8"))
    for method in METHODS.values():
        entry = pipeline.get(method.upstream_key)
        if entry is None:
            raise ValueError(f"{method.name} maps to missing upstream key {method.upstream_key}")
        allowed = set(entry["allowed_target_model_types"])
        claimed = {UPSTREAM_TARGET_TYPES[target_type] for target_type in method.target_types}
        if not claimed.issubset(allowed):
            raise ValueError(f"{method.name} claims unsupported target types: {sorted(claimed - allowed)}")
    return len(METHODS)


def _upstream_config(upstream: Path, method_name: str, experiment: str) -> tuple[str, str]:
    method = get_method(method_name)
    if method.generation == "client_fresh":
        raise ValueError("client-fresh PAIR/TAP requires a verified client-model target binding")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    pipeline_path = upstream / "configs/pipeline_configs/run_pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    upstream_entry = pipeline.get(method.upstream_key)
    if upstream_entry is None:
        raise ValueError(f"{method.upstream_key} is absent from the pinned HarmBench pipeline")
    template = upstream_entry["experiment_name_template"]
    if "<model_name>" not in template and experiment != template:
        raise ValueError(f"{method.name} requires the pinned experiment {template!r}, not {experiment!r}")
    return upstream_entry["class_name"], hashlib.sha256(pipeline_path.read_bytes()).hexdigest()


def build_generate_command(
    *,
    upstream: Path,
    method_name: str,
    experiment: str,
    behaviors: Path,
    output_dir: Path,
    upstream_python: Path,
    models_config: Path | None = None,
    method_config: Path | None = None,
    run_id: str | None = None,
) -> tuple[list[str], str, str]:
    class_name, pipeline_sha256 = _upstream_config(upstream, method_name, experiment)
    if method_name in ENSEMBLE_RUNS and run_id not in {"0", "1", "2", "3", "4"}:
        raise ValueError(f"{method_name} requires one of the pinned run IDs 0–4")
    if method_name not in ENSEMBLE_RUNS and run_id is not None:
        raise ValueError(f"{method_name} has no upstream run IDs; omit --run-id")
    command = [
        str(upstream_python.resolve(strict=True)),
        "generate_test_cases.py",
        "--method_name",
        class_name,
        "--experiment_name",
        experiment,
        "--behaviors_path",
        str(behaviors.resolve(strict=True)),
        "--save_dir",
        str(output_dir.resolve()),
    ]
    if models_config is not None:
        command.extend(["--models_config_file", str(models_config.resolve(strict=True))])
    if method_config is not None:
        command.extend(["--method_config_file", str(method_config.resolve(strict=True))])
    if run_id is not None:
        command.extend(["--run_id", run_id])
    return command, class_name, pipeline_sha256


def run(
    *,
    upstream: Path,
    method_name: str,
    experiment: str,
    behaviors: Path,
    output_dir: Path,
    target_type: str,
    upstream_python: Path,
    models_config: Path | None = None,
    method_config: Path | None = None,
    run_id: str | None = None,
) -> Path | None:
    method = get_method(method_name)
    if target_type not in method.target_types:
        raise ValueError(f"{method_name} does not support {target_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "test_cases.json"
    if cases_path.exists() or (method_name not in ENSEMBLE_RUNS and any(output_dir.iterdir())):
        raise FileExistsError("use a new output directory; existing attack artifacts will not be overwritten")
    if method_name in ENSEMBLE_RUNS and (output_dir / f"test_cases_{run_id}.json").exists():
        raise FileExistsError(f"upstream run ID {run_id} already exists; refusing to overwrite it")
    command, class_name, pipeline_sha256 = build_generate_command(
        upstream=upstream,
        method_name=method_name,
        experiment=experiment,
        behaviors=behaviors,
        output_dir=output_dir,
        upstream_python=upstream_python,
        models_config=models_config,
        method_config=method_config,
        run_id=run_id,
    )
    effective_models_config = models_config or upstream / "configs/model_configs/models.yaml"
    effective_method_config = method_config or upstream / f"configs/method_configs/{class_name}_config.yaml"
    common_receipt = {
        "method": method_name,
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": experiment,
        "behaviors_sha256": sha256(behaviors),
        "models_config_sha256": sha256(effective_models_config),
        "method_config_sha256": sha256(effective_method_config),
        "upstream_pipeline_sha256": pipeline_sha256,
    }
    # Upstream generate_test_cases.py prints the entire resolved method config,
    # which can include API tokens. Never forward that output into Codex or CI
    # logs. The generated artifacts and their hashes remain available for QA.
    subprocess.run(command, cwd=upstream, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if method_name in ENSEMBLE_RUNS:
        part = output_dir / f"test_cases_{run_id}.json"
        if not part.is_file():
            raise FileNotFoundError(f"upstream did not produce {part}")
        part_receipt = common_receipt | {"run_id": run_id, "part_sha256": sha256(part)}
        (output_dir / f"run_{run_id}.receipt.json").write_text(
            json.dumps(part_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        missing = [index for index in range(5) if not (output_dir / f"test_cases_{index}.json").is_file()]
        if missing:
            print(f"Generated {method_name} run {run_id}; waiting for upstream run IDs {missing} before merging")
            return None
        for index in range(5):
            member = output_dir / f"test_cases_{index}.json"
            member_receipt_path = output_dir / f"run_{index}.receipt.json"
            if not member_receipt_path.is_file():
                raise ValueError(f"missing generation receipt for ensemble run {index}")
            member_receipt = json.loads(member_receipt_path.read_text(encoding="utf-8"))
            if any(member_receipt.get(key) != value for key, value in common_receipt.items()) or member_receipt.get(
                "part_sha256"
            ) != sha256(member):
                raise ValueError(f"mixed or changed ensemble run {index}; refusing to merge")
        subprocess.run(
            [
                str(upstream_python.resolve(strict=True)),
                "merge_test_cases.py",
                "--method_name",
                class_name,
                "--save_dir",
                str(output_dir.resolve()),
            ],
            cwd=upstream,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif not cases_path.is_file():
        subprocess.run(
            [
                str(upstream_python.resolve(strict=True)),
                "merge_test_cases.py",
                "--method_name",
                class_name,
                "--save_dir",
                str(output_dir.resolve()),
            ],
            cwd=upstream,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not cases_path.is_file():
        raise FileNotFoundError(f"upstream did not produce {cases_path}")
    correction_manifest = behaviors.with_suffix(".corrections.json")
    receipt = {
        "method": method_name,
        "upstream_method": method.upstream_key,
        "upstream_class": class_name,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_pipeline_sha256": pipeline_sha256,
        "upstream_python": str(upstream_python.resolve(strict=True)),
        "experiment": experiment,
        "run_id": "0-4" if method_name in ENSEMBLE_RUNS else "upstream-default",
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behaviors),
        "behavior_corrections_sha256": sha256(correction_manifest) if correction_manifest.is_file() else None,
        "models_config_sha256": sha256(effective_models_config),
        "method_config_sha256": sha256(effective_method_config),
        "source_experiment": experiment if method_name in {"GCG-Transfer", "TAP-Transfer"} else None,
        "ensemble_run_hashes": (
            {str(index): sha256(output_dir / f"test_cases_{index}.json") for index in range(5)}
            if method_name in ENSEMBLE_RUNS
            else None
        ),
    }
    receipt_path = output_dir / "generation-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    image_dir = output_dir / "images" if target_type.startswith("vision") else None
    rows, manifest = materialize(
        method_name=method_name,
        test_cases_path=cases_path,
        behaviors_path=behaviors,
        experiment=experiment,
        run_id=receipt["run_id"],
        target_type=target_type,
        generation_receipt=receipt,
        image_dir=image_dir,
    )
    output_path = output_dir / "gym-inputs.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest["gym_inputs_sha256"] = sha256(output_path)
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} {method_name} Gym rows at {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-python", required=True, type=Path)
    parser.add_argument(
        "--target-type", required=True, choices=["text_api", "text_weights", "vision_api", "vision_weights"]
    )
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run(
        upstream=args.upstream,
        method_name=args.method,
        experiment=args.experiment,
        behaviors=args.behaviors,
        output_dir=args.output_dir,
        target_type=args.target_type,
        upstream_python=args.upstream_python,
        models_config=args.models_config,
        method_config=args.method_config,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
