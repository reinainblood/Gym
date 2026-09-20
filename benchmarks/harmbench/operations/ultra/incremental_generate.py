# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an upstream HarmBench method and persist each completed behavior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


def run(args: argparse.Namespace) -> None:
    import torch

    gym_root = Path("/app/Gym")
    if gym_root.is_dir():
        sys.path.insert(0, str(gym_root))
    sys.path.insert(0, str(args.upstream))
    from baselines import get_method_class, init_method
    from eval_utils import get_experiment_config

    model_configs = yaml.full_load(args.models_config.read_text(encoding="utf-8"))
    method_configs = yaml.full_load(args.method_config.read_text(encoding="utf-8"))
    method_config = dict(method_configs["default_method_hyperparameters"] or {})
    method_config.update(get_experiment_config(args.experiment, model_configs, method_configs))
    method_class = get_method_class(args.class_name)
    method = init_method(method_class, method_config)
    if args.class_name == "GCG":
        from benchmarks.harmbench.cache_compat import install_gcg_dynamic_cache_adapter

        install_gcg_dynamic_cache_adapter(method)
        print(
            json.dumps(
                {
                    "event": "transformers_cache_compat_installed",
                    "class_name": args.class_name,
                    "conversion": "DynamicCache.batch_repeat_interleave",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with args.behaviors.open(newline="", encoding="utf-8") as stream:
        behaviors = list(csv.DictReader(stream))
    if args.behavior_ids:
        selected = set(args.behavior_ids.split(","))
        behaviors = [behavior for behavior in behaviors if behavior["BehaviorID"] in selected]
    if not behaviors:
        raise ValueError("no behaviors selected")

    print(
        json.dumps(
            {
                "event": "harmbench_method_start",
                "method": args.public_name,
                "class_name": args.class_name,
                "behaviors": len(behaviors),
                "run_id": args.run_id,
                "world_size": torch.cuda.device_count(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.class_name == "EnsembleGCG":
        path = Path(method_class.get_output_file_path(args.output_dir, "", "test_cases", args.run_id))
        if path.exists():
            print(json.dumps({"event": "existing_run_reused", "path": str(path)}), flush=True)
            return
        test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=False)
        method.save_test_cases(
            str(args.output_dir),
            test_cases,
            logs,
            method_config=method_config,
            run_id=args.run_id,
        )
        return

    completed = 0
    for behavior in behaviors:
        behavior_id = behavior["BehaviorID"]
        path = Path(method_class.get_output_file_path(str(args.output_dir), behavior_id, "test_cases", args.run_id))
        if path.exists():
            completed += 1
            continue
        test_cases, logs = method.generate_test_cases(behaviors=[behavior], verbose=False)
        method.save_test_cases(
            str(args.output_dir),
            test_cases,
            logs,
            method_config=method_config,
            run_id=args.run_id,
        )
        completed += 1
        print(
            json.dumps(
                {
                    "event": "harmbench_behavior_saved",
                    "method": args.public_name,
                    "behavior_id": behavior_id,
                    "completed": completed,
                    "total": len(behaviors),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--method-config", required=True, type=Path)
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-name", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--behavior-ids", default="")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
