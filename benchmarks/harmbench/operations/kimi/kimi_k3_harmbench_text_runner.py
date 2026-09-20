# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the immutable public HarmBench text white-box classes on Kimi K3."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml


UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
METHODS = {
    "GCG": (
        "GCG",
        "GCG_config.yaml",
        "f046d305c30938a29ee51af011cee6380c833d6a11246d7a68f78288da8bfc99",
    ),
    "GCG-Multi": (
        "EnsembleGCG",
        "EnsembleGCG_config.yaml",
        "eece35ed068032df9f12c6eaf852078dcfae3d00a92b1ee0b2f75fd2a9c64d4f",
    ),
    "AutoPrompt": (
        "AutoPrompt",
        "AutoPrompt_config.yaml",
        "0887dcb1fb8637b05b44adb475cb1eefb9ff7356cc2a7aa96ec745e155dc4972",
    ),
    "GBDA": (
        "GBDA",
        "GBDA_config.yaml",
        "14572275354078065f67f8eb95b7b023f5525e72ce13efcf6e9a13d728327c2d",
    ),
    "PEZ": (
        "PEZ",
        "PEZ_config.yaml",
        "4ccb3c01ecb39f0579b347f7f4b15e40d9d73c5d6d7a9072c67b2211013253b0",
    ),
    "UAT": (
        "UAT",
        "UAT_config.yaml",
        "4fc01891c14d81383a2c9be002e5b521d15806ab60a7c50342844b86212622b3",
    ),
    "AutoDAN": (
        "AutoDAN",
        "AutoDAN_config.yaml",
        "beaba0873558e1de3b9e61301c9948abedf63e0227ea5d365e8a2fe891e7bfaf",
    ),
    "FewShot": (
        "FewShot",
        "FewShot_config.yaml",
        "11919d2720a357e0a68f175e66d54db2b7e0245319b90b09c96eabf79eed3c65",
    ),
}
_MODEL_CACHE: dict[str, tuple[object, object]] = {}
UNCOMPRESSED_PROJECTION_PATTERN = (
    r"re:.*(self_attention_res_proj|mlp_res_proj|output_attn_res_proj|routed_expert_(up|down)_proj).*"
)


def install_kimi_transformers_compatibility() -> None:
    """Restore the retired HF output-recording metadata used by served K3."""
    from transformers.utils import generic

    if hasattr(generic, "OutputRecorder"):
        return

    @dataclass
    class OutputRecorder:
        target_class: type
        index: int = 0
        layer_name: str | None = None
        class_name: str | None = None

    generic.OutputRecorder = OutputRecorder

    from transformers.modeling_utils import PreTrainedModel

    original_post_init = PreTrainedModel.post_init

    def post_init(self) -> None:
        if self.__class__.__name__ == "KimiK3ForConditionalGeneration":
            original_tie_weights = self.tie_weights
            self.tie_weights = lambda *args, **kwargs: original_tie_weights()
        original_post_init(self)

    PreTrainedModel.post_init = post_init


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_loader_view(snapshot: Path) -> Path:
    """Correct checkpoint compression metadata without copying or changing weights."""
    view = Path("/tmp/kimi-k3-loader-view")
    view.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        target = view / source.name
        if source.name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            ignored = config["text_config"]["quantization_config"]["ignore"]
            if UNCOMPRESSED_PROJECTION_PATTERN not in ignored:
                ignored.append(UNCOMPRESSED_PROJECTION_PATTERN)
            target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
    return view


def _target(snapshot: Path, *, reserve_attacker_gpu: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "model_name_or_path": str(snapshot),
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "dtype": "auto",
        "device_map": "auto",
        "local_files_only": True,
    }
    if reserve_attacker_gpu:
        # AutoDAN and FewShot themselves reserve GPU 0 for their public
        # Mistral/Mixtral attacker and build the target max_memory map.
        value["num_gpus"] = 7
    return value


def _existing_path(method_class, output_dir: Path, behavior_id: str, run_id: str | None) -> Path:
    return Path(method_class.get_output_file_path(str(output_dir), behavior_id, "test_cases", run_id=run_id))


def _install_cached_target_loader(snapshot: Path) -> None:
    import torch
    from baselines import model_utils

    if getattr(model_utils.load_model_and_tokenizer, "_kimi_k3_cached", False):
        return
    original = model_utils.load_model_and_tokenizer

    def cached_loader(model_name_or_path, *args, **kwargs):
        model_path = str(model_name_or_path)
        if model_path != str(snapshot):
            return original(model_name_or_path, *args, **kwargs)
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]
        if "max_memory" not in kwargs:
            kwargs["max_memory"] = {
                0: 0,
                **{index: torch.cuda.mem_get_info(index)[0] for index in range(1, 8)},
            }
        value = original(model_name_or_path, *args, **kwargs)
        _MODEL_CACHE[model_path] = value
        return value

    cached_loader._kimi_k3_cached = True
    model_utils.load_model_and_tokenizer = cached_loader


def run(
    *,
    upstream: Path,
    snapshot: Path,
    output_root: Path,
    method_name: str,
    run_id: str | None,
    commit: Callable[[], None],
) -> dict[str, object]:
    if method_name not in METHODS:
        raise ValueError(f"unknown text white-box method: {method_name}")
    if method_name == "GCG-Multi" and run_id not in {"0", "1", "2", "3", "4"}:
        raise ValueError("GCG-Multi requires public run ID 0, 1, 2, 3, or 4")
    if method_name != "GCG-Multi" and run_id is not None:
        raise ValueError(f"{method_name} has no public run ID")
    revision = (upstream / ".pinned-revision").read_text(encoding="utf-8").strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision is {revision}, expected {UPSTREAM_REVISION}")
    class_name, config_name, config_hash = METHODS[method_name]
    config_path = upstream / "configs/method_configs" / config_name
    if _sha256(config_path) != config_hash:
        raise ValueError(f"public config hash mismatch for {method_name}")
    method_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["default_method_hyperparameters"]
    reserve = method_name in {"AutoDAN", "FewShot"}
    loader_view = prepare_loader_view(snapshot)
    target = _target(loader_view, reserve_attacker_gpu=reserve)
    if method_name == "GCG-Multi":
        method_config["target_models"] = [{"target_model": target, "num_gpus": 8}]
    else:
        method_config["target_model"] = target
    if method_name == "AutoDAN":
        target.update({"model_short_name": "Kimi K3", "developer_name": "Moonshot AI"})
    if method_name == "FewShot":
        method_config["attack_model"]["num_gpus"] = 1

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    install_kimi_transformers_compatibility()
    from baselines import get_method_class, init_method

    _install_cached_target_loader(loader_view)

    method_class = get_method_class(class_name)
    output_dir = output_root / method_name / (run_id or "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    behaviors_path = upstream / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    with behaviors_path.open(newline="", encoding="utf-8") as stream:
        behaviors = list(csv.DictReader(stream))
    started = time.time()
    method = init_method(method_class, method_config)

    try:
        if method_name == "GCG-Multi":
            result_path = _existing_path(method_class, output_dir, "", run_id)
            if not result_path.exists():
                test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=True)
                method.save_test_cases(
                    str(output_dir),
                    test_cases,
                    logs,
                    method_config=method_config,
                    run_id=run_id,
                )
                commit()
        else:
            for index, behavior in enumerate(behaviors):
                behavior_id = behavior["BehaviorID"]
                if _existing_path(method_class, output_dir, behavior_id, None).exists():
                    continue
                try:
                    test_cases, logs = method.generate_test_cases(behaviors=[behavior], verbose=True)
                    method.save_test_cases(str(output_dir), test_cases, logs, method_config=method_config)
                    commit()
                except Exception as exc:
                    failure = {
                        "schema_version": 1,
                        "status": "failed",
                        "method": method_name,
                        "behavior_id": behavior_id,
                        "behavior_index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failure_path = output_dir / "failures" / f"{behavior_id}.json"
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_path.write_text(
                        json.dumps(failure, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    commit()
                    raise
    finally:
        progress = {
            "schema_version": 1,
            "status": "running_or_complete",
            "method": method_name,
            "run_id": run_id,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "upstream_revision": revision,
            "public_config_sha256": config_hash,
            "expected_behaviors": len(behaviors),
            "present_outputs": sum(
                _existing_path(method_class, output_dir, b["BehaviorID"], run_id).exists() for b in behaviors
            ),
            "elapsed_seconds": time.time() - started,
        }
        (output_dir / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        commit()
    return progress
