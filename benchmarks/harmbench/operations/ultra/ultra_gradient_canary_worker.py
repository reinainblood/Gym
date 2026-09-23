# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove target-loss input gradients through Ultra using HarmBench's loader style."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"
WORLD_SIZE = 8


def _stats(tensor) -> dict[str, float]:
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "l1": float(tensor.abs().sum().item()),
        "l2": float(tensor.float().norm().item()),
    }


def run(model_dir: Path, manifest_path: Path, output_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError(f"expected {WORLD_SIZE} visible GPUs, received {torch.cuda.device_count()}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("resolved_revision") != MODEL_REVISION:
        raise ValueError("checkpoint identity mismatch")
    if manifest.get("total_bytes") != 1_121_078_448_762:
        raise ValueError("checkpoint byte count mismatch")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
        max_memory={index: "170GiB" for index in range(WORLD_SIZE)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    torch.cuda.empty_cache()

    messages = [{"role": "user", "content": "Return the name of the blue-green color in one word."}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target = " teal"
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    before = embedding(prompt_ids).detach()
    target_embeds = embedding(target_ids).detach()
    optim = before[:, -1:, :].clone().requires_grad_(True)
    prefix = before[:, :-1, :]
    inputs_embeds = torch.cat((prefix, optim, target_embeds), dim=1)
    labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
    labels[:, -target_ids.shape[1] :] = target_ids

    for index in range(WORLD_SIZE):
        torch.cuda.reset_peak_memory_stats(index)
    output = model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False)
    loss_before = output.loss
    if loss_before is None or not torch.isfinite(loss_before):
        raise ValueError("target loss is absent or non-finite")
    gradient = torch.autograd.grad(loss_before, optim)[0]
    if not torch.isfinite(gradient).all() or not bool((gradient != 0).any()):
        raise ValueError("input embedding gradient is absent, zero, or non-finite")

    scale = max(float(gradient.float().norm().item()), 1e-12)
    updated = (optim.detach() - 0.05 * gradient / scale).to(optim.dtype)
    with torch.no_grad():
        after_output = model(
            inputs_embeds=torch.cat((prefix, updated, target_embeds), dim=1),
            labels=labels,
            use_cache=False,
        )
    loss_after = after_output.loss
    if loss_after is None or not torch.isfinite(loss_after):
        raise ValueError("updated target loss is absent or non-finite")
    before_value = float(loss_before.item())
    after_value = float(loss_after.item())
    if not after_value < before_value:
        raise ValueError("the signed input-gradient step did not lower target loss")

    per_device = [
        {
            "device": index,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(WORLD_SIZE)
    ]
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "world_size": WORLD_SIZE,
        "distributed_strategy": "accelerate_device_map_auto",
        "model_parameters_frozen": True,
        "autograd_input": "input_embedding",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "input_token_count": int(prompt_ids.shape[1]),
        "target_token_count": int(target_ids.shape[1]),
        "loss_before": before_value,
        "loss_after_one_step": after_value,
        "loss_changed": not math.isclose(before_value, after_value, rel_tol=0.0, abs_tol=1e-7),
        "gradient": _stats(gradient),
        "gradient_finite": True,
        "gradient_nonzero": True,
        "loss_decreased_all_ranks": True,
        "hf_device_map": {key: str(value) for key, value in model.hf_device_map.items()},
        "per_device": per_device,
        "torch_version": str(torch.__version__),
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.manifest, args.output)


if __name__ == "__main__":
    main()
