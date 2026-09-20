# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed Super VL input-gradient canary worker.

This is launched under torchrun by ``modal_super_vl_gradient_canary.py``. It
loads the verified BF16 checkpoint with FSDP2 sharding, freezes every model
parameter, and differentiates one safe target-completion loss with respect to
one synthetic RGB image. The receipt contains only hashes, shapes, memory, and
numeric diagnostics; it never records prompts, target text, or model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


MODEL_VERSION = "hf-ea-0e636f7"
WORLD_SIZE = 8


def _suffix_start(sequence: list[int], suffix: list[int]) -> int:
    if not suffix or len(suffix) > len(sequence):
        raise ValueError("target token sequence is absent")
    for start in range(len(sequence) - len(suffix), -1, -1):
        if sequence[start : start + len(suffix)] == suffix:
            return start
    raise ValueError("target token sequence is not present in the processed prompt")


def _tensor_stats(tensor) -> dict[str, float]:
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "l1": float(tensor.abs().sum().item()),
        "l2": float(tensor.float().norm().item()),
    }


def run(model_dir: Path, output_path: Path) -> None:
    import torch
    import torch.distributed as dist
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise ValueError(f"expected {WORLD_SIZE} ranks, received {world_size}")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    config_path = model_dir / "config.json"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    distributed = DistributedConfig(fsdp_size=world_size)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        distributed_config=distributed,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    # Sharded checkpoint loading allocates large temporary staging buffers. Give
    # those blocks back before FSDP's first layer all-gather.
    torch.cuda.empty_cache()
    dist.barrier()

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    # A deterministic, non-sensitive image with structure in all three channels.
    axis = torch.arange(512, dtype=torch.uint8)
    red = axis.repeat(512, 1)
    green = axis.view(-1, 1).repeat(1, 512)
    blue = ((red.to(torch.int16) + green.to(torch.int16)) // 2).to(torch.uint8)
    image_array = torch.stack((red, green, blue), dim=-1).numpy()
    image = Image.fromarray(image_array, mode="RGB")

    user_text = "Identify the requested safe color word from this synthetic image."
    target_text = " azure"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eos = processor.tokenizer.eos_token or ""
    full_text = prompt + target_text + eos
    encoded = processor(images=[image], text=[full_text], return_tensors="pt")
    target_ids = processor.tokenizer(target_text + eos, add_special_tokens=False)["input_ids"]
    input_ids_list = encoded["input_ids"][0].tolist()
    target_start = _suffix_start(input_ids_list, target_ids)
    labels = torch.full_like(encoded["input_ids"], -100)
    labels[:, target_start : target_start + len(target_ids)] = encoded["input_ids"][
        :, target_start : target_start + len(target_ids)
    ]
    if int((labels != -100).sum().item()) != len(target_ids):
        raise ValueError("target label mask is malformed")

    pixel_values = encoded.pop("pixel_values")
    mean = torch.tensor(model.config.norm_mean, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(model.config.norm_std, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    raw_pixels = (pixel_values * std + mean).clamp(0.0, 1.0).to(device).detach().requires_grad_(True)
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "image_flags"}
    }
    labels = labels.to(device)
    mean = mean.to(device)
    std = std.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    normalized_pixels = (raw_pixels - mean) / std
    output = model(**model_inputs, pixel_values=normalized_pixels, labels=labels, use_cache=False)
    loss_before = output.loss
    if loss_before is None or not torch.isfinite(loss_before):
        raise ValueError("Super VL target loss is missing or non-finite")
    gradient = torch.autograd.grad(loss_before, raw_pixels, retain_graph=False, create_graph=False)[0]
    if gradient is None or not torch.isfinite(gradient).all() or not bool((gradient != 0).any()):
        raise ValueError("Super VL pixel gradient is missing, non-finite, or zero")

    step_size = 1.0 / 255.0
    updated_pixels = (raw_pixels.detach() - step_size * gradient.sign()).clamp(0.0, 1.0)
    with torch.no_grad():
        updated_output = model(
            **model_inputs,
            pixel_values=(updated_pixels - mean) / std,
            labels=labels,
            use_cache=False,
        )
    loss_after = updated_output.loss
    if loss_after is None or not torch.isfinite(loss_after):
        raise ValueError("updated Super VL target loss is missing or non-finite")

    local = {
        "rank": rank,
        "loss_before": float(loss_before.item()),
        "loss_after_one_step": float(loss_after.item()),
        "gradient": _tensor_stats(gradient),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        losses = [item["loss_before"] for item in gathered if item is not None]
        losses_after = [item["loss_after_one_step"] for item in gathered if item is not None]
        gradient_l2 = [item["gradient"]["l2"] for item in gathered if item is not None]
        if len(gathered) != world_size or not all(item is not None for item in gathered):
            raise ValueError("missing distributed rank receipt")
        loss_spread = max(losses) - min(losses)
        loss_after_spread = max(losses_after) - min(losses_after)
        loss_scale = max(abs(sum(losses) / len(losses)), 1e-12)
        loss_after_scale = max(abs(sum(losses_after) / len(losses_after)), 1e-12)
        loss_relative_spread = loss_spread / loss_scale
        loss_after_relative_spread = loss_after_spread / loss_after_scale
        gradient_scale = max(abs(sum(gradient_l2) / len(gradient_l2)), 1e-12)
        gradient_relative_spread = (max(gradient_l2) - min(gradient_l2)) / gradient_scale
        loss_decreased_all_ranks = all(after < before for before, after in zip(losses, losses_after, strict=True))
        print(
            json.dumps(
                {
                    "event": "gradient_canary_rank_summary",
                    "loss_min": min(losses),
                    "loss_max": max(losses),
                    "loss_after_min": min(losses_after),
                    "loss_after_max": max(losses_after),
                    "gradient_l2_min": min(gradient_l2),
                    "gradient_l2_max": max(gradient_l2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if loss_relative_spread > 1e-4:
            raise ValueError("distributed ranks disagree on the initial target loss")
        if gradient_relative_spread > 0.02:
            raise ValueError("distributed rank gradient norms differ by more than two percent")
        if not loss_decreased_all_ranks:
            raise ValueError("one signed pixel step did not lower loss on every distributed rank")
        receipt = {
            "schema_version": 1,
            "status": "passed",
            "model_version": MODEL_VERSION,
            "model_config_sha256": config_sha256,
            "world_size": world_size,
            "distributed_strategy": "transformers_distributed_config_fsdp2",
            "model_parameters_frozen": True,
            "autograd_input": "raw_rgb_pixels",
            "image_shape": list(raw_pixels.shape),
            "image_sha256": hashlib.sha256(image_array.tobytes()).hexdigest(),
            "input_token_count": len(input_ids_list),
            "target_token_count": len(target_ids),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "target_sha256": hashlib.sha256(target_text.encode()).hexdigest(),
            "loss_before": losses[0],
            "loss_after_one_step": losses_after[0],
            "loss_changed": all(
                not math.isclose(before, after, rel_tol=0.0, abs_tol=1e-7)
                for before, after in zip(losses, losses_after, strict=True)
            ),
            "loss_rank_spread": loss_spread,
            "loss_rank_relative_spread": loss_relative_spread,
            "loss_after_rank_spread": loss_after_spread,
            "loss_after_rank_relative_spread": loss_after_relative_spread,
            "loss_decreased_all_ranks": loss_decreased_all_ranks,
            "gradient_l2_rank_relative_spread": gradient_relative_spread,
            "step_size": step_size,
            "gradient_finite": True,
            "gradient_nonzero": min(gradient_l2) > 0.0,
            "per_rank": gathered,
            "torch_version": torch.__version__,
        }
        output_path.parent.mkdir(parents=True, exist_ok=False)
        output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.output)


if __name__ == "__main__":
    main()
