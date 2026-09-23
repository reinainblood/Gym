# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native differentiable Kimi K3 adapter for HarmBench MultiModalPGD."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class KimiK3HarmBenchModel:
    """Expose K3 through HarmBench's public multimodal-model interface."""

    def __init__(self, snapshot: Path):
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            dtype="auto",
            device_map="auto",
            max_memory={index: "278GiB" for index in range(8)},
            low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.patch_size = int(self.image_processor.media_proc_cfg["patch_size"])
        self.image_mean = torch.tensor(self.image_processor.media_proc_cfg["image_mean"]).view(3, 1, 1)
        self.image_std = torch.tensor(self.image_processor.media_proc_cfg["image_std"]).view(3, 1, 1)
        self._image_prompt = ""

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        resize = self.image_processor.get_resize_config({"type": "image", "image": image})
        array = self.image_processor.resize_image(
            image,
            resize["new_width"],
            resize["new_height"],
            resize["pad_width"],
            resize["pad_height"],
        )
        self._image_prompt = self.image_processor.make_image_prompt(width, height)
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    def _patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.model.vision_tower.patch_embed.proj.weight.device
        mean = self.image_mean.to(device=device, dtype=image.dtype)
        std = self.image_std.to(device=device, dtype=image.dtype)
        pixels = (image.to(device) - mean) / std
        channels, height, width = pixels.shape
        patch = self.patch_size
        if channels != 3 or height % patch or width % patch:
            raise ValueError(f"invalid K3 image tensor shape: {tuple(pixels.shape)}")
        patches = pixels.reshape(channels, height // patch, patch, width // patch, patch)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, channels, patch, patch)
        grid = torch.tensor([[1, height // patch, width // patch]], dtype=torch.long, device=device)
        return patches, grid

    def _text_inputs(self, behavior: str, target: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        messages = [{"role": "user", "content": f"<|kimi_image_placeholder|>{behavior}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            image_prompts=[self._image_prompt],
        )
        embedding_device = self.model.get_input_embeddings().weight.device
        input_ids = prompt["input_ids"].to(embedding_device)
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            embedding_device
        )
        input_ids = torch.cat((input_ids, target_ids), dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[:, -target_ids.shape[1] :] = target_ids
        return input_ids, attention_mask, labels

    def compute_loss(self, behavior: str, target: str, image_input: torch.Tensor) -> torch.Tensor:
        if not self._image_prompt:
            raise RuntimeError("preprocess must be called before compute_loss")
        pixel_values, grid_thws = self._patchify(image_input)
        input_ids, attention_mask, labels = self._text_inputs(behavior, target)
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
        )
        if result.loss is None or not torch.isfinite(result.loss):
            raise ValueError("K3 multimodal target loss is absent or non-finite")
        return result.loss
