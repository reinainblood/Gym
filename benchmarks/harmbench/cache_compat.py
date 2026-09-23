# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility boundary for public HarmBench GCG and Transformers 5 caches."""

from __future__ import annotations

import copy
import gc
from typing import Any


def _clone_cache_value(value: Any) -> Any:
    """Clone cache state while turning autograd-produced tensors into leaves."""
    try:
        import torch
    except ImportError:  # Unit tests for the installation boundary do not require torch.
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    return value


def clone_transformers_cache(cache: Any) -> Any:
    """Clone a Transformers 5 Cache without deepcopying non-leaf tensors."""
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("Transformers cache does not expose a layers list")
    cloned = copy.copy(cache)
    cloned.layers = []
    for layer in layers:
        cloned_layer = copy.copy(layer)
        for name, value in vars(layer).items():
            setattr(cloned_layer, name, _clone_cache_value(value))
        cloned.layers.append(cloned_layer)
    return cloned


def _repeat_batch_value(value: Any, repeats: int) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == 1:
            return value.repeat_interleave(repeats, dim=0)
        return value
    if isinstance(value, dict):
        return {key: _repeat_batch_value(item, repeats) for key, item in value.items()}
    if isinstance(value, list):
        return [_repeat_batch_value(item, repeats) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_batch_value(item, repeats) for item in value)
    return value


def repeat_hybrid_cache_batch(cache: Any, repeats: int) -> None:
    """Expand attention and Mamba state from batch one to candidate batch."""
    if repeats < 1:
        raise ValueError("cache repeat count must be positive")
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("Transformers cache does not expose a layers list")
    for layer in layers:
        for name, value in vars(layer).items():
            setattr(layer, name, _repeat_batch_value(value, repeats))


def install_gcg_dynamic_cache_adapter(method: Any) -> bool:
    """Use Transformers 5 Cache batching without changing GCG's loss math.

    HarmBench's pinned GCG code indexes ``past_key_values`` as the legacy tuple
    returned by the Transformers version used by the public release.
    Transformers 5 returns a ``DynamicCache`` and no longer accepts legacy
    tuples. This replaces only ``compute_candidates_loss`` with the same
    candidate slicing, logits shift, cross entropy, and mean reduction while
    cloning and batch-expanding the native cache object.
    """
    model = getattr(method, "model", None)
    if model is None or not callable(model):
        raise TypeError("HarmBench method does not expose a callable model")

    def compute_candidates_loss(search_batch_size: int, input_embeds: Any, target_ids: Any) -> Any:
        import torch
        from torch.nn import CrossEntropyLoss

        if method.search_batch_size != search_batch_size:
            method.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()
        prefix_cache = method.prefix_cache
        all_loss = []
        for start in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[start : start + search_batch_size]
                prefix_cache_batch = clone_transformers_cache(prefix_cache)
                repeat_hybrid_cache_batch(prefix_cache_batch, input_embeds_batch.shape[0])
                outputs = model(
                    inputs_embeds=input_embeds_batch,
                    past_key_values=prefix_cache_batch,
                )
            logits = outputs.logits
            target_start = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., target_start - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            loss = CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)
            del outputs, logits, loss, prefix_cache_batch
            torch.cuda.empty_cache()
            gc.collect()
        return torch.cat(all_loss, dim=0)

    method.compute_candidates_loss = compute_candidates_loss
    return True
