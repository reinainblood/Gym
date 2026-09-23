# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove a real target-loss gradient through the served Kimi K3 text embedding path."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


MODEL_ID = "moonshotai/Kimi-K3"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
TOTAL_BYTES = 1_560_998_984_390
GPU_COUNT = 8


def run(snapshot: Path, manifest_path: Path, output: Path) -> None:
    import torch
    from transformers.utils import generic

    if not hasattr(generic, "OutputRecorder"):

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

    from kimi_k3_harmbench_text_runner import prepare_loader_view
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != GPU_COUNT:
        raise ValueError(f"expected {GPU_COUNT} visible GPUs, received {torch.cuda.device_count()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": REVISION,
        "file_count": 118,
        "total_bytes": TOTAL_BYTES,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if manifest.get("status") not in {"verified", "preflight_verified"}:
        mismatch["status"] = (manifest.get("status"), "verified|preflight_verified")
    if mismatch:
        raise ValueError(f"checkpoint manifest mismatch: {mismatch}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    loader_view = prepare_loader_view(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        loader_view,
        trust_remote_code=True,
        local_files_only=True,
        dtype="auto",
        device_map="auto",
        max_memory={index: "278GiB" for index in range(GPU_COUNT)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(loader_view, trust_remote_code=True, local_files_only=True)
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    prompt = "Return the name of the blue-green color in one word."
    target = " teal"
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prompt_embeds = embedding(prompt_ids).detach()
    target_embeds = embedding(target_ids).detach()
    optim = prompt_embeds[:, -1:, :].clone().requires_grad_(True)
    inputs = torch.cat((prompt_embeds[:, :-1], optim, target_embeds), dim=1)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    labels[:, -target_ids.shape[1] :] = target_ids
    result = model(inputs_embeds=inputs, labels=labels, use_cache=False)
    loss = result.loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("target loss is absent or non-finite")
    gradient = torch.autograd.grad(loss, optim)[0]
    if not torch.isfinite(gradient).all() or not bool((gradient != 0).any()):
        raise ValueError("text input gradient is absent, zero, or non-finite")
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "loader_config_sha256": hashlib.sha256((loader_view / "config.json").read_bytes()).hexdigest(),
        "autograd_input": "text_input_embedding",
        "loss": float(loss.item()),
        "gradient_l1": float(gradient.abs().sum().item()),
        "gradient_l2": float(gradient.float().norm().item()),
        "gradient_finite": True,
        "gradient_nonzero": True,
        "gpu_count": GPU_COUNT,
        "hf_device_map": {key: str(value) for key, value in model.hf_device_map.items()},
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "torch_version": str(torch.__version__),
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.snapshot, args.manifest, args.output)


if __name__ == "__main__":
    main()
