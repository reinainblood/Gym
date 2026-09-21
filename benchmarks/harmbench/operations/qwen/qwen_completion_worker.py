# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate Qwen white-box target completions at HarmBench's 512-token protocol cap.

The attack workers wrote each case's target completion with ``max_new_tokens=256``, which is
``generate_completions.py``'s argparse default rather than the protocol HarmBench reports (see
commit 642f7092). This module re-runs completion only: it reloads each case's published PNG,
verifies the hash recorded in the parent receipt, and regenerates against the same BF16 checkpoint,
processor, and chat template. No attack optimization is repeated and no parent receipt is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
MODEL_REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"
SNAPSHOT = Path("/weights/snapshot")
RESULTS = Path("/results")
PROTOCOL_MAX_NEW_TOKENS = 512
COMPLETIONS_DIRNAME = "completions-512"
WHITEBOX_METHODS = {"MultiModalPGD", "MultiModalPGDPatch", "MultiModalPGDBlankImage"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def validate_completion_receipt(
    output_path: Path,
    parent_path: Path,
    *,
    checkpoint_config_sha256: str,
    checkpoint_index_sha256: str,
    processor_file_sha256: dict[str, str],
    rendered_prompt_sha256: str,
) -> dict:
    """Accept an existing canonical completion only when its full hash chain survives readback."""
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "artifact_kind": "target_completion",
        "status": "completed",
        "method": parent.get("method"),
        "public_method_class": parent.get("public_method_class"),
        "index": parent.get("index"),
        "behavior_id": parent.get("behavior_id"),
        "behavior": parent.get("behavior"),
        "context": parent.get("context", ""),
        "functional_category": parent.get("functional_category"),
        "semantic_category": parent.get("semantic_category"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "checkpoint_index_sha256": checkpoint_index_sha256,
        "processor_file_sha256": processor_file_sha256,
        "parent_receipt": parent_path.name,
        "parent_receipt_sha256": sha256(parent_path),
        "parent_generation_sha256": parent.get("generation_sha256"),
        "parent_max_new_tokens": 256,
        "test_case_image": parent.get("test_case_image"),
        "test_case_image_sha256": parent.get("test_case_image_sha256"),
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
        "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
    }
    mismatched = [key for key, value in required.items() if receipt.get(key) != value]
    image_path = Path(str(parent.get("test_case_image", "")))
    if not image_path.is_file() or parent.get("test_case_image_sha256") != sha256(image_path):
        mismatched.append("test_case_image_sha256")
    generation = receipt.get("generation")
    if not isinstance(generation, str) or receipt.get("generation_sha256") != text_sha256(generation):
        mismatched.append("generation_sha256")
    token_count = receipt.get("generation_token_count")
    if not isinstance(token_count, int) or not 0 <= token_count <= PROTOCOL_MAX_NEW_TOKENS:
        mismatched.append("generation_token_count")
    elif receipt.get("finish_reason") != ("length" if token_count >= PROTOCOL_MAX_NEW_TOKENS else "stop"):
        mismatched.append("finish_reason")
    if not isinstance(receipt.get("prompt_token_count"), int) or receipt["prompt_token_count"] <= 0:
        mismatched.append("prompt_token_count")
    if mismatched:
        raise ValueError(f"existing canonical completion failed fields {','.join(sorted(set(mismatched)))}")
    return receipt


def validate_attack_manifest(method_dir: Path, method: str) -> dict:
    """Rehash a finalized 110-case attack campaign before target completion."""
    if method not in WHITEBOX_METHODS:
        raise ValueError("unknown Qwen white-box method")
    manifest_path = method_dir / "attack-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Qwen white-box attack manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_attack_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "behaviors": 110,
        "cases": 110,
        "images": 110,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    cases = manifest.get("case_receipts")
    images = manifest.get("optimized_images")
    if not isinstance(cases, list) or len(cases) != 110:
        mismatched.append("case_receipts")
    if not isinstance(images, list) or len(images) != 110:
        mismatched.append("optimized_images")
    if mismatched:
        raise ValueError(f"Qwen white-box attack manifest failed fields {','.join(sorted(set(mismatched)))}")
    canonical_cases = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    canonical_images = json.dumps(images, sort_keys=True, separators=(",", ":")).encode()
    if manifest.get("case_receipts_sha256") != hashlib.sha256(canonical_cases).hexdigest():
        raise ValueError("Qwen white-box attack case manifest hash mismatch")
    if manifest.get("optimized_images_sha256") != hashlib.sha256(canonical_images).hexdigest():
        raise ValueError("Qwen white-box attack image manifest hash mismatch")
    observed_case_names = [path.name for path in sorted((method_dir / "cases").glob("*.json"))]
    observed_image_names = [path.name for path in sorted((method_dir / "images").glob("*.png"))]
    if observed_case_names != [row.get("name") for row in cases] or observed_image_names != [
        row.get("name") for row in images
    ]:
        raise ValueError("Qwen white-box attack manifest filenames disagree with the Volume")
    for directory, rows in ((method_dir / "cases", cases), (method_dir / "images", images)):
        for row in rows:
            path = directory / row["name"]
            if not path.is_file() or sha256(path) != row.get("sha256"):
                raise ValueError("Qwen white-box attack manifest file readback failed")
    return manifest


def finalize_completion_artifact(*, method_dir: Path, method: str) -> dict:
    """Reconcile exactly 110 canonical 512-token completions against finalized attacks."""
    attack_manifest = validate_attack_manifest(method_dir, method)
    parent_dir = method_dir / "cases"
    completion_dir = method_dir / COMPLETIONS_DIRNAME
    parent_paths = sorted(parent_dir.glob("*.json"))
    completion_paths = sorted(completion_dir.glob("*.json"))
    if len(parent_paths) != 110 or [path.name for path in completion_paths] != [path.name for path in parent_paths]:
        raise ValueError("canonical completion finalizer requires exactly 110 parent-matched receipts")
    first = json.loads(completion_paths[0].read_text(encoding="utf-8"))
    processor_hashes = first.get("processor_file_sha256")
    if not isinstance(processor_hashes, dict) or not processor_hashes:
        raise ValueError("canonical completion processor hash manifest is missing")
    if any(not isinstance(value, str) or len(value) != 64 for value in processor_hashes.values()):
        raise ValueError("canonical completion processor hash manifest is invalid")
    completion_manifest = []
    for parent_path, completion_path in zip(parent_paths, completion_paths, strict=True):
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        validate_completion_receipt(
            completion_path,
            parent_path,
            checkpoint_config_sha256=attack_manifest["checkpoint_config_sha256"],
            checkpoint_index_sha256=attack_manifest["checkpoint_index_sha256"],
            processor_file_sha256=processor_hashes,
            rendered_prompt_sha256=parent["prompt_sha256"],
        )
        completion_manifest.append({"name": completion_path.name, "sha256": sha256(completion_path)})
    canonical = json.dumps(completion_manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_completion_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "completions": 110,
        "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
        "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
        "attack_manifest_sha256": sha256(method_dir / "attack-manifest.json"),
        "checkpoint_config_sha256": attack_manifest["checkpoint_config_sha256"],
        "checkpoint_index_sha256": attack_manifest["checkpoint_index_sha256"],
        "processor_file_sha256": processor_hashes,
        "completion_receipts": completion_manifest,
        "completion_receipts_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    output_path = method_dir / "completion-manifest-512.json"
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return manifest


def validate_completion_manifest(method_dir: Path, method: str) -> dict:
    """Rehash a finalized canonical-completion campaign before scoring."""
    attack_manifest = validate_attack_manifest(method_dir, method)
    path = method_dir / "completion-manifest-512.json"
    if not path.is_file():
        raise FileNotFoundError("Qwen white-box completion manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "artifact_kind": "qwen_whitebox_completion_manifest",
        "status": "completed",
        "method": method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "completions": 110,
        "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
        "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
        "attack_manifest_sha256": sha256(method_dir / "attack-manifest.json"),
        "checkpoint_config_sha256": attack_manifest["checkpoint_config_sha256"],
        "checkpoint_index_sha256": attack_manifest["checkpoint_index_sha256"],
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    receipts = manifest.get("completion_receipts")
    if not isinstance(receipts, list) or len(receipts) != 110:
        mismatched.append("completion_receipts")
    if mismatched:
        raise ValueError(f"Qwen white-box completion manifest failed fields {','.join(sorted(set(mismatched)))}")
    canonical = json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()
    if manifest.get("completion_receipts_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Qwen white-box completion receipt manifest hash mismatch")
    completion_dir = method_dir / COMPLETIONS_DIRNAME
    observed_names = [file.name for file in sorted(completion_dir.glob("*.json"))]
    if observed_names != [row.get("name") for row in receipts]:
        raise ValueError("Qwen white-box completion manifest filenames disagree with the Volume")
    for row in receipts:
        completion_path = completion_dir / row["name"]
        if not completion_path.is_file() or sha256(completion_path) != row.get("sha256"):
            raise ValueError("Qwen white-box completion manifest file readback failed")
    return manifest


class QwenCompletionRuntime:
    """Generation-only runtime bound to the exact differentiable checkpoint."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not SNAPSHOT.is_dir():
            raise FileNotFoundError(f"missing exact Qwen snapshot: {SNAPSHOT}")
        manifest = json.loads((SNAPSHOT.parent / "manifest.json").read_text())
        if manifest.get("status") != "verified" or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("Qwen BF16 checkpoint manifest is absent or mismatched")
        self.config_sha256 = sha256(SNAPSHOT / "config.json")
        self.index_sha256 = sha256(SNAPSHOT / "model.safetensors.index.json")

        self.processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "184GiB", 1: "184GiB"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        # The attack runtime generated with the cache disabled; keep that identical so the token cap
        # is the only difference between the parent completion and this one.
        self.model.config.use_cache = False
        self.device = next(self.model.parameters()).device
        self.torch = torch
        self.processor_sha256 = self._processor_sha256()
        torch.cuda.empty_cache()

    def _processor_sha256(self) -> dict[str, str]:
        names = ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
        return {name: sha256(SNAPSHOT / name) for name in names if (SNAPSHOT / name).exists()}

    def render_prompt(self, behavior: str) -> str:
        """Reproduce the attack runtime's prompt exactly, trailing space included."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": behavior + " "}]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def complete_case(self, parent_path: Path, output_dir: Path) -> dict:
        from PIL import Image

        output_path = output_dir / parent_path.name
        if output_path.exists():
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            prompt = self.render_prompt(parent["behavior"])
            validate_completion_receipt(
                output_path,
                parent_path,
                checkpoint_config_sha256=self.config_sha256,
                checkpoint_index_sha256=self.index_sha256,
                processor_file_sha256=self.processor_sha256,
                rendered_prompt_sha256=text_sha256(prompt),
            )
            return {"case": parent_path.name, "status": "already_complete"}

        parent = json.loads(parent_path.read_text())
        if parent.get("status") != "completed":
            return {"case": parent_path.name, "status": "skipped_incomplete_parent"}

        image_path = Path(parent["test_case_image"])
        if not image_path.exists():
            raise FileNotFoundError(f"published test-case image is missing: {image_path}")
        observed_image_sha256 = sha256(image_path)
        if observed_image_sha256 != parent["test_case_image_sha256"]:
            raise ValueError(f"test-case image hash mismatch for {parent_path.name}")

        image = Image.open(image_path).convert("RGB")
        prompt = self.render_prompt(parent["behavior"])
        encoded = self.processor(images=[image], text=[prompt], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_token_count = int(encoded["input_ids"].shape[1])

        started = time.time()
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=PROTOCOL_MAX_NEW_TOKENS
            )
        generated = output[:, prompt_token_count:]
        generation_token_count = int(generated.shape[1])
        generation = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        receipt = {
            "schema_version": 1,
            "artifact_kind": "target_completion",
            "status": "completed",
            "method": parent["method"],
            "public_method_class": parent["public_method_class"],
            "index": parent["index"],
            "behavior_id": parent["behavior_id"],
            "behavior": parent["behavior"],
            "context": parent.get("context", ""),
            "functional_category": parent.get("functional_category"),
            "semantic_category": parent.get("semantic_category"),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_config_sha256": self.config_sha256,
            "checkpoint_index_sha256": self.index_sha256,
            "processor_file_sha256": self.processor_sha256,
            "parent_receipt": parent_path.name,
            "parent_receipt_sha256": sha256(parent_path),
            "parent_generation_sha256": parent.get("generation_sha256"),
            "parent_max_new_tokens": 256,
            "test_case_image": str(image_path),
            "test_case_image_sha256": observed_image_sha256,
            "rendered_prompt_sha256": text_sha256(prompt),
            "prompt_token_count": prompt_token_count,
            "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
            "generation_token_count": generation_token_count,
            "finish_reason": "length" if generation_token_count >= PROTOCOL_MAX_NEW_TOKENS else "stop",
            "generation": generation,
            "generation_sha256": text_sha256(generation),
            "sampling": {"do_sample": False, "num_beams": 1, "use_cache": False},
            "elapsed_seconds": time.time() - started,
            "upstream_protocol_note": (
                "HarmBench reports every experiment with max_new_tokens=512; the parent receipt's 256 "
                "is generate_completions.py's argparse default, not the protocol."
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
        return {
            "case": parent_path.name,
            "status": "completed",
            "generation_token_count": generation_token_count,
            "finish_reason": receipt["finish_reason"],
            "changed_from_parent": receipt["generation_sha256"] != parent.get("generation_sha256"),
        }
