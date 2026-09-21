# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate exact-BF16 Qwen target completions for a finalized full-corpus GCG artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
MODEL_REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"  # pragma: allowlist secret
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
EXPERIMENT = "nemotron_3_5_qwen_gcg"
PUBLIC_BEHAVIORS = 400
PUBLIC_STEPS = 500
PUBLIC_SEARCH_WIDTH = 512
MAX_NEW_TOKENS = 512
SNAPSHOT = Path("/weights/snapshot")
BEHAVIORS = Path("/app/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def generation_paths(output_root: Path) -> tuple[Path, Path]:
    cases = output_root / "GCG" / EXPERIMENT / "test_cases" / "test_cases.json"
    receipt = output_root / "generation-receipt.json"
    return cases, receipt


def load_generation_artifact(
    *, output_root: Path, behaviors_path: Path, artifact_id: str
) -> tuple[list[dict[str, str]], dict[str, list[str]], dict[str, Any]]:
    """Validate the finalized Qwen GCG artifact without exposing attack payloads."""
    cases_path, receipt_path = generation_paths(output_root)
    if not cases_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("finalized Qwen GCG cases and generation receipt are required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "method": "GCG",
        "upstream_method": "GCG",
        "upstream_revision": UPSTREAM_REVISION,
        "experiment": EXPERIMENT,
        "run_id": artifact_id,
        "target_type": "text_weights",
        "source_target_model": MODEL_ID,
        "source_target_revision": MODEL_REVISION,
        "test_cases_sha256": sha256(cases_path),
        "behaviors_sha256": sha256(behaviors_path),
        "behaviors": PUBLIC_BEHAVIORS,
        "cases": PUBLIC_BEHAVIORS,
        "num_steps": PUBLIC_STEPS,
        "search_width": PUBLIC_SEARCH_WIDTH,
    }
    mismatched = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatched:
        raise ValueError(f"Qwen GCG generation receipt failed fields {','.join(sorted(mismatched))}")
    validate_generation_shard_evidence(output_root=output_root, receipt=receipt)

    with behaviors_path.open(newline="", encoding="utf-8") as stream:
        behaviors = list(csv.DictReader(stream))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len(behaviors) != PUBLIC_BEHAVIORS or not isinstance(cases, dict) or len(cases) != PUBLIC_BEHAVIORS:
        raise ValueError("Qwen GCG completion requires the complete 400-behavior public corpus")
    behavior_ids = [row["BehaviorID"] for row in behaviors]
    if set(cases) != set(behavior_ids) or any(
        not isinstance(cases[key], list)
        or len(cases[key]) != 1
        or not isinstance(cases[key][0], str)
        or not cases[key][0].strip()
        for key in cases
    ):
        raise ValueError("Qwen GCG cases must contain exactly one case for every public behavior")
    return behaviors, cases, receipt


def validate_generation_shard_evidence(*, output_root: Path, receipt: dict[str, Any]) -> None:
    """Read back the finalized Qwen attack shard receipts before completing targets."""
    num_shards = receipt.get("num_shards")
    manifest = receipt.get("shard_receipts")
    if not isinstance(num_shards, int) or not 1 <= num_shards <= 16:
        raise ValueError("Qwen GCG generation receipt has an invalid shard count")
    if not isinstance(manifest, list) or len(manifest) != num_shards:
        raise ValueError("Qwen GCG generation receipt has an incomplete shard manifest")
    expected_names = [f"qwen-{index:02d}-of-{num_shards:02d}.json" for index in range(num_shards)]
    observed_names = [row.get("name") if isinstance(row, dict) else None for row in manifest]
    if observed_names != expected_names:
        raise ValueError("Qwen GCG generation shard manifest is out of source order")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if receipt.get("shard_receipts_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Qwen GCG generation shard manifest hash mismatch")
    shard_dir = output_root / "shard-receipts"
    for row in manifest:
        digest = row.get("sha256")
        path = shard_dir / row["name"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Qwen GCG generation shard manifest contains an invalid hash")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Qwen GCG generation shard receipt readback failed")


def shard_indexes(shard_index: int, num_shards: int, limit: int = 0) -> list[int]:
    if not 0 <= shard_index < num_shards <= 16:
        raise ValueError("invalid completion shard selection")
    selected = list(range(shard_index, PUBLIC_BEHAVIORS, num_shards))
    return selected[:limit] if limit else selected


def render_upstream_prompt(tokenizer: Any, attack: str) -> str:
    """Reproduce HarmBench get_template's default tokenizer-template branch."""
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "{instruction}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "")
    return template.format(instruction=attack)


class QwenGCGCompletionRuntime:
    """Generation-only runtime bound to the exact Qwen BF16 checkpoint."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not SNAPSHOT.is_dir():
            raise FileNotFoundError(f"missing exact Qwen snapshot: {SNAPSHOT}")
        manifest = json.loads((SNAPSHOT.parent / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "verified"
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("revision") != MODEL_REVISION
        ):
            raise ValueError("Qwen BF16 checkpoint manifest is absent or mismatched")
        self.config_sha256 = sha256(SNAPSHOT / "config.json")
        self.index_sha256 = sha256(SNAPSHOT / "model.safetensors.index.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            max_memory={0: "184GiB", 1: "184GiB"},
            low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.torch = torch
        torch.cuda.empty_cache()

    def complete(
        self,
        *,
        artifact_id: str,
        index: int,
        behavior: dict[str, str],
        attack: str,
        generation_receipt_sha256: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_path = output_dir / f"{index:03d}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "completed"
                or existing.get("artifact_id") != artifact_id
                or existing.get("index") != index
            ):
                raise ValueError(f"invalid existing Qwen GCG completion receipt at index {index}")
            return {"index": index, "status": "already_complete"}

        prompt = render_upstream_prompt(self.tokenizer, attack)
        encoded = self.tokenizer([prompt], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        generated = output[:, prompt_tokens:]
        generation_tokens = int(generated.shape[1])
        generation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        receipt = {
            "schema_version": 1,
            "artifact_kind": "gcg_target_completion",
            "status": "completed",
            "artifact_id": artifact_id,
            "method": "GCG",
            "index": index,
            "behavior_id": behavior["BehaviorID"],
            "behavior": behavior["Behavior"],
            "context": behavior.get("ContextString", ""),
            "functional_category": behavior["FunctionalCategory"],
            "semantic_category": behavior["SemanticCategory"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_config_sha256": self.config_sha256,
            "checkpoint_index_sha256": self.index_sha256,
            "generation_receipt_sha256": generation_receipt_sha256,
            "attack_sha256": text_sha256(attack),
            "rendered_prompt_sha256": text_sha256(prompt),
            "prompt_token_count": prompt_tokens,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_token_count": generation_tokens,
            "finish_reason": "length" if generation_tokens >= MAX_NEW_TOKENS else "stop",
            "sampling": {"do_sample": False},
            "generation": generation,
            "generation_sha256": text_sha256(generation),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "index": index,
            "status": "completed",
            "generation_token_count": generation_tokens,
            "finish_reason": receipt["finish_reason"],
        }


def write_completion_shard_receipt(
    *,
    output_root: Path,
    artifact_id: str,
    shard_index: int,
    num_shards: int,
    selected: list[int],
    behaviors_path: Path = BEHAVIORS,
) -> dict[str, Any]:
    expected_selected = shard_indexes(shard_index, num_shards)
    if selected != expected_selected:
        raise ValueError("canonical Qwen GCG completion receipt requires the complete source-order shard")
    completion_dir = output_root / "target-completions" / "qwen-bf16" / "individual"
    cases_path, generation_receipt_path = generation_paths(output_root)
    behaviors, cases, _ = load_generation_artifact(
        output_root=output_root,
        behaviors_path=behaviors_path,
        artifact_id=artifact_id,
    )
    generation_receipt_sha256 = sha256(generation_receipt_path)
    individual_manifest = []
    checkpoint_configs: set[str] = set()
    checkpoint_indexes: set[str] = set()
    for index in selected:
        path = completion_dir / f"{index:03d}.json"
        if not path.is_file():
            raise ValueError("Qwen GCG completion shard is missing one or more receipts")
        behavior = behaviors[index]
        attack = cases[behavior["BehaviorID"]][0]
        completion = validate_individual_completion(
            path=path,
            artifact_id=artifact_id,
            index=index,
            behavior_id=behavior["BehaviorID"],
            attack=attack,
            generation_receipt_sha256=generation_receipt_sha256,
        )
        checkpoint_configs.add(completion["checkpoint_config_sha256"])
        checkpoint_indexes.add(completion["checkpoint_index_sha256"])
        individual_manifest.append({"name": path.name, "sha256": sha256(path)})
    if len(checkpoint_configs) != 1 or len(checkpoint_indexes) != 1:
        raise ValueError("Qwen GCG completion shard mixes checkpoint identities")
    canonical_manifest = json.dumps(individual_manifest, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "artifact_id": artifact_id,
        "target": "qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "GCG",
        "max_new_tokens": MAX_NEW_TOKENS,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "selected_behaviors": len(selected),
        "completed_behaviors": len(selected),
        "indexes_sha256": hashlib.sha256("\n".join(map(str, selected)).encode()).hexdigest(),
        "test_cases_sha256": sha256(cases_path),
        "generation_receipt_sha256": generation_receipt_sha256,
        "checkpoint_config_sha256": next(iter(checkpoint_configs)),
        "checkpoint_index_sha256": next(iter(checkpoint_indexes)),
        "individual_receipts": individual_manifest,
        "individual_receipts_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
    }
    receipt_path = (
        output_root
        / "target-completions"
        / "qwen-bf16"
        / "shard-receipts"
        / f"qwen-{shard_index:02d}-of-{num_shards:02d}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate_individual_completion(
    *,
    path: Path,
    artifact_id: str,
    index: int,
    behavior_id: str,
    attack: str,
    generation_receipt_sha256: str,
) -> dict[str, Any]:
    """Read back one private completion while exposing no payload in errors."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "artifact_kind": "gcg_target_completion",
        "status": "completed",
        "artifact_id": artifact_id,
        "method": "GCG",
        "index": index,
        "behavior_id": behavior_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "generation_receipt_sha256": generation_receipt_sha256,
        "attack_sha256": text_sha256(attack),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    mismatched = [key for key, value in required.items() if receipt.get(key) != value]
    for key in ("checkpoint_config_sha256", "checkpoint_index_sha256", "rendered_prompt_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            mismatched.append(key)
    generation = receipt.get("generation")
    if not isinstance(generation, str) or receipt.get("generation_sha256") != text_sha256(generation):
        mismatched.append("generation_sha256")
    if mismatched:
        raise ValueError(f"Qwen GCG completion {index} failed fields {','.join(sorted(set(mismatched)))}")
    return receipt


def finalize_completion_artifact(*, output_root: Path, behaviors_path: Path, artifact_id: str) -> dict[str, Any]:
    """Reconcile exactly 400 Qwen BF16 completions and their shard receipts."""
    behaviors, cases, _ = load_generation_artifact(
        output_root=output_root,
        behaviors_path=behaviors_path,
        artifact_id=artifact_id,
    )
    _, generation_receipt_path = generation_paths(output_root)
    generation_receipt_sha256 = sha256(generation_receipt_path)
    base = output_root / "target-completions" / "qwen-bf16"
    individual_dir = base / "individual"
    expected_paths = [individual_dir / f"{index:03d}.json" for index in range(PUBLIC_BEHAVIORS)]
    observed_paths = sorted(individual_dir.glob("*.json"))
    if observed_paths != expected_paths:
        raise ValueError("Qwen GCG completion finalizer requires exactly 400 indexed receipts")

    individual_manifest = []
    checkpoint_configs: set[str] = set()
    checkpoint_indexes: set[str] = set()
    for index, path in enumerate(expected_paths):
        behavior = behaviors[index]
        receipt = validate_individual_completion(
            path=path,
            artifact_id=artifact_id,
            index=index,
            behavior_id=behavior["BehaviorID"],
            attack=cases[behavior["BehaviorID"]][0],
            generation_receipt_sha256=generation_receipt_sha256,
        )
        checkpoint_configs.add(receipt["checkpoint_config_sha256"])
        checkpoint_indexes.add(receipt["checkpoint_index_sha256"])
        individual_manifest.append({"name": path.name, "sha256": sha256(path)})
    if len(checkpoint_configs) != 1 or len(checkpoint_indexes) != 1:
        raise ValueError("Qwen GCG completion receipts disagree on checkpoint identity")

    shard_dir = base / "shard-receipts"
    shard_paths = sorted(shard_dir.glob("qwen-*-of-*.json"))
    if not shard_paths:
        raise ValueError("Qwen GCG completion finalizer requires shard receipts")
    shard_receipts = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    shard_counts = {row.get("num_shards") for row in shard_receipts}
    if len(shard_counts) != 1:
        raise ValueError("Qwen GCG completion shard receipts disagree on num_shards")
    num_shards = shard_counts.pop()
    if not isinstance(num_shards, int) or not 1 <= num_shards <= 16:
        raise ValueError("Qwen GCG completion shard count is invalid")
    expected_shard_paths = [shard_dir / f"qwen-{index:02d}-of-{num_shards:02d}.json" for index in range(num_shards)]
    if shard_paths != expected_shard_paths:
        raise ValueError("Qwen GCG completion shard receipt set is incomplete")
    for index, receipt in enumerate(shard_receipts):
        selected = shard_indexes(index, num_shards)
        selected_manifest = [individual_manifest[value] for value in selected]
        canonical_selected = json.dumps(selected_manifest, sort_keys=True, separators=(",", ":")).encode()
        required = {
            "schema_version": 1,
            "status": "completed",
            "artifact_id": artifact_id,
            "target": "qwen",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "method": "GCG",
            "max_new_tokens": MAX_NEW_TOKENS,
            "shard_index": index,
            "num_shards": num_shards,
            "selected_behaviors": len(selected),
            "completed_behaviors": len(selected),
            "indexes_sha256": hashlib.sha256("\n".join(map(str, selected)).encode()).hexdigest(),
            "test_cases_sha256": sha256(generation_paths(output_root)[0]),
            "generation_receipt_sha256": generation_receipt_sha256,
            "checkpoint_config_sha256": next(iter(checkpoint_configs)),
            "checkpoint_index_sha256": next(iter(checkpoint_indexes)),
            "individual_receipts": selected_manifest,
            "individual_receipts_sha256": hashlib.sha256(canonical_selected).hexdigest(),
        }
        mismatched = [key for key, value in required.items() if receipt.get(key) != value]
        if mismatched:
            raise ValueError(f"Qwen GCG completion shard receipt {index} failed fields {','.join(sorted(mismatched))}")

    canonical_individual = json.dumps(individual_manifest, sort_keys=True, separators=(",", ":")).encode()
    shard_manifest = [{"name": path.name, "sha256": sha256(path)} for path in shard_paths]
    canonical_shards = json.dumps(shard_manifest, sort_keys=True, separators=(",", ":")).encode()
    final = {
        "schema_version": 1,
        "artifact_kind": "gcg_target_completion_manifest",
        "status": "completed",
        "artifact_id": artifact_id,
        "target": "qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "GCG",
        "behaviors": PUBLIC_BEHAVIORS,
        "completions": PUBLIC_BEHAVIORS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_receipt_sha256": generation_receipt_sha256,
        "checkpoint_config_sha256": next(iter(checkpoint_configs)),
        "checkpoint_index_sha256": next(iter(checkpoint_indexes)),
        "individual_receipts": individual_manifest,
        "individual_receipts_sha256": hashlib.sha256(canonical_individual).hexdigest(),
        "num_shards": num_shards,
        "shard_receipts": shard_manifest,
        "shard_receipts_sha256": hashlib.sha256(canonical_shards).hexdigest(),
    }
    output_path = base / "target-completion-receipt.json"
    output_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final
