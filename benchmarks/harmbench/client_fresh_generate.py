# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned upstream PAIR or TAP while targeting a verified client endpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
SUPPORTED = {"PAIR", "TAP"}
SENTINEL_MODEL = "gpt-client-fresh"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ClientTarget:
    """Minimal HarmBench language-model interface for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        revision: str,
        request_json: Callable[[str, dict[str, Any] | None], dict[str, Any]] | None = None,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ValueError("client target requires an HTTPS base URL")
        if not api_key:
            raise ValueError("client target API key is unavailable")
        if not model or not revision:
            raise ValueError("client target model and revision are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.revision = revision
        self._request_override = request_json
        self.calls: list[dict[str, Any]] = []
        self.catalog_sha256 = self._verify_catalog()

    def _request_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._request_override is not None:
            return self._request_override(path, payload)
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="GET" if body is None else "POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    return json.loads(response.read())
            except (TimeoutError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _verify_catalog(self) -> str:
        payload = self._request_json("/v1/models")
        model_ids = sorted(row.get("id") for row in payload.get("data", []) if row.get("id"))
        if self.model not in model_ids:
            raise ValueError("client target model is absent from the served model catalog")
        return text_sha256(json.dumps(model_ids, separators=(",", ":")))

    def batched_generate(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        max_n_tokens: int,
        temperature: float,
        stop_tokens: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        def generate_one(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
            if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
                raise ValueError("client target requires OpenAI message dictionaries")
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_n_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if stop_tokens:
                payload["stop"] = stop_tokens
            response = self._request_json("/v1/chat/completions", payload)
            if response.get("model") != self.model:
                raise ValueError("client target response model identity changed")
            text = response["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("client target returned a non-text completion")
            return text, {
                "request_sha256": text_sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
                "response_sha256": text_sha256(text),
                "response_id_sha256": text_sha256(str(response.get("id", ""))),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "model": response["model"],
            }

        concurrency = int(os.environ.get("HARMBENCH_CLIENT_TARGET_CONCURRENCY", "1"))
        if concurrency < 1:
            raise ValueError("HARMBENCH_CLIENT_TARGET_CONCURRENCY must be positive")
        if concurrency == 1 or len(conversations) < 2:
            generated = [generate_one(messages) for messages in conversations]
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(conversations))) as pool:
                generated = list(pool.map(generate_one, conversations))
        outputs = [text for text, _receipt in generated]
        self.calls.extend(receipt for _text, receipt in generated)
        return outputs


def validate_upstream(upstream: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"HarmBench checkout is {head}, expected {UPSTREAM_REVISION}")
    return head


def build_method_config(upstream: Path, method: str) -> tuple[dict[str, Any], Path]:
    if method not in SUPPORTED:
        raise ValueError(f"client-fresh method must be one of {sorted(SUPPORTED)}")
    path = upstream / f"configs/method_configs/{method}_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["default_method_hyperparameters"]
    config["target_model"] = {
        "model_name_or_path": SENTINEL_MODEL,
        "fschat_template": "chatgpt",
    }
    return config, path


def validate_binding_receipt(
    path: Path,
    *,
    client_base_url: str,
    client_model: str,
    client_revision: str,
) -> str:
    """Require an independently persisted deployment binding for the endpoint checkpoint."""
    receipt = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "model": client_model,
        "revision": client_revision,
        "endpoint_base_url_sha256": text_sha256(client_base_url.rstrip("/")),
    }
    observed = {
        "model": receipt.get("model_id", receipt.get("model")),
        "revision": receipt.get("model_revision", receipt.get("revision")),
        "endpoint_base_url_sha256": receipt.get("endpoint_base_url_sha256"),
    }
    if receipt.get("status") not in {"completed", "verified"} or observed != expected:
        raise ValueError("client deployment-binding receipt does not match the requested endpoint checkpoint")
    if not receipt.get("deployment_app_id"):
        raise ValueError("client deployment-binding receipt is missing the deployment app ID")
    return sha256(path)


def run(
    *,
    upstream: Path,
    method: str,
    behaviors: Path,
    output_dir: Path,
    client_base_url: str,
    client_model: str,
    client_revision: str,
    api_key_env: str,
    client_binding_receipt: Path,
) -> Path:
    """Run a resumable full upstream method and write payload-free target-binding receipts."""
    validate_upstream(upstream)
    if not api_key_env or api_key_env not in os.environ:
        raise ValueError("configured client API key environment variable is unavailable")
    config, config_path = build_method_config(upstream, method)
    binding_receipt_sha256 = validate_binding_receipt(
        client_binding_receipt,
        client_base_url=client_base_url,
        client_model=client_model,
        client_revision=client_revision,
    )
    with behaviors.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("client-fresh behavior source is empty")

    target = ClientTarget(
        base_url=client_base_url,
        api_key=os.environ[api_key_env],
        model=client_model,
        revision=client_revision,
    )
    sys.path.insert(0, str(upstream))
    from baselines import get_method_class, init_method  # type: ignore[import-not-found]

    conversers_name = "baselines.pair.conversers" if method == "PAIR" else "baselines.tap.conversers"
    conversers = __import__(conversers_name, fromlist=["load_indiv_model"])
    original_loader = conversers.load_indiv_model

    def load_model(*, model_name_or_path: str, **kwargs: Any) -> Any:
        if model_name_or_path == SENTINEL_MODEL:
            return target
        return original_loader(model_name_or_path=model_name_or_path, **kwargs)

    conversers.load_indiv_model = load_model
    method_class = get_method_class(method)
    instance = init_method(method_class, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir = output_dir / "client-call-receipts"
    receipt_dir.mkdir(exist_ok=True)

    for row in rows:
        behavior_id = row["BehaviorID"]
        case_path = Path(method_class.get_output_file_path(output_dir, behavior_id, "test_cases"))
        call_receipt_path = receipt_dir / f"{behavior_id}.json"
        if case_path.is_file() and call_receipt_path.is_file():
            continue
        before = len(target.calls)
        cases, logs = instance.generate_test_cases([row], verbose=False)
        instance.save_test_cases(output_dir, cases, logs, method_config=config)
        new_calls = target.calls[before:]
        if not new_calls:
            raise ValueError(f"client target was not queried for behavior {behavior_id}")
        call_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "behavior_id": behavior_id,
                    "client_model": client_model,
                    "client_revision": client_revision,
                    "calls": new_calls,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    instance.merge_test_cases(output_dir)
    cases_path = output_dir / "test_cases.json"
    if not cases_path.is_file():
        raise FileNotFoundError("upstream client-fresh generation did not produce merged test cases")
    call_receipts = sorted(receipt_dir.glob("*.json"))
    if len(call_receipts) != len(rows):
        raise ValueError(f"client-fresh call receipts cover {len(call_receipts)}/{len(rows)} behaviors")
    total_calls = 0
    receipt_hashes = {}
    for path in call_receipts:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "completed" or value.get("client_model") != client_model:
            raise ValueError(f"invalid client call receipt: {path.name}")
        total_calls += len(value["calls"])
        receipt_hashes[value["behavior_id"]] = sha256(path)
    final = {
        "schema_version": 1,
        "status": "completed",
        "client_target_verified": True,
        "client_model": client_model,
        "client_revision": client_revision,
        "source_target_model": client_model,
        "endpoint_base_url_sha256": text_sha256(client_base_url.rstrip("/")),
        "served_model_catalog_sha256": target.catalog_sha256,
        "client_binding_receipt_sha256": binding_receipt_sha256,
        "behaviors_sha256": sha256(behaviors),
        "method_config_sha256": sha256(config_path),
        "test_cases_sha256": sha256(cases_path),
        "behavior_receipts": len(call_receipts),
        "target_calls": total_calls,
        "behavior_receipt_sha256": receipt_hashes,
    }
    target_path = output_dir / "client-target-receipt.json"
    target_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--client-base-url", required=True)
    parser.add_argument("--client-model", required=True)
    parser.add_argument("--client-revision", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--client-binding-receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = run(
        upstream=args.upstream,
        method=args.method,
        behaviors=args.behaviors,
        output_dir=args.output_dir,
        client_base_url=args.client_base_url,
        client_model=args.client_model,
        client_revision=args.client_revision,
        api_key_env=args.api_key_env,
        client_binding_receipt=args.client_binding_receipt,
    )
    print(json.dumps({"status": "completed", "receipt": str(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
