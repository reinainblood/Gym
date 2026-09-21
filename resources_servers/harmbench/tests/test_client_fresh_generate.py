# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.client_fresh_generate import (
    SENTINEL_MODEL,
    ClientTarget,
    build_method_config,
    text_sha256,
    validate_binding_receipt,
)
from resources_servers.harmbench.tests.upstream_checkout import harmbench_upstream


UPSTREAM = harmbench_upstream()


@pytest.mark.parametrize("concurrency", [1, 4])
def test_client_target_preserves_messages_and_records_only_hashes(monkeypatch, concurrency):
    requests = []

    def request(path, payload):
        requests.append((path, payload))
        if path == "/v1/models":
            return {"data": [{"id": "client-model"}]}
        return {
            "id": "response-id",
            "model": "client-model",
            "choices": [{"message": {"content": "synthetic response"}, "finish_reason": "stop"}],
        }

    target = ClientTarget(
        base_url="https://client.example",
        api_key="test-only-key",
        model="client-model",
        revision="client-revision",
        request_json=request,
    )
    monkeypatch.setenv("HARMBENCH_CLIENT_TARGET_CONCURRENCY", str(concurrency))
    messages = [
        [{"role": "user", "content": "synthetic request"}],
        [{"role": "user", "content": "second synthetic request"}],
    ]
    assert target.batched_generate(messages, max_n_tokens=150, temperature=0.0, stop_tokens=["}"]) == [
        "synthetic response",
        "synthetic response",
    ]
    assert requests[1][1]["messages"] == messages[0]
    assert requests[1][1]["max_tokens"] == 150
    assert requests[1][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(target.calls) == 2
    assert set(target.calls[0]) == {
        "request_sha256",
        "response_sha256",
        "response_id_sha256",
        "finish_reason",
        "model",
    }
    assert "synthetic request" not in json.dumps(target.calls)
    assert "synthetic response" not in json.dumps(target.calls)


def test_client_target_rejects_wrong_served_identity():
    with pytest.raises(ValueError, match="absent from the served model catalog"):
        ClientTarget(
            base_url="https://client.example",
            api_key="test-only-key",
            model="expected",
            revision="revision",
            request_json=lambda *_: {"data": [{"id": "different"}]},
        )


def test_client_binding_receipt_binds_app_endpoint_model_and_revision(tmp_path):
    path = tmp_path / "binding.yaml"
    path.write_text(
        json.dumps(
            {
                "status": "verified",
                "deployment_app_id": "ap-test",
                "model_id": "client-model",
                "model_revision": "client-revision",
                "endpoint_base_url_sha256": text_sha256("https://client.example"),
            }
        ),
        encoding="utf-8",
    )
    assert (
        len(
            validate_binding_receipt(
                path,
                client_base_url="https://client.example",
                client_model="client-model",
                client_revision="client-revision",
            )
        )
        == 64
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_binding_receipt(
            path,
            client_base_url="https://different.example",
            client_model="client-model",
            client_revision="client-revision",
        )


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="optional pinned HarmBench checkout is absent")
@pytest.mark.parametrize("method", ["PAIR", "TAP"])
def test_client_fresh_config_preserves_public_defaults_and_replaces_only_target(method):
    config, path = build_method_config(UPSTREAM, method)
    source = __import__("yaml").safe_load(path.read_text())["default_method_hyperparameters"]
    assert {key: value for key, value in config.items() if key != "target_model"} == source
    assert config["target_model"] == {
        "model_name_or_path": SENTINEL_MODEL,
        "fschat_template": "chatgpt",
    }
