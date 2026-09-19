# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string

from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.agentdojo_agent.model_bridge import (
    NeMoGymAgentDojoLLM,
    agentdojo_messages_to_response_output,
    agentdojo_messages_to_responses_input,
    response_to_agentdojo_message,
)


def test_agentdojo_transcript_maps_to_monotonic_responses_input() -> None:
    call = FunctionCall(function="read_file", args={"file_path": "bill.txt"}, id="call-1")
    messages = [
        {"role": "system", "content": [text_content_block_from_string("system")]},
        {"role": "user", "content": [text_content_block_from_string("read it")]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call": call,
            "tool_call_id": "call-1",
            "content": [text_content_block_from_string("contents")],
            "error": None,
        },
    ]

    assert agentdojo_messages_to_responses_input(messages) == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "system"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "read it"}],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"file_path": "bill.txt"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
    ]
    serialized = agentdojo_messages_to_response_output(messages)
    assert [item.type for item in serialized] == ["function_call", "function_call_output"]
    assert serialized[0].name == "read_file"
    assert serialized[1].output == "contents"


def test_assistant_text_is_preserved_in_both_directions() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [text_content_block_from_string("finished")],
            "tool_calls": None,
        }
    ]
    assert agentdojo_messages_to_responses_input(messages) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "finished"}],
        }
    ]
    assert agentdojo_messages_to_response_output(messages)[0].content[0].text == "finished"


def test_tool_result_without_call_id_is_rejected_for_model_replay() -> None:
    call = FunctionCall(function="read_file", args={})
    messages = [
        {
            "role": "tool",
            "tool_call": call,
            "tool_call_id": None,
            "content": [text_content_block_from_string("contents")],
            "error": None,
        }
    ]
    with pytest.raises(ValueError, match="require a call id"):
        agentdojo_messages_to_responses_input(messages)


def test_output_serialization_pairs_missing_tool_result_id() -> None:
    call = FunctionCall(function="read_file", args={})
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call": call,
            "tool_call_id": None,
            "content": [text_content_block_from_string("contents")],
            "error": None,
        },
    ]
    output = agentdojo_messages_to_response_output(messages)
    assert output[0].call_id == output[1].call_id == "agentdojo-0-0"


def test_openai_proxy_strips_legacy_tool_result_name() -> None:
    bridge = object.__new__(NeMoGymAgentDojoLLM)
    bridge._event_loop = Mock()
    bridge._request_chat = Mock()
    bridge.responses = []

    with (
        patch("responses_api_agents.agentdojo_family.model_bridge.asyncio.run_coroutine_threadsafe") as submit,
        patch.object(bridge, "_chat_to_response", return_value=Mock()),
    ):
        submit.return_value.result.return_value = Mock()
        bridge.create_chat_completion(
            model="compatibility-alias",
            messages=[
                {
                    "role": "tool",
                    "content": "result",
                    "tool_call_id": "call-1",
                    "name": "search_product",
                }
            ],
        )

    submitted_payload = bridge._request_chat.call_args.args[0]
    assert submitted_payload["messages"][0] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-1",
    }


def test_unsupported_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported AgentDojo message role"):
        agentdojo_messages_to_responses_input([{"role": "alien", "content": []}])


def _response_with_arguments(arguments: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response-1",
            "created_at": 0.0,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": arguments,
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


@pytest.mark.parametrize("arguments", ["{broken", json.dumps(["not", "an", "object"])])
def test_invalid_model_tool_arguments_are_rejected(arguments: str) -> None:
    with pytest.raises(ValueError, match="invalid arguments|non-object arguments"):
        response_to_agentdojo_message(_response_with_arguments(arguments))
