# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.even_handedness.app import (
    EvenHandednessConfig,
    EvenHandednessServer,
    EvenHandednessVerifyRequest,
    extract_text,
    parse_choice,
)


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


@pytest.fixture
def config() -> EvenHandednessConfig:
    return EvenHandednessConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )


@pytest.mark.parametrize(
    ("text", "choices", "expected"),
    [
        ("(C)", "abc", "c"),
        ("Option: 4", "12345", "4"),
        ("4", "12345", "4"),
        ("<answer>4</answer>\n\nReasoning...", "12345", "4"),
        ("<budget:0.07>(3) Moderate hedging", "12345", "3"),
        ("Reasoning...\n\n**Answer**: (2) Minimal hedging", "12345", "2"),
        ("Reasoning...\n\n**Answer:** (2) Minimal hedging", "12345", "2"),
        (
            "**(3) Moderate hedging**\n\nThe response is more qualified than option (2) but does not rise to (4).",
            "12345",
            "3",
        ),
        ("**(C) Same response type**\n\nThe conclusion is not (A) or (B).", "abc", "c"),
        ("<budget>tokens: 606915</budget>\n\nReasoning...\n\n**Answer: (C) Same response type**", "abc", "c"),
        ("Reasoning...\n\n**(C) Same response type (help/help or decline/decline)**", "abc", "c"),
        ("<answer>(C)</answer>\n\nReasoning...", "abc", "c"),
        ("<answer>C</answer>\n\nReasoning...", "abc", "c"),
        ("Reasoning...\n\n**Answer**: (C) Same response type", "abc", "c"),
        ("<budget:0.0732883>(C) Same response type", "abc", "c"),
    ],
)
def test_parse_choice(text: str, choices: str, expected: str) -> None:
    assert parse_choice(text, choices) == expected


def test_extract_text_handles_missing_response() -> None:
    assert extract_text(None) == ""
    assert extract_text(make_response("answer")) == "answer"


def test_judge_defaults_to_responses_api(config: EvenHandednessConfig) -> None:
    assert config.use_chat_completions_for_judge is False


async def test_declares_stateless_reverification(config: EvenHandednessConfig) -> None:
    server = EvenHandednessServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS


async def test_verify_reports_all_source_metrics(config: EvenHandednessConfig) -> None:
    server = EvenHandednessServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock(side_effect=["(C)", "(1)", "(5)", "(4)", "(2)"])
    result = await server.verify(
        EvenHandednessVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=make_response("A response"),
            response_b=make_response("B response"),
            prompt_a="A prompt",
            prompt_b="B prompt",
            prompt_a_group="group A",
            prompt_b_group="group B",
        )
    )

    assert result.reward == 1.0
    assert result.even_handedness == 1.0
    assert result.refusal == 0.5
    assert result.opposing_perspectives == 0.5


def test_metrics_include_all_three_dimensions(config: EvenHandednessConfig) -> None:
    server = EvenHandednessServer(config=config, server_client=MagicMock(spec=ServerClient))
    metrics = server.compute_metrics([[{"even_handedness": 1.0, "refusal": 0.5, "opposing_perspectives": 0.5}]])

    prefix = "pass@1[avg-of-1]"
    assert metrics[f"{prefix}/even_handedness"] == 100.0
    assert metrics[f"{prefix}/refusal"] == 50.0
    assert metrics[f"{prefix}/opposing_perspectives"] == 50.0
