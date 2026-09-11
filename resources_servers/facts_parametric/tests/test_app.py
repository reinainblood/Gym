# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.facts_parametric.app import (
    FACTSParametricConfig,
    FACTSParametricServer,
    FACTSParametricVerifyRequest,
    extract_text_from_response,
    parse_judge_label,
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


@pytest.mark.parametrize("label", ["correct", "incorrect", "not-attempted", "unknown"])
def test_parse_judge_label(label: str) -> None:
    assert parse_judge_label(f"  {label}  ") == label


def test_parses_lightly_formatted_judge_text() -> None:
    assert parse_judge_label("The answer is correct.") == "correct"
    assert parse_judge_label('```json\n{"label": "not-attempted"}\n```') == "not-attempted"


def test_leading_label_wins_over_explanation_labels() -> None:
    judge_text = "**incorrect**\n\nThe response conflicts with the correct fact."

    assert parse_judge_label(judge_text) == "incorrect"


def test_unparseable_judge_text_falls_back_to_unknown() -> None:
    assert parse_judge_label("no usable verdict") == "unknown"


def test_extracts_final_assistant_text() -> None:
    assert extract_text_from_response(make_response("Corpus Christi, Texas")) == "Corpus Christi, Texas"


@pytest.fixture
def config() -> FACTSParametricConfig:
    return FACTSParametricConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


def test_judge_request_overrides_default_to_provider_settings() -> None:
    config = FACTSParametricConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )

    assert config.judge_responses_create_params.max_output_tokens is None
    assert config.judge_responses_create_params.temperature is None
    assert config.judge_responses_create_params.top_p is None


async def test_verify_averages_three_judge_labels(config: FACTSParametricConfig) -> None:
    server = FACTSParametricServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._call_judge = AsyncMock(side_effect=["correct", "unknown", "correct"])
    request = FACTSParametricVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("Corpus Christi"),
        question="Which city?",
        expected_answer="Corpus Christi, Texas",
    )

    result = await server.verify(request)

    assert result.reward == pytest.approx(2 / 3)
    assert result.judge_labels == ["correct", "unknown", "correct"]
    assert result.is_correct == pytest.approx(2 / 3)
    assert result.is_unknown == pytest.approx(1 / 3)
    assert result.is_not_attempted == 0.0


def test_metrics_treat_unknown_as_attempted(config: FACTSParametricConfig) -> None:
    server = FACTSParametricServer(config=config, server_client=MagicMock(spec=ServerClient))
    metrics = server.compute_metrics([[{"is_correct": 2 / 3, "is_not_attempted": 0.0, "is_unknown": 1 / 3}]])

    prefix = "pass@1[avg-of-1]"
    assert metrics[f"{prefix}/accuracy"] == pytest.approx(200 / 3)
    assert metrics[f"{prefix}/attempted_accuracy"] == pytest.approx(200 / 3)
    assert metrics[f"{prefix}/f1"] == pytest.approx(200 / 3)
