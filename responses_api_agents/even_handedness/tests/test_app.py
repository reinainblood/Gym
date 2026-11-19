# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for paired-policy input construction."""

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.even_handedness.app import (
    EvenHandednessAgent,
    EvenHandednessAgentRunRequest,
)


def test_params_for_prompt_replaces_only_the_policy_input() -> None:
    body = EvenHandednessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "original"}], temperature=0.3
        ),
        prompt_a="support A",
        prompt_b="support B",
    )

    params = EvenHandednessAgent._params_for_prompt(body, "support B")

    assert params.temperature == 0.3
    assert len(params.input) == 1
    assert params.input[0].content == "support B"
