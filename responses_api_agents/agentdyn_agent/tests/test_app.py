# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.agentdyn_agent.app import (
    AGENTDYN_DEFENSES,
    AGENTDYN_SUITES,
    AgentDynAgent,
    AgentDynAgentConfig,
    AgentDynRunRequest,
)


def _agent(*, default_defense: str | None = None) -> AgentDynAgent:
    config = AgentDynAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentdyn",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        default_defense=default_defense,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    return AgentDynAgent(config=config, server_client=server_client)


def _request() -> AgentDynRunRequest:
    return AgentDynRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "selector"}]},
            "suite": "shopping",
            "user_task_id": "user_task_0",
            "injection_task_id": None,
            "attack": None,
            "defense": None,
            "benchmark_version": "v1.2.2",
        }
    )


def test_contract_exposes_only_agentdyn_suites_and_requested_defenses() -> None:
    assert AGENTDYN_SUITES == ("shopping", "github", "dailylife")
    assert AGENTDYN_DEFENSES == (
        "prompt_guard_2_detector",
        "piguard_detector",
        "camel",
        "progent",
        "drift",
    )
    with pytest.raises(ValueError):
        AgentDynRunRequest.model_validate({**_request().model_dump(), "suite": "banking"})


async def test_configured_defense_is_applied_to_request() -> None:
    agent = _agent(default_defense="piguard_detector")
    agent._run_agentdojo = MagicMock(return_value=(True, True))

    result = await agent.run(_request())

    assert result.defense == "piguard_detector"
    assert result.reward == 1.0


def test_agentdyn_metrics_use_separate_namespace() -> None:
    agent = _agent()
    metrics = agent.compute_metrics(
        [
            [{"utility": True, "security": True, "attack_success": False, "injection_task_id": None}],
            [{"utility": False, "security": False, "attack_success": True, "injection_task_id": "i0"}],
        ]
    )
    assert metrics["agentdyn/benign_utility"] == 1.0
    assert metrics["agentdyn/utility_under_attack"] == 0.0
    assert metrics["agentdyn/attack_success_rate"] == 1.0
