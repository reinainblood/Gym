# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentDyn backend for the shared AgentDojo-family NeMo Gym adapter."""

from __future__ import annotations

from typing import Literal

from fastapi import Body
from pydantic import ConfigDict

from responses_api_agents.agentdojo_family.app import (
    AgentDojoFamilyAgent,
    AgentDojoFamilyAgentConfig,
    AgentDojoFamilyRunRequest,
    AgentDojoFamilyVerifyResponse,
)


AGENTDYN_SUITES = ("shopping", "github", "dailylife")
AGENTDYN_DEFENSES = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "camel",
    "progent",
    "drift",
)


class AgentDynAgentConfig(AgentDojoFamilyAgentConfig):
    metric_prefix: str = "agentdyn"


class AgentDynRunRequest(AgentDojoFamilyRunRequest):
    model_config = ConfigDict(extra="allow")

    suite: Literal["shopping", "github", "dailylife"]
    defense: (
        Literal[
            "prompt_guard_2_detector",
            "piguard_detector",
            "camel",
            "progent",
            "drift",
        ]
        | None
    ) = None


class AgentDynAgent(AgentDojoFamilyAgent):
    config: AgentDynAgentConfig

    async def run(self, body: AgentDynRunRequest = Body()) -> AgentDojoFamilyVerifyResponse:
        return await super().run(body)


if __name__ == "__main__":
    AgentDynAgent.run_webserver()
