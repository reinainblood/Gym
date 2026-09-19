# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Official AgentDojo backend for the shared AgentDojo-family adapter."""

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


AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")
AgentDojoAgentConfig = AgentDojoFamilyAgentConfig
AgentDojoVerifyResponse = AgentDojoFamilyVerifyResponse


class AgentDojoRunRequest(AgentDojoFamilyRunRequest):
    model_config = ConfigDict(extra="allow")

    suite: Literal["banking", "slack", "travel", "workspace"]


class AgentDojoAgent(AgentDojoFamilyAgent):
    config: AgentDojoAgentConfig

    async def run(self, body: AgentDojoRunRequest = Body()) -> AgentDojoVerifyResponse:
        return await super().run(body)


if __name__ == "__main__":
    AgentDojoAgent.run_webserver()
