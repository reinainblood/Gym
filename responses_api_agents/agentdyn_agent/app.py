# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentDyn backend for the shared AgentDojo-family NeMo Gym adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import agentdojo
import agentdojo.agent_pipeline.agent_pipeline as agentdyn_pipeline
import openai
from fastapi import Body
from huggingface_hub import snapshot_download
from pydantic import ConfigDict

from responses_api_agents.agentdojo_family.app import (
    AgentDojoFamilyAgent,
    AgentDojoFamilyAgentConfig,
    AgentDojoFamilyRunRequest,
    AgentDojoFamilyVerifyResponse,
)
from responses_api_agents.agentdojo_family.model_bridge import NeMoGymOpenAIProxy


AGENTDYN_SUITES = ("shopping", "github", "dailylife")
AGENTDYN_DEFENSES = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "camel",
    "progent",
    "drift",
)
PROMPT_GUARD_2_UPSTREAM_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"


class AgentDynAgentConfig(AgentDojoFamilyAgentConfig):
    metric_prefix: str = "agentdyn"
    defense_model_alias: str | None = "gpt-4o-2024-08-06"
    prompt_guard_2_model_name: str = PROMPT_GUARD_2_UPSTREAM_MODEL
    prompt_guard_2_model_revision: str | None = None
    prompt_guard_2_local_path: str | None = None


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
    external_model_base_url: str | None = None
    detector_model_name: str | None = None
    detector_model_revision: str | None = None


class AgentDynAgent(AgentDojoFamilyAgent):
    config: AgentDynAgentConfig

    async def run(self, body: AgentDynRunRequest = Body()) -> AgentDojoFamilyVerifyResponse:
        defense = body.defense or self.config.default_defense
        if defense in {"camel", "progent", "drift"}:
            body = body.model_copy(
                update={
                    "external_model_base_url": self.resolve_model_base_url(
                        self.config.model_server.name,
                        self.rollout_id_from_run(body),
                    )
                }
            )
        elif defense == "prompt_guard_2_detector":
            body = body.model_copy(
                update={
                    "detector_model_name": self.config.prompt_guard_2_model_name,
                    "detector_model_revision": self.config.prompt_guard_2_model_revision,
                }
            )
        return await super().run(body)

    def _run_agentdojo(self, body, bridge, benchmark_version):
        if body.defense == "prompt_guard_2_detector":
            model_path = self.config.prompt_guard_2_local_path or snapshot_download(
                repo_id=self.config.prompt_guard_2_model_name,
                revision=self.config.prompt_guard_2_model_revision,
            )
            original_detector = agentdyn_pipeline.TransformersBasedPIDetector

            def build_prompt_guard_detector(*args, **kwargs):
                if kwargs.get("model_name") == PROMPT_GUARD_2_UPSTREAM_MODEL:
                    kwargs["model_name"] = model_path
                return original_detector(*args, **kwargs)

            with patch.object(agentdyn_pipeline, "TransformersBasedPIDetector", build_prompt_guard_detector):
                return super()._run_agentdojo(body, bridge, benchmark_version)
        if body.defense not in {"camel", "progent", "drift"}:
            return super()._run_agentdojo(body, bridge, benchmark_version)
        if body.external_model_base_url is None:
            raise RuntimeError(f"{body.defense} requires the routed NeMo model-server URL")
        model_alias = self.config.defense_model_alias or self.config.attack_model_alias
        routed_environment = {
            "OPENAI_BASE_URL": body.external_model_base_url,
            "OPENAI_API_KEY": "EMPTY",
            "SECAGENT_POLICY_MODEL": model_alias,
            "SECAGENT_IGNORE_UPDATE_ERROR": "False",
        }
        installed_defenses = Path(agentdojo.__file__).resolve().parent / "defenses"
        with (
            patch.dict(os.environ, routed_environment, clear=False),
            patch.object(agentdyn_pipeline, "_defenses_root", return_value=installed_defenses),
            patch.object(openai, "OpenAI", side_effect=lambda *args, **kwargs: NeMoGymOpenAIProxy(bridge)),
        ):
            return super()._run_agentdojo(body, bridge, benchmark_version)


if __name__ == "__main__":
    AgentDynAgent.run_webserver()
