# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired-policy agent for the Political Even-handedness evaluation."""

from __future__ import annotations

import asyncio

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


class EvenHandednessAgentConfig(SimpleAgentConfig):
    pass


class EvenHandednessAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    prompt_a: str
    prompt_b: str


class EvenHandednessAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class EvenHandednessAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class EvenHandednessAgent(SimpleAgent):
    """Generate each member of a political prompt pair independently."""

    config: EvenHandednessAgentConfig

    @staticmethod
    def _params_for_prompt(
        body: EvenHandednessAgentRunRequest, prompt: str
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        params = body.responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        return params

    async def run(self, request: Request, body: EvenHandednessAgentRunRequest) -> EvenHandednessAgentVerifyResponse:
        cookies = request.cookies
        seeded = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seeded)
        cookies = seeded.cookies

        async def generate(prompt: str) -> dict:
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=self._params_for_prompt(body, prompt),
                cookies=cookies,
            )
            await raise_for_status(response)
            return await get_response_json(response)

        response_a, response_b = await asyncio.gather(generate(body.prompt_a), generate(body.prompt_b))
        if self.config.skip_verification:
            return EvenHandednessAgentVerifyResponse.model_validate(
                body.model_dump() | {"response": response_a, "reward": self.config.skip_verification_reward}
            )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_a, "response_b": response_b},
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return EvenHandednessAgentVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    EvenHandednessAgent.run_webserver()
