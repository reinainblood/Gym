# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KORA conversation loop: a child model talks to the model under test for a scenario.

Reproduces ``runTest`` in ``packages/benchmark/src/kora.ts`` at the pinned upstream
revision. For each row (one scenario under one tier):

    turn 1   child  = the scenario's stored first_user_message (never generated)
             model  <- [system prompt from the row] + child message 1
             model  -> reply 1
    turn 2.. child  <- scenarioToNextUserMessagePrompt(risk, scenario, transcript so far)
             child  -> child message n
             model  <- [system prompt] + the whole transcript
             model  -> reply n

for ``planned_turns`` exchanges, after which the transcript goes to the resources server
for judging. The child model never sees the target's system prompt, and the target never
sees the narrative: the two only meet through the transcript.

Two upstream details are kept exactly. A child reply containing a bracketed placeholder
such as ``[name]`` is retried up to three times with upstream's correction appended
(``generateUserMessage.ts``). And every target reply passes upstream's capture-integrity
gate (``validateAssistantTurn.ts``): an empty reply, an echo of the child, a doubled
reply. Upstream abandons the conversation on a defect; this agent records the defect
code as ``assistant_turn_issue``, stops the conversation there, and still sends it to be
judged, so the rate is visible in the aggregate instead of silently lowering ``n``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from benchmarks.kora.upstream_spec import (
    CHILD_MAX_ATTEMPTS,
    CHILD_NEXT_MESSAGE_SYSTEM,
    CHILD_PLACEHOLDER_RETRY_MESSAGE,
    build_child_next_message_user_prompt,
    has_placeholder,
    validate_assistant_turn,
)
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class KoraChildActorConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    child_model_server: ModelServerRef
    child_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    child_max_attempts: int = Field(
        default=CHILD_MAX_ATTEMPTS,
        ge=1,
        description="Retries when the child reply contains a bracketed placeholder (upstream: 3).",
    )
    max_turns: Optional[int] = Field(
        default=None,
        description=(
            "Cap on exchanges per conversation. Unset runs each row's planned_turns, which is the "
            "published protocol; set it only for smoke tests."
        ),
    )


class KoraChildActorRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    kora_id: str = ""
    risk_name: str = ""
    risk_description: str = ""
    child_age: int = 0
    child_cognitive_maturity: str = ""
    child_emotional_maturity: str = ""
    child_maturity: str = ""
    child_background: str = ""
    narrative: str = ""
    first_user_message: str = ""
    planned_turns: int = 3


class KoraChildActorVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def message_text(response: NeMoGymResponse) -> str:
    """The assistant text of a reply, ignoring reasoning items."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "\n".join(part.text for part in item.content if getattr(part, "text", None)).strip()
    return ""


class KoraChildActorAgent(SimpleResponsesAPIAgent):
    config: KoraChildActorConfig

    async def _call_model(
        self,
        *,
        server_name: str,
        params: NeMoGymResponseCreateParamsNonStreaming,
        messages: list[Any],
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Any]:
        request_params = params.model_copy(deep=True)
        request_params.input = messages
        http_response = await self.server_client.post(
            server_name=server_name,
            url_path="/v1/responses",
            json=request_params,
            cookies=cookies,
        )
        await raise_for_status(http_response)
        return NeMoGymResponse.model_validate(await get_response_json(http_response)), http_response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Pass-through to the model under test; the loop lives in :meth:`run`."""
        model_response, cookies = await self._call_model(
            server_name=self.config.model_server.name,
            params=body,
            messages=body.input
            if isinstance(body.input, list)
            else [NeMoGymEasyInputMessage(role="user", content=body.input)],
            cookies=request.cookies,
        )
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return model_response

    async def _child_message(self, body: KoraChildActorRunRequest, transcript: list[dict[str, str]]) -> str:
        """The child's next message, with upstream's placeholder retry."""
        prompt = build_child_next_message_user_prompt(
            risk_name=body.risk_name,
            risk_description=body.risk_description,
            child_age=body.child_age,
            child_cognitive_maturity=body.child_cognitive_maturity,
            child_emotional_maturity=body.child_emotional_maturity,
            child_maturity=body.child_maturity,
            child_background=body.child_background,
            narrative=body.narrative,
            messages=transcript,
        )
        messages: list[Any] = [
            NeMoGymEasyInputMessage(role="system", content=CHILD_NEXT_MESSAGE_SYSTEM),
            NeMoGymEasyInputMessage(role="user", content=prompt),
        ]
        cookies: Any = None
        previous: Optional[str] = None
        for attempt in range(1, self.config.child_max_attempts + 1):
            attempt_messages = list(messages)
            if previous is not None:
                attempt_messages += [
                    NeMoGymEasyInputMessage(role="assistant", content=previous),
                    NeMoGymEasyInputMessage(role="user", content=CHILD_PLACEHOLDER_RETRY_MESSAGE),
                ]
            child_response, cookies = await self._call_model(
                server_name=self.config.child_model_server.name,
                params=self.config.child_responses_create_params,
                messages=attempt_messages,
                cookies=cookies,
            )
            text = message_text(child_response)
            if not has_placeholder(text):
                return text
            previous = text
            LOG.info("kora child reply for %s had a placeholder on attempt %d", body.kora_id or "<row>", attempt)
        # Upstream throws here and the conversation is lost. Sending the last attempt keeps
        # the row; the placeholder is then visible to the judge, which is the honest record.
        return previous or ""

    async def run(self, request: Request, body: KoraChildActorRunRequest) -> KoraChildActorVerifyResponse:
        cookies = request.cookies

        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        responder_params = body.responses_create_params
        # The row's `input` carries only the target's system prompt; the child supplies
        # every user turn from here on.
        system_messages: list[Any] = list(responder_params.input) if isinstance(responder_params.input, list) else []

        planned = body.planned_turns
        if self.config.max_turns is not None:
            planned = min(planned, self.config.max_turns)

        transcript: list[dict[str, str]] = []
        child_turns: list[str] = []
        responder_outputs: list[Any] = []
        usage = None
        model_response: Optional[NeMoGymResponse] = None
        issue: Optional[str] = None
        truncated = 0

        for turn in range(planned):
            child_message = body.first_user_message if turn == 0 else await self._child_message(body, transcript)
            child_turns.append(child_message)
            transcript.append({"role": "user", "content": child_message})

            model_response, cookies = await self._call_model(
                server_name=self.config.model_server.name,
                params=responder_params,
                messages=system_messages
                + [NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in transcript],
                cookies=cookies,
            )
            usage = accumulate_response_usage(usage, model_response.usage)
            if model_response.incomplete_details:
                truncated += 1
            assistant_message = message_text(model_response)

            issue = validate_assistant_turn(assistant_message, transcript)
            if issue:
                LOG.warning(
                    "kora %s: assistant turn %d rejected (%s); ending conversation", body.kora_id, turn + 1, issue
                )
                transcript.pop()
                child_turns.pop()
                break

            transcript.append({"role": "assistant", "content": assistant_message})
            # Only assistant messages ride on the returned response, oldest first, so the
            # verifier and reverify read them positionally as turns 1..n.
            responder_outputs.extend(
                item for item in model_response.output if item.type == "message" and item.role == "assistant"
            )

        if model_response is None:
            raise RuntimeError(f"kora {body.kora_id}: no turns were run (planned_turns={body.planned_turns})")
        final_response = model_response.model_copy(update={"output": responder_outputs, "usage": usage})

        verify_payload = body.model_dump() | {
            "response": final_response.model_dump(mode="json"),
            "conversation": transcript,
            "child_turns": child_turns,
            "turns_completed": len(child_turns),
            "assistant_turn_issue": issue,
            "truncated_turns": truncated,
        }
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_payload,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return KoraChildActorVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    KoraChildActorAgent.run_webserver()
