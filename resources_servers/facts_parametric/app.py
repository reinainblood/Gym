# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACTS Parametric resources server.

The verifier uses an LLM as a semantic-equivalence judge.  It never compares
the policy answer and gold answer with string matching or numeric tolerances.
Each answer is judged three times; the returned reward and label indicators are
the mean across those independent judgements.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


_DEFAULT_JUDGE_PROMPT_PATH = str(Path(__file__).parent / "prompts" / "judge.yaml")
JudgeLabel = Literal["correct", "incorrect", "not-attempted", "unknown"]
_VALID_LABELS: frozenset[str] = frozenset({"correct", "incorrect", "not-attempted", "unknown"})


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the final assistant-message text from a Responses API result."""
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [item.text for item in content if isinstance(getattr(item, "text", None), str)]
            if texts:
                return "\n".join(texts).strip()
    return ""


def parse_judge_label(judge_text: str) -> JudgeLabel:
    """Parse a FACTS label from direct or lightly formatted judge output.

    A received but malformed reply is not a judge-call failure: the shared
    judge failsafe reserves ``judge_failed`` for transport, authentication,
    HTTP, and empty-response failures. Fall back to ``unknown`` when a label
    cannot be recovered, while accepting harmless prose or Markdown around an
    otherwise valid label.
    """
    normalized = judge_text.strip().lower().replace("not attempted", "not-attempted")
    if normalized in _VALID_LABELS:
        return normalized  # type: ignore[return-value]

    # Judge prompts require one label, but providers can still wrap it in
    # Markdown or an explanation. A leading label is the strongest signal:
    # explanatory prose can mention the other labels (for example, “the
    # correct fact”) and must not override ``**incorrect**`` at the start.
    leading_label = re.match(r'^\s*(?:[`*_#"\']\s*)*(correct|incorrect|not-attempted|unknown)(?![a-z-])', normalized)
    if leading_label:
        return leading_label.group(1)  # type: ignore[return-value]

    # Accept conventional structured verdict wording anywhere in the response.
    explicit_labels = re.findall(
        r'\b(?:label|verdict|grade)\s*(?::|is)?\s*[`*_"\']*(correct|incorrect|not-attempted|unknown)(?![a-z-])',
        normalized,
    )
    if explicit_labels:
        return explicit_labels[-1]  # type: ignore[return-value]

    labels = re.findall(r"(?<![a-z-])(correct|incorrect|not-attempted|unknown)(?![a-z-])", normalized)
    if len(set(labels)) == 1:
        return labels[0]  # type: ignore[return-value]
    return "unknown"


class FACTSParametricConfig(BaseResourcesServerConfig):
    """Configuration for a model-server-backed FACTS semantic judge."""

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[]),
        description=(
            "Optional judge request overrides. Unset generation controls are omitted so the judge provider uses its "
            "defaults."
        ),
    )
    judge_prompt_path: str = Field(default=_DEFAULT_JUDGE_PROMPT_PATH)
    use_chat_completions_for_judge: bool = Field(
        default=True,
        description="Use the OpenAI-compatible Chat Completions endpoint, required by Gemini's compatibility API.",
    )
    judge_samples: Literal[3] = 3


class FACTSParametricRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class FACTSParametricVerifyRequest(FACTSParametricRunRequest, BaseVerifyRequest):
    pass


class FACTSParametricVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: str = ""
    expected_answer: str = ""
    judge_labels: list[JudgeLabel] = Field(default_factory=list)
    judge_outputs: list[str] = Field(default_factory=list)
    is_correct: float = 0.0
    is_incorrect: float = 0.0
    is_not_attempted: float = 0.0
    is_unknown: float = 0.0


class FACTSParametricServer(SimpleResourcesServer):
    """Grade parametric facts with three semantic-equivalence judgements."""

    config: FACTSParametricConfig

    def model_post_init(self, context: object) -> None:
        prompt_data = yaml.safe_load(Path(self.config.judge_prompt_path).read_text(encoding="utf-8"))
        self._judge_prompt_template: str = prompt_data["user"]
        super().model_post_init(context)

    async def _call_judge(self, judge_prompt: str) -> str:
        if self.config.use_chat_completions_for_judge:
            chat_params_kwargs: dict = {"messages": [{"role": "user", "content": judge_prompt}]}
            request_overrides = self.config.judge_responses_create_params
            if request_overrides.max_output_tokens is not None:
                chat_params_kwargs["max_tokens"] = request_overrides.max_output_tokens
            if request_overrides.temperature is not None:
                chat_params_kwargs["temperature"] = request_overrides.temperature
            if request_overrides.top_p is not None:
                chat_params_kwargs["top_p"] = request_overrides.top_p
            chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(**chat_params_kwargs)
            chat_response = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
                response_model=NeMoGymChatCompletion,
            )
            content = chat_response.choices[0].message.content if chat_response.choices else None
            if content:
                return content.strip()
            raise JudgeError("empty FACTS judge response")

        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]
        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        text = extract_text_from_response(judge_response)
        if not text:
            raise JudgeError("empty FACTS judge response")
        return text

    @staticmethod
    def _score_fn(result: dict) -> dict[str, float]:
        return {
            "accuracy": result.get("is_correct", 0.0),
            "hedging_rate": result.get("is_not_attempted", 0.0),
            "unknown_rate": result.get("is_unknown", 0.0),
        }

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Report FACTS accuracy, hedging, attempted accuracy, and F1.

        ``unknown`` is attempted: it is neither a model abstention nor evidence
        that the response was wrong. This makes attempted accuracy
        ``correct / (correct + incorrect + unknown)``.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        for key, accuracy in list(metrics.items()):
            if not key.endswith("/accuracy"):
                continue
            prefix = key.removesuffix("/accuracy")
            hedging_rate = metrics.get(f"{prefix}/hedging_rate")
            if hedging_rate is None:
                continue
            attempted_rate = 100.0 - hedging_rate
            attempted_accuracy = 100.0 * accuracy / attempted_rate if attempted_rate else 0.0
            f1 = (
                2.0 * accuracy * attempted_accuracy / (accuracy + attempted_accuracy)
                if accuracy + attempted_accuracy
                else 0.0
            )
            metrics[f"{prefix}/attempted_accuracy"] = attempted_accuracy
            metrics[f"{prefix}/f1"] = f1
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        return highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]")

    async def verify(self, body: FACTSParametricVerifyRequest) -> FACTSParametricVerifyResponse:
        generation = extract_text_from_response(body.response)
        judge_prompt = self._judge_prompt_template.format(
            question=body.question or "",
            expected_answer=body.expected_answer or "",
            generation=generation,
        )
        judge_outputs = await asyncio.gather(
            *[self._call_judge(judge_prompt) for _ in range(self.config.judge_samples)]
        )
        judge_labels = [parse_judge_label(output) for output in judge_outputs]

        def fraction(label: JudgeLabel) -> float:
            return judge_labels.count(label) / self.config.judge_samples

        is_correct = fraction("correct")
        return FACTSParametricVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=is_correct,
            extracted_answer=generation,
            expected_answer=body.expected_answer or "",
            judge_labels=judge_labels,
            judge_outputs=judge_outputs,
            is_correct=is_correct,
            is_incorrect=fraction("incorrect"),
            is_not_attempted=fraction("not-attempted"),
            is_unknown=fraction("unknown"),
        )


if __name__ == "__main__":
    FACTSParametricServer.run_webserver()
