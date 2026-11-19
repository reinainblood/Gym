# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentDyn backend for the shared AgentDojo-family NeMo Gym adapter."""

from __future__ import annotations

import ast
import importlib
import os
import threading
import time
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
# Upstream's full DEFENSES list, in upstream's own names.
AGENTDYN_DEFENSES = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "camel",
    "progent",
    "drift",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
)
PROMPT_GUARD_2_UPSTREAM_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
PIGUARD_UPSTREAM_MODEL = "leolee99/PIGuard"
TRANSFORMERS_PI_UPSTREAM_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
# Detector defenses hard-code their classifier upstream and resolve moving Hub `main` at
# construction time. Both are pinned here instead, so a published treatment names the exact
# weights it scored with. PIGuard additionally ships custom modeling code that upstream loads
# with `trust_remote_code=True`, which makes the pin a supply-chain control, not just provenance.
DETECTOR_UPSTREAM_MODELS = {
    "prompt_guard_2_detector": PROMPT_GUARD_2_UPSTREAM_MODEL,
    "piguard_detector": PIGUARD_UPSTREAM_MODEL,
    "transformers_pi_detector": TRANSFORMERS_PI_UPSTREAM_MODEL,
}
ROUTED_DEFENSES = frozenset({"camel", "progent", "drift"})
CAMEL_MAX_EVAL_STEPS = 100_000
CAMEL_MAX_EVAL_SECONDS = 30.0
CAMEL_MAX_AST_NODES = 3_500
_CAMEL_EVAL_STATE = threading.local()


def _patch_camel_constant_error() -> None:
    """Repair bounded-error paths in AgentDyn's pinned CaMeL interpreter.

    The pinned interpreter tries to format the unsupported type through
    ``type(value).__type__``. For bytes, complex numbers, and Ellipsis that
    raises ``AttributeError`` before CaMeL can return its structured error to
    the policy model for correction.
    """
    interpreter = importlib.import_module("camel.interpreter.interpreter")
    original = interpreter._eval_constant
    if getattr(original, "_nemo_gym_unsupported_constant_fix", False):
        return

    def eval_constant(node, namespace, tool_calls_chain, dependencies, eval_args):
        if isinstance(node, ast.Constant) and (isinstance(node.value, (bytes, complex)) or node.value is Ellipsis):
            error = interpreter.CaMeLException(
                NotImplementedError(f"unsupported constant type {type(node.value).__name__}"),
                (node,),
                (),
            )
            return interpreter.EvalResult(
                interpreter.result.Error(error),
                namespace,
                tool_calls_chain,
                dependencies,
            )
        return original(node, namespace, tool_calls_chain, dependencies, eval_args)

    eval_constant._nemo_gym_unsupported_constant_fix = True
    interpreter._eval_constant = eval_constant

    original_camel_eval = interpreter.camel_eval
    if getattr(original_camel_eval, "_nemo_gym_evaluation_budget", False):
        return

    def bounded_camel_eval(node, namespace, tool_calls_chain, dependencies, eval_args):
        is_top_level = getattr(_CAMEL_EVAL_STATE, "depth", 0) == 0
        if is_top_level:
            _CAMEL_EVAL_STATE.steps = 0
            _CAMEL_EVAL_STATE.started_at = time.monotonic()
        _CAMEL_EVAL_STATE.depth = getattr(_CAMEL_EVAL_STATE, "depth", 0) + 1
        _CAMEL_EVAL_STATE.steps += 1
        elapsed = time.monotonic() - _CAMEL_EVAL_STATE.started_at
        try:
            oversized_program = is_top_level and sum(1 for _ in ast.walk(node)) > CAMEL_MAX_AST_NODES
            if (
                oversized_program
                or _CAMEL_EVAL_STATE.steps > CAMEL_MAX_EVAL_STEPS
                or elapsed > CAMEL_MAX_EVAL_SECONDS
            ):
                error_node = node
                if not hasattr(error_node, "lineno"):
                    error_node = ast.Constant(
                        value=None,
                        lineno=1,
                        col_offset=0,
                        end_lineno=1,
                        end_col_offset=1,
                    )
                error = interpreter.CaMeLException(
                    TimeoutError(
                        f"CaMeL evaluation budget exceeded ({CAMEL_MAX_AST_NODES} AST nodes, "
                        f"{CAMEL_MAX_EVAL_STEPS} steps, or {CAMEL_MAX_EVAL_SECONDS:g}s)"
                    ),
                    (error_node,),
                    (),
                )
                return interpreter.EvalResult(
                    interpreter.result.Error(error),
                    namespace,
                    tool_calls_chain,
                    dependencies,
                )
            try:
                return original_camel_eval(node, namespace, tool_calls_chain, dependencies, eval_args)
            except RecursionError:
                error_node = node if hasattr(node, "lineno") else ast.Constant(value=None, lineno=1, col_offset=0)
                error = interpreter.CaMeLException(
                    RecursionError("CaMeL evaluation recursion limit exceeded"),
                    (error_node,),
                    (),
                )
                return interpreter.EvalResult(
                    interpreter.result.Error(error),
                    namespace,
                    tool_calls_chain,
                    dependencies,
                )
        finally:
            _CAMEL_EVAL_STATE.depth -= 1
            if is_top_level:
                for attribute in ("steps", "started_at", "depth"):
                    try:
                        delattr(_CAMEL_EVAL_STATE, attribute)
                    except AttributeError:
                        pass

    bounded_camel_eval._nemo_gym_evaluation_budget = True
    interpreter.camel_eval = bounded_camel_eval


class AgentDynAgentConfig(AgentDojoFamilyAgentConfig):
    # One rollout at a time, enforced rather than advised. The routed defenses redirect their
    # internal OpenAI clients with `mock.patch` over process globals (`openai.OpenAI`,
    # `os.environ`), and those scopes are not thread-safe: overlapping rollouts restore each
    # other's patches out of order, which can record calls on the wrong rollout or hand a
    # defense the real OpenAI client. Scale out with more agent processes instead.
    concurrency: Literal[1] = 1
    metric_prefix: str = "agentdyn"
    defense_model_alias: str | None = "gpt-4o-2024-08-06"
    prompt_guard_2_model_name: str = PROMPT_GUARD_2_UPSTREAM_MODEL
    prompt_guard_2_model_revision: str | None = None
    prompt_guard_2_local_path: str | None = None
    piguard_model_name: str = PIGUARD_UPSTREAM_MODEL
    piguard_model_revision: str | None = None
    piguard_local_path: str | None = None
    transformers_pi_model_name: str = TRANSFORMERS_PI_UPSTREAM_MODEL
    transformers_pi_model_revision: str | None = None
    transformers_pi_local_path: str | None = None

    def detector_source(self, defense: str) -> tuple[str, str | None, str | None]:
        """Return the (repository, revision, local cache override) backing a detector defense."""
        if defense == "prompt_guard_2_detector":
            return (
                self.prompt_guard_2_model_name,
                self.prompt_guard_2_model_revision,
                self.prompt_guard_2_local_path,
            )
        if defense == "transformers_pi_detector":
            return (
                self.transformers_pi_model_name,
                self.transformers_pi_model_revision,
                self.transformers_pi_local_path,
            )
        return (self.piguard_model_name, self.piguard_model_revision, self.piguard_local_path)


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
            "transformers_pi_detector",
            "spotlighting_with_delimiting",
            "repeat_user_prompt",
            "tool_filter",
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
        if defense in ROUTED_DEFENSES:
            body = body.model_copy(
                update={
                    "external_model_base_url": self.resolve_model_base_url(
                        self.config.model_server.name,
                        self.rollout_id_from_run(body),
                    )
                }
            )
        elif defense in DETECTOR_UPSTREAM_MODELS:
            detector_model_name, detector_model_revision, _ = self.config.detector_source(defense)
            body = body.model_copy(
                update={
                    "detector_model_name": detector_model_name,
                    "detector_model_revision": detector_model_revision,
                }
            )
        return await super().run(body)

    def _run_agentdojo(self, body, bridge, benchmark_version):
        if body.defense in DETECTOR_UPSTREAM_MODELS:
            upstream_model_name = DETECTOR_UPSTREAM_MODELS[body.defense]
            repo_id, revision, local_path = self.config.detector_source(body.defense)
            model_path = local_path or snapshot_download(repo_id=repo_id, revision=revision)
            original_detector = agentdyn_pipeline.TransformersBasedPIDetector

            def build_pinned_detector(*args, **kwargs):
                if kwargs.get("model_name") == upstream_model_name:
                    kwargs["model_name"] = model_path
                return original_detector(*args, **kwargs)

            with patch.object(agentdyn_pipeline, "TransformersBasedPIDetector", build_pinned_detector):
                return super()._run_agentdojo(body, bridge, benchmark_version)
        if body.defense not in ROUTED_DEFENSES:
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
        original_camel_builder = agentdyn_pipeline._build_camel_pipeline

        def build_patched_camel_pipeline(*args, **kwargs):
            pipeline = original_camel_builder(*args, **kwargs)
            _patch_camel_constant_error()
            return pipeline

        with (
            patch.dict(os.environ, routed_environment, clear=False),
            patch.object(agentdyn_pipeline, "_defenses_root", return_value=installed_defenses),
            patch.object(agentdyn_pipeline, "_build_camel_pipeline", build_patched_camel_pipeline),
            patch.object(openai, "OpenAI", side_effect=lambda *args, **kwargs: NeMoGymOpenAIProxy(bridge)),
        ):
            return super()._run_agentdojo(body, bridge, benchmark_version)


if __name__ == "__main__":
    AgentDynAgent.run_webserver()
