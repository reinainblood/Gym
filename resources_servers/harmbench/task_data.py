# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the harmbench server.

HarmBench measures attack success: a target generation is scored by the pinned
``cais/HarmBench-Llama-2-13b-cls`` classifier, except for copyright behaviors, which are scored by
MinHash similarity against a reference text instead.

Unlike most servers here, no field lives in ``verifier_metadata``: ``HarmBenchVerifyRequest``
(app.py:115) types these keys directly on the request, and the committed rows carry them at the top
level, so none of them carries a ``legacy_location`` marker. Requiredness mirrors that model
exactly -- ``harmbench_id``, ``behavior``, ``functional_category`` and ``semantic_category`` have no
default there and so are required here; ``context``, ``tags`` and ``method`` carry the model's own
defaults. ``tags`` is the scoring switch: a row tagged ``hash_check`` is routed to the copyright
MinHash scorer and never reaches the classifier.
"""

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    harmbench_id: str = Field(
        description=(
            "Upstream HarmBench BehaviorID. Identifies the behavior a generation is scored against, "
            "groups rollouts into behaviors for behavior-averaged ASR, and selects the reference text "
            "for copyright rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    behavior: str = Field(
        description=(
            "The harmful behavior the attack is trying to elicit. Interpolated into the upstream "
            "LLAMA2_CLS_PROMPT that the classifier judges the generation against."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    context: str = Field(
        default="",
        description=(
            "Optional context string for contextual behaviors. When non-empty, verify() selects the "
            "contextual classifier prompt variant instead of the standard one."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    functional_category: str = Field(
        description="Upstream functional cohort ('standard', 'contextual', 'copyright', 'multimodal').",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    semantic_category: str = Field(
        description="Upstream semantic cohort (e.g. 'cybercrime_intrusion'), carried through for reporting.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "Upstream behavior tags. Scoring switch: a row tagged 'hash_check' is scored by copyright "
            "MinHash similarity rather than the HarmBench classifier."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    method: str = Field(
        default="DirectRequest",
        description=(
            "Public HarmBench attack method that produced this row's prompt (DirectRequest, ZeroShot, "
            "PAP-top5, GCG, MultiModalPGD, ...). Recorded so rollouts from different attacks are not "
            "pooled as one result."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    upstream_revision: Any = Field(
        default=None,
        description=(
            "Pinned upstream HarmBench commit the row was generated from. Provenance only; verify() does not read it."
        ),
    )
