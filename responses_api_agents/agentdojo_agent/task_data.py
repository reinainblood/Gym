# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained agentdojo_agent (no resources server).

A row selects one official AgentDojo benchmark cell. The suite and user task say which task to
run; adding an injection task and an attack turns that same task into its attacked counterpart,
which is why the attacked and clean arms share a schema rather than being two row types.

This is the official `ethz-spylab/agentdojo` backend, so the suites are the original four. The
AgentDyn suites (`shopping`, `github`, `dailylife`) belong to the separate agentdyn_agent, which
pins a different upstream: both install the same top-level `agentdojo` package, so they cannot be
one server, and their schemas are deliberately kept apart rather than unioned into a shape that
would accept a row neither backend can run.

Required-ness mirrors `AgentDojoRunRequest` in `app.py`. Only `suite` and `user_task_id` are
required there; everything else has a wire default.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    # `extra="allow"` matches the run request: rows may carry fields the agent does not read.
    model_config = ConfigDict(extra="allow")

    suite: Literal["banking", "slack", "travel", "workspace"] = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    user_task_id: str = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # None marks the clean arm of the matrix; a string names the injection task to run alongside
    # the user task. The adapter rejects this without `attack`, and `attack` without this.
    injection_task_id: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    attack: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # Left as an open string rather than a literal: the defenses this backend exposes are
    # whatever the pinned upstream registers, and that set is not the AgentDyn one.
    defense: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    benchmark_version: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
