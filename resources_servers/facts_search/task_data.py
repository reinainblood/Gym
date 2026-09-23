# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for FACTS Search."""

from pydantic import BaseModel, ConfigDict, Field


class UpstreamMetadata(BaseModel):
    """Published dataset provenance retained on every prepared row."""

    dataset: str
    version: int
    file: str
    csv_sha256: str
    license: str
    published_rows: int
    advertised_v2_public_rows: int


class TaskData(BaseModel):
    """Fields consumed by the FACTS Search verifier and retained for provenance."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Stable example identifier from the public FACTS Search dataset.")
    problem: str = Field(
        description="Factual research question presented to the search agent.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    gold_answer: str = Field(
        description="Reference answer used by the Gemini A/B/C grader.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    row_sha256: str = Field(description="SHA-256 over the source CSV fields for this row.")
    upstream: UpstreamMetadata = Field(description="Pinned public dataset provenance.")
