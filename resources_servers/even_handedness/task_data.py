# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for Anthropic Political Even-handedness rows."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Paired prompt fields consumed by the paired agent and verifier."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = Field(default=None, description="Stable identifier from the CC-BY source CSV.")
    prompt_a: Optional[str] = Field(default=None, description="First political prompt in the paired task.")
    prompt_b: Optional[str] = Field(default=None, description="Opposing political prompt in the paired task.")
    prompt_a_group: Optional[str] = Field(default=None, description="Interest group represented by prompt A.")
    prompt_b_group: Optional[str] = Field(default=None, description="Interest group represented by prompt B.")
    stance_a: Optional[str] = Field(default=None, description="Source stance associated with prompt A.")
    stance_b: Optional[str] = Field(default=None, description="Source stance associated with prompt B.")
    split: Optional[str] = Field(default=None, description="Source dataset split label.")
    main_category: Optional[str] = Field(default=None, description="Source high-level political category.")
    partisan: Optional[str] = Field(default=None, description="Source partisan-topic indicator.")
    template_category: Optional[str] = Field(default=None, description="Source prompt-category label.")
    template: Optional[str] = Field(default=None, description="Source prompt template.")
    topic_name: Optional[str] = Field(default=None, description="Source political-topic label.")
    source_revision: Optional[str] = Field(default=None, description="Pinned upstream Git revision.")
    source_sha256: Optional[str] = Field(default=None, description="SHA-256 of the pinned source CSV.")
