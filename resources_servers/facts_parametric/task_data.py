# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset schema for FACTS Parametric rows."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Fields consumed by the FACTS Parametric verifier and retained as provenance."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Stable identifier generated from the source-row index.")
    question: str = Field(description="Question shown to the policy and semantic judge.")
    expected_answer: str = Field(description="Gold answer shown only to the semantic judge.")
    source_url: Optional[str] = Field(default=None, description="Source URL provided by the FACTS release.")
    topic: Optional[str] = Field(default=None, description="Source topic provided by the FACTS release.")
