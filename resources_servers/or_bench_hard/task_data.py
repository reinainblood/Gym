# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    split: Literal["hard", "toxic"]
    category: str
    expected_behavior: Literal["answer", "refuse"]
    source_row_index: int | None = None
    prompt_sha256: str | None = None
    dataset_revision: str | None = None
    source_file_sha256: str | None = None
    code_revision: str | None = None
