# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility for immutable served K3 code in all local and Ray workers."""

from __future__ import annotations

from dataclasses import dataclass

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import generic


if not hasattr(generic, "OutputRecorder"):

    @dataclass
    class OutputRecorder:
        target_class: type
        index: int = 0
        layer_name: str | None = None
        class_name: str | None = None

    generic.OutputRecorder = OutputRecorder

    _original_post_init = PreTrainedModel.post_init

    def _kimi_compatible_post_init(self) -> None:
        if self.__class__.__name__ == "KimiK3ForConditionalGeneration":
            original_tie_weights = self.tie_weights
            self.tie_weights = lambda *args, **kwargs: original_tie_weights()
        _original_post_init(self)

    PreTrainedModel.post_init = _kimi_compatible_post_init
