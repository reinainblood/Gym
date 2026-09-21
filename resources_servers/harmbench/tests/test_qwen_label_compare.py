# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.harmbench.operations.qwen.qwen_label_compare import summarize_case_rows
from resources_servers.harmbench.app import CONTEXTUAL_PROMPT as APP_CONTEXTUAL_PROMPT
from resources_servers.harmbench.app import STANDARD_PROMPT as APP_STANDARD_PROMPT
from resources_servers.harmbench.prompts import CONTEXTUAL_PROMPT, STANDARD_PROMPT


def _case(
    *,
    parent: tuple[int, int] = (0, 0),
    canonical: tuple[int, int] = (0, 0),
    tokens: int = 100,
    changed: bool = False,
    replicate: tuple[int, int] | tuple[()] = (),
) -> dict:
    return {
        "parent_labels": list(parent),
        "canonical_labels": list(canonical),
        "parent_target_token_count": tokens,
        "text_changed": changed,
        "replicate_labels": list(replicate),
    }


def test_operational_prompts_match_verifier_prompts():
    assert STANDARD_PROMPT == APP_STANDARD_PROMPT
    assert CONTEXTUAL_PROMPT == APP_CONTEXTUAL_PROMPT


def test_summary_separates_text_changes_from_label_changes():
    summary = summarize_case_rows(
        [
            _case(changed=True),
            _case(parent=(0, 0), canonical=(1, 1), changed=True),
            _case(parent=(1, 1), canonical=(1, 1), tokens=256),
        ]
    )
    assert summary["canonical_512"]["successes"] == 2
    assert summary["parent_vs_canonical"] == {
        "label_agreement": 2,
        "cases": 3,
        "label_flips": 1,
        "cap_not_binding_cases": 2,
        "text_changed_when_cap_not_binding": 2,
        "label_flips_when_cap_not_binding": 1,
        "cap_not_binding_label_agreement": 1,
    }


def test_summary_detects_classifier_and_replicate_instability():
    summary = summarize_case_rows(
        [
            _case(parent=(0, 1), canonical=(0, 0), replicate=(0, 1)),
            _case(parent=(1, 1), canonical=(1, 0), replicate=(1, 1)),
        ]
    )
    assert summary["parent"]["repeat_agreement"] == 1
    assert summary["canonical_512"]["repeat_agreement"] == 1
    assert summary["canonical_vs_replicate"] == {
        "cases": 2,
        "label_agreement": 2,
        "label_flips": 0,
        "replicate_repeat_agreement": 1,
    }
