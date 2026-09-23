# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the kora server (KORA child-safety conversations).

Fields are top-level row columns written by ``benchmarks/kora/prepare.py`` from the run
package's ``scenarios`` and ``risks`` tables. Names follow the package's column names so a
Gym row joins back to it on ``scenario_id``. ``responses_create_params.input`` holds only
the target's system prompt; the child's turns are generated at run time by the
``kora_child_actor`` agent from ``first_user_message`` and the scenario.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    kora_id: str = Field(
        description="`<scenario_id>:<tier>`; unique across the benchmark. The same scenario runs once per tier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    scenario_id: str = Field(
        description="The package's scenario seed id; joins `scenarios.parquet` and, via `results`, the published grades.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    tier: Literal["adult", "child"] = Field(
        description=(
            "Which system prompt the target received: `adult` (upstream's `default` variant, no age declared) "
            "or `child` (the age band's child-safe prompt). The leaderboard reports the two separately."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    age_range: Literal["7to9", "10to12", "13to17"] = Field(
        description="The age band the scenario was written for; selects the child-tier system prompt.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    risk_category_id: str = Field(
        description="Taxonomy category (8). compute_metrics groups on it.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    risk_id: str = Field(
        description="Taxonomy risk (26). Sets the turn budget and whether memory is injected; compute_metrics groups on it.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    risk_name: str = Field(
        description="Display name of the risk; rendered into the judge's and the child simulator's prompts.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    risk_description: str = Field(
        description="The taxonomy's definition of the risk; rendered into the judge's and the child simulator's prompts.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    planned_turns: int = Field(
        description=(
            "Exchanges the conversation runs for: 3 for most risks, 8 for the relationship risks, 4 for the "
            "gradual privacy flavour. One exchange is one child message and one reply."
        ),
        json_schema_extra={"consumed_by": ["agent", "verify"]},
    )
    short_title: str = Field(
        description="Human label for the scenario; the behaviour judge sees it as `Scenario:`.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    child_age: int = Field(
        description="The simulated child's age, inside `age_range`; in every simulator and judge prompt.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    child_cognitive_maturity: str = Field(
        description="low, medium or high; in the child simulator's prompt.",
        json_schema_extra={"consumed_by": ["agent"]},
    )
    child_emotional_maturity: str = Field(
        description="low, medium or high; in the child simulator's prompt.",
        json_schema_extra={"consumed_by": ["agent"]},
    )
    child_maturity: str = Field(
        description="Expanded prose about the child's maturity; in the simulator and both judge prompts.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    child_background: str = Field(
        description="Expanded prose about the child's background; in the simulator and both judge prompts.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    narrative: str = Field(
        description="The scenario the simulator plays and the safety judge calibrates against. The target never sees it.",
        json_schema_extra={"consumed_by": ["verify", "agent"]},
    )
    first_user_message: str = Field(
        description="The child's opening message, stored by upstream and sent verbatim as turn one.",
        json_schema_extra={"consumed_by": ["agent", "verify"]},
    )
    model_memory: Optional[str] = Field(
        default=None,
        description=(
            "Fabricated memory of prior conversations, present only for the three relationship risks. Already "
            "rendered into the system prompt as USER MEMORY; kept here for provenance."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    evaluation_criteria: Optional[str] = Field(
        default=None,
        description="The scenario's descriptive rubric. Upstream stores it but its judge prompts do not use it.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    scenario_flavor_id: Optional[str] = Field(
        default=None,
        description="Sub-variant of the risk the scenario was drawn under, when the risk defines flavours.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    motivation_name: Optional[str] = Field(
        default=None,
        description="The child motivation the scenario was drawn from (10 in the corpus).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
