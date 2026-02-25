"""
Reasoning tool для SGR агентов.
Адаптировано из sgr-agent-core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from api.agents.base_tool import BaseTool

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext


class ReasoningTool(BaseTool):
    """Agent core logic determines the next reasoning step with adaptive
    planning by schema-guided-reasoning capabilities. Keep all text fields
    concise and focused.

    Usage: Required tool. Use this tool before any other tool execution.

    CRITICAL: reasoning_steps must be a list of exactly 2 or 3 strings. One item only is invalid and causes a validation error.
    CRITICAL: remaining_steps must be a list of 1 to 3 strings. Empty list is invalid and causes a validation error.
    """

    # Reasoning chain - step-by-step thinking process (helps stabilize model)
    reasoning_steps: list[str] = Field(
        description="Step-by-step reasoning (brief). MANDATORY: provide exactly 2 or 3 items. One item is invalid and will be rejected. Example: [\"First step...\", \"Second step...\"] or three steps.",
        min_length=2,
        max_length=3,
    )

    # Reasoning and state assessment
    current_situation: str = Field(
        description="Current research situation (2-3 sentences MAX). Max 300 characters.",
        max_length=300,
    )
    plan_status: str = Field(
        description="Status of current plan (1 sentence). Max 150 characters.",
        max_length=150,
    )
    enough_data: bool = Field(
        default=False,
        description="Sufficient data collected for comprehensive report?",
    )

    # Next step planning
    remaining_steps: list[str] = Field(
        description="1-3 remaining steps (brief, action-oriented). MANDATORY: provide at least 1 and at most 3 items. Empty list is invalid and will be rejected. Example: [\"Next: do X\"] or [\"Step A\", \"Step B\"].",
        min_length=1,
        max_length=3,
    )
    task_completed: bool = Field(description="Is the research task finished?")

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs):
        return self.model_dump_json(
            indent=2,
        )
