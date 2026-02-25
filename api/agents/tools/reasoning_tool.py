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
    CRITICAL: remaining_steps must be a list of 1 to 3 strings. First element = THE IMMEDIATE NEXT STEP to execute now; then optional following steps. Empty list is invalid.
    """

    # Reasoning chain - step-by-step thinking process (helps stabilize model)
    reasoning_steps: list[str] = Field(
        description="Step-by-step reasoning (brief). MANDATORY: provide exactly 2 or 3 items. One item is invalid and will be rejected. Example: [\"First step...\", \"Second step...\"] or three steps.",
        min_length=1,
        max_length=30,
    )

    # Reasoning and state assessment
    current_situation: str = Field(
        description="Current research situation (2-3 sentences MAX). Max 300 characters.",
        max_length=3000,
    )
    plan_status: str = Field(
        description="Status of current plan (1 sentence). Max 150 characters.",
        max_length=1500,
    )
    enough_data: bool = Field(
        default=False,
        description="Sufficient data collected for comprehensive report?",
    )

    # Next step planning: first item = immediate next action; then optional follow-ups
    remaining_steps: list[str] = Field(
        description="The next 1-3 steps TO DO, in order. Item [0] is THE IMMEDIATE NEXT STEP (the very next action to execute right after this call). Items [1], [2] are the following steps after that. NOT 'steps that will remain after the next one' — [0] is the next one. MANDATORY: 1-3 items. Example: [\"Search RAG for X\", \"Then synthesize answer\"].",
        min_length=1,
        max_length=30,
    )
    task_completed: bool = Field(description="Is the research task finished?")

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs):
        return self.model_dump_json(
            indent=2,
        )
