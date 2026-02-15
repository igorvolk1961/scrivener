"""
Final Answer tool для завершения работы агента.
Адаптировано из sgr-agent-core.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from api.agents.base_tool import BaseTool
from api.agents.models import AgentStatesEnum

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FinalRAGAnswerTool(BaseTool):
    """Finalize a task and complete agent execution after all steps are
    completed.

    Usage: Call after you are ready to finalize your work and provide the final answer to the user.
    
    For knowledge base queries: Always specify knowledge_base_coverage and knowledge_base_sources.
    """

    reasoning: str = Field(description="Why task is now complete and how answer was verified")
    completed_steps: list[str] = Field(
        description="Summary of completed steps including verification", min_length=1, max_length=5
    )
    answer: str = Field(
        description="Comprehensive final answer with EXACT factual details (dates, numbers, names). "
                    "For knowledge base queries: answer must contain ONLY information from the knowledge base."
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(description="Task completion status")
    knowledge_base_coverage: Literal["full", "partial", "none"] = Field(
        default="full",
        description="Полнота информации в базе знаний: 'full' - полный ответ, 'partial' - частичный, 'none' - информации нет. "
                    "Обязательно указывай для ответов из базы знаний."
    )
    knowledge_base_sources: list[str] = Field(
        default_factory=list,
        description="Список источников из базы знаний, использованных для ответа (формат: 'file_name (IRV ID: irv_id)'). "
                    "Обязательно указывай для ответов из базы знаний."
    )

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        context.state = self.status
        context.execution_result = self.answer
        
        # Формируем структурированный ответ
        result = {
            "knowledge_base_coverage": self.knowledge_base_coverage,
            "answer": self.answer,
            "sources": self.knowledge_base_sources,
            "status": self.status.value,
            "reasoning": self.reasoning,
            "completed_steps": self.completed_steps
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
