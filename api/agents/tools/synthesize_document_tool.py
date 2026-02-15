"""
Synthesize Document Tool для синтеза нового документа.
Заглушка для будущей реализации синтеза документа на основе найденной информации.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any

from pydantic import Field

from api.agents.base_tool import BaseTool

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SynthesizeDocumentTool(BaseTool):
    """Synthesize a new document based on retrieved information from the knowledge base using predefined templates.
    
    This tool creates a new structured document by synthesizing information from multiple sources
    found in the knowledge base according to predefined document templates/structures.
    Use this when you need to create a formal document (report, summary, analysis, plan, etc.)
    that follows a specific template or structure.
    
    IMPORTANT: This tool is for creating NEW documents using templates. For simple answers
    without templates, use FinalAnswerTool instead.
    
    Currently this is a stub implementation. The actual implementation will
    synthesize documents based on retrieved chunks and their metadata using document templates.
    """

    reasoning: str = Field(description="Why document synthesis is needed and what type of document should be created")
    retrieved_chunks: list[Dict[str, Any]] = Field(
        description="List of retrieved chunks with their metadata to synthesize into a document"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document to create (e.g., 'report', 'summary', 'analysis', 'plan')"
    )
    document_structure: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structure/template for the document (REQUIRED for document synthesis). "
                    "This defines the template that the new document should follow. "
                    "For simple answers without templates, use FinalAnswerTool instead."
    )

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        """
        Выполняет синтез нового документа на основе найденной информации.
        
        Args:
            context: Контекст агента
            config: Конфигурация агента
            
        Returns:
            JSON строка с результатами синтеза документа (пока заглушка)
        """
        logger.info(f"SynthesizeDocumentTool called for {len(self.retrieved_chunks)} chunks, type: {self.document_type}")
        
        # Заглушка - возвращаем сообщение о том, что инструмент не реализован
        return json.dumps({
            "status": "not_implemented",
            "message": "SynthesizeDocumentTool is not yet implemented. This tool will synthesize documents based on retrieved chunks and their metadata.",
            "requested_document_type": self.document_type,
            "chunks_count": len(self.retrieved_chunks),
            "reasoning": self.reasoning,
            "document_structure": self.document_structure
        }, ensure_ascii=False, indent=2)

